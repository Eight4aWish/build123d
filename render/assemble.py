"""Assemble a populated panel (panel + labels + placed hardware) from a
module's FaceplateParams, then export per-material STLs + a JSON manifest for
the Blender renderer.

Placement is driven entirely by the module's own parameters:
  - every enabled hole that isn't a pot  -> jack
  - every pot hole (pot_hole_indices)    -> knob
  - every LED centre (button_led_*)      -> 3 mm LED
  - the screen rect (if screen_enable)   -> OLED carrier + emissive plane

So this works for any panel in the repo that follows the CortHex/duallpg
lineage; modules with extra structure just expose more (LEDs, screen).

Usage:
  ./.venv/bin/python render/assemble.py duallpg --out render/out/duallpg
  ./.venv/bin/python render/assemble.py duallpg --show         # OCP preview
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

# Make sibling module (hardware) and repo root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Panel generator modules live in ../panels and are imported by bare name.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "panels"))

from build123d import (  # noqa: E402
    Align,
    Axis,
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    Color,
    Compound,
    FontStyle,
    Location,
    Locations,
    Mode,
    Plane,
    Rectangle,
    Text,
    export_stl,
    extrude,
)

import hardware as hw  # noqa: E402


# Nut-colour scheme: Befaco knurled nuts by signal type.
#   audio In -> black, control(CV/trig) In -> silver,
#   audio Out -> red,  sum/CV Out -> gold.
import re  # noqa: E402

_OUT_KW = {"OUT", "OUTPUT", "MIX", "SUM", "EOC", "EOR", "EOF"}
_GOLD_OUT_KW = {"MIX", "SUM", "CV", "EOC", "EOR", "EOF", "GATE", "TRIG", "CLK", "CLOCK"}
_CONTROL_KW = {
    "CV", "PING", "STRIKE", "GATE", "TRIG", "TRIGGER", "CLK", "CLOCK", "RESET",
    "RUN", "LINK", "SYNC", "MOD", "ATTEN", "MANUAL", "DEPTH", "REC", "RECORD",
    "ACCENT", "ACCT", "VOCT", "PITCH", "PIT",
}


def classify_nut(label: str, *, is_output: bool = False) -> str:
    """Map a hole label to a Befaco nut finish material label.

    `is_output=True` forces output treatment for jacks whose label doesn't
    say "out" (e.g. CortHex's CV1–6), via the module's output_hole_indices.
    """
    toks = set(re.findall(r"[A-Za-z]+", (label or "").upper()))
    is_out = is_output or bool(toks & _OUT_KW)
    is_control = bool(toks & _CONTROL_KW)
    if is_out:
        return "nut_gold" if (toks & _GOLD_OUT_KW) else "nut_red"
    return "nut_silver" if is_control else "nut_black"


# Labels that mount a short black trimmer pot rather than a jack.
_TRIMMER_KW = {"ATTEN", "MANUAL", "TRIM", "TRIMMER"}


def is_trimmer(label: str) -> bool:
    toks = set(re.findall(r"[A-Za-z]+", (label or "").upper()))
    return bool(toks & _TRIMMER_KW)


def _screen_rect(params) -> dict | None:
    """Return {x,y,w,h} of the screen active opening in panel coords, or None."""
    if not getattr(params, "screen_enable", False):
        return None
    # n8synth-style API: screen_rect() -> (p0, p1) opposite corners, or
    # explicit screen_w/screen_h + a centre helper. Try the common shapes.
    for attr in ("screen_rect", "screen_opening", "screen_bbox"):
        fn = getattr(params, attr, None)
        if callable(fn):
            try:
                p0, p1 = fn()
                x = (p0[0] + p1[0]) / 2
                y = (p0[1] + p1[1]) / 2
                return {"x": x, "y": y, "w": abs(p1[0] - p0[0]), "h": abs(p1[1] - p0[1])}
            except Exception:  # noqa: BLE001
                pass
    return None


# Hole labels that mount a momentary button cap on the daisy-style panels.
_DAISY_BUTTON_KW = {"MODE", "BTN", "BUTTON", "SELECT"}
# A 2-throw toggle is usually spotted by having a label above *and* below - one
# per throw. A toggle whose two positions are not worth naming separately gets a
# single label instead, and would otherwise fall through to the jack branch, so
# name it here.
_DAISY_TOGGLE_KW = {"FLIP", "TOGGLE", "TOG"}


def _daisy_labels(mod, params, xf):
    """Readable hole labels + branding, positioned in the real-front frame.

    `xf(x, y)` maps a raw hole position to the rotated (real-front) frame;
    labels sit just below their hole and read upright.
    """
    holes = mod._get_labelable_holes(params)
    below = params.label_offset[1]  # negative = below the hole
    with BuildPart() as p:
        def add(txt: str, x: float, y: float, size: float) -> None:
            txt = (txt or "").strip()
            if not txt:
                return
            dx0, dy0 = mod._text_local_offset(
                txt, font=params.label_font, style=params.label_font_style, font_size=size
            )
            with BuildSketch(Plane.XY.offset(params.thickness)) as sk:
                with Locations(Location((x, y, 0), (0, 0, 0))):
                    with Locations((dx0, dy0)):
                        Text(txt, font_size=size, font=params.label_font, font_style=params.label_font_style)
            extrude(to_extrude=sk.sketch, amount=params.label_height, mode=Mode.ADD)

        add(params.brand_text_top, params.panel_w / 2, params.panel_h - 4.01, params.brand_size)
        add(params.brand_text_bottom, params.panel_w / 2, 4.01, params.brand_size)
        above = tuple(getattr(mod, "HOLE_LABELS_ABOVE", ()))
        for idx, hh in enumerate(holes):
            txt = mod.HOLE_LABELS[idx] if idx < len(mod.HOLE_LABELS) else ""
            fx, fy = xf(hh.x, hh.y)
            add(txt, fx, fy + below, params.label_size)
            if idx < len(above) and above[idx].strip():
                add(above[idx], fx, fy - below, params.label_size)  # above the hole
    return p.part


def build_assembly_daisy(mod, module_name: str):
    """Adapter for the declarative-Hole / RECT_CUTOUTS panel API
    (daisy_braids, daisy_grids etc.).

    The KiCad-derived geometry is rotated 180° to the real-front orientation
    (TIMBRE trimmer top-left, OUT-R jack bottom-right). Jacks/trimmers/LEDs
    come from HOLES; the MODE hole gets a button cap; the screen + SD slot
    come from RECT_CUTOUTS; silver bolts sit at the 3.0 mm OLED mount holes.
    """
    params = mod.PanelParams()
    t = float(params.thickness)
    w, h_ = float(params.panel_w), float(params.panel_h)

    def xf(x: float, y: float) -> tuple[float, float]:
        return (w - x, h_ - y)

    # Base: rotate 180° about the panel centre (== xf for every point).
    base = mod.build_base(params)
    base = base.rotate(Axis.Z, 180).moved(Location((w, h_, 0)))
    labels = _daisy_labels(mod, params, xf)
    base.label = "panel"
    base.color = Color(*params.base_color)
    labels.label = "labels"
    labels.color = Color(*params.label_color)

    labelable = mod._get_labelable_holes(params)
    above = tuple(getattr(mod, "HOLE_LABELS_ABOVE", ()))
    label_map = {
        (round(hh.x, 2), round(hh.y, 2)): (mod.HOLE_LABELS[i] if i < len(mod.HOLE_LABELS) else "")
        for i, hh in enumerate(labelable)
    }
    above_map = {
        (round(hh.x, 2), round(hh.y, 2)): (above[i] if i < len(above) else "")
        for i, hh in enumerate(labelable)
    }
    nut_overrides = dict(getattr(params, "nut_overrides", ()))

    placements: list[tuple[Compound, float, float, float]] = []
    placed_meta: list[dict] = []
    led_meta: list[dict] = []

    for hh in mod.HOLES:
        if hh.shape != "circle":
            continue  # oval = mount slot, no hardware
        d = hh.drill[0]
        key = (round(hh.x, 2), round(hh.y, 2))
        label = label_map.get(key, "")
        abv = above_map.get(key, "")
        fx, fy = xf(hh.x, hh.y)
        toks = set(label.upper().replace("/", " ").split())
        if abs(d - 3.0) < 0.05:  # OLED screen mounting bolt
            placements.append((hw.bolt(head_d=4.0), fx, fy, t))
            placed_meta.append({"x": fx, "y": fy, "kind": "bolt"})
        elif d < 5.0:  # small hole -> 3 mm LED
            placements.append((hw.led(color_name="red"), fx, fy, t))
            led_meta.append({"x": fx, "y": fy, "color": "red"})
        elif (label.strip() and abv.strip()) or toks & _DAISY_TOGGLE_KW:
            # Both labels (one per throw), or a single label that names a toggle.
            placements.append((hw.toggle_switch(), fx, fy, t))
            placed_meta.append({"x": fx, "y": fy, "kind": "toggle",
                                "label": f"{abv}/{label}" if abv.strip() else label})
        elif toks & _DAISY_BUTTON_KW:  # MODE etc. -> momentary button
            placements.append((hw.switch_cap(), fx, fy, t))
            placed_meta.append({"x": fx, "y": fy, "kind": "button", "label": label})
        elif d < 7.0:  # jack
            nut = nut_overrides.get(label) or classify_nut(label)
            placements.append((hw.jack(nut_label=nut), fx, fy, t))
            placed_meta.append({"x": fx, "y": fy, "kind": "jack", "label": label, "nut": nut})
        else:  # >= 7 mm -> black trimmer
            placements.append((hw.trimmer(indicator_deg=90.0), fx, fy, t))
            placed_meta.append({"x": fx, "y": fy, "kind": "trimmer", "label": label})

    screens: list[dict] = []
    for r in mod.RECT_CUTOUTS:
        cx, cy = xf(r.x + r.w / 2, r.y + r.h / 2)
        if r.w > 8.0:  # OLED window
            screens.append({
                "x": cx, "y": cy, "z": t - 0.4, "w": 13.5, "h": 10.5, "dy": 0.0,
                "image": f"{module_name}_screen.png",
            })
        elif getattr(params, "sd_card_present", True):  # narrow slot -> SD card
            placements.append((hw.sd_card(height=min(r.h - 1.5, 11.0), protrude=7.0), cx, cy, t))
            placed_meta.append({"x": cx, "y": cy, "kind": "sd_card"})

    manifest = {
        "module": module_name,
        "panel_w": float(params.panel_w),
        "panel_h": float(params.panel_h),
        "thickness": t,
        "base_color": list(params.base_color),
        "label_color": list(params.label_color),
        "holes": placed_meta,
        "leds": led_meta,
        "screens": screens,
    }
    return params, base, labels, placements, manifest


def build_assembly_6hp(mod, params, module_name: str):
    """Adapter for the 6HP-family API (col_x / row_y / labels_left+right /
    content_y_offset / screen_origin), e.g. pico2w_onclite.

    The 2xN control grid: POT* -> trimmer, SELECT/MENU -> button, IN*/OUT* ->
    jacks. Screen from screen_origin() with 4 mounting bolts.
    """
    t = float(params.thickness)
    w, h_ = float(params.panel_w), float(params.panel_h)
    # Some 6HP panels install upside-down: build_labels pre-rotates the text
    # 180° while build_base does not, so rotate the whole assembly 180° to the
    # as-installed (upright) orientation.
    rot = bool(getattr(params, "rotate_labels", False))

    def xf(x: float, y: float) -> tuple[float, float]:
        return (w - x, h_ - y) if rot else (x, y)

    base = mod.build_base(params)
    labels = mod.build_labels(params)
    if rot:
        base = base.rotate(Axis.Z, 180).moved(Location((w, h_, 0)))
        labels = labels.rotate(Axis.Z, 180).moved(Location((w, h_, 0)))
    base.label = "panel"
    base.color = Color(*params.base_color)
    labels.label = "labels"
    labels.color = Color(*params.label_color)

    dy = float(getattr(params, "content_y_offset", 0.0)) + float(getattr(params, "labels_y_offset", 0.0))
    col_x = params.col_x
    row_y = params.row_y
    ll = tuple(getattr(params, "labels_left", ()))
    lr = tuple(getattr(params, "labels_right", ()))
    sl = tuple(getattr(params, "secondary_left", ()))
    sr = tuple(getattr(params, "secondary_right", ()))
    nut_overrides = dict(getattr(params, "nut_overrides", ()))

    placements: list[tuple[Compound, float, float, float]] = []
    placed_meta: list[dict] = []

    hole_enabled = getattr(params, "_hole_enabled", None)
    for ci, cx in enumerate(col_x):
        labels_col = ll if ci == 0 else lr
        sec_col = sl if ci == 0 else sr
        for ri, y in enumerate(row_y):
            if hole_enabled is not None and not hole_enabled(ci, ri):
                continue  # hole removed in the panel; no hardware here
            px, py = xf(cx, y + dy)
            lbl = labels_col[ri] if ri < len(labels_col) else ""
            sec = sec_col[ri] if ri < len(sec_col) else ""
            toks = set((lbl + " " + sec).upper().replace("/", " ").split())
            if "POT" in toks:
                placements.append((hw.trimmer(indicator_deg=90.0), px, py, t))
                placed_meta.append({"x": px, "y": py, "kind": "trimmer", "label": lbl})
            elif toks & {"SELECT", "MENU", "BTN", "BUTTON", "MODE"}:
                placements.append((hw.switch_cap(), px, py, t))
                placed_meta.append({"x": px, "y": py, "kind": "button", "label": lbl})
            else:
                nut = nut_overrides.get(lbl) or classify_nut(lbl)
                placements.append((hw.jack(nut_label=nut), px, py, t))
                placed_meta.append({"x": px, "y": py, "kind": "jack", "label": lbl, "nut": nut})

    screens: list[dict] = []
    if getattr(params, "screen_enable", False) and hasattr(params, "screen_origin"):
        sx, sy = params.screen_origin()
        sy += dy
        sw = float(params.screen_w)
        sh = float(params.screen_h)
        cx_s, cy_s = xf(sx + sw / 2, sy + sh / 2)
        pw, ph = sw - 1.5, sh - 1.5
        spx = getattr(params, "screen_px", None)
        if spx:
            aspect = float(spx[0]) / float(spx[1])
            if pw / ph > aspect:
                pw = ph * aspect
            else:
                ph = pw / aspect
        screens.append({
            "x": cx_s, "y": cy_s, "z": t - 0.4, "w": pw, "h": ph, "dy": 0.0,
            "image": f"{module_name}_screen.png",
        })
        # 4 screen mounting bolts (matches build_base inset/offsets).
        hxL, hxR = sx + 0.5, sx + sw - 0.5
        yB, yT = sy, sy + sh
        for (mx, my) in [
            (hxL, yB - params.screen_bottom_hole_offset),
            (hxR, yB - params.screen_bottom_hole_offset),
            (hxL, yT + params.screen_top_hole_offset),
            (hxR, yT + params.screen_top_hole_offset),
        ]:
            bx, by = xf(mx, my)
            placements.append((hw.bolt(head_d=4.0), bx, by, t))
            placed_meta.append({"x": bx, "y": by, "kind": "bolt"})

    manifest = {
        "module": module_name,
        "panel_w": float(params.panel_w),
        "panel_h": float(params.panel_h),
        "thickness": t,
        "base_color": list(params.base_color),
        "label_color": list(params.label_color),
        "holes": placed_meta,
        "leds": [],
        "screens": screens,
    }
    return params, base, labels, placements, manifest


def _center_offset(txt: str, size: float) -> tuple[float, float]:
    with BuildSketch(Plane.XY) as sk:
        Text(txt, font_size=size, font="Arial", font_style=FontStyle.BOLD)
    bb = sk.sketch.bounding_box()
    return (-(bb.min.X + bb.max.X) / 2, -(bb.min.Y + bb.max.Y) / 2)


def _layout_labels(L: dict):
    """Readable labels + branding for an explicit LAYOUT panel."""
    w, h, t = float(L["panel_w"]), float(L["panel_h"]), float(L["thickness"])
    with BuildPart() as p:
        def add(txt: str, x: float, y: float, size: float) -> None:
            if not (txt or "").strip():
                return
            dx0, dy0 = _center_offset(txt, size)
            with BuildSketch(Plane.XY.offset(t)) as sk:
                with Locations(Location((x, y, 0), (0, 0, 0))):
                    with Locations((dx0, dy0)):
                        Text(txt, font_size=size, font="Arial", font_style=FontStyle.BOLD)
            extrude(to_extrude=sk.sketch, amount=0.2, mode=Mode.ADD)

        bs = L.get("brand_size", 3.4)
        add(L.get("brand_top", ""), w / 2, h - 4.5, bs)
        add(L.get("brand_bottom", ""), w / 2, 4.5, bs)
        for c in L["controls"]:
            lbl = c.get("label", "")
            if lbl:
                dy = c.get("label_dy")
                if dy is None:
                    dy = -10.0 if c["kind"] in ("knob", "encoder") else -7.0
                add(lbl, c["x"] + c.get("label_dx", 0.0), c["y"] + dy, c.get("label_size", 2.5))
            if c.get("label_above"):
                add(c["label_above"], c["x"], c["y"] + c.get("above_dy", 6.5), c.get("label_size", 2.5))
    return p.part


# Default panel-hole diameter (mm) per control kind for the LAYOUT adapter.
_LAYOUT_HOLE_D = {"jack": 6.9, "knob": 7.0, "encoder": 7.0, "button": 6.9,
                  "led": 3.0, "toggle": 6.0, "bolt": 3.2, "midi": 6.0, "trimmer": 6.0}


def build_assembly_layout(mod, module_name: str):
    """Adapter for hand-authored panels that expose a `LAYOUT` dict (an
    explicit control list). Used for reconstructed commercial boards such as
    the Ksoloti Big Genes / AMYboard where no parametric source exists."""
    L = mod.LAYOUT
    w, h, t = float(L["panel_w"]), float(L["panel_h"]), float(L["thickness"])
    controls = L["controls"]

    with BuildPart() as p:
        Box(w, h, t, align=(Align.MIN, Align.MIN, Align.MIN))
        with BuildSketch(Plane.XY.offset(-0.2)):
            for c in controls:
                x, y, k = c["x"], c["y"], c["kind"]
                if k in ("screen", "sd_slot", "usb"):
                    with Locations((x, y)):
                        Rectangle(c["w"], c["h"])
                else:
                    d = c.get("d", _LAYOUT_HOLE_D.get(k, 6.9))
                    with Locations((x, y)):
                        Circle(d / 2)
            for (mx, my) in L.get("mounts", []):
                with Locations((mx, my)):
                    Circle(3.2 / 2)
        extrude(amount=t + 0.4, mode=Mode.SUBTRACT)
    base = p.part
    base.label = "panel"
    base.color = Color(*L.get("base_color", (0.028, 0.028, 0.032)))
    labels = _layout_labels(L)
    labels.label = "labels"
    labels.color = Color(*L.get("label_color", (0.92, 0.92, 0.93)))

    placements: list[tuple[Compound, float, float, float]] = []
    placed_meta: list[dict] = []
    led_meta: list[dict] = []
    screens: list[dict] = []

    for c in controls:
        x, y, k, lbl = c["x"], c["y"], c["kind"], c.get("label", "")
        if k in ("jack", "midi"):
            nut = c.get("nut") or classify_nut(lbl)
            placements.append((hw.jack(nut_label=nut), x, y, t))
            placed_meta.append({"x": x, "y": y, "kind": "jack", "label": lbl, "nut": nut})
        elif k == "knob":
            placements.append((hw.knob(base_d=16, top_d=13, height=13, metal_cap=True), x, y, t))
            placed_meta.append({"x": x, "y": y, "kind": "knob", "label": lbl})
        elif k == "encoder":
            placements.append((hw.knob(base_d=14, top_d=11, height=13, metal_cap=True), x, y, t))
            placed_meta.append({"x": x, "y": y, "kind": "encoder", "label": lbl})
        elif k == "trimmer":
            placements.append((hw.trimmer(indicator_deg=90.0), x, y, t))
            placed_meta.append({"x": x, "y": y, "kind": "trimmer", "label": lbl})
        elif k == "button":
            placements.append((hw.switch_cap(body_d=c.get("cap_d", 6.0), silver=c.get("silver", False)), x, y, t))
            placed_meta.append({"x": x, "y": y, "kind": "button", "label": lbl})
        elif k == "led":
            placements.append((hw.led(color_name=c.get("color", "red")), x, y, t))
            led_meta.append({"x": x, "y": y, "color": c.get("color", "red")})
        elif k == "toggle":
            placements.append((hw.toggle_switch(), x, y, t))
            placed_meta.append({"x": x, "y": y, "kind": "toggle", "label": lbl})
        elif k == "screen":
            sw, sh = float(c["w"]) - 1.0, float(c["h"]) - 1.0
            px = c.get("px")
            if px:
                aspect = float(px[0]) / float(px[1])
                if sw / sh > aspect:
                    sw = sh * aspect
                else:
                    sh = sw / aspect
            screens.append({"x": x, "y": y, "z": t - 0.4, "w": sw, "h": sh, "dy": 0.0,
                            "image": f"{module_name}_screen.png"})
        # sd_slot / usb: cut only, no hardware (dark recess)

    manifest = {
        "module": module_name,
        "panel_w": w, "panel_h": h, "thickness": t,
        "base_color": list(L.get("base_color", (0.028, 0.028, 0.032))),
        "label_color": list(L.get("label_color", (0.92, 0.92, 0.93))),
        "holes": placed_meta, "leds": led_meta, "screens": screens,
    }
    return None, base, labels, placements, manifest


def build_assembly(module_name: str):
    mod = importlib.import_module(module_name)
    if hasattr(mod, "LAYOUT"):
        return build_assembly_layout(mod, module_name)
    if hasattr(mod, "HOLES") and hasattr(mod, "PanelParams"):
        return build_assembly_daisy(mod, module_name)
    params = mod.FaceplateParams()
    if hasattr(params, "col_x") and hasattr(params, "row_y"):
        return build_assembly_6hp(mod, params, module_name)
    t = float(params.thickness)

    base = mod.build_base(params)
    labels = mod.build_labels(params)

    base.label = "panel"
    base.color = Color(*getattr(params, "base_color", (0.86, 0.86, 0.86)))
    if labels is not None:
        labels.label = "labels"
        labels.color = Color(*getattr(params, "label_color", (0.1, 0.1, 0.1)))

    holes = params.holes_sorted()
    remove = set(int(i) for i in getattr(params, "remove_hole_indices", ()))
    pots = set(int(i) for i in getattr(params, "pot_hole_indices", ()))
    buttons = set(int(i) for i in getattr(params, "button_led_indices", ()))
    buttons_plain = set(int(i) for i in getattr(params, "button_hole_indices", ()))
    trimmers_idx = set(int(i) for i in getattr(params, "trimmer_hole_indices", ()))
    outputs = set(int(i) for i in getattr(params, "output_hole_indices", ()))
    # Explicit per-label nut overrides ((label, material) pairs) win over the
    # keyword classifier — e.g. "X"/"Y" CV outs -> gold, "AUD-IN" -> silver.
    nut_overrides = dict(getattr(params, "nut_overrides", ()))
    labels_below = tuple(getattr(params, "hole_labels_below", ()))
    labels_above = tuple(getattr(params, "hole_labels_above", ()))

    def label_at(i: int) -> str:
        b = labels_below[i].strip() if i < len(labels_below) else ""
        if b:
            return b
        return labels_above[i].strip() if i < len(labels_above) else ""

    # Each placement is (hardware Compound built at origin, x, y, z). We keep
    # the placement separate from the geometry so we can bake the world
    # location into each leaf at export time (avoids losing the transform when
    # regrouping leaves by material).
    placements: list[tuple[Compound, float, float, float]] = []
    placed_meta: list[dict] = []

    for idx, (x, y) in enumerate(holes):
        if idx in remove:
            continue
        label = label_at(idx)
        if idx in pots:
            placements.append((hw.knob(indicator_deg=90.0), x, y, t))
            placed_meta.append({"idx": idx, "x": x, "y": y, "kind": "pot", "label": label})
        elif idx in buttons or idx in buttons_plain:
            # Momentary switch in the main hole; an LED (if any) is a companion
            # hole placed separately from button_led_centers() below.
            placements.append((hw.switch_cap(), x, y, t))
            placed_meta.append({"idx": idx, "x": x, "y": y, "kind": "button", "label": label})
        elif idx in trimmers_idx or is_trimmer(label):
            placements.append((hw.trimmer(indicator_deg=90.0), x, y, t))
            placed_meta.append({"idx": idx, "x": x, "y": y, "kind": "trimmer", "label": label})
        elif not label.strip():
            # Unlabelled grid holes carry a 5 mm red LED in a black holder.
            placements.append((hw.led_holder(), x, y, t))
            placed_meta.append({"idx": idx, "x": x, "y": y, "kind": "led_holder", "label": label})
        else:
            nut = nut_overrides.get(label) or classify_nut(label, is_output=(idx in outputs))
            placements.append((hw.jack(nut_label=nut), x, y, t))
            placed_meta.append({"idx": idx, "x": x, "y": y, "kind": "jack", "label": label, "nut": nut})

    # LEDs (CortHex lineage). button_led_centers() returns panel-coord centres.
    led_meta: list[dict] = []
    if getattr(params, "button_led_indices", ()):
        led_colors = tuple(getattr(params, "button_led_colors", ()))
        try:
            for i, (x, y) in enumerate(params.button_led_centers()):
                cname = led_colors[i] if i < len(led_colors) else "red"
                placements.append((hw.led(color_name=cname), x, y, t))
                led_meta.append({"x": x, "y": y, "color": cname})
        except Exception:  # noqa: BLE001
            pass

    # Screen.
    screens: list[dict] = []
    if getattr(params, "screen_enable", False) and hasattr(params, "screen_geometry"):
        # Panel-cut window with 4 mounting bolts (e.g. teensy_chaos): emissive
        # plane recessed in the window, silver bolts at the mount holes.
        (scx, scy), mount_pts = params.screen_geometry()
        sw = float(getattr(params, "screen_w", 24.0)) - 1.0
        sh = float(getattr(params, "screen_h", 12.0)) - 1.0
        # Fit the lit area to the OLED's pixel aspect inside the window so wide
        # panels (e.g. 128x32) don't vertically stretch the image.
        spx = getattr(params, "screen_px", None)
        if spx:
            aspect = float(spx[0]) / float(spx[1])
            if sw / sh > aspect:
                sw = sh * aspect
            else:
                sh = sw / aspect
        screens.append({
            "x": scx, "y": scy, "z": t - 0.4, "w": sw, "h": sh, "dy": 0.0,
            "image": f"{module_name}_screen.png",
        })
        for (mx, my) in mount_pts:
            placements.append((hw.bolt(head_d=4.0), mx, my, t))
            placed_meta.append({"x": mx, "y": my, "kind": "bolt"})
    else:
        rect = _screen_rect(params)
        if rect is not None:
            ol = hw.oled(active_w=rect["w"], active_h=rect["h"])
            placements.append((ol, rect["x"], rect["y"], t))
            screens.append(
                {
                    "x": rect["x"],
                    "y": rect["y"],
                    "z": t + float(ol.active_z),
                    "w": float(ol.active_w),
                    "h": float(ol.active_h),
                    "dy": float(ol.active_dy),
                    "image": f"{module_name}_screen.png",
                }
            )

    manifest = {
        "module": module_name,
        "panel_w": float(params.panel_w),
        "panel_h": float(params.panel_h),
        "thickness": t,
        "base_color": list(getattr(params, "base_color", (0.86, 0.86, 0.86))),
        "label_color": list(getattr(params, "label_color", (0.1, 0.1, 0.1))),
        "holes": placed_meta,
        "leds": led_meta,
        "screens": screens,
    }

    return params, base, labels, placements, manifest


def _place_world(asm: Compound, x: float, y: float, z: float) -> list:
    """Bake the (x,y,z) world location into each child solid of `asm`.

    `asm` is built at the origin with children carrying material labels and
    their own local locations; multiplying each child by the placement
    Location yields a standalone solid at world coordinates with the label
    preserved, so regrouping by material keeps every part in place.
    """
    loc = Location((x, y, z))
    out = []
    for child in asm.children:
        moved = loc * child
        moved.label = child.label
        out.append(moved)
    return out


def export(module_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, base, labels, placements, manifest = build_assembly(module_name)

    # Group every solid by its material label and export one STL each.
    groups: dict[str, list] = {}
    for item in (base, labels):
        if item is not None:
            groups.setdefault(item.label, []).append(item)
    for asm, x, y, z in placements:
        for solid in _place_world(asm, x, y, z):
            groups.setdefault(solid.label or "misc", []).append(solid)

    stl_files: dict[str, str] = {}
    for label, solids in groups.items():
        comp = Compound(children=[s for s in solids])
        path = out_dir / f"{label}.stl"
        export_stl(comp, str(path))
        stl_files[label] = path.name

    manifest["stl"] = stl_files
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Exported {len(stl_files)} material STLs + manifest.json to {out_dir}")
    for label, fn in sorted(stl_files.items()):
        print(f"  {label:14s} {fn}")
    if manifest["screens"]:
        print(f"  screens: {len(manifest['screens'])} (emissive plane built in Blender)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble a populated panel and export render assets")
    ap.add_argument("module", help="panel module name, e.g. duallpg or CortHex")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default render/out/<module>)")
    ap.add_argument("--show", action="store_true", help="preview the colored assembly in OCP")
    args = ap.parse_args()

    if args.show:
        import os

        from build123d import Pos
        from ocp_vscode import Camera, show

        _, base, labels, placements, _ = build_assembly(args.module)
        objs = [o for o in (base, labels) if o is not None]
        objs += [Pos(x, y, z) * asm for asm, x, y, z in placements]
        try:
            show(*objs, reset_camera=Camera.RESET, port=int(os.environ.get("OCP_VSCODE_PORT", "3939")))
        except Exception as ex:  # noqa: BLE001
            print("preview unavailable:", ex)
        return

    out_dir = args.out or (Path(__file__).resolve().parent / "out" / args.module)
    export(args.module, out_dir)


if __name__ == "__main__":
    main()
