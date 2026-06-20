"""Parametric Eurorack panel hardware, for photoreal assembly renders.

Each factory returns a build123d Part positioned so that z=0 is the *front
face of the panel* and the part sits proud of it (i.e. the panel occupies
z<=0, hardware grows in +z toward the viewer). Compose by translating to a
hole's (x, y) in panel coordinates and dropping the assembly onto the panel.

All parts carry a `.label` (used to group them into per-material STL files at
export time) and a `.color` (used for the live OCP preview). The Blender
renderer assigns real materials by the STL filename / label, so the colors
here only need to be plausible for the quick preview.

Material groups (labels):
  nut, washer, jack_body, knob, knob_pointer, led, led_dome, screen_body,
  screen_bezel, screw
"""

from __future__ import annotations

from math import cos, radians

from build123d import (
    Axis,
    Box,
    Color,
    Compound,
    Cone,
    Cylinder,
    Part,
    Plane,
    Pos,
    RegularPolygon,
    Rot,
    Sphere,
    chamfer,
    extrude,
    fillet,
)

# Plausible preview colors (linear-ish RGB 0..1).
COL_NICKEL = (0.72, 0.73, 0.75)
COL_BLACK_PLASTIC = (0.06, 0.06, 0.07)
COL_DARK = (0.02, 0.02, 0.02)
COL_KNOB = (0.08, 0.08, 0.09)
COL_POINTER = (0.90, 0.90, 0.92)
COL_LED_RED = (0.85, 0.10, 0.10)
COL_LED_BLUE = (0.10, 0.25, 0.95)
COL_LED_GREEN = (0.12, 0.85, 0.20)
LED_PREVIEW = {"red": COL_LED_RED, "blue": COL_LED_BLUE, "green": COL_LED_GREEN}
COL_PCB = (0.05, 0.20, 0.08)
COL_GLASS = (0.02, 0.02, 0.03)

# Befaco knurled-nut finishes (preview colors; the renderer maps the same
# material labels to proper metal/anodized shaders).
NUT_COLORS = {
    "nut_silver": (0.78, 0.79, 0.81),
    "nut_black": (0.04, 0.04, 0.045),
    "nut_red": (0.55, 0.04, 0.04),
    "nut_gold": (0.86, 0.66, 0.26),
}


def _tag(part: Part, label: str, color: tuple[float, float, float]) -> Part:
    part.label = label
    part.color = Color(*color)
    return part


# ---------------------------------------------------------------------------
# 3.5 mm jack (Thonkiconn / PJ398SM look): hex nut + star washer + visible
# black bushing face with the socket opening.
# ---------------------------------------------------------------------------
def jack(
    *,
    nut_d: float = 7.8,
    nut_h: float = 2.6,
    bushing_d: float = 6.0,
    socket_d: float = 3.6,
    nut_label: str = "nut_silver",
    knurl_facets: int = 30,
) -> Compound:
    """A panel-mount 3.5 mm jack with a round Befaco-style knurled nut.

    `nut_label` selects the nut finish (one of NUT_COLORS keys); the renderer
    assigns the matching metal/anodized material. Knurl is approximated by a
    many-sided prism (cheap, no booleans) for a faceted edge highlight.
    """
    parts: list[Part] = []

    # Knurled round nut: high-facet prism + chamfered top, threaded bore.
    poly = RegularPolygon(nut_d / 2, max(6, knurl_facets), major_radius=True)
    nut = extrude(poly, nut_h)
    try:
        top = nut.edges().group_by(Axis.Z)[-1]
        nut = chamfer(top, length=0.35)
    except Exception:  # noqa: BLE001
        pass
    nut -= Cylinder(bushing_d / 2 - 0.1, nut_h * 4)  # thread bore
    color = NUT_COLORS.get(nut_label, COL_NICKEL)
    parts.append(_tag(nut, nut_label, color))

    # Black plastic bushing face flush at the nut top, with the socket recess.
    bushing = Pos(0, 0, nut_h - 0.4) * Cylinder(bushing_d / 2 - 0.05, 0.6)
    bushing -= Pos(0, 0, nut_h) * Cylinder(socket_d / 2, 2.0)
    parts.append(_tag(bushing, "jack_body", COL_BLACK_PLASTIC))

    asm = Compound(children=parts)
    asm.label = "jack"
    return asm


# ---------------------------------------------------------------------------
# Knob: tapered skirt with a flat top and a pointer indicator line.
# ---------------------------------------------------------------------------
def _knob_core(
    *,
    base_d: float,
    top_d: float,
    height: float,
    skirt_gap: float,
    pointer: bool,
    indicator_deg: float,
    fillet_r: float,
    body_label: str,
    pointer_label: str,
    body_color: tuple[float, float, float],
    metal_cap: bool = False,
) -> Compound:
    parts: list[Part] = []

    body = Pos(0, 0, skirt_gap + height / 2) * Cone(base_d / 2, top_d / 2, height)
    try:
        top_edges = body.edges().group_by(Axis.Z)[-1]
        body = fillet(top_edges, radius=fillet_r)
    except Exception:  # noqa: BLE001
        pass
    parts.append(_tag(body, body_label, body_color))

    top_z = skirt_gap + height
    if metal_cap:  # brushed-aluminium top cap (Davies-style)
        cap = Pos(0, 0, top_z - 0.4) * Cylinder((top_d - 0.6) / 2, 1.0)
        parts.append(_tag(cap, "knob_cap", (0.74, 0.75, 0.77)))

    if pointer:
        line = Box(top_d / 2 - 1.0, 0.9, 0.6)
        line = Pos((top_d / 2 - 1.0) / 2 + 1.0, 0, top_z + (0.4 if metal_cap else -0.15)) * line
        line = Rot(0, 0, indicator_deg) * line
        parts.append(_tag(line, pointer_label, COL_POINTER))

    asm = Compound(children=parts)
    asm.label = body_label
    return asm


def knob(
    *,
    base_d: float = 12.0,
    top_d: float = 10.0,
    height: float = 11.0,
    skirt_gap: float = 1.0,
    pointer: bool = True,
    indicator_deg: float = 90.0,
    metal_cap: bool = False,
) -> Compound:
    """A collet-style synth knob (the wider DEPTH-style control).

    indicator_deg measured CCW from +X. `metal_cap` adds a brushed-aluminium
    top cap (Davies 1900-style) for boards like the Ksoloti Big Genes.
    """
    return _knob_core(
        base_d=base_d, top_d=top_d, height=height, skirt_gap=skirt_gap,
        pointer=pointer, indicator_deg=indicator_deg, fillet_r=0.8,
        body_label="knob", pointer_label="knob_pointer", body_color=COL_KNOB,
        metal_cap=metal_cap,
    )


def trimmer(
    *,
    base_d: float = 7.0,
    top_d: float = 6.0,
    height: float = 5.5,
    skirt_gap: float = 0.8,
    indicator_deg: float = 90.0,
) -> Compound:
    """A short, narrow black trimmer pot with a white indicator line."""
    return _knob_core(
        base_d=base_d, top_d=top_d, height=height, skirt_gap=skirt_gap,
        pointer=True, indicator_deg=indicator_deg, fillet_r=0.5,
        body_label="trimmer", pointer_label="trimmer_pointer", body_color=COL_KNOB,
    )


def switch_cap(
    *,
    body_d: float = 6.0,
    proud_h: float = 4.0,
    through_depth: float = 3.0,
    dome_fillet: float = 2.4,
    silver: bool = False,
    color: tuple[float, float, float] = (0.16, 0.16, 0.18),
) -> Compound:
    """Momentary-switch cap as actually mounted.

    The cap's Ø6 body slots *through* the panel hole (which is wider, ~6.9 mm,
    so a thin gap shows around it) and stands ~4 mm proud with a rounded dome.
    The wide gripping base sits behind the panel and is not modelled, since it
    is hidden from the front. z=0 is the panel front face; the body continues
    down to -through_depth so it fills the hole.
    """
    h = proud_h + through_depth
    body = Pos(0, 0, (proud_h - through_depth) / 2) * Cylinder(body_d / 2, h)
    try:
        top = body.edges().group_by(Axis.Z)[-1]
        body = fillet(top, radius=dome_fillet)
    except Exception:  # noqa: BLE001
        pass
    label = "switch_cap_silver" if silver else "switch_cap"
    cap = _tag(body, label, (0.74, 0.75, 0.77) if silver else color)
    asm = Compound(children=[cap])
    asm.label = label
    return asm


def bolt(*, head_d: float = 5.0, head_h: float = 1.8, socket: bool = True) -> Compound:
    """A silver socket-head cap bolt (e.g. screen mounting screws)."""
    head = Pos(0, 0, head_h / 2) * Cylinder(head_d / 2, head_h)
    try:
        top = head.edges().group_by(Axis.Z)[-1]
        head = fillet(top, radius=0.3)
    except Exception:  # noqa: BLE001
        pass
    if socket:
        head -= Pos(0, 0, head_h - 0.5) * extrude(RegularPolygon(1.4, 6, major_radius=False), 1.0)
    head = _tag(head, "bolt", (0.78, 0.79, 0.81))
    asm = Compound(children=[head])
    asm.label = "bolt"
    return asm


def sd_card(*, thickness: float = 2.0, height: float = 11.0, protrude: float = 10.0,
            behind: float = 3.0) -> Compound:
    """An SD card seated edge-on in a panel slot, protruding `protrude` mm.

    The card plane is vertical (Y-Z); from the front only the ~2 mm edge shows,
    as it does on a real card-edge slot."""
    h = protrude + behind
    card = Pos(0, 0, (protrude - behind) / 2) * Box(thickness, height, h)
    try:  # clip the protruding outer-top corner (SD notch read)
        edge = card.edges().filter_by(Axis.Y).group_by(Axis.Z)[-1].sort_by(Axis.Y)[-1]
        card = chamfer(edge, length=1.6)
    except Exception:  # noqa: BLE001
        pass
    card = _tag(card, "sd_card", (0.05, 0.22, 0.6))
    asm = Compound(children=[card])
    asm.label = "sd_card"
    return asm


def toggle_switch(
    *,
    nut_d: float = 6.0,
    nut_h: float = 2.0,
    lever_len: float = 8.0,
    lever_d: float = 1.8,
    tilt_deg: float = 24.0,
) -> Compound:
    """A vertical mini-toggle switch: round bushing nut on the panel face and a
    metal lever leaning up (+Y) with a rounded tip. All metal (silver)."""
    parts: list[Part] = []

    nut = Pos(0, 0, nut_h / 2) * Cylinder(nut_d / 2, nut_h)
    parts.append(_tag(nut, "toggle", (0.72, 0.73, 0.75)))

    lever = Pos(0, 0, lever_len / 2) * Cylinder(lever_d / 2, lever_len)
    lever += Pos(0, 0, lever_len) * Sphere(lever_d * 0.8)
    lever = Rot(-tilt_deg, 0, 0) * lever   # lean toward +Y (up the panel)
    lever = Pos(0, 0, nut_h) * lever
    parts.append(_tag(lever, "toggle", (0.72, 0.73, 0.75)))

    asm = Compound(children=parts)
    asm.label = "toggle"
    return asm


def led_holder(*, led_d: float = 5.0, bezel_d: float = 8.6) -> Compound:
    """A 5 mm LED in a black panel holder/bezel (red LED)."""
    parts: list[Part] = []
    # Black bezel: a flange on the panel face + a short collar.
    flange = Pos(0, 0, 0.6) * Cylinder(bezel_d / 2, 1.2)
    collar = Pos(0, 0, 1.2 + 1.0) * Cylinder(led_d / 2 + 0.9, 2.0)
    bezel = flange + collar
    bezel -= Cylinder(led_d / 2 + 0.1, 20)  # bore for the LED
    parts.append(_tag(bezel, "led_holder", COL_BLACK_PLASTIC))

    # 5 mm red LED seated in the bezel, dome proud of the collar.
    rim_h = 1.0
    rim = Pos(0, 0, 1.0 + rim_h / 2) * Cylinder(led_d / 2, rim_h)
    dome = Pos(0, 0, 1.0 + rim_h) * Sphere(led_d / 2, arc_size1=0, arc_size2=90)
    parts.append(_tag(rim, "led", COL_LED_RED))
    parts.append(_tag(dome, "led_dome", COL_LED_RED))

    asm = Compound(children=parts)
    asm.label = "led_holder"
    return asm


# ---------------------------------------------------------------------------
# 3 mm LED: short rim + hemispherical dome, emissive.
# ---------------------------------------------------------------------------
def led(*, d: float = 3.0, rim_h: float = 1.0, color_name: str = "red") -> Compound:
    """A domed LED. `color_name` ("red"/"blue"/...) selects the material; red
    keeps the base "led"/"led_dome" labels, others get "led_<name>[_dome]"."""
    base = "led" if color_name == "red" else f"led_{color_name}"
    col = LED_PREVIEW.get(color_name, COL_LED_RED)
    rim = Pos(0, 0, rim_h / 2) * Cylinder(d / 2, rim_h)
    dome = Pos(0, 0, rim_h) * Sphere(d / 2, arc_size1=0, arc_size2=90)
    rim = _tag(rim, base, col)
    dome = _tag(dome, base + "_dome", col)
    asm = Compound(children=[rim, dome])
    asm.label = "led"
    return asm


# ---------------------------------------------------------------------------
# OLED module: black PCB carrier + dark glass. The *active area* is not built
# here — the renderer adds an emissive textured plane from the manifest, so
# the screen image stays decoupled from geometry. We expose the active-area
# size/offset as attributes for the manifest.
# ---------------------------------------------------------------------------
def oled(
    *,
    module_w: float = 26.0,
    module_h: float = 19.0,
    pcb_h: float = 1.2,
    glass_h: float = 1.4,
    active_w: float = 21.7,
    active_h: float = 10.9,
    active_dy: float = 1.0,
) -> Compound:
    """A 0.96\" style OLED carrier. Active-area is rendered as emissive plane."""
    pcb = Pos(0, 0, pcb_h / 2) * Box(module_w, module_h, pcb_h)
    pcb = _tag(pcb, "screen_body", COL_PCB)

    glass = Pos(0, active_dy, pcb_h + glass_h / 2) * Box(module_w - 1.5, module_h - 4.0, glass_h)
    glass = _tag(glass, "screen_bezel", COL_GLASS)

    asm = Compound(children=[pcb, glass])
    asm.label = "oled"
    # Surface z where the emissive plane should sit, plus active-area geometry.
    asm.active_z = pcb_h + glass_h + 0.05  # type: ignore[attr-defined]
    asm.active_w = active_w  # type: ignore[attr-defined]
    asm.active_h = active_h  # type: ignore[attr-defined]
    asm.active_dy = active_dy  # type: ignore[attr-defined]
    return asm


# ---------------------------------------------------------------------------
# M3 panel screw head (for the mount slots) — optional dressing.
# ---------------------------------------------------------------------------
def screw(*, head_d: float = 5.5, head_h: float = 1.6) -> Compound:
    head = Pos(0, 0, head_h / 2) * Cylinder(head_d / 2, head_h)
    head = _tag(head, "screw", COL_NICKEL)
    asm = Compound(children=[head])
    asm.label = "screw"
    return asm


if __name__ == "__main__":
    # Quick visual sanity check of one of each part.
    import os
    from ocp_vscode import show

    demo = [
        Pos(-20, 0, 0) * jack(),
        Pos(0, 0, 0) * knob(),
        Pos(15, 0, 0) * led(),
        Pos(40, 0, 0) * oled(),
        Pos(-30, 0, 0) * screw(),
    ]
    try:
        show(*demo, port=int(os.environ.get("OCP_VSCODE_PORT", "3939")))
    except Exception as ex:  # noqa: BLE001
        print("preview unavailable:", ex)
