"""Daisy Patch Init + Grove 0.66" OLED 10HP faceplate — FACE-DOWN 2-colour, ribbed.

Reusable template for the "Patch Init + small OLED" panel family (daisy_mfx,
daisy_braids, daisy_multiosc, ...). Unlike the older face-up scripts in this repo,
this one is built to print **face-down** with **flush** lettering and adds
**back-side stiffening ribs** so the panel doesn't bend.

Orientation
-----------
The KiCad hole data has y increasing the opposite way to how the module mounts, so
everything is run through ``phys()`` (a 180° flip) to the real layout: **pots at the
top, jacks at the bottom**, MOD 1 top-left, OUT-R bottom-right. Labels are authored
upright in that final orientation and sit *below* their hole.

The part is modelled **front-face-up** (front at z = thickness, ribs hanging below the
back face) so the viewer shows it the way you'd look at the mounted module. For STL
export it is flipped front-down onto the bed (`export_transform`).

How it prints (Bambu A1 mini, 0.1 mm layers)
--------------------------------------------
Exported, the front (lettered) face is at z = 0 so it lands on the bed. The labels are
a flush *inlay*: the base has letter-shaped pockets in its front skin (`label_height`,
default 0.2 mm = 2 layers) and the `labels` solid fills them. Printed face-down this is
the classic "two white layers first, then build the black panel up on top" trick — the
white letters are the first ~2 layers on the bed (mirrored, so they read correctly
through the front), then the rest is black.

In Bambu Studio: import both STLs together (keep positions), assign **white → labels,
black → base**, print **by object with a Z hop / clearance** so the toolhead can't drag
the freshly-laid white letters before the black locks them in. No supports — the ribs
are raised features on the upward (back) face.

Run / export
------------
  ./.venv/bin/python panels/patch_init_oled.py                      # view assembly
  ./.venv/bin/python panels/patch_init_oled.py --stl-base   base.stl
  ./.venv/bin/python panels/patch_init_oled.py --stl-labels labels.stl
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    Cylinder,
    FontStyle,
    Location,
    Locations,
    Mode,
    Plane,
    Rectangle,
    SlotOverall,
    Text,
    extrude,
    export_stl,
)
from ocp_vscode import Camera, show

# ===========================================================================
# Panel parameters
# ===========================================================================


@dataclass(frozen=True)
class PanelParams:
    # Panel dimensions
    panel_w: float = 50.8  # 10HP
    panel_h: float = 128.5
    thickness: float = 2.0

    # Text mode:
    #   "emboss" — RECOMMENDED. Prints FACE-UP (flat back on the bed) with the letters
    #              raised proud of the front. They're the only thing above z = thickness, so
    #              one height-based filament change at Z = thickness prints them white.
    #              Printed last onto solid material => crisp. Ribs go on a SEPARATE glue-on
    #              plate (see build_rib_plate), because ribs can only print pointing up and
    #              would otherwise force the panel face-down.
    #   "inlay"  — face-down 2-colour flush: white letter solids fill pockets in the front.
    #   "deboss" — face-down, recessed letters revealed by two whole-layer filament changes.
    #              Both face-down modes fight the over-extruded first layer and print mushy.
    text_mode: Literal["emboss", "inlay", "deboss"] = "emboss"
    deboss_depth: float = 0.3  # recess depth = black layers x layer height (3 x 0.1 mm)

    # The first layer is over-extruded for bed adhesion (0.25 mm line vs 0.22 mm), which
    # squeezes black into the thin letter voids and closes them up. Widen the void over the
    # FIRST LAYER ONLY to cancel that. Determined from the text_relief_test.py coupon.
    first_layer_relief: float = 0.04  # mm per side
    first_layer_h: float = 0.1        # must match the slicer's first layer height

    # Flush inlay labels (front face is up at z = thickness while modelling)
    label_height: float = 0.2  # two 0.1 mm layers of white
    label_font: str = "Arial"
    label_font_style: FontStyle = FontStyle.BOLD
    # Constrained by the 12.17 mm jack pitch: two neighbouring labels collide once
    # (w1 + w2)/2 exceeds it. The widest adjacent pair is GATE2|GATE1, which needs 12.42 mm
    # at 4.0 mm (they touch) but only 11.18 mm at 3.6 mm. 3.8 mm is the hard ceiling.
    # Bigger is better for printing (0.50 mm stems at 3.6 = ~2.3 x the 0.22 mm line width,
    # vs only ~2.1 at 3.2, which printed mushy) — so 3.6 is the practical compromise.
    label_size: float = 3.6
    label_below_offset: float = -7.0  # label sits this far below its hole centre (mm)

    # Branding
    brand_text_top: str = "DaisyMultiOsc"
    brand_text_bottom: str = "Eight4aWish"
    brand_size: float = 4.0
    brand_margin: float = 4.01  # from the relevant panel edge

    # Only label round holes at least this wide (skips mounting holes / LEDs)
    label_hole_min_d: float = 5.0

    # --- Back-side stiffening ribs --------------------------------------
    # Ribs are placed on explicit centrelines (physical/mounted panel coords, mm) that
    # sit in the gaps *between* component rows/columns, then trimmed by the keep-outs.
    rib_enable: bool = True
    rib_height: float = 3.0  # how far ribs stand off the back (mm)
    rib_thickness: float = 2.0
    # Vertical ribs as (x, y_start, y_end) in physical panel coords; None = full band.
    #  - edges run full height
    #  - centreline + the two jack-column-gap ribs run only the bottom (jack) half
    #  - two ribs between the LED/SD column and the pots run the pot area (above the board)
    rib_v: tuple[tuple[float, float | None, float | None], ...] = (
        (1.6, None, None),
        (49.2, None, None),
        (13.235, None, 64.0),     # jack gap, extended up to brace the button rib (y=63)
        (37.565, None, 64.0),     # jack gap, extended up to brace the jack rib (y=63)
        (25.4, None, 54.0),       # centreline: bottom half only
        (18.275, 78.0, None),     # pot area: between LED/SD and left pots
        (32.51, 78.0, None),      # pot area: between LED/SD and right pots
    )
    # Horizontal ribs as (y, x_start, x_end); None = full width. Bottom -> top: below
    # jacks, the two jack-row gaps, a full-width rib below the screen, two short ribs
    # under the button/jack either side of the screen, above the OLED board, between the
    # pot rows, and above the pots.
    rib_h: tuple[tuple[float, float | None, float | None], ...] = (
        (10.0, None, None),
        (23.4, None, None),
        (37.06, None, None),
        (52.0, None, None),
        (63.0, 1.6, 14.5),        # under the button left of the screen
        (63.0, 36.3, 49.2),       # under the jack right of the screen
        (78.8, None, None),
        (96.04, None, None),
        (116.0, None, None),
    )
    rail_margin: float = 8.0  # keep ribs out of the top/bottom rack-rail border
    edge_margin: float = 1.4  # keep ribs off the left/right edge
    rib_keepout_clearance: float = 0.5  # gap left around every component (mm)

    # --- Component preview / keep-out envelopes -------------------------
    # Rectangular body footprints (so a straight rib can pass between them).
    jack_body: tuple[float, float] = (8.6, 8.6)  # Thonkiconn PJ398SM housing
    jack_body_depth: float = 10.0
    pot_body: tuple[float, float] = (10.0, 11.0)  # Alpha RV09 9 mm
    pot_body_depth: float = 6.5
    mount_keepout_r: float = 4.5  # generous clearance around LED / M3 mount holes

    # Grove 0.66" SSD1306 board: mounting holes sit at the board's vertical centre,
    # the glass is offset from centre (verify against the real board).
    oled_board: tuple[float, float] = (20.0, 20.0)
    oled_board_center: tuple[float, float] = (25.4, 61.75)  # = OLED mount-hole midline
    oled_board_depth: float = 5.0
    oled_window_center: tuple[float, float] = (25.4, 59.0)
    oled_glass: tuple[float, float] = (14.0, 11.5)

    # Display colours (dark-grey panel, white lettering)
    base_color: tuple[float, float, float] = (0.028, 0.028, 0.032)
    label_color: tuple[float, float, float] = (0.92, 0.92, 0.93)
    jack_color: tuple[float, float, float] = (0.75, 0.66, 0.20)
    pot_color: tuple[float, float, float] = (0.20, 0.22, 0.28)
    oled_color: tuple[float, float, float] = (0.10, 0.25, 0.45)


# ===========================================================================
# Hole / cutout data (Daisy Patch Init OLED — from KiCad, B8 freed for the OLED)
# ===========================================================================


@dataclass(frozen=True)
class Hole:
    shape: Literal["circle", "oval"]
    x: float
    y: float
    drill: tuple[float, float]  # (major, minor) for oval; (d, d) for circle


@dataclass(frozen=True)
class RectCutout:
    x: float  # min X
    y: float  # min Y
    w: float
    h: float


OLED_Y_OFFSET = -3  # nudge OLED window + mounting holes up/down together

HOLES: tuple[Hole, ...] = (
    Hole(shape="circle", x=7.149999, y=111.900001, drill=(6.2, 6.2)),
    Hole(shape="circle", x=19.316694, y=111.900001, drill=(6.2, 6.2)),
    Hole(shape="circle", x=31.483297, y=111.900001, drill=(6.2, 6.2)),
    Hole(shape="circle", x=43.650000, y=111.900001, drill=(6.2, 6.2)),
    Hole(shape="circle", x=7.149999, y=98.311904, drill=(6.2, 6.2)),
    Hole(shape="circle", x=19.316694, y=98.311904, drill=(6.2, 6.2)),
    Hole(shape="circle", x=31.483297, y=98.311904, drill=(6.2, 6.2)),
    Hole(shape="circle", x=43.650000, y=98.311904, drill=(6.2, 6.2)),
    Hole(shape="circle", x=7.149999, y=84.561952, drill=(6.2, 6.2)),
    Hole(shape="circle", x=19.316694, y=84.561903, drill=(6.2, 6.2)),
    Hole(shape="circle", x=31.483297, y=84.561903, drill=(6.2, 6.2)),
    Hole(shape="circle", x=43.650000, y=84.561903, drill=(6.2, 6.2)),
    # B8 removed: was Hole(circle, 25.502894, 61.956597, 6.2) — now the OLED screen
    Hole(shape="circle", x=8.649995, y=59.288225, drill=(6.2, 6.2)),
    Hole(shape="circle", x=42.155087, y=59.288206, drill=(6.2, 6.2)),
    Hole(shape="circle", x=11.175500, y=42.027311, drill=(7.2, 7.2)),
    Hole(shape="circle", x=39.649996, y=42.027311, drill=(7.2, 7.2)),
    Hole(shape="circle", x=11.175500, y=22.904289, drill=(7.2, 7.2)),
    Hole(shape="circle", x=39.649996, y=22.904289, drill=(7.2, 7.2)),
    Hole(shape="circle", x=25.399999, y=19.252005, drill=(3.2, 3.2)),
    Hole(shape="oval", x=43.100000, y=125.500000, drill=(5.0, 3.0)),
    Hole(shape="oval", x=7.500000, y=3.000000, drill=(5.0, 3.0)),
    # OLED mounting holes (verify spacing against the real Grove 0.66" board)
    Hole(shape="circle", x=15.65, y=64.75 + OLED_Y_OFFSET, drill=(3.0, 3.0)),
    Hole(shape="circle", x=35.15, y=64.75 + OLED_Y_OFFSET, drill=(3.0, 3.0)),
)

RECT_CUTOUTS: tuple[RectCutout, ...] = (
    RectCutout(x=24.14, y=33.493, w=27.348 - 24.14, h=46.295 - 33.493),  # SD-card slot
    # OLED glass window: 14 x 11.5 mm
    RectCutout(x=18.4, y=56.25 + OLED_Y_OFFSET, w=14.0, h=11.5),
)

# Labels below each labelable hole, in KiCad top-to-bottom / left-to-right order
# (the order matches _labelable_holes; the physical layout is flipped at build time).
HOLE_LABELS: tuple[str, ...] = (
    "OUT-R", "OUT-L", "", "",
    "MOD3", "MOD2", "MOD1", "V/OCT",
    "", "", "GATE2", "GATE1",
    "", "SELECT",
    "MOD3", "MOD2", "MOD1", "TUNE",
)
HOLE_LABELS_ABOVE: tuple[str, ...] = tuple("" for _ in HOLE_LABELS)


# ===========================================================================
# Component classification (drill diameter -> mounted part type)
# ===========================================================================

PartType = Literal["jack", "pot", "mount", "rail", "other"]


def classify(hole: Hole) -> PartType:
    if hole.shape == "oval":
        return "rail"
    d = hole.drill[0]
    if 5.8 <= d <= 6.8:
        return "jack"
    if 6.9 <= d <= 7.8:
        return "pot"
    if d <= 3.6:
        return "mount"
    return "other"


# ===========================================================================
# Helpers
# ===========================================================================


def _phys(x: float, y: float, params: PanelParams) -> tuple[float, float]:
    """KiCad coords -> physical (mounted) layout: 180° flip about the panel centre."""
    return (params.panel_w - x, params.panel_h - y)


def _text_local_offset(txt: str, *, font: str, style: FontStyle, font_size: float) -> tuple[float, float]:
    """Offset that recentres ``txt`` on the origin."""
    with BuildSketch(Plane.XY) as sk:
        Text(txt, font_size=font_size, font=font, font_style=style)
    bb = sk.sketch.bounding_box()
    return (-(bb.min.X + bb.max.X) / 2, -(bb.min.Y + bb.max.Y) / 2)


def _labelable_holes(params: PanelParams, holes: tuple[Hole, ...]) -> list[Hole]:
    hs = [h for h in holes if h.shape == "circle" and h.drill[0] >= params.label_hole_min_d]
    hs.sort(key=lambda h: (-h.y, h.x))  # KiCad top-to-bottom, left-to-right (label order)
    return hs


def _label_placements(
    params: PanelParams,
    holes: tuple[Hole, ...],
    labels_below: tuple[str, ...],
    labels_above: tuple[str, ...],
) -> list[tuple[str, float, float]]:
    """Return (text, x, y) in physical coords for every non-empty label + the brands."""
    out: list[tuple[str, float, float]] = []
    cx = params.panel_w / 2
    if params.brand_text_top.strip():
        out.append((params.brand_text_top, cx, params.panel_h - params.brand_margin))
    if params.brand_text_bottom.strip():
        out.append((params.brand_text_bottom, cx, params.brand_margin))

    for idx, h in enumerate(_labelable_holes(params, holes)):
        px, py = _phys(h.x, h.y, params)
        if idx < len(labels_below) and labels_below[idx].strip():
            out.append((labels_below[idx].strip(), px, py + params.label_below_offset))
        if idx < len(labels_above) and labels_above[idx].strip():
            out.append((labels_above[idx].strip(), px, py - params.label_below_offset))
    return out


def _add_text(txt: str, x: float, y: float, size: float, params: PanelParams,
              ox: float = 0.0, oy: float = 0.0) -> None:
    dx, dy = _text_local_offset(txt, font=params.label_font, style=params.label_font_style, font_size=size)
    with Locations(Location((x + dx + ox, y + dy + oy, 0))):
        Text(txt, font_size=size, font=params.label_font, font_style=params.label_font_style)


# 8 directions is ample: the dilation error is relief * (1 - cos(pi/8)) ~= 0.003 mm at
# relief 0.04, far below one line width. More steps just multiply the boolean cost.
_DILATE_STEPS = 8


def _add_text_dilated(txt: str, x: float, y: float, size: float, params: PanelParams,
                      relief: float) -> None:
    """Add `txt` dilated outward by `relief` to the active BuildSketch.

    OCC's 2D offset self-intersects on glyph counters (it returns an invalid shape past
    ~0.1 mm), so approximate a dilation by a disc: union the glyph with copies of itself
    translated `relief` in a ring of directions. Every copy is known-good geometry, so the
    result stays valid/manifold. The un-shifted copy is included so thin strokes can't end
    up with a gap down the middle."""
    _add_text(txt, x, y, size, params)
    for k in range(_DILATE_STEPS):
        th = 2.0 * math.pi * k / _DILATE_STEPS
        _add_text(txt, x, y, size, params, relief * math.cos(th), relief * math.sin(th))


def _is_brand(s: str, params: PanelParams) -> bool:
    return s in (params.brand_text_top, params.brand_text_bottom)


def _hole_sketch(params: PanelParams, holes: tuple[Hole, ...], rects: tuple[RectCutout, ...]) -> None:
    """Add every hole + rectangular cutout (physical coords) to the active BuildSketch."""
    for h in holes:
        px, py = _phys(h.x, h.y, params)
        with Locations((px, py)):
            if h.shape == "circle":
                Circle(h.drill[0] / 2)
            else:
                SlotOverall(h.drill[0], h.drill[1])
    for r in rects:
        px, py = _phys(r.x + r.w / 2, r.y + r.h / 2, params)  # physical centre
        with Locations((px, py)):
            Rectangle(r.w, r.h)


# ===========================================================================
# Build functions  (modelled front-face-up: front at z = thickness, ribs below)
# ===========================================================================


def build_base(
    params: PanelParams,
    holes: tuple[Hole, ...] = HOLES,
    rects: tuple[RectCutout, ...] = RECT_CUTOUTS,
    labels_below: tuple[str, ...] = HOLE_LABELS,
    labels_above: tuple[str, ...] = HOLE_LABELS_ABOVE,
) -> object:
    """Panel solid with holes cut through, flush text pockets in the front (top) face,
    and stiffening ribs hanging off the back."""
    t = params.thickness
    with BuildPart() as p:
        Box(params.panel_w, params.panel_h, t, align=(Align.MIN, Align.MIN, Align.MIN))

        # Cut all holes / windows (overshoot both faces so they're clean)
        with BuildSketch(Plane.XY.offset(-0.1)) as sk:
            _hole_sketch(params, holes, rects)
        extrude(to_extrude=sk.sketch, amount=t + 0.2, mode=Mode.SUBTRACT)

        placements = _label_placements(params, holes, labels_below, labels_above)

        def _size_of(s: str) -> float:
            return params.brand_size if _is_brand(s, params) else params.label_size

        # EMBOSS (face-up): raise the letters `label_height` proud of the front face. They
        # are then the ONLY geometry above z = thickness, so a single height-based filament
        # change at Z = thickness prints them white. Printed last, on top of solid material,
        # so they come out crisp — no bed squish, no isolated islands.
        if params.text_mode == "emboss":
            if params.label_height > 0:
                with BuildSketch(Plane.XY.offset(t)) as txt:
                    for s, x, y in placements:
                        _add_text(s, x, y, _size_of(s), params)
                extrude(to_extrude=txt.sketch, amount=params.label_height, mode=Mode.ADD)
            if params.rib_enable and params.rib_height > 0:
                ribs = build_ribs(params, holes, rects)
                if ribs is not None:
                    p.part += ribs
            return p.part

        # Recessed letters in the FRONT (top) face: z = t-depth .. t. In "inlay" mode the
        # pocket is filled by build_labels (white); in "deboss" mode the white is revealed
        # by two whole-layer filament changes while printing.
        depth = params.label_height if params.text_mode == "inlay" else params.deboss_depth

        if depth > 0:
            # True letter shape, full recess depth — this gives the crisp edge.
            with BuildSketch(Plane.XY.offset(t - depth)) as txt:
                for s, x, y in placements:
                    _add_text(s, x, y, _size_of(s), params)
            extrude(to_extrude=txt.sketch, amount=depth + 0.05, mode=Mode.SUBTRACT)

            # First layer only: widen the void so the over-extruded first layer can't
            # squeeze black into the letters and close them up.
            relief = params.first_layer_relief
            if params.text_mode == "deboss" and relief > 0:
                flh = min(params.first_layer_h, depth)
                with BuildSketch(Plane.XY.offset(t - flh)) as big:
                    for s, x, y in placements:
                        _add_text_dilated(s, x, y, _size_of(s), params, relief)
                extrude(to_extrude=big.sketch, amount=flh + 0.05, mode=Mode.SUBTRACT)

        # Back-side ribs (hang below the back face at z = 0)
        if params.rib_enable and params.rib_height > 0:
            ribs = build_ribs(params, holes, rects)
            if ribs is not None:
                p.part += ribs

    return p.part


def build_ribs(
    params: PanelParams,
    holes: tuple[Hole, ...] = HOLES,
    rects: tuple[RectCutout, ...] = RECT_CUTOUTS,
) -> object | None:
    """Lattice of vertical + horizontal ribs on the back (z = 0 .. -rib_height), trimmed
    to the rail-free band and cleared (rectangular keep-outs) around every component."""
    h = params.rib_height
    x_lo, x_hi = params.edge_margin, params.panel_w - params.edge_margin
    y_lo, y_hi = params.rail_margin, params.panel_h - params.rail_margin
    band_w, band_h = x_hi - x_lo, y_hi - y_lo
    if band_w <= 0 or band_h <= 0:
        return None

    # Verticals stop flush with the OUTER EDGES of the outermost horizontal ribs, rather
    # than running on to the rail band — otherwise they poke out as stubs top and bottom.
    hys = [y for y, _, _ in params.rib_h]
    v_lo = max(y_lo, min(hys) - params.rib_thickness / 2) if hys else y_lo
    v_hi = min(y_hi, max(hys) + params.rib_thickness / 2) if hys else y_hi

    clr = params.rib_keepout_clearance
    with BuildPart() as p:
        # Ribs grow downward from the back face (align MAX at z = 0)
        for x, v0, v1 in params.rib_v:
            ry0 = v_lo if v0 is None else max(v0, v_lo)
            ry1 = v_hi if v1 is None else min(v1, v_hi)
            if ry1 - ry0 <= 0:
                continue
            with Locations(Location((x, (ry0 + ry1) / 2, 0))):
                Box(params.rib_thickness, ry1 - ry0, h, align=(Align.CENTER, Align.CENTER, Align.MAX))
        for y, u0, u1 in params.rib_h:
            rx0 = x_lo if u0 is None else max(u0, x_lo)
            rx1 = x_hi if u1 is None else min(u1, x_hi)
            if rx1 - rx0 <= 0:
                continue
            with Locations(Location(((rx0 + rx1) / 2, y, 0))):
                Box(rx1 - rx0, params.rib_thickness, h, align=(Align.CENTER, Align.CENTER, Align.MAX))

        # Carve rectangular clearance around every component + cutout (physical coords)
        with BuildSketch(Plane.XY.offset(0.1)) as ko:
            for hole in holes:
                px, py = _phys(hole.x, hole.y, params)
                t = classify(hole)
                with Locations((px, py)):
                    if t == "jack":
                        Rectangle(params.jack_body[0] + 2 * clr, params.jack_body[1] + 2 * clr)
                    elif t == "pot":
                        Rectangle(params.pot_body[0] + 2 * clr, params.pot_body[1] + 2 * clr)
                    elif t == "mount":
                        Circle(params.mount_keepout_r)
                    elif hole.shape == "circle":
                        Circle(hole.drill[0] / 2 + clr)
                    else:
                        SlotOverall(hole.drill[0] + 2 * clr, hole.drill[1] + 2 * clr)
            # OLED board footprint
            obx, oby = _phys(*params.oled_board_center, params)
            with Locations((obx, oby)):
                Rectangle(params.oled_board[0] + 2 * clr, params.oled_board[1] + 2 * clr)
            # Rectangular cutouts (SD slot, OLED window)
            for r in rects:
                rx, ry = _phys(r.x + r.w / 2, r.y + r.h / 2, params)
                with Locations((rx, ry)):
                    Rectangle(r.w + 2 * clr, r.h + 2 * clr)
        extrude(to_extrude=ko.sketch, amount=-(h + 0.2), mode=Mode.SUBTRACT)

    return p.part


def build_labels(
    params: PanelParams,
    holes: tuple[Hole, ...] = HOLES,
    labels_below: tuple[str, ...] = HOLE_LABELS,
    labels_above: tuple[str, ...] = HOLE_LABELS_ABOVE,
) -> object:
    """White inlay solids that fill the base's front-face pockets (z = t-label_height .. t)."""
    t = params.thickness
    with BuildPart() as p:
        with BuildSketch(Plane.XY.offset(t - params.label_height)) as txt:
            for s, x, y in _label_placements(params, holes, labels_below, labels_above):
                _add_text(s, x, y, params.brand_size if _is_brand(s, params) else params.label_size, params)
        extrude(to_extrude=txt.sketch, amount=params.label_height, mode=Mode.ADD)
    return p.part


def build_components(params: PanelParams, holes: tuple[Hole, ...] = HOLES) -> dict[str, object]:
    """Show-only preview solids for each mounted part, grouped by type for colouring.
    Bodies hang behind the back face (z < 0), the same side as the ribs."""
    groups: dict[str, list[object]] = {"jack": [], "pot": [], "oled": []}
    t = params.thickness

    for hole in holes:
        kind = classify(hole)
        px, py = _phys(hole.x, hole.y, params)
        if kind == "jack":
            jw, jh = params.jack_body
            with BuildPart() as part:
                with Locations(Location((px, py, 0))):  # body behind the back face
                    Box(jw, jh, params.jack_body_depth, align=(Align.CENTER, Align.CENTER, Align.MAX))
                with Locations(Location((px, py, t))):  # nut on the front
                    Cylinder(4.5, 1.2, align=(Align.CENTER, Align.CENTER, Align.MIN))
            groups["jack"].append(part.part)
        elif kind == "pot":
            pw, ph = params.pot_body
            with BuildPart() as part:
                with Locations(Location((px, py, 0))):
                    Box(pw, ph, params.pot_body_depth, align=(Align.CENTER, Align.CENTER, Align.MAX))
                with Locations(Location((px, py, t))):  # shaft out the front
                    Cylinder(3.0, 7.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
            groups["pot"].append(part.part)

    # OLED board (behind) + glass (in the window)
    obx, oby = _phys(*params.oled_board_center, params)
    wx, wy = _phys(*params.oled_window_center, params)
    gw, gh = params.oled_glass
    with BuildPart() as oled:
        with Locations(Location((obx, oby, 0))):
            Box(params.oled_board[0], params.oled_board[1], params.oled_board_depth,
                align=(Align.CENTER, Align.CENTER, Align.MAX))
        with Locations(Location((wx, wy, t))):
            Box(gw, gh, 0.2, align=(Align.CENTER, Align.CENTER, Align.MAX))
    groups["oled"].append(oled.part)

    out: dict[str, object] = {}
    for name, parts in groups.items():
        if not parts:
            continue
        acc = parts[0]
        for extra in parts[1:]:
            acc = acc + extra
        out[name] = acc
    return out


# ===========================================================================
# Orientation transforms
# ===========================================================================


def view_transform(obj: object, params: PanelParams) -> object:
    """Centre only — front face up, as you'd look at the mounted module.

    This is also the FACE-UP print orientation used by `emboss`: the flat back sits on the
    bed and the raised letters are on top, printed last."""
    return obj.moved(Location((-params.panel_w / 2, -params.panel_h / 2, 0)))


def build_rib_plate(params: PanelParams, holes: tuple[Hole, ...] = HOLES,
                    rects: tuple[RectCutout, ...] = RECT_CUTOUTS) -> object | None:
    """The stiffening rib grid as a SEPARATE part, ready to print and glue to the panel back.

    Rotating 180° about Y does two things at once: it seats the *glue* face on the bed (so it
    comes out flat and smooth for a good bond) and mirrors X into the panel's back-view. So
    the print can be lifted straight off the bed and set glue-face-down onto the panel back,
    with the ribs standing away from it into the case — no flipping, no guessing handedness."""
    ribs = build_ribs(params, holes, rects)
    if ribs is None:
        return None
    centred = view_transform(ribs, params)
    flipped = centred.rotate(Axis.Y, 180)
    return flipped.moved(Location((0, 0, -flipped.bounding_box().min.Z)))


def export_transform(obj: object, params: PanelParams) -> object:
    """Flip front-down onto the bed for face-down printing (front at z = 0).

    Rotating 180° about Y both lands the front face on the bed and mirrors the lettering
    so it reads correctly through the front; ribs end up standing up off the bed."""
    centred = view_transform(obj, params)
    flipped = centred.rotate(Axis.Y, 180)
    return flipped.moved(Location((0, 0, params.thickness)))


# ===========================================================================
# Template SVG (1:1 alignment print, physical front view)
# ===========================================================================


def write_template_svg(path: Path, params: PanelParams, holes: tuple[Hole, ...] = HOLES,
                       rects: tuple[RectCutout, ...] = RECT_CUTOUTS) -> None:
    W, H = params.panel_w, params.panel_h
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" '
        f'viewBox="0 0 {W} {H}">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="none" stroke="black" stroke-width="0.2"/>',
    ]
    for h in holes:
        px, py = _phys(h.x, h.y, params)
        sy = H - py  # SVG y is top-down
        if h.shape == "circle":
            lines.append(f'<circle cx="{px}" cy="{sy}" r="{h.drill[0] / 2}" '
                         f'fill="none" stroke="black" stroke-width="0.2"/>')
        else:
            lines.append(f'<circle cx="{px}" cy="{sy}" r="0.5" fill="black"/>')
    for r in rects:
        px, py = _phys(r.x + r.w / 2, r.y + r.h / 2, params)
        lines.append(f'<rect x="{px - r.w / 2}" y="{H - py - r.h / 2}" width="{r.w}" height="{r.h}" '
                     f'fill="none" stroke="black" stroke-width="0.2"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines))


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Patch Init + Grove OLED 10HP faceplate")
    ap.add_argument("--stl-base", type=Path, default=None, help="Export the panel STL")
    ap.add_argument("--stl-ribs", type=Path, default=None,
                    help="Export the stiffening rib grid as a separate glue-on part (emboss mode)")
    ap.add_argument("--stl-labels", type=Path, default=None, help="Export labels (white) STL (inlay mode only)")
    ap.add_argument("--text-mode", choices=("emboss", "inlay", "deboss"), default="emboss",
                    help="emboss = face-up, raised letters, one filament change, ribs separate "
                         "(recommended); inlay/deboss = face-down variants")
    ap.add_argument("--label-height", type=float, default=None,
                    help="raised letter height (emboss) / recess depth (inlay, deboss) in mm")
    ap.add_argument("--template-svg", type=Path, default=None, help="Write a 1:1 alignment SVG")
    ap.add_argument("--integrated-ribs", action="store_true",
                    help="emboss only: put the ribs on the panel itself (forces it face-down; not printable face-up)")
    ap.add_argument("--no-ribs", action="store_true", help="no ribs at all")
    ap.add_argument("--no-preview", action="store_true", help="don't draw component preview in the viewer")
    args = ap.parse_args()

    kw = {} if args.label_height is None else {"label_height": args.label_height, "deboss_depth": args.label_height}
    face_up = args.text_mode == "emboss"
    # Face-up printing cannot carry the ribs: they'd hang below the bed. So in emboss mode
    # the panel is built rib-free and the ribs become a separate glue-on plate.
    ribs_on_panel = (not args.no_ribs) and (not face_up or args.integrated_ribs)
    params = PanelParams(rib_enable=ribs_on_panel, text_mode=args.text_mode, **kw)
    rib_params = PanelParams(rib_enable=not args.no_ribs, text_mode=args.text_mode, **kw)
    inlay = params.text_mode == "inlay"

    orient = view_transform if face_up else export_transform

    if args.template_svg is not None:
        write_template_svg(args.template_svg, params)
        print(f"Wrote template {args.template_svg}")

    if args.stl_base is not None:
        export_stl(orient(build_base(params), params), args.stl_base)
        print(f"Wrote {args.stl_base}  ({'FACE-UP' if face_up else 'face-down'}"
              f"{', ribs separate' if face_up and not args.integrated_ribs else ''})")
    if args.stl_ribs is not None:
        plate = build_rib_plate(rib_params)
        if plate is None:
            print("Skipping --stl-ribs: ribs are disabled")
        else:
            export_stl(plate, args.stl_ribs)
            print(f"Wrote {args.stl_ribs}  (glue face on the bed, ribs standing up)")
    if args.stl_labels is not None:
        if not inlay:
            print(f"Skipping --stl-labels: {params.text_mode} mode is a single STL")
        else:
            export_stl(orient(build_labels(params), params), args.stl_labels)
            print(f"Wrote {args.stl_labels}")

    # Viewer: mounted orientation, panel + (separate) ribs behind it + component preview
    base = view_transform(build_base(params), params)
    comps = {} if args.no_preview else {
        k: view_transform(v, params) for k, v in build_components(params).items()
    }

    objs = [base]
    names = ["base"]
    colors = [params.base_color]
    if face_up and not args.no_ribs and not args.integrated_ribs:
        ribs = build_ribs(rib_params)
        if ribs is not None:
            objs.append(view_transform(ribs, rib_params))
            names.append("ribs (separate)")
            colors.append((0.35, 0.35, 0.38))
    if inlay:  # white inlay is a separate part only in inlay mode
        objs.append(view_transform(build_labels(params), params))
        names.append("labels")
        colors.append(params.label_color)
    objs += list(comps.values())
    names += list(comps.keys())
    cmap = {"jack": params.jack_color, "pot": params.pot_color, "oled": params.oled_color}
    colors += [cmap[k] for k in comps]

    show(*objs, names=names, colors=colors, reset_camera=Camera.RESET, grid=True)


if __name__ == "__main__":
    main()
