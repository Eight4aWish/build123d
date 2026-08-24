"""Interval Oscillator 10HP faceplate - Face-up 2-color printing.

This script generates a Eurorack faceplate with:
- Base panel solid (printed first, black filament)
- Raised labels solid (printed on top, white filament)

The panel prints FACE-UP with embossed labels on the top surface.

Run:
  ./.venv/bin/python interval_osc_simple.py

Export STLs:
  ./.venv/bin/python daisy_patch_init.py --stl-base base.stl --stl-labels labels.stl

In Bambu Studio:
1. Import both STLs together (select both files, choose "Yes" to combine as one object)
2. Assign filaments: black to base, white to labels
3. Print face-up - labels are raised on the top surface
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import math

from build123d import (
    Align,
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    FontStyle,
    Location,
    Locations,
    Mode,
    Plane,
    Rectangle,
    RectangleRounded,
    SlotOverall,
    Text,
    extrude,
    export_stl,
)
from ocp_vscode import Camera, show

# ---------------------------------------------------------------------------
# Label Configuration
# ---------------------------------------------------------------------------

# Labels placed below each hole (top-to-bottom, left-to-right ordering)
HOLE_LABELS = (
    "OUT-R",# OUT-R
    "OUT-L",# OUT-L
    "",# IN-R
    "",# IN-L
    "CHAOS",# CV_8
    "DENS",# CV_7
    "Y",# CV_6
    "X",# CV_5
    "SNARE",# B6
    "KICK",# B5
    "RESET",# B9
    "CLOCK",# B10
    "FLIP",# B8
    "HAT",# C10
    "MODE",# B7
    "WILDNESS",# CV_4
    "DENS-HT",# CV_3
    "DENS-SN",# CV_2
    "DENS-KC",# CV_1
)

# The four pots carry both pages: the Home name ABOVE the pot, the Kit name
# BELOW it and one size smaller (LABELS_SECONDARY, further down). Everything else
# on the panel keeps its single label below the hole.
#
# Labels placed ABOVE each hole. For the pots this is the Home page: a short press on B7 pulses
# the LED once and the four pots stop being X / Y / DENSITY / CHAOS and become the
# per-drum density trims plus wildness (main.cpp, `if(g_sub_mode == 1)`).
#
# A panel that prints only the Home row is documenting half the module, and the
# half it leaves out contains WILDNESS - which is the control the whole module is
# named around. Bracketed so the alternate reads as an alternate.
# Rendered at 78% and in italic. Size alone was doing the work of saying "this is
# the other page", and it was not quite enough at a glance - the slant is what
# makes a secondary read as a secondary rather than as a shorter word.
LABELS_SECONDARY = ("DENS-KC", "DENS-SN", "DENS-HT", "WILDNESS")

HOLE_LABELS_ABOVE = (
    "",# OUT-R
    "",# OUT-L
    "",# IN-R
    "",# IN-L
    "",# CV_8
    "",# CV_7
    "",# CV_6
    "",# CV_5
    "",# B6
    "",# B5
    "",# B9
    "",# B10
    "",# B8
    "",# C10
    "",# B7
    "CHAOS",# CV_4
    "DENSITY",# CV_3
    "Y",# CV_2
    "X",# CV_1
)

# ---------------------------------------------------------------------------
# Hole Definitions (from KiCad)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Hole:
    shape: Literal["circle", "oval"]
    x: float
    y: float
    drill: tuple[float, float]  # (major, minor) for oval; (d, d) for circle


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
    Hole(shape="circle", x=25.502894, y=61.956597, drill=(6.2, 6.2)),
    Hole(shape="circle", x=8.649995, y=59.288225, drill=(6.2, 6.2)),
    Hole(shape="circle", x=42.155087, y=59.288206, drill=(6.2, 6.2)),
    Hole(shape="circle", x=11.175500, y=42.027311, drill=(7.2, 7.2)),
    Hole(shape="circle", x=39.649996, y=42.027311, drill=(7.2, 7.2)),
    Hole(shape="circle", x=11.175500, y=22.904289, drill=(7.2, 7.2)),
    Hole(shape="circle", x=39.649996, y=22.904289, drill=(7.2, 7.2)),
    Hole(shape="circle", x=25.399999, y=19.252005, drill=(3.2, 3.2)),
    Hole(shape="oval", x=43.100000, y=125.500000, drill=(5.0, 3.0)),
    Hole(shape="oval", x=7.500000, y=3.000000, drill=(5.0, 3.0)),
)

# ---------------------------------------------------------------------------
# Rectangular Cutouts (from KiCad Edge.Cuts)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RectCutout:
    x: float  # min X
    y: float  # min Y
    w: float  # width
    h: float  # height


# Offset rectangle cutout (display window/slot)
RECT_CUTOUTS: tuple[RectCutout, ...] = (
    RectCutout(x=24.14, y=33.493, w=27.348 - 24.14, h=46.295 - 33.493),
)

# ---------------------------------------------------------------------------
# Panel Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PanelParams:
    # Panel dimensions
    panel_w: float = 50.8  # 10HP
    panel_h: float = 128.5
    thickness: float = 2.0

    # Label settings
    label_height: float = 0.2  # Two 0.1mm layers
    label_font: str = "Arial"
    label_font_style: FontStyle = FontStyle.BOLD
    label_size: float = 3.2
    label_offset: tuple[float, float] = (0.0, -7.0)  # Offset from hole center

    # Branding
    brand_text_top: str = "Sorrow"
    brand_text_bottom: str = "Eight4aWish"
    brand_size: float = 4.0

    # Minimum hole diameter to label (excludes mounting holes)
    label_hole_min_d: float = 5.0

    # SD slot present but empty (no card) for renders.
    sd_card_present: bool = False

    # Befaco nut colours: drum trigger outputs (HAT/KICK/SNARE) gold.
    nut_overrides: tuple[tuple[str, str], ...] = (
        ("HAT", "nut_gold"), ("KICK", "nut_gold"), ("SNARE", "nut_gold"),
    )

    # Display colors — real-life finish: dark grey panel, white lettering.
    base_color: tuple[float, float, float] = (0.028, 0.028, 0.032)
    label_color: tuple[float, float, float] = (0.92, 0.92, 0.93)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _text_local_offset(
    txt: str,
    *,
    font: str,
    style: FontStyle,
    font_size: float,
) -> tuple[float, float]:
    """Compute offset to center text at the origin."""
    with BuildSketch(Plane.XY) as sk:
        Text(txt, font_size=font_size, font=font, font_style=style)
    bb = sk.sketch.bounding_box()
    ax = (bb.min.X + bb.max.X) / 2
    ay = (bb.min.Y + bb.max.Y) / 2
    return (-ax, -ay)


def _get_labelable_holes(params: PanelParams) -> list[Hole]:
    """Get holes that should have labels (circles >= min diameter)."""
    holes = [h for h in HOLES if h.shape == "circle" and h.drill[0] >= params.label_hole_min_d]
    # Sort: top-to-bottom, left-to-right
    holes.sort(key=lambda h: (-h.y, h.x))
    return holes


def _brand_positions(params: PanelParams) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return (top_xyrot, bottom_xyrot) for branding text."""
    x = params.panel_w / 2
    margin = 4.01
    # Labels are flipped 180° so swap top/bottom
    return ((x, margin, 180.0), (x, params.panel_h - margin, 180.0))


# ---------------------------------------------------------------------------
# Build Functions
# ---------------------------------------------------------------------------

def build_base(params: PanelParams) -> object:
    """Build the base panel with holes cut through."""
    with BuildPart() as p:
        # Main panel box
        Box(
            params.panel_w,
            params.panel_h,
            params.thickness,
            align=(Align.MIN, Align.MIN, Align.MIN),
        )

        # Cut all holes and rectangular cutouts
        with BuildSketch(Plane.XY.offset(-0.1)) as sk:
            for h in HOLES:
                with Locations((h.x, h.y)):
                    if h.shape == "circle":
                        Circle(h.drill[0] / 2)
                    else:
                        SlotOverall(h.drill[0], h.drill[1])

            # Rectangular cutouts (e.g., display window)
            for r in RECT_CUTOUTS:
                with Locations((r.x, r.y)):
                    Rectangle(r.w, r.h, align=(Align.MIN, Align.MIN))

        extrude(to_extrude=sk.sketch, amount=params.thickness + 0.2, mode=Mode.SUBTRACT)

    return p.part


def build_labels(params: PanelParams) -> object:
    """Build the raised labels on top of the panel."""
    
    def add_text_extrusion(txt: str, x: float, y: float, rot: float, font_size: float) -> None:
        dx, dy = _text_local_offset(txt, font=params.label_font, style=params.label_font_style, font_size=font_size)
        with BuildSketch(Plane.XY.offset(params.thickness)) as sk:
            with Locations(Location((x, y, 0), (0, 0, rot))):
                with Locations((dx, dy)):
                    Text(txt, font_size=font_size, font=params.label_font, font_style=params.label_font_style)
        extrude(to_extrude=sk.sketch, amount=params.label_height, mode=Mode.ADD)

    with BuildPart() as p:
        # Branding
        (top_x, top_y, top_rot), (bot_x, bot_y, bot_rot) = _brand_positions(params)
        if params.brand_text_top.strip():
            add_text_extrusion(params.brand_text_top, top_x, top_y, top_rot, params.brand_size)
        if params.brand_text_bottom.strip():
            add_text_extrusion(params.brand_text_bottom, bot_x, bot_y, bot_rot, params.brand_size)

        # Hole labels
        dx_off, dy_off = -params.label_offset[0], -params.label_offset[1]  # Flip offset
        rot = 180.0  # Flip text orientation

        holes = _get_labelable_holes(params)
        for idx, h in enumerate(holes):
            # Below label
            if idx < len(HOLE_LABELS):
                txt = HOLE_LABELS[idx].strip()
                if txt:
                    add_text_extrusion(txt, h.x + dx_off, h.y + dy_off, rot, params.label_size)

            # Above label
            if idx < len(HOLE_LABELS_ABOVE):
                txt = HOLE_LABELS_ABOVE[idx].strip()
                if txt:
                    add_text_extrusion(txt, h.x + dx_off, h.y - dy_off, rot, params.label_size)

    return p.part


def export_transform(obj: object, params: PanelParams) -> object:
    """Center the object and mirror for correct orientation."""
    from build123d import Plane as B123dPlane

    # Center at origin
    result = obj.moved(Location((-params.panel_w / 2, -params.panel_h / 2, 0)))
    # Mirror so cutouts align with hardware
    result = result.mirror(B123dPlane.YZ)
    return result


def export_transform_labels(obj: object, params: PanelParams) -> object:
    """Center labels only (no mirror - text must read correctly)."""
    return obj.moved(Location((-params.panel_w / 2, -params.panel_h / 2, 0)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sorrow (daisy_grids) faceplate")
    parser.add_argument("--stl-base", type=Path, default=None, help="Export base STL")
    parser.add_argument("--stl-labels", type=Path, default=None, help="Export labels STL")

    args = parser.parse_args()

    params = PanelParams()

    # Build parts
    base = build_base(params)
    labels = build_labels(params)

    # Transform for export/display
    base_out = export_transform(base, params)
    labels_out = export_transform_labels(labels, params)

    # Export STLs
    if args.stl_base is not None:
        export_stl(base_out, args.stl_base)
    if args.stl_labels is not None:
        export_stl(labels_out, args.stl_labels)

    # Display
    show(
        base_out,
        labels_out,
        names=["base", "labels"],
        colors=[params.base_color, params.label_color],
        reset_camera=Camera.RESET,
        grid=True,
    )


if __name__ == "__main__":
    main()
