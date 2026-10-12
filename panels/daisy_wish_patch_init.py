"""Wish patch.Init() 10HP faceplate - face-up 2-colour printing.

Sorrow's panel (daisy_grids.py), the stock patch.init() hole set from KiCad, with Wish
patch.Init()'s labels (eurorack_daisy_patch_init/daisy_beads_patch_init/src/main.cpp).
The four pots carry two layers: PLAY above each pot (toggle down), TONE below it (toggle
up). The third layer, RANDOM, is B7 held and is not printed. The toggle is labelled per
throw, TONE up and PLAY down, which is also how the renderer knows it is a toggle.

- Base panel solid (printed first, black filament)
- Raised labels solid (printed on top, white filament)

Export STLs:
  ./.venv/bin/python panels/daisy_wish_patch_init.py --stl-base base.stl --stl-labels labels.stl

In Bambu Studio: import both STLs together as one object, black to base, white to
labels, print face-up.
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
    "IN-R",# IN-R
    "IN-L",# IN-L
    "DENS",# CV_8   DENSITY CV
    "SIZE",# CV_7
    "TIME",# CV_6
    "V/OCT",# CV_5
    "",# B6     gate out 2, unused
    "GATE",# B5    gate out 1: Beads' grain gates
    "FREEZE",# B9  gate in 2
    "SEED",# B10   gate in 1
    "PLAY",# B8    toggle down
    "",# C10       CV out 1, unused
    "FREEZE",# B7  tap; hold for RANDOM
    "REVERB",# CV_4
    "FEEDBACK",# CV_3
    "DRY/WET",# CV_2
    "SHAPE",# CV_1
)

# The pots' PLAY layer above each pot; their TONE layer is the label below.
LABELS_SECONDARY = ("SHAPE", "DRY/WET", "FEEDBACK", "REVERB")

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
    "TONE",# B8    toggle up
    "",# C10
    "",# B7
    "PITCH",# CV_4
    "DENSITY",# CV_3
    "SIZE",# CV_2
    "TIME",# CV_1
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
    brand_text_top: str = "Wish"
    brand_text_bottom: str = "Eight4aWish"
    brand_size: float = 4.0

    # Minimum hole diameter to label (excludes mounting holes)
    label_hole_min_d: float = 5.0

    # SD slot present but empty (no card) for renders.
    sd_card_present: bool = False

    # Befaco nut colours: the gate output gold; pitch and gate inputs silver.
    nut_overrides: tuple[tuple[str, str], ...] = (
        ("GATE", "nut_gold"), ("V/OCT", "nut_silver"),
        ("SEED", "nut_silver"), ("FREEZE", "nut_silver"),
    )

    # B7 is labelled FREEZE like gate in 2, so the renderer is told which hole is
    # the button (KiCad coordinates, as HOLES).
    button_holes: tuple[tuple[float, float], ...] = ((42.155087, 59.288206),)

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

    parser = argparse.ArgumentParser(description="Wish patch.Init() (daisy_beads_patch_init) faceplate")
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
