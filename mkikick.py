"""mkikick 6HP 2x6 faceplate.

Based on teensyexpander.py. All 12 control holes (2 columns x 6 rows) are
enabled except:

  - (col=1, row=1) "MOD" on the DRM1 row is removed entirely
  - (col=0, row=0) "OUT 3" is enlarged to 9 mm (replacing teensyexpander's
    default 6.9 mm) and its label is shifted down to clear the bigger hole

These local overrides are applied here so teensyexpander.py stays untouched.

Run:
    ./.venv/bin/python mkikick.py

Export STLs:
    ./.venv/bin/python mkikick.py --export-mode base   --stl mkikick_base.stl
    ./.venv/bin/python mkikick.py --export-mode labels --stl mkikick_labels.stl
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from build123d import (
    BuildPart,
    BuildSketch,
    Circle,
    Location,
    Locations,
    Mode,
    Plane,
    Text,
    export_step,
    export_stl,
    extrude,
)
from ocp_vscode import Camera, show

from teensyexpander import (
    FaceplateParams,
    build_faceplate,
)


def _ocp_port(default: int = 3939) -> int:
    try:
        raw = os.environ.get("OCP_VSCODE_PORT") or os.environ.get("OCP_PORT") or str(default)
        return int(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# mkikick-specific overrides
# ---------------------------------------------------------------------------
# MOD (col=1, row=1) is removed. OUT 3 (col=0, row=0) is also removed from the
# standard 6.9 mm hole set so we can cut a larger 9 mm hole in its place.
MKIKICK_REMOVE_HOLES = ((1, 1), (0, 0))

# Enlarged OUT 3 hole
OUT3_COL, OUT3_ROW = 0, 0
OUT3_DIAMETER = 9.0  # mm

# Label offset for OUT 3 (panel-space dy; negative = below the hole).
# Default is -7.0 for a 6.9 mm hole; with a 9 mm hole we need ~1 mm more
# clearance so the label sits cleanly below the enlarged opening.
OUT3_LABEL_OFFSET = (0.0, -8.5)

# Additional holes to enlarge beyond teensyexpander's 6.9 mm default.
# These are cut on top of the default hole (overlay subtract); labels stay
# at their default offset since the size change is small.
# Format: (col_idx, row_idx, diameter_mm)
EXTRA_HOLE_CUTS = (
    (1, 0, 7.5),  # DECAY
    (0, 1, 7.5),  # PIT-CV (upper)
    (0, 2, 7.5),  # TN-DEC
    (1, 2, 7.5),  # TN-DEP
    (0, 3, 7.5),  # DIST
    (1, 3, 7.5),  # TONE
)

# Per-hole labels (6 rows, top-to-bottom in panel coords).
# remove_holes entries are still rendered as "" when shown below, but build_labels
# and build_base both skip disabled holes, so the text is used only when the hole
# is active. OUT 3 (labels_left[0]) is rendered by _apply_out3_label regardless.
MKIKICK_LABELS_LEFT  = ("PITCH", "PIT-CV", "TN-DEC", "DIST", "PIT-CV", "OUT")
MKIKICK_LABELS_RIGHT = ("DECAY",  "",  "TN-DEP", "TONE", "ACCT", "TRIG")

# Optional small secondary labels under each main label ("" for none).
MKIKICK_SECONDARY_LEFT  = ("", "", "", "", "", "")
MKIKICK_SECONDARY_RIGHT = ("", "", "", "", "", "")

MKIKICK_PARAMS = replace(
    FaceplateParams(),
    remove_holes=MKIKICK_REMOVE_HOLES,
    labels_left=MKIKICK_LABELS_LEFT,
    labels_right=MKIKICK_LABELS_RIGHT,
    secondary_left=MKIKICK_SECONDARY_LEFT,
    secondary_right=MKIKICK_SECONDARY_RIGHT,
    brand_text="mkikick",
    model_text="Eight4aWish",
)


# ---------------------------------------------------------------------------
# Post-processing: enlarged OUT 3 hole + custom-offset OUT 3 label
# ---------------------------------------------------------------------------
def _out3_hole_xy(params: FaceplateParams) -> tuple[float, float]:
    x = params.col_x[OUT3_COL]
    y = params.row_y[OUT3_ROW] + params.labels_y_offset + params.content_y_offset
    return (x, y)


def _apply_out3_hole(base, params: FaceplateParams):
    """Cut a 9 mm through-hole at OUT 3's position on the base solid."""
    if base is None:
        return base

    x, y = _out3_hole_xy(params)
    cut_z0 = -0.2
    cut_h = params.thickness + 0.4

    with BuildPart() as cutter:
        with BuildSketch(Plane.XY.offset(cut_z0)):
            with Locations((x, y)):
                Circle(OUT3_DIAMETER / 2)
        extrude(amount=cut_h, mode=Mode.ADD)

    return base - cutter.part


def _apply_out3_label(labels, params: FaceplateParams):
    """Add the OUT 3 label with an enlarged-hole-friendly offset."""
    if labels is None:
        return labels
    # Skip if the label text is empty or text is engraved into the base.
    text = params.labels_left[OUT3_ROW] if OUT3_ROW < len(params.labels_left) else ""
    if not text or params.text_mode == "deboss":
        return labels

    hx, hy = _out3_hole_xy(params)

    # Translate panel-space (dx, dy) into internal coords; rotate_labels flips the sign.
    odx, ody = OUT3_LABEL_OFFSET
    if params.rotate_labels:
        odx, ody = -odx, -ody
    lx, ly = hx + odx, hy + ody
    rot = params._label_rotation()

    # emboss/inlay both build a text solid on/above the panel top face.
    z_base = params.thickness if params.text_mode == "emboss" else params.thickness - max(0.0, float(params.inlay_depth))
    height = params.label_height if params.text_mode == "emboss" else max(0.0, float(params.inlay_depth))
    if height <= 0:
        return labels

    with BuildPart() as tp:
        with BuildSketch(Plane.XY.offset(z_base)) as sk:
            with Locations(Location((lx, ly, 0), (0, 0, rot))):
                Text(
                    text,
                    font_size=params.label_size,
                    font=params.label_font,
                    font_style=params.label_font_style,
                )
        extrude(to_extrude=sk.sketch, amount=height, mode=Mode.ADD)

    return labels + tp.part


def _apply_extra_hole_cuts(base, params: FaceplateParams):
    """Overlay-cut larger-than-default holes listed in EXTRA_HOLE_CUTS."""
    if base is None or not EXTRA_HOLE_CUTS:
        return base

    cut_z0 = -0.2
    cut_h = params.thickness + 0.4
    dy = params.content_y_offset

    with BuildPart() as cutter:
        with BuildSketch(Plane.XY.offset(cut_z0)):
            for col, row, d in EXTRA_HOLE_CUTS:
                x = params.col_x[col]
                y = params.row_y[row] + params.labels_y_offset + dy
                with Locations((x, y)):
                    Circle(d / 2)
        extrude(amount=cut_h, mode=Mode.ADD)

    return base - cutter.part


def build_mkikick(params: FaceplateParams, export_mode: str = "combined"):
    base, labels = build_faceplate(params, export_mode=export_mode)
    base = _apply_out3_hole(base, params)
    base = _apply_extra_hole_cuts(base, params)
    labels = _apply_out3_label(labels, params)
    return base, labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build mkikick 6HP 2x6 faceplate")
    parser.add_argument("--export-mode", choices=("combined", "base", "labels"), default="combined")
    parser.add_argument("--stl",         type=Path, default=None)
    parser.add_argument("--stl-base",    type=Path, default=None)
    parser.add_argument("--stl-labels",  type=Path, default=None)
    parser.add_argument("--step",        type=Path, default=None)
    parser.add_argument("--text-mode",   choices=("emboss", "deboss", "inlay"), default="emboss")
    parser.add_argument("--inlay-depth", type=float, default=0.2)
    args = parser.parse_args()

    params = replace(MKIKICK_PARAMS, text_mode=args.text_mode, inlay_depth=float(args.inlay_depth))

    if params.text_mode in ("deboss", "inlay") and params.inverse_label_enable:
        params = replace(params, inverse_label_enable=False)

    base, labels = build_mkikick(params, export_mode=args.export_mode)

    if args.stl is not None:
        if args.export_mode == "combined":
            from build123d import Compound
            export_stl(Compound([o for o in (base, labels) if o is not None]), args.stl)
        elif args.export_mode == "base" and base is not None:
            export_stl(base, args.stl)
        elif args.export_mode == "labels" and labels is not None:
            export_stl(labels, args.stl)

    if args.stl_base is not None and base is not None:
        export_stl(base, args.stl_base)
    if args.stl_labels is not None and labels is not None:
        export_stl(labels, args.stl_labels)

    if args.step is not None:
        if args.export_mode == "combined":
            from build123d import Compound
            export_step(Compound([o for o in (base, labels) if o is not None]), args.step)
        elif args.export_mode == "base" and base is not None:
            export_step(base, args.step)
        elif args.export_mode == "labels" and labels is not None:
            export_step(labels, args.step)

    try:
        if args.export_mode == "combined":
            show(
                base, labels,
                names=["base", "labels"],
                colors=[params.base_color, params.label_color],
                reset_camera=Camera.RESET, grid=True, port=_ocp_port(),
            )
        elif args.export_mode == "base" and base is not None:
            show(base, names=["base"], colors=[params.base_color],
                 reset_camera=Camera.RESET, grid=True, port=_ocp_port())
        elif args.export_mode == "labels" and labels is not None:
            show(labels, names=["labels"], colors=[params.label_color],
                 reset_camera=Camera.RESET, grid=True, port=_ocp_port())
    except RuntimeError as ex:
        print("\nOCP viewer is not reachable.")
        print("- Open 'OCP CAD Viewer' in VS Code and ensure the backend is running.")
        print(f"\nDetails: {ex}")


if __name__ == "__main__":
    main()
