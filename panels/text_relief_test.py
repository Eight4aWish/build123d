"""First-layer relief test coupon for the face-down colour-change lettering.

Why
---
Printing face-down, the slicer over-extrudes the FIRST layer for bed adhesion. That
squished plastic closes up the thin letter voids, so the lettering loses its fine detail.
The fix is to make the letter void **more generous on the first layer only**, so that
after squish the aperture ends up the right size — the un-squished layers above then
define the crisp edge.

This coupon prints the real panel label text at a sweep of first-layer relief values so
you can see which one gives crisp white letters.

Geometry (per row, all face-down):
  - z = 0 .. first_layer_h   : letter void DILATED outward by that row's `relief`
  - z = first_layer_h .. recess : letter void at its TRUE shape (crisp edge)
  - z = recess ..            : solid (this is where the white caps the letters)

Print (same recipe as the panel):
  - black from z = 0 to z = recess_depth   (front face, letter holes)
  - white for ~2 layers                    (caps/bridges the letters -> they read white)
  - black for the rest
Add the two filament changes on the layer slider BY Z HEIGHT.

Run:
  ./.venv/bin/python panels/text_relief_test.py
  ./.venv/bin/python panels/text_relief_test.py --stl exports/panels/text_relief_test.stl
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    BuildSketch,
    FontStyle,
    Location,
    Locations,
    Mode,
    Plane,
    Text,
    extrude,
    export_stl,
)
from ocp_vscode import Camera, show


@dataclass(frozen=True)
class TestParams:
    # Panel print setup: 0.2 mm nozzle, 0.1 mm layers.
    recess_depth: float = 0.3   # 3 black layers @ 0.1 mm, then white caps the letters
    first_layer_h: float = 0.1  # MUST match your slicer's FIRST layer height

    # The sweep: how far to dilate the letter void on the first layer (mm per side).
    # With a 0.2 mm nozzle the extrusion is ~0.2-0.25 mm wide, so useful relief is small.
    reliefs: tuple[float, ...] = (0.0, 0.04, 0.08, 0.12, 0.16, 0.20)

    # Sample text = real panel labels. Printed at TWO sizes per row so one print tells us
    # both (a) which relief works and (b) whether the panel's 3.2 mm text is simply too
    # fine. At 0.22 mm line width, 3.2 mm Arial Bold has ~0.45 mm strokes = ~2 lines;
    # 4.5 mm gives ~0.63 mm = ~3 lines.
    sample_text: str = "OUT-R CV 1"
    label_size: float = 3.2   # the panel's current size
    label_size2: float = 4.5  # a larger alternative
    value_size: float = 4.5   # the row marker (big enough to stay readable regardless)
    font: str = "Arial"
    style: FontStyle = FontStyle.BOLD

    thickness: float = 1.6
    # Face-up variant: match the production panels — 2.0 mm plate with the
    # letters raised 0.2 mm proud (2 layers @ 0.1 mm).
    top_thickness: float = 2.0
    raised_height: float = 0.2
    # Face-up variant sweeps FONT SIZE (one row per size) instead of relief.
    # Covers every size used across the modules: 2.1 (6HP secondary), 2.4 (4HP),
    # 2.6 (6HP brand), 3.0 (old N8), 3.2 (label standard), 3.6, 4.0 (brand
    # standard), 4.5.
    font_sizes: tuple[float, ...] = (2.1, 2.4, 2.6, 3.0, 3.2, 3.6, 4.0, 4.5)
    row_pitch: float = 11.0
    margin_x: float = 6.0
    margin_y: float = 7.0
    col_gap: float = 6.0

    base_color: tuple[float, float, float] = (0.028, 0.028, 0.032)


def _text_width(txt: str, size: float, p: TestParams) -> float:
    with BuildSketch(Plane.XY) as sk:
        Text(txt, font_size=size, font=p.font, font_style=p.style)
    return sk.sketch.bounding_box().size.X


def _columns(p: TestParams) -> tuple[float, float, float]:
    """x positions of the three columns: row marker, sample@label_size, sample@label_size2."""
    x0 = p.margin_x
    x1 = x0 + _text_width("0.00", p.value_size, p) + p.col_gap
    x2 = x1 + _text_width(p.sample_text, p.label_size, p) + p.col_gap
    return (x0, x1, x2)


def _plate_size(p: TestParams) -> tuple[float, float]:
    _, _, x2 = _columns(p)
    w = x2 + _text_width(p.sample_text, p.label_size2, p) + p.margin_x
    h = p.margin_y * 2 + p.row_pitch * len(p.reliefs)
    return (round(w, 1), round(h, 1))


def _row_y(p: TestParams, i: int, h: float) -> float:
    """Row centres, top row first."""
    top = h - p.margin_y - p.row_pitch / 2
    return top - i * p.row_pitch


def _add_row_text(p: TestParams, i: int, h: float, dx: float = 0.0, dy: float = 0.0) -> None:
    """Add one row (marker + sample at both sizes) to the active BuildSketch, shifted by (dx, dy)."""
    y = _row_y(p, i, h)
    x0, x1, x2 = _columns(p)
    for x, txt, size in (
        (x0, f"{p.reliefs[i]:.2f}", p.value_size),
        (x1, p.sample_text, p.label_size),
        (x2, p.sample_text, p.label_size2),
    ):
        with Locations(Location((x + dx, y + dy, 0))):
            Text(txt, font_size=size, font=p.font, font_style=p.style,
                 align=(Align.MIN, Align.CENTER))


def _add_row_text_dilated(p: TestParams, i: int, h: float, relief: float, steps: int = 16) -> None:
    """Add one row's text DILATED outward by `relief`, to the active BuildSketch.

    OCC's 2D offset self-intersects on glyph counters (it produces an invalid shape past
    ~0.1 mm), so instead we approximate a dilation by a disc: union the glyph with copies
    of itself translated `relief` in `steps` directions around a circle. Every copy is
    known-good geometry, so the union stays valid. The original (un-shifted) copy is
    included so thin strokes can't end up with a gap down the middle."""
    _add_row_text(p, i, h)  # original — keeps thin strokes filled
    for k in range(steps):
        th = 2.0 * math.pi * k / steps
        _add_row_text(p, i, h, relief * math.cos(th), relief * math.sin(th))


def build_coupon(p: TestParams) -> object:
    w, h = _plate_size(p)
    t = p.thickness

    with BuildPart() as part:
        Box(w, h, t, align=(Align.MIN, Align.MIN, Align.MIN))

        # 1) TRUE letter voids, full recess depth (these give the crisp edge)
        with BuildSketch(Plane.XY.offset(t - p.recess_depth)) as sk_true:
            for i in range(len(p.reliefs)):
                _add_row_text(p, i, h)
        extrude(to_extrude=sk_true.sketch, amount=p.recess_depth + 0.05, mode=Mode.SUBTRACT)

        # 2) DILATED voids, first layer only — one cut per row (each has its own relief)
        for i, relief in enumerate(p.reliefs):
            if relief <= 0:
                continue
            with BuildSketch(Plane.XY.offset(t - p.first_layer_h)) as sk_big:
                _add_row_text_dilated(p, i, h, relief)
            extrude(to_extrude=sk_big.sketch, amount=p.first_layer_h + 0.05, mode=Mode.SUBTRACT)

    return part.part


def _plate_size_top(p: TestParams) -> tuple[float, float]:
    """Plate size for the face-up FONT-SIZE sweep: marker column + one sample."""
    marker_w = _text_width("0.0", p.value_size, p)
    sample_w = max(_text_width(p.sample_text, s, p) for s in p.font_sizes)
    w = p.margin_x + marker_w + p.col_gap + sample_w + p.margin_x
    h = p.margin_y * 2 + p.row_pitch * len(p.font_sizes)
    return (round(w, 1), round(h, 1))


def build_coupon_top(p: TestParams) -> object:
    """Face-UP variant: letters RAISED on top, one row per FONT SIZE.

    This is the recipe the production panels moved to — print base-down, letters
    laid last on solid material. One filament change at Z = top_thickness prints
    them white. With no first-layer squish to fight, the interesting variable is
    legibility vs size, so each row prints the sample text at one of
    `font_sizes`, with the size as the row marker.
    """
    w, h = _plate_size_top(p)
    t = p.top_thickness
    x_marker = p.margin_x
    x_sample = x_marker + _text_width("0.0", p.value_size, p) + p.col_gap

    with BuildPart() as part:
        Box(w, h, t, align=(Align.MIN, Align.MIN, Align.MIN))
        with BuildSketch(Plane.XY.offset(t)) as sk:
            for i, size in enumerate(p.font_sizes):
                y = h - p.margin_y - p.row_pitch / 2 - i * p.row_pitch
                for x, txt, fs in (
                    (x_marker, f"{size:.1f}", p.value_size),
                    (x_sample, p.sample_text, size),
                ):
                    with Locations(Location((x, y, 0))):
                        Text(txt, font_size=fs, font=p.font, font_style=p.style,
                             align=(Align.MIN, Align.CENTER))
        extrude(to_extrude=sk.sketch, amount=p.raised_height, mode=Mode.ADD)

    return part.part


def export_transform(obj: object, p: TestParams) -> object:
    """Flip front-down onto the bed (front at z=0) and mirror, as the panel does."""
    w, h = _plate_size(p)
    centred = obj.moved(Location((-w / 2, -h / 2, 0)))
    flipped = centred.rotate(Axis.Y, 180)
    return flipped.moved(Location((0, 0, p.thickness)))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="First-layer relief test coupon")
    ap.add_argument("--stl", type=Path, default=None, help="Export the coupon STL")
    ap.add_argument("--top", action="store_true",
                    help="face-UP variant: letters raised on top, one row per font size")
    ap.add_argument("--sizes", type=float, nargs="+", default=None,
                    help="font-size sweep for --top in mm, e.g. --sizes 2.6 3.2 4.0")
    ap.add_argument("--first-layer", type=float, default=None,
                    help="your slicer's FIRST layer height in mm (must match, default 0.1)")
    ap.add_argument("--recess", type=float, default=None,
                    help="recess depth in mm = black layers x layer height (default 0.3)")
    ap.add_argument("--reliefs", type=float, nargs="+", default=None,
                    help="relief sweep in mm, e.g. --reliefs 0 0.04 0.08 0.12")
    args = ap.parse_args()

    kw = {}
    if args.first_layer is not None:
        kw["first_layer_h"] = args.first_layer
    if args.recess is not None:
        kw["recess_depth"] = args.recess
    if args.reliefs is not None:
        kw["reliefs"] = tuple(args.reliefs)
    if args.sizes is not None:
        kw["font_sizes"] = tuple(args.sizes)
    p = TestParams(**kw)
    if args.top:
        # Face-up: prints as modelled (base on the bed, letters on top).
        w, h = _plate_size_top(p)
        out = build_coupon_top(p)
        print(f"Coupon (face-UP, letters on top) {w} x {h} x {p.top_thickness}+{p.raised_height} mm")
        print(f"  print base-down; ONE filament change at Z = {p.top_thickness} mm -> white letters")
        print(f"  font-size sweep: {', '.join(f'{s:.1f}' for s in p.font_sizes)} mm")
    else:
        coupon = build_coupon(p)
        out = export_transform(coupon, p)
        w, h = _plate_size(p)
        print(f"Coupon (face-DOWN) {w} x {h} x {p.thickness} mm")
        print(f"  recess depth   : {p.recess_depth} mm  <- black up to here, then white")
        print(f"  first layer    : {p.first_layer_h} mm  <- dilated zone")
        print(f"  relief sweep   : {', '.join(f'{r:.2f}' for r in p.reliefs)} mm")

    if args.stl is not None:
        export_stl(out, args.stl)
        print(f"Wrote {args.stl}")

    show(out, names=["coupon"], colors=[p.base_color], reset_camera=Camera.RESET, grid=True)


if __name__ == "__main__":
    main()
