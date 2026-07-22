"""Fondant cutter + embosser for parkrun celebration cake toppers.

One press does both jobs:
  - the outer rounded-rectangle wall is a tapered cutting blade that cuts the
    fondant into a 50 x 30 mm rounded rectangle, and
  - the recessed face inside the blade carries the design standing proud
    (mirrored), so the same press debosses "10 YR", "V50" and the parkrun
    icon into the surface.

Geometry, printed back-plate-down / features-up (no supports needed):
  z = 0 .. back_t                    : solid back plate
  z = back_t                         : the stamp face
  z = back_t .. back_t + emboss_h    : design features (MIRRORED)
  z = back_t .. back_t + cut_depth   : perimeter cutting blade, outside face
                                       tapered from wall_t down to blade_tip

Use: roll the fondant to about `cut_depth` (4.5 mm) thick, dust the stamp with
icing sugar / cornflour, press until the blade reaches the board. The face
meets the fondant surface just as the blade cuts through, so the design ends
up `emboss_h` deep.

The parkrun icon is imported from reference/parkrun_icon.svg (cleaned copy of
the official "About parkrun" homepage icon, background removed).

Run:
  ./.venv/bin/python stamps/parkrun_fondant_stamp.py
  ./.venv/bin/python stamps/parkrun_fondant_stamp.py --stl exports/stamps/parkrun_fondant_stamp.stl
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from build123d import (
    Align,
    BuildSketch,
    Face,
    FontStyle,
    Location,
    Plane,
    Pos,
    RectangleRounded,
    Sketch,
    Text,
    export_stl,
    extrude,
    import_svg,
    loft,
    mirror,
    scale,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ICON_SVG = REPO_ROOT / "reference" / "parkrun_icon.svg"


@dataclass(frozen=True)
class StampParams:
    # The cut fondant piece (blade inner profile)
    piece_w: float = 50.0
    piece_h: float = 30.0
    corner_r: float = 6.0

    # Cutting blade: vertical inner face (defines the piece), outer face
    # tapered from wall_t at the root to blade_tip at the cutting edge.
    wall_t: float = 1.8
    blade_tip: float = 0.5
    cut_depth: float = 4.5   # blade beyond the face; roll fondant to ~this

    back_t: float = 3.0      # solid back plate
    emboss_h: float = 1.8    # how proud the design stands = impression depth

    clearance: float = 1.5   # min gap between design and blade inner wall

    # parkrun icon: scaled to logo_d diameter, centred at (logo_cx, 0)
    logo_d: float = 22.0
    logo_cx: float = -11.5
    # The icon's strokes are ~0.66 mm at this scale — a bit fine to print and
    # press. Dilate outward by this much per side (union-of-translated-copies,
    # same trick as panels/text_relief_test.py; plain 2D offset self-intersects
    # on curvy glyph-like shapes).
    logo_boost: float = 0.15

    # Text block, right of the logo
    lines: tuple[str, ...] = ("10 YR", "V50")
    text_cx: float = 12.5
    line_y: tuple[float, ...] = (7.0, -7.0)
    text_max_size: float = 9.0
    text_max_w: float = 21.0
    font: str = "Arial"
    style: FontStyle = FontStyle.BOLD


def _text_width(txt: str, size: float, p: StampParams) -> float:
    with BuildSketch(Plane.XY) as sk:
        Text(txt, font_size=size, font=p.font, font_style=p.style)
    return sk.sketch.bounding_box().size.X


def _fitted_font_size(p: StampParams) -> float:
    """One font size for all lines, shrunk until the widest line fits."""
    widest = max(p.lines, key=lambda t: _text_width(t, p.text_max_size, p))
    w = _text_width(widest, p.text_max_size, p)
    return p.text_max_size * min(1.0, p.text_max_w / w)


def _logo_faces(p: StampParams) -> list[Face]:
    """Import the icon, centre it, scale to logo_d, dilate, move into place."""
    raw = [s for s in import_svg(str(ICON_SVG)) if isinstance(s, Face)]
    if not raw:
        raise RuntimeError(f"No faces imported from {ICON_SVG}")
    # The ring is the largest face; its bbox is the icon's circle.
    ring = max(raw, key=lambda f: f.area)
    bb = ring.bounding_box()
    cx, cy = bb.center().X, bb.center().Y
    k = p.logo_d / max(bb.size.X, bb.size.Y)

    # scale() acts on the raw geometry (ignoring any Location), so scale first
    # about the SVG origin, then translate the scaled centre into place.
    placed = [
        Pos(p.logo_cx - k * cx, -k * cy) * scale(f, by=k) for f in raw
    ]
    if p.logo_boost <= 0:
        return placed

    dilated: list[Face] = []
    steps = 12
    for f in placed:
        copies = f
        for i in range(steps):
            th = 2 * math.pi * i / steps
            copies += Pos(p.logo_boost * math.cos(th), p.logo_boost * math.sin(th)) * f
        dilated.extend(copies.faces())
    return dilated


def _design_sketch(p: StampParams) -> Sketch:
    """The face design in READING orientation (mirrored later)."""
    parts: list[Face] = _logo_faces(p)
    fs = _fitted_font_size(p)
    for txt, y in zip(p.lines, p.line_y):
        with BuildSketch(Plane.XY) as sk:
            Text(txt, font_size=fs, font=p.font, font_style=p.style,
                 align=(Align.CENTER, Align.CENTER))
        parts.extend((Pos(p.text_cx, y) * sk.sketch).faces())
    return Sketch() + parts


def build_stamp(p: StampParams):
    inner = RectangleRounded(p.piece_w, p.piece_h, p.corner_r)
    outer = RectangleRounded(
        p.piece_w + 2 * p.wall_t, p.piece_h + 2 * p.wall_t, p.corner_r + p.wall_t
    )
    tip = RectangleRounded(
        p.piece_w + 2 * p.blade_tip, p.piece_h + 2 * p.blade_tip,
        p.corner_r + p.blade_tip,
    )

    face_z = p.back_t
    plate = extrude(outer, p.back_t)
    blade = loft(
        [
            Plane.XY.offset(face_z) * outer,
            Plane.XY.offset(face_z + p.cut_depth) * tip,
        ]
    ) - extrude(Plane.XY.offset(face_z) * inner, p.cut_depth + 0.1)

    design = _design_sketch(p)

    # Sanity: design must clear the blade's inner wall
    bb = design.bounding_box()
    max_x = p.piece_w / 2 - p.clearance
    max_y = p.piece_h / 2 - p.clearance
    if abs(bb.min.X) > max_x or abs(bb.max.X) > max_x \
            or abs(bb.min.Y) > max_y or abs(bb.max.Y) > max_y:
        raise ValueError(
            f"Design bbox ({bb.min.X:.1f},{bb.min.Y:.1f})-({bb.max.X:.1f},"
            f"{bb.max.Y:.1f}) too close to the blade (limits ±{max_x:.1f}, ±{max_y:.1f})"
        )

    features = extrude(Plane.XY.offset(face_z) * design, p.emboss_h)
    stamp = plate + blade + features

    # Mirror so the impression reads correctly in the fondant
    return mirror(stamp, Plane.YZ)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="parkrun fondant cutter/embosser")
    ap.add_argument("--stl", type=Path, default=None, help="Export the stamp STL")
    ap.add_argument("--no-show", action="store_true", help="Skip the OCP viewer")
    args = ap.parse_args()

    p = StampParams()
    stamp = build_stamp(p)

    bb = stamp.bounding_box()
    print(f"Stamp {bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm, "
          f"volume {stamp.volume / 1000:.1f} cm^3, valid={stamp.is_valid}")
    print(f"  cuts a {p.piece_w:.0f} x {p.piece_h:.0f} mm rounded rectangle "
          f"(corner r{p.corner_r:.0f})")
    print(f"  impression depth {p.emboss_h} mm; roll fondant to ~{p.cut_depth} mm")
    print(f"  text size {_fitted_font_size(p):.1f} mm ({p.font} Bold)")

    if args.stl is not None:
        args.stl.parent.mkdir(parents=True, exist_ok=True)
        export_stl(stamp, args.stl)
        print(f"Wrote {args.stl}")

    if not args.no_show:
        try:
            from ocp_vscode import Camera, show
            show(stamp, names=["stamp"], reset_camera=Camera.RESET, grid=True)
        except Exception as exc:  # viewer not running / not installed
            print(f"(viewer skipped: {exc})")


if __name__ == "__main__":
    main()
