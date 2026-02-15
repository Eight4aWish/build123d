"""Gridfinity desktop organiser modules (build123d).

Parametric Gridfinity-compatible desktop organiser bins with press-fit
6 × 2 mm magnet holes.  Three modules are included:

  1. Post-it note holder  – 2 × 2 grid, 4U high, open front
  2. Pen / pencil cup     – 1 × 1 grid, 10U high
  3. Shallow tray          – 3 × 1 grid, 3U high, 2 dividers

All dimensions and grid sizes are easily adjustable via the dataclass
parameters at the top of each builder function.

Gridfinity spec reference (from gridfinity.xyz & kennetek/gridfinity-rebuilt):
  Grid pitch   42 mm         Base top       41.5 mm
  Height unit   7 mm         Corner radius   3.75 mm
  Magnet        6 mm Ø × 2 mm thick  (press-fit)
  Hole centre   8 mm from grid-pitch edge  (= 13 mm from unit centre)

Base profile (cross-section, inside-bottom → outside-top):
  z = 0.00  35.60 mm  r = 0.80   (bottom)
  z = 0.80  37.20 mm  r = 1.60   (45° chamfer)
  z = 2.60  37.20 mm  r = 1.60   (straight wall)
  z = 4.75  41.50 mm  r = 3.75   (45° chamfer to top)
  z = 7.00  41.50 mm  r = 3.75   (bridge / floor)

Run:
  ./.venv/bin/python gridfinity_desktop.py

Export:
  ./.venv/bin/python gridfinity_desktop.py --stl-postit postit.stl
  ./.venv/bin/python gridfinity_desktop.py --stl-pen pen.stl
  ./.venv/bin/python gridfinity_desktop.py --stl-tray tray.stl
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from build123d import (
    Align,
    Box,
    BuildPart,
    BuildSketch,
    Cylinder,
    Location,
    Locations,
    Mode,
    Part,
    Plane,
    Polygon,
    Rectangle,
    RectangleRounded,
    add,
    extrude,
    export_step,
    export_stl,
    loft,
)
from ocp_vscode import Camera, show


def _ocp_port(default: int = 3939) -> int:
    try:
        raw = os.environ.get("OCP_VSCODE_PORT") or os.environ.get("OCP_PORT") or str(default)
        return int(raw)
    except ValueError:
        return default


# ── Gridfinity specification constants ────────────────────────────────────

GRID      = 42.0    # grid pitch (mm)
BASE_W    = 41.5    # base top dimension per unit (mm)
BASE_R    = 3.75    # corner radius at base top (mm)
BASE_H    = 7.0     # one height-unit / total base height (mm)
PROF_H    = 4.75    # base profile height (mm)
BRIDGE_H  = BASE_H - PROF_H   # bridge connecting units (2.25 mm)
HOLE_FROM_GRID_SIDE = 8.0  # magnet-hole centre from grid-pitch edge (mm)
# → offset from unit centre = GRID/2 − 8 = 13.0 mm
# (spec: 4.8 mm from base *bottom* edge, which is 35.6 mm wide)


# ── Parameters ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GFParams:
    """Shared Gridfinity build parameters.

    Magnet hole diameter is set *slightly* under 6 mm for a press fit.
    Adjust ``mag_hole_d`` if your printer over/under-extrudes.
    """
    mag_hole_d: float     = 6.1    # press-fit hole Ø  (< 6 mm magnet)
    mag_hole_depth: float = 2.15   # pocket depth      (> 2 mm magnet)
    wall_t: float         = 2.6    # bin wall thickness
    floor_t: float        = 1.0    # floor thickness inside the bin


# ── Geometry helpers ──────────────────────────────────────────────────────

def _dim(n: int) -> float:
    """Outer dimension for *n* grid units (= n × 42 − 0.5)."""
    return GRID * n - (GRID - BASE_W)


def _centres(gx: int, gy: int) -> list[tuple[float, float, float]]:
    """Centre (cx, cy, 0) of every grid unit in a gx × gy layout."""
    return [
        ((ix - (gx - 1) / 2) * GRID,
         (iy - (gy - 1) / 2) * GRID,
         0.0)
        for ix in range(gx)
        for iy in range(gy)
    ]


# ── Single base-profile bump (1 × 1 unit) ────────────────────────────────

def _base_bump() -> Part:
    """Ruled-loft base profile for one grid unit (z = 0 → PROF_H).

    Four cross-sections recreate the standard stepped profile:
      z = 0.00  35.60 × 35.60 mm,  r = 0.80  (bottom chamfer start)
      z = 0.80  37.20 × 37.20 mm,  r = 1.60  (45° transition done)
      z = 2.60  37.20 × 37.20 mm,  r = 1.60  (straight wall)
      z = 4.75  41.50 × 41.50 mm,  r = 3.75  (top of profile)
    """
    sections = [
        (0.00, BASE_W - 2 * 2.95, max(BASE_R - 2.95, 0.10)),  # 35.60, 0.80
        (0.80, BASE_W - 2 * 2.15, max(BASE_R - 2.15, 0.10)),  # 37.20, 1.60
        (2.60, BASE_W - 2 * 2.15, max(BASE_R - 2.15, 0.10)),  # 37.20, 1.60
        (PROF_H, BASE_W,          BASE_R),                      # 41.50, 3.75
    ]
    with BuildPart() as bp:
        for z, w, r in sections:
            with BuildSketch(Plane.XY.offset(z)):
                RectangleRounded(w, w, r)
        loft(ruled=True)
    return bp.part


# ── Multi-unit base with magnet holes ─────────────────────────────────────

def make_base(gx: int, gy: int, p: GFParams = GFParams()) -> Part:
    """Gridfinity base grid (gx × gy) with press-fit magnet pockets.

    Each grid unit gets the standard stepped base profile and four
    6 mm magnet pockets in the corners (cut from below).
    """
    tw, td = _dim(gx), _dim(gy)
    bump = _base_bump()

    with BuildPart() as bp:
        # Flat bridge deck tying all units together (z = PROF_H → BASE_H)
        with BuildSketch(Plane.XY.offset(PROF_H)):
            RectangleRounded(tw, td, BASE_R)
        extrude(amount=BRIDGE_H)

        # Per-unit profiled base bumps
        with Locations(_centres(gx, gy)):
            add(bump)

        # Magnet pockets – four per unit, cut from the bottom face
        if p.mag_hole_d > 0:
            ho = GRID / 2 - HOLE_FROM_GRID_SIDE   # 13.0 mm from unit centre
            mag_pts: list[tuple[float, float, float]] = []
            for cx, cy, _ in _centres(gx, gy):
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        mag_pts.append((cx + sx * ho, cy + sy * ho, 0.0))
            with Locations(mag_pts):
                Cylinder(
                    radius=p.mag_hole_d / 2,
                    height=p.mag_hole_depth,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT,
                )

    return bp.part


# ── Generic hollow bin ────────────────────────────────────────────────────

def make_bin(
    gx: int, gy: int, height_u: int, p: GFParams = GFParams(),
) -> Part:
    """Gridfinity bin: base + walls, hollowed from inside.

    *height_u* is the total height in Gridfinity units (7 mm each),
    including the 1-unit base.  E.g. height_u = 4  → 28 mm overall.
    """
    tw, td = _dim(gx), _dim(gy)
    body_h = height_u * BASE_H           # total body height (mm)
    wall_h = body_h - BASE_H             # wall height above base

    with BuildPart() as bp:
        add(make_base(gx, gy, p))

        # Outer walls (z = BASE_H → body_h)
        with BuildSketch(Plane.XY.offset(BASE_H)):
            RectangleRounded(tw, td, BASE_R)
        extrude(amount=wall_h)

        # Hollow interior (keep floor)
        iw  = tw - 2 * p.wall_t
        id_ = td - 2 * p.wall_t
        ir  = max(BASE_R - p.wall_t, 0.5)
        fz  = BASE_H + p.floor_t       # top of floor

        with BuildSketch(Plane.XY.offset(fz)):
            RectangleRounded(iw, id_, ir)
        extrude(amount=body_h - fz, mode=Mode.SUBTRACT)

    return bp.part


# ── 1. Post-it note holder ────────────────────────────────────────────────

def make_postit_holder(
    gx: int = 2,
    gy: int = 2,
    height_u: int = 6,
    open_front: bool = True,
    front_lip_h: float = 5.0,
    p: GFParams = GFParams(),
) -> Part:
    """Post-it note holder.

    Standard 76 × 76 mm (3″) post-it notes fit comfortably inside
    a 2 × 2 grid bin (≈ 81 mm interior).  6U high (42 mm) holds a
    full pad of notes.

    When *open_front* is True the −Y wall is removed above a small
    retaining lip of *front_lip_h* mm, so notes can be grabbed easily.
    """
    part = make_bin(gx, gy, height_u, p)
    if not open_front:
        return part

    tw, td = _dim(gx), _dim(gy)
    iw     = tw - 2 * p.wall_t
    body_h = height_u * BASE_H
    cut_z  = BASE_H + p.floor_t + front_lip_h
    cut_h  = body_h - cut_z + 1.0       # generously through top

    with BuildPart() as bp:
        add(part)
        # Remove front wall above retaining lip
        with Locations([(0, -td / 2, cut_z)]):
            Box(
                iw + 0.2,
                p.wall_t * 4,
                cut_h,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
                mode=Mode.SUBTRACT,
            )
    return bp.part


# ── 2. Pen / pencil holder ────────────────────────────────────────────────

def make_pen_holder(
    gx: int = 1,
    gy: int = 2,
    height_u: int = 10,
    p: GFParams = GFParams(),
) -> Part:
    """Tall cup for pens, pencils, and markers.

    Default 1 × 2 grid, 10 U high  →  42 × 84 × 70 mm.
    A divider separates the two 1 × 1 cells so shorter
    items (e.g. pencils) don't get lost behind tall pens.
    """
    part = make_bin(gx, gy, height_u, p)

    # Add divider wall between the two 1×1 cells
    if gy >= 2:
        tw, td = _dim(gx), _dim(gy)
        iw  = tw - 2 * p.wall_t
        id_ = td - 2 * p.wall_t
        fz  = BASE_H + p.floor_t
        dh  = height_u * BASE_H - fz

        with BuildPart() as bp:
            add(part)
            with Locations([(0, 0, fz)]):
                Box(
                    iw,
                    p.wall_t,
                    dh,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                )
        return bp.part

    return part


# ── 3. Shallow tray with dividers ────────────────────────────────────────

def make_tray(
    gx: int = 2,
    gy: int = 2,
    height_u: int = 6,
    dividers_x: int = 1,
    dividers_y: int = 1,
    p: GFParams = GFParams(),
) -> Part:
    """Shallow tray for paper-clips, erasers, SD-cards, etc.

    Default 2 × 2 grid with cross dividers creating 4 equal cells.
    *dividers_x* / *dividers_y* set the number of dividing walls
    along each axis.
    """
    part = make_bin(gx, gy, height_u, p)
    if dividers_x < 1 and dividers_y < 1:
        return part

    tw, td = _dim(gx), _dim(gy)
    iw  = tw - 2 * p.wall_t
    id_ = td - 2 * p.wall_t
    fz  = BASE_H + p.floor_t
    dh  = height_u * BASE_H - fz     # divider height

    with BuildPart() as bp:
        add(part)
        # X-axis dividers (walls running along Y)
        if dividers_x >= 1:
            spacing_x = iw / (dividers_x + 1)
            for i in range(1, dividers_x + 1):
                x = -iw / 2 + i * spacing_x
                with Locations([(x, 0, fz)]):
                    Box(
                        p.wall_t,
                        id_,
                        dh,
                        align=(Align.CENTER, Align.CENTER, Align.MIN),
                    )
        # Y-axis dividers (walls running along X)
        if dividers_y >= 1:
            spacing_y = id_ / (dividers_y + 1)
            for j in range(1, dividers_y + 1):
                y = -id_ / 2 + j * spacing_y
                with Locations([(0, y, fz)]):
                    Box(
                        iw,
                        p.wall_t,
                        dh,
                        align=(Align.CENTER, Align.CENTER, Align.MIN),
                    )
    return bp.part


# ── CLI / viewer ──────────────────────────────────────────────────────────

def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--stl-postit",  type=str, default=None, help="Export post-it holder STL")
    ap.add_argument("--stl-pen",     type=str, default=None, help="Export pen holder STL")
    ap.add_argument("--stl-tray",    type=str, default=None, help="Export tray STL")
    ap.add_argument("--step-postit", type=str, default=None, help="Export post-it holder STEP")
    ap.add_argument("--step-pen",    type=str, default=None, help="Export pen holder STEP")
    ap.add_argument("--step-tray",   type=str, default=None, help="Export tray STEP")
    return ap.parse_args()


def main() -> None:
    args = _cli()
    p = GFParams()

    # Build all three modules
    print("Building post-it holder (2×2, 6U) …")
    postit = make_postit_holder(p=p)
    print("Building pen holder (1×2, 10U) …")
    pen = make_pen_holder(p=p)
    print("Building tray (2×2, 3U, 4 cells) …")
    tray = make_tray(p=p)
    print("Done.\n")

    # Exports
    for part, stl, step, name in [
        (postit, args.stl_postit,  args.step_postit, "post-it holder"),
        (pen,    args.stl_pen,     args.step_pen,    "pen holder"),
        (tray,   args.stl_tray,    args.step_tray,   "tray"),
    ]:
        if stl:
            export_stl(part, stl)
            print(f"  ✓ STL  {name} → {stl}")
        if step:
            export_step(part, step)
            print(f"  ✓ STEP {name} → {step}")

    # Display in OCP viewer – space the three modules apart
    x_spacing = _dim(3) + 20          # enough room between parts

    try:
        show(
            postit,
            pen.moved(Location((x_spacing, 0, 0))),
            tray.moved(Location((x_spacing * 2, 0, 0))),
            names=["Post-it Holder (2×2, 6U)",
                   "Pen Cup (1×2, 10U)",
                   "Tray (2×2, 3U)"],
            colors=[(0.25, 0.55, 0.85),
                    (0.85, 0.45, 0.25),
                    (0.35, 0.75, 0.45)],
            reset_camera=Camera.RESET,
            grid=True,
            port=_ocp_port(),
        )
    except RuntimeError as ex:
        print("\nOCP viewer is not reachable.")
        print("- Open 'OCP CAD Viewer' and ensure the backend is running.")
        print("- Or start: ./.venv/bin/python -m ocp_vscode --port 3939")
        print(f"\nDetails: {ex}")


if __name__ == "__main__":
    main()
