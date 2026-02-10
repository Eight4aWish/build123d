"""Desk headphone holder (build123d) generated from `drawing.svg`.

This model is built by *copying the exact 2D profile* drawn in `drawing.svg`.
The SVG's `<rect>` elements (in mm) are imported, unioned, then extruded into a
3D solid.

Run:
  ./.venv/bin/python headphone_holder_build123d.py

Export:
  ./.venv/bin/python headphone_holder_build123d.py --stl headphone_holder.stl

Notes:
- Dimensions are millimetres.
- Coordinate system:
    - SVG X maps to CAD X.
    - SVG Y maps to CAD Z (down in SVG becomes negative Z).
    - The profile is extruded along CAD Y.
"""

from __future__ import annotations

import os

from dataclasses import dataclass, replace
from pathlib import Path
import xml.etree.ElementTree as ET

from build123d import (
    Align,
    Box,
    BuildPart,
    Location,
    Locations,
    Mode,
    export_step,
    export_stl,
    fillet,
)
from ocp_vscode import Camera, show


def _ocp_port(default: int = 3939) -> int:
    try:
        raw = os.environ.get("OCP_VSCODE_PORT") or os.environ.get("OCP_PORT") or str(default)
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class HolderParams:
    # Source SVG profile
    profile_svg: Path = Path("drawing.svg")

    # Extrusion width (Y direction)
    width: float = 40.0

    # Widen the bar capture "U" by stretching the profile in X.
    # This keeps the outer wall fixed and shifts the inner/right-hand geometry.
    u_widen: float = 2.0

    # SVG coordinate snapping (mm). Inkscape often writes values like 109.20416
    # when the intent was 109.2 or 109.0, which can create tiny steps.
    snap: float = 1.0

    # Edge rounding (mm). Set to 0 to keep sharp edges.
    fillet_radius: float = 1.0

    color: tuple[float, float, float] = (0.15, 0.15, 0.15)

def _resolve_profile_path(profile_svg: Path) -> Path:
    if profile_svg.is_absolute():
        return profile_svg

    # Allow running from any working directory.
    candidate = Path(__file__).resolve().with_name(str(profile_svg))
    if candidate.exists():
        return candidate
    return profile_svg


def _svg_rects_mm(svg_path: Path) -> list[tuple[float, float, float, float]]:
    """Return list of (x, y, width, height) from SVG <rect> elements.

    This SVG was authored with Inkscape using mm units.
    """

    tree = ET.parse(svg_path)
    root = tree.getroot()

    def snap(value: float, step: float) -> float:
        if step <= 0:
            return value
        return round(value / step) * step

    rects: list[tuple[float, float, float, float]] = []
    for el in root.iter():
        if not el.tag.endswith("rect"):
            continue
        try:
            x = float(el.attrib.get("x", "0"))
            y = float(el.attrib.get("y", "0"))
            w = float(el.attrib["width"])
            h = float(el.attrib["height"])
        except (KeyError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        rects.append((x, y, w, h))

    if not rects:
        raise ValueError(f"No <rect> elements found in {svg_path}")
    return rects


def _detect_u_seam_x(rects: list[tuple[float, float, float, float]]) -> float:
    """Pick an X position to "stretch" the profile for the bar U opening.

    Heuristic: find vertical-wall-like rectangles (width ~6mm, tall-ish), then
    pick the middle X (outer wall / inner wall / far wall => inner wall).
    """

    candidates = sorted({x for x, _, w, h in rects if 5.0 <= w <= 7.0 and h >= 12.0})
    if len(candidates) >= 3:
        return candidates[1]
    if len(candidates) == 2:
        return candidates[1]

    # Fallback: median of all x positions.
    xs = sorted(x for x, _, _, _ in rects)
    return xs[len(xs) // 2]


def _stretch_rects_x(
    rects: list[tuple[float, float, float, float]],
    seam_x: float,
    widen: float,
) -> list[tuple[float, float, float, float]]:
    """Stretch rectangles in X at seam_x by widen.

    - Rectangles entirely left of seam are unchanged.
    - Rectangles entirely right of seam are shifted +widen.
    - Rectangles that cross the seam get wider by +widen.
    """

    if abs(widen) < 1e-9:
        return rects

    out: list[tuple[float, float, float, float]] = []
    for x, y, w, h in rects:
        x2 = x + w
        if x2 <= seam_x:
            out.append((x, y, w, h))
        elif x >= seam_x:
            out.append((x + widen, y, w, h))
        else:
            out.append((x, y, w + widen, h))
    return out


def build_holder(params: HolderParams):
    svg_path = _resolve_profile_path(params.profile_svg)
    rects_raw = _svg_rects_mm(svg_path)

    def s(value: float) -> float:
        if params.snap <= 0:
            return value
        return round(value / params.snap) * params.snap

    rects = [(s(x), s(y), s(w), s(h)) for x, y, w, h in rects_raw]

    # Stretch the profile to widen the bar "U" while keeping the outer wall fixed.
    seam_x = _detect_u_seam_x(rects)
    rects = _stretch_rects_x(rects, seam_x=seam_x, widen=params.u_widen)

    min_x = min(x for x, _, _, _ in rects)
    min_y = min(y for _, y, _, _ in rects)

    with BuildPart() as p:
        # Union all SVG rectangles as boxes extruded along Y.
        # Mapping:
        # - CAD X = SVG X (shifted so leftmost is X=0)
        # - CAD Z = -(SVG Y - min_y), so topmost is Z=0
        if params.width <= 0:
            raise ValueError("width must be > 0")

        for x, y, w, h in rects:
            x0 = x - min_x
            z0 = -((y - min_y) + h)
            with Locations(Location((x0, 0, z0))):
                Box(w, params.width, h, align=(Align.MIN, Align.CENTER, Align.MIN), mode=Mode.ADD)

    part = p.part
    if params.fillet_radius > 0:
        part = fillet(part.edges(), params.fillet_radius)
    return part


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Desk headphone holder generated from drawing.svg (build123d)")
    parser.add_argument("--stl", type=Path, default=None, help="Export STL to this path")
    parser.add_argument("--step", type=Path, default=None, help="Export STEP to this path")

    parser.add_argument("--svg", type=Path, default=None, help="Path to SVG profile (defaults to drawing.svg)")
    parser.add_argument("--width", type=float, default=None, help="Extrusion width in mm (Y direction)")
    parser.add_argument(
        "--snap",
        type=float,
        default=None,
        help="Snap SVG coordinates to this grid in mm (e.g. 0.01). Use 0 to disable.",
    )
    parser.add_argument(
        "--u-widen",
        type=float,
        default=None,
        help="Widen the underside bar U opening by this many mm (default: 2). Use 0 to disable.",
    )
    parser.add_argument(
        "--fillet",
        type=float,
        default=None,
        help="Edge fillet radius in mm (default: 1). Use 0 to keep sharp edges.",
    )

    args = parser.parse_args()

    params = HolderParams()
    if args.svg is not None:
        params = replace(params, profile_svg=args.svg)
    if args.width is not None:
        params = replace(params, width=float(args.width))
    if args.snap is not None:
        params = replace(params, snap=float(args.snap))
    if args.u_widen is not None:
        params = replace(params, u_widen=float(args.u_widen))
    if args.fillet is not None:
        params = replace(params, fillet_radius=float(args.fillet))

    holder = build_holder(params)

    if args.stl is not None:
        export_stl(holder, args.stl)
    if args.step is not None:
        export_step(holder, args.step)

    try:
        show(
            holder,
            names=["headphone_holder"],
            colors=[params.color],
            reset_camera=Camera.RESET,
            grid=True,
            port=_ocp_port(),
        )
    except RuntimeError as ex:
        print("\nOCP viewer is not reachable.")
        print("- If you're using the VS Code extension: open 'OCP CAD Viewer' and ensure the backend is running.")
        print("- Or start the standalone viewer with: ./.venv/bin/python -m ocp_vscode --port 3939")
        print(f"\nDetails: {ex}")


if __name__ == "__main__":
    main()
