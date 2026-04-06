"""Tiptop Audio Eurorack Z-Rail adapter bracket.

Dimensions: 23.8 x 132 x 2.5 mm (width x length x thickness)
Four M4 flathead countersunk mounting holes.

  x=5 and x=127  — outer holes, countersunk from the TOP face
  x=40 and x=92  — inner holes, countersunk from the BOTTOM face

Hole diameter is 4.7 mm (spec is 4.5 mm; +0.2 mm compensates for FDM
filament expansion that typically prints holes ~0.2 mm undersize).

Run:
    ./.venv/bin/python zrail_bracket.py

Export:
    ./.venv/bin/python zrail_bracket.py --stl zrail_bracket.stl
    ./.venv/bin/python zrail_bracket.py --step zrail_bracket.step
"""

from __future__ import annotations

import os
from pathlib import Path

from build123d import (
    Align,
    Box,
    BuildPart,
    CounterSinkHole,
    Location,
    Locations,
    Mode,
    export_stl,
    export_step,
)
from ocp_vscode import Camera, show


def _ocp_port(default: int = 3939) -> int:
    try:
        raw = os.environ.get("OCP_VSCODE_PORT") or os.environ.get("OCP_PORT") or str(default)
        return int(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
LENGTH    = 132.0  # mm — X axis (long edge)
WIDTH     =  23.8  # mm — Y axis
THICKNESS =   2.5  # mm — Z axis

# Hole specs
# Nominal per spec: 4.5 mm. Using 4.7 mm to account for FDM under-size.
# Countersink: 90° included angle (standard M4 ISO flathead), 8.2 mm outer Ø.
HOLE_D     =  4.7  # mm — clearance hole diameter
CS_OUTER_D =  8.2  # mm — countersink outer diameter at the face
CS_ANGLE   = 90    # degrees — included angle of the countersink cone

# (x along length, y across width) — origin at part corner
HOLES_TOP = [(  5.0, 14.3), (127.0, 14.3)]  # countersunk from top  (+Z) face
HOLES_BOT = [( 40.0, 14.3), ( 92.0, 14.3)]  # countersunk from bottom (-Z) face


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build() -> object:
    with BuildPart() as p:
        Box(LENGTH, WIDTH, THICKNESS, align=(Align.MIN, Align.MIN, Align.MIN))

        # Countersinks on the TOP face (z = THICKNESS).
        # Default hole orientation drills in -Z, so place at z = THICKNESS
        # and the countersink opens at the top surface.
        with Locations(*[(x, y, THICKNESS) for x, y in HOLES_TOP]):
            CounterSinkHole(
                radius=HOLE_D / 2,
                counter_sink_radius=CS_OUTER_D / 2,
                depth=THICKNESS,
                counter_sink_angle=CS_ANGLE,
                mode=Mode.SUBTRACT,
            )

        # Countersinks on the BOTTOM face (z = 0).
        # Rotate 180° around the X axis so the hole drills upward (+Z),
        # with the countersink opening at the bottom surface.
        with Locations(*[Location((x, y, 0), (180, 0, 0)) for x, y in HOLES_BOT]):
            CounterSinkHole(
                radius=HOLE_D / 2,
                counter_sink_radius=CS_OUTER_D / 2,
                depth=THICKNESS,
                counter_sink_angle=CS_ANGLE,
                mode=Mode.SUBTRACT,
            )

    return p.part


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Tiptop Audio Z-Rail adapter bracket")
    parser.add_argument("--stl",  type=Path, default=None, help="Export STL to this path")
    parser.add_argument("--step", type=Path, default=None, help="Export STEP to this path")
    args = parser.parse_args()

    part = build()

    if args.stl is not None:
        export_stl(part, args.stl)
        print(f"STL written to {args.stl}")
    if args.step is not None:
        export_step(part, args.step)
        print(f"STEP written to {args.step}")

    try:
        show(part, reset_camera=Camera.RESET, grid=True, port=_ocp_port())
    except RuntimeError as ex:
        print("\nOCP viewer is not reachable.")
        print("- Open 'OCP CAD Viewer' in VS Code and ensure the backend is running.")
        print(f"\nDetails: {ex}")


if __name__ == "__main__":
    main()
