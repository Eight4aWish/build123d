"""Switch cap generator for tapered switch shafts (build123d).

Cylindrical press-fit switch caps with internal gripping fins.
Multiple variants with different fin grip diameters for test fitting.

Dimensions:
  Body outer Ø     6.0 mm
  Lip outer Ø      7.5 mm   (1.5 mm tall skirt at open end)
  Total height     12.0 mm
  Bore Ø           4.6 mm   (wall thickness ≈ 0.7 mm)
  Ceiling          1.5 mm   (solid dome end)
  4 internal fins gripping a tapered shaft (3.5 mm base → 3.2 mm tip)

Designed for 0.2 mm nozzle FDM (PLA).
Print dome-down (closed end on build plate) for best surface finish.

Run:
  ./.venv/bin/python switch_cap.py

Export:
  ./.venv/bin/python switch_cap.py --no-show
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

from build123d import (
    Align,
    Axis,
    BuildPart,
    BuildSketch,
    Circle,
    Cylinder,
    Location,
    Locations,
    Mode,
    Part,
    Plane,
    Rectangle,
    extrude,
    export_stl,
    fillet,
)
from ocp_vscode import Camera, show


def _ocp_port(default: int = 3939) -> int:
    try:
        return int(os.environ.get("OCP_VSCODE_PORT", str(default)))
    except ValueError:
        return default


# ── Parameters ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CapParams:
    """Switch cap dimensions."""
    body_d: float       = 6.0     # main body outer Ø (mm)
    lip_d: float        = 7.5     # lip outer Ø (mm)
    total_h: float      = 12.0    # total cap height (mm)
    lip_h: float        = 1.5     # lip height from open end (mm)
    hole_d: float       = 4.6     # bore Ø (mm)
    hole_ceiling: float = 1.5     # solid ceiling at dome end (mm)
    dome_fillet: float  = 1.0     # fillet radius for rounded dome (mm)
    fin_width: float    = 0.4     # tangential width of each fin (mm)
    n_fins: int         = 4       # number of internal gripping fins
    # Switch shaft (reference only)
    shaft_d_base: float = 3.5     # shaft Ø at base (mm)
    shaft_d_tip: float  = 3.2     # shaft Ø at tip (mm)


# Variants: label → fin inner diameter (mm)
# The fin tips define the grip Ø that the switch shaft must squeeze past.
# FDM printing adds ~0.1 mm to internal features, so model slightly larger.
VARIANTS = {
    "A_tight":  3.20,   # matches shaft tip — very firm grip
    "B_medium": 3.35,   # halfway between tip and base
    "C_loose":  3.50,   # matches shaft base — gentle grip
}


# ── Cap builder ───────────────────────────────────────────────────────────

def make_switch_cap(
    fin_inner_d: float,
    p: CapParams = CapParams(),
) -> Part:
    """Build one switch cap with the given fin inner diameter.

    The bore is a 4.6 mm cylinder with four thin fins protruding inward.
    *fin_inner_d* sets the diameter at the fin tips — the shaft must flex
    the fins outward to pass through.
    """
    bore_r    = p.hole_d / 2
    fin_tip_r = fin_inner_d / 2
    fin_depth = bore_r - fin_tip_r          # radial protrusion per fin
    r_mid     = (bore_r + fin_tip_r) / 2    # centre of fin in radial dir
    hole_h    = p.total_h - p.hole_ceiling  # bore depth from open end

    with BuildPart() as cap:
        # ── outer shell ──────────────────────────────────────────────
        # Lip (wider skirt at open end)
        Cylinder(
            radius=p.lip_d / 2,
            height=p.lip_h,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        # Main body
        Cylinder(
            radius=p.body_d / 2,
            height=p.total_h,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

        # ── dome (rounded closed end) ───────────────────────────────
        top_edges = cap.edges().group_by(Axis.Z)[-1]
        fillet(top_edges, radius=p.dome_fillet)

        # ── bore with fin notches ────────────────────────────────────
        # Start with the bore circle, then subtract thin rectangles
        # where each fin should remain as solid material.
        with BuildSketch(Plane.XY):
            Circle(radius=bore_r)
            for i in range(p.n_fins):
                a_deg = i * (360.0 / p.n_fins)
                a_rad = math.radians(a_deg)
                cx = r_mid * math.cos(a_rad)
                cy = r_mid * math.sin(a_rad)
                with Locations([(cx, cy)]):
                    Rectangle(
                        fin_depth, p.fin_width,
                        rotation=a_deg,
                        mode=Mode.SUBTRACT,
                    )
        extrude(amount=hole_h, mode=Mode.SUBTRACT)

    return cap.part


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Switch cap generator")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip OCP viewer, just export STLs")
    args = parser.parse_args()

    p = CapParams()
    caps: dict[str, Part] = {}

    for label, fin_d in VARIANTS.items():
        desc = f"fin Ø {fin_d:.2f} mm, grip clearance {p.hole_d - fin_d:.2f} mm/side"
        print(f"Building {label} ({desc}) …")
        caps[label] = make_switch_cap(fin_d, p)

    # Export individual STLs
    for label, part in caps.items():
        path = f"switch_cap_{label}.stl"
        export_stl(part, path)
        print(f"  ✓ STL → {path}")

    # Show all variants side by side in viewer
    if not args.no_show:
        spacing = 12.0
        parts_shifted = []
        for i, (label, part) in enumerate(caps.items()):
            shifted = part.move(Location((i * spacing, 0, 0)))
            parts_shifted.append(shifted)
        show(
            *parts_shifted,
            names=list(caps.keys()),
            port=_ocp_port(),
            reset_camera=Camera.KEEP,
        )

    print("Done.")


if __name__ == "__main__":
    main()
