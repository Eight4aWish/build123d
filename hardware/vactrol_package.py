"""Two-piece vactrol package for a 3 mm LED + GL55-series LDR.

Architecture
============
Clamshell along the wire plane (horizontal split at Y = outer_y/2).
  - Tray (bottom): half-cylinder cradles for LDR + LED, alignment pins,
                   wire grooves in each Z-end.
  - Cap  (top):    mirror cavity, alignment sockets, matching wire grooves.

Axial registration
------------------
- LDR back face is recessed into the LDR-side end wall by ldr_recess_depth
  (default 0.5 mm), shortening the cavity and bringing the components
  closer together while preserving a ≥0.9 mm light-tight back wall.
- The cavity is wide (Ø(LDR cavity)) through BOTH the LDR disk AND the LED
  dome zone — there is no LDR-front shoulder — so the full LDR sensor face
  is exposed to the dome's light cone.
- The LDR is held axially by its back face against the recess floor and
  its front face against the LED dome tip.
- The LED is held axially by its lip catching on the body→lip cavity step,
  and radially by the body cylinder fitting snugly in the body cavity.

Light-tightness
---------------
Walls are 1.4 mm thick (block visible LED light through PLA/PETG at
typical 0.2 mm layers). The +X and -X long-edge seams have a lap step
so light can't traverse the parting plane along the long sides.
The Z-end seams stay butt-jointed; the wire-OD-tight grooves and the
1.4 mm end walls handle leakage there.

Coordinate convention
---------------------
  X  width (across)            origin at -X face
  Y  vertical (split axis)     origin at -Y face (tray bottom)
  Z  component axis            origin at -Z face (LDR wires exit)
Parting plane: Y = outer_y / 2.

Run:
    ./.venv/bin/python vactrol_package.py
Export:
    ./.venv/bin/python vactrol_package.py \\
        --stl-tray vactrol_tray.stl --stl-cap vactrol_cap.stl
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from build123d import (
    Align,
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    Compound,
    Location,
    Locations,
    Mode,
    Plane,
    extrude,
    export_step,
    export_stl,
)
from ocp_vscode import Camera, show


def _ocp_port(default: int = 3939) -> int:
    try:
        raw = os.environ.get("OCP_VSCODE_PORT") or os.environ.get("OCP_PORT") or str(default)
        return int(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VactrolParams:
    # --- 3 mm LED (cylinder + dome, then lip) -----------------------------
    led_lamp_d: float = 3.01            # lamp cylinder OD (dome is same OD)
    led_lamp_l: float = 4.4             # lip top → dome tip
    led_lip_d: float = 3.88             # base lip OD
    led_lip_h: float = 1.22             # lip thickness
    led_wire_d: float = 0.72
    led_wire_pitch: float = 2.54

    # --- GL55-series LDR (disk) -------------------------------------------
    ldr_d: float = 5.05
    ldr_h: float = 2.18
    ldr_wire_d: float = 0.63
    ldr_wire_pitch: float = 3.6

    # --- Clearances (radial = per-side; axial = per-component) ------------
    clear_lamp_radial: float = 0.10
    clear_lip_radial: float = 0.15
    clear_ldr_radial: float = 0.15
    clear_led_axial: float = 0.05       # LED lip-cavity slack to end wall
    clear_wire: float = 0.10            # over wire OD

    # --- Wall thicknesses (for light-tightness) ---------------------------
    wall_t: float = 1.4                 # cavity wall (X and Y)
    end_wall_t: float = 1.4             # Z-end wall

    # --- LDR recess: LDR back face sits this far into the back wall, ------
    # bringing the dome contact point closer to the LDR-end face. Effective
    # LDR-side wall thickness = end_wall_t - ldr_recess_depth (keep ≥0.7 mm
    # for light-tightness).
    ldr_recess_depth: float = 0.5

    # --- Light-tight lap on +X / -X long edges ----------------------------
    lap_inset: float = 0.5              # strip width inboard from outer face
    lap_h: float = 0.5                  # vertical overlap height
    lap_clearance: float = 0.10         # mating clearance in the lap

    # --- Alignment pins / sockets -----------------------------------------
    pin_d: float = 1.6
    pin_h: float = 1.8                  # protrusion above parting plane
    socket_d_extra: float = 0.10        # socket Ø = pin Ø + this
    socket_h_extra: float = 0.30        # socket extra depth beyond pin top
    pin_x_inset: float = 1.0            # pin centre X from outer face
    pin_z_inset: float = 0.7            # pin centre Z from outer face (in end wall)

    # --- Display ----------------------------------------------------------
    tray_color: tuple[float, float, float] = (0.85, 0.60, 0.30)
    cap_color: tuple[float, float, float] = (0.30, 0.55, 0.85)


# ---------------------------------------------------------------------------
# Derived dimensions
# ---------------------------------------------------------------------------
def _dimensions(p: VactrolParams) -> dict:
    cavity_d = p.ldr_d + 2 * p.clear_ldr_radial         # 5.35 (wide cavity Ø)
    outer_x = cavity_d + 2 * p.wall_t                    # 8.15
    outer_y = cavity_d + 2 * p.wall_t

    # LED lamp = hemispherical dome + cylindrical body
    dome_h = p.led_lamp_d / 2                            # 1.505
    body_l = p.led_lamp_l - dome_h                       # 2.895

    # Z stack (LDR end at Z=0)
    z_ldr_back  = p.end_wall_t - p.ldr_recess_depth                   # 0.90
    z_ldr_face  = z_ldr_back + p.ldr_h                                # 3.08 (LDR sensor face)
    z_dome_base = z_ldr_face + dome_h                                  # 4.585 (wide→body cavity transition)
    z_body_back = z_dome_base + body_l                                 # 7.48  (body→lip cavity transition; lip catches here)
    z_lip_back  = z_body_back + p.led_lip_h + p.clear_led_axial        # 8.75
    outer_z     = z_lip_back + p.end_wall_t                            # 10.15

    return {
        "cavity_d": cavity_d,
        "outer_x": outer_x,
        "outer_y": outer_y,
        "outer_z": outer_z,
        "parting_y": outer_y / 2,
        "z_ldr_back": z_ldr_back,
        "z_ldr_face": z_ldr_face,
        "z_dome_base": z_dome_base,
        "z_body_back": z_body_back,
        "z_lip_back": z_lip_back,
    }


# ---------------------------------------------------------------------------
# Cavity + groove subtractions (identical for tray and cap; each half is
# clipped to its own Y range by the surrounding box)
# ---------------------------------------------------------------------------
def _cut_cavities_and_grooves(p: VactrolParams, d: dict) -> None:
    cx = d["outer_x"] / 2
    cy = d["parting_y"]

    r_wide = (p.ldr_d + 2 * p.clear_ldr_radial) / 2     # LDR + dome chamber
    r_body = (p.led_lamp_d + 2 * p.clear_lamp_radial) / 2
    r_lip  = (p.led_lip_d + 2 * p.clear_lip_radial) / 2

    # Wide cavity: LDR disk AND LED dome chamber — no LDR-front shoulder,
    # so the full LDR sensor face is exposed to the dome.
    with BuildSketch(Plane.XY.offset(d["z_ldr_back"])) as sk:
        with Locations((cx, cy)):
            Circle(r_wide)
    extrude(to_extrude=sk.sketch, amount=d["z_dome_base"] - d["z_ldr_back"], mode=Mode.SUBTRACT)

    # Body cavity: snug around the LED's cylindrical body section for radial
    # alignment. The lip catches at the body→lip step (next cavity).
    with BuildSketch(Plane.XY.offset(d["z_dome_base"])) as sk:
        with Locations((cx, cy)):
            Circle(r_body)
    extrude(to_extrude=sk.sketch, amount=d["z_body_back"] - d["z_dome_base"], mode=Mode.SUBTRACT)

    # Lip cavity
    with BuildSketch(Plane.XY.offset(d["z_body_back"])) as sk:
        with Locations((cx, cy)):
            Circle(r_lip)
    extrude(to_extrude=sk.sketch, amount=d["z_lip_back"] - d["z_body_back"], mode=Mode.SUBTRACT)

    # Wire grooves through the LDR-side end wall
    r_ldr_w = (p.ldr_wire_d + p.clear_wire) / 2
    with BuildSketch(Plane.XY.offset(-0.1)) as sk:
        for xs in (-p.ldr_wire_pitch / 2, +p.ldr_wire_pitch / 2):
            with Locations((cx + xs, cy)):
                Circle(r_ldr_w)
    extrude(to_extrude=sk.sketch, amount=d["z_ldr_back"] + 0.2, mode=Mode.SUBTRACT)

    # Wire grooves through the LED-side end wall
    r_led_w = (p.led_wire_d + p.clear_wire) / 2
    with BuildSketch(Plane.XY.offset(d["z_lip_back"] - 0.1)) as sk:
        for xs in (-p.led_wire_pitch / 2, +p.led_wire_pitch / 2):
            with Locations((cx + xs, cy)):
                Circle(r_led_w)
    extrude(to_extrude=sk.sketch, amount=p.end_wall_t + 0.2, mode=Mode.SUBTRACT)


def _pin_positions(p: VactrolParams, d: dict) -> list[tuple[float, float]]:
    """Two diagonally opposite pins, each embedded in an end-wall corner."""
    return [
        (p.pin_x_inset, p.pin_z_inset),
        (d["outer_x"] - p.pin_x_inset, d["outer_z"] - p.pin_z_inset),
    ]


# ---------------------------------------------------------------------------
# Tray (bottom half)
# ---------------------------------------------------------------------------
def build_tray(p: VactrolParams):
    d = _dimensions(p)
    outer_x = d["outer_x"]
    parting_y = d["parting_y"]
    outer_z = d["outer_z"]

    with BuildPart() as part:
        # Solid lower half
        with Locations(Location((outer_x / 2, parting_y / 2, outer_z / 2))):
            Box(outer_x, parting_y, outer_z)

        # Cavities and wire grooves (clipped to the lower half by the box itself)
        _cut_cavities_and_grooves(p, d)

        # Lap recess on +X and -X long edges: cut the top lap_h × lap_inset strip
        for x_start in (0.0, outer_x - p.lap_inset):
            x_center = x_start + p.lap_inset / 2
            y_center = parting_y - p.lap_h / 2
            with Locations(Location((x_center, y_center, outer_z / 2))):
                Box(p.lap_inset, p.lap_h, outer_z, mode=Mode.SUBTRACT)

        # Alignment pins protruding +Y from the parting plane.
        # Plane.ZX has normal +Y; sketch coords map (local_x=world_Z, local_y=world_X).
        for px, pz in _pin_positions(p, d):
            with BuildSketch(Plane.ZX.offset(parting_y)) as sk:
                with Locations((pz, px)):
                    Circle(p.pin_d / 2)
            extrude(to_extrude=sk.sketch, amount=p.pin_h, mode=Mode.ADD)

    return part.part


# ---------------------------------------------------------------------------
# Cap (top half)
# ---------------------------------------------------------------------------
def build_cap(p: VactrolParams):
    d = _dimensions(p)
    outer_x = d["outer_x"]
    outer_y = d["outer_y"]
    outer_z = d["outer_z"]
    parting_y = d["parting_y"]
    cap_h = outer_y - parting_y

    with BuildPart() as part:
        # Main upper body (Y in [parting_y, outer_y], full X and Z)
        with Locations(Location((outer_x / 2, parting_y + cap_h / 2, outer_z / 2))):
            Box(outer_x, cap_h, outer_z)

        # Lap skirts: extend down past parting_y on +X and -X edges. The
        # skirt is inset from the tray's recess by lap_clearance on both X
        # (inner face) and Y (bottom face). The outer X face is flush so the
        # package's outer profile remains continuous.
        skirt_w = p.lap_inset - p.lap_clearance       # 0.40
        skirt_h = p.lap_h - p.lap_clearance           # 0.40
        for x_start in (0.0, outer_x - skirt_w):
            x_center = x_start + skirt_w / 2
            y_center = parting_y - skirt_h / 2
            with Locations(Location((x_center, y_center, outer_z / 2))):
                Box(skirt_w, skirt_h, outer_z)

        # Cavities and wire grooves (clipped to the upper half by the box itself)
        _cut_cavities_and_grooves(p, d)

        # Alignment sockets (Ø + 0.1, deeper than pin by 0.3).
        # Plane.ZX normal is +Y — start 0.05 mm below parting plane to
        # guarantee the cut overlaps the cap's bottom face.
        for px, pz in _pin_positions(p, d):
            with BuildSketch(Plane.ZX.offset(parting_y - 0.05)) as sk:
                with Locations((pz, px)):
                    Circle((p.pin_d + p.socket_d_extra) / 2)
            extrude(
                to_extrude=sk.sketch,
                amount=p.pin_h + p.socket_h_extra + 0.05,
                mode=Mode.SUBTRACT,
            )

    return part.part


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Two-piece vactrol package (3 mm LED + GL55 LDR)")
    parser.add_argument("--stl-tray", type=Path, default=None)
    parser.add_argument("--stl-cap", type=Path, default=None)
    parser.add_argument("--step", type=Path, default=None)
    parser.add_argument(
        "--explode",
        type=float,
        default=0.0,
        help="Y-offset to apply to cap for preview (mm); 0 = assembled.",
    )
    args = parser.parse_args()

    p = VactrolParams()
    d = _dimensions(p)

    tray = build_tray(p)
    cap = build_cap(p)

    print(f"Outer dims (X×Y×Z): {d['outer_x']:.2f} × {d['outer_y']:.2f} × {d['outer_z']:.2f} mm")
    dome_h = p.led_lamp_d / 2
    body_l = p.led_lamp_l - dome_h
    print(
        f"Z stack: wall={p.end_wall_t - p.ldr_recess_depth:.2f}  "
        f"LDR={p.ldr_h:.2f}  "
        f"dome={dome_h:.2f}  "
        f"body={body_l:.2f}  "
        f"lip={p.led_lip_h:.2f}(+{p.clear_led_axial:.2f})  "
        f"wall={p.end_wall_t:.2f}"
    )

    if args.stl_tray is not None:
        export_stl(tray, args.stl_tray)
        print(f"Tray STL → {args.stl_tray}")
    if args.stl_cap is not None:
        export_stl(cap, args.stl_cap)
        print(f"Cap STL → {args.stl_cap}")
    if args.step is not None:
        export_step(Compound([tray, cap]), args.step)
        print(f"STEP → {args.step}")

    cap_placed = Location((0, args.explode, 0)) * cap if args.explode else cap

    try:
        show(
            tray,
            cap_placed,
            names=["tray", "cap"],
            colors=[p.tray_color, p.cap_color],
            reset_camera=Camera.RESET,
            grid=True,
            port=_ocp_port(),
        )
    except RuntimeError as ex:
        print("\nOCP viewer is not reachable.")
        print(f"Details: {ex}")


if __name__ == "__main__":
    main()
