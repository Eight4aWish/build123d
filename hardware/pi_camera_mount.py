"""Tilting slide-in mount for the Waveshare RPi Camera (B) (build123d).

The Pi Hut sells this board as "Camera Board for Raspberry Pi - Adjustable-Focus
Lens (5MP)", SKU WAV-8193: OV5647, 6 mm M12-style lens, 32 x 32 mm PCB with an
M2 hole in each corner on a 28 mm square. Dimensions from Waveshare's outline
drawing; PCB thickness 1.66 mm by calipers (the grooves are sized from it;
--pcb-t overrides).

Three printed parts:

  cradle  U-channel the camera board slides down into. Edge grooves hold it,
          corner ledges stop it, the front is open so the lens can still be
          turned to focus, and the FFC connector can face up or down (the gap
          between the ledges lets the cable out of the bottom). A tab on the
          back carries the tilt pivot.
  base    Sits on the 12 mm material over its 6 mm slot. A fork holds the
          cradle's tab; one prong traps an M5 nut for the tilt bolt, and a
          trapped M5 nut between the prongs takes the bolt that comes up
          through the slot.
  knob    Print two. Each grips an M5 hex-head bolt, so both the tilt and the
          slot clamp tighten by hand.

Hardware: 2x M5x25 hex-head bolts (ISO 4017, 8 mm AF), 2x M5 hex nuts.

Print orientation is the model's own: cradle upright on its bottom edge,
base flat, knob face-down. No supports.

Run:
  ./.venv/bin/python hardware/pi_camera_mount.py --tilt 30

Export:
  ./.venv/bin/python hardware/pi_camera_mount.py --no-show --stl-dir exports/hardware
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path

from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    Compound,
    Cylinder,
    Location,
    Locations,
    Mode,
    Part,
    Plane,
    Polygon,
    RectangleRounded,
    RegularPolygon,
    chamfer,
    export_step,
    export_stl,
    extrude,
)
from ocp_vscode import Camera, show


def _ocp_port(default: int = 3939) -> int:
    try:
        return int(os.environ.get("OCP_VSCODE_PORT", str(default)))
    except ValueError:
        return default


# ── Parameters ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MountParams:
    # Camera board
    pcb_w: float = 32.0
    pcb_h: float = 32.0
    pcb_t: float = 1.66         # measured; the groove gap is pcb_t + groove_fit

    # Cradle
    side_fit: float = 0.3       # board edge to groove floor, each side
    groove_fit: float = 0.4     # front-back play in the groove
    wall: float = 2.2           # side wall outside the groove floor
    lip_w: float = 1.5          # lip overlap onto the board face (parts start ~2.7 in)
    lip_t: float = 1.6
    back_t: float = 3.0
    relief: float = 1.5         # recess behind the board for back-side parts
    ledge_h: float = 2.0        # floor under the board's corners
    ledge_gap: float = 24.4     # opening between the ledges (FFC connector is 23.2 wide)
    rail_h: float = 32.0        # flush with the board top; lift it out by the lens holder
    lead_in: float = 0.6        # chamfer on the groove mouths

    # Tilt joint (pivot axis runs left-right, behind the cradle)
    tab_w: float = 8.0
    joint_r: float = 6.5        # tab and prong radius about the pivot
    pivot_back: float = 11.5    # back plate rear face to pivot axis; > knob radius
    joint_gap: float = 0.3      # tab to prong, each side
    prong_knob_t: float = 5.0   # prong the knob bears on
    prong_nut_t: float = 8.0    # prong with the trapped nut
    prong_flare: float = 6.0    # prong foot runs this far back for stiffness

    # Base
    pivot_z: float = 26.0       # pivot axis above the underside of the base
    base_t: float = 5.0
    base_w: float = 32.0
    base_corner_r: float = 3.0
    block_h: float = 10.0       # solid block between the prongs, holds the clamp nut
    clamp_y: float = -3.0       # slot bolt position (pivot is y = 0)
    clamp_floor: float = 2.5    # base material under the clamp nut

    # M5 hardware
    bolt_d: float = 5.4         # clearance
    nut_af: float = 8.3         # M5 nut / hex head is 8.0 AF
    nut_depth: float = 4.3      # nut is 4.0 thick

    # Knob
    knob_d: float = 20.0
    knob_h: float = 8.0
    knob_floor: float = 3.0     # under the bolt head
    knob_head_af: float = 8.15  # press fit on the 8.0 AF head; glue it if loose
    knob_scallops: int = 8
    knob_scallop_r: float = 2.5

    # Mounting surface (for the assembly view and bolt length)
    material_t: float = 12.0
    slot_w: float = 6.0

    # Derived ────────────────────────────────────────────────────────────

    @property
    def channel_half(self) -> float:
        return self.pcb_w / 2 + self.side_fit

    @property
    def outer_half(self) -> float:
        return self.channel_half + self.wall

    @property
    def groove_t(self) -> float:
        return self.pcb_t + self.groove_fit

    @property
    def cradle_front(self) -> float:
        return self.groove_t + self.lip_t

    @property
    def cradle_top(self) -> float:
        return self.ledge_h + self.rail_h

    @property
    def pivot_local(self) -> tuple[float, float]:
        """Pivot (y, z) in the cradle's own frame: board back face y = 0, bottom z = 0."""
        return (-self.back_t - self.pivot_back, self.joint_r)

    @property
    def prong_knob_x(self) -> tuple[float, float]:
        inner = self.tab_w / 2 + self.joint_gap
        return (-inner - self.prong_knob_t, -inner)

    @property
    def prong_nut_x(self) -> tuple[float, float]:
        inner = self.tab_w / 2 + self.joint_gap
        return (inner, inner + self.prong_nut_t)

    @property
    def base_y(self) -> tuple[float, float]:
        return (-self.joint_r - self.prong_flare - 2.0, self.joint_r + 2.0)

    @property
    def tilt_bolt_len(self) -> float:
        """Shank needed to pass the tilt joint and fill the nut."""
        span = self.prong_nut_x[1] - self.prong_knob_x[0]
        return self.knob_floor + span - self.nut_depth + 4.0

    @property
    def clamp_bolt_len(self) -> float:
        """Shank needed to pass knob, material and floor and fill the nut."""
        return self.knob_floor + self.material_t + self.clamp_floor + 4.0


# ── Parts ─────────────────────────────────────────────────────────────────

def make_cradle(p: MountParams = MountParams()) -> Part:
    """Cradle in its print pose: board back face at y = 0, lens towards +Y."""
    ch, oh = p.channel_half, p.outer_half
    yb, yf, top = -p.back_t, p.cradle_front, p.cradle_top
    yp, zp = p.pivot_local
    r = p.joint_r

    with BuildPart() as cradle:
        with Locations((0, (yb + yf) / 2, 0)):
            Box(2 * oh, yf - yb, top, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # Board channel: the full board width, pcb_t + fit deep
        with Locations((0, 0, p.ledge_h)):
            Box(2 * ch, p.groove_t, top, align=(Align.CENTER, Align.MIN, Align.MIN),
                mode=Mode.SUBTRACT)
        # Front window between the lips
        with Locations((0, p.groove_t, p.ledge_h)):
            Box(p.pcb_w - 2 * p.lip_w, p.lip_t + 1, top,
                align=(Align.CENTER, Align.MIN, Align.MIN), mode=Mode.SUBTRACT)
        # Gap between the corner ledges for the connector and FFC
        Box(p.ledge_gap, yf + 1, p.ledge_h, align=(Align.CENTER, Align.MIN, Align.MIN),
            mode=Mode.SUBTRACT)
        # Recess behind the board so only its edge strips touch the back plate
        with Locations((0, 0, p.ledge_h)):
            Box(p.pcb_w - 2 * p.lip_w, p.relief, top,
                align=(Align.CENTER, Align.MAX, Align.MIN), mode=Mode.SUBTRACT)

        # Lead-in on the groove mouths at the top
        mouth = [
            e for e in cradle.edges().filter_by(Axis.Z, reverse=True)
            if abs(e.center().Z - top) < 1e-6
            and p.pcb_w / 2 - p.lip_w - 1e-6 < abs(e.center().X) < ch + 1e-6
            and -1e-6 < e.center().Y < p.groove_t + 1e-6
        ]
        chamfer(mouth, p.lead_in)

        # Pivot tab: a stadium end on the back plate, flat on the bed, with a
        # 45-degree fill under the rear of the circle so it prints unsupported.
        with BuildSketch(Plane.YZ):
            with Locations((yp, zp)):
                Circle(r)
            Polygon((yp, 0), (yb + 0.5, 0), (yb + 0.5, 2 * r), (yp, 2 * r), align=None)
            Polygon(
                (yp, 0),
                (yp - r * (math.sqrt(2) - 1), 0),
                (yp - r / math.sqrt(2), zp - r / math.sqrt(2)),
                (yp, zp),
                align=None,
            )
        extrude(amount=p.tab_w / 2, both=True)

        with Locations((0, yp, zp)):
            Cylinder(p.bolt_d / 2, p.tab_w + 2, rotation=(0, 90, 0), mode=Mode.SUBTRACT)

    return cradle.part


def make_base(p: MountParams = MountParams()) -> Part:
    """Base in its print pose: underside at z = 0, pivot axis at (y=0, z=pivot_z)."""
    r, pz = p.joint_r, p.pivot_z
    y0, y1 = p.base_y
    s2 = math.sqrt(2)

    with BuildPart() as base:
        with BuildSketch():
            with Locations((0, (y0 + y1) / 2)):
                RectangleRounded(p.base_w, y1 - y0, p.base_corner_r)
        extrude(amount=p.base_t)

        # Prongs: circle about the pivot on a stem that stays behind y = 0, so
        # the cradle's back plate can swing down in front of it. The 45-degree
        # chamfer under the front of the circle prints without support.
        def prong_profile():
            with Locations((0, pz)):
                Circle(r)
            Polygon(
                (-r - p.prong_flare, 0),
                (0, 0),
                (0, pz - r * s2),
                (r / s2, pz - r / s2),
                (0, pz),
                (-r, pz),
                align=None,
            )

        for x0, x1 in (p.prong_knob_x, p.prong_nut_x):
            with BuildSketch(Plane.YZ.offset(x0)):
                prong_profile()
            extrude(amount=x1 - x0)

        # Block between the prongs: ties them together and houses the clamp nut
        inner = p.tab_w / 2 + p.joint_gap
        clamp_front = p.clamp_y + p.nut_af / math.sqrt(3) + 1.5
        with Locations((0, -r - p.prong_flare, 0)):
            Box(2 * inner, clamp_front + r + p.prong_flare, p.block_h,
                align=(Align.CENTER, Align.MIN, Align.MIN))

        # Tilt bolt through both prongs, nut trapped in the outer face of one
        with Locations((0, 0, pz)):
            Cylinder(p.bolt_d / 2, 2 * p.prong_nut_x[1] + 10, rotation=(0, 90, 0),
                     mode=Mode.SUBTRACT)
        with BuildSketch(Plane.YZ.offset(p.prong_nut_x[1])):
            with Locations((0, pz)):
                # vertex up so the pocket roof is self-supporting
                RegularPolygon(p.nut_af / 2, 6, major_radius=False, rotation=30)
        extrude(amount=-p.nut_depth, mode=Mode.SUBTRACT)

        # Slot clamp: bolt up from under the material into a nut dropped in
        # from above, flats facing the prongs so it passes between them.
        with Locations((0, p.clamp_y, 0)):
            Cylinder(p.bolt_d / 2, p.block_h, align=(Align.CENTER, Align.CENTER, Align.MIN),
                     mode=Mode.SUBTRACT)
        with BuildSketch(Plane.XY.offset(p.block_h)):
            with Locations((0, p.clamp_y)):
                RegularPolygon(p.nut_af / 2, 6, major_radius=False, rotation=30)
        extrude(amount=-(p.block_h - p.clamp_floor), mode=Mode.SUBTRACT)

    return base.part


def make_knob(p: MountParams = MountParams()) -> Part:
    """Thumb knob in its print pose: bearing face at z = 0, bolt head pocket on top."""
    R = p.knob_d / 2

    with BuildPart() as knob:
        Cylinder(R, p.knob_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
        chamfer(knob.edges().filter_by(Axis.Z, reverse=True), 0.8)
        with Locations(*[
            (math.cos(a) * (R + 1.2), math.sin(a) * (R + 1.2), 0)
            for a in (2 * math.pi * i / p.knob_scallops for i in range(p.knob_scallops))
        ]):
            Cylinder(p.knob_scallop_r, p.knob_h, align=(Align.CENTER, Align.CENTER, Align.MIN),
                     mode=Mode.SUBTRACT)
        Cylinder(p.bolt_d / 2, p.knob_h, align=(Align.CENTER, Align.CENTER, Align.MIN),
                 mode=Mode.SUBTRACT)
        with BuildSketch(Plane.XY.offset(p.knob_h)):
            RegularPolygon(p.knob_head_af / 2, 6, major_radius=False)
        extrude(amount=-(p.knob_h - p.knob_floor), mode=Mode.SUBTRACT)

    return knob.part


def make_camera(p: MountParams = MountParams()) -> Part:
    """Rough stand-in for the camera board, in the cradle's frame, connector down."""
    z0 = p.ledge_h
    zc = z0 + p.pcb_h / 2
    with BuildPart() as cam:
        with Locations((0, 0, z0)):
            Box(p.pcb_w, p.pcb_t, p.pcb_h, align=(Align.CENTER, Align.MIN, Align.MIN))
        with Locations((0, p.pcb_t, zc)):
            Box(17.4, 6.0, 17.4, align=(Align.CENTER, Align.MIN, Align.CENTER))
            Box(25.0, 3.0, 5.0, align=(Align.CENTER, Align.MIN, Align.CENTER))
        with Locations((0, p.pcb_t + 6.0, zc)):
            Cylinder(7.25, 15.0, rotation=(-90, 0, 0), align=(Align.CENTER, Align.CENTER, Align.MIN))
        with Locations((0, p.pcb_t, z0 - 0.4)):
            Box(23.2, 2.5, 5.5, align=(Align.CENTER, Align.MIN, Align.MIN))
    return cam.part


# ── Assembly ──────────────────────────────────────────────────────────────

def cradle_placement(p: MountParams, tilt: float) -> Location:
    """Cradle frame -> base frame. Positive tilt aims the lens up."""
    yp, zp = p.pivot_local
    return (
        Location((0, 0, p.pivot_z))
        * Location((0, 0, 0), (tilt, 0, 0))
        * Location((0, -yp, -zp))
    )


def tilt_knob_placement(p: MountParams) -> Location:
    return Location((p.prong_knob_x[0], 0, p.pivot_z), (0, -90, 0))


def clamp_knob_placement(p: MountParams) -> Location:
    return Location((0, p.clamp_y, -p.material_t), (180, 0, 0))


def make_material(p: MountParams) -> Part:
    """The 12 mm board with its 6 mm slot running front to back, for the view only."""
    with BuildPart() as mat:
        Box(90, 90, p.material_t, align=(Align.CENTER, Align.CENTER, Align.MAX))
        with Locations((0, p.clamp_y, 0)):
            Box(p.slot_w, 40, p.material_t, align=(Align.CENTER, Align.CENTER, Align.MAX),
                mode=Mode.SUBTRACT)
    return mat.part


def tilt_range(p: MountParams, step: float = 5.0) -> tuple[float, float]:
    """Widest tilt either way with no contact between cradle+camera and base,
    knob or the mounting surface."""
    cradle = make_cradle(p) + make_camera(p)
    fixed = make_base(p) + tilt_knob_placement(p) * make_knob(p)
    surface = Box(200, 200, 50, align=(Align.CENTER, Align.CENTER, Align.MAX))

    def touches(a: Part, b: Part) -> bool:
        common = a & b  # None when they don't meet
        return common is not None and common.volume > 1e-3

    def clear(angle: float) -> bool:
        moving = cradle_placement(p, angle) * cradle
        return not touches(moving, fixed) and not touches(moving, surface)

    limits = []
    for sign in (1, -1):
        angle = 0.0
        while angle + step <= 120 and clear(sign * (angle + step)):
            angle += step
        limits.append(sign * angle)
    return limits[1], limits[0]


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Tilting slide-in mount for the Waveshare RPi Camera (B)")
    parser.add_argument("--pcb-t", type=float, default=None, help="Measured camera PCB thickness (mm)")
    parser.add_argument("--tilt", type=float, default=0.0, help="Viewer pose, degrees (+ aims up)")
    parser.add_argument("--stl-dir", type=Path, default=None, help="Export the three printable STLs here")
    parser.add_argument("--step", type=Path, default=None, help="Export the posed assembly as STEP")
    parser.add_argument("--check", action="store_true", help="Report the collision-free tilt range")
    parser.add_argument("--no-show", action="store_true", help="Skip the OCP viewer")
    args = parser.parse_args()

    p = MountParams()
    if args.pcb_t is not None:
        p = replace(p, pcb_t=args.pcb_t)

    cradle, base, knob = make_cradle(p), make_base(p), make_knob(p)
    print(f"Groove gap {p.groove_t:.2f} mm for a {p.pcb_t:.2f} mm board")
    print(f"Bolts: tilt needs {p.tilt_bolt_len:.1f} mm of shank, "
          f"slot clamp {p.clamp_bolt_len:.1f} mm  -> 2x M5x25 hex head")

    if args.stl_dir is not None:
        args.stl_dir.mkdir(parents=True, exist_ok=True)
        for name, part in (("cradle", cradle), ("base", base), ("knob", knob)):
            path = args.stl_dir / f"pi_camera_{name}.stl"
            export_stl(part, path)
            print(f"  STL -> {path}")

    if args.check:
        lo, hi = tilt_range(p)
        print(f"Tilt range {lo:+.0f} to {hi:+.0f} degrees")

    posed = {
        "base": base,
        "cradle": cradle_placement(p, args.tilt) * cradle,
        "camera": cradle_placement(p, args.tilt) * make_camera(p),
        "tilt_knob": tilt_knob_placement(p) * knob,
        "clamp_knob": clamp_knob_placement(p) * knob,
        "material": make_material(p),
    }

    if args.step is not None:
        export_step(Compound(children=list(posed.values())), args.step)
        print(f"  STEP -> {args.step}")

    if not args.no_show:
        try:
            show(
                *posed.values(),
                names=list(posed.keys()),
                colors=["steelblue", "orange", "green", "steelblue", "steelblue", "burlywood"],
                alphas=[1, 1, 1, 1, 1, 0.4],
                reset_camera=Camera.RESET,
                port=_ocp_port(),
            )
        except RuntimeError as ex:
            print("\nOCP viewer is not reachable.")
            print("- Open 'OCP CAD Viewer' in VS Code and ensure the backend is running.")
            print(f"\nDetails: {ex}")


if __name__ == "__main__":
    main()
