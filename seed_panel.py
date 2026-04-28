"""Seeed XIAO RP2040 + tactile switch + 3 mm LED on 8HP Intellijel 1U panel.

Panel: 40.64 × 39.3 × 2.0 mm, 4× M3 corner mount holes (Tiptop rails).

Front:
  top-left  — 3 mm LED hole
  top-right — switch cap clearance hole
  bottom-centre — USB-C cutout

Back:
  switch pocket (floor + lead-exit slots) below the switch hole
  LED retainer rails either side of the LED hole
  XIAO cage: PCB perpendicular to panel, USB-C aligned with cutout

Separate part: LED retainer clip (slides into rails).

Run:
    ./.venv/bin/python seed_panel.py

Export STLs:
    ./.venv/bin/python seed_panel.py --stl-panel panel.stl --stl-retainer retainer.stl
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
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
    Rectangle,
    RectangleRounded,
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
class PanelParams:
    # Panel (8HP × Intellijel 1U)
    panel_w: float = 40.64
    panel_h: float = 39.3
    thickness: float = 2.0

    # Mount holes — 3 mm clearance, Tiptop rails.
    mount_hole_d: float = 3.2
    mount_x_from_edge: float = 2.54
    mount_y_from_edge: float = 3.0

    # -- Top row: LED + switch --
    # Panel height is ~39 mm but rail-to-rail clear band is only ~20 mm, so
    # top_row_y and usbc_y are pulled in toward the panel vertical centre.
    top_row_y: float = 25.0

    # LED (3 mm body, 3.88 mm × 1.31 mm lip retained from the back)
    # Retainer slides in from −X (panel left), stop wall is on +X side.
    led_x: float = 11.0
    led_hole_d: float = 3.2
    led_lip_d: float = 3.88
    led_lip_h: float = 1.31

    # LED retainer housing (slide-in plate, 1.33 mm behind panel back).
    led_lip_gap: float = 1.33                # clearance behind panel for LED lip
    led_retainer_t: float = 1.5              # retainer plate thickness (Z)
    led_retainer_w: float = 8.0              # plate Y extent (across LED, perpendicular to slide)
    led_retainer_l: float = 9.0              # plate X extent (slide axis)
    led_retainer_slot_w: float = 3.3         # plate's internal slot width (> body, < lip)
    led_retainer_slot_d: float = 5.5         # depth of that slot into the plate
    led_housing_wall: float = 1.5            # wall around the groove

    # Switch cap clearance (cap body is Ø 6 mm per switch_cap.py)
    switch_x: float = 29.64
    switch_hole_d: float = 6.4

    # Switch stack behind panel (measured in −Z from panel back = 0):
    #   [−body_h, 0]                              — switch body (6.35 mm sq × 4 mm), top flush with panel
    #   [−(body_h + ret_t), −body_h]              — slide-in retainer bars
    # The cap (switch_cap.py) has a Ø 7.5 × 1.5 lip that rests on the panel
    # FRONT, so the body top butts straight against the panel back — no
    # extra behind-panel stand-off is needed.
    switch_cap_base_offset: float = 0.0      # body top flush with panel back
    switch_body_size: float = 6.55           # 6.35 + 0.2 tol  (body pocket, Z ∈ [−body_h, 0])
    switch_back_hole_size: float = 7.0       # back floor opening (≥ 6.5, clears body during insertion)
    switch_body_h: float = 4.0
    switch_retainer_t: float = 1.5
    switch_retainer_w: float = 10.0          # bar X extent (slide axis)
    switch_retainer_l: float = 9.0           # overall Y envelope of the two-bar assembly
    switch_housing_wall: float = 1.5

    # Split-bar retainer: two bars that pass either side of the switch's base
    # pins (pins sit at midpoints of the ±X sides of the 6.35 mm body).
    switch_bar_w: float = 2.5                # Y width of each bar
    switch_bar_gap: float = 2.0              # Y gap between bars (for the pin row at y = top_row_y)
    switch_bar_offset_y: float = 2.25        # (bar_w + bar_gap) / 2 — bar centre distance from switch centre
    switch_pin_offset_x: float = 3.175       # pin X offset from switch centre (= body/2)
    switch_pin_clearance_d: float = 1.5      # Ø of the pin through-hole cut in the tower floor/ridge

    # -- Bottom row: USB-C cutout --
    usbc_x: float = 20.32
    usbc_y: float = 13.0
    usbc_cutout_w: float = 10.5              # clears most cable overmolds
    usbc_cutout_h: float = 5.0
    usbc_cutout_r: float = 1.2               # corner radius

    # XIAO cage — PCB perpendicular to panel, USB-C at front edge facing panel.
    # USB-C receptacle (and chip) on PCB −Y face; the slot must clear the
    # full PCB+receptacle envelope during insertion from the rear.
    xiao_pcb_w: float = 17.8                 # panel-plane X dimension of PCB
    xiao_pcb_l: float = 21.0                 # depth into module (length along Z) — matches actual XIAO board
    xiao_pcb_t: float = 1.0
    xiao_usbc_height: float = 4.0            # effective −Y extent from PCB to the lowest component (USB-C + passives)
    xiao_pcb_slot_w: float = 5.5             # slot Y gap — fits pcb_t (1.0) + usbc_height (4.0) + 0.25 clearance per side
    xiao_slot_wall_t: float = 1.5            # wall thickness on the top/bottom/side walls
    xiao_pcb_x_clearance: float = 0.5        # gap between PCB edge and inner face of side walls
    xiao_usbc_recess: float = 0.0            # receptacle face flush with panel back (receptacle protrudes into cutout)
    xiao_usbc_depth: float = 7.5             # receptacle Z depth (bottom wall skips this front region)
    xiao_rear_retainer_t: float = 1.5        # thickness of the rear slide-in stop
    xiao_retainer_l: float = 22.0            # X extent of the retainer bar (1.7 mm sticks out past +X wall for grip)
    xiao_retainer_w: float = 3.5             # Y extent of the retainer bar (slot is this + 0.4)

    # Colour preview
    panel_color: tuple[float, float, float] = (0.86, 0.86, 0.86)
    retainer_color: tuple[float, float, float] = (0.2, 0.5, 0.9)


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------
# Coordinate convention:
#   Origin at bottom-left corner of panel.
#   +Z = panel front (toward user).
#   Panel occupies z in [0, thickness]. Back features extrude in −Z.
def build_panel(p: PanelParams):
    with BuildPart() as part:
        # Base plate — Align.MIN puts it at (0, 0, 0) growing in +X+Y+Z.
        Box(
            p.panel_w,
            p.panel_h,
            p.thickness,
            align=(Align.MIN, Align.MIN, Align.MIN),
        )

        # ---- Front-face through-cuts --------------------------------------
        # Mount holes (corners), LED, switch, USB-C cutout.
        cut_z0 = -0.2
        cut_h = p.thickness + 0.4

        with BuildSketch(Plane.XY.offset(cut_z0)) as front_cuts:
            mount_pts = [
                (p.mount_x_from_edge,               p.mount_y_from_edge),
                (p.panel_w - p.mount_x_from_edge,   p.mount_y_from_edge),
                (p.mount_x_from_edge,               p.panel_h - p.mount_y_from_edge),
                (p.panel_w - p.mount_x_from_edge,   p.panel_h - p.mount_y_from_edge),
            ]
            with Locations(*mount_pts):
                Circle(p.mount_hole_d / 2)
            with Locations((p.led_x, p.top_row_y)):
                Circle(p.led_hole_d / 2)
            with Locations((p.switch_x, p.top_row_y)):
                Circle(p.switch_hole_d / 2)
            with Locations((p.usbc_x, p.usbc_y)):
                RectangleRounded(p.usbc_cutout_w, p.usbc_cutout_h, p.usbc_cutout_r)
        extrude(to_extrude=front_cuts.sketch, amount=cut_h, mode=Mode.SUBTRACT)

        # ---- Back-face features -------------------------------------------
        _add_led_housing(p)
        _add_switch_housing(p)
        _add_xiao_cage(p)

    return part.part


def _add_led_housing(p: PanelParams) -> None:
    """Block behind the LED hole with a horizontal groove for a slide-in plate.

    Slide axis is X: the retainer enters from −X (panel-left) and seats against
    the +X stop wall. Z layout (panel back at z=0, module interior is −Z):
      [−led_lip_gap, 0]                                   — LED lip pocket (Ø 3.88)
      [−(led_lip_gap + led_retainer_t), −led_lip_gap]     — retainer plate groove
    """
    plate_x_far  = p.led_x + p.led_retainer_slot_d
    plate_x_near = plate_x_far - p.led_retainer_l
    housing_x_min = plate_x_near - 1.0
    housing_x_max = plate_x_far + p.led_housing_wall
    housing_x_center = (housing_x_min + housing_x_max) / 2
    housing_x_len = housing_x_max - housing_x_min
    housing_h = p.led_retainer_w + 2 * p.led_housing_wall
    floor_t = 0.8
    housing_d = p.led_lip_gap + p.led_retainer_t + floor_t

    with BuildSketch(Plane.XY.offset(-housing_d)) as hb_sk:
        with Locations((housing_x_center, p.top_row_y)):
            Rectangle(housing_x_len, housing_h)
    extrude(to_extrude=hb_sk.sketch, amount=housing_d, mode=Mode.ADD)

    # LED lip pocket (Ø lip + tolerance) from z=0 to z=-led_lip_gap
    with BuildSketch(Plane.XY.offset(-p.led_lip_gap)) as lip_sk:
        with Locations((p.led_x, p.top_row_y)):
            Circle((p.led_lip_d + 0.3) / 2)
    extrude(to_extrude=lip_sk.sketch, amount=p.led_lip_gap + 0.1, mode=Mode.SUBTRACT)

    # Retainer groove — open at −X (entry), closed at +X (stop wall)
    slot_z_top = -p.led_lip_gap
    slot_z_bot = -(p.led_lip_gap + p.led_retainer_t)
    slot_h = p.led_retainer_w + 0.4
    slot_x_min = housing_x_min - 0.5
    slot_x_max = plate_x_far + 0.2
    slot_x_center = (slot_x_min + slot_x_max) / 2
    slot_x_len = slot_x_max - slot_x_min

    with BuildSketch(Plane.XY.offset(slot_z_bot)) as slot_sk:
        with Locations((slot_x_center, p.top_row_y)):
            Rectangle(slot_x_len, slot_h)
    extrude(to_extrude=slot_sk.sketch, amount=slot_z_top - slot_z_bot, mode=Mode.SUBTRACT)

    # Pass-through in the floor at the LED position — lip + body enter here
    with BuildSketch(Plane.XY.offset(-housing_d - 0.1)) as body_sk:
        with Locations((p.led_x, p.top_row_y)):
            Circle((p.led_lip_d + 0.3) / 2)
    extrude(to_extrude=body_sk.sketch, amount=floor_t + 0.2, mode=Mode.SUBTRACT)


def _add_switch_housing(p: PanelParams) -> None:
    """Tower behind the switch hole: offset pocket + TWO slide-in retainer bars.

    The switch has base pins at the midpoints of its ±X sides. The retainer is
    therefore split into two parallel bars, one on each Y side of the switch
    centreline, separated by a central ridge. Pins pass through the ridge via
    dedicated clearance holes cut all the way through the tower floor.

    Z layout:
      [−cap_off, 0]                             — cap-base clearance
      [−(cap_off + body_h), −cap_off]           — switch body (6.35 sq × 4)
      [−(cap_off + body_h + ret_t), −(cap_off + body_h)] — retainer bar grooves
    """
    cap_off = p.switch_cap_base_offset
    body_h  = p.switch_body_h
    ret_t   = p.switch_retainer_t
    wall    = p.switch_housing_wall

    body_top_z = -cap_off
    body_bot_z = body_top_z - body_h
    ret_top_z  = body_bot_z
    ret_bot_z  = ret_top_z - ret_t
    floor_t    = 0.8
    tower_d    = -ret_bot_z + floor_t

    plate_w = p.switch_retainer_w
    plate_l = p.switch_retainer_l
    tower_x_len = plate_w + wall
    tower_y_len = plate_l + 2 * wall
    tower_x_center = p.switch_x
    tower_y_center = p.top_row_y

    with BuildSketch(Plane.XY.offset(-tower_d)) as tower_sk:
        with Locations((tower_x_center, tower_y_center)):
            Rectangle(tower_x_len, tower_y_len)
    extrude(to_extrude=tower_sk.sketch, amount=tower_d, mode=Mode.ADD)

    # Square column for the switch body — extends from the bottom of the
    # retainer groove (ret_bot_z, i.e., the floor's top face) all the way up
    # to the panel back. Taking it through the ridge Z-range in the body's
    # X,Y footprint clears the insertion path; the ridge survives outside the
    # body footprint (within the slot X range) so the bars still have an
    # inner-wall detent at their ends.
    with BuildSketch(Plane.XY.offset(ret_bot_z)) as bore_sk:
        with Locations((p.switch_x, p.top_row_y)):
            Rectangle(p.switch_body_size, p.switch_body_size)
    extrude(to_extrude=bore_sk.sketch, amount=-ret_bot_z, mode=Mode.SUBTRACT)

    # TWO retainer grooves — open at +X (entry), closed at −X (stop wall),
    # separated in Y by the surviving portions of the central ridge.
    slot_x_min = tower_x_center - tower_x_len / 2 + wall
    slot_x_max = tower_x_center + tower_x_len / 2 + 0.5
    slot_x_center = (slot_x_min + slot_x_max) / 2
    slot_x_len = slot_x_max - slot_x_min
    slot_y_len = p.switch_bar_w + 0.4

    for y_off in (+p.switch_bar_offset_y, -p.switch_bar_offset_y):
        with BuildSketch(Plane.XY.offset(ret_bot_z)) as slot_sk:
            with Locations((slot_x_center, tower_y_center + y_off)):
                Rectangle(slot_x_len, slot_y_len)
        extrude(to_extrude=slot_sk.sketch, amount=ret_top_z - ret_bot_z, mode=Mode.SUBTRACT)

    # Pass-through in the floor — body enters here from behind (≥ 6.5 mm).
    with BuildSketch(Plane.XY.offset(-tower_d - 0.1)) as body_sk:
        with Locations((p.switch_x, p.top_row_y)):
            Rectangle(p.switch_back_hole_size, p.switch_back_hole_size)
    extrude(to_extrude=body_sk.sketch, amount=floor_t + 0.2, mode=Mode.SUBTRACT)


def _add_xiao_cage(p: PanelParams) -> None:
    """PCB cage tied to the panel back via TOP wall + two SIDE walls.

    Geometry along Z (panel back at z=0, module interior is −Z):
      pcb_front_z = −xiao_usbc_recess                 — PCB front edge (≈0.5 mm behind panel)
      pcb_rear_z  = pcb_front_z − xiao_pcb_l           — PCB rear edge
      ret_z_top   = pcb_rear_z − 1.3                   — retainer groove top
      ret_z_bot   = ret_z_top − xiao_rear_retainer_t   — retainer groove bottom
      cage_z_back = ret_z_bot − 0.5                    — cage rear face

    Structural walls:
      - Top wall: full depth z ∈ [cage_z_back, 0], bonded to the panel back.
      - Side walls (×2, at ±X): full depth z ∈ [cage_z_back, 0], also bonded to
        the panel. These bridge top↔bottom walls in Y along the whole length,
        so the bottom wall is never cantilevered.
      - Bottom wall: starts at bot_wall_front_z (to skip the USB-C receptacle
        overhang on the cutout side) and runs to cage_z_back. Supported at
        both X edges by the side walls.
      - NO rear end-cap block. PCB enters from the rear; the retainer's
        stop is the inner face of the −X side wall.
    """
    slot_gap = p.xiao_pcb_slot_w
    wall_t   = p.xiao_slot_wall_t

    # Slot is centred on the full PCB+receptacle assembly (not on the PCB
    # body). With the receptacle hanging off the PCB −Y face, the USB-C
    # midline ends up at usbc_y when the assembly is centred in the slot:
    #   slot_center_y = usbc_y + pcb_t/2
    slot_center_y = p.usbc_y + p.xiao_pcb_t / 2

    pcb_front_z = -p.xiao_usbc_recess
    pcb_rear_z  = pcb_front_z - p.xiao_pcb_l
    ret_z_top   = pcb_rear_z - 0.2
    ret_z_bot   = ret_z_top - p.xiao_rear_retainer_t
    cage_z_back = ret_z_bot - 1.5
    bot_wall_front_z = pcb_front_z - p.xiao_usbc_depth

    wall_x_extent = p.xiao_pcb_w + 2 * (p.xiao_pcb_x_clearance + wall_t)
    top_wall_y = slot_center_y + (slot_gap / 2 + wall_t / 2)
    bot_wall_y = slot_center_y - (slot_gap / 2 + wall_t / 2)
    side_wall_y_extent = (top_wall_y + wall_t / 2) - (bot_wall_y - wall_t / 2)

    # Top wall — full depth, bonded to panel at z=0.
    top_wall_depth = -cage_z_back
    with BuildSketch(Plane.XY.offset(cage_z_back)) as tw_sk:
        with Locations((p.usbc_x, top_wall_y)):
            Rectangle(wall_x_extent, wall_t)
    extrude(to_extrude=tw_sk.sketch, amount=top_wall_depth, mode=Mode.ADD)

    # Bottom wall — only the rear portion (front skipped for USB-C receptacle).
    # Structural support comes from the side walls below, not cantilever.
    bot_wall_depth = bot_wall_front_z - cage_z_back
    with BuildSketch(Plane.XY.offset(cage_z_back)) as bw_sk:
        with Locations((p.usbc_x, bot_wall_y)):
            Rectangle(wall_x_extent, wall_t)
    extrude(to_extrude=bw_sk.sketch, amount=bot_wall_depth, mode=Mode.ADD)

    # Side walls — full depth, also bonded to the panel. These bridge top
    # and bottom walls along the entire cage length and are what prevent the
    # bottom wall from printing as a mid-air bridge.
    for sw_x in (p.usbc_x - (wall_x_extent / 2 - wall_t / 2),
                 p.usbc_x + (wall_x_extent / 2 - wall_t / 2)):
        with BuildSketch(Plane.XY.offset(cage_z_back)) as sw_sk:
            with Locations((sw_x, slot_center_y)):
                Rectangle(wall_t, side_wall_y_extent)
        extrude(to_extrude=sw_sk.sketch, amount=top_wall_depth, mode=Mode.ADD)

    # Retainer groove — open at +X (retainer enters from outside the cage),
    # closed at −X (the −X side wall's inner face is the stop).
    slot_x_min = p.usbc_x - wall_x_extent / 2 - 0.5       # 0.5 past −X outer face (entry side too)
    slot_x_max = p.usbc_x + wall_x_extent / 2 + 0.5       # 0.5 past +X outer face
    slot_x_center = (slot_x_min + slot_x_max) / 2
    slot_x_len = slot_x_max - slot_x_min
    slot_y_len = p.xiao_retainer_w + 0.4

    with BuildSketch(Plane.XY.offset(ret_z_bot)) as g_sk:
        with Locations((slot_x_center, slot_center_y)):
            Rectangle(slot_x_len, slot_y_len)
    extrude(to_extrude=g_sk.sketch, amount=ret_z_top - ret_z_bot, mode=Mode.SUBTRACT)



# ---------------------------------------------------------------------------
# Separate slide-in retainer parts
# ---------------------------------------------------------------------------
def build_led_retainer(p: PanelParams):
    """Flat plate with a rectangular slot that slips around the LED body,
    pinching the lip against the panel back. Slide axis is X: the entry edge
    is at plate-local X=0 and the slot is cut inward from that edge."""
    with BuildPart() as part:
        with BuildSketch(Plane.XY) as sk:
            Rectangle(p.led_retainer_l, p.led_retainer_w, align=(Align.MIN, Align.CENTER))
            with Locations((p.led_retainer_slot_d / 2, 0)):
                Rectangle(p.led_retainer_slot_d, p.led_retainer_slot_w, mode=Mode.SUBTRACT)
        extrude(to_extrude=sk.sketch, amount=p.led_retainer_t, mode=Mode.ADD)
    return part.part


def build_switch_retainer(p: PanelParams):
    """Two parallel bars that slide in on either side of the switch's base
    pins (pins lie on the Y-centreline, at x = switch_x ± pin_offset). Both
    bars are emitted from a single BuildPart so they ship as one file/STL
    but print as two separate pieces for installation around the pins."""
    with BuildPart() as part:
        with BuildSketch(Plane.XY) as sk:
            with Locations(
                (0, +p.switch_bar_offset_y),
                (0, -p.switch_bar_offset_y),
            ):
                Rectangle(p.switch_retainer_w, p.switch_bar_w)
        extrude(to_extrude=sk.sketch, amount=p.switch_retainer_t, mode=Mode.ADD)
    return part.part


def build_xiao_retainer(p: PanelParams):
    """Flat bar that slides through the rear of the XIAO cage to stop the
    PCB from backing out."""
    with BuildPart() as part:
        with BuildSketch(Plane.XY) as sk:
            Rectangle(p.xiao_retainer_l, p.xiao_retainer_w, align=(Align.CENTER, Align.CENTER))
        extrude(to_extrude=sk.sketch, amount=p.xiao_rear_retainer_t, mode=Mode.ADD)
    return part.part


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _place_led_retainer(part, p: PanelParams):
    """Move the LED retainer into its assembled pose behind the panel."""
    plate_x_far  = p.led_x + p.led_retainer_slot_d
    plate_x_near = plate_x_far - p.led_retainer_l
    # Local frame: X in [0, plate_l], Y centred, Z in [0, plate_t].
    dx = plate_x_near
    dy = p.top_row_y
    dz = -(p.led_lip_gap + p.led_retainer_t)
    return Location((dx, dy, dz)) * part


def _place_switch_retainer(part, p: PanelParams):
    """Assembled pose — bars' −X edges flush with the tower's −X stop wall.
    Bars are already Y-symmetric around plate-local y=0, so dy = top_row_y."""
    ret_bot_z = -(p.switch_cap_base_offset + p.switch_body_h + p.switch_retainer_t)
    stop_wall_inside_x = p.switch_x - p.switch_retainer_w / 2 + p.switch_housing_wall / 2
    dx = stop_wall_inside_x + p.switch_retainer_w / 2
    dy = p.top_row_y
    dz = ret_bot_z
    return Location((dx, dy, dz)) * part


def _place_xiao_retainer(part, p: PanelParams):
    """Assembled pose — retainer fully inserted against the −X side wall."""
    slot_center_y = p.usbc_y + p.xiao_pcb_t / 2
    pcb_front_z = -p.xiao_usbc_recess
    pcb_rear_z = pcb_front_z - p.xiao_pcb_l
    ret_z_bot = pcb_rear_z - 0.2 - p.xiao_rear_retainer_t
    wall_x_extent = p.xiao_pcb_w + 2 * (p.xiao_pcb_x_clearance + p.xiao_slot_wall_t)
    stop_wall_inside_x = p.usbc_x - wall_x_extent / 2 + p.xiao_slot_wall_t
    dx = stop_wall_inside_x + p.xiao_retainer_l / 2
    dy = slot_center_y
    dz = ret_z_bot
    return Location((dx, dy, dz)) * part


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build XIAO RP2040 1U 8HP panel")
    parser.add_argument("--stl-panel",          type=Path, default=None)
    parser.add_argument("--stl-led-retainer",   type=Path, default=None)
    parser.add_argument("--stl-switch-retainer",type=Path, default=None)
    parser.add_argument("--stl-xiao-retainer",  type=Path, default=None)
    parser.add_argument("--stl-xiao-retainer-pack", type=Path, default=None,
                        help="Stem path for a test pack of XIAO retainers at widths 3.3/3.5/3.7/3.9 mm "
                             "(e.g. 'seed_xiao.stl' writes seed_xiao_loose.stl, _std.stl, _snug.stl, _press.stl)")
    parser.add_argument("--step",               type=Path, default=None)
    args = parser.parse_args()

    p = PanelParams()
    panel            = build_panel(p)
    led_retainer     = build_led_retainer(p)
    switch_retainer  = build_switch_retainer(p)
    xiao_retainer    = build_xiao_retainer(p)

    led_placed    = _place_led_retainer(led_retainer, p)
    switch_placed = _place_switch_retainer(switch_retainer, p)
    xiao_placed   = _place_xiao_retainer(xiao_retainer, p)

    if args.stl_panel is not None:
        export_stl(panel, args.stl_panel)
        print(f"Panel STL → {args.stl_panel}")
    if args.stl_led_retainer is not None:
        export_stl(led_retainer, args.stl_led_retainer)
        print(f"LED retainer STL → {args.stl_led_retainer}")
    if args.stl_switch_retainer is not None:
        export_stl(switch_retainer, args.stl_switch_retainer)
        print(f"Switch retainer STL → {args.stl_switch_retainer}")
    if args.stl_xiao_retainer is not None:
        export_stl(xiao_retainer, args.stl_xiao_retainer)
        print(f"XIAO retainer STL → {args.stl_xiao_retainer}")
    if args.stl_xiao_retainer_pack is not None:
        stem = args.stl_xiao_retainer_pack
        for label, w in (("loose", 3.3), ("std", 3.5), ("snug", 3.7), ("press", 3.9)):
            variant_part = build_xiao_retainer(replace(p, xiao_retainer_w=w))
            out = stem.with_name(f"{stem.stem}_{label}{stem.suffix}")
            export_stl(variant_part, out)
            print(f"XIAO retainer ({label}, w={w} mm) → {out}")
    if args.step is not None:
        export_step(Compound([panel, led_placed, switch_placed, xiao_placed]), args.step)
        print(f"STEP → {args.step}")

    try:
        show(
            panel, led_placed, switch_placed, xiao_placed,
            names=["panel", "led_retainer", "switch_retainer", "xiao_retainer"],
            colors=[p.panel_color, p.retainer_color, p.retainer_color, p.retainer_color],
            reset_camera=Camera.RESET, grid=True, port=_ocp_port(),
        )
    except RuntimeError as ex:
        print("\nOCP viewer is not reachable.")
        print("- Open 'OCP CAD Viewer' in VS Code and ensure the backend is running.")
        print(f"\nDetails: {ex}")


if __name__ == "__main__":
    main()
