"""ClkLinkRec — N8Synth 4HP 1x6 panel.

Eurorack module derived from the "N8 Synth 4HP 1x6 Label Template v1.1" SVG
(single column of six holes on a 4HP/3U panel) and the combo-hole logic of
`CortHex.py`.

Hole layout (top -> bottom), all on the panel centreline:
  - Top    : main jack + companion 3 mm LED through-hole
  - Middle : four standard jack holes
  - Bottom : main jack + companion 3 mm LED through-hole

The main-jack diameter and the button/LED pairing (LED diameter + relative
offset) are copied from CortHex so the LED combo matches the rest of the range.
Two offset Eurorack mounting slots (top + bottom) come straight from the 4HP
template.

Run:
  ./.venv/bin/python clklinkrec.py

Export:
  ./.venv/bin/python clklinkrec.py --export-mode base   --stl-base clklinkrec_base.stl
  ./.venv/bin/python clklinkrec.py --export-mode labels --stl-labels clklinkrec_labels.stl

Paper template (1:1):
  ./.venv/bin/python clklinkrec.py --template-svg clklinkrec_template.svg
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from build123d import (
    Align,
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    FontStyle,
    Location,
    Locations,
    Mode,
    Plane,
    Rectangle,
    SlotOverall,
    Text,
    extrude,
    export_step,
    export_stl,
)
from build123d.exporters import ExportDXF, ExportSVG
from ocp_vscode import Camera, show

ExportMode = Literal["combined", "base", "labels"]


def _ocp_port(default: int = 3939) -> int:
    try:
        raw = os.environ.get("OCP_VSCODE_PORT") or os.environ.get("OCP_PORT") or str(default)
        return int(raw)
    except ValueError:
        return default


# --- SVG extraction helpers ------------------------------------------------
#
# Unlike the 10HP template, the 4HP "N8 Synth 4HP 1x6 Label Template" SVG is
# authored directly in millimetres (width="297mm", viewBox in mm), so the unit
# scale is 1:1.  We keep the same conversion machinery as CortHex for clarity.

SVG_UNIT_TO_MM = 1.0

# Panel rectangle in SVG units (rect20228).
RECT_X_U = 20.132034
RECT_Y_U = 20.000034
RECT_W_U = 19.999929
RECT_H_U = 128.45914


def _u2mm(u: float) -> float:
    return u * SVG_UNIT_TO_MM


SVG_PANEL_W_MM = _u2mm(RECT_W_U)
SVG_PANEL_H_MM = _u2mm(RECT_H_U)


def _svg_to_panel_xy(cx_u: float, cy_u: float) -> tuple[float, float]:
    """Convert SVG (cx,cy) in SVG units to panel XY in mm.

    SVG Y increases downward. Panel Y increases upward from bottom edge.
    """

    x_mm = _u2mm(cx_u - RECT_X_U)
    y_mm = _u2mm((RECT_Y_U + RECT_H_U) - cy_u)
    return (x_mm, y_mm)


# Jack-hole centres (single column, 6 rows) from the SVG ellipses.
# All share the same cx (panel centreline); cy runs top -> bottom.
HOLE_CX_U = 30.130159
HOLE_CY_U = (36.939816, 54.209827, 71.479752, 88.749763, 106.0197, 123.2897)

# Eurorack mounting slots (path20273 top, path20277 bottom). Horizontal
# stadium slots, both offset to the same side of the centreline, taken
# straight from the template.
MOUNT_SLOT_CX_U = 27.65
MOUNT_SLOT_CY_U = (22.795, 145.448)


def _text_local_offset(
    txt: str,
    *,
    font: str,
    style: FontStyle,
    font_size: float,
) -> tuple[float, float]:
    """Compute offset to center text at the origin."""
    with BuildSketch(Plane.XY) as sk:
        Text(txt, font_size=font_size, font=font, font_style=style)
    bb = sk.sketch.bounding_box()
    ax = (bb.min.X + bb.max.X) / 2
    ay = (bb.min.Y + bb.max.Y) / 2
    return (-ax, -ay)


@dataclass(frozen=True)
class FaceplateParams:
    # Panel — standard 4HP/3U. Doepfer cuts 4 HP to 20.0 mm (nominal pitch is
    # 4 x 5.08 = 20.32, minus ~0.3 mm clearance so modules don't bind in a row).
    panel_w: float = 20.0
    panel_h: float = 128.5
    thickness: float = 2.0

    # Eurorack mounting slots (top + bottom, copied from the 4HP template).
    add_mount_slots: bool = True
    mount_slot_overall_len: float = 7.0
    mount_hole_d: float = 3.2

    # Control holes — Ø from the SVG ellipse rx (matches CortHex jacks).
    hole_d: float = _u2mm(2 * 3.4509125)  # ~6.90 mm
    hole_oversize: float = 0.0

    # Labels (separate solid).
    label_height: float = 0.2
    label_font: str = "Arial"
    label_font_style: FontStyle = FontStyle.BOLD
    label_size: float = 2.4
    label_offset: tuple[float, float] = (0.0, -6.0)

    # Branding text — two horizontal lines stacked in the gap between the
    # bottom mount slot and the lowest LED. Module name on top, maker below.
    brand_text_module: str = "ClkLinkRec"
    brand_text_maker: str = "Eight4aWish"
    brand_size: float = 2.4
    brand_height: float = 0.2
    brand_line_gap: float = 3.2  # vertical spacing between the two brand lines

    # Per-hole labels — 6 entries, top -> bottom. All labels sit *above* their
    # hole so they clear the LEDs on the top/bottom holes.
    hole_labels_above: tuple[str, ...] = (
        "LINK",    # 0 top    + LED
        "CLOCK",   # 1
        "RESET OUT",     # 2
        "RUNNING",    # 3
        "RESET IN",    # 4
        "RECORD",    # 5 bottom + LED
    )

    # Keep every hole.
    remove_hole_indices: tuple[int, ...] = ()

    # Jack + 3 mm LED combo holes (top and bottom). Each entry is the index
    # (in `holes_sorted`) of a jack that gets a paired LED hole. Sizing and
    # relative offset are copied verbatim from CortHex.
    # LINK (idx 0) and RECORD (idx 5) are momentary switches, each with a
    # companion 3 mm LED.
    button_led_indices: tuple[int, ...] = (0, 5)
    # Offset of the LED hole centre from the jack centre, in panel-frame
    # coordinates: (+X right, -Y down the faceplate).
    button_led_offset: tuple[float, float] = (2.5, -6.5)
    button_led_d: float = 3.2
    # LED colours, parallel to button_led_indices: LINK = blue, RECORD = red.
    button_led_colors: tuple[str, ...] = ("blue", "red")

    # Real-life finish: dark grey panel with white lettering.
    base_color: tuple[float, float, float] = (0.028, 0.028, 0.032)
    label_color: tuple[float, float, float] = (0.92, 0.92, 0.93)

    def _content_offsets(self) -> tuple[float, float]:
        """Center SVG-derived geometry onto the standard panel size."""
        dx = (self.panel_w - SVG_PANEL_W_MM) / 2
        dy = (self.panel_h - SVG_PANEL_H_MM) / 2
        return (dx, dy)

    def holes_xy(self) -> list[tuple[float, float]]:
        dx, dy = self._content_offsets()
        holes: list[tuple[float, float]] = []
        for cy_u in HOLE_CY_U:
            x0, y0 = _svg_to_panel_xy(HOLE_CX_U, cy_u)
            holes.append((x0 + dx, y0 + dy))
        return holes

    def holes_sorted(self) -> list[tuple[float, float]]:
        holes = self.holes_xy()
        holes.sort(key=lambda pt: (-pt[1], pt[0]))  # top-to-bottom, left-to-right
        return holes

    def holes_enabled_sorted(self) -> list[tuple[float, float]]:
        holes = self.holes_sorted()
        disabled = set(int(i) for i in self.remove_hole_indices)
        return [pt for idx, pt in enumerate(holes) if idx not in disabled]

    def button_led_centers(self) -> list[tuple[float, float]]:
        """LED through-hole centres for each jack+LED combo hole."""
        holes = self.holes_sorted()
        dx, dy = self.button_led_offset
        return [(holes[i][0] + dx, holes[i][1] + dy) for i in self.button_led_indices]

    def eurorack_mount_slot_centers(self) -> list[tuple[float, float]]:
        dx, dy = self._content_offsets()
        centers: list[tuple[float, float]] = []
        for cy_u in MOUNT_SLOT_CY_U:
            x0, y0 = _svg_to_panel_xy(MOUNT_SLOT_CX_U, cy_u)
            centers.append((x0 + dx, y0 + dy))
        return centers

    def brand_gap_center_y(self) -> float:
        """Y midpoint of the gap between the bottom mount slot and lowest LED."""
        mount_top = min(y for _, y in self.eurorack_mount_slot_centers()) + self.mount_hole_d / 2
        led_centers = self.button_led_centers()
        led_bottom = min(y for _, y in led_centers) - self.button_led_d / 2 if led_centers else self.panel_h
        return (mount_top + led_bottom) / 2

    def brand_module_pos(self) -> tuple[float, float]:
        return (self.panel_w / 2, self.brand_gap_center_y() + self.brand_line_gap / 2)

    def brand_maker_pos(self) -> tuple[float, float]:
        return (self.panel_w / 2, self.brand_gap_center_y() - self.brand_line_gap / 2)


def build_base(params: FaceplateParams) -> object:
    """Return the base panel solid with cutouts subtracted."""

    cut_z0 = -0.2
    cut_h = params.thickness + 0.4

    holes = params.holes_enabled_sorted()

    with BuildPart() as p:
        Box(
            params.panel_w,
            params.panel_h,
            params.thickness,
            align=(Align.MIN, Align.MIN, Align.MIN),
            mode=Mode.ADD,
        )

        with BuildSketch(Plane.XY.offset(cut_z0)):
            if params.add_mount_slots:
                with Locations(*params.eurorack_mount_slot_centers()):
                    SlotOverall(params.mount_slot_overall_len, params.mount_hole_d)

            with Locations(*holes):
                Circle((params.hole_d + params.hole_oversize) / 2)

            led_centers = params.button_led_centers()
            if led_centers:
                with Locations(*led_centers):
                    Circle(params.button_led_d / 2)

        extrude(amount=cut_h, mode=Mode.SUBTRACT)

    return p.part


def build_labels(params: FaceplateParams) -> object | None:
    """Return a separate solid containing raised labels/branding."""

    height = max(params.label_height, params.brand_height)
    if height <= 0:
        return None

    with BuildPart() as p:
        def add_text_extrusion(txt: str, *, x: float, y: float, rot: float, font_size: float) -> None:
            dx, dy = _text_local_offset(txt, font=params.label_font, style=params.label_font_style, font_size=font_size)
            with BuildSketch(Plane.XY.offset(params.thickness)) as sk:
                with Locations(Location((x, y, 0), (0, 0, rot))):
                    with Locations((dx, dy)):
                        Text(txt, font_size=font_size, font=params.label_font, font_style=params.label_font_style)
            extrude(to_extrude=sk.sketch, amount=height, mode=Mode.ADD)

        # Branding — two horizontal lines stacked in the bottom gap.
        if params.brand_text_module.strip():
            bx, by = params.brand_module_pos()
            add_text_extrusion(params.brand_text_module, x=bx, y=by, rot=0.0, font_size=params.brand_size)
        if params.brand_text_maker.strip():
            bx, by = params.brand_maker_pos()
            add_text_extrusion(params.brand_text_maker, x=bx, y=by, rot=0.0, font_size=params.brand_size)

        holes_all = params.holes_sorted()
        dx_off, dy_off = params.label_offset

        disabled = set(int(i) for i in params.remove_hole_indices)
        for idx, (hx, hy) in enumerate(holes_all):
            if idx in disabled:
                continue
            if idx < len(params.hole_labels_above):
                txt = params.hole_labels_above[idx].strip()
                if txt:
                    add_text_extrusion(txt, x=hx + dx_off, y=hy - dy_off, rot=0.0, font_size=params.label_size)

    return p.part


def build_faceplate(params: FaceplateParams, export_mode: ExportMode = "combined") -> tuple[object | None, object | None]:
    base = build_base(params) if export_mode in ("combined", "base") else None
    labels = build_labels(params) if export_mode in ("combined", "labels") else None
    return base, labels


def export_print_template(params: FaceplateParams, svg: Path | None = None, dxf: Path | None = None) -> None:
    """Export a 1:1 outline/cutouts template for paper printing."""

    holes = params.holes_enabled_sorted()

    with BuildSketch(Plane.XY) as outline:
        Rectangle(params.panel_w, params.panel_h, align=(Align.MIN, Align.MIN))

    with BuildSketch(Plane.XY) as cutouts:
        if params.add_mount_slots:
            with Locations(*params.eurorack_mount_slot_centers()):
                SlotOverall(params.mount_slot_overall_len, params.mount_hole_d)
        with Locations(*holes):
            Circle(params.hole_d / 2)

        led_centers = params.button_led_centers()
        if led_centers:
            with Locations(*led_centers):
                Circle(params.button_led_d / 2)

    if svg is not None:
        exp = ExportSVG(margin=5, line_weight=0.18)
        exp.add_layer("outline")
        exp.add_layer("cutouts")
        exp.add_shape(outline.sketch, layer="outline")
        exp.add_shape(cutouts.sketch, layer="cutouts")
        exp.write(svg)

    if dxf is not None:
        exp = ExportDXF()
        exp.add_layer("outline")
        exp.add_layer("cutouts")
        exp.add_shape(outline.sketch, layer="outline")
        exp.add_shape(cutouts.sketch, layer="cutouts")
        exp.write(dxf)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build ClkLinkRec 4HP 1x6 Eurorack panel")
    parser.add_argument("--export-mode", choices=("combined", "base", "labels"), default="combined")
    parser.add_argument("--stl", type=Path, default=None, help="Export STL to this path")
    parser.add_argument("--stl-base", type=Path, default=None, help="Export base STL to this path")
    parser.add_argument("--stl-labels", type=Path, default=None, help="Export labels STL to this path")
    parser.add_argument("--step", type=Path, default=None, help="Export STEP to this path")
    parser.add_argument("--template-svg", type=Path, default=None, help="Export a 1:1 SVG print template")
    parser.add_argument("--template-dxf", type=Path, default=None, help="Export a 1:1 DXF print template")
    args = parser.parse_args()

    params = FaceplateParams()

    if args.template_svg is not None or args.template_dxf is not None:
        export_print_template(params, svg=args.template_svg, dxf=args.template_dxf)

    base, labels = build_faceplate(params, export_mode=args.export_mode)

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
                base,
                labels,
                names=["base", "labels"],
                colors=[params.base_color, params.label_color],
                reset_camera=Camera.RESET,
                grid=True,
                port=_ocp_port(),
            )
        elif args.export_mode == "base" and base is not None:
            show(
                base,
                names=["base"],
                colors=[params.base_color],
                reset_camera=Camera.RESET,
                grid=True,
                port=_ocp_port(),
            )
        elif args.export_mode == "labels" and labels is not None:
            show(
                labels,
                names=["labels"],
                colors=[params.label_color],
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
