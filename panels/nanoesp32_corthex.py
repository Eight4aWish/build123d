"""CortHex — N8Synth 10HP 3x6 panel, all 18 holes, no screen.

Eurorack module that uses neural networks / LLMs to suggest patches,
emitting suggestions across six CV outputs. Name: cortex + hex (6).

Derived from `n8synth_10HP.py`. Same SVG-extracted panel rectangle and
hole-grid geometry, but with the OLED cutout disabled and every hole in
the 3×6 grid kept; seven holes (row1 col1 + all of column 2) are paired
with 3 mm LED through-holes.

Run:
  ./.venv/bin/python corthex.py

Export:
  ./.venv/bin/python corthex.py --export-mode base   --stl-base corthex_base.stl
  ./.venv/bin/python corthex.py --export-mode labels --stl-labels corthex_labels.stl

Paper template (1:1):
  ./.venv/bin/python corthex.py --template-svg corthex_template.svg
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


# --- SVG extraction helpers (from N8Synth 10HP print template) -------------

SVG_UNIT_TO_MM = 0.00254  # 1/10000 inch -> mm

# Panel rectangle in SVG units (rect20228)
RECT_X_U = 9862.2051
RECT_Y_U = 9862.2031
RECT_W_U = 19861.91
RECT_H_U = 50546.559


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


# Hole centers (3 columns × 6 rows) from the SVG ellipses.
HOLE_CX_U = (13805.304, 19805.305, 25805.303)
HOLE_CY_U = (16536.588, 23335.805, 30134.988, 36934.203, 43733.395, 50532.605)


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
    # Panel — 10HP/3U. Doepfer cuts 10 HP to 50.5 mm (nominal pitch is
    # 10 x 5.08 = 50.8, minus ~0.3 mm clearance so modules don't bind in a row).
    panel_w: float = 50.5
    panel_h: float = 128.5
    thickness: float = 2.0

    # Match Boy (teensy_move.py) module orientation.
    flip_y: bool = True

    # Shift all content (holes + labels) up by this much so the bottom row
    # clears the rack rail. Mounting slots are NOT affected.
    content_y_offset: float = 2.0

    # Eurorack mounting slots.
    add_mount_slots: bool = True
    mount_slot_overall_len: float = 4.0
    mount_hole_d: float = 3.2
    mount_x_from_left: float = 7.5
    mount_y_from_top: float = 3.0

    # Control holes.
    hole_d: float = _u2mm(2 * 1358.627)  # from SVG ellipse rx
    hole_oversize: float = 0.0

    # Labels (separate solid).
    label_height: float = 0.2
    label_font: str = "Arial"
    label_font_style: FontStyle = FontStyle.BOLD
    label_size: float = 3.2
    labels_y_offset: float = 0.0
    label_offset: tuple[float, float] = (0.0, -7.0)

    # Branding text (top and bottom).
    brand_text_top: str = "Eight4aWish"
    brand_text_bottom: str = "CortHex"
    brand_size: float = 4.0
    brand_height: float = 0.2
    brand_margin: float = 4.0

    # Per-hole labels — 18 entries for the 3×6 grid (top-to-bottom,
    # left-to-right). Carried over from Boy (teensy_move.py); row 1 col1+col2
    # were the OLED slots, so set whatever CortHex uses there.
    hole_labels_below: tuple[str, ...] = (
        # Row 1 (top):  col1   col2   col3
        "",            "",   "CV1",
        # Row 2:
        "CLOCK",        "",   "CV2",
        # Row 3:
        "IN-L",         "",  "CV3",
        # Row 4:
        "IN-R",       "",   "CV4",
        # Row 5:
        "OUT-L",      "",  "CV5",
        # Row 6 (bottom):
        "OUT-R",       "",  "CV6",
    )
    hole_labels_above: tuple[str, ...] = (
        "", "", "",
        "", "", "",
        "", "", "",
        "", "", "",
        "", "", "",
        "", "", "",
    )

    # CortHex keeps every hole — no removals, no screen.
    remove_hole_indices: tuple[int, ...] = ()
    screen_enable: bool = False

    # Jacks that are outputs (CV1–CV6 = column 3). Used by the render
    # pipeline to colour their Befaco nuts as outputs; a bare "CVn" label
    # can't otherwise be told apart from a CV input.
    output_hole_indices: tuple[int, ...] = (2, 5, 8, 11, 14, 17)

    # Button + 3 mm LED combo holes. Each entry is the index (in
    # `holes_sorted`) of a button hole that gets a paired LED hole.
    # Defaults: row1 col1 + all of column 2.
    button_led_indices: tuple[int, ...] = (0, 1, 4, 7, 10, 13, 16)
    # Offset of the LED hole centre from the button hole centre, in
    # panel-frame coordinates: (+X right, −Y down the faceplate).
    button_led_offset: tuple[float, float] = (2.5, -6.5)
    # 3 mm LED body passes through; the slightly-wider lip sits on the
    # rear of the panel, so the LED can be positioned before soldering.
    button_led_d: float = 3.2

    # Real-life finish: dark grey panel with white lettering.
    base_color: tuple[float, float, float] = (0.028, 0.028, 0.032)
    label_color: tuple[float, float, float] = (0.92, 0.92, 0.93)

    def _content_offsets(self) -> tuple[float, float]:
        """Center SVG-derived geometry onto the standard panel size."""
        dx = (self.panel_w - SVG_PANEL_W_MM) / 2
        dy = (self.panel_h - SVG_PANEL_H_MM) / 2
        return (dx, dy)

    def _maybe_flip(self, x: float, y: float) -> tuple[float, float]:
        if self.flip_y:
            return (x, self.panel_h - y)
        return (x, y)

    def holes_xy(self) -> list[tuple[float, float]]:
        dx, dy = self._content_offsets()
        holes: list[tuple[float, float]] = []
        for cx_u in HOLE_CX_U:
            for cy_u in HOLE_CY_U:
                x0, y0 = _svg_to_panel_xy(cx_u, cy_u)
                x, y = (x0 + dx, y0 + dy + self.labels_y_offset)
                x, y = self._maybe_flip(x, y)
                # Shift content up after the flip so it always moves the holes
                # toward the top edge regardless of flip_y.
                holes.append((x, y + self.content_y_offset))
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
        """LED through-hole centres for each button+LED combo hole."""
        holes = self.holes_sorted()
        dx, dy = self.button_led_offset
        return [(holes[i][0] + dx, holes[i][1] + dy) for i in self.button_led_indices]

    def eurorack_mount_slot_centers(self) -> list[tuple[float, float]]:
        xL = float(self.mount_x_from_left)
        xR = float(self.panel_w - self.mount_x_from_left)
        yT = float(self.panel_h - self.mount_y_from_top)
        yB = float(self.mount_y_from_top)
        return [(xL, yT), (xR, yT), (xL, yB), (xR, yB)]

    def brand_bottom_pos(self) -> tuple[float, float]:
        return self._maybe_flip(self.panel_w / 2, self.brand_margin)

    def brand_top_pos(self) -> tuple[float, float]:
        return self._maybe_flip(self.panel_w / 2, self.panel_h - self.brand_margin)


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

        if params.brand_text_top.strip():
            bx, by = params.brand_top_pos()
            add_text_extrusion(params.brand_text_top, x=bx, y=by, rot=0.0, font_size=params.brand_size)
        if params.brand_text_bottom.strip():
            bx, by = params.brand_bottom_pos()
            add_text_extrusion(params.brand_text_bottom, x=bx, y=by, rot=0.0, font_size=params.brand_size)

        holes_all = params.holes_sorted()
        dx_off, dy_off = params.label_offset

        disabled = set(int(i) for i in params.remove_hole_indices)
        for idx, (hx, hy) in enumerate(holes_all):
            if idx in disabled:
                continue
            if idx < len(params.hole_labels_below):
                txt = params.hole_labels_below[idx].strip()
                if txt:
                    add_text_extrusion(txt, x=hx + dx_off, y=hy + dy_off, rot=0.0, font_size=params.label_size)
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

    parser = argparse.ArgumentParser(description="Build CortHex 10HP 3x6 Eurorack panel (no screen, all 18 holes)")
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
