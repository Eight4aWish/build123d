"""N8Synth 10HP 3x6 panel (derived from SVG print template).

This script was created by extracting the true panel rectangle and feature locations
from the provided KiCad/PCBNEW SVG output ("label template").

Key detail: the SVG viewBox uses units of 1/10000 inch.
- 1 SVG unit = 0.00254 mm

Geometry extracted:
- Panel outline rectangle: width ~50.448 mm, height ~128.388 mm (≈ 10HP, 3U)
- 3 columns x 6 rows of circular holes (Ø ~6.90 mm)
- 4 Eurorack mounting slots (7.0 mm x 3.2 mm) near the corners

Notes:
- Coordinates in this file follow the same convention as `n8synth_6HP.py`:
  XY origin at panel bottom-left, Z up.
- Labels/branding are scaffolded to match the existing workflow; you can update
  text strings and per-hole labels later when you decide which holes to remove
  and where the OLED cutout should go.

Run:
  ./.venv/bin/python n8synth_10HP.py

Export (optional):
  ./.venv/bin/python n8synth_10HP.py --export-mode base   --stl-base n8_10hp_base.stl
  ./.venv/bin/python n8synth_10HP.py --export-mode labels --stl-labels n8_10hp_labels.stl

Paper template (1:1):
  ./.venv/bin/python n8synth_10HP.py --template-svg n8_10hp_template.svg
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


# --- SVG extraction helpers -------------------------------------------------

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


# Hole centers from SVG ellipses (layer "Panel Outline")
HOLE_CX_U = (13805.304, 19805.305, 25805.303)
HOLE_CY_U = (16536.588, 23335.805, 30134.988, 36934.203, 43733.395, 50532.605)

# Mounting slots extracted from the SVG path bounding coordinates.
# Left top slot path20273: x in [11437.014, 14192.919], y in [10393.873, 11654.177]
# Left bottom slot path20277: x in [11437.014, 14192.919], y in [58622.985, 59883.288]
# Right top slot path1918: x in [25414.255, 28170.16],  y in [10393.873, 11654.176]
# Right bottom slot path1920: x in [25434.625, 28190.53], y in [58622.985, 59883.288]
MOUNT_SLOTS_BOUNDS_U = (
	(11437.014, 14192.919, 10393.873, 11654.177),
	(11437.014, 14192.919, 58622.985, 59883.288),
	(25414.255, 28170.16, 10393.873, 11654.176),
	(25434.625, 28190.53, 58622.985, 59883.288),
)


def _slot_centers_and_size_mm() -> tuple[list[tuple[float, float]], float, float]:
	centers: list[tuple[float, float]] = []

	# Size is consistent across all 4 slots; compute from the first.
	x0, x1, y0, y1 = MOUNT_SLOTS_BOUNDS_U[0]
	slot_len = _u2mm(x1 - x0)  # overall length
	slot_w = _u2mm(y1 - y0)  # slot width

	for x0, x1, y0, y1 in MOUNT_SLOTS_BOUNDS_U:
		cx_u = (x0 + x1) / 2
		cy_u = (y0 + y1) / 2
		centers.append(_svg_to_panel_xy(cx_u, cy_u))

	return centers, slot_len, slot_w


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
	# Panel
	# Use standard 10HP/3U by default.
	# The SVG-derived rectangle is slightly undersized (PCB/print template friendly),
	# so we center its features onto the standard panel.
	panel_w: float = 50.8
	panel_h: float = 128.5
	thickness: float = 2.0

	# Flip the module vertically (top/bottom swapped) - matches your note that the
	# designed module is "the other way up".
	flip_y: bool = True

	# Eurorack mounting (scripted, do not use SVG-derived slots)
	add_mount_slots: bool = True
	mount_slot_overall_len: float = 4.0
	mount_hole_d: float = 3.2
	# Standard-ish Eurorack slot centers for 10HP are close to the edges.
	# (The SVG-derived template had centers around x≈7.5mm and y≈3mm.)
	mount_x_from_left: float = 7.5
	mount_y_from_top: float = 3.0

	# Control holes
	hole_d: float = _u2mm(2 * 1358.627)  # from ellipse rx
	hole_oversize: float = 0.0

	# Labels (separate solid)
	label_height: float = 0.2
	label_font: str = "Arial"
	label_font_style: FontStyle = FontStyle.BOLD
	label_size: float = 3.0
	labels_y_offset: float = 0.0

	# Place labels under holes with this offset from the hole center (dx, dy)
	label_offset: tuple[float, float] = (0.0, -7.0)

	# Branding text (top and bottom)
	brand_text_top: str = "Eight4aWish"
	brand_text_bottom: str = "TeensyMove"
	brand_size: float = 3.2
	brand_height: float = 0.2
	brand_margin: float = 4.0

	# Labels placed below each hole (top-to-bottom, left-to-right ordering).
	# 18 entries for the 3×6 grid.  Use "" to skip a label.
	hole_labels_below: tuple[str, ...] = (
		# Row 1 (top):  col1   col2   col3
		"",            "",   "MENU",
		# Row 2:
		"GATE",            "ROOT",   "GATE",
		# Row 3:
		"MOD",            "CHORD",   "MOD",
		# Row 4:
		"PITCH",           "PROG",  "PITCH",
		# Row 5:
		"IN L-R",           "VOICE",  "OUT-L",
		# Row 6 (bottom):
		"CLOCK",           "RESET",  "OUT-R",
	)
	# Labels placed above each hole ("" for none)
	hole_labels_above: tuple[str, ...] = (
		"", "", "",
		"", "", "",
		"", "", "",
		"", "", "",
		"", "", "",
		"", "", "",
	)

	# Remove/disable the first N holes (sorted top-to-bottom, left-to-right)
	# to make room for the OLED.
	remove_hole_indices: tuple[int, ...] = (0, 1)

	# Which two holes define the OLED placement area (by sorted index).
	# Default matches remove_hole_indices, but kept separate so you can
	# remove holes and anchor the screen somewhere else if needed.
	screen_anchor_indices: tuple[int, int] = (0, 1)

	# Screen (OLED)
	screen_enable: bool = True
	screen_w: float = 26.5
	screen_h: float = 11.5
	# Screen mounting hole rectangle spacing (x, y)
	screen_mount_dx: float = 28.5
	screen_mount_dy: float = 16.5
	# Hole diameter for OLED mounting
	screen_mount_hole_d: float = 3.0
	# Vertical offset applied to screen + mount holes (positive = up on panel)
	screen_y_offset: float = 2.0
	# Screen horizontal relationship to mount holes:
	# left hole center is (left edge - 0.5mm), right hole center is (right edge + 1.5mm)
	screen_left_hole_center_from_edge: float = 0.5
	screen_right_hole_center_from_edge: float = 1.5
	# OLED native pixel size (128x32) so the render fits the lit area's aspect
	# inside the wider window rather than stretching it.
	screen_px: tuple[int, int] = (128, 32)

	# Render hints: column-2 controls are trimmers; MENU (idx 2) is a button.
	trimmer_hole_indices: tuple[int, ...] = (4, 7, 10, 13)
	button_hole_indices: tuple[int, ...] = (2,)

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
				holes.append(self._maybe_flip(x, y))
		return holes

	def holes_sorted(self) -> list[tuple[float, float]]:
		holes = self.holes_xy()
		holes.sort(key=lambda pt: (-pt[1], pt[0]))  # top-to-bottom, left-to-right
		return holes

	def holes_enabled_sorted(self) -> list[tuple[float, float]]:
		holes = self.holes_sorted()
		disabled = set(int(i) for i in self.remove_hole_indices)
		return [pt for idx, pt in enumerate(holes) if idx not in disabled]

	def eurorack_mount_slot_centers(self) -> list[tuple[float, float]]:
		"""Return 4 slot centers derived from top-left at (x=4, y=11 from top)."""
		xL = float(self.mount_x_from_left)
		xR = float(self.panel_w - self.mount_x_from_left)
		yT = float(self.panel_h - self.mount_y_from_top)
		yB = float(self.mount_y_from_top)
		return [(xL, yT), (xR, yT), (xL, yB), (xR, yB)]

	def screen_geometry(self) -> tuple[tuple[float, float], list[tuple[float, float]]]:
		"""Return (screen_center_xy, mount_hole_centers).

		Screen is positioned using the centers of the removed holes as the target area.
		"""

		holes = self.holes_sorted()
		if len(holes) < 2:
			raise ValueError("Expected at least 2 holes for screen placement")

		a0, a1 = self.screen_anchor_indices
		p0 = holes[int(a0)]
		p1 = holes[int(a1)]
		screen_cx = (p0[0] + p1[0]) / 2
		screen_cy = (p0[1] + p1[1]) / 2 + self.screen_y_offset

		# Derive mount center so that hole-to-screen edge offsets match the spec.
		# If left hole center is left of screen edge by 0.5 and right is right by 1.5,
		# then mount-center is 0.5mm to the right of screen center.
		mount_cx = screen_cx + (self.screen_right_hole_center_from_edge - self.screen_left_hole_center_from_edge) / 2
		mount_cy = screen_cy

		dx = self.screen_mount_dx / 2
		dy = self.screen_mount_dy / 2
		mount_pts = [
			(mount_cx - dx, mount_cy - dy),
			(mount_cx + dx, mount_cy - dy),
			(mount_cx - dx, mount_cy + dy),
			(mount_cx + dx, mount_cy + dy),
		]

		return (screen_cx, screen_cy), mount_pts

	def mount_slots_mm(self) -> tuple[list[tuple[float, float]], float, float]:
		dx, dy = self._content_offsets()
		centers, slot_len, slot_w = _slot_centers_and_size_mm()
		centers2 = [self._maybe_flip(c[0] + dx, c[1] + dy) for c in centers]
		return (centers2, slot_len, slot_w)

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
			# Eurorack mounting slots
			if params.add_mount_slots:
				with Locations(*params.eurorack_mount_slot_centers()):
					SlotOverall(params.mount_slot_overall_len, params.mount_hole_d)

			with Locations(*holes):
				Circle((params.hole_d + params.hole_oversize) / 2)

			# OLED cutout + mount holes
			if params.screen_enable:
				(sx, sy), mount_pts = params.screen_geometry()
				with Locations((sx, sy)):
					Rectangle(params.screen_w, params.screen_h, align=(Align.CENTER, Align.CENTER))
				with Locations(*mount_pts):
					Circle(params.screen_mount_hole_d / 2)

		extrude(amount=cut_h, mode=Mode.SUBTRACT)

	return p.part


def build_labels(params: FaceplateParams) -> object | None:
	"""Return a separate solid containing raised labels/branding."""

	if params.label_height <= 0 and params.brand_height <= 0:
		return None

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

		# Branding
		if params.brand_text_top.strip():
			bx, by = params.brand_top_pos()
			add_text_extrusion(params.brand_text_top, x=bx, y=by, rot=0.0, font_size=params.brand_size)
		if params.brand_text_bottom.strip():
			bx, by = params.brand_bottom_pos()
			add_text_extrusion(params.brand_text_bottom, x=bx, y=by, rot=0.0, font_size=params.brand_size)

		# Hole labels
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

	from build123d import Rectangle

	holes = params.holes_enabled_sorted()

	with BuildSketch(Plane.XY) as outline:
		Rectangle(params.panel_w, params.panel_h, align=(Align.MIN, Align.MIN))

	with BuildSketch(Plane.XY) as cutouts:
		if params.add_mount_slots:
			with Locations(*params.eurorack_mount_slot_centers()):
				SlotOverall(params.mount_slot_overall_len, params.mount_hole_d)
		with Locations(*holes):
			Circle(params.hole_d / 2)

		if params.screen_enable:
			(sx, sy), mount_pts = params.screen_geometry()
			with Locations((sx, sy)):
				Rectangle(params.screen_w, params.screen_h, align=(Align.CENTER, Align.CENTER))
			with Locations(*mount_pts):
				Circle(params.screen_mount_hole_d / 2)

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

	parser = argparse.ArgumentParser(description="Build N8Synth 10HP 3x6 Eurorack panel from SVG-derived geometry")
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