"""Face masks for Boy Rev A: one per app, Move and Warps so far.

David's design (2026-11-21): a thin sheet the full size of the panel that goes on over it, held by the panel's
own four rack screws through the same slots. It replaces every label on the panel, its holes clear the jack nuts,
the encoder knob and the screen's screws so it goes on with the knob still fitted, and it carries the app's name
where the panel says "Boy". It's only a mask: the panel behind it is the structure.

Everything is placed from Boy Rev A's panel (teensy_move_reva.FaceplateParams), so a mask can't drift from the
panel it covers. The labels use the panel's font, size and offset, so each one lands over the label it hides.
MENU moves to just above the encoder: on the panel it sits 10 mm below, crowding the CLK jack.

Printing: as the panels, face up, the letters raised `label_height` on top of a `thickness` sheet; one
filament change at Z = thickness prints them white (panels/README.md).

The screen and its four screws share one opening, a bay that runs out through the panel's left edge. The left two
screws sit 2.9 mm from that edge and about 2.5 mm from the window's corners, so a round hole over each head would
leave slivers of a few tenths of a millimetre, too thin to print or to survive.

The clearances are settings, checked on a 1:1 paper template before printing:
- the encoder: David's knobs are 10 mm wide (12.5 mm tall), under them the PEC11R's M7 nut;
- the jacks: the knurled nuts, 7.8 mm in the renderer (render/hardware.py);
- the screen's four screws: their heads, which set the bay's size.

Run:
  ./.venv/bin/python panels/teensy_move_reva_mask.py              # every mask
  ./.venv/bin/python panels/teensy_move_reva_mask.py warps        # one

Writes, for each app, exports/panels/boy_<app>_mask_base.stl and _labels.stl; all of them side by side as a 1:1
paper template, exports/panels/boy_masks_template.pdf (print at 100 %); and a preview, render/out/boy_masks/boy_masks_flat.png.
It checks every label clears every hole, the window and the panel edge, and stops if one doesn't.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from build123d import (
	Align,
	BuildSketch,
	Circle,
	Compound,
	Location,
	Locations,
	Mode,
	Plane,
	Rectangle,
	RectangleRounded,
	SlotOverall,
	Text,
	export_stl,
	extrude,
)

from teensy_move import _text_local_offset
from teensy_move_reva import FaceplateParams as PanelParams

PANEL = PanelParams()
ROOT = Path(__file__).resolve().parent.parent

# The panel's holes, sorted top to bottom and left to right (PanelParams.holes_sorted): 0 and 1 are under the
# screen, 2 is the encoder, 3-17 the fifteen jacks, three to a row.
ENCODER = 2
JACKS = tuple(range(3, 18))


@dataclass(frozen=True)
class MaskParams:
	app: str
	# below each jack, in the panel's order (PanelParams.hole_labels_below); "" for a jack the app doesn't use
	labels: tuple[str, ...]
	thickness: float = 1.0
	label_height: float = 0.2      # as the panels
	encoder_clear_d: float = 11.0  # a 10 mm knob, and the M7 nut under it
	jack_clear_d: float = 9.2      # the knurled nuts (7.8 mm) and FDM holes printing small
	screw_clear_d: float = 5.0     # the screen screws' heads: the bay reaches this far round each
	bay_corner_r: float = 1.0
	menu: str = "MENU"
	menu_gap: float = 1.0          # from the encoder hole's edge to MENU's lower edge
	min_gap: float = 0.6           # the checks: a label to an opening or the edge
	min_web: float = 1.0           # ...and the mask between two openings, or an opening and the edge


def _panel_labels(**changes: str) -> tuple[str, ...]:
	"""The panel's labels, with some jacks renamed or blanked; the encoder's MENU is placed separately."""
	labels = list(PANEL.hole_labels_below)
	labels[ENCODER] = ""
	for old, new in changes.items():
		labels[labels.index(old)] = new
	return tuple(labels)


MASKS = {
	# Move: Boy's own labels; only the name and MENU change
	"move": MaskParams(app="Move", labels=_panel_labels()),
	# Warps (META, stock Warps): the names on Boy's screen in Warps mode (eurorack_modules warps_mode.cpp), paired as
	# Warps' controls read, ALGO and TIMB over LEV1 and LEV2 (David, 2026-11-21; the firmware moved its CVs to suit).
	# IN-L is Warps' IN 1, the carrier; IN-R its IN 2, the modulator; OUT-R its AUX. MIDI plays the chord engine.
	# GATE3, PITCH3 and the gate outs (CLK, DRM1, DRM2, RST) are unused.
	"warps": MaskParams(app="Warps", labels=_panel_labels(
		GATE1="ALGO", PITCH1="TIMB", CLK="",
		GATE2="LEV1", PITCH2="LEV2", DRM1="",
		GATE3="", PITCH3="", DRM2="",
		**{"IN-L": "CARR", "RST": "", "OUT-L": "OUT", "IN-R": "MOD", "OUT-R": "AUX"},
	)),
}


# ---------------------------------------------------------------------------------------------------- geometry

def holes(m: MaskParams) -> list[tuple[str, float, float, float]]:
	"""(kind, x, y, diameter) of every round cut-out."""
	pts = PANEL.holes_sorted()
	out = [("encoder", *pts[ENCODER], m.encoder_clear_d)]
	out += [("jack", *pts[i], m.jack_clear_d) for i in JACKS]
	return out


def bay(m: MaskParams) -> tuple[float, float, float, float]:
	"""The screen bay as (x0, y0, x1, y1): round the four screws' heads, and so round the window, out through the
	left edge (x0 lies off the panel, taking the bay's left corners with it)."""
	_, screws = PANEL.screen_geometry()
	r = m.screw_clear_d / 2
	xs, ys = [x for x, _ in screws], [y for _, y in screws]
	return (-2 * m.bay_corner_r - 1, min(ys) - r, max(xs) + r, max(ys) + r)


def window() -> tuple[float, float, float, float]:
	"""The panel's screen window as (x0, y0, x1, y1), inside the bay."""
	(sx, sy), _ = PANEL.screen_geometry()
	return (sx - PANEL.screen_w / 2, sy - PANEL.screen_h / 2, sx + PANEL.screen_w / 2, sy + PANEL.screen_h / 2)


def slots() -> list[tuple[float, float]]:
	return PANEL.eurorack_mount_slot_centers()


def texts(m: MaskParams) -> list[tuple[str, float, float, float]]:
	"""(text, x, y, size) of every label, each centred on (x, y) as the panel centres them."""
	pts = PANEL.holes_sorted()
	dx, dy = PANEL.label_offset
	out = []
	for i in JACKS:
		if m.labels[i].strip():
			out.append((m.labels[i], pts[i][0] + dx, pts[i][1] + dy, PANEL.label_size))
	# MENU just above the encoder: the hole's top edge, the gap, then half the letters' height
	ex, ey = pts[ENCODER]
	h = _text_height(m.menu, PANEL.label_size)
	out.append((m.menu, ex, ey + m.encoder_clear_d / 2 + m.menu_gap + h / 2, PANEL.label_size))
	# the app's name where the panel says "Boy", Eight4aWish where it does (flip_y puts "bottom" at the top)
	out.append((m.app, *PANEL.brand_bottom_pos(), PANEL.brand_size))
	out.append((PANEL.brand_text_top, *PANEL.brand_top_pos(), PANEL.brand_size))
	return out


def _text_sketch(txt: str, x: float, y: float, size: float):
	ox, oy = _text_local_offset(txt, font=PANEL.label_font, style=PANEL.label_font_style, font_size=size)
	with BuildSketch() as sk:
		with Locations(Location((x + ox, y + oy, 0))):
			Text(txt, font_size=size, font=PANEL.label_font, font_style=PANEL.label_font_style)
	return sk.sketch


def _text_height(txt: str, size: float) -> float:
	bb = _text_sketch(txt, 0, 0, size).bounding_box()
	return bb.max.Y - bb.min.Y


def mask_sketch(m: MaskParams):
	with BuildSketch() as sk:
		Rectangle(PANEL.panel_w, PANEL.panel_h, align=(Align.MIN, Align.MIN))
		with Locations(*slots()):
			SlotOverall(PANEL.mount_slot_overall_len, PANEL.mount_hole_d, mode=Mode.SUBTRACT)
		for _, x, y, d in holes(m):
			with Locations((x, y)):
				Circle(d / 2, mode=Mode.SUBTRACT)
		x0, y0, x1, y1 = bay(m)
		with Locations(((x0 + x1) / 2, (y0 + y1) / 2)):
			RectangleRounded(x1 - x0, y1 - y0, m.bay_corner_r, mode=Mode.SUBTRACT)
	return sk.sketch


def label_sketches(m: MaskParams) -> list[tuple[str, object]]:
	return [(t, _text_sketch(t, x, y, s)) for t, x, y, s in texts(m)]


def build(m: MaskParams):
	base = extrude(mask_sketch(m), amount=m.thickness)
	labels = [extrude(Plane.XY.offset(m.thickness) * sk, amount=m.label_height) for _, sk in label_sketches(m)]
	return base, Compound(labels)


# ---------------------------------------------------------------------------------------------------- checks

def _circle_to_box(x: float, y: float, r: float, box: tuple[float, float, float, float]) -> float:
	"""Gap from a circle to a rectangle, negative where they overlap."""
	bx0, by0, bx1, by1 = box
	nx, ny = min(max(x, bx0), bx1), min(max(y, by0), by1)
	return ((nx - x) ** 2 + (ny - y) ** 2) ** 0.5 - r


def check(m: MaskParams) -> list[str]:
	"""Every label clear of every opening, the slots and the panel edge by min_gap; and no web of mask thinner than
	min_web between two openings, or between an opening and the edge."""
	problems = []
	x0w, y0w, x1w, y1w = bay(m)
	slot_boxes = [(x - PANEL.mount_slot_overall_len / 2, y - PANEL.mount_hole_d / 2,
	               x + PANEL.mount_slot_overall_len / 2, y + PANEL.mount_hole_d / 2) for x, y in slots()]
	hs = holes(m)
	for i, (kind, x, y, d) in enumerate(hs):
		r = d / 2
		edge = min(x - r, y - r, PANEL.panel_w - x - r, PANEL.panel_h - y - r)
		if edge < m.min_web:
			problems.append(f"{m.app}: the {kind} hole at ({x:.1f}, {y:.1f}) leaves {edge:.2f} mm to the edge")
		for kind2, x2, y2, d2 in hs[i + 1:]:
			web = ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5 - r - d2 / 2
			if web < m.min_web:
				problems.append(f"{m.app}: {web:.2f} mm between the {kind} and {kind2} holes at ({x:.1f}, {y:.1f})")
		for box, what in [(bay(m), "the screen bay"), *[(b, "a rack slot") for b in slot_boxes]]:
			web = _circle_to_box(x, y, r, box)
			if web < m.min_web:
				problems.append(f"{m.app}: {web:.2f} mm between the {kind} hole at ({x:.1f}, {y:.1f}) and {what}")
	for b in slot_boxes:
		if not (b[0] > x1w + m.min_web or b[2] < x0w - m.min_web or b[1] > y1w + m.min_web or b[3] < y0w - m.min_web):
			problems.append(f"{m.app}: a rack slot is within {m.min_web} mm of the screen bay")
	for txt, sk in label_sketches(m):
		bb = sk.bounding_box()
		bx0, by0, bx1, by1 = bb.min.X, bb.min.Y, bb.max.X, bb.max.Y
		if bx0 < m.min_gap or by0 < m.min_gap or bx1 > PANEL.panel_w - m.min_gap or by1 > PANEL.panel_h - m.min_gap:
			problems.append(f"{m.app}: {txt} runs off the panel")
		for kind, x, y, d in holes(m):
			nx, ny = min(max(x, bx0), bx1), min(max(y, by0), by1)  # nearest point of the label's box
			gap = ((nx - x) ** 2 + (ny - y) ** 2) ** 0.5 - d / 2
			if gap < m.min_gap:
				problems.append(f"{m.app}: {txt} is {gap:.2f} mm from a {kind} hole at ({x:.1f}, {y:.1f})")
		if bx0 < x1w + m.min_gap and bx1 > x0w - m.min_gap and by0 < y1w + m.min_gap and by1 > y0w - m.min_gap:
			problems.append(f"{m.app}: {txt} touches the screen bay")
		for x, y in slots():
			sx0, sx1 = x - PANEL.mount_slot_overall_len / 2, x + PANEL.mount_slot_overall_len / 2
			sy0, sy1 = y - PANEL.mount_hole_d / 2, y + PANEL.mount_hole_d / 2
			if bx0 < sx1 + m.min_gap and bx1 > sx0 - m.min_gap and by0 < sy1 + m.min_gap and by1 > sy0 - m.min_gap:
				problems.append(f"{m.app}: {txt} touches a rack slot")
	return problems


# ---------------------------------------------------------------------------------------------------- pictures

def _paths(sketch):
	"""A sketch's faces as matplotlib paths, holes included (the inner wires run the other way)."""
	import numpy as np
	from matplotlib.path import Path as MPath

	def ring(wire):
		# edge by edge, so a straight edge keeps its exact corners and a curve gets enough points
		pts = []
		for e in wire.order_edges():
			n = 2 if e.geom_type.name == "LINE" else 24
			pts += [e.position_at(t) for t in np.linspace(0, 1, n)][:-1]
		return [(p.X, p.Y) for p in pts]

	out = []
	for f in sketch.faces():
		verts, codes = [], []
		for w in [f.outer_wire(), *f.inner_wires()]:
			r = ring(w)
			verts += r + [r[0]]
			codes += [MPath.MOVETO] + [MPath.LINETO] * (len(r) - 1) + [MPath.CLOSEPOLY]
		out.append(MPath(verts, codes))
	return out


def draw(ax, m: MaskParams, x_off: float, *, paper: bool) -> None:
	from matplotlib.patches import Circle as Disc, PathPatch, Rectangle as Box2D
	from matplotlib.transforms import Affine2D

	shift = Affine2D().translate(x_off, 0)
	if not paper:  # Boy underneath, roughly: the panel, its screen, the jack nuts, the knob and the screen screws
		ax.add_patch(Box2D((x_off, 0), PANEL.panel_w, PANEL.panel_h, facecolor="#37373c"))
		x0, y0, x1, y1 = window()
		ax.add_patch(Box2D((x_off + x0, y0), x1 - x0, y1 - y0, facecolor="#050506"))
		for kind, x, y, _ in holes(m):
			r = 5.0 if kind == "encoder" else 3.9
			ax.add_patch(Disc((x_off + x, y), r, facecolor="#b9b9bd" if kind == "encoder" else "#8d8d92"))
			if kind == "jack":
				ax.add_patch(Disc((x_off + x, y), 1.8, facecolor="#050506"))
		for x, y in PANEL.screen_geometry()[1]:
			ax.add_patch(Disc((x_off + x, y), 2.0, facecolor="#9a9a9f"))
	for p in _paths(mask_sketch(m)):
		ax.add_patch(PathPatch(p.transformed(shift), facecolor="none" if paper else "#141416",
		                       edgecolor="black" if paper else "none", lw=0.3))
	for _, sk in label_sketches(m):
		for p in _paths(sk):
			ax.add_patch(PathPatch(p.transformed(shift), facecolor="black" if paper else "#ececee", edgecolor="none"))


def pictures(masks: list[MaskParams], template: Path, preview: Path) -> None:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	gap = 15.0
	w = len(masks) * PANEL.panel_w + (len(masks) - 1) * gap
	# 1:1 on A4 portrait: the figure is the page in inches, the axes a box of exactly w x panel_h mm
	page_w, page_h = 210.0, 297.0
	fig = plt.figure(figsize=(page_w / 25.4, page_h / 25.4))
	ax = fig.add_axes(((page_w - w) / 2 / page_w, (page_h - PANEL.panel_h) / 2 / page_h, w / page_w, PANEL.panel_h / page_h))
	for i, m in enumerate(masks):
		draw(ax, m, i * (PANEL.panel_w + gap), paper=True)
	ax.set_xlim(0, w)
	ax.set_ylim(0, PANEL.panel_h)
	ax.axis("off")
	fig.text(0.5, 0.06, "Print at 100 %: the outline is 50.5 x 128.5 mm. Cut the holes out and lay it over Boy.",
	         ha="center", fontsize=8)
	fig.savefig(template)
	plt.close(fig)

	fig = plt.figure(figsize=(w / 25.4 * 2.5, PANEL.panel_h / 25.4 * 2.5), facecolor="#8a8a90")
	ax = fig.add_axes((0, 0, 1, 1))
	for i, m in enumerate(masks):
		draw(ax, m, i * (PANEL.panel_w + gap), paper=False)
	ax.set_xlim(0, w)
	ax.set_ylim(0, PANEL.panel_h)
	ax.axis("off")
	fig.savefig(preview, dpi=150, facecolor=fig.get_facecolor())
	plt.close(fig)


def main() -> None:
	ap = argparse.ArgumentParser(description="Face masks for Boy Rev A")
	ap.add_argument("apps", nargs="*", choices=[[], *MASKS], default=[], help="the masks to make (default: all)")
	ap.add_argument("--out", type=Path, default=ROOT / "exports" / "panels")
	ap.add_argument("--preview", type=Path, default=ROOT / "render" / "out" / "boy_masks" / "boy_masks_flat.png")
	a = ap.parse_args()
	masks = [MASKS[k] for k in (a.apps or MASKS)]

	problems = [p for m in masks for p in check(m)]
	if problems:
		sys.exit("Not written:\n  " + "\n  ".join(problems))
	a.out.mkdir(parents=True, exist_ok=True)
	a.preview.parent.mkdir(parents=True, exist_ok=True)
	for m in masks:
		base, labels = build(m)
		stem = f"boy_{m.app.lower()}_mask"
		export_stl(base, a.out / f"{stem}_base.stl")
		export_stl(labels, a.out / f"{stem}_labels.stl")
		print(f"{m.app}: {a.out / stem}_base.stl and _labels.stl ({m.thickness} mm, letters {m.label_height} mm)")
	pictures(masks, a.out / "boy_masks_template.pdf", a.preview)
	print(f"template {a.out / 'boy_masks_template.pdf'}, preview {a.preview}")


if __name__ == "__main__":
	main()
