"""Face masks for Boy: what the panel says, for the jacks that change with the app.

Boy's panel is labelled for mode 1, the Move companion. A mask goes on in front of it and says what
those jacks do in another mode, with no change to the panel. The screen, the encoder and the audio
jacks are the same in every mode, so a mask leaves them and their labels showing (David, 2026-11-05):

- a band over rows 2-4: the CV and gate jacks. The panel's MENU label sits just above the third jack in
  row 2, closer than a wall will print, so the band steps up over it and prints MENU in the same place
  and size (David: "feel free to add it to the face mask");
- a spine down the middle column, over RST and MIDI, to carry the band to the foot;
- a foot across the bottom, with the panel's two lower rack screw slots. Knurled thumbscrews in
  those two hold the mask on, and the mode's name sits between them, over the panel's own "Boy".

The upper rack screws and the panel's top branding are left alone.

It keeps the panel's coordinates (teensy_move_reva.py), so every hole lines up by construction. It
clears the jack nuts, whatever their kind: holes of 10 mm, against 8 mm round nuts or an 8 mm hex
nut's 9.2 mm corners, with room for FDM printing holes small.

Printing: as the panels, one height-based filament change at the top of the card, Z = 1.6 mm, so the
labels print white.

Run (exports both, checks them and writes a 1:1 PDF of each to print on paper first):
  ./.venv/bin/python panels/teensy_move_reva_mask.py

Another mode: a new class below, with its `hole_labels_below` and `brand_text_bottom`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from build123d import (
	Align,
	Axis,
	BuildPart,
	BuildSketch,
	Circle,
	Cylinder,
	Locations,
	Mode,
	Plane,
	Pos,
	Rectangle,
	SlotOverall,
	export_stl,
	extrude,
	fillet,
)

from teensy_move import build_labels
from teensy_move_reva import FaceplateParams as BoyRevAParams

BASE = BoyRevAParams()

# What the mask must leave showing: indices into the panel's 18 cells
SCREEN_ROW = (0, 1, 2)
AUDIO = (12, 14, 15, 17)


@dataclass(frozen=True)
class MaskParams(BoyRevAParams):
	# Stiff enough for a 15 mm spine that gets handled, and the nuts still stand about 1 mm proud of it.
	thickness: float = 1.6
	# Inside the panel's width, so it can't foul the next module.
	card_w: float = 49.9
	# The band's top: under the MENU label and the screen's lower screw heads, over row 2's holes.
	band_y1: float = 95.8
	# The band's bottom: over IN-L's and OUT-L's nuts, under row 4's labels.
	band_y0: float = 42.0
	# The foot's top: under the IN-R and OUT-R labels.
	foot_y1: float = 9.6
	foot_y0: float = 0.0
	spine_w: float = 15.0
	# The step up over the panel's MENU label: under the reach of a knob up to 16 mm across, and right of
	# the screen's lower right screw head
	menu_x0: float = 35.0
	menu_y1: float = 97.0
	corner_r: float = 1.5

	add_mount_slots: bool = False
	# An M3 thumbscrew, with room for FDM printing holes small. The panel's slot runs the same way.
	slot_len: float = 4.4
	slot_w: float = 3.4
	# What the checks allow for
	knurled_head_d: float = 11.0
	audio_nut_d: float = 10.0
	knob_d: float = 16.0
	screw_head_clear_d: float = 6.5

	remove_hole_indices: tuple[int, ...] = SCREEN_ROW + AUDIO
	encoder_hole_indices: tuple[int, ...] = ()
	hole_d: float = 10.0

	# Halfway between a hole's edge and the next hole's, as the bigger holes leave less room
	label_offset: tuple[float, float] = (0.0, -8.6)

	brand_text_top: str = ""
	brand_size: float = 4.0

	def brand_bottom_pos(self) -> tuple[float, float]:
		return (self.panel_w / 2, 4.3)

	def spine_x(self) -> float:
		return self.holes_sorted()[13][0]

	def slots(self) -> list[tuple[float, float]]:
		return self.eurorack_mount_slot_centers()[2:]


@dataclass(frozen=True)
class WarpsMask(MaskParams):
	# As eurorack_modules docs/TEENSY_MOVE.md has them
	hole_labels_below: tuple[str, ...] = (
		# Row 1 (top):  col1      col2        col3
		"",             "",         "",
		"ALGO",         "LEV1 V/O", "",
		"TIMBRE",       "LEV2",     "",
		"",             "",         "",
		"",             "",         "",
		"",             "MIDI",     "",
	)
	brand_text_bottom: str = "WARPS"


@dataclass(frozen=True)
class ChordalMask(MaskParams):
	# As eurorack_modules docs/TEENSY_MOVE.md has them. The triggers are in the order the notes enter.
	hole_labels_below: tuple[str, ...] = (
		# Row 1 (top):  col1      col2        col3
		"",             "",         "",
		"NOTE4",        "NOTE1",    "TRIG1",
		"GATE",         "NOTE2",    "TRIG3",
		"MOTION",       "NOTE3",    "TRIG2",
		"",             "TRIG4",    "",
		"",             "MIDI",     "",
	)
	brand_text_bottom: str = "CHORDAL"


MASKS = {"warps": WarpsMask(), "chordal": ChordalMask()}


def build_card(p: MaskParams) -> object:
	x0 = (p.panel_w - p.card_w) / 2
	with BuildPart() as part:
		with BuildSketch() as outline:
			with Locations((x0, p.band_y0)):
				Rectangle(p.card_w, p.band_y1 - p.band_y0, align=(Align.MIN, Align.MIN))
			with Locations((x0, p.foot_y0)):
				Rectangle(p.card_w, p.foot_y1 - p.foot_y0, align=(Align.MIN, Align.MIN))
			with Locations((p.spine_x() - p.spine_w / 2, p.foot_y0)):
				Rectangle(p.spine_w, p.band_y1 - p.foot_y0, align=(Align.MIN, Align.MIN))
			fillet(outline.vertices(), radius=p.corner_r)
			with Locations((p.menu_x0, p.band_y0 + 20.0)):
				Rectangle(x0 + p.card_w - p.menu_x0, p.menu_y1 - p.band_y0 - 20.0, align=(Align.MIN, Align.MIN))
			fillet([v for v in outline.vertices() if abs(v.Y - p.menu_y1) < 1e-6], radius=0.8)
		extrude(amount=p.thickness)
		with BuildSketch(Plane.XY.offset(-0.2)):
			for d, pts in p.holes_by_diameter().items():
				with Locations(*pts):
					Circle((d + p.hole_oversize) / 2)
			with Locations(*p.slots()):
				SlotOverall(p.slot_len, p.slot_w)
		extrude(amount=p.thickness + 0.4, mode=Mode.SUBTRACT)
	return part.part


def _volume(shape: object) -> float:
	return getattr(shape, "volume", 0.0) if shape is not None else 0.0


def _discs(pts: list[tuple[float, float]], d: float, height: float = 3.0) -> object:
	with BuildPart() as part:
		with Locations(*[(x, y, -0.5) for x, y in pts]):
			Cylinder(d / 2, height, align=(Align.CENTER, Align.CENTER, Align.MIN))
	return part.part


def build_mask_labels(p: MaskParams) -> object:
	"""The mask's lettering, with MENU as the panel has it: the same place and size."""
	menu = tuple(t if i == 2 else "" for i, t in enumerate(BASE.hole_labels_below))
	return build_labels(p) + build_labels(replace(BASE, hole_labels_below=menu, brand_text_top="", brand_text_bottom="", thickness=p.thickness))


def panel_labels_kept() -> object:
	"""The panel's own labels that a mask must leave showing, at a height inside the card."""
	kept = tuple(t if i in AUDIO else "" for i, t in enumerate(BASE.hole_labels_below))
	return build_labels(replace(BASE, hole_labels_below=kept, brand_text_top="", brand_text_bottom="", thickness=0.5))


def check(p: MaskParams, card: object, labels: object, kept: object) -> list[str]:
	"""What would spoil a print or a fit: damaged copies of these parameters each make one of them fail."""
	errs: list[str] = []
	if len(card.solids()) != 1:
		errs.append(f"the card is {len(card.solids())} pieces")

	flat = Pos(0, 0, -p.thickness) * labels  # the lettering, moved down into the card
	off = flat - card
	if _volume(off) > 1e-6:
		errs.append(f"lettering off the card or over a hole: {off.volume:.3f} mm3")

	covered = kept & card
	if _volume(covered) > 1e-6:
		errs.append(f"the card covers a label it should leave showing: {covered.volume:.3f} mm3")

	holes = BASE.holes_sorted()
	(_, _), mount_pts = BASE.screen_geometry()
	lower_screws = sorted(mount_pts, key=lambda pt: pt[1])[:2]
	stand_proud = {
		"an audio jack's nut": _discs([holes[i] for i in AUDIO], p.audio_nut_d),
		"the encoder's knob": _discs([holes[2]], p.knob_d),
		"a screw head on the screen": _discs(lower_screws, p.screw_head_clear_d),
	}
	for what, discs in stand_proud.items():
		hit = discs & card
		if _volume(hit) > 1e-6:
			errs.append(f"the card runs into {what}: {hit.volume:.3f} mm3")

	heads = _discs(p.slots(), p.knurled_head_d)
	on_text = heads & flat
	if _volume(on_text) > 1e-6:
		errs.append(f"a thumbscrew's head sits on the lettering: {on_text.volume:.3f} mm3")

	# The thinnest wall: a hole to an edge, a hole to a hole, a slot to the foot's edge
	x0 = (p.panel_w - p.card_w) / 2
	x1 = x0 + p.card_w
	round_holes = [(d / 2, pt) for d, pts in p.holes_by_diameter().items() for pt in pts]
	thinnest = 1e9
	for i, (r, (x, y)) in enumerate(round_holes):
		if y > p.band_y0:
			thinnest = min(thinnest, x - r - x0, x1 - x - r, y - r - p.band_y0, p.band_y1 - y - r)
		else:
			thinnest = min(thinnest, x - r - (p.spine_x() - p.spine_w / 2), p.spine_x() + p.spine_w / 2 - x - r)
		for r2, (x2, y2) in round_holes[i + 1 :]:
			thinnest = min(thinnest, ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5 - r - r2)
	for x, y in p.slots():
		thinnest = min(thinnest, y - p.slot_w / 2 - p.foot_y0, p.foot_y1 - y - p.slot_w / 2, x - p.slot_len / 2 - x0, x1 - x - p.slot_len / 2)
	print(f"  thinnest wall: {thinnest:.2f} mm")
	if thinnest < 1.2:
		errs.append(f"a wall of {thinnest:.2f} mm: too thin to print well")
	return errs


def export_pdf(p: MaskParams, card: object, path: Path, preview: Path | None = None) -> None:
	"""The mask at 1:1 on A4, over the panel's outline, with a scale bar to check the print's scale."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.patches import Circle as Disc, Rectangle as Rect

	fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
	ax = fig.add_axes((0, 0, 1, 1))
	ax.set_xlim(-80, 130)
	ax.set_ylim(-110, 187)
	ax.set_axis_off()
	pts_of = lambda mm: mm * 72 / 25.4 * 0.72  # about a cap height in mm, as points

	# The panel, in grey, with what the mask leaves showing
	ax.add_patch(Rect((0, 0), p.panel_w, p.panel_h, fill=False, ec="0.65", lw=0.6, ls="--"))
	holes = BASE.holes_sorted()
	for i, (x, y) in enumerate(holes):
		d = BASE.encoder_hole_d if i == 2 else BASE.hole_d
		if i not in SCREEN_ROW[:2]:
			ax.add_patch(Disc((x, y), d / 2, fill=False, ec="0.65", lw=0.4))
	(sx, sy), mount_pts = BASE.screen_geometry()
	ax.add_patch(Rect((sx - BASE.screen_w / 2, sy - BASE.screen_h / 2), BASE.screen_w, BASE.screen_h, fill=False, ec="0.65", lw=0.5, ls="--"))
	for x, y in mount_pts:
		ax.add_patch(Disc((x, y), BASE.screen_mount_hole_d / 2, fill=False, ec="0.65", lw=0.4))
	for x, y in BASE.eurorack_mount_slot_centers()[:2]:
		ax.add_patch(Disc((x, y), BASE.mount_hole_d / 2, fill=False, ec="0.65", lw=0.4))
	for i in AUDIO:
		x, y = holes[i]
		ax.text(x, y + BASE.label_offset[1], BASE.hole_labels_below[i], ha="center", va="center", fontsize=pts_of(BASE.label_size), fontweight="bold", family="Arial", color="0.55")

	# The mask, in black: every edge of its top face
	top = card.faces().sort_by(Axis.Z)[-1]
	for edge in top.edges():
		pts = edge.positions([t / 40 for t in range(41)])
		ax.plot([q.X for q in pts], [q.Y for q in pts], color="k", lw=0.8, solid_capstyle="round")
	for i, txt in enumerate(p.hole_labels_below):
		if txt and i not in p.remove_hole_indices:
			x, y = holes[i]
			ax.text(x, y + p.label_offset[1], txt, ha="center", va="center", fontsize=pts_of(p.label_size), fontweight="bold", family="Arial")
	mx, my = holes[2]
	ax.text(mx, my + BASE.encoder_label_dy, BASE.hole_labels_below[2], ha="center", va="center", fontsize=pts_of(p.label_size), fontweight="bold", family="Arial")
	tx, ty = p.brand_bottom_pos()
	ax.text(tx, ty, p.brand_text_bottom, ha="center", va="center", fontsize=pts_of(p.brand_size), fontweight="bold", family="Arial")

	# scale bar
	ax.plot([0, 50], [-12, -12], color="k", lw=1)
	for x in (0, 50):
		ax.plot([x, x], [-14, -10], color="k", lw=1)
	ax.text(25, -17, "50 mm: print at 100 %, no fit-to-page", ha="center", fontsize=9, family="Arial")
	ax.text(25, -24, "Black: the mask. Grey: Boy's panel.", ha="center", fontsize=9, family="Arial")
	fig.savefig(path)
	if preview is not None:
		ax.set_xlim(-6, 56.5)
		ax.set_ylim(-28, 134)
		fig.set_size_inches(6.25, 16.2)
		fig.savefig(preview, dpi=120)
	plt.close(fig)


def main() -> None:
	out = Path(__file__).resolve().parent.parent / "exports" / "panels"
	previews = os.environ.get("MASK_PREVIEW_DIR")
	kept = panel_labels_kept()
	failed = False
	for name, p in MASKS.items():
		print(name)
		card = build_card(p)
		labels = build_mask_labels(p)
		errs = check(p, card, labels, kept)
		stem = f"boy_{name}_mask"
		export_stl(card, out / f"{stem}_base.stl")
		export_stl(labels, out / f"{stem}_labels.stl")
		export_pdf(p, card, out / f"{stem}_template.pdf", Path(previews) / f"{stem}.png" if previews else None)
		for e in errs:
			print("  PROBLEM:", e)
		failed = failed or bool(errs)
		print(f"  {'FAILS its checks' if errs else 'passes its checks'}")
	raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
	main()
