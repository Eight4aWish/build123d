"""Warps overlay for Boy: a card that slips over the panel when Boy runs mode 2.

Boy's panel is labelled for mode 1, the Move companion. This card sits on it and says
what the jacks do in Warps, with no tools and no change to the panel. The jacks are
as eurorack_modules docs/TEENSY_MOVE.md has them.

It keeps the panel's coordinates (teensy_move_reva.py), so every hole lines up by
construction, and it clears what stands proud of the panel:
- the jack nuts, whatever their kind: holes of 10 mm, against 8 mm round nuts or an 8 mm
  hex nut's 9.2 mm corners, with room for FDM printing holes small;
- the screen and the encoder's knob: the card stops below them, covering rows 2-6, with
  notches for the screen's two lower screw heads. A window round the screen would leave
  a thin frame, and a loose strip where its screw clearance meets the card's edge;
- the rack screws: the card stops short of them, so it goes on with the module in the
  rack, and leaves the panel's own branding showing.

Printing: as the panels, one height-based filament change at the top of the card, Z = 1.2 mm,
so the labels print white.

Run (exports, checks and writes a 1:1 PDF to print on paper first):
  ./.venv/bin/python panels/teensy_move_reva_warps_overlay.py

Copy it for another mode: change `hole_labels_below` and `brand_text_top`.
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
	Locations,
	Mode,
	Plane,
	Pos,
	Rectangle,
	extrude,
	export_stl,
)

from teensy_move import build_labels
from teensy_move_reva import FaceplateParams as BoyRevAParams


@dataclass(frozen=True)
class OverlayParams(BoyRevAParams):
	# The card. Thin, because the nuts stand about 2.6 mm proud and should stand clear of it.
	thickness: float = 1.2
	# Inside the panel's width, so it can't foul the next module.
	card_w: float = 49.9
	# Rack screws are at y = 3 mm from the end, with heads of about 6 mm: the card ends 1 mm clear.
	card_y0: float = 7.0
	# Above the panel's MENU label, and below the knob's reach for any knob up to 16 mm across
	card_y1: float = 97.0

	add_mount_slots: bool = False

	# The top row stays uncovered: the screen and the encoder
	remove_hole_indices: tuple[int, ...] = (0, 1, 2)
	encoder_hole_indices: tuple[int, ...] = ()

	# Clearance over the jack nuts, and over the screen's screw heads
	hole_d: float = 10.0
	screw_head_clear_d: float = 6.5

	# Halfway between a hole's edge and the next hole's, as the bigger holes leave less room
	label_offset: tuple[float, float] = (0.0, -8.6)

	# In the order of mode 2's jacks: docs/TEENSY_MOVE.md
	hole_labels_below: tuple[str, ...] = (
		# Row 1 (top):  col1      col2        col3
		"",             "",         "",
		"ALGO",         "LEV1 V/O", "",
		"TIMBRE",       "LEV2",     "",
		"",             "",         "",
		"CARRIER",      "",         "OUT",
		"MOD",          "",         "AUX",
	)

	# The mode's name, in the empty band below the unused fourth row
	brand_text_top: str = "WARPS"
	brand_text_bottom: str = ""
	brand_size: float = 6.0

	def brand_top_pos(self) -> tuple[float, float]:
		row4 = self.holes_sorted()[9]
		return (self.panel_w / 2, row4[1] + self.label_offset[1])

	def screw_notches(self) -> list[tuple[float, float, float]]:
		"""(x, y, width) of each notch's bottom-left: from below the screw head to past the card's top."""
		_, mount_pts = self.screen_geometry()
		r = self.screw_head_clear_d / 2
		lower = sorted(pt for pt in mount_pts if pt[1] < mount_pts[0][1] + 1)
		out = []
		for x, y in lower:
			# the left notch runs off the card's edge, so it leaves no sliver there
			x_lo = 0.0 if x - r < 1.0 else x - r
			out.append((x_lo, y - r, x + r - x_lo))
		return out


def build_card(p: OverlayParams) -> object:
	x0 = (p.panel_w - p.card_w) / 2
	with BuildPart() as part:
		with Locations((x0, p.card_y0, 0)):
			Box(p.card_w, p.card_y1 - p.card_y0, p.thickness, align=(Align.MIN, Align.MIN, Align.MIN))
		with BuildSketch(Plane.XY.offset(-0.2)):
			for d, pts in p.holes_by_diameter().items():
				with Locations(*pts):
					Circle((d + p.hole_oversize) / 2)
			for x, y, w in p.screw_notches():
				with Locations((x, y)):
					Rectangle(w, p.card_y1 - y + 1.0, align=(Align.MIN, Align.MIN))
		extrude(amount=p.thickness + 0.4, mode=Mode.SUBTRACT)
	return part.part


def check(p: OverlayParams, card: object, labels: object) -> list[str]:
	"""What would spoil a print or a fit: lettering over a hole or off the card, thin walls, more than one piece."""
	x0 = (p.panel_w - p.card_w) / 2
	errs: list[str] = []
	with BuildPart() as whole:
		with Locations((x0, p.card_y0, 0)):
			Box(p.card_w, p.card_y1 - p.card_y0, p.thickness, align=(Align.MIN, Align.MIN, Align.MIN))
	flat = Pos(0, 0, -p.thickness) * labels  # the lettering, moved down into the card
	over_hole = flat & (whole.part - card)
	if over_hole is not None and getattr(over_hole, "volume", 0) > 1e-6:
		errs.append(f"lettering over a hole or notch: {over_hole.volume:.3f} mm3")
	off_card = flat - whole.part
	if off_card is not None and getattr(off_card, "volume", 0) > 1e-6:
		errs.append(f"lettering off the card: {off_card.volume:.3f} mm3")
	if len(card.solids()) != 1:
		errs.append(f"the card is {len(card.solids())} pieces")

	# The thinnest wall: between holes, a hole and the card's edge, a hole and a notch
	x1 = x0 + p.card_w
	holes = [(d / 2, pt) for d, pts in p.holes_by_diameter().items() for pt in pts]
	thinnest = 1e9
	for i, (r, (x, y)) in enumerate(holes):
		thinnest = min(thinnest, x - r - x0, x1 - x - r, y - r - p.card_y0, p.card_y1 - y - r)
		for r2, (x2, y2) in holes[i + 1:]:
			thinnest = min(thinnest, ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5 - r - r2)
		for nx, ny, w in p.screw_notches():
			dx = max(nx - x, 0.0, x - (nx + w))
			dy = max(ny - y, 0.0)
			thinnest = min(thinnest, (dx * dx + dy * dy) ** 0.5 - r)
	print(f"thinnest wall: {thinnest:.2f} mm")
	if thinnest < 1.2:
		errs.append(f"a wall of {thinnest:.2f} mm: too thin to print well")
	return errs


def export_pdf(p: OverlayParams, path: Path, preview: Path | None = None) -> None:
	"""The card at 1:1 on A4, over the panel's outline, with a scale bar to check the print's scale."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.patches import Circle as Disc, Rectangle as Rect

	base = BoyRevAParams()
	fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
	ax = fig.add_axes((0, 0, 1, 1))
	ax.set_xlim(-80, 130)
	ax.set_ylim(-110, 187)
	ax.set_axis_off()
	ax.add_patch(Rect((0, 0), p.panel_w, p.panel_h, fill=False, ec="0.65", lw=0.6, ls="--"))
	x0 = (p.panel_w - p.card_w) / 2
	ax.add_patch(Rect((x0, p.card_y0), p.card_w, p.card_y1 - p.card_y0, fill=False, ec="k", lw=0.8))
	# the panel's top row, for reference, and its hole under each of the card's
	for i, (x, y) in enumerate(base.holes_sorted()):
		if i >= 3:
			ax.add_patch(Disc((x, y), base.hole_d / 2, fill=False, ec="0.65", lw=0.4))
	(sx, sy), mount_pts = base.screen_geometry()
	ax.add_patch(Rect((sx - base.screen_w / 2, sy - base.screen_h / 2), base.screen_w, base.screen_h, fill=False, ec="0.65", lw=0.5, ls="--"))
	for x, y in mount_pts:
		ax.add_patch(Disc((x, y), base.screen_mount_hole_d / 2, fill=False, ec="0.65", lw=0.4))
	ex, ey = base.holes_sorted()[2]
	ax.add_patch(Disc((ex, ey), base.encoder_hole_d / 2, fill=False, ec="0.65", lw=0.4))
	for d, pts in p.holes_by_diameter().items():
		for x, y in pts:
			ax.add_patch(Disc((x, y), d / 2, fill=False, ec="k", lw=0.8))
	for x, y, w in p.screw_notches():
		ax.add_patch(Rect((x, y), w, p.card_y1 - y, fc="w", ec="k", lw=0.8))
	pts_of = lambda mm: mm * 72 / 25.4 * 0.72  # about a cap height in mm, as points
	holes = p.holes_sorted()
	for i, txt in enumerate(p.hole_labels_below):
		if txt and i not in p.remove_hole_indices:
			x, y = holes[i]
			ax.text(x, y + p.label_offset[1], txt, ha="center", va="center", fontsize=pts_of(p.label_size), fontweight="bold", family="Arial")
	tx, ty = p.brand_top_pos()
	ax.text(tx, ty, p.brand_text_top, ha="center", va="center", fontsize=pts_of(p.brand_size), fontweight="bold", family="Arial")
	# scale bar
	ax.plot([0, 50], [-12, -12], color="k", lw=1)
	for x in (0, 50):
		ax.plot([x, x], [-14, -10], color="k", lw=1)
	ax.text(25, -17, "50 mm: print at 100 %, no fit-to-page", ha="center", fontsize=9, family="Arial")
	ax.text(25, -24, "Solid: the card. Grey: Boy's panel.", ha="center", fontsize=9, family="Arial")
	fig.savefig(path)
	if preview is not None:
		ax.set_xlim(-6, 56.5)
		ax.set_ylim(-28, 134)
		fig.set_size_inches(6.25, 16.2)
		fig.savefig(preview, dpi=120)
	plt.close(fig)


def main() -> None:
	p = OverlayParams()
	card = build_card(p)
	labels = build_labels(p)
	errs = check(p, card, labels)
	out = Path(__file__).resolve().parent.parent / "exports" / "panels"
	stem = "boy_warps_overlay"
	export_stl(card, out / f"{stem}_base.stl")
	export_stl(labels, out / f"{stem}_labels.stl")
	preview = Path(os.environ["OVERLAY_PREVIEW"]) if "OVERLAY_PREVIEW" in os.environ else None
	export_pdf(p, out / f"{stem}_template.pdf", preview)
	for e in errs:
		print("PROBLEM:", e)
	print(f"card {p.card_w} x {p.card_y1 - p.card_y0} x {p.thickness} mm, {'FAILS its checks' if errs else 'passes its checks'}")
	raise SystemExit(1 if errs else 0)


if __name__ == "__main__":
	main()
