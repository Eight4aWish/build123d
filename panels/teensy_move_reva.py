"""Boy on the analogue board Rev A (Teensy 4.1) — N8Synth 10HP 3x6 panel.

The Boy panel (teensy_move.py) for the "Boy2" build: the Teensy 4.1 on the
analogue board Rev A, with a push encoder and TRS MIDI in. The cell-by-cell
deck layout is stage 8 of eurorack_electronics' docs/analogue_board_bringup.md.

Changes from Boy:
- an encoder (Bourns PEC11R, M7 bushing) in place of the top-right MENU button,
  cut at 7.5 mm: it needs at least 7.1, and FDM prints holes slightly small;
- the screen and its four mount holes 1 mm further left, so the screen module
  stays clear of the encoder;
- the Rev A labels.

Everything else comes from teensy_move.py unchanged, including the 2 mm content
lift that keeps the bottom of the control deck clear of the rack rail.

Run:
  ./.venv/bin/python panels/teensy_move_reva.py

Export:
  ./.venv/bin/python panels/teensy_move_reva.py \\
      --stl-base exports/panels/boy2_base.stl --stl-labels exports/panels/boy2_labels.stl
"""

from __future__ import annotations

from dataclasses import dataclass

from teensy_move import FaceplateParams as BoyParams
# Re-exported so render/assemble.py finds them on this module.
from teensy_move import build_base, build_faceplate, build_labels, export_print_template, main  # noqa: F401


@dataclass(frozen=True)
class FaceplateParams(BoyParams):
	# Cells JPS1–18 as the front reads; the screen sits over JPS1 and JPS2.
	hole_labels_below: tuple[str, ...] = (
		# Row 1 (top):  col1      col2      col3
		"",             "",       "MENU",
		"GATE1",        "PITCH1", "CLK",
		"GATE2",        "PITCH2", "DRM1",
		"GATE3",        "PITCH3", "DRM2",
		"IN-L",         "RST",    "OUT-L",
		"IN-R",         "MIDI",   "OUT-R",
	)

	encoder_hole_indices: tuple[int, ...] = (2,)
	encoder_hole_d: float = 7.5  # at least 7.1 for the M7 bushing

	screen_x_offset: float = -1.0

	# Render hints: no trimmers or buttons on this panel; the drum and reset
	# outs get Boy's silver control-jack nuts (the classifier doesn't know DRM/RST).
	trimmer_hole_indices: tuple[int, ...] = ()
	button_hole_indices: tuple[int, ...] = ()
	nut_overrides: tuple[tuple[str, str], ...] = (
		("DRM1", "nut_silver"),
		("DRM2", "nut_silver"),
		("RST", "nut_silver"),
	)


if __name__ == "__main__":
	main(FaceplateParams(), description="Boy on the analogue board Rev A — N8Synth 10HP 3x6 Eurorack panel")
