"""Generate the TeensyMove OLED default screen (128x32) as an emissive texture.

Reproduces the default CV-MODE screen (page 0) from
eurorack_modules/src/teensy-move/main.cpp: four 8px rows reporting CV / gate /
drum / MIDI activity, idle defaults. Drawn with the shared 5x7 font.

  CV MODE  G1:- G2:-     (row 0, y=0)
  P1:+0.00V  P2:+0.00V   (row 1, y=8)
  Drums:---- CLK:-       (row 2, y=16)
  ch1-4:CV ch10:Drum     (row 3, y=24)

Output: render/out/teensy_move/teensy_move_screen.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oled_font import draw_string, load_font  # noqa: E402

W, H = 128, 32
SCALE = 6
ON = (225, 238, 255)
OFF = (0, 0, 0)

ROWS = (
    "CV MODE  G1:- G2:-",
    "P1:+0.00V  P2:+0.00V",
    "Drums:---- CLK:-",
    "ch1-4:CV ch10:Drum",
)


def main() -> None:
    font = load_font()
    img = Image.new("1", (W, H), 0)
    px = img.load()

    for i, row in enumerate(ROWS):
        draw_string(px, 0, i * 8, row, font)

    big = img.resize((W * SCALE, H * SCALE), Image.NEAREST)
    rgb = Image.new("RGB", big.size, OFF)
    mask = big.convert("L").point(lambda v: 255 if v > 127 else 0)
    rgb.paste(Image.new("RGB", big.size, ON), (0, 0), mask)

    out_dir = Path(__file__).resolve().parents[1] / "out" / "teensy_move"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "teensy_move_screen.png"
    rgb.save(out)
    print(f"wrote {out}  ({rgb.size[0]}x{rgb.size[1]})")


if __name__ == "__main__":
    main()
