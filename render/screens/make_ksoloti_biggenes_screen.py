"""Generate the Ksoloti Big Genes (Elements) OLED screen (128x64).

Reproduces the single-page idle UI from eurorack_modules/src/ksoloti_elements/
main.cc (default state: resonator_model=Mod, pot_mode=Levels, cv_assign =
Flow/Mallet/None, cv_sel=A), drawn with the shared 5x7 font.

Output: render/out/ksoloti_biggenes/ksoloti_biggenes_screen.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oled_font import draw_string, load_font  # noqa: E402

W, H = 128, 64
SCALE = 6
ON = (225, 238, 255)
OFF = (0, 0, 0)


def hline(px, x, y, w):
    for xx in range(x, min(x + w, W)):
        px[xx, y] = 1


def main() -> None:
    font = load_font()
    img = Image.new("1", (W, H), 0)
    px = img.load()

    draw_string(px, 0, 0, "S1 Mod E1 Con SE2 Cv", font)
    hline(px, 0, 9, 128)
    draw_string(px, 0, 11, "P1-4 Geo Brt Dmp Pos", font)
    draw_string(px, 0, 21, "P5-8 Bow Blw Stk Spc", font)
    hline(px, 0, 29, 24)                      # underline P5-8 (levels active)
    draw_string(px, 0, 31, "P5-8 BlT Flw Mal StT", font)
    draw_string(px, 0, 42, "CvAD Flw Mal --- Gte", font)
    hline(px, 30, 50, 18)                     # underline active CV (A)
    draw_string(px, 0, 53, "S34 PyPge CvXY VO FM", font)

    big = img.resize((W * SCALE, H * SCALE), Image.NEAREST)
    rgb = Image.new("RGB", big.size, OFF)
    mask = big.convert("L").point(lambda v: 255 if v > 127 else 0)
    rgb.paste(Image.new("RGB", big.size, ON), (0, 0), mask)

    out_dir = Path(__file__).resolve().parents[1] / "out" / "ksoloti_biggenes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ksoloti_biggenes_screen.png"
    rgb.save(out)
    print(f"wrote {out}  ({rgb.size[0]}x{rgb.size[1]})")


if __name__ == "__main__":
    main()
