"""Generate the Ksoloti Big Genes (Elements) OLED screen (128x64).

Reproduces the idle UI from eurorack_modules/src/ksoloti_elements/main.cc
(default state: resonator_model=Mod, pot state=Levels, cv_assign =
Flow/Mallet/None, cv_sel=A), drawn with the shared 5x7 font.

Six rows, each under the controls it describes. Kept in step with main.cc by hand —
if the screen layout changes there, it changes here too.

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

    draw_string(px, 0, 0, "S1:Mod", font)
    hline(px, 0, 9, 128)
    draw_string(px, 0, 11, "Geom Brgt Damp Posn", font)   # P1-P4
    draw_string(px, 0, 21, "BowL BloL StkL Spce", font)   # P5-P8, levels state
    draw_string(px, 0, 31, "Cont Play Page Asgn", font)   # E1  S3  S4  E2
    draw_string(px, 0, 42, "A:Flw  B:Mal  C:---", font)   # assignable CV only
    hline(px, 0, 50, 30)                                  # underline slot A
    # bottom row is left blank: it shows the last control touched, and the idle module
    # in a panel render is not touching anything

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
