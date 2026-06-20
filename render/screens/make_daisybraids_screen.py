"""Generate the DaisyBraids OLED default patch screen as an emissive texture.

Reproduces the 64x48 SSD1306 "Patch mode" screen from daisy_braids_oled
(src/main.cpp UpdateDisplay), rendered with the firmware's own 5x7 bitmap
font (common/oled_soft_i2c.cpp) so the glyphs match the real display exactly.

  ANA:CSAW          (bank 0 "ANA", patch 0 "CSAW", centred, y=0)
  ----------        (HLine y=9)
  TIMB  COLR        (y=14)
  ATK   DCY         (y=24)

Output: render/out/daisybraids/daisybraids_screen.png
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

W, H = 64, 48
SCALE = 10
ON = (225, 238, 255)
OFF = (0, 0, 0)

FW_FONT = Path("/Users/dbaghurst/GitHub/eurorack_daisy_patch_init/common/oled_soft_i2c.cpp")


def load_font() -> dict[int, list[int]]:
    """Parse the firmware font5x7 table -> {ascii: [5 column bytes]}."""
    text = FW_FONT.read_text()
    block = text[text.index("font5x7[][5]"):]
    block = block[: block.index("};")]
    font: dict[int, list[int]] = {}
    code = 32
    for line in block.splitlines():
        m = re.findall(r"0x[0-9A-Fa-f]{2}", line)
        if len(m) == 5:
            font[code] = [int(b, 16) for b in m]
            code += 1
    return font


def draw_char(px, x: int, y: int, cols: list[int]) -> None:
    for c, byte in enumerate(cols):       # 5 columns
        for r in range(7):                # 7 rows, bit r = row r
            if byte & (1 << r):
                px[x + c, y + r] = 1


def draw_string(px, x: int, y: int, s: str, font) -> int:
    for ch in s:
        cols = font.get(ord(ch), [0, 0, 0, 0, 0])
        draw_char(px, x, y, cols)
        x += 6                            # 5px glyph + 1px gap
    return x


def main() -> None:
    font = load_font()
    img = Image.new("1", (W, H), 0)
    px = img.load()

    title = "ANA:CSAW"
    tw = len(title) * 6
    draw_string(px, (W - tw) // 2, 0, title, font)

    for x in range(W):                    # HLine at y=9
        px[x, 9] = 1

    draw_string(px, 0, 14, "TIMB  COLR", font)
    draw_string(px, 0, 24, "ATK   DCY", font)

    big = img.resize((W * SCALE, H * SCALE), Image.NEAREST)
    rgb = Image.new("RGB", big.size, OFF)
    mask = big.convert("L").point(lambda v: 255 if v > 127 else 0)
    rgb.paste(Image.new("RGB", big.size, ON), (0, 0), mask)

    out_dir = Path(__file__).resolve().parents[1] / "out" / "daisybraids"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "daisybraids_screen.png"
    rgb.save(out)
    print(f"wrote {out}  ({rgb.size[0]}x{rgb.size[1]})")


if __name__ == "__main__":
    main()
