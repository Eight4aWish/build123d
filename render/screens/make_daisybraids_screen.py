"""Generate the Joy (DaisyBraids) OLED default patch screen as an emissive texture.

Reproduces the 64x48 SSD1306 "Patch mode" screen from daisy_braids_oled
(src/main.cpp UpdateDisplay), rendered with the firmware's own 5x7 bitmap
font (common/oled_soft_i2c.cpp) so the glyphs match the real display exactly.

Joy v1.1 layout — bank name and patch name on their own centred lines, and a
PER-MODEL knob-label line (from the Braids manual fold-out) above the fixed AD:

  ANALOG            (bank 0 full name, centred, y=0)
  CSAW              (patch 0 name, centred, y=10)
  ----------        (HLine y=19)
  WIDT  POLR        (per-model knob labels for CSAW, y=23)
  ATK   DCY         (fixed internal AD envelope, y=33)

Output: render/out/daisy_braids/daisy_braids_screen.png (where the Blender
render reads it from, per the module manifest).
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

    def centered(y: int, s: str) -> None:
        draw_string(px, (W - len(s) * 6) // 2, y, s, font)

    centered(0, "ANALOG")                 # bank 0 full name
    centered(10, "CSAW")                  # patch 0 name

    for x in range(W):                    # HLine at y=19
        px[x, 19] = 1

    draw_string(px, 0, 23, "WIDT  POLR", font)   # per-model knob labels (CSAW)
    draw_string(px, 0, 33, "ATK   DCY", font)    # fixed internal AD

    big = img.resize((W * SCALE, H * SCALE), Image.NEAREST)
    rgb = Image.new("RGB", big.size, OFF)
    mask = big.convert("L").point(lambda v: 255 if v > 127 else 0)
    rgb.paste(Image.new("RGB", big.size, ON), (0, 0), mask)

    out_dir = Path(__file__).resolve().parents[1] / "out" / "daisy_braids"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "daisy_braids_screen.png"
    rgb.save(out)
    print(f"wrote {out}  ({rgb.size[0]}x{rgb.size[1]})")


if __name__ == "__main__":
    main()
