"""Generate the DaisyMultiOsc OLED default screen as an emissive texture.

Reproduces the 64x48 SSD1306 "Play mode" screen from daisy_multiosc
(common/multiosc_core/host.cpp Host::DrawLegend) for the power-on default
engine FM4OP, algorithm 0 = Parallel ("PARA"). Drawn with the firmware's own
5x7 bitmap font (common/oled_soft_i2c.cpp) so the glyphs match the real display.

  FM4OP:PARA        (Name():Selection(), centred, y=0)
  ----------        (HLine y=9)
  Tune  Low         (Tune + MOD1 label, y=14)
  Mid   Hi          (MOD2 + MOD3 labels, y=24)

Output: render/out/daisy_multiosc/daisy_multiosc_screen.png
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

    title = "FM4OP:PARA"                  # active_->Name() + ":" + Selection()
    tw = len(title) * 6
    draw_string(px, (W - tw) // 2, 0, title, font)

    for x in range(W):                    # HLine at y=9
        px[x, 9] = 1

    # row1 = "%-6s%s" % ("Tune", ModLabel(0)) ; row2 = "%-6s%s" % (ModLabel(1), ModLabel(2))
    # FM4OP algo 0 (Parallel): MOD1..3 = Low / Mid / Hi
    draw_string(px, 0, 14, f"{'Tune':<6}Low", font)
    draw_string(px, 0, 24, f"{'Mid':<6}Hi", font)

    big = img.resize((W * SCALE, H * SCALE), Image.NEAREST)
    rgb = Image.new("RGB", big.size, OFF)
    mask = big.convert("L").point(lambda v: 255 if v > 127 else 0)
    rgb.paste(Image.new("RGB", big.size, ON), (0, 0), mask)

    out_dir = Path(__file__).resolve().parents[1] / "out" / "daisy_multiosc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "daisy_multiosc_screen.png"
    rgb.save(out)
    print(f"wrote {out}  ({rgb.size[0]}x{rgb.size[1]})")


if __name__ == "__main__":
    main()
