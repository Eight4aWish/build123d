"""Generate the DaisyMultiFX OLED default screen as an emissive texture.

Reproduces the 64x48 SSD1306 effect view from daisy_multifx_oled
(src/main.cpp UpdateDisplay), patch 0 = REVERB, drawn with the firmware's
own 5x7 font (shared oled_font).

  REVERB            (effect name, centred, y=0)
  --------          (HLine y=9)
  Time Damp         (p1 p2, each %-5s, y=14)
  InLv Send         (p3 p4, y=24)

Output: render/out/daisy_mfx/daisy_mfx_screen.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oled_font import draw_string, load_font  # noqa: E402

W, H = 64, 48
SCALE = 10
ON = (225, 238, 255)
OFF = (0, 0, 0)


def main() -> None:
    font = load_font()
    img = Image.new("1", (W, H), 0)
    px = img.load()

    title = "REVERB"
    draw_string(px, (W - len(title) * 6) // 2, 0, title, font)

    for x in range(W):  # HLine y=9
        px[x, 9] = 1

    # Params padded to 5-char columns: p1=Time p2=Damp / p3=InLv p4=Send
    draw_string(px, 0, 14, "Time Damp", font)
    draw_string(px, 0, 24, "InLv Send", font)

    big = img.resize((W * SCALE, H * SCALE), Image.NEAREST)
    rgb = Image.new("RGB", big.size, OFF)
    mask = big.convert("L").point(lambda v: 255 if v > 127 else 0)
    rgb.paste(Image.new("RGB", big.size, ON), (0, 0), mask)

    out_dir = Path(__file__).resolve().parents[1] / "out" / "daisy_mfx"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "daisy_mfx_screen.png"
    rgb.save(out)
    print(f"wrote {out}  ({rgb.size[0]}x{rgb.size[1]})")


if __name__ == "__main__":
    main()
