"""Generate the OnCLite (pico2w-oc) OLED home-menu screen (128x64).

Reproduces eurorack_ui::OledHomeMenu.draw() with the pico2w-oc item list
(src/pico2w-oc/main.cpp kHomeItems): a "Home" title and a 2-column list,
with the default selection (index 0 = "Clock") highlighted (inverted).

Output: render/out/pico2w_onclite/pico2w_onclite_screen.png
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

ITEMS = ["Clock", "Quant", "Euclid", "LFO", "Env", "Scope", "MIDI", "Turing", "Acid", "Diag"]
SELECTED = 0
TOP_MARGIN = 12  # OledHomeMenu default


def fill_rect(px, x0, y0, w, h, val=1):
    for yy in range(y0, min(y0 + h, H)):
        for xx in range(x0, min(x0 + w, W)):
            if 0 <= xx < W and 0 <= yy < H:
                px[xx, yy] = val


def main() -> None:
    font = load_font()
    img = Image.new("1", (W, H), 0)
    px = img.load()

    draw_string(px, 0, 0, "Home", font)

    cols = 2 if len(ITEMS) > 3 else 1
    rows = (len(ITEMS) + cols - 1) // cols
    cell_w = W // cols
    for i, name in enumerate(ITEMS):
        col = i // rows
        row = i % rows
        x = col * cell_w + 2
        y = TOP_MARGIN + row * 10
        if i == SELECTED:
            fill_rect(px, col * cell_w, y, cell_w, 10, 1)   # white cell
            draw_string(px, x, y + 1, name, font, val=0)     # black text
        else:
            draw_string(px, x, y + 1, name, font, val=1)

    big = img.resize((W * SCALE, H * SCALE), Image.NEAREST)
    rgb = Image.new("RGB", big.size, OFF)
    mask = big.convert("L").point(lambda v: 255 if v > 127 else 0)
    rgb.paste(Image.new("RGB", big.size, ON), (0, 0), mask)

    out_dir = Path(__file__).resolve().parents[1] / "out" / "pico2w_onclite"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "pico2w_onclite_screen.png"
    rgb.save(out)
    print(f"wrote {out}  ({rgb.size[0]}x{rgb.size[1]})")


if __name__ == "__main__":
    main()
