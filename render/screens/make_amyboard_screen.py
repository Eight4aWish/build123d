"""Generate the AMYboard patchbank OLED screen (128x128, 4-bit grayscale).

Reproduces the "macro" page from eurorack_modules/src/amyboard_patchbank/
patchbank.py (_draw_macros): patch name, divider, four parameter rows
(name + value bar) with the selected one highlighted. Filtered patches derive
macros TONE / RES / SPACE / MOVE (pbdata._derive_macros); here we show the
"Warm Analog" pad. Grayscale levels match the firmware (WHITE/GRAY/DIM/SELBAR).

Output: render/out/amyboard/amyboard_screen.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oled_font import load_font  # noqa: E402

W, H = 128, 128
SCALE = 5
ON = (225, 238, 255)  # OLED tint at full brightness

# Firmware 4-bit levels (0..15) -> 0..255.
WHITE, GRAY, DIM, SELBAR = 255, 153, 85, 68

PATCH = "Warm Analog"
MACROS = [("TONE", 0.55), ("RES", 0.22), ("SPACE", 0.42), ("MOVE", 0.50)]
SEL = 0


def main() -> None:
    font = load_font()
    img = Image.new("L", (W, H), 0)
    px = img.load()

    def text(x, y, s, lvl):
        for ch in s:
            cols = font.get(ord(ch), [0, 0, 0, 0, 0])
            for c, byte in enumerate(cols):
                for r in range(7):
                    if byte & (1 << r):
                        if 0 <= x + c < W and 0 <= y + r < H:
                            px[x + c, y + r] = lvl
            x += 6

    def hline(x, y, w, lvl):
        for xx in range(x, min(x + w, W)):
            px[xx, y] = lvl

    def fill_rect(x, y, w, h, lvl):
        for yy in range(max(y, 0), min(y + h, H)):
            for xx in range(max(x, 0), min(x + w, W)):
                px[xx, yy] = lvl

    def rect(x, y, w, h, lvl):
        hline(x, y, w, lvl); hline(x, y + h - 1, w, lvl)
        for yy in range(y, min(y + h, H)):
            px[x, yy] = lvl
            px[min(x + w - 1, W - 1), yy] = lvl

    text(2, 2, PATCH, WHITE)
    hline(0, 12, W, DIM)
    y = 24
    for i, (nm, v) in enumerate(MACROS):
        if i == SEL:
            fill_rect(0, y - 4, W, 18, SELBAR)
        col = WHITE if i == SEL else GRAY
        text(4, y, nm, col)
        bx, by, bw, bh = 50, y - 1, 72, 10
        rect(bx, by, bw, bh, col)
        fillw = int((bw - 2) * v)
        if fillw > 0:
            fill_rect(bx + 1, by + 1, fillw, bh - 2, col)
        y += 22
    text(2, H - 10, "hold = back", DIM)

    big = img.resize((W * SCALE, H * SCALE), Image.NEAREST)
    rgb = Image.new("RGB", big.size, (0, 0, 0))
    bpx = big.load()
    opx = rgb.load()
    for j in range(big.size[1]):
        for i in range(big.size[0]):
            g = bpx[i, j] / 255.0
            opx[i, j] = (int(ON[0] * g), int(ON[1] * g), int(ON[2] * g))

    out_dir = Path(__file__).resolve().parents[1] / "out" / "amyboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "amyboard_screen.png"
    rgb.save(out)
    print(f"wrote {out}  ({rgb.size[0]}x{rgb.size[1]})")


if __name__ == "__main__":
    main()
