"""Generate Mirth's (daisy_neural) OLED RUN page as an emissive texture.

Reproduces the 64x48 SSD1306 RUN page from daisy_neural (src/main.cpp DrawRunPage),
rendered with the firmware's own 5x7 bitmap font (common/oled_soft_i2c.cpp), playing the
Orange TH100 from the starter set:

  ORANGE TH         (the capture's name, y=0)
  [=========   ]    (input peak meter: outline y=9..15, bar y=10..14)
  CPU 64(64)        (CPU average and peak, y=18)
  T+0.5M+1.0        (knob 1 trim at noon = unity, knob 2 mix fully wet; raw 0..1, y=26)
  AMP2              (capture 2 in the AMPS bank, y=34)

64% average and peak is the figure measured on the module with a capture running. A
not-amp's page differs only in its last line (NT01 S0.42: not-amp 1, steer 0.42), and
its CPU depends on the not-amp.

Output: render/out/daisy_neural/daisy_neural_screen.png (where the Blender render reads
it from, per the module manifest).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from make_daisybraids_screen import H, OFF, ON, SCALE, W, draw_string, load_font


def main() -> None:
    font = load_font()
    img = Image.new("1", (W, H), 0)
    px = img.load()

    draw_string(px, 0, 0, "ORANGE TH", font)

    for x in range(64):                    # DrawRect(0, 9, 64, 7)
        px[x, 9] = px[x, 15] = 1
    for y in range(9, 16):
        px[0, y] = px[63, y] = 1
    for x in range(1, 1 + 40):             # FillRect(1, 10, w, 5): a healthy input level
        for y in range(10, 15):
            px[x, y] = 1

    draw_string(px, 0, 18, "CPU 64(64)", font)
    draw_string(px, 0, 26, "T+0.5M+1.0", font)
    draw_string(px, 0, 34, "AMP2", font)

    big = img.resize((W * SCALE, H * SCALE), Image.NEAREST)
    rgb = Image.new("RGB", big.size, OFF)
    mask = big.convert("L").point(lambda v: 255 if v > 127 else 0)
    rgb.paste(Image.new("RGB", big.size, ON), (0, 0), mask)

    out_dir = Path(__file__).resolve().parents[1] / "out" / "daisy_neural"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "daisy_neural_screen.png"
    rgb.save(out)
    print(f"wrote {out}  ({rgb.size[0]}x{rgb.size[1]})")


if __name__ == "__main__":
    main()
