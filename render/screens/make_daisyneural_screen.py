"""Generate Mirth's (daisy_neural) OLED RUN page as an emissive texture.

Reproduces the 64x48 SSD1306 RUN page from daisy_neural (src/main.cpp DrawRunPage, the
six rows of v1.1.0), rendered with the firmware's own 5x7 bitmap font
(common/oled_soft_i2c.cpp), playing the Orange TH100 from the starter set:

  ORANGE TH         (the capture's name, y=0)
  [=========   ]    (input peak meter: outline y=9..14, bar y=10..13)
  CPU 64(64)        (CPU average and peak, y=16)
  AMP 2/12          (capture 2 of the twelve starters in AMPS, y=24)
  M100              (knob 1, the mix, fully wet; knob 2 does nothing on an amp, y=32)
        D+0         (knob 3 does nothing on an amp; knob 4, the drive, at noon, x=34 y=40)

64% average and peak is the figure measured on the module with a capture running. On a
not-amp the patch line reads NT 1/19 and knob 2's column the steer (S0.42); on a filter
FLT 3/7, F5.0 and R6.5, and its CPU line the load and cycles (61%  7637c).

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

    for x in range(64):                    # DrawRect(0, 9, 64, 6)
        px[x, 9] = px[x, 14] = 1
    for y in range(9, 15):
        px[0, y] = px[63, y] = 1
    for x in range(1, 1 + 40):             # FillRect(1, 10, w, 4): a healthy input level
        for y in range(10, 14):
            px[x, y] = 1

    draw_string(px, 0, 16, "CPU 64(64)", font)
    draw_string(px, 0, 24, "AMP 2/12", font)
    draw_string(px, 0, 32, "M100", font)
    draw_string(px, 34, 40, "D+0", font)

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
