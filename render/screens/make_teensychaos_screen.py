"""Generate the Teensy Chaos OLED screen (128x64) as an emissive texture.

Reproduces the running screen from eurorack_modules/src/teensy-chaos/main.cpp:
a phase-space attractor plot in the top 32 rows, then three text rows of
parameters. Here the LORENZ mode is synthesised — the plot is X vs (Z-rho),
exactly what the firmware plots (getX/getY), integrated with RK4.

  [ Lorenz butterfly, top 32px ]
  LORENZ            r:28.0     (y=34)
  s:10.00     dt:0.0020        (y=44)
  dp:0.50     L5 R4            (y=54)

Output: render/out/teensy_chaos/teensy_chaos_screen.png
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

# Lorenz parameters (firmware defaults).
SIGMA, RHO, BETA = 10.0, 28.0, 2.667
# Display normalisation (firmware: xMin/xRange, yMin/yRange for z-rho).
X_MIN, X_RANGE = -20.0, 40.0
Y_MIN, Y_RANGE = -28.0, 55.0
PLOT_W, PLOT_H = 128, 32


def lorenz_points(n: int = 900, dt: float = 0.006, warmup: int = 2500):
    x, y, z = 0.1, 0.0, 0.0

    def deriv(x, y, z):
        return (SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z)

    def rk4(x, y, z):
        k1 = deriv(x, y, z)
        k2 = deriv(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1], z + 0.5 * dt * k1[2])
        k3 = deriv(x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1], z + 0.5 * dt * k2[2])
        k4 = deriv(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
        return (
            x + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            y + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            z + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
        )

    for _ in range(warmup):
        x, y, z = rk4(x, y, z)
    pts = []
    for _ in range(n):
        x, y, z = rk4(x, y, z)
        pts.append((x, z - RHO))  # firmware getX(), getY()=z-rho
    return pts


def main() -> None:
    font = load_font()
    img = Image.new("1", (W, H), 0)
    px = img.load()

    for (gx, gy) in lorenz_points():
        sx = int((gx - X_MIN) * (PLOT_W - 1) / X_RANGE)
        sy = int((gy - Y_MIN) * (PLOT_H - 1) / Y_RANGE)
        sy = PLOT_H - 1 - sy  # firmware flips Y
        if 0 <= sx < PLOT_W and 0 <= sy < PLOT_H:
            px[sx, sy] = 1

    draw_string(px, 0, 34, "LORENZ", font)
    draw_string(px, 78, 34, "r:28.0", font)
    draw_string(px, 0, 44, "s:10.00", font)
    draw_string(px, 72, 44, "dt:0.0020", font)
    draw_string(px, 0, 54, "dp:0.50", font)
    draw_string(px, 72, 54, "L5 R4", font)

    big = img.resize((W * SCALE, H * SCALE), Image.NEAREST)
    rgb = Image.new("RGB", big.size, OFF)
    mask = big.convert("L").point(lambda v: 255 if v > 127 else 0)
    rgb.paste(Image.new("RGB", big.size, ON), (0, 0), mask)

    out_dir = Path(__file__).resolve().parents[1] / "out" / "teensy_chaos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "teensy_chaos_screen.png"
    rgb.save(out)
    print(f"wrote {out}  ({rgb.size[0]}x{rgb.size[1]})")


if __name__ == "__main__":
    main()
