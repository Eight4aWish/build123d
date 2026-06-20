"""Shared 5x7 OLED bitmap font (the standard glyph set used by both the daisy
soft-I2C driver and Adafruit GFX). Loaded from the daisy firmware's font table
so glyphs match the real displays."""

from __future__ import annotations

import re
from pathlib import Path

_FW = Path("/Users/dbaghurst/GitHub/eurorack_daisy_patch_init/common/oled_soft_i2c.cpp")


def load_font() -> dict[int, list[int]]:
    """Return {ascii: [5 column bytes]} for ASCII 32+."""
    text = _FW.read_text()
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


def draw_char(px, x: int, y: int, cols: list[int], val: int = 1) -> None:
    for c, byte in enumerate(cols):       # 5 columns
        for r in range(7):                # bit r = row r (top..bottom)
            if byte & (1 << r):
                px[x + c, y + r] = val


def draw_string(px, x: int, y: int, s: str, font, val: int = 1) -> int:
    for ch in s:
        draw_char(px, x, y, font.get(ord(ch), [0, 0, 0, 0, 0]), val)
        x += 6                            # 5px glyph + 1px gap
    return x
