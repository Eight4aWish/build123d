"""AMYboard — 10HP faceplate reconstructed from the real acrylic-panel photo.

Layout matches amyboard.com/img/amyboard_modular.jpg (the assembled module):
title "AMY" top, OLED near the top, a rotary encoder on the left, and the
jacks as a 2-column x 5-row block on the right with row labels
spdif / line / MIDI / CV1 / CV2 (left col = in, right col = out).

The real panel is white acrylic with red lettering; rendered here in the
series' dark finish (easy to flip). Consumed by render/assemble.py (LAYOUT).
Screen: render/screens/make_amyboard_screen.py (patchbank macro page).
"""

PW, PH = 50.5, 128.5  # 10HP x 3U
CX = PW / 2

# Jack block: 2 columns (in / out) x 5 rows
_COL_IN, _COL_OUT = 33.0, 43.0
_ROWS = {  # row label -> y
    "spdif": 74.0,
    "line": 61.0,
    "MIDI": 48.0,
    "CV1": 35.0,
    "CV2": 22.0,
}


def _row(label, y, nut_in, nut_out):
    # left jack carries the red row label (to its left); right jack unlabelled.
    return [
        {"kind": "jack", "x": _COL_IN, "y": y, "label": label, "nut": nut_in,
         "label_dx": -9.5, "label_dy": 0.0, "label_size": 2.0},
        {"kind": "jack", "x": _COL_OUT, "y": y, "label": "", "nut": nut_out},
    ]


LAYOUT = {
    "panel_w": PW, "panel_h": PH, "thickness": 2.0,
    "base_color": (0.028, 0.028, 0.032),
    "label_color": (0.92, 0.92, 0.93),
    "brand_top": "PatchBank",
    "brand_bottom": "Eight4aWish",
    "controls": [
        # OLED near the top (the hero)
        {"kind": "screen", "x": CX, "y": 100.0, "w": 26.0, "h": 26.0, "px": (128, 128)},
        # rotary encoder on the left
        {"kind": "encoder", "x": 11.0, "y": 55.0, "label": "ENC", "label_dy": -8.0, "label_size": 2.0},
        # jack block: rows spdif / line / MIDI / CV1 / CV2 (in | out)
        *_row("spdif", _ROWS["spdif"], "nut_black", "nut_red"),
        *_row("line", _ROWS["line"], "nut_black", "nut_red"),
        *_row("MIDI", _ROWS["MIDI"], "nut_black", "nut_red"),
        *_row("CV1", _ROWS["CV1"], "nut_silver", "nut_gold"),
        *_row("CV2", _ROWS["CV2"], "nut_silver", "nut_gold"),
    ],
    "mounts": [(4.0, 124.5), (46.5, 124.5), (4.0, 4.0), (46.5, 4.0)],
}
