"""Ksoloti Big Genes (Elements) — approximate 20HP faceplate for renders.

This is NOT a print-accurate panel. The Big Genes is a commercial open-hardware
board (Ksoloti, GPLv3) whose exact faceplate CAD isn't published, so this is a
photo-reconstructed layout (from ksoloti.github.io/7-big_genes.html) used only
to frame the software OLED screen in renders.

Consumed by render/assemble.py via the `LAYOUT` dict (explicit control list).
Screen content: render/screens/make_ksoloti_biggenes_screen.py (Elements UI).
"""

PW, PH = 101.3, 128.5  # 20HP x 3U

# Knob rows
_KY1, _KY2 = 110.0, 90.0
_KX = (17.0, 39.0, 61.0, 83.0)

# Jack rows
_JA, _JB = 31.0, 17.0

LAYOUT = {
    "panel_w": PW, "panel_h": PH, "thickness": 2.0,
    "base_color": (0.028, 0.028, 0.032),
    "label_color": (0.92, 0.92, 0.93),
    "brand_top": "KsolotiElements",
    "brand_bottom": "Eight4aWish",
    "controls": [
        # --- 8 trimmer pots (standard trimmer style) ---
        {"kind": "trimmer", "x": _KX[0], "y": _KY1, "label": "P1", "label_dy": -6.5, "label_size": 2.0},
        {"kind": "trimmer", "x": _KX[1], "y": _KY1, "label": "P2", "label_dy": -6.5, "label_size": 2.0},
        {"kind": "trimmer", "x": _KX[2], "y": _KY1, "label": "P3", "label_dy": -6.5, "label_size": 2.0},
        {"kind": "trimmer", "x": _KX[3], "y": _KY1, "label": "P4", "label_dy": -6.5, "label_size": 2.0},
        {"kind": "trimmer", "x": _KX[0], "y": _KY2, "label": "P5", "label_dy": -6.5, "label_size": 2.0},
        {"kind": "trimmer", "x": _KX[1], "y": _KY2, "label": "P6", "label_dy": -6.5, "label_size": 2.0},
        {"kind": "trimmer", "x": _KX[2], "y": _KY2, "label": "P7", "label_dy": -6.5, "label_size": 2.0},
        {"kind": "trimmer", "x": _KX[3], "y": _KY2, "label": "P8", "label_dy": -6.5, "label_size": 2.0},
        # --- status LEDs L1/L2: green, in the gap between P3 and P4 ---
        {"kind": "led", "x": 72.0, "y": 113.0, "color": "green"},
        {"kind": "led", "x": 72.0, "y": 107.0, "color": "green"},
        # --- encoders flanking the screen ---
        {"kind": "encoder", "x": 14.5, "y": 66.0, "label": "E1-S1", "label_dy": -9.5, "label_size": 2.0},
        {"kind": "encoder", "x": 86.5, "y": 66.0, "label": "E2-S2", "label_dy": -9.5, "label_size": 2.0},
        # --- OLED screen (the hero) ---
        {"kind": "screen", "x": 50.6, "y": 66.0, "w": 38.0, "h": 20.0, "px": (128, 64)},
        # --- mid row: MIDI / SD / buttons / USB ---
        {"kind": "midi", "x": 9.0, "y": 49.0, "nut": "nut_black", "label": "MIDI", "label_dy": -5.0, "label_size": 1.9},
        {"kind": "midi", "x": 18.0, "y": 49.0, "nut": "nut_black"},
        {"kind": "sd_slot", "x": 30.5, "y": 49.0, "w": 2.6, "h": 12.0},
        {"kind": "button", "x": 41.0, "y": 49.0, "label": "S3", "label_dy": -5.5, "label_size": 1.9},
        {"kind": "led", "x": 48.0, "y": 49.0, "color": "red"},
        {"kind": "button", "x": 57.0, "y": 49.0, "label": "S4", "label_dy": -5.5, "label_size": 1.9},
        {"kind": "led", "x": 64.0, "y": 49.0, "color": "red"},
        {"kind": "usb", "x": 79.0, "y": 49.0, "w": 9.5, "h": 3.4, "label": "PROG", "label_dy": -4.5, "label_size": 1.9},
        {"kind": "usb", "x": 92.0, "y": 49.0, "w": 9.5, "h": 3.4, "label": "HOST", "label_dy": -4.5, "label_size": 1.9},
        # --- jack bank, row A (y=31) ---
        {"kind": "jack", "x": 9.0, "y": _JA, "label": "P1", "nut": "nut_silver"},
        {"kind": "jack", "x": 18.5, "y": _JA, "label": "P2", "nut": "nut_silver"},
        {"kind": "jack", "x": 28.0, "y": _JA, "label": "P3", "nut": "nut_silver"},
        {"kind": "jack", "x": 37.5, "y": _JA, "label": "P4", "nut": "nut_silver"},
        {"kind": "jack", "x": 49.0, "y": _JA, "label": "CV-X", "nut": "nut_silver"},
        {"kind": "jack", "x": 59.0, "y": _JA, "label": "GTE-1", "nut": "nut_gold"},
        {"kind": "jack", "x": 68.0, "y": _JA, "label": "CV-1", "nut": "nut_gold"},
        {"kind": "jack", "x": 82.0, "y": _JA, "label": "IN-L", "nut": "nut_black"},
        {"kind": "jack", "x": 93.0, "y": _JA, "label": "OUT-L", "nut": "nut_red"},
        # --- jack bank, row B (y=17) ---
        {"kind": "jack", "x": 9.0, "y": _JB, "label": "CV-A", "nut": "nut_silver"},
        {"kind": "jack", "x": 18.5, "y": _JB, "label": "CV-B", "nut": "nut_silver"},
        {"kind": "jack", "x": 28.0, "y": _JB, "label": "CV-C", "nut": "nut_silver"},
        {"kind": "jack", "x": 37.5, "y": _JB, "label": "CV-D", "nut": "nut_silver"},
        {"kind": "jack", "x": 49.0, "y": _JB, "label": "CV-Y", "nut": "nut_silver"},
        {"kind": "jack", "x": 59.0, "y": _JB, "label": "GTE-2", "nut": "nut_gold"},
        {"kind": "jack", "x": 68.0, "y": _JB, "label": "CV-2", "nut": "nut_gold"},
        {"kind": "jack", "x": 82.0, "y": _JB, "label": "IN-R", "nut": "nut_black"},
        {"kind": "jack", "x": 93.0, "y": _JB, "label": "OUT-R", "nut": "nut_red"},
    ],
    # M3 corner-ish mount holes (cosmetic)
    "mounts": [(4.0, 124.5), (97.3, 124.5), (4.0, 4.0), (97.3, 4.0)],
}
