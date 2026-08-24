"""Ksoloti Big Genes (Elements) — 20HP faceplate, geometry from the real board.

Provenance
----------
This started as a photo reconstruction from ksoloti.github.io/7-big_genes.html,
because the panel CAD was not published. It is now derived from the real thing:
``ksoloti_big_genes_panel-brd.svg``, a KiCad plot of the Ksoloti Big Genes panel
PCB **v0.8** (the version string is on its own silkscreen). Every position below
is measured off that plot.

What changed when the real board turned up, for anyone comparing renders:

    pots        x 17/39/61/83   -> 16.49/39.35/62.21/85.07   (22.86 = 9 x 2.54)
    pot rows    y 90 and 110    -> 88.20 and 111.10
    encoders    (14.5, 86.5)    -> (16.49, 85.07), y 66 -> 64.12
    jack rows   y 17 and 31     -> 15.80 and 28.50
    jack x      9 .. 93         -> 5.00 .. 96.44, all on a 2.54 grid
    mid row     y 49            -> y 41.2
    screen      38.0 x 20.0     -> 31.2 x 17.2, centre (50.78, 64.40)
    SD slot     (30.5, 49)      -> (24.32, 42.50), 3.0 x 13.2
    USB         y 49, h 3.4     -> y 41.26, h 3.6  (x was nearly right)
    LEDs        2 by the pots   -> none there; 8 in the middle band
    mounts      round holes     -> 3 mm slots, 4 mm wide

The GEOMETRY is now exact. The LABELS are still our reading of what each jack
does - the real silkscreen is drawn as stroked outlines rather than text, so it
cannot be lifted mechanically. Treat the wording as informed guesswork and the
positions as measured.

Coordinates are panel coordinates: origin bottom-left, +Y up, so jacks are low
and pots are high. The plot is in KiCad coordinates (+Y down); convert with
``y_panel = 128.5 - y_kicad``.

Consumed by render/assemble.py via the `LAYOUT` dict (explicit control list).
Screen content: render/screens/make_ksoloti_biggenes_screen.py (Elements UI).
"""

PW, PH = 101.3, 128.5  # 20HP x 3U, measured off the plot's Edge.Cuts

# Pots: two rows of four, 22.86 mm apart (9 x 2.54).
_PX = (16.49, 39.35, 62.21, 85.07)
_PY1, _PY2 = 111.10, 88.20

# Jacks: two rows of nine. Spacing is 10.16 (2HP) within a group and 12.70
# between groups, which is what puts the visible gaps in the row.
_JX = (5.00, 15.16, 25.32, 35.48, 48.18, 60.88, 71.04, 83.74, 96.44)
_JA, _JB = 28.50, 15.80

_MID = 41.20   # the MIDI / SD / button / USB band
_EY = 64.12    # encoders, level with the screen

LAYOUT = {
    "panel_w": PW, "panel_h": PH, "thickness": 2.0,
    "base_color": (0.028, 0.028, 0.032),
    "label_color": (0.92, 0.92, 0.93),
    "brand_top": "Girl",
    "brand_bottom": "Eight4aWish",
    "controls": [
        # --- 8 trimmer pots, Ø8.0 ---
        {"kind": "trimmer", "d": 8.0, "x": _PX[0], "y": _PY1, "label": "GEOM", "label_dy": -6.6, "label_size": 2.6},
        {"kind": "trimmer", "d": 8.0, "x": _PX[1], "y": _PY1, "label": "BRGT", "label_dy": -6.6, "label_size": 2.6},
        {"kind": "trimmer", "d": 8.0, "x": _PX[2], "y": _PY1, "label": "DAMP", "label_dy": -6.6, "label_size": 2.6},
        {"kind": "trimmer", "d": 8.0, "x": _PX[3], "y": _PY1, "label": "POSN", "label_dy": -6.6, "label_size": 2.6},
        {"kind": "trimmer", "d": 8.0, "x": _PX[0], "y": _PY2, "labels": ["BOW", "TIMB", "TIMB"], "label_dy": -6.6, "label_size": 2.6, "alt_size": 2.0, "label_pitch": 2.8},
        {"kind": "trimmer", "d": 8.0, "x": _PX[1], "y": _PY2, "labels": ["BLOW", "FLOW", "TIMB"], "label_dy": -6.6, "label_size": 2.6, "alt_size": 2.0, "label_pitch": 2.8},
        {"kind": "trimmer", "d": 8.0, "x": _PX[2], "y": _PY2, "labels": ["STRIKE", "MALLET", "TIMB"], "label_dy": -6.6, "label_size": 2.6, "alt_size": 2.0, "label_pitch": 2.8},
        {"kind": "trimmer", "d": 8.0, "x": _PX[3], "y": _PY2, "label": "SPACE", "label_dy": -6.6, "label_size": 2.6, "alt_size": 2.0, "label_pitch": 2.8},
        # --- encoders flanking the screen, Ø8.0, on the pot x-grid ---
        {"kind": "encoder", "d": 8.0, "x": _PX[0], "y": _EY, "labels": ["CONTOUR", "push: MODEL"], "label_dy": -9.5, "label_size": 2.0, "alt_size": 1.5},
        {"kind": "encoder", "d": 8.0, "x": _PX[3], "y": _EY, "labels": ["ASSIGN", "push: CV A/B/C"], "label_dy": -9.5, "label_size": 2.0, "alt_size": 1.5},
        # --- OLED window, 31.2 x 17.2 (the hero) ---
        {"kind": "screen", "x": 50.78, "y": 64.40, "w": 31.2, "h": 17.2, "px": (128, 64)},
        # --- mid band: MIDI TRS, SD, buttons, LEDs, USB ---
        # The MIDI pair share one label. It hangs off the left edge if it is centred
        # on the first jack at x=5.50, so it sits between the two instead.
        {"kind": "midi", "d": 7.2, "x": 5.50, "y": _MID, "nut": "nut_black"},
        {"kind": "midi", "d": 7.2, "x": 15.66, "y": _MID, "nut": "nut_black",
         "label": ""},
        {"kind": "sd_slot", "x": 24.32, "y": 42.50, "w": 3.0, "h": 13.2},
        {"kind": "button", "d": 7.5, "x": 39.35, "y": 41.30, "label": "PLAY", "label_dy": -5.5, "label_size": 1.9},
        {"kind": "button", "d": 7.5, "x": 62.21, "y": 41.30, "label": "PAGE", "label_dy": -5.5, "label_size": 1.9},
        # four LEDs in a 2x2 between the buttons
        {"kind": "led", "d": 2.2, "x": 44.77, "y": 44.30, "color": "red"},
        {"kind": "led", "d": 2.2, "x": 52.27, "y": 44.30, "color": "red"},
        {"kind": "led", "d": 2.2, "x": 44.77, "y": 39.20, "color": "green"},
        {"kind": "led", "d": 2.2, "x": 52.27, "y": 39.20, "color": "green"},
        # four in a row above the USB pair, 5.33 mm pitch
        {"kind": "led", "d": 2.2, "x": 77.02, "y": 49.30, "color": "green"},
        {"kind": "led", "d": 2.2, "x": 82.35, "y": 49.30, "color": "green"},
        {"kind": "led", "d": 2.2, "x": 87.69, "y": 49.30, "color": "red"},
        {"kind": "led", "d": 2.2, "x": 93.02, "y": 49.30, "color": "red"},
        # Calibration access. Ø1.7 at (27.92, 99.70), between the pot rows - a
        # screwdriver hole for a trimmer on the board behind, not an indicator.
        # It is the one hole on the plot that appears on every layer while being
        # under 2 mm, which is what marks it out from the silkscreen dial dots.
        {"kind": "bolt", "d": 1.7, "x": 27.92, "y": 99.70, "label": ""},
        {"kind": "usb", "x": 79.10, "y": 41.26, "w": 9.5, "h": 3.6, "label": "PROG", "label_dy": -4.5, "label_size": 2.0},
        {"kind": "usb", "x": 94.10, "y": 41.26, "w": 9.5, "h": 3.6, "label": ""},
        # --- jack bank, row A ---
        {"kind": "jack", "d": 7.2, "x": _JX[0], "y": _JA, "label": "GEOM", "label_size": 2.6, "nut": "nut_silver"},
        {"kind": "jack", "d": 7.2, "x": _JX[1], "y": _JA, "label": "BRGT", "label_size": 2.6, "nut": "nut_silver"},
        {"kind": "jack", "d": 7.2, "x": _JX[2], "y": _JA, "label": "DAMP", "label_size": 2.6, "nut": "nut_silver"},
        {"kind": "jack", "d": 7.2, "x": _JX[3], "y": _JA, "label": "POSN", "label_size": 2.6, "nut": "nut_silver"},
        {"kind": "jack", "d": 7.2, "x": _JX[4], "y": _JA, "label": "V/OCT", "label_size": 2.6, "nut": "nut_silver"},
        {"kind": "jack", "d": 7.2, "x": _JX[5], "y": _JA, "nut": "nut_gold"},
        {"kind": "jack", "d": 7.2, "x": _JX[6], "y": _JA, "nut": "nut_gold"},
        {"kind": "jack", "d": 7.2, "x": _JX[7], "y": _JA, "nut": "nut_black"},
        {"kind": "jack", "d": 7.2, "x": _JX[8], "y": _JA, "label": "OUT-L", "label_size": 2.6, "nut": "nut_red"},
        # --- jack bank, row B ---
        {"kind": "jack", "d": 7.2, "x": _JX[0], "y": _JB, "label": "CV-A", "label_size": 2.6, "nut": "nut_silver"},
        {"kind": "jack", "d": 7.2, "x": _JX[1], "y": _JB, "label": "CV-B", "label_size": 2.6, "nut": "nut_silver"},
        {"kind": "jack", "d": 7.2, "x": _JX[2], "y": _JB, "label": "CV-C", "label_size": 2.6, "nut": "nut_silver"},
        {"kind": "jack", "d": 7.2, "x": _JX[3], "y": _JB, "label": "GATE", "label_size": 2.6, "nut": "nut_silver"},
        {"kind": "jack", "d": 7.2, "x": _JX[4], "y": _JB, "label": "FM", "label_size": 2.6, "nut": "nut_silver"},
        {"kind": "jack", "d": 7.2, "x": _JX[5], "y": _JB, "nut": "nut_gold"},
        {"kind": "jack", "d": 7.2, "x": _JX[6], "y": _JB, "nut": "nut_gold"},
        {"kind": "jack", "d": 7.2, "x": _JX[7], "y": _JB, "nut": "nut_black"},
        {"kind": "jack", "d": 7.2, "x": _JX[8], "y": _JB, "label": "OUT-R", "label_size": 2.6, "nut": "nut_red"},
    ],
    # 3 mm mounting slots, 4 mm wide, measured off the plot's Edge.Cuts.
    "mounts": [(7.525, 125.475), (93.885, 125.475), (7.525, 2.975), (93.885, 2.975)],
}
