"""Health, the gesture module on the analogue board Rev B (ESP32-P4) — N8Synth 10HP 3x6 panel.

Named for the rhyme's "eleven for health" (David, 2026-11-03).

Hand tracking from the P4's camera, which sits on top of the rack, not on this panel.
Each hand's x, y and size come out as CV for the Befaco Oneiroi. The TLQ-SCREEN is the
display, so this panel has no screen. The layout is David's (2026-11-03); the brief is
eurorack_electronics docs/analogue_board_revb.md, section 8.

Geometry
--------
Every position is Boy's on the analogue board Rev A (teensy_move_reva.py): the same
N8Synth 10HP deck, built and working on 2026-11-01. That includes the 2 mm content lift
that keeps the bottom of the control deck clear of the rack rail; the rail slots are not
lifted. Hole sizes are that panel's too: the deck template's 6.9 mm for jacks, pots and
buttons, and 7.5 mm for the encoder, which needs more than the pots (PEC11R, M7 bushing).
A PCB holds those sizes exactly where a print comes out slightly small, so the PCB is a
little looser than the print, which only matters under the nuts.

Cells and Rev B channels (a part in top cell JPSn uses POTn or SWn)
------------------------------------------------------------------
    row 1   JPS1 encoder: A and B on GIN1 and GIN2, push on SW1, common on 3.3 V digital
            (not POT_REF)     JPS2 button, SW2     JPS3 button, SW3
    row 2   JPS4 X pot, POT4   JPS5 Y pot, POT5    JPS6 Z pot, POT6
    row 3   hand 1 X, Y, Z on two-way CV channels set as outputs (JPS7, 8, 9)
    row 4   hand 2 X, Y, Z, the same (JPS10, 11, 12); the firmware maps any feature to
            any channel, so a row need not follow the channel numbers
    row 5   JPS13 IN-L, AIN1   JPS14 RST, GIN4      JPS15 OUT-L, AOUT1
    row 6   JPS16 IN-R, AIN2   JPS17 CLK, GIN3      JPS18 OUT-R, AOUT2
The encoder takes two of the four gate inputs, RST and CLK the other two. CLK's cell
reaches the outputs side of the board, so its wire crosses to GIN3. The audio outs are
DC-coupled, so OUT-L and OUT-R can also carry CV. SW4-SW6 stay free, since the pot cells
use their POT channels: room for two footswitches on a TRS jack later (tip and ring, as
Boy's MIDI jack uses A and B), if a cell is ever given up for them.

The buttons are BTN1 and BTN2 because there is no firmware yet to say what they do.

Consumed by kicad_faceplate.py and render/assemble.py via the `LAYOUT` dict.
"""

PW, PH = 50.5, 128.5  # 10HP x 3U, as Boy's

# Deck cells, panel coordinates (origin bottom-left, +Y up), with the 2 mm lift.
_X = (10.041, 25.281, 40.521)
_Y = (105.359, 88.089, 70.819, 53.549, 36.279, 19.009)  # rows 1-6, top to bottom

_D = 6.9    # the deck template's hole: jacks, pots, buttons
_ENC = 7.5  # the encoder's M7 bushing needs more than the pots

# Written out in full: kicad_faceplate.py reads this file with ast, without running it.
LAYOUT = {
    "panel_w": PW, "panel_h": PH, "thickness": 2.0,
    "base_color": (0.028, 0.028, 0.032),
    "label_color": (0.92, 0.92, 0.93),
    "brand_top": "Health",
    "brand_bottom": "Eight4aWish",
    "controls": [
        # --- row 1: the encoder works the menu on the TLQ-SCREEN; two buttons ---
        {"kind": "encoder", "d": _ENC, "x": _X[0], "y": _Y[0], "label": "MENU", "label_dy": -10.0, "label_size": 2.6},
        {"kind": "button", "d": _D, "x": _X[1], "y": _Y[0], "label": "BTN1", "label_dy": -7.0, "label_size": 2.6},
        {"kind": "button", "d": _D, "x": _X[2], "y": _Y[0], "label": "BTN2", "label_dy": -7.0, "label_size": 2.6},
        # --- row 2: one pot per axis, for both hands ---
        {"kind": "pot", "d": _D, "x": _X[0], "y": _Y[1], "label": "X", "label_dy": -8.5, "label_size": 2.6},
        {"kind": "pot", "d": _D, "x": _X[1], "y": _Y[1], "label": "Y", "label_dy": -8.5, "label_size": 2.6},
        {"kind": "pot", "d": _D, "x": _X[2], "y": _Y[1], "label": "Z", "label_dy": -8.5, "label_size": 2.6},
        # --- rows 3-4: each hand's x, y and size (Z, the distance from the camera) ---
        {"kind": "jack", "d": _D, "x": _X[0], "y": _Y[2], "label": "X1", "label_dy": -7.0, "label_size": 2.6, "nut": "nut_silver"},
        {"kind": "jack", "d": _D, "x": _X[1], "y": _Y[2], "label": "Y1", "label_dy": -7.0, "label_size": 2.6, "nut": "nut_silver"},
        {"kind": "jack", "d": _D, "x": _X[2], "y": _Y[2], "label": "Z1", "label_dy": -7.0, "label_size": 2.6, "nut": "nut_silver"},
        {"kind": "jack", "d": _D, "x": _X[0], "y": _Y[3], "label": "X2", "label_dy": -7.0, "label_size": 2.6, "nut": "nut_silver"},
        {"kind": "jack", "d": _D, "x": _X[1], "y": _Y[3], "label": "Y2", "label_dy": -7.0, "label_size": 2.6, "nut": "nut_silver"},
        {"kind": "jack", "d": _D, "x": _X[2], "y": _Y[3], "label": "Z2", "label_dy": -7.0, "label_size": 2.6, "nut": "nut_silver"},
        # --- rows 5-6: audio either side, reset and clock in the middle ---
        {"kind": "jack", "d": _D, "x": _X[0], "y": _Y[4], "label": "IN-L", "label_dy": -7.0, "label_size": 2.6, "nut": "nut_black"},
        {"kind": "jack", "d": _D, "x": _X[1], "y": _Y[4], "label": "RST", "label_dy": -7.0, "label_size": 2.6, "nut": "nut_black"},
        {"kind": "jack", "d": _D, "x": _X[2], "y": _Y[4], "label": "OUT-L", "label_dy": -7.0, "label_size": 2.6, "nut": "nut_red"},
        {"kind": "jack", "d": _D, "x": _X[0], "y": _Y[5], "label": "IN-R", "label_dy": -7.0, "label_size": 2.6, "nut": "nut_black"},
        {"kind": "jack", "d": _D, "x": _X[1], "y": _Y[5], "label": "CLK", "label_dy": -7.0, "label_size": 2.6, "nut": "nut_black"},
        {"kind": "jack", "d": _D, "x": _X[2], "y": _Y[5], "label": "OUT-R", "label_dy": -7.0, "label_size": 2.6, "nut": "nut_red"},
    ],
    # Boy's rail slots: 7.5 mm in from each side, 3 mm from top and bottom, not lifted.
    "mounts": [(7.5, 125.5), (43.0, 125.5), (7.5, 3.0), (43.0, 3.0)],
}
