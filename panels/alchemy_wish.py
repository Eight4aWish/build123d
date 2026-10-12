"""Wish — the Hermetic Modular Alchemy Lab (12HP) running Wish, in the series style.

Render-only, like Secret: the module is Hermetic's, so this is not a panel to print.
The geometry is Secret's (alchemy_secret.py), which took every position from Hermetic's
own front-panel template in the Alchemy SDK; only the labels change.

The labels are Wish's (daisy_beads/src/wish.cpp): knobs P1-P6 in two columns of three,
each row's button between its knobs, B1 FREEZE, B2 SEED, B3 QUALITY (B3 is also the
shift for the second layer, which a panel label cannot show). Jacks: J1/J2 the audio
inputs, then the stock panel's CV1-CV6 taken as J3-J8, so the top row reads IN-L,
V/OCT (J3), FREEZE (J5), SIZE (J7), OUT-L (J9) and the bottom row IN-R, SEED (J4),
TIME (J6), DENSITY (J8), OUT-R (J10). Not drawn: the LED rings round each knob, which
the series style has no hardware model for.

Consumed by render/assemble.py (LAYOUT).
"""

from alchemy_secret import LAYOUT as _SECRET, PW, PH, _c, _p

_KNOB_DY = -10.0  # the series' default for knobs
_KNOB = {"indicator_deg": 90.0}  # the series' plain dark knob, as on Secret
_ROW1, _ROW2 = 132.20, 145.13  # jack rows (KiCad y)
_JX = (63.13, 74.92, 86.69, 98.42, 110.31)  # jack columns (KiCad x)

LAYOUT = {
    **_SECRET,
    "brand_top": "Wish",
    "controls": [
        # Knobs P1-P6 and buttons B1-B3, three rows: knob, button, knob.
        _c("knob", 69.16, 53.39, "TIME", label_dy=_KNOB_DY, knob_kw=_KNOB),
        _c("button", 86.64, 53.39, "FREEZE"),
        _c("knob", 104.16, 53.39, "SIZE", label_dy=_KNOB_DY, knob_kw=_KNOB),
        _c("knob", 69.16, 81.41, "SHAPE", label_dy=_KNOB_DY, knob_kw=_KNOB),
        _c("button", 86.64, 81.41, "SEED"),
        _c("knob", 104.16, 81.39, "DENSITY", label_dy=_KNOB_DY, knob_kw=_KNOB),
        _c("knob", 69.16, 109.42, "PITCH", label_dy=_KNOB_DY, knob_kw=_KNOB),
        _c("button", 86.64, 109.42, "QUALITY"),
        _c("knob", 104.16, 109.39, "DRY/WET", label_dy=_KNOB_DY, knob_kw=_KNOB),
        # USB-C: the template's slot between its two end holes (4.58 mm, 6.25 apart)
        {"kind": "usb", **dict(zip(("x", "y"), _p(86.72, 123.37))), "w": 10.83, "h": 4.58},
        # Jacks. Nut colours as the series uses them: black audio and CV in, white
        # (silver) pitch and gates in, red audio out.
        _c("jack", _JX[0], _ROW1, "IN-L", nut="nut_black"),
        _c("jack", _JX[1], _ROW1, "V/OCT", nut="nut_silver"),
        _c("jack", _JX[2], _ROW1, "FREEZE", nut="nut_silver"),
        _c("jack", _JX[3], _ROW1, "SIZE", nut="nut_black"),
        _c("jack", _JX[4], _ROW1, "OUT-L", nut="nut_red"),
        _c("jack", _JX[0], _ROW2, "IN-R", nut="nut_black"),
        _c("jack", _JX[1], _ROW2, "SEED", nut="nut_silver"),
        _c("jack", _JX[2], _ROW2, "TIME", nut="nut_black"),
        _c("jack", _JX[3], _ROW2, "DENSITY", nut="nut_black"),
        _c("jack", _JX[4], _ROW2, "OUT-R", nut="nut_red"),
    ],
}
