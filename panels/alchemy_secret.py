"""Secret — the Hermetic Modular Alchemy Lab (12HP) running Secret, in the series style.

Render-only, like AMYboard: the module is Hermetic's, so this is not a panel to print.
Every position comes from Hermetic's own front-panel template in the Alchemy SDK
(eurorack_daisy_patch_init/deps/alchemy-sdk/panel/front-panel-template.kicad_pcb), not
from a photo. Its panel rectangle runs (56.40, 32.09) to (117.36, 160.59) in KiCad
coordinates: 60.96 x 128.50 mm, 12HP x 3U, with the top mounting holes 3.0 mm down as
Eurorack specifies. Converted here to panel coordinates, x from the left edge and y up
from the bottom:  x = x_kicad - 56.40,  y = 160.59 - y_kicad.

The labels are Secret's (daisy_chaos/src/secret.cpp): knobs P1-P6 in two columns of three,
each row's button between its knobs (B2 TAME MODE beside TAME, B3 ENV between AD and SR,
swapped that way 2026-10-02). Jacks: the stock panel's CV1-CV6 taken as J3-J8, so the top
row reads IN 1, V/OCT (J3), CHAOS (J5), X (J7), OUT L (J9) and the bottom row IN 2,
GATE (J4), TAME (J6), Y (J8), OUT R (J10). J1/J2 are audio inputs Secret leaves unused,
so they are unlabelled. Not drawn: the LED rings round each knob, which the series style
has no hardware model for. The physical faceplate is a separate thing, made from
Hermetic's template itself (daisy_chaos/panel/).

Consumed by render/assemble.py (LAYOUT).
"""

PW, PH = 60.96, 128.5  # 12HP x 3U, from the template's panel rectangle

_X0, _Y1 = 56.40, 160.59  # the template's panel left edge and bottom edge (KiCad)


def _p(xk: float, yk: float) -> tuple[float, float]:
    """KiCad template coordinates -> panel coordinates (y up)."""
    return round(xk - _X0, 2), round(_Y1 - yk, 2)


def _c(kind: str, xk: float, yk: float, label: str = "", **kw) -> dict:
    x, y = _p(xk, yk)
    return {"kind": kind, "x": x, "y": y, "label": label, **kw}


_KNOB_DY = -10.0  # the series' default for knobs
_KNOB = {"indicator_deg": 90.0}  # the series' plain dark knob, as on Chaos and Joy
_ROW1, _ROW2 = 132.20, 145.13  # jack rows (KiCad y)
_JX = (63.13, 74.92, 86.69, 98.42, 110.31)  # jack columns (KiCad x)

LAYOUT = {
    "panel_w": PW, "panel_h": PH, "thickness": 2.0,
    "base_color": (0.028, 0.028, 0.032),
    "label_color": (0.92, 0.92, 0.93),
    "brand_top": "Secret",
    "brand_bottom": "Eight4aWish",
    "controls": [
        # Knobs P1-P6 and buttons B1-B3, three rows: knob, button, knob.
        _c("knob", 69.16, 53.39, "TUNE", label_dy=_KNOB_DY, knob_kw=_KNOB),
        _c("button", 86.64, 53.39, "MODEL"),
        _c("knob", 104.16, 53.39, "CHAOS", label_dy=_KNOB_DY, knob_kw=_KNOB),
        _c("knob", 69.16, 81.41, "CHAR", label_dy=_KNOB_DY, knob_kw=_KNOB),
        _c("button", 86.64, 81.41, "TAME MODE"),
        _c("knob", 104.16, 81.39, "TAME", label_dy=_KNOB_DY, knob_kw=_KNOB),
        _c("knob", 69.16, 109.42, "AD", label_dy=_KNOB_DY, knob_kw=_KNOB),
        _c("button", 86.64, 109.42, "ENV"),
        _c("knob", 104.16, 109.39, "SR", label_dy=_KNOB_DY, knob_kw=_KNOB),
        # USB-C: the template's slot between its two end holes (4.58 mm, 6.25 apart)
        {"kind": "usb", **dict(zip(("x", "y"), _p(86.72, 123.37))), "w": 10.83, "h": 4.58},
        # Jacks. Nut colours as the series uses them: white (silver) pitch and gate in,
        # black CV in, gold CV out, red audio out.
        _c("jack", _JX[0], _ROW1, "", nut="nut_black"),
        _c("jack", _JX[1], _ROW1, "V/OCT", nut="nut_silver"),
        _c("jack", _JX[2], _ROW1, "CHAOS", nut="nut_black"),
        _c("jack", _JX[3], _ROW1, "X", nut="nut_gold"),
        _c("jack", _JX[4], _ROW1, "OUT-L", nut="nut_red"),
        _c("jack", _JX[0], _ROW2, "", nut="nut_black"),
        _c("jack", _JX[1], _ROW2, "GATE", nut="nut_silver"),
        _c("jack", _JX[2], _ROW2, "TAME", nut="nut_black"),
        _c("jack", _JX[3], _ROW2, "Y", nut="nut_gold"),
        _c("jack", _JX[4], _ROW2, "OUT-R", nut="nut_red"),
    ],
    "mounts": [_p(63.49, 35.09), _p(110.27, 35.10), _p(63.49, 157.59), _p(110.28, 157.60)],
}
