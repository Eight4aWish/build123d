"""Seed Recorder — 8HP Intellijel 1U panel (the hardware trigger).

Minimal 1U front for the Seeed XIAO RP2040 recorder module: a red LED, a
slightly-larger silver momentary button cap, and a USB-C cutout. Geometry from
seed_panel.py. Consumed by render/assemble.py via the LAYOUT dict; composited
with the Retrospective Mac menu-bar app by render/screens/make_seed_recorder_viz.py.
"""

PW, PH = 40.64, 39.3  # 8HP x Intellijel 1U
CX = PW / 2

LAYOUT = {
    "panel_w": PW, "panel_h": PH, "thickness": 2.0,
    "base_color": (0.028, 0.028, 0.032),
    "label_color": (0.92, 0.92, 0.93),
    "brand_top": "Retrospective",
    "brand_bottom": "Eight4aWish",
    "brand_size": 3.0,
    "controls": [
        # red LED, top-left
        {"kind": "led", "x": 11.0, "y": 26.5, "color": "red"},
        # slightly larger silver momentary button cap, top-right
        {"kind": "button", "x": 29.5, "y": 26.5, "d": 8.0, "cap_d": 7.4, "silver": True},
        # USB-C cutout, bottom-centre
        {"kind": "usb", "x": CX, "y": 9.0, "w": 9.5, "h": 3.4},
    ],
    # 4x M3 corner mount holes (Intellijel 1U 8HP: 2.54 mm from sides, 3.0 from top/bottom)
    "mounts": [(2.54, 3.0), (PW - 2.54, 3.0), (2.54, PH - 3.0), (PW - 2.54, PH - 3.0)],
}
