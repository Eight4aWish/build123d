"""Tiliqua (apf.audio) — 6HP faceplate, geometry measured from the published CAD.

Render-only, like `ksoloti_biggenes.py`. This is somebody else's product and there is
no reason to remake their panel; the model exists so Tiliqua can appear on a title card
and a module page in the same visual language as the rest of the range.

Provenance
----------
Measured from ``tiliqua-panel.kicad_pcb`` in `apfaudio/tiliqua-hardware` (the CAD is not
in the main `apfaudio/tiliqua` repo — that carries schematic PDFs only, to keep clones
small). Positions below are read from Edge.Cuts geometry and footprint placements, then
converted from KiCad coordinates (origin top-left, +Y down, panel spanning x −3.00..27.00)
to panel coordinates (origin bottom-left, +Y up)::

    panel_x = kicad_x + 3.00
    panel_y = 128.50 - kicad_y

MEASURED — taken straight from the board file:

    outline         30.00 x 128.50 mm, i.e. 6HP at 3U
    jack column     x 7.20, eight holes, y 16.61..113.11, pitch 13.78
                    (drawn in the CAD as MountingHole_6.4mm_M6 footprints — a
                    convenient 6.4 mm circle, not a mounting hole)
    USB-C x2        6.15 x 3.70, centres (20.87, 101.50) and (20.80, 92.65)
    PMOD x2         6.00 x 16.25, x 15.95..21.95, at y 48.12..64.38 and 25.25..41.50
                    (drawn +2.00 to the right of this — see RIGHT_SHIFT)
                    (a 2x6 header on 2.54 mm pitch is 15.24 x 5.08 plus clearance)
    encoder         6.4 mm at (22.80, 113.10)
    TRS MIDI        6.4 mm at (23.40, 16.43)
    mounting        four corners, (7.53, 125.50) (22.52, 125.50) (7.47, 3.00) (22.52, 3.00)

ASSUMED — not found in Edge.Cuts, so this is our reading and should be treated as
approximate:

    GPDI slot       There is an unexplained 26 mm gap between the lower USB-C and the
                    upper PMOD, and David's own walk down the panel puts the GPDI slot
                    exactly there. Placed at (18.95, 77.50), 6.0 x 15.5 — portrait, on
                    the PMOD x, which is the only way it fits a 30 mm panel. If the aperture
                    ever matters, it is in the motherboard CAD rather than the panel's.
                    Drawn as a plain rectangle: the real aperture follows the HDMI socket's
                    trapezoid with chamfered corners, which is not worth modelling for a
                    render this size. Deliberate, not an omission.
                    It is a GPDI slot, not HDMI: an HDMI-type connector carrying DVI
                    signalling, so the panel should not say HDMI.
    label placement Sizing and the right-hand wording come from David reading the module
                    in front of him: text at the bottom-right of each hole, and
                    dbg / usb2 / gpdi / ex0 / ex1 / midi down the right, encoder blank,
                    all in roughly 6 point. Exact offsets are ours — at 6HP there is no
                    room to put text beside the right-hand apertures without colliding
                    with the jack digits, so those labels sit above their aperture.

DELIBERATELY OMITTED

    LED slits       The real panel has an LED slit beneath each I/O jack. Left out on
                    purpose: they read as noise at title-card size and add nothing to a
                    render. Do not "fix" this.

    unused labels   Out 1-3 are drilled and fitted but carry no label, which is the house
                    convention: a control the software does not use stays where the panel
                    puts it and simply goes unlabelled. The blank entries in JACK_LABELS
                    are deliberate. ORBITA uses one output; LACUNA uses two.

The right-hand column, top to bottom, is: push-button encoder, debug USB-C
(programming), second USB-C (host), the GPDI slot, PMOD EX0, PMOD EX1, TRS MIDI.

Run:
  ./.venv/bin/python panels/tiliqua.py

Export:
  ./.venv/bin/python panels/tiliqua.py --export-mode base   --stl tiliqua_base.stl
  ./.venv/bin/python panels/tiliqua.py --export-mode labels --stl tiliqua_labels.stl
  ./.venv/bin/python panels/tiliqua.py --template-svg tiliqua_template.svg
"""

from __future__ import annotations

import argparse
from typing import Literal

from build123d import (
    Align,
    BuildPart,
    BuildSketch,
    Circle,
    FontStyle,
    Locations,
    Mode,
    Plane,
    Rectangle,
    SlotOverall,
    Text,
    extrude,
    export_stl,
)
from build123d.exporters import ExportSVG

ExportMode = Literal["combined", "base", "labels"]

# --- panel -----------------------------------------------------------------
HP = 5.08
W = 30.00          # 6HP, as cut (nominal 6 x 5.08 = 30.48)
H = 128.50         # 3U
T = 2.0            # plate thickness
LABEL_D = 0.6      # label relief, for the two-colour print the range uses

JACK_D = 6.40
ENC_D = 6.40
MIDI_D = 6.40
MOUNT_D = 3.20

# --- measured positions ----------------------------------------------------
JACK_X = 7.20
JACK_Y = [16.61, 30.38, 44.17, 57.96, 71.74, 85.54, 99.31, 113.11]  # bottom -> top

USB_W, USB_H = 6.15, 3.70
USB = [(20.87, 101.50), (20.80, 92.65)]        # dbg (upper), host (lower)

PMOD_W, PMOD_H = 6.00, 16.25
# DELIBERATE DIVERGENCE from the measured CAD. The slots and the GPDI aperture sit at
# x 18.95 on the real panel, which leaves only ~1 mm between them and our three-character
# jack labels. Shifted +2.00 here, which also lands them on x 20.95 — level with dbg
# (20.87) and usb2 (20.80), so the right-hand side reads as one column instead of two.
# This is our render, not a copy of the panel; do not "correct" it back.
RIGHT_SHIFT = 2.00
PMOD = [(18.95 + RIGHT_SHIFT, 56.25), (18.95 + RIGHT_SHIFT, 33.38)]   # EX0, EX1

ENC = (22.80, 113.10)
MIDI = (23.40, 16.43)
MOUNTS = [(7.53, 125.50), (22.52, 125.50), (7.47, 3.00), (22.52, 3.00)]

# --- assumed ---------------------------------------------------------------
# Portrait, not landscape: it stands in the right-hand strip on the same x as the PMOD
# slots. Drawn landscape at first, which made it the widest thing on the panel and pushed
# it across the jack column — visibly wrong as soon as it was rendered.
GPDI = (18.95 + RIGHT_SHIFT, 77.50)
GPDI_W, GPDI_H = 6.00, 15.50

# The real silkscreen is tiny — roughly 6 point, so ~2.1 mm. Jacks are numbered from
# zero, inputs then outputs, with the digit at the bottom-right of each hole.
LABEL_PT = 2.1

# Top -> bottom: in 0-3, then out 0-3. Stock Tiliqua is numbered, and the real silkscreen
# says so; our bitstreams get their jacks named instead, which makes the render plainly
# ours rather than a copy of apf.audio's panel. Names are the control tables in
# LACUNA.md / ORBITA.md. LACUNA is stereo (out0/out1); ORBITA is mono, so its out 1-3
# stay blank.
JACK_LABELS = {
    "TILIQUA": ["0", "1", "2", "3", "0", "1", "2", "3"],
    # Deliberately identical across both instruments. Reading the two control tables
    # side by side, every input is the same *kind* of thing: in0 a gate (LACUNA strikes on
    # a rising edge, ORBITA plucks on one and drones on a held level), in1 1 V/oct from
    # 55 Hz in both, in2 a radial position hub-to-rim, in3 the hole. Only the outputs
    # differ. That is the whole argument of the pair made visible — same membrane, same
    # four controls, two instruments that sound nothing alike.
    "LACUNA":  ["GTE", "V/O", "RAD", "GEO", "OUTL", "OUTR", "", ""],
    "ORBITA":  ["GTE", "V/O", "RAD", "GEO", "OUT", "", "", ""],
}

# House Befaco nut scheme (render/assemble.py, classify_nut): audio in black, control in
# silver, audio out red, sum/CV out gold. Set explicitly rather than inferred, because
# classify_nut works off keywords and none of RAD / GEO / V/O are in its vocabulary — they
# would all fall through to "audio in" black despite being CV.
JACK_NUTS = {
    # Stock hardware: four audio in, four audio out.
    "TILIQUA": ["nut_black"] * 4 + ["nut_red"] * 4,
    # LACUNA: strike, tension, strike position and geometry are all control voltages.
    # Stereo since 49faf3c — a second pickup a quarter turn round — so two audio outs.
    "LACUNA":  ["nut_silver"] * 4 + ["nut_red"] * 2 + ["nut_black"] * 2,
    # ORBITA: all four inputs are control. in0 was black here on a misreading of "drive"
    # as audio-rate — ORBITA.md is explicit that it is a gate edge, so it is silver like
    # LACUNA's strike.
    "ORBITA":  ["nut_silver"] * 4 + ["nut_red"] + ["nut_black"] * 3,
}

# Right-hand column, top to bottom. The encoder carries no label on the real panel.
# All caps, which is the house convention across every panel in ../panels (ATTEN, CLOCK,
# MANUAL, RESET...). The real Tiliqua silkscreen is lowercase; this is another small,
# deliberate departure that keeps the render ours rather than a copy of theirs.
RIGHT_LABELS = [(None, "ENC"), ("DBG", "USB0"), ("USB2", "USB1"),
                ("GPDI", "GPDI"), ("EX0", "PMOD0"), ("EX1", "PMOD1"), ("MIDI", "MIDI")]


def _draw_apertures():
    """Every hole. Draws into whichever BuildSketch is active, so the caller decides
    whether it is being added or subtracted."""
    if True:
        with Locations(*[(JACK_X, y) for y in JACK_Y]):
            Circle(JACK_D / 2)
        with Locations(ENC):
            Circle(ENC_D / 2)
        with Locations(MIDI):
            Circle(MIDI_D / 2)
        with Locations(*USB):
            Rectangle(USB_W, USB_H)
        with Locations(*PMOD):
            Rectangle(PMOD_W, PMOD_H)
        with Locations(GPDI):
            Rectangle(GPDI_W, GPDI_H)
        # Eurorack mounting: slots, not round holes, so the module can be nudged.
        with Locations(*MOUNTS):
            SlotOverall(MOUNT_D + 1.8, MOUNT_D, rotation=0)


def build_base():
    with BuildPart() as part:
        with BuildSketch(Plane.XY):
            with Locations((W / 2, H / 2)):
                Rectangle(W, H)
        extrude(amount=T)
        with BuildSketch(Plane.XY):
            _draw_apertures()
        extrude(amount=T, mode=Mode.SUBTRACT)
    return part.part


def build_labels(title: str = "TILIQUA"):
    """Raised text, so a two-colour print picks it out — same as the rest of the range.

    `title` is the word across the top. The hardware is Tiliqua; the bitstreams running on
    it are LACUNA and ORBITA, and each wants its own title card off the same geometry.
    """
    with BuildPart() as part:
        with BuildSketch(Plane.XY.offset(T)) as sk:
            # the digit tucks into the bottom-right of its jack
            for y, digit in zip(reversed(JACK_Y),
                                JACK_LABELS.get(title, JACK_LABELS["TILIQUA"])):
                if not digit:
                    continue
                with Locations((JACK_X + JACK_D / 2 + 0.5, y - JACK_D / 2 - 0.2)):
                    Text(digit, font_size=LABEL_PT, font_style=FontStyle.BOLD,
                         align=(Align.MIN, Align.MAX))
            # right-hand column: above each aperture, since 6HP leaves no room beside it
            for text, (cx, cy, half) in zip(
                [t for t, _ in RIGHT_LABELS],
                [(ENC[0], ENC[1], ENC_D / 2),
                 (USB[0][0], USB[0][1], USB_H / 2),
                 (USB[1][0], USB[1][1], USB_H / 2),
                 (GPDI[0], GPDI[1], GPDI_H / 2),
                 (PMOD[0][0], PMOD[0][1], PMOD_H / 2),
                 (PMOD[1][0], PMOD[1][1], PMOD_H / 2),
                 (MIDI[0], MIDI[1], MIDI_D / 2)],
            ):
                if text is None:
                    continue
                with Locations((cx, cy + half + 0.6)):
                    Text(text, font_size=LABEL_PT, font_style=FontStyle.BOLD,
                         align=(Align.CENTER, Align.MIN))
            with Locations((W / 2, H - 8.0)):
                Text(title, font_size=3.4, font_style=FontStyle.BOLD,
                     align=(Align.CENTER, Align.CENTER))
        extrude(amount=LABEL_D)
    return part.part


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-mode", choices=["combined", "base", "labels"],
                    default="combined")
    ap.add_argument("--stl")
    ap.add_argument("--template-svg")
    ap.add_argument("--name", default="TILIQUA",
                    help="word across the top: TILIQUA, LACUNA or ORBITA")
    args = ap.parse_args()

    base, labels = build_base(), build_labels(args.name)
    shape = {"base": base, "labels": labels, "combined": base + labels}[args.export_mode]

    if args.stl:
        export_stl(shape, args.stl)
        print(f"wrote {args.stl}")
    if args.template_svg:
        svg = ExportSVG(scale=1)
        svg.add_shape(shape)
        svg.write(args.template_svg)
        print(f"wrote {args.template_svg}")
    if not args.stl and not args.template_svg:
        try:
            from ocp_vscode import show
            show(shape)
        except Exception as exc:  # headless is normal here
            bb = shape.bounding_box()
            print(f"viewer unavailable ({exc.__class__.__name__}); "
                  f"built {bb.size.X:.2f} x {bb.size.Y:.2f} x {bb.size.Z:.2f} mm")


if __name__ == "__main__":
    main()
