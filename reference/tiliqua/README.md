# Tiliqua front panel — reference geometry

`tiliqua-panel.kicad_pcb` is the front panel board file for the **apf.audio Tiliqua**,
copied here so `panels/tiliqua.py` has a source to measure against and so the numbers in
it can be re-checked without a network.

**Source:** [apfaudio/tiliqua-hardware](https://github.com/apfaudio/tiliqua-hardware),
`hardware/tiliqua-panel/`. Note the CAD is *not* in the main
[apfaudio/tiliqua](https://github.com/apfaudio/tiliqua) repo — that carries schematic
PDFs only, to keep clones small.

**Copyright (C) 2024 Sebastian Holzapfel. Licensed CERN-OHL-S v2** (Open Hardware
Licence, Strongly Reciprocal), the same as the rest of the Tiliqua project. It is
included here unmodified, as reference material. Our panel script is a render-only
approximation for video and web use and is not a derivative board design; if that ever
changes — if anyone were to manufacture from it — the reciprocal terms apply and would
need reading properly first.

## What is and is not in this file

The panel board carries the jack column, both USB-C apertures, both PMOD slots, the
encoder and TRS MIDI holes, and the mounting corners. The **encoder, GPDI, USB and MIDI
electronics are on the motherboard**, a separate project in the same repo, so the GPDI
aperture does not appear in this file at all — see the ASSUMED section of
`panels/tiliqua.py`.

The jack holes are drawn as `MountingHole_6.4mm_M6` footprints. They are jacks; 6.4 mm is
just a convenient circle.
