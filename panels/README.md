# Eurorack faceplate generators

One build123d script per module. Most build a **base** solid plus a separate
**labels** solid for 2-colour printing, and share the same CLI:

```sh
./.venv/bin/python panels/<script>.py                       # view in OCP CAD Viewer
./.venv/bin/python panels/<script>.py --export-mode base   --stl base.stl
./.venv/bin/python panels/<script>.py --export-mode labels --stl labels.stl
./.venv/bin/python panels/<script>.py --template-svg out.svg   # 1:1 paper template
```

Run a script with `--help` for its own options (text mode, fit clearances, etc.).
Checked-in printables for these live in [../exports/panels/](../exports/panels/).

## PCB faceplates

[`kicad_faceplate.py`](kicad_faceplate.py) turns any of these scripts into a
KiCad board and a JLCPCB-ready gerber zip — same holes, same cutouts, same
labels, but as a black-solder-mask PCB with white silkscreen instead of a
two-colour print:

```sh
python3 panels/kicad_faceplate.py panels/daisy_braids.py \
    --outdir exports/pcb/joy_10hp --name joy_10hp --gerbers
```

It parses the panel script rather than importing it, so it needs nothing but
stock Python 3 (and `kicad-cli` for `--gerbers`). Output and ordering notes live
in [../exports/pcb/](../exports/pcb/).

## Modules

| Script | Module | Size / layout | Notes |
| --- | --- | --- | --- |
| `amyboard.py` | AMYboard | 10HP | Rebuilt from the assembled-panel photo |
| `daisy_braids.py` | DaisyBraids | 10HP, OLED | |
| `daisy_grids.py` | DaisyGrids | 10HP, OLED | Same hole layout as `daisy_intervalosc.py`, different labels |
| `daisy_intervalosc.py` | IntervalOsc | 10HP | Same hole layout as `daisy_grids.py`, different labels. *Renamed from `daisy_intervalosc copy.py`* |
| `daisy_mfx.py` | Daisy Patch Init OLED (MFX) | 10HP | |
| `daisy_multiosc.py` | DaisyMultiOsc | 10HP, OLED | Based on `daisy_braids.py` (same holes), different labels |
| `duallpg.py` | DualLPG (N8Synth) | 10HP, 3×6 | Dual low-pass-gate layout |
| `esp32_clklinkrec.py` | ClkLinkRec (N8Synth, ESP32) | 4HP, 1×6 | Name maps to the matching eurorack firmware repo |
| `ksoloti_biggenes.py` | Ksoloti Big Genes (Elements) | ~20HP | Approximate, render-only |
| `mkikick.py` | mkikick | 6HP, 2×6 | Derives `FaceplateParams` from `teensy_expander.py` |
| `nanoesp32_corthex.py` | CortHex (N8Synth, Nano-ESP32) | 10HP, 3×6 | All 18 holes, no screen. Name maps to the matching eurorack repo |
| `patch_init_oled.py` | Daisy Patch Init + Grove 0.66" OLED | 10HP, OLED | **Face-down 2-colour + back-side stiffening ribs.** Superseded by the face-up recipe, kept for the rib/assembly demo — see note below |
| `pico2w_onclite.py` | OnCLite (N8Synth 6HP, Pico 2 W) | 6HP, 2×N | Name maps to the matching eurorack repo; docstring/argparse text still say generic "N8Synth 6HP" |
| `seed_panel.py` | Seeed XIAO RP2040 | 8HP Intellijel 1U | Tactile switch + 3 mm LED; includes retainer fit variants |
| `seed_recorder_panel.py` | Seed Recorder | 8HP Intellijel 1U | |
| `teensy_chaos.py` | Chaos | 10HP, 3×6 | Derived from a KiCad/PCBNEW label-template SVG. Renamed from "TeensyChaos" (trademark) |
| `teensy_expander.py` | Teensy Expander | 6HP, 2×N | Based on N8Synth 6HP |
| `teensy_move.py` | Boy | 10HP, 3×6 | Derived from a KiCad/PCBNEW label-template SVG. Renamed from "TeensyMove" (trademark) |

> Note: `teensy_chaos.py` and `teensy_move.py` have no default export paths — export
> with explicit `--stl-base`/`--stl-labels` paths, named after the product
> (`chaos_*.stl`, `boy_*.stl`).

## Ribbed panel + raised lettering (`patch_init_oled.py`)

Template for the Patch-Init/OLED family. It prints as **two parts**: a face-up panel with
raised letters, and a stiffening rib grid you glue to its back.

```sh
./.venv/bin/python panels/patch_init_oled.py \
    --stl-base exports/panels/patch_init_oled_panel.stl \
    --stl-ribs exports/panels/patch_init_oled_ribgrid.stl
```

**Why two parts.** Ribs can only print pointing *up*, so an integrated rib grid forces the
panel **face-down** — which puts the lettering on the bed, where the over-extruded first
layer squashes the thin letter voids and the small counters (the holes in O/D/R) lift off
as isolated islands. Every face-down lettering trick fights that. Printing the panel
**face-up** instead puts the letters on *top*, laid last onto solid material: crisp, and
trivially sliceable. The ribs then simply become their own part.

- **Panel (`emboss`, the default).** Flat back on the bed, letters raised `label_height`
  proud of the front. They're the only geometry above `z = thickness`, so **one
  height-based filament change at `Z = thickness` (2.0 mm)** prints them white — no AMS,
  no per-region colour, no multi-part assembly in the slicer.
- **Rib grid (`--stl-ribs`).** Exported with its **glue face on the bed** (flat and smooth
  for a good bond) and already mirrored into the panel's back-view, so you lift it straight
  off the plate and set it glue-face-down on the panel back, ribs standing away into the
  case. Glue the *whole* footprint — a rib only stiffens the panel if the bond transfers
  shear, so spots of glue do very little.
- **Layout.** KiCad hole data is run through a 180° `phys()` flip to the real mounted layout
  (**pots at the top, jacks at the bottom**, MOD 1 top-left, OUT-R bottom-right).
- **Label size.** `label_size` is 4.0 mm — the largest that still fits every label inside the
  12.17 mm jack pitch. Its 0.58 mm stems are ~2.6× a 0.22 mm line width; 3.2 mm gave only
  ~2.1× and printed mushy.
- **Ribs.** A lattice trimmed by **rectangular** keep-outs around every jack / pot / OLED
  board / SD slot / LED, kept clear of the top/bottom rack-rail border. Ribs sit on explicit
  centrelines (`rib_v` verticals as `(x, y_start, y_end)`, `rib_h` horizontals as
  `(y, x_start, x_end)`, physical/mounted coords). Tune those plus `rib_height`,
  `rib_thickness`, `rail_margin`, `rib_keepout_clearance`, the `*_body` footprints and
  `mount_keepout_r`; disable with `--no-ribs`.
- Running it shows the panel **plus the rib grid and preview models** of the mounted
  jacks/pots/OLED, so you can check the ribs clear the hardware. `--no-preview` hides them.

### Other text modes (face-down, not recommended)
`--text-mode inlay` (flush white inlay filling front-face pockets) and `--text-mode deboss`
(recessed letters revealed by two whole-layer filament changes, or paint-filled) both print
face-down and both suffer the squashed-first-layer problem above. Kept for reference.

```sh
./.venv/bin/python panels/patch_init_oled.py                              # view assembly + ribs + parts
./.venv/bin/python panels/patch_init_oled.py --stl-base base.stl --stl-labels labels.stl
```

In Bambu Studio: import both STLs together (keep positions), assign **white → labels,
black → base**, and print **by object with a Z hop / clearance** so the toolhead can't drag
the freshly-laid white letters before the black locks them in. No supports — the ribs are
raised features on the upward (back) face.

The `build_base` / `build_labels` / `build_ribs` / `build_components` functions take the hole
list + label tuples as arguments, so the other Patch-Init/OLED panels can import them and pass
their own labels/branding while keeping the face-down + rib behaviour.
