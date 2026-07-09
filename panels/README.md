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
| `patch_init_oled.py` | Daisy Patch Init + Grove 0.66" OLED | 10HP, OLED | **Face-down 2-colour + back-side stiffening ribs.** Reusable template — see note below |
| `pico2w_onclite.py` | OnCLite (N8Synth 6HP, Pico 2 W) | 6HP, 2×N | Name maps to the matching eurorack repo; docstring/argparse text still say generic "N8Synth 6HP" |
| `seed_panel.py` | Seeed XIAO RP2040 | 8HP Intellijel 1U | Tactile switch + 3 mm LED; includes retainer fit variants |
| `seed_recorder_panel.py` | Seed Recorder | 8HP Intellijel 1U | |
| `teensy_chaos.py` | TeensyChaos | 10HP, 3×6 | Derived from a KiCad/PCBNEW label-template SVG |
| `teensy_expander.py` | Teensy Expander | 6HP, 2×N | Based on N8Synth 6HP |
| `teensy_move.py` | N8Synth 10HP (Teensy Move) | 10HP, 3×6 | Derived from a KiCad/PCBNEW label-template SVG |

> Note: `teensy_chaos.py` and `teensy_move.py` both default their export name to
> `n8_10hp_base/labels.stl` — pass explicit `--stl` paths so they don't overwrite
> each other.

## Face-down 2-colour + ribbed panels (`patch_init_oled.py`)

Most scripts here print **face-up** (raised white labels on top of a black base).
`patch_init_oled.py` is the template for the alternative **face-down** workflow:

- The KiCad hole data is run through a 180° `phys()` flip to the real mounted layout
  (**pots at the top, jacks at the bottom**, MOD 1 top-left, OUT-R bottom-right). It's
  modelled **front-face-up** so the viewer matches the mounted module; on **export** the
  front face is flipped onto the bed (`z = 0`) for face-down printing.
- Two `--text-mode` options for the lettering:
  - **`inlay`** (default, 2-colour) — the base has letter-shaped pockets in its front skin
    (`label_height`, default 0.2 mm = 2 layers) and the `labels` solid fills them. Printed
    face-down this is the "two white layers first, then build the black panel up on top"
    trick; the letters end up **mirrored** so they read correctly through the front. Crisp,
    but the tiny white letter islands can lift off the bed.
  - **`deboss`** — single-colour black panel with recessed letters (`deboss_depth`). Two
    uses, no white STL and reliable adhesion (solid black first layer with holes):
    - **Layer colour-change reveal (recommended, single extruder):** the recess *is* the
      "letter holes" in the first black layers. Print black up to `Z = recess depth`, swap to
      white for ~2 layers (which cap/bridge the letters), then back to black. Two whole-layer
      filament changes (add them on Bambu's layer slider **by Z height**) reveal white
      letters — no AMS, no per-region colour. Set the recess to `black layers × layer height`
      via `--label-height` (e.g. 0.3 mm = 3 × 0.1 mm).
    - **Paint-fill:** flood the recesses and wipe — but watery paint tends to bead out of
      narrow grooves, so the colour-change reveal above is usually better.
- The back carries a **lattice of stiffening ribs** (vertical + horizontal) trimmed by
  **rectangular** keep-outs around every jack / pot / OLED board / SD slot / LED, so a solid
  ~2 mm rib threads the gaps between them; ribs also stay out of the top/bottom rack-rail
  border. Ribs sit on explicit centrelines (`rib_v` verticals as `(x, y_start, y_end)`,
  `rib_h` horizontals as `(y, x_start, x_end)`, physical/mounted coords) chosen to fall in
  the gaps between rows/columns; tune those plus `rib_height`,
  `rib_thickness`, `rail_margin`, `rib_keepout_clearance`, the `*_body` footprints and
  `mount_keepout_r` in `PanelParams`; disable with `--no-ribs`.
- Running it shows the panel **plus preview models** of the mounted jacks/pots/OLED so you
  can check the ribs clear the hardware. `--no-preview` hides them.

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
