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
| `pico2w_onclite.py` | OnCLite (N8Synth 6HP, Pico 2 W) | 6HP, 2×N | Name maps to the matching eurorack repo; docstring/argparse text still say generic "N8Synth 6HP" |
| `seed_panel.py` | Seeed XIAO RP2040 | 8HP Intellijel 1U | Tactile switch + 3 mm LED; includes retainer fit variants |
| `seed_recorder_panel.py` | Seed Recorder | 8HP Intellijel 1U | |
| `teensy_chaos.py` | TeensyChaos | 10HP, 3×6 | Derived from a KiCad/PCBNEW label-template SVG |
| `teensy_expander.py` | Teensy Expander | 6HP, 2×N | Based on N8Synth 6HP |
| `teensy_move.py` | N8Synth 10HP (Teensy Move) | 10HP, 3×6 | Derived from a KiCad/PCBNEW label-template SVG |

> Note: `teensy_chaos.py` and `teensy_move.py` both default their export name to
> `n8_10hp_base/labels.stl` — pass explicit `--stl` paths so they don't overwrite
> each other.
