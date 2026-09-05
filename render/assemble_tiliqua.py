"""Assemble the Tiliqua panel into per-material STLs + manifest.json for render_panel.py.

A one-off standing in for `assemble.py`, which drives off a `FaceplateParams` built around
regular col_x / row_y grids. Tiliqua's layout is irregular — one jack column, a stack of
connector apertures of three different shapes — so it does not fit that mould. Everything
downstream (materials, lighting, camera) is unchanged: this writes the same asset set.

  ./.venv/bin/python render/assemble_tiliqua.py ORBITA
  blender -b -P render/render_panel.py -- --assets render/out/tiliqua_orbita \
      --out render/out/tiliqua_orbita/orbita.png --view three-quarter
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent / "panels")]

from build123d import Compound, Location, export_stl
import tiliqua as t
import hardware as hw
from assemble import _place_world   # the house placement helper — see note below

TITLE = (sys.argv[1] if len(sys.argv) > 1 else "ORBITA").upper()
OUT = Path(__file__).parent / "out" / f"tiliqua_{TITLE.lower()}"
OUT.mkdir(parents=True, exist_ok=True)

# Hardware is built AT THE ORIGIN and placed with _place_world, which multiplies each
# child by the placement Location and carries its material label across. Moving the parent
# compound instead looks identical until you regroup by material for export: re-parenting
# drops the parent transform, so every jack collapses and the knob lands mirrored. That is
# what the first render of this panel did, and it is why the helper exists.
placements: list = []     # (assembly, x, y, z)
holes: list[dict] = []

base = t.build_base(); base.label = "panel"
labels = t.build_labels(TITLE); labels.label = "labels"

names = t.JACK_LABELS.get(TITLE, t.JACK_LABELS["TILIQUA"])
nuts = t.JACK_NUTS.get(TITLE, t.JACK_NUTS["TILIQUA"])
for y, name, nut in zip(reversed(t.JACK_Y), names, nuts):
    placements.append((hw.jack(nut_label=nut), t.JACK_X, y, t.T))
    holes.append({"x": t.JACK_X, "y": y, "kind": "jack", "label": name, "nut": nut})

placements.append((hw.knob(base_d=9.0, top_d=7.5, height=8.0, indicator_deg=125),
                   t.ENC[0], t.ENC[1], t.T))
holes.append({"x": t.ENC[0], "y": t.ENC[1], "kind": "encoder", "label": ""})

groups: dict[str, list] = {"panel": [base], "labels": [labels]}
for asm, x, y, z in placements:
    for solid in _place_world(asm, x, y, z):
        groups.setdefault(solid.label or "misc", []).append(solid)

stl = {}
for label, shapes in groups.items():
    fn = f"{label}.stl"
    export_stl(Compound(children=list(shapes)), str(OUT / fn))
    stl[label] = fn

(OUT / "manifest.json").write_text(json.dumps({
    "module": f"tiliqua_{TITLE.lower()}",
    "panel_w": t.W, "panel_h": t.H, "thickness": t.T,
    "base_color": [0.028, 0.028, 0.032],
    "label_color": [0.92, 0.92, 0.93],
    "holes": holes, "leds": [], "screens": [], "stl": stl,
}, indent=1))
print(f"{OUT}  ->  {len(stl)} materials: {', '.join(sorted(stl))}")
