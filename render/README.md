# Photoreal render pipeline

Turns a panel module's `FaceplateParams` into a populated, photoreal render of the
assembled module (panel + raised labels + placed hardware: jacks, knobs, nuts,
LEDs, OLED screens).

It imports the panel generators from [../panels/](../panels/) by bare module name,
so it stays in sync with the source geometry automatically.

## Pieces

| File | Role |
| --- | --- |
| `hardware.py` | Parametric build123d models of panel hardware (jacks, knobs, nuts, OLED, LEDs), positioned relative to the panel front face. |
| `assemble.py` | Loads a module's `FaceplateParams`, places hardware from the module's own parameters, and exports per-material STLs + `manifest.json`. |
| `render_panel.py` | Headless Blender (4.x) renderer: reads the STLs + manifest, assigns PBR materials, builds emissive screen planes, lights and renders with Cycles. |
| `screens/` | Screen images composited onto OLED planes. |
| `out/` | Generated per-module assets and renders (one subdir per module). |
| `firmware_screen_prompt.md` | Prompt used to generate the screen artwork. |

## Usage

```sh
# 1. assemble: panel + hardware -> per-material STLs + manifest
./.venv/bin/python render/assemble.py duallpg --out render/out/duallpg
./.venv/bin/python render/assemble.py duallpg --show          # OCP preview instead

# 2. render with Blender 4.x
blender -b -P render/render_panel.py -- \
    --assets render/out/duallpg \
    --out render/out/duallpg/duallpg.png \
    --view three-quarter
```

The module argument (`duallpg` above) is the panel script's module name from
[../panels/](../panels/).
