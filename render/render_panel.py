"""Headless Blender renderer for an assembled Eurorack panel.

Reads the per-material STLs + manifest.json produced by `assemble.py`, assigns
PBR materials, builds emissive screen planes, lights a studio scene and renders
with Cycles.

Run (Blender 4.x):
  blender -b -P render/render_panel.py -- --assets render/out/duallpg \
          --out render/out/duallpg/duallpg.png [--samples 160] [--res 1100x2800] \
          [--view three-quarter|front]

Everything is in millimetres (STL units); the camera/lights are placed in the
same units, so distances look large but are internally consistent.
"""

import json
import math
import sys
from pathlib import Path

import bpy
import mathutils


# --------------------------------------------------------------------------- #
# Arg parsing (everything after the first standalone "--").
# --------------------------------------------------------------------------- #
def parse_args() -> dict:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    out: dict = {"samples": 160, "res": (1100, 2800), "views": ["flat", "tilt"]}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--assets":
            out["assets"] = Path(argv[i + 1]); i += 2
        elif a == "--outdir":
            out["outdir"] = Path(argv[i + 1]); i += 2
        elif a == "--samples":
            out["samples"] = int(argv[i + 1]); i += 2
        elif a == "--res":
            w, h = argv[i + 1].lower().split("x"); out["res"] = (int(w), int(h)); i += 2
        elif a == "--views":
            out["views"] = [v.strip() for v in argv[i + 1].split(",") if v.strip()]; i += 2
        elif a == "--view":
            out["views"] = [argv[i + 1]]; i += 2
        elif a == "--no-backing":
            out["no_backing"] = True; i += 1
        elif a == "--transparent":
            out["transparent"] = True; i += 1
        else:
            i += 1
    if "assets" not in out:
        raise SystemExit("--assets <dir> is required")
    out.setdefault("outdir", out["assets"])
    return out


# --------------------------------------------------------------------------- #
# Scene helpers.
# --------------------------------------------------------------------------- #
def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def set_principled(bsdf, **kw) -> None:
    """Set Principled BSDF inputs by friendly name, tolerating 3.x/4.x names."""
    aliases = {
        "Specular": ["Specular IOR Level", "Specular"],
        "Emission Color": ["Emission Color", "Emission"],
        "Coat Weight": ["Coat Weight", "Clearcoat"],
        "Coat Roughness": ["Coat Roughness", "Clearcoat Roughness"],
        "Transmission": ["Transmission Weight", "Transmission"],
    }
    for key, val in kw.items():
        names = aliases.get(key, [key])
        for n in names:
            if n in bsdf.inputs:
                bsdf.inputs[n].default_value = val
                break


def make_material(name: str, **kw):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    set_principled(bsdf, **kw)
    return mat


def material_library(manifest: dict) -> dict:
    base = tuple(manifest["base_color"]) + (1.0,)
    label = tuple(manifest["label_color"]) + (1.0,)
    nickel = (0.70, 0.72, 0.75, 1.0)
    mats = {
        "panel": make_material("panel", **{"Base Color": base, "Roughness": 0.78, "Metallic": 0.0, "Specular": 0.3}),
        "labels": make_material("labels", **{"Base Color": label, "Roughness": 0.62, "Metallic": 0.0}),
        "nut": make_material("nut", **{"Base Color": nickel, "Metallic": 1.0, "Roughness": 0.34}),
        "washer": make_material("washer", **{"Base Color": nickel, "Metallic": 1.0, "Roughness": 0.40}),
        "screw": make_material("screw", **{"Base Color": nickel, "Metallic": 1.0, "Roughness": 0.30}),
        # Befaco knurled nuts: anodised aluminium (metallic, low-ish roughness).
        "nut_silver": make_material("nut_silver", **{"Base Color": (0.80, 0.81, 0.83, 1), "Metallic": 1.0, "Roughness": 0.30}),
        "nut_black": make_material("nut_black", **{"Base Color": (0.015, 0.015, 0.018, 1), "Metallic": 0.9, "Roughness": 0.40, "Coat Weight": 0.2}),
        "nut_red": make_material("nut_red", **{"Base Color": (0.42, 0.025, 0.025, 1), "Metallic": 0.9, "Roughness": 0.33, "Coat Weight": 0.2}),
        "nut_gold": make_material("nut_gold", **{"Base Color": (0.90, 0.68, 0.26, 1), "Metallic": 1.0, "Roughness": 0.28}),
        "jack_body": make_material("jack_body", **{"Base Color": (0.02, 0.02, 0.02, 1), "Roughness": 0.5}),
        "knob": make_material("knob", **{"Base Color": (0.03, 0.03, 0.035, 1), "Roughness": 0.55, "Coat Weight": 0.3, "Coat Roughness": 0.25}),
        "knob_pointer": make_material("knob_pointer", **{"Base Color": (0.9, 0.9, 0.92, 1), "Roughness": 0.4}),
        "knob_cap": make_material("knob_cap", **{"Base Color": (0.74, 0.75, 0.77, 1), "Metallic": 1.0, "Roughness": 0.34}),
        "trimmer": make_material("trimmer", **{"Base Color": (0.02, 0.02, 0.022, 1), "Roughness": 0.5, "Coat Weight": 0.2}),
        "trimmer_pointer": make_material("trimmer_pointer", **{"Base Color": (0.9, 0.9, 0.92, 1), "Roughness": 0.4}),
        "led_holder": make_material("led_holder", **{"Base Color": (0.015, 0.015, 0.017, 1), "Roughness": 0.35, "Coat Weight": 0.3}),
        "switch_cap": make_material("switch_cap", **{"Base Color": (0.16, 0.16, 0.18, 1), "Roughness": 0.6, "Coat Weight": 0.12}),
        "switch_cap_silver": make_material("switch_cap_silver", **{"Base Color": (0.74, 0.75, 0.77, 1), "Metallic": 1.0, "Roughness": 0.32}),
        "bolt": make_material("bolt", **{"Base Color": (0.40, 0.40, 0.43, 1), "Metallic": 0.8, "Roughness": 0.5}),
        "toggle": make_material("toggle", **{"Base Color": (0.72, 0.73, 0.75, 1), "Metallic": 1.0, "Roughness": 0.3}),
        "sd_card": make_material("sd_card", **{"Base Color": (0.04, 0.20, 0.55, 1), "Roughness": 0.45, "Coat Weight": 0.2}),
        # Idle red LEDs: glossy saturated-red epoxy dome, only faintly self-lit
        # (high emission desaturates to pink under the tonemap).
        "led": make_material("led", **{"Base Color": (0.45, 0.0, 0.0, 1), "Emission Color": (0.7, 0.02, 0.02, 1), "Emission Strength": 0.6, "Roughness": 0.25}),
        "led_dome": make_material("led_dome", **{"Base Color": (0.55, 0.0, 0.0, 1), "Emission Color": (0.75, 0.02, 0.02, 1), "Emission Strength": 0.7, "Roughness": 0.08, "Coat Weight": 0.6}),
        "led_blue": make_material("led_blue", **{"Base Color": (0.0, 0.05, 0.5, 1), "Emission Color": (0.03, 0.12, 0.85, 1), "Emission Strength": 0.7, "Roughness": 0.25}),
        "led_blue_dome": make_material("led_blue_dome", **{"Base Color": (0.0, 0.08, 0.6, 1), "Emission Color": (0.04, 0.16, 0.9, 1), "Emission Strength": 0.85, "Roughness": 0.08, "Coat Weight": 0.6}),
        "led_green": make_material("led_green", **{"Base Color": (0.0, 0.4, 0.05, 1), "Emission Color": (0.05, 0.75, 0.1, 1), "Emission Strength": 0.7, "Roughness": 0.25}),
        "led_green_dome": make_material("led_green_dome", **{"Base Color": (0.0, 0.5, 0.07, 1), "Emission Color": (0.06, 0.8, 0.12, 1), "Emission Strength": 0.85, "Roughness": 0.08, "Coat Weight": 0.6}),
        "screen_body": make_material("screen_body", **{"Base Color": (0.02, 0.12, 0.05, 1), "Roughness": 0.5}),
        "screen_bezel": make_material("screen_bezel", **{"Base Color": (0.01, 0.01, 0.012, 1), "Roughness": 0.08, "Coat Weight": 0.8}),
        "misc": make_material("misc", **{"Base Color": (0.5, 0.5, 0.5, 1), "Roughness": 0.6}),
    }
    return mats


def import_stls(assets: Path, manifest: dict, mats: dict):
    objs = []
    for label, fn in manifest["stl"].items():
        path = assets / fn
        if not path.exists():
            print(f"  (skip missing {fn})")
            continue
        before = set(bpy.data.objects)
        try:
            bpy.ops.wm.stl_import(filepath=str(path))
        except AttributeError:
            bpy.ops.import_mesh.stl(filepath=str(path))  # legacy
        new = [o for o in bpy.data.objects if o not in before]
        mat = mats.get(label, mats["misc"])
        for o in new:
            o.data.materials.clear()
            o.data.materials.append(mat)
            smooth = label.startswith("nut") or label.startswith("led") or label.startswith("switch_cap") or label in {
                "knob", "knob_cap", "trimmer", "washer", "screw", "bolt", "toggle"
            }
            for poly in o.data.polygons:
                poly.use_smooth = smooth
            objs.append(o)
    return objs


def build_screens(manifest: dict, assets: Path):
    for s in manifest.get("screens", []):
        img_path = assets / s["image"]
        if not img_path.exists():
            # No screen content -> leave it off (dark recess shows through the
            # window); don't fake a glow.
            print(f"  (screen image {s['image']} missing -> screen left dark/off)")
            continue
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(s["x"], s["y"] + s.get("dy", 0.0), s["z"]))
        plane = bpy.context.active_object
        plane.scale = (s["w"], s["h"], 1.0)
        mat = bpy.data.materials.new(f"screen_emit_{s['x']:.0f}")
        mat.use_nodes = True
        nt = mat.node_tree
        emis = nt.nodes.new("ShaderNodeEmission")
        emis.inputs["Strength"].default_value = 6.0
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(img_path))
        tex.interpolation = "Closest"
        nt.links.new(tex.outputs["Color"], emis.inputs["Color"])
        out = nt.nodes.get("Material Output")
        nt.links.new(emis.outputs["Emission"], out.inputs["Surface"])
        plane.data.materials.clear()
        plane.data.materials.append(mat)


def add_backing(manifest: dict) -> None:
    """A near-black matte plane just behind the panel so holes/gaps read as
    dark recesses (the module interior) rather than the bright background."""
    w, h = manifest["panel_w"], manifest["panel_h"]
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(w / 2, h / 2, -1.6))
    plane = bpy.context.active_object
    plane.scale = (w + 10, h + 10, 1.0)
    mat = make_material("backing", **{"Base Color": (0.006, 0.006, 0.008, 1), "Roughness": 0.95, "Specular": 0.1})
    plane.data.materials.clear()
    plane.data.materials.append(mat)


def setup_world() -> None:
    """Sky texture world for clean reflections + soft ambient fill."""
    world = bpy.data.worlds.new("studio")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    bg.inputs["Strength"].default_value = 0.18
    try:
        sky = nt.nodes.new("ShaderNodeTexSky")
        sky.sky_type = "NISHITA"
        sky.sun_elevation = math.radians(35)
        nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    except Exception:  # noqa: BLE001
        bg.inputs["Color"].default_value = (0.6, 0.62, 0.66, 1.0)


def setup_lights(cx: float, cy: float) -> None:
    # Suns are distance-independent, so they behave predictably in mm units.
    # Key sun, upper-front-left.
    bpy.ops.object.light_add(type="SUN")
    key = bpy.context.active_object
    key.data.energy = 2.4
    key.data.angle = math.radians(4)  # soft-ish shadows
    key.rotation_euler = (math.radians(22), math.radians(-20), 0)
    # Fill sun, opposite side, dim.
    bpy.ops.object.light_add(type="SUN")
    fill = bpy.context.active_object
    fill.data.energy = 0.7
    fill.rotation_euler = (math.radians(15), math.radians(28), 0)
    # Soft highlight sun from above to pop the knurled-nut edges (kept gentle).
    bpy.ops.object.light_add(type="SUN")
    rim = bpy.context.active_object
    rim.data.energy = 1.1
    rim.data.angle = math.radians(8)
    rim.rotation_euler = (math.radians(8), 0, math.radians(10))


def setup_camera(manifest: dict, view: str):
    w, h = manifest["panel_w"], manifest["panel_h"]
    cx, cy = w / 2, h / 2

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 78
    cam_data.sensor_width = 36
    cam_data.sensor_fit = "VERTICAL"  # panel's long axis is vertical (world Y)
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)

    # Explicit transform keeps the panel upright (world +Y = screen up).
    if view in ("flat", "front"):
        # True flat elevation: orthographic, straight down -Z.
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = manifest["panel_h"] + 14
        cam.location = (cx, cy, 400)
        cam.rotation_euler = (0.0, 0.0, 0.0)
    else:  # tilt / three-quarter: yaw about Y, camera on the look ray.
        yaw = math.radians(16)
        dist = 485
        cam.location = (cx + dist * math.sin(yaw), cy, dist * math.cos(yaw))
        cam.rotation_euler = (0.0, yaw, 0.0)

    bpy.context.scene.camera = cam
    return cx, cy


def setup_render(args: dict) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args["samples"]
    scene.cycles.use_denoising = True
    # Try GPU (Metal on macOS); fall back to CPU silently.
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        scene.cycles.device = "GPU"
    except Exception as ex:  # noqa: BLE001
        print("  (GPU unavailable, using CPU):", ex)
    scene.render.resolution_x, scene.render.resolution_y = args["res"]
    scene.render.film_transparent = bool(args.get("transparent"))
    available = [vt.name for vt in bpy.types.ColorManagedViewSettings.bl_rna.properties["view_transform"].enum_items]
    for vt in ("AgX", "Filmic", "Standard"):
        if vt in available:
            scene.view_settings.view_transform = vt
            break


def main() -> None:
    args = parse_args()
    assets = args["assets"]
    manifest = json.loads((assets / "manifest.json").read_text())

    reset_scene()
    mats = material_library(manifest)
    import_stls(assets, manifest, mats)
    build_screens(manifest, assets)
    if not args.get("no_backing"):
        add_backing(manifest)
    setup_world()
    setup_lights(manifest["panel_w"] / 2, manifest["panel_h"] / 2)
    setup_render(args)

    module = manifest.get("module", "panel")
    outdir = args["outdir"]
    outdir.mkdir(parents=True, exist_ok=True)

    # Build the scene once; render each requested view from its own camera.
    for view in args["views"]:
        setup_camera(manifest, view)
        out_path = outdir / f"{module}_{view}.png"
        bpy.context.scene.render.filepath = str(out_path)
        print(f"Rendering {view} -> {out_path}  ({args['res'][0]}x{args['res'][1]}, {args['samples']} spp)")
        bpy.ops.render.render(write_still=True)
    print("Done.")


if __name__ == "__main__":
    main()
