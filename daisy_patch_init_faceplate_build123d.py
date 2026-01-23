"""Daisy Patch Init 10HP faceplate rebuilt in build123d.

This script mirrors the workflow/standards used by `n8synth_faceplate_build123d.py`:
- Base panel solid (for the main print)
- Raised label solid (for 2-color printing)
- Optional 1:1 SVG/DXF print template

Unlike the N8Synth model (which is parameterized), this one primarily imports the
hole and label placement from the KiCad panel file in:
  Daisy_Patch_Init/Frontpanel-DesignFiles/blank.kicad_pcb

Run:
  ./.venv/bin/python daisy_patch_init_faceplate_build123d.py

Export STLs:
  ./.venv/bin/python daisy_patch_init_faceplate_build123d.py --export-mode base   --stl patch_init_base.stl
  ./.venv/bin/python daisy_patch_init_faceplate_build123d.py --export-mode labels --stl patch_init_labels.stl

Notes:
- Coordinates follow the established convention: XY origin at panel bottom-left, Z up.
- Text is imported from KiCad `gr_text` blocks on F.SilkS (size/rotation/justification).
  The font face is normalized to `Arial` bold for consistency with the N8Synth script.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import math

from build123d import (
    Align,
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    FontStyle,
    Location,
    Locations,
    Mode,
    Plane,
    Rectangle,
    RectangleRounded,
    SlotOverall,
    Text,
    extrude,
    export_step,
    export_stl,
)
from build123d.exporters import ExportDXF, ExportSVG
from ocp_vscode import Camera, show


def _ocp_port(default: int = 3939) -> int:
    try:
        raw = os.environ.get("OCP_VSCODE_PORT") or os.environ.get("OCP_PORT") or str(default)
        return int(raw)
    except ValueError:
        return default

ExportMode = Literal["combined", "base", "labels"]
LabelSource = Literal["holes", "kicad"]
TextMode = Literal["emboss", "deboss", "inlay"]


"""Label text configuration.

`text_labels` are the default labels placed using `label_offset` (typically below holes).

`text_labels_above` are optional additional labels placed above holes using the same X
offset and mirrored Y offset. Use "" to omit an above label for that hole.
"""


text_labels = [
    "OUTR",
    "OUTL",
    "INR",
    "INL",
    "CV_8",
    "CV_7",
    "CV_6",
    "CV_5",
    "B6",
    "B5",
    "B9",
    "B10",
    "B8",
    "C10",
    "B7",
    "CV_4",
    "CV_3",
    "CV_2",
    "CV_1",
]


# Above-hole labels (same ordering/length as `text_labels`).
# Leave entries as "" for no above label at that hole.
text_labels_above = [
    "",  # OUTR
    "",  # OUTL
    "",  # INR
    "",  # INL
    "",  # CV_8
    "",  # CV_7
    "",  # CV_6
    "",  # CV_5
    "",  # B6
    "",  # B5
    "",  # B9
    "",  # B10
    "ALT",  # B8
    "",  # C10
    "",  # B7
    "",  # CV_4
    "",  # CV_3
    "",  # CV_2
    "",  # CV_1
]


@dataclass(frozen=True)
class KiCadHole:
    shape: Literal["circle", "oval"]
    x: float
    y: float
    # circle => (d, d), oval => (major, minor)
    drill: tuple[float, float]


# Hard-coded drill locations (np_thru_hole pads) for this panel.
#
# This intentionally decouples hole placement/sizing from the KiCad file so you can
# directly edit drill sizes here.
#
# NOTE: One hole is smaller in the KiCad source (drill 5.5 at ~x=8.65, y=59.29).
# It is set to 6.2 here to match the other B holes.
HARDCODED_HOLES: tuple[KiCadHole, ...] = (
    KiCadHole(shape="circle", x=7.149999, y=111.900001, drill=(6.200000, 6.200000)),
    KiCadHole(shape="circle", x=19.316694, y=111.900001, drill=(6.200000, 6.200000)),
    KiCadHole(shape="circle", x=31.483297, y=111.900001, drill=(6.200000, 6.200000)),
    KiCadHole(shape="circle", x=43.650000, y=111.900001, drill=(6.200000, 6.200000)),
    KiCadHole(shape="circle", x=7.149999, y=98.311904, drill=(6.200000, 6.200000)),
    KiCadHole(shape="circle", x=19.316694, y=98.311904, drill=(6.200000, 6.200000)),
    KiCadHole(shape="circle", x=31.483297, y=98.311904, drill=(6.200000, 6.200000)),
    KiCadHole(shape="circle", x=43.650000, y=98.311904, drill=(6.200000, 6.200000)),
    KiCadHole(shape="circle", x=7.149999, y=84.561952, drill=(6.200000, 6.200000)),
    KiCadHole(shape="circle", x=19.316694, y=84.561903, drill=(6.200000, 6.200000)),
    KiCadHole(shape="circle", x=31.483297, y=84.561903, drill=(6.200000, 6.200000)),
    KiCadHole(shape="circle", x=43.650000, y=84.561903, drill=(6.200000, 6.200000)),
    KiCadHole(shape="circle", x=25.502894, y=61.956597, drill=(6.200000, 6.200000)),
    KiCadHole(shape="circle", x=8.649995, y=59.288225, drill=(6.200000, 6.200000)),
    KiCadHole(shape="circle", x=42.155087, y=59.288206, drill=(6.200000, 6.200000)),
    KiCadHole(shape="circle", x=11.175500, y=42.027311, drill=(7.200000, 7.200000)),
    KiCadHole(shape="circle", x=39.649996, y=42.027311, drill=(7.200000, 7.200000)),
    KiCadHole(shape="circle", x=11.175500, y=22.904289, drill=(7.200000, 7.200000)),
    KiCadHole(shape="circle", x=39.649996, y=22.904289, drill=(7.200000, 7.200000)),
    KiCadHole(shape="circle", x=25.399999, y=19.252005, drill=(3.200000, 3.200000)),
    KiCadHole(shape="oval", x=43.100000, y=125.500000, drill=(5.000000, 3.000000)),
    KiCadHole(shape="oval", x=7.500000, y=3.000000, drill=(5.000000, 3.000000)),
)


@dataclass(frozen=True)
class KiCadRect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return abs(self.x1 - self.x0)

    @property
    def h(self) -> float:
        return abs(self.y1 - self.y0)

    @property
    def min_x(self) -> float:
        return min(self.x0, self.x1)

    @property
    def min_y(self) -> float:
        return min(self.y0, self.y1)


@dataclass(frozen=True)
class KiCadText:
    text: str
    x: float
    y: float
    rot_deg: float
    size_mm: float
    justify_h: Literal["left", "center", "right"]
    justify_v: Literal["bottom", "center", "top"]


@dataclass(frozen=True)
class KiCadPanelImport:
    # Transform from board coordinates to panel-local coordinates:
    # local = R(-rot) * (board_xy - origin)
    origin_x: float
    origin_y: float
    origin_rot_deg: float
    outline: KiCadRect
    cutouts: tuple[KiCadRect, ...]
    holes: tuple[KiCadHole, ...]
    labels: tuple[KiCadText, ...]


@dataclass(frozen=True)
class FaceplateParams:
    # Panel
    panel_w: float = 50.8  # 10HP
    panel_h: float = 128.5
    thickness: float = 3.0

    # Optional Eurorack mounting (if not using the KiCad-provided slots)
    add_mount_holes: bool = False
    mount_hole_d: float = 3.2
    mount_x_from_edge: float = 3.0
    mount_y_from_edge: float = 3.0
    mount_both_sides: bool = True
    mount_slot: bool = True
    mount_slot_len: float = 4.0

    # KiCad import
    kicad_panel_path: Path = Path("Daisy_Patch_Init/Frontpanel-DesignFiles/blank.kicad_pcb")
    # If True, omit KiCad oval slots that match Eurorack mounting slots
    # (useful if you prefer the scripted mounting hole pattern).
    drop_kicad_mount_slots: bool = False

    # Labels
    label_height: float = 0.4
    label_font: str = "Arial"
    label_font_style: FontStyle = FontStyle.BOLD
    # Match the established N8Synth defaults unless explicitly overridden.
    label_size: float = 3.2
    label_use_kicad_size: bool = False
    label_use_kicad_justify: bool = False
    # Optional scaling if you do choose to use KiCad sizes.
    label_scale: float = 1.0

    # Match N8Synth: labels are offset from the feature they describe.
    # (dx, dy) in mm applied to the hole center.
    label_offset: tuple[float, float] = (0.0, -7.0)

    # Optional: add a second set of labels on the opposite side of the same holes.
    # This is useful for putting text both below and above specific holes.
    # The second label uses the same X offset and a mirrored Y offset.
    label_above_enable: bool = False
    # Text to place above each hole (same ordering as `hole_labels` / `text_labels`).
    # Use "" for no above-label at that hole.
    hole_labels_above: tuple[str, ...] = tuple(text_labels_above)

    # Secondary (shift/alt) labels under the main label.
    # Used for CV_1..CV_4 (pots) which may have an alternate function depending on B8.
    # Set entries to "" to omit that secondary label.
    secondary_label_size: float = 2.1
    secondary_label_height: float = 0.4
    secondary_label_style: FontStyle = FontStyle.BOLD
    # Offset from the main label position (dx, dy) in mm.
    secondary_label_offset_from_main: tuple[float, float] = (0.0, -3.0)
    # Order is (CV_1, CV_2, CV_3, CV_4)
    cv_secondary_labels: tuple[str, str, str, str] = ("ALT1", "ALT2", "ALT3", "ALT4")

    # Inverse label style (like n8synth): a raised rounded rectangle plaque with the
    # text cut out, leaving the base color to show through.
    inverse_label_enable: bool = True
    # Any label whose text matches one of these strings will use the inverse style.
    # Common convention: invert outputs to distinguish them from inputs.
    inverse_labels: tuple[str, ...] = ("OUTR", "OUTL", "C10","B5","B6")
    inverse_corner_r: float = 0.8
    inverse_pad_x: float = 0.6
    inverse_pad_y: float = 0.6
    inverse_min_w: float = 0.0
    inverse_min_h: float = 0.0

    # Flip label placement/orientation by 180 degrees around each hole.
    # This addresses cases where the panel is effectively "upside down" relative
    # to the viewer/physical reference: the offset vector is negated and the text
    # is rotated 180 degrees.
    label_flip_180: bool = True

    # Default to hole-relative labels to avoid any KiCad mirroring/offset issues.
    # Use --labels-from kicad if you want to import F.SilkS text instead.
    label_source: LabelSource = "holes"

    # Editable label names for hole-relative labeling.
    # Ordering follows the internal hole ordering (top-to-bottom, left-to-right).
    # Keep this in sync with `text_labels` above.
    hole_labels: tuple[str, ...] = tuple(text_labels)

    # When label_source="holes": label only circles >= this diameter (mm)
    # to avoid eurorack mounting holes.
    label_hole_min_d: float = 5.0
    # Include oval holes (e.g. mounting slots) when generating labels.
    label_include_ovals: bool = False

    base_color: tuple[float, float, float] = (0.86, 0.86, 0.86)
    label_color: tuple[float, float, float] = (0.10, 0.10, 0.10)

    # Text rendering mode
    # - emboss: raised text (current behaviour)
    # - deboss: engraved text (subtracted from base)
    # - inlay: recessed pocket in base + separate inlay solid (for 2-color printing)
    text_mode: TextMode = "emboss"
    # Depth used for deboss/inlay (mm). With 0.2 mm layers, 0.4 mm = 2 layers.
    inlay_depth: float = 0.4

    # Branding (match n8synth conventions)
    brand_text_top: str = "patch.init()"
    brand_text_bottom: str = "84aW"
    brand_size: float = 2.6
    brand_height: float = 0.4
    brand_margin: float = 4.01

    # Export behavior
    # Many slicers (including Bambu Studio) auto-center each imported STL independently.
    # When exporting base + labels as separate STLs, this can cause XY misalignment.
    # If enabled, exported solids are translated so the panel center is at (0,0).
    export_centered: bool = True


def _export_transform(obj: object, params: FaceplateParams) -> object:
    if not params.export_centered:
        return obj
    try:
        # Keep Z unchanged; only unify XY reference.
        return obj.moved(Location((-params.panel_w / 2, -params.panel_h / 2, 0)))
    except Exception:
        return obj


def _slot_or_hole_2d(params: FaceplateParams) -> None:
    if params.mount_slot:
        SlotOverall(params.mount_slot_len, params.mount_hole_d)
    else:
        Circle(params.mount_hole_d / 2)


def _rotate_xy(dx: float, dy: float, rot_deg: float) -> tuple[float, float]:
    if abs(rot_deg) < 1e-12:
        return (dx, dy)
    a = math.radians(rot_deg)
    ca = math.cos(a)
    sa = math.sin(a)
    return (dx * ca - dy * sa, dx * sa + dy * ca)


def _secondary_label_for_main(params: FaceplateParams, main_txt: str) -> str:
    t = main_txt.strip()
    if t == "CV_1":
        return params.cv_secondary_labels[0].strip()
    if t == "CV_2":
        return params.cv_secondary_labels[1].strip()
    if t == "CV_3":
        return params.cv_secondary_labels[2].strip()
    if t == "CV_4":
        return params.cv_secondary_labels[3].strip()
    return ""


def _parse_kicad_fp_rects_edge_cuts(kicad_text: str) -> list[KiCadRect]:
    rects: list[KiCadRect] = []
    # KiCad 6+: fp_rect block within a footprint
    for m in __import__("re").finditer(
        r"\(fp_rect\s*\n\s*\(start\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\)\s*\n\s*\(end\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\)[\s\S]*?\(layer\s+\"Edge\.Cuts\"\)",
        kicad_text,
    ):
        x0, y0, x1, y1 = map(float, m.groups())
        rects.append(KiCadRect(x0, y0, x1, y1))
    return rects


def _find_panel_footprint_transform(kicad_text: str, *, panel_w: float, panel_h: float) -> tuple[float, float, float]:
    """Return (origin_x, origin_y, rot_deg) for the panel footprint.

    The Patch Init panel KiCad file stores the panel outline and cutouts as fp_* primitives
    inside a footprint. The footprint itself is placed at some (at x y [rot]) in board
    coordinates. Board-level gr_text uses board coordinates, while fp_rect/pads use
    footprint-local coordinates.

    To bring everything into the same panel-local coordinate system (origin at bottom-left),
    we subtract the footprint (at x y) and inverse-rotate by the footprint rotation.
    """
    import re

    w_s = f"{panel_w:g}"
    h_s = f"{panel_h:g}"

    fp_pat = re.compile(r"\(footprint[\s\S]*?\)\s*\n\)\s*", re.MULTILINE)
    for m in fp_pat.finditer(kicad_text):
        block = m.group(0)
        if "(fp_rect" not in block or '(layer "Edge.Cuts")' not in block:
            continue
        if f"(start 0 0)" not in block or f"(end {w_s} {h_s})" not in block:
            continue

        at = re.search(r"\(at\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)(?:\s+(-?\d+\.?\d*))?\)", block)
        if not at:
            continue
        ox = float(at.group(1))
        oy = float(at.group(2))
        rot = float(at.group(3) or 0.0)
        return (ox, oy, rot)

    # Fallback: assume already panel-local
    return (0.0, 0.0, 0.0)


def _board_to_local_xy(x: float, y: float, *, origin_x: float, origin_y: float, origin_rot_deg: float) -> tuple[float, float]:
    # local = R(-rot) * (board - origin)
    dx = x - origin_x
    dy = y - origin_y
    if abs(origin_rot_deg) < 1e-9:
        return (dx, dy)
    a = math.radians(-origin_rot_deg)
    ca = math.cos(a)
    sa = math.sin(a)
    return (dx * ca - dy * sa, dx * sa + dy * ca)


def _parse_kicad_np_thru_hole_pads(kicad_text: str) -> list[KiCadHole]:
    import re

    holes: list[KiCadHole] = []
    for m in re.finditer(
        r"\(pad\s+\"\"\s+np_thru_hole\s+(circle|oval)[\s\S]*?\(at\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)[^)]*\)[\s\S]*?\(drill\s+([^\)]+)\)",
        kicad_text,
    ):
        shape = m.group(1)
        x = float(m.group(2))
        y = float(m.group(3))
        drill_raw = m.group(4).strip()

        if drill_raw.startswith("oval"):
            _, major, minor = drill_raw.split()
            drill = (float(major), float(minor))
        else:
            d = float(drill_raw)
            drill = (d, d)

        holes.append(KiCadHole(shape=shape, x=x, y=y, drill=drill))

    return holes


def _parse_kicad_gr_text_f_silks(
    kicad_text: str,
    *,
    origin_x: float,
    origin_y: float,
    origin_rot_deg: float,
) -> list[KiCadText]:
    import re

    labels: list[KiCadText] = []

    # Avoid ingesting the enormous render_cache polygons; stop the match at (render_cache.
    # Not all KiCad exports include render_cache, so we fall back to a simpler match.
    pat_with_cache = re.compile(
        r"\(gr_text\s+\"(?P<txt>[^\"]*)\"(?P<body>[\s\S]*?)\n\s*\(render_cache",
        re.MULTILINE,
    )
    pat_no_cache = re.compile(
        r"\(gr_text\s+\"(?P<txt>[^\"]*)\"(?P<body>[\s\S]*?)\n\s*\)\s*\n",
        re.MULTILINE,
    )

    def parse_body(txt: str, body: str) -> None:
        if "(layer \"F.SilkS\")" not in body:
            return

        at_m = re.search(r"\(at\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)(?:\s+(-?\d+\.?\d*))?\)", body)
        if not at_m:
            return
        x_board = float(at_m.group(1))
        y_board = float(at_m.group(2))
        rot = float(at_m.group(3) or 0.0)

        x, y = _board_to_local_xy(
            x_board,
            y_board,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_rot_deg=origin_rot_deg,
        )

        size_m = re.search(r"\(size\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\)", body)
        size_mm = float(size_m.group(1)) if size_m else 1.5

        justify_h: Literal["left", "center", "right"] = "center"
        justify_v: Literal["bottom", "center", "top"] = "center"

        just_m = re.search(r"\(justify\s+([^\)]+)\)", body)
        if just_m:
            toks = just_m.group(1).split()
            for t in toks:
                if t in ("left", "center", "right"):
                    justify_h = t  # type: ignore[assignment]
                if t in ("top", "bottom"):
                    justify_v = t  # type: ignore[assignment]

        s = txt.strip()
        if not s:
            return

        labels.append(
            KiCadText(
                text=s,
                x=x,
                y=y,
                rot_deg=rot,
                size_mm=size_mm,
                justify_h=justify_h,
                justify_v=justify_v,
            )
        )

    # First pass: prefer matches that end at render_cache.
    for m in pat_with_cache.finditer(kicad_text):
        parse_body(m.group("txt"), m.group("body"))

    # If we got nothing, fall back to a rough block match.
    if not labels:
        for m in pat_no_cache.finditer(kicad_text):
            parse_body(m.group("txt"), m.group("body"))

    return labels


def load_kicad_panel(params: FaceplateParams) -> KiCadPanelImport:
    kicad_text = params.kicad_panel_path.read_text(errors="ignore")

    rects = _parse_kicad_fp_rects_edge_cuts(kicad_text)
    if not rects:
        raise ValueError(f"No Edge.Cuts fp_rect found in {params.kicad_panel_path}")

    # Pick the largest Edge.Cuts rectangle as the outline.
    outline = max(rects, key=lambda r: r.w * r.h)
    cutouts = tuple(r for r in rects if r is not outline)

    origin_x, origin_y, origin_rot_deg = _find_panel_footprint_transform(
        kicad_text,
        panel_w=params.panel_w,
        panel_h=params.panel_h,
    )

    holes = list(HARDCODED_HOLES)

    if params.drop_kicad_mount_slots:
        # Heuristic: drop oval 5x3 slots very near the panel corners.
        filtered: list[KiCadHole] = []
        for h in holes:
            if h.shape == "oval" and abs(h.drill[0] - 5.0) < 0.2 and abs(h.drill[1] - 3.0) < 0.2:
                # near (0,0) or (panel_w,panel_h)
                if (h.x < 15 and h.y < 15) or (h.x > params.panel_w - 15 and h.y > params.panel_h - 15):
                    continue
            filtered.append(h)
        holes = filtered

    labels = _parse_kicad_gr_text_f_silks(
        kicad_text,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_rot_deg=origin_rot_deg,
    )

    return KiCadPanelImport(
        origin_x=origin_x,
        origin_y=origin_y,
        origin_rot_deg=origin_rot_deg,
        outline=outline,
        cutouts=cutouts,
        holes=tuple(holes),
        labels=tuple(labels),
    )


def _text_anchor_xy_from_bbox(bb, *, h: str, v: str) -> tuple[float, float]:
    # build123d BoundingBox has min/max vectors with X/Y.
    x0 = bb.min.X
    x1 = bb.max.X
    y0 = bb.min.Y
    y1 = bb.max.Y

    if h == "left":
        ax = x0
    elif h == "right":
        ax = x1
    else:
        ax = (x0 + x1) / 2

    if v == "bottom":
        ay = y0
    elif v == "top":
        ay = y1
    else:
        ay = (y0 + y1) / 2

    return (ax, ay)


def _text_local_offset(
    txt: str,
    *,
    font: str,
    style: FontStyle,
    font_size: float,
    justify_h: str,
    justify_v: str,
) -> tuple[float, float]:
    # Create the text at the origin and compute an offset that moves the requested
    # anchor point (based on bbox) to the origin.
    with BuildSketch(Plane.XY) as sk:
        Text(txt, font_size=font_size, font=font, font_style=style)

    bb = sk.sketch.bounding_box()
    ax, ay = _text_anchor_xy_from_bbox(bb, h=justify_h, v=justify_v)
    return (-ax, -ay)


def _holes_for_labels(params: FaceplateParams, kicad: KiCadPanelImport) -> list[KiCadHole]:
    holes: list[KiCadHole] = []
    for h in kicad.holes:
        if h.shape == "oval" and not params.label_include_ovals:
            continue
        if h.shape == "circle" and h.drill[0] < params.label_hole_min_d:
            continue
        holes.append(h)

    # Stable ordering: top-to-bottom, left-to-right.
    holes.sort(key=lambda hh: (-hh.y, hh.x))
    return holes


def _brand_positions(params: FaceplateParams) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return (top_xyrot, bottom_xyrot) for branding text.

    If label_flip_180 is enabled (the common case for this panel), swap the top/bottom
    positions and rotate 180° so the text reads correctly in the same orientation as the
    hole labels.
    """
    x = params.panel_w / 2
    top_y = params.panel_h - params.brand_margin
    bot_y = params.brand_margin
    rot = 0.0
    if params.label_flip_180:
        top_y, bot_y = bot_y, top_y
        rot = 180.0
    return ((x, top_y, rot), (x, bot_y, rot))


def build_base(params: FaceplateParams, kicad: KiCadPanelImport) -> "object":
    cut_z0 = -0.2
    cut_h = params.thickness + 0.4

    with BuildPart() as p:
        Box(
            params.panel_w,
            params.panel_h,
            params.thickness,
            align=(Align.MIN, Align.MIN, Align.MIN),
            mode=Mode.ADD,
        )

        # Build all cutouts as a single sketch and extrude through.
        with BuildSketch(Plane.XY.offset(cut_z0)) as sk:
            if params.add_mount_holes:
                mount_points = [
                    (params.mount_x_from_edge, params.mount_y_from_edge),
                    (params.mount_x_from_edge, params.panel_h - params.mount_y_from_edge),
                ]
                if params.mount_both_sides:
                    x2 = params.panel_w - params.mount_x_from_edge
                    mount_points += [
                        (x2, params.mount_y_from_edge),
                        (x2, params.panel_h - params.mount_y_from_edge),
                    ]
                with Locations(*mount_points):
                    _slot_or_hole_2d(params)

            for r in kicad.cutouts:
                with Locations((r.min_x, r.min_y)):
                    Rectangle(r.w, r.h, align=(Align.MIN, Align.MIN))

            for h in kicad.holes:
                with Locations((h.x, h.y)):
                    if h.shape == "circle":
                        Circle(h.drill[0] / 2)
                    else:
                        major, minor = h.drill
                        SlotOverall(major, minor)

        extrude(to_extrude=sk.sketch, amount=cut_h, mode=Mode.SUBTRACT)

        # Optional recessed text pocket (for deboss/inlay modes)
        if params.text_mode in ("deboss", "inlay"):
            pocket_depth = max(0.0, float(params.inlay_depth))
            if pocket_depth > 0:
                z0 = params.thickness - pocket_depth

                def add_text(
                    txt: str,
                    *,
                    at_xy: tuple[float, float],
                    rot_deg: float,
                    font_size: float,
                    style: FontStyle,
                    justify_h: Literal["left", "center", "right"],
                    justify_v: Literal["bottom", "center", "top"],
                ) -> None:
                    dx, dy = _text_local_offset(
                        txt,
                        font=params.label_font,
                        style=style,
                        font_size=font_size,
                        justify_h=justify_h,
                        justify_v=justify_v,
                    )
                    with Locations(Location((at_xy[0], at_xy[1], 0), (0, 0, rot_deg))):
                        with Locations((dx, dy)):
                            Text(txt, font_size=font_size, font=params.label_font, font_style=style)

                with BuildSketch(Plane.XY.offset(z0)) as pocket_sk:
                    # Branding
                    (top_x, top_y, top_rot), (bot_x, bot_y, bot_rot) = _brand_positions(params)
                    if params.brand_text_top.strip():
                        add_text(
                            params.brand_text_top.strip(),
                            at_xy=(top_x, top_y),
                            rot_deg=top_rot,
                            font_size=params.brand_size,
                            style=params.label_font_style,
                            justify_h="center",
                            justify_v="center",
                        )
                    if params.brand_text_bottom.strip():
                        add_text(
                            params.brand_text_bottom.strip(),
                            at_xy=(bot_x, bot_y),
                            rot_deg=bot_rot,
                            font_size=params.brand_size,
                            style=params.label_font_style,
                            justify_h="center",
                            justify_v="center",
                        )

                    # Main labels (no inverse plaques in pocket)
                    if params.label_source == "kicad":
                        for lab in kicad.labels:
                            s = lab.text.strip()
                            if not s:
                                continue
                            if params.label_use_kicad_size:
                                size = max(0.5, lab.size_mm * params.label_scale)
                            else:
                                size = params.label_size

                            justify_h = lab.justify_h if params.label_use_kicad_justify else "center"
                            justify_v = lab.justify_v if params.label_use_kicad_justify else "center"

                            add_text(
                                s,
                                at_xy=(lab.x, lab.y),
                                rot_deg=lab.rot_deg,
                                font_size=size,
                                style=params.label_font_style,
                                justify_h=justify_h,  # type: ignore[arg-type]
                                justify_v=justify_v,  # type: ignore[arg-type]
                            )

                            sec = _secondary_label_for_main(params, s)
                            if sec:
                                sec_dx, sec_dy = params.secondary_label_offset_from_main
                                odx, ody = _rotate_xy(sec_dx, sec_dy, lab.rot_deg)
                                add_text(
                                    sec,
                                    at_xy=(lab.x + odx, lab.y + ody),
                                    rot_deg=lab.rot_deg,
                                    font_size=params.secondary_label_size,
                                    style=params.secondary_label_style,
                                    justify_h=justify_h,  # type: ignore[arg-type]
                                    justify_v=justify_v,  # type: ignore[arg-type]
                                )
                    else:
                        dx_off, dy_off = params.label_offset
                        rot_extra = 0.0
                        if params.label_flip_180:
                            dx_off, dy_off = (-dx_off, -dy_off)
                            rot_extra = 180.0
                        holes = _holes_for_labels(params, kicad)

                        if params.label_above_enable and len(params.hole_labels_above) < len(holes):
                            raise ValueError(
                                f"hole_labels_above has {len(params.hole_labels_above)} entries but needs at least {len(holes)} for the current hole set. "
                                "Provide one above-label per hole (use '' for none)."
                            )
                        for idx, h in enumerate(holes, start=1):
                            if idx - 1 >= len(params.hole_labels):
                                break
                            s = params.hole_labels[idx - 1].strip()
                            if not s:
                                continue
                            add_text(
                                s,
                                at_xy=(h.x + dx_off, h.y + dy_off),
                                rot_deg=rot_extra,
                                font_size=params.label_size,
                                style=params.label_font_style,
                                justify_h="center",
                                justify_v="center",
                            )
                            if params.label_above_enable and abs(dy_off) > 1e-6:
                                above_txt = params.hole_labels_above[idx - 1].strip()
                                if above_txt:
                                    add_text(
                                        above_txt,
                                        at_xy=(h.x + dx_off, h.y - dy_off),
                                        rot_deg=rot_extra,
                                        font_size=params.label_size,
                                        justify_h="center",
                                        justify_v="center",
                                    )

                            sec = _secondary_label_for_main(params, s)
                            if sec:
                                sec_dx, sec_dy = params.secondary_label_offset_from_main
                                odx, ody = _rotate_xy(sec_dx, sec_dy, rot_extra)
                                add_text(
                                    sec,
                                    at_xy=(h.x + dx_off + odx, h.y + dy_off + ody),
                                    rot_deg=rot_extra,
                                    font_size=params.secondary_label_size,
                                    style=params.secondary_label_style,
                                    justify_h="center",
                                    justify_v="center",
                                )

                extrude(to_extrude=pocket_sk.sketch, amount=pocket_depth + 0.1, mode=Mode.SUBTRACT)

    return p.part


def build_labels(params: FaceplateParams, kicad: KiCadPanelImport) -> "object":
    # In deboss mode there is no separate print part, but returning the inlay volume
    # is still useful for preview/debugging.
    if params.text_mode in ("deboss", "inlay"):
        depth = max(0.0, float(params.inlay_depth))
        if depth <= 0:
            with BuildPart() as p:
                return p.part

        z0 = params.thickness - depth

        with BuildPart() as p:
            def add_text(
                txt: str,
                *,
                at_xy: tuple[float, float],
                rot_deg: float,
                font_size: float,
                style: FontStyle,
                justify_h: Literal["left", "center", "right"],
                justify_v: Literal["bottom", "center", "top"],
            ) -> None:
                dx, dy = _text_local_offset(
                    txt,
                    font=params.label_font,
                    style=style,
                    font_size=font_size,
                    justify_h=justify_h,
                    justify_v=justify_v,
                )
                with BuildSketch(Plane.XY.offset(z0)) as sk:
                    with Locations(Location((at_xy[0], at_xy[1], 0), (0, 0, rot_deg))):
                        with Locations((dx, dy)):
                            Text(txt, font_size=font_size, font=params.label_font, font_style=style)
                extrude(to_extrude=sk.sketch, amount=depth, mode=Mode.ADD)

            # Branding
            (top_x, top_y, top_rot), (bot_x, bot_y, bot_rot) = _brand_positions(params)
            if params.brand_text_top.strip():
                add_text(
                    params.brand_text_top.strip(),
                    at_xy=(top_x, top_y),
                    rot_deg=top_rot,
                    font_size=params.brand_size,
                    style=params.label_font_style,
                    justify_h="center",
                    justify_v="center",
                )
            if params.brand_text_bottom.strip():
                add_text(
                    params.brand_text_bottom.strip(),
                    at_xy=(bot_x, bot_y),
                    rot_deg=bot_rot,
                    font_size=params.brand_size,
                    style=params.label_font_style,
                    justify_h="center",
                    justify_v="center",
                )

            if params.label_source == "kicad":
                for lab in kicad.labels:
                    s = lab.text.strip()
                    if not s:
                        continue
                    if params.label_use_kicad_size:
                        size = max(0.5, lab.size_mm * params.label_scale)
                    else:
                        size = params.label_size

                    justify_h = lab.justify_h if params.label_use_kicad_justify else "center"
                    justify_v = lab.justify_v if params.label_use_kicad_justify else "center"

                    add_text(
                        s,
                        at_xy=(lab.x, lab.y),
                        rot_deg=lab.rot_deg,
                        font_size=size,
                        style=params.label_font_style,
                        justify_h=justify_h,  # type: ignore[arg-type]
                        justify_v=justify_v,  # type: ignore[arg-type]
                    )

                    sec = _secondary_label_for_main(params, s)
                    if sec:
                        sec_dx, sec_dy = params.secondary_label_offset_from_main
                        odx, ody = _rotate_xy(sec_dx, sec_dy, lab.rot_deg)
                        add_text(
                            sec,
                            at_xy=(lab.x + odx, lab.y + ody),
                            rot_deg=lab.rot_deg,
                            font_size=params.secondary_label_size,
                            style=params.secondary_label_style,
                            justify_h=justify_h,  # type: ignore[arg-type]
                            justify_v=justify_v,  # type: ignore[arg-type]
                        )
            else:
                dx_off, dy_off = params.label_offset
                rot_extra = 0.0
                if params.label_flip_180:
                    dx_off, dy_off = (-dx_off, -dy_off)
                    rot_extra = 180.0
                holes = _holes_for_labels(params, kicad)

                if params.label_above_enable and len(params.hole_labels_above) < len(holes):
                    raise ValueError(
                        f"hole_labels_above has {len(params.hole_labels_above)} entries but needs at least {len(holes)} for the current hole set. "
                        "Provide one above-label per hole (use '' for none)."
                    )
                for idx, h in enumerate(holes, start=1):
                    if idx - 1 >= len(params.hole_labels):
                        break
                    s = params.hole_labels[idx - 1].strip()
                    if not s:
                        continue
                    add_text(
                        s,
                        at_xy=(h.x + dx_off, h.y + dy_off),
                        rot_deg=rot_extra,
                        font_size=params.label_size,
                        style=params.label_font_style,
                        justify_h="center",
                        justify_v="center",
                    )
                    if params.label_above_enable and abs(dy_off) > 1e-6:
                        above_txt = params.hole_labels_above[idx - 1].strip()
                        if above_txt:
                            add_text(
                                above_txt,
                                at_xy=(h.x + dx_off, h.y - dy_off),
                                rot_deg=rot_extra,
                                font_size=params.label_size,
                                justify_h="center",
                                justify_v="center",
                            )

                    sec = _secondary_label_for_main(params, s)
                    if sec:
                        sec_dx, sec_dy = params.secondary_label_offset_from_main
                        odx, ody = _rotate_xy(sec_dx, sec_dy, rot_extra)
                        add_text(
                            sec,
                            at_xy=(h.x + dx_off + odx, h.y + dy_off + ody),
                            rot_deg=rot_extra,
                            font_size=params.secondary_label_size,
                            style=params.secondary_label_style,
                            justify_h="center",
                            justify_v="center",
                        )

        return p.part

    inverse_set = {s.strip() for s in params.inverse_labels if s.strip()}

    def _is_inverse_label(txt: str) -> bool:
        if not params.inverse_label_enable:
            return False
        return txt.strip() in inverse_set

    def _make_text_sketch(
        txt: str,
        *,
        font_size: float,
        style: FontStyle,
        at_xy: tuple[float, float],
        rot_deg: float,
        justify_h: Literal["left", "center", "right"],
        justify_v: Literal["bottom", "center", "top"],
    ):
        dx, dy = _text_local_offset(
            txt,
            font=params.label_font,
            style=style,
            font_size=font_size,
            justify_h=justify_h,
            justify_v=justify_v,
        )
        with BuildSketch(Plane.XY.offset(params.thickness)) as sk:
            with Locations(Location((at_xy[0], at_xy[1], 0), (0, 0, rot_deg))):
                with Locations((dx, dy)):
                    Text(txt, font_size=font_size, font=params.label_font, font_style=style)
        return sk.sketch

    def _plaque_size_for_text(txt: str, *, font_size: float, style: FontStyle) -> tuple[float, float]:
        # Use an unrotated sketch to estimate bounds in the label's local frame.
        # This matches n8synth and is sufficient for this panel where rotations are 0/180.
        sk = _make_text_sketch(
            txt,
            font_size=font_size,
            style=style,
            at_xy=(0.0, 0.0),
            rot_deg=0.0,
            justify_h="center",
            justify_v="center",
        )
        bb = sk.bounding_box().size
        w = max(params.inverse_min_w, bb.X + 2 * params.inverse_pad_x)
        h = max(params.inverse_min_h, bb.Y + 2 * params.inverse_pad_y)
        return (w, h)

    def _add_inverse_label(
        txt: str,
        *,
        at_xy: tuple[float, float],
        rot_deg: float,
        font_size: float,
        style: FontStyle,
        height: float,
    ) -> None:
        # 1) Add plaque
        w, h = _plaque_size_for_text(txt, font_size=font_size, style=style)
        with BuildSketch(Plane.XY.offset(params.thickness)) as plaque_sk:
            with Locations(Location((at_xy[0], at_xy[1], 0), (0, 0, rot_deg))):
                RectangleRounded(w, h, params.inverse_corner_r)
        extrude(to_extrude=plaque_sk.sketch, amount=height, mode=Mode.ADD)

        # 2) Cut text out of the plaque
        txt_sk = _make_text_sketch(
            txt,
            font_size=font_size,
            style=style,
            at_xy=at_xy,
            rot_deg=rot_deg,
            justify_h="center",
            justify_v="center",
        )
        extrude(to_extrude=txt_sk, amount=height + 0.1, mode=Mode.SUBTRACT)

    with BuildPart() as p:
        # Branding (always present; matches n8synth style)
        (top_x, top_y, top_rot), (bot_x, bot_y, bot_rot) = _brand_positions(params)
        for txt, x, y, rot in (
            (params.brand_text_top, top_x, top_y, top_rot),
            (params.brand_text_bottom, bot_x, bot_y, bot_rot),
        ):
            if txt.strip():
                dx, dy = _text_local_offset(
                    txt,
                    font=params.label_font,
                    style=params.label_font_style,
                    font_size=params.brand_size,
                    justify_h="center",
                    justify_v="center",
                )
                with BuildSketch(Plane.XY.offset(params.thickness)) as sk:
                    with Locations(Location((x, y, 0), (0, 0, rot))):
                        with Locations((dx, dy)):
                            Text(
                                txt,
                                font_size=params.brand_size,
                                font=params.label_font,
                                font_style=params.label_font_style,
                            )
                extrude(to_extrude=sk.sketch, amount=params.brand_height, mode=Mode.ADD)

        if params.label_source == "kicad":
            for lab in kicad.labels:
                if not lab.text.strip():
                    continue
                if params.label_use_kicad_size:
                    size = max(0.5, lab.size_mm * params.label_scale)
                else:
                    size = params.label_size

                justify_h = lab.justify_h if params.label_use_kicad_justify else "center"
                justify_v = lab.justify_v if params.label_use_kicad_justify else "center"

                if _is_inverse_label(lab.text):
                    _add_inverse_label(
                        lab.text.strip(),
                        at_xy=(lab.x, lab.y),
                        rot_deg=lab.rot_deg,
                        font_size=size,
                        style=params.label_font_style,
                        height=params.label_height,
                    )
                else:
                    sk = _make_text_sketch(
                        lab.text,
                        font_size=size,
                        style=params.label_font_style,
                        at_xy=(lab.x, lab.y),
                        rot_deg=lab.rot_deg,
                        justify_h=justify_h,
                        justify_v=justify_v,
                    )
                    extrude(to_extrude=sk, amount=params.label_height, mode=Mode.ADD)

                sec = _secondary_label_for_main(params, lab.text)
                if sec:
                    sec_dx, sec_dy = params.secondary_label_offset_from_main
                    odx, ody = _rotate_xy(sec_dx, sec_dy, lab.rot_deg)
                    sec_sk = _make_text_sketch(
                        sec,
                        font_size=params.secondary_label_size,
                        style=params.secondary_label_style,
                        at_xy=(lab.x + odx, lab.y + ody),
                        rot_deg=lab.rot_deg,
                        justify_h=justify_h,
                        justify_v=justify_v,
                    )
                    extrude(to_extrude=sec_sk, amount=params.secondary_label_height, mode=Mode.ADD)
        else:
            dx_off, dy_off = params.label_offset
            rot_extra = 0.0
            if params.label_flip_180:
                dx_off, dy_off = (-dx_off, -dy_off)
                rot_extra = 180.0
            holes = _holes_for_labels(params, kicad)

            if params.label_above_enable and len(params.hole_labels_above) < len(holes):
                raise ValueError(
                    f"hole_labels_above has {len(params.hole_labels_above)} entries but needs at least {len(holes)} for the current hole set. "
                    "Provide one above-label per hole (use '' for none)."
                )

            if len(params.hole_labels) < len(holes):
                raise ValueError(
                    f"hole_labels has {len(params.hole_labels)} entries but needs at least {len(holes)} for the current hole set. "
                    "Edit text_labels near the top of this file to provide one label per hole (in top-to-bottom, left-to-right order)."
                )

            for idx, h in enumerate(holes, start=1):
                txt = params.hole_labels[idx - 1].strip()
                if not txt:
                    continue
                at_xy = (h.x + dx_off, h.y + dy_off)
                if _is_inverse_label(txt):
                    _add_inverse_label(
                        txt,
                        at_xy=at_xy,
                        rot_deg=rot_extra,
                        font_size=params.label_size,
                        style=params.label_font_style,
                        height=params.label_height,
                    )
                else:
                    sk = _make_text_sketch(
                        txt,
                        font_size=params.label_size,
                        style=params.label_font_style,
                        at_xy=at_xy,
                        rot_deg=rot_extra,
                        justify_h="center",
                        justify_v="center",
                    )
                    extrude(to_extrude=sk, amount=params.label_height, mode=Mode.ADD)

                if params.label_above_enable and abs(dy_off) > 1e-6:
                    above_txt = params.hole_labels_above[idx - 1].strip()
                    if above_txt:
                        at_xy2 = (h.x + dx_off, h.y - dy_off)
                        if _is_inverse_label(above_txt):
                            _add_inverse_label(
                                above_txt,
                                at_xy=at_xy2,
                                rot_deg=rot_extra,
                                font_size=params.label_size,
                                style=params.label_font_style,
                                height=params.label_height,
                            )
                        else:
                            sk2 = _make_text_sketch(
                                above_txt,
                                font_size=params.label_size,
                                style=params.label_font_style,
                                at_xy=at_xy2,
                                rot_deg=rot_extra,
                                justify_h="center",
                                justify_v="center",
                            )
                            extrude(to_extrude=sk2, amount=params.label_height, mode=Mode.ADD)

                sec = _secondary_label_for_main(params, txt)
                if sec:
                    sec_dx, sec_dy = params.secondary_label_offset_from_main
                    odx, ody = _rotate_xy(sec_dx, sec_dy, rot_extra)
                    sec_sk = _make_text_sketch(
                        sec,
                        font_size=params.secondary_label_size,
                        style=params.secondary_label_style,
                        at_xy=(at_xy[0] + odx, at_xy[1] + ody),
                        rot_deg=rot_extra,
                        justify_h="center",
                        justify_v="center",
                    )
                    extrude(to_extrude=sec_sk, amount=params.secondary_label_height, mode=Mode.ADD)

    return p.part


def build_faceplate(params: FaceplateParams, *, export_mode: ExportMode = "combined") -> tuple[object | None, object | None]:
    kicad = load_kicad_panel(params)

    # Basic sanity check: warn if KiCad outline doesn't match the expected 10HP panel.
    if abs(kicad.outline.w - params.panel_w) > 0.2 or abs(kicad.outline.h - params.panel_h) > 0.2:
        raise ValueError(
            "KiCad outline does not match FaceplateParams panel size: "
            f"kicad=({kicad.outline.w:.3f}x{kicad.outline.h:.3f}) vs params=({params.panel_w:.3f}x{params.panel_h:.3f}). "
            "Update FaceplateParams.panel_w/panel_h or point --kicad to the correct panel file."
        )

    base = build_base(params, kicad) if export_mode in ("combined", "base") else None
    labels = build_labels(params, kicad) if export_mode in ("combined", "labels") else None
    return base, labels


def export_print_template(params: FaceplateParams, *, svg: Path | None, dxf: Path | None) -> None:
    kicad = load_kicad_panel(params)

    def make_layer_outline():
        with BuildSketch(Plane.XY) as sk:
            Rectangle(params.panel_w, params.panel_h, align=(Align.MIN, Align.MIN))
        return sk.sketch

    def make_layer_cutouts():
        with BuildSketch(Plane.XY) as sk:
            # Optional scripted mounting holes
            if params.add_mount_holes:
                mount_points: list[tuple[float, float]] = [
                    (params.mount_x_from_edge, params.mount_y_from_edge),
                    (params.mount_x_from_edge, params.panel_h - params.mount_y_from_edge),
                ]
                if params.mount_both_sides:
                    x2 = params.panel_w - params.mount_x_from_edge
                    mount_points += [
                        (x2, params.mount_y_from_edge),
                        (x2, params.panel_h - params.mount_y_from_edge),
                    ]
                with Locations(*mount_points):
                    _slot_or_hole_2d(params)

            for r in kicad.cutouts:
                with Locations((r.min_x, r.min_y)):
                    Rectangle(r.w, r.h, align=(Align.MIN, Align.MIN))

            for h in kicad.holes:
                with Locations((h.x, h.y)):
                    if h.shape == "circle":
                        Circle(h.drill[0] / 2)
                    else:
                        major, minor = h.drill
                        SlotOverall(major, minor)

        return sk.sketch

    def make_layer_labels():
        sketches: list[object] = []

        inverse_set = {s.strip() for s in params.inverse_labels if s.strip()}

        def is_inverse(txt: str) -> bool:
            return params.inverse_label_enable and txt.strip() in inverse_set

        def text_sketch(
            txt: str,
            *,
            at_xy: tuple[float, float],
            rot_deg: float,
            font_size: float,
            justify_h: Literal["left", "center", "right"],
            justify_v: Literal["bottom", "center", "top"],
        ):
            dx, dy = _text_local_offset(
                txt,
                font=params.label_font,
                style=params.label_font_style,
                font_size=font_size,
                justify_h=justify_h,
                justify_v=justify_v,
            )
            with BuildSketch(Plane.XY) as sk:
                with Locations(Location((at_xy[0], at_xy[1], 0), (0, 0, rot_deg))):
                    with Locations((dx, dy)):
                        Text(txt, font_size=font_size, font=params.label_font, font_style=params.label_font_style)
            return sk.sketch

        def plaque_sketch(*, at_xy: tuple[float, float], rot_deg: float, w: float, h: float):
            with BuildSketch(Plane.XY) as sk:
                with Locations(Location((at_xy[0], at_xy[1], 0), (0, 0, rot_deg))):
                    RectangleRounded(w, h, params.inverse_corner_r)
            return sk.sketch

        # Branding
        (top_x, top_y, top_rot), (bot_x, bot_y, bot_rot) = _brand_positions(params)
        for txt, x, y, rot in (
            (params.brand_text_top, top_x, top_y, top_rot),
            (params.brand_text_bottom, bot_x, bot_y, bot_rot),
        ):
            if txt.strip():
                dx, dy = _text_local_offset(
                    txt,
                    font=params.label_font,
                    style=params.label_font_style,
                    font_size=params.brand_size,
                    justify_h="center",
                    justify_v="center",
                )
                with BuildSketch(Plane.XY) as sk:
                    with Locations(Location((x, y, 0), (0, 0, rot))):
                        with Locations((dx, dy)):
                            Text(
                                txt,
                                font_size=params.brand_size,
                                font=params.label_font,
                                font_style=params.label_font_style,
                            )
                sketches.append(sk.sketch)

        if params.label_source == "kicad":
            for lab in kicad.labels:
                if not lab.text.strip():
                    continue
                if params.label_use_kicad_size:
                    size = max(0.5, lab.size_mm * params.label_scale)
                else:
                    size = params.label_size

                justify_h = lab.justify_h if params.label_use_kicad_justify else "center"
                justify_v = lab.justify_v if params.label_use_kicad_justify else "center"

                tsk = text_sketch(
                    lab.text,
                    at_xy=(lab.x, lab.y),
                    rot_deg=lab.rot_deg,
                    font_size=size,
                    justify_h=justify_h,
                    justify_v=justify_v,
                )
                sketches.append(tsk)

                sec = _secondary_label_for_main(params, lab.text)
                if sec:
                    sec_dx, sec_dy = params.secondary_label_offset_from_main
                    odx, ody = _rotate_xy(sec_dx, sec_dy, lab.rot_deg)
                    sketches.append(
                        text_sketch(
                            sec,
                            at_xy=(lab.x + odx, lab.y + ody),
                            rot_deg=lab.rot_deg,
                            font_size=params.secondary_label_size,
                            justify_h=justify_h,
                            justify_v=justify_v,
                        )
                    )

                if is_inverse(lab.text):
                    bb = tsk.bounding_box().size
                    w = max(params.inverse_min_w, bb.X + 2 * params.inverse_pad_x)
                    h = max(params.inverse_min_h, bb.Y + 2 * params.inverse_pad_y)
                    sketches.append(plaque_sketch(at_xy=(lab.x, lab.y), rot_deg=lab.rot_deg, w=w, h=h))
        else:
            dx_off, dy_off = params.label_offset
            rot_extra = 0.0
            if params.label_flip_180:
                dx_off, dy_off = (-dx_off, -dy_off)
                rot_extra = 180.0
            holes = _holes_for_labels(params, kicad)

            if params.label_above_enable and len(params.hole_labels_above) < len(holes):
                raise ValueError(
                    f"hole_labels_above has {len(params.hole_labels_above)} entries but needs at least {len(holes)} for the current hole set. "
                    "Provide one above-label per hole (use '' for none)."
                )

            if len(params.hole_labels) < len(holes):
                raise ValueError(
                    f"hole_labels has {len(params.hole_labels)} entries but needs at least {len(holes)} for the current hole set. "
                    "Edit text_labels near the top of this file to provide one label per hole (in top-to-bottom, left-to-right order)."
                )

            for idx, h in enumerate(holes, start=1):
                txt = params.hole_labels[idx - 1].strip()
                if not txt:
                    continue
                at_xy = (h.x + dx_off, h.y + dy_off)
                tsk = text_sketch(
                    txt,
                    at_xy=at_xy,
                    rot_deg=rot_extra,
                    font_size=params.label_size,
                    justify_h="center",
                    justify_v="center",
                )
                sketches.append(tsk)

                sec = _secondary_label_for_main(params, txt)
                if sec:
                    sec_dx, sec_dy = params.secondary_label_offset_from_main
                    odx, ody = _rotate_xy(sec_dx, sec_dy, rot_extra)
                    sketches.append(
                        text_sketch(
                            sec,
                            at_xy=(at_xy[0] + odx, at_xy[1] + ody),
                            rot_deg=rot_extra,
                            font_size=params.secondary_label_size,
                            justify_h="center",
                            justify_v="center",
                        )
                    )

                if is_inverse(txt):
                    bb = tsk.bounding_box().size
                    w = max(params.inverse_min_w, bb.X + 2 * params.inverse_pad_x)
                    h = max(params.inverse_min_h, bb.Y + 2 * params.inverse_pad_y)
                    sketches.append(plaque_sketch(at_xy=at_xy, rot_deg=rot_extra, w=w, h=h))

                if params.label_above_enable and abs(dy_off) > 1e-6:
                    above_txt = params.hole_labels_above[idx - 1].strip()
                    if not above_txt:
                        continue
                    at_xy2 = (h.x + dx_off, h.y - dy_off)
                    tsk2 = text_sketch(
                        above_txt,
                        at_xy=at_xy2,
                        rot_deg=rot_extra,
                        font_size=params.label_size,
                        justify_h="center",
                        justify_v="center",
                    )
                    sketches.append(tsk2)
                    if is_inverse(above_txt):
                        bb2 = tsk2.bounding_box().size
                        w2 = max(params.inverse_min_w, bb2.X + 2 * params.inverse_pad_x)
                        h2 = max(params.inverse_min_h, bb2.Y + 2 * params.inverse_pad_y)
                        sketches.append(plaque_sketch(at_xy=at_xy2, rot_deg=rot_extra, w=w2, h=h2))
        return sketches

    def make_layer_calibration():
        x0 = params.panel_w + 6
        y0 = 6
        with BuildSketch(Plane.XY) as sk:
            with Locations((x0, y0)):
                Rectangle(10, 10, align=(Align.MIN, Align.MIN))
        return sk.sketch

    outline = make_layer_outline()
    cutouts = make_layer_cutouts()
    labels = make_layer_labels()
    calib = make_layer_calibration()

    if svg is not None:
        exp = ExportSVG(margin=5, line_weight=0.18)
        exp.add_layer("outline")
        exp.add_layer("cutouts")
        exp.add_layer("labels")
        exp.add_layer("calibration")
        exp.add_shape(outline, layer="outline")
        exp.add_shape(cutouts, layer="cutouts")
        exp.add_shape(labels, layer="labels")
        exp.add_shape(calib, layer="calibration")
        exp.write(svg)

    if dxf is not None:
        exp = ExportDXF()
        exp.add_layer("outline")
        exp.add_layer("cutouts")
        exp.add_layer("labels")
        exp.add_layer("calibration")
        exp.add_shape(outline, layer="outline")
        exp.add_shape(cutouts, layer="cutouts")
        exp.add_shape(labels, layer="labels")
        exp.add_shape(calib, layer="calibration")
        exp.write(dxf)


def main() -> None:
    import argparse
    from dataclasses import replace

    parser = argparse.ArgumentParser(description="Build Daisy Patch Init 10HP faceplate from KiCad panel file")
    parser.add_argument(
        "--export-mode",
        choices=("combined", "base", "labels"),
        default="combined",
        help="Which solids to generate",
    )
    parser.add_argument("--stl", type=Path, default=None, help="Export STL to this path")
    parser.add_argument(
        "--stl-base",
        type=Path,
        default=None,
        help="Export base STL to this path (recommended for 2-part printing workflows)",
    )
    parser.add_argument(
        "--stl-labels",
        type=Path,
        default=None,
        help="Export labels/inlay STL to this path (recommended for 2-part printing workflows)",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=None,
        help="Panel thickness in mm (default: 3.0)",
    )
    parser.add_argument(
        "--label-height",
        type=float,
        default=None,
        help="Height of raised labels in mm (default: 0.4). Should be a multiple of your layer height.",
    )
    parser.add_argument(
        "--labels-above-cv-b8",
        action="store_true",
        help="Also place labels above CV_1..CV_4 and B8 using the mirrored Y offset",
    )
    parser.add_argument("--step", type=Path, default=None, help="Export STEP to this path")
    parser.add_argument(
        "--text-mode",
        choices=("emboss", "deboss", "inlay"),
        default="emboss",
        help="How to render text: emboss (raised), deboss (engraved), inlay (recess + separate inlay solid)",
    )
    parser.add_argument(
        "--inlay-depth",
        type=float,
        default=0.4,
        help="Depth in mm used for deboss/inlay modes (e.g. 0.4 = 2 layers at 0.2 mm)",
    )
    parser.add_argument(
        "--template-svg",
        type=Path,
        default=None,
        help="Export a 1:1 SVG template (outline/cutouts/labels) for paper printing",
    )
    parser.add_argument(
        "--template-dxf",
        type=Path,
        default=None,
        help="Export a 1:1 DXF template (outline/cutouts/labels) for paper printing",
    )
    parser.add_argument(
        "--kicad",
        type=Path,
        default=None,
        help="Override the KiCad panel file path",
    )
    parser.add_argument(
        "--label-scale",
        type=float,
        default=None,
        help="Scale imported KiCad font sizes (e.g. 1.2)",
    )
    parser.add_argument(
        "--label-size",
        type=float,
        default=None,
        help="Use a fixed label font size in mm (defaults to the n8synth convention)",
    )
    parser.add_argument(
        "--use-kicad-label-size",
        action="store_true",
        help="Use the KiCad per-label font sizes (optionally scaled by --label-scale)",
    )
    parser.add_argument(
        "--use-kicad-label-justify",
        action="store_true",
        help="Use KiCad label justification (otherwise labels are centered like the n8synth template)",
    )
    parser.add_argument(
        "--labels-from",
        choices=("holes", "kicad"),
        default=None,
        help="Label source: 'holes' generates placeholder A/B/C... positioned using --label-offset; 'kicad' imports F.SilkS gr_text",
    )
    parser.add_argument(
        "--label-offset",
        nargs=2,
        type=float,
        default=None,
        metavar=("DX", "DY"),
        help="Label offset from hole center in mm (defaults to n8synth: 0 -7)",
    )
    parser.add_argument(
        "--flip-labels",
        action="store_true",
        help="Flip hole-relative labels by 180° (negate offset + rotate text). Enabled by default in params.",
    )
    parser.add_argument(
        "--no-flip-labels",
        action="store_true",
        help="Disable the default 180° label flip.",
    )
    parser.add_argument(
        "--use-script-mount-holes",
        action="store_true",
        help="Generate Eurorack mounting holes using the script (and drop KiCad corner slots)",
    )
    parser.add_argument(
        "--no-export-centered",
        action="store_true",
        help="Do not translate exported STL/STEP to a common centered origin",
    )

    args = parser.parse_args()

    params = replace(FaceplateParams(), text_mode=args.text_mode, inlay_depth=float(args.inlay_depth))
    if args.no_export_centered:
        params = replace(params, export_centered=False)

    if args.thickness is not None:
        params = replace(params, thickness=args.thickness)
    if args.label_height is not None:
        params = replace(params, label_height=args.label_height)

    # In non-emboss modes, inverse plaques aren't meaningful for 2-color inlay/deboss.
    if params.text_mode in ("deboss", "inlay") and params.inverse_label_enable:
        params = replace(params, inverse_label_enable=False)
    if args.kicad is not None:
        params = FaceplateParams(**{**params.__dict__, "kicad_panel_path": args.kicad})
    if args.label_scale is not None:
        params = FaceplateParams(**{**params.__dict__, "label_scale": args.label_scale})
    if args.label_size is not None:
        params = FaceplateParams(**{**params.__dict__, "label_size": args.label_size})
    if args.use_kicad_label_size:
        params = FaceplateParams(**{**params.__dict__, "label_use_kicad_size": True})
    if args.use_kicad_label_justify:
        params = FaceplateParams(**{**params.__dict__, "label_use_kicad_justify": True})
    if args.labels_from is not None:
        params = FaceplateParams(**{**params.__dict__, "label_source": args.labels_from})
    if args.label_offset is not None:
        params = FaceplateParams(**{**params.__dict__, "label_offset": (args.label_offset[0], args.label_offset[1])})
    if args.labels_above_cv_b8:
        params = FaceplateParams(**{**params.__dict__, "label_above_enable": True})
    if args.flip_labels:
        params = FaceplateParams(**{**params.__dict__, "label_flip_180": True})
    if args.no_flip_labels:
        params = FaceplateParams(**{**params.__dict__, "label_flip_180": False})
    if args.use_script_mount_holes:
        params = FaceplateParams(
            **{
                **params.__dict__,
                "add_mount_holes": True,
                "drop_kicad_mount_slots": True,
            }
        )

    if args.template_svg is not None or args.template_dxf is not None:
        export_print_template(params, svg=args.template_svg, dxf=args.template_dxf)

    base, labels = build_faceplate(params, export_mode=args.export_mode)

    # Export
    if args.stl is not None:
        if args.export_mode == "combined":
            from build123d import Compound

            export_stl(_export_transform(Compound([o for o in (base, labels) if o is not None]), params), args.stl)
        elif args.export_mode == "base" and base is not None:
            export_stl(_export_transform(base, params), args.stl)
        elif args.export_mode == "labels" and labels is not None:
            export_stl(_export_transform(labels, params), args.stl)

    # Convenience exports (can be used alongside --export-mode)
    if args.stl_base is not None and base is not None:
        export_stl(_export_transform(base, params), args.stl_base)
    if args.stl_labels is not None and labels is not None:
        export_stl(_export_transform(labels, params), args.stl_labels)

    if args.step is not None:
        if args.export_mode == "combined":
            from build123d import Compound

            export_step(_export_transform(Compound([o for o in (base, labels) if o is not None]), params), args.step)
        elif args.export_mode == "base" and base is not None:
            export_step(_export_transform(base, params), args.step)
        elif args.export_mode == "labels" and labels is not None:
            export_step(_export_transform(labels, params), args.step)

    # View
    try:
        if args.export_mode == "combined":
            show(
                base,
                labels,
                names=["base", "labels"],
                colors=[params.base_color, params.label_color],
                reset_camera=Camera.RESET,
                grid=True,
                port=_ocp_port(),
            )
        elif args.export_mode == "base" and base is not None:
            show(
                base,
                names=["base"],
                colors=[params.base_color],
                reset_camera=Camera.RESET,
                grid=True,
                port=_ocp_port(),
            )
        elif args.export_mode == "labels" and labels is not None:
            show(
                labels,
                names=["labels"],
                colors=[params.label_color],
                reset_camera=Camera.RESET,
                grid=True,
                port=_ocp_port(),
            )
    except RuntimeError as ex:
        print("\nOCP viewer is not reachable.")
        print("- If you're using the VS Code extension: open 'OCP CAD Viewer' and ensure the backend is running.")
        print("- Or start the standalone viewer with: ./.venv/bin/python -m ocp_vscode --port 3939")
        print(f"\nDetails: {ex}")


if __name__ == "__main__":
    main()
