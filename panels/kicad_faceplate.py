#!/usr/bin/env python3
"""Turn a build123d faceplate script into a KiCad PCB faceplate (JLCPCB-ready).

Why this exists
---------------
The Daisy Patch Init panels in this repo are 3D-printed: a black base plus a
white raised-label solid. The exact same hole table and label list also describe
a *PCB* faceplate — black solder mask instead of black filament, white
silkscreen instead of white filament. This script reads a panel script
(``daisy_braids.py`` and friends), and writes a KiCad board you can plot
gerbers from and upload to JLCPCB.

It does **not** import the panel script (that would pull in build123d and
ocp_vscode). It parses it with ``ast``, so it runs anywhere with plain Python 3.

Coordinate systems — the one thing to get right
-----------------------------------------------
The ``HOLES`` tables in the panel scripts are lifted verbatim from Electrosmith's
``blank.kicad_pcb`` front-panel file, i.e. they are **KiCad coordinates**: origin
top-left of the panel, +X right, +Y *down*, ``patch.init()`` branding at y≈4,
jacks at y≈85..112. That is exactly the coordinate system this script emits, so
holes and cutouts are copied through unchanged.

Labels are the exception. ``build_base()`` in the panel scripts mirrors the base
about X (``export_transform``) while ``build_labels()`` does not
(``export_transform_labels``), so on the finished print ``HOLE_LABELS[i]`` ends up
next to the hole at ``(panel_w - x_i, y_i)`` — the X-mirror of hole *i*, not hole
*i* itself. That is why ``HOLE_LABELS[0] == "OUT-R"`` while hole 0 is the
left-hand jack of the top row. The jack/pot grid is X-symmetric so the print
comes out right; here we resolve the mirror explicitly and attach each label to
the hole it is really meant for.

Everything else (hole positions, drill sizes, the SD-card slot, the OLED window
and its two mounting holes, the mounting slots) is emitted in true, unmirrored
vendor coordinates.

Usage
-----
    python3 panels/kicad_faceplate.py panels/daisy_braids.py \
        --outdir exports/pcb/joy_10hp --name joy_10hp --gerbers

``--gerbers`` shells out to ``kicad-cli`` (KiCad 7+) to plot the fab layers and
zip them for JLCPCB. Without it you get just the board, which you can open in
KiCad and plot by hand via File -> Fabrication Outputs.

The board is written in the KiCad 6/7 file format (version 20221018), which
KiCad 7, 8 and 9 all open directly. Ordering notes live in
``exports/pcb/README.md``.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Panel-script extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hole:
    shape: str  # "circle" | "oval"
    x: float
    y: float
    drill: tuple[float, float]  # (major, minor); (d, d) for a circle


@dataclass(frozen=True)
class RectCutout:
    x: float  # min X
    y: float  # min Y
    w: float
    h: float


@dataclass(frozen=True)
class SilkDot:
    """A filled dot printed on the front silkscreen. Not a hole.

    The Big Genes panel prints eleven of these in a ring around every pot - a
    dial scale, 30 degrees apart at r=8, with the bottom 60 degrees left out
    because that is the dead zone of a 300-degree pot. They only exist on the
    silkscreen layer of the plot, which is how you tell them from a drilled hole:
    a real hole shows up on the mask and drill layers too.
    """

    x: float
    y: float
    d: float


@dataclass
class PanelSource:
    """Everything ``kicad_faceplate`` needs out of a build123d panel script."""

    holes: list[Hole]
    cutouts: list[RectCutout]
    labels_below: list[str]
    labels_above: list[str]
    panel_w: float = 50.8
    panel_h: float = 128.5
    label_offset: tuple[float, float] = (0.0, -7.0)
    label_hole_min_d: float = 5.0
    brand_top: str = ""
    brand_bottom: str = ""
    brand_margin: float = 4.01
    stem: str = "panel"
    # Labels naming a control's SECOND function - what it becomes on another
    # page. Set smaller than the primary so the panel reads at a glance as one
    # name per control with an alternate, rather than two equal names.
    secondary: frozenset = frozenset()
    secondary_ratio: float = 0.78
    # LAYOUT panels attach a label to each control, so there is no HOLE_LABELS
    # order to reconstruct and no X-mirror to undo. When this is set,
    # build_labels() uses it verbatim instead of pairing labels to holes.
    explicit_labels: "list[Label] | None" = None
    # Silkscreen markings that are not holes - the pot dial scales.
    dots: "list[SilkDot]" = field(default_factory=list)


class _Consts(ast.NodeVisitor):
    """Collect module-level ``NAME = <literal>`` bindings for constant folding."""

    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802 (ast API)
        if len(node.targets) != 1:
            return
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                self.values[target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
            return
        # Tuple unpacking - `PW, PH = 101.3, 128.5` and `_JA, _JB = 28.5, 15.8`
        # are how the hand-authored panels declare their dimensions, and missing
        # them meant the LAYOUT dict could not be folded at all.
        if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
            if len(target.elts) != len(node.value.elts):
                return
            for name, val in zip(target.elts, node.value.elts):
                if not isinstance(name, ast.Name):
                    continue
                try:
                    self.values[name.id] = ast.literal_eval(val)
                except ValueError:
                    pass


def _eval(node: ast.AST, consts: dict[str, object]) -> object:
    """Evaluate a literal expression, resolving module-level constants.

    Handles the ``64.75 + OLED_Y_OFFSET`` style arithmetic the panel scripts use
    to nudge the OLED window and its mounting holes.
    """
    if isinstance(node, ast.Name):
        if node.id not in consts:
            raise ValueError(f"unknown constant {node.id!r}")
        return consts[node.id]
    if isinstance(node, ast.BinOp):
        lhs, rhs = _eval(node.left, consts), _eval(node.right, consts)
        if isinstance(node.op, ast.Add):
            return lhs + rhs  # type: ignore[operator]
        if isinstance(node.op, ast.Sub):
            return lhs - rhs  # type: ignore[operator]
        if isinstance(node.op, ast.Mult):
            return lhs * rhs  # type: ignore[operator]
        if isinstance(node.op, ast.Div):
            return lhs / rhs  # type: ignore[operator]
        raise ValueError("unsupported operator")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand, consts)  # type: ignore[operator]
    if isinstance(node, ast.Dict):
        return {_eval(k, consts): _eval(v, consts) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Subscript):
        seq = _eval(node.value, consts)
        return seq[_eval(node.slice, consts)]  # type: ignore[index]
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_eval(e, consts) for e in node.elts)
    return ast.literal_eval(node)


def _call_kwargs(node: ast.Call, consts: dict[str, object]) -> dict[str, object]:
    return {kw.arg: _eval(kw.value, consts) for kw in node.keywords if kw.arg}


def _panel_params(tree: ast.Module, consts: dict[str, object]) -> dict[str, object]:
    """Read the defaults off the ``PanelParams`` dataclass."""
    out: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PanelParams":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value:
                    try:
                        out[stmt.target.id] = _eval(stmt.value, consts)
                    except ValueError:
                        pass
    return out


def load_panel(path: Path) -> PanelSource:
    tree = ast.parse(path.read_text(), filename=str(path))
    consts = _Consts()
    consts.visit(tree)
    cvals = consts.values

    holes: list[Hole] = []
    cutouts: list[RectCutout] = []
    labels_below: list[str] = []
    labels_above: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0] if isinstance(node.targets[0], ast.Name) else None
        name = target.id if target else None
        if name is None and isinstance(node, ast.AnnAssign):
            continue
        if name == "HOLE_LABELS":
            labels_below = [str(v) for v in _eval(node.value, cvals)]  # type: ignore[union-attr]
        elif name == "HOLE_LABELS_ABOVE":
            labels_above = [str(v) for v in _eval(node.value, cvals)]  # type: ignore[union-attr]

    # HOLES / RECT_CUTOUTS are annotated assignments (``HOLES: tuple[...] = (...)``).
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name) or node.value is None:
            continue
        if node.target.id == "HOLES":
            for elt in node.value.elts:  # type: ignore[attr-defined]
                kw = _call_kwargs(elt, cvals)
                drill = kw["drill"]
                holes.append(
                    Hole(
                        shape=str(kw["shape"]),
                        x=float(kw["x"]),  # type: ignore[arg-type]
                        y=float(kw["y"]),  # type: ignore[arg-type]
                        drill=(float(drill[0]), float(drill[1])),  # type: ignore[index]
                    )
                )
        elif node.target.id == "RECT_CUTOUTS":
            for elt in node.value.elts:  # type: ignore[attr-defined]
                kw = _call_kwargs(elt, cvals)
                cutouts.append(
                    RectCutout(
                        x=float(kw["x"]),  # type: ignore[arg-type]
                        y=float(kw["y"]),  # type: ignore[arg-type]
                        w=float(kw["w"]),  # type: ignore[arg-type]
                        h=float(kw["h"]),  # type: ignore[arg-type]
                    )
                )

    if not holes:
        layout = _find_layout(tree, cvals)
        if layout is not None:
            return _from_layout(layout, path)
        raise SystemExit(f"{path}: no HOLES table and no LAYOUT dict found")

    p = _panel_params(tree, cvals)
    return PanelSource(
        holes=holes,
        cutouts=cutouts,
        labels_below=labels_below,
        labels_above=labels_above,
        panel_w=float(p.get("panel_w", 50.8)),  # type: ignore[arg-type]
        panel_h=float(p.get("panel_h", 128.5)),  # type: ignore[arg-type]
        label_offset=tuple(p.get("label_offset", (0.0, -7.0))),  # type: ignore[arg-type]
        label_hole_min_d=float(p.get("label_hole_min_d", 5.0)),  # type: ignore[arg-type]
        brand_top=str(p.get("brand_text_top", "")),
        brand_bottom=str(p.get("brand_text_bottom", "")),
        stem=path.stem,
        secondary=frozenset(cvals.get("LABELS_SECONDARY", ()) or ()),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# LAYOUT panels (hand-authored, not Daisy Patch Init)
# ---------------------------------------------------------------------------
#
# The Daisy panels carry a HOLES table in vendor (KiCad) coordinates plus two
# parallel label lists whose order has to be un-mirrored. The hand-authored
# panels - ksoloti_biggenes.py and friends - instead carry a LAYOUT dict: one
# entry per control, each with its own x, y, diameter and label. That is a nicer
# thing to read, so this converts it rather than asking the panel to change.
#
# Two differences that matter:
#
#   * LAYOUT y measures UP from the bottom edge (a jack is low, a pot is high);
#     KiCad measures DOWN from the top. So y_kicad = panel_h - y_layout.
#   * LAYOUT x/y for a slot or window is its CENTRE. RectCutout wants a corner.

# Kinds that are a rectangular opening rather than a drilled hole. The value is
# just documentation - the width and height come from the control itself.
_LAYOUT_RECT_KINDS = {"screen", "sd_slot", "usb"}

# Mount slots. The plot gives 3 mm of travel; the width is opened to M3
# clearance, which is what the printed panels in this repo use too.
_MOUNT_SLOT = (6.2, 3.2)


def _find_layout(tree: ast.Module, consts: dict[str, object]) -> dict | None:
    """The ``LAYOUT = {...}`` dict, or None if the script has no such thing."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if isinstance(tgt, ast.Name) and tgt.id == "LAYOUT":
            got = _eval(node.value, consts)
            return got if isinstance(got, dict) else None
    return None


def _from_layout(layout: dict, path: Path) -> PanelSource:
    pw = float(layout.get("panel_w", 101.3))
    ph = float(layout.get("panel_h", 128.5))
    flip = lambda y: ph - float(y)  # noqa: E731 - LAYOUT is +Y up, KiCad is +Y down

    holes: list[Hole] = []
    cutouts: list[RectCutout] = []
    labels: list[Label] = []
    dots: list[SilkDot] = []

    for c in layout.get("controls", []):
        kind = str(c.get("kind", ""))
        x, y = float(c["x"]), flip(c["y"])

        if kind in _LAYOUT_RECT_KINDS:
            w, h = float(c["w"]), float(c["h"])
            cutouts.append(RectCutout(x=x - w / 2, y=y - h / 2, w=w, h=h))
        elif "d" in c:
            d = float(c["d"])
            holes.append(Hole("circle", x, y, (d, d)))
        else:
            # No diameter and not a rectangle: nothing to cut. Better to say so
            # than to guess a size and put a wrong hole in a fab file.
            print(f"  skipped {kind!r} at ({c['x']}, {c['y']}): no d and not a cutout")
            continue

        # A dial scale: {"r": 8.0, "d": 1.2, "step": 30, "skip": 60} prints ticks
        # every `step` degrees at radius `r`, leaving a gap of `skip` degrees at
        # the bottom for the pot's dead zone. Measured off the Big Genes plot.
        dial = c.get("dial")
        if dial:
            r_d = float(dial["r"])
            dd = float(dial.get("d", 1.2))
            step = float(dial.get("step", 30))
            skip = float(dial.get("skip", 60))
            n = int(round(360 / step))
            for i in range(n):
                # 0 deg is straight down; sweep clockwise. The dead zone is
                # centred on straight down, so drop anything inside it.
                ang = i * step
                if min(ang, 360 - ang) < skip / 2:
                    continue
                th = math.radians(ang - 90)
                dots.append(SilkDot(x + r_d * math.cos(th), y + r_d * math.sin(th), dd))

        # `label` is one line. `labels` is a stack, for a control whose function
        # changes with a page or a mode - Girl's exciter pots cycle level, meta
        # and timbre on S4, and Sorrow's four pots become per-drum densities and
        # wildness on the Kit page. A panel that prints only the first state is
        # documenting a third of the module.
        stack = c.get("labels") or ([c["label"]] if c.get("label") else [])
        dy = float(c.get("label_dy", -6.0))
        pitch = float(c.get("label_pitch", 2.1))
        # label_x lets one label serve a pair - the MIDI jacks share one, and
        # centring it on either jack pushes it off the panel edge.
        lx = float(c.get("label_x", c["x"]))
        size = float(c.get("label_size", 2.0))
        for n, text in enumerate(stack):
            text = str(text).strip()
            if not text:
                continue
            # The first line is the control's name; the rest are what it becomes
            # on later pages, so they are set smaller and read as subordinate.
            sz = size if n == 0 else float(c.get("alt_size", max(1.3, size - 0.5)))
            labels.append(Label(text, lx, flip(float(c["y"]) + dy - n * pitch), sz))

    for mx, my in layout.get("mounts", []):
        holes.append(Hole("oval", float(mx), flip(my), _MOUNT_SLOT))

    return PanelSource(
        holes=holes,
        cutouts=cutouts,
        labels_below=[],
        labels_above=[],
        panel_w=pw,
        panel_h=ph,
        brand_top=str(layout.get("brand_top", "")),
        brand_bottom=str(layout.get("brand_bottom", "")),
        stem=path.stem,
        explicit_labels=labels,
        dots=dots,
    )


# ---------------------------------------------------------------------------
# Label placement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Label:
    text: str
    x: float
    y: float
    size: float  # glyph height, mm


def labelable_holes(src: PanelSource) -> list[Hole]:
    """The holes the panel scripts label, in the order ``HOLE_LABELS`` uses."""
    hs = [h for h in src.holes if h.shape == "circle" and h.drill[0] >= src.label_hole_min_d]
    hs.sort(key=lambda h: (-h.y, h.x))  # top-to-bottom, left-to-right in KiCad coords
    return hs


def _mirror_partner(src: PanelSource, hole: Hole, tol: float = 0.2) -> Hole:
    """The hole at ``(panel_w - x, y)`` — the one this hole's label really belongs to.

    ``build_base()`` mirrors the base about X but ``build_labels()`` does not, so
    a label authored against hole *i* lands next to hole *i*'s X-mirror on the
    finished panel. The jack/pot grid is X-symmetric to within a few tens of a
    micron, so the partner always exists; falling back to the hole itself keeps
    a genuinely asymmetric panel from crashing.
    """
    tx = src.panel_w - hole.x
    best, best_d = hole, tol
    for cand in src.holes:
        d = math.hypot(cand.x - tx, cand.y - hole.y)
        if d < best_d:
            best, best_d = cand, d
    return best


def build_labels(src: PanelSource, *, label_size: float, brand_size: float) -> list[Label]:
    out: list[Label] = []
    if src.explicit_labels is not None:
        out.extend(src.explicit_labels)
        cx = src.panel_w / 2
        if src.brand_top.strip():
            out.append(Label(src.brand_top.strip(), cx, src.brand_margin, brand_size))
        if src.brand_bottom.strip():
            out.append(Label(src.brand_bottom.strip(), cx, src.panel_h - src.brand_margin, brand_size))
        return out
    below_dy = -src.label_offset[1]  # (0, -7) -> label sits 7 mm *below* in KiCad's +Y-down
    holes = labelable_holes(src)

    def _size(txt: str) -> float:
        return round(label_size * src.secondary_ratio, 3) if txt in src.secondary else label_size

    for idx, hole in enumerate(holes):
        target = _mirror_partner(src, hole)
        if idx < len(src.labels_below):
            txt = src.labels_below[idx].strip()
            if txt:
                out.append(Label(txt, target.x, target.y + below_dy, _size(txt)))
        if idx < len(src.labels_above):
            txt = src.labels_above[idx].strip()
            if txt:
                out.append(Label(txt, target.x, target.y - below_dy, _size(txt)))

    cx = src.panel_w / 2
    if src.brand_top.strip():
        out.append(Label(src.brand_top.strip(), cx, src.brand_margin, brand_size))
    if src.brand_bottom.strip():
        out.append(Label(src.brand_bottom.strip(), cx, src.panel_h - src.brand_margin, brand_size))
    return out


# ---------------------------------------------------------------------------
# KiCad board emission
# ---------------------------------------------------------------------------

_LAYERS = """	(layers
		(0 "F.Cu" signal)
		(31 "B.Cu" signal)
		(32 "B.Adhes" user "B.Adhesive")
		(33 "F.Adhes" user "F.Adhesive")
		(34 "B.Paste" user)
		(35 "F.Paste" user)
		(36 "B.SilkS" user "B.Silkscreen")
		(37 "F.SilkS" user "F.Silkscreen")
		(38 "B.Mask" user)
		(39 "F.Mask" user)
		(40 "Dwgs.User" user "User.Drawings")
		(41 "Cmts.User" user "User.Comments")
		(42 "Eco1.User" user "User.Eco1")
		(43 "Eco2.User" user "User.Eco2")
		(44 "Edge.Cuts" user)
		(45 "Margin" user)
		(46 "B.CrtYd" user "B.Courtyard")
		(47 "F.CrtYd" user "F.Courtyard")
		(48 "B.Fab" user)
		(49 "F.Fab" user)
	)"""


def _uuid(seed: int) -> str:
    """Deterministic UUIDs, so regenerating the board produces a clean diff."""
    h = f"{(seed * 0x9E3779B97F4A7C15) & ((1 << 128) - 1):032x}"
    return f"{h[0:8]}-{h[8:12]}-4{h[13:16]}-8{h[17:20]}-{h[20:32]}"


class _Ids:
    def __init__(self) -> None:
        self.n = 0

    def next(self) -> str:
        self.n += 1
        return _uuid(self.n)


def _f(v: float) -> str:
    return f"{v:.6f}".rstrip("0").rstrip(".") or "0"


def render_pcb(
    src: PanelSource,
    labels: list[Label],
    *,
    thickness: float,
    edge_width: float,
    text_thickness_ratio: float,
    text_width_ratio: float,
    washer_pads: bool,
    jlc_marker: tuple[float, float] | None,
    title: str,
) -> str:
    ids = _Ids()
    o: list[str] = []
    o.append("(kicad_pcb (version 20221018) (generator kicad_faceplate)")
    o.append("")
    o.append("\t(general")
    o.append(f"\t\t(thickness {_f(thickness)})")
    o.append("\t)")
    o.append("")
    o.append('\t(paper "A4")')
    o.append("\t(title_block")
    o.append(f'\t\t(title "{title}")')
    o.append('\t\t(comment 1 "Eurorack faceplate - no electrical function")')
    o.append('\t\t(comment 2 "Generated by panels/kicad_faceplate.py")')
    o.append("\t)")
    o.append(_LAYERS)
    o.append("")
    # Only the settings that actually matter for a faceplate. Everything else is
    # left to KiCad's defaults so the file stays readable in 7/8/9 alike.
    o.append("\t(setup")
    o.append("\t\t(pad_to_mask_clearance 0)")
    o.append("\t\t(pcbplotparams")
    o.append("\t\t\t(usegerberextensions false)")
    o.append("\t\t\t(usegerberattributes true)")
    o.append("\t\t\t(usegerberadvancedattributes true)")
    o.append("\t\t\t(creategerberjobfile true)")
    o.append("\t\t\t(plotframeref false)")
    o.append("\t\t\t(subtractmaskfromsilk false)")
    o.append("\t\t\t(mirror false)")
    o.append("\t\t\t(drillshape 0)")
    o.append('\t\t\t(outputdirectory "gerbers/")')
    o.append("\t\t)")
    o.append("\t)")
    o.append("")
    o.append('\t(net 0 "")')
    o.append("")

    # --- Edge.Cuts: panel outline, then every rectangular cutout ------------
    def gr_rect(x0: float, y0: float, x1: float, y1: float) -> None:
        o.append(f"\t(gr_rect (start {_f(x0)} {_f(y0)}) (end {_f(x1)} {_f(y1)})")
        o.append(f"\t\t(stroke (width {_f(edge_width)}) (type solid)) (fill none)")
        o.append(f'\t\t(layer "Edge.Cuts") (tstamp {ids.next()}))')

    gr_rect(0, 0, src.panel_w, src.panel_h)
    for r in src.cutouts:
        gr_rect(r.x, r.y, r.x + r.w, r.y + r.h)
    o.append("")

    # --- Silkscreen -------------------------------------------------------
    for lb in labels:
        th = round(lb.size * text_thickness_ratio, 3)
        w = round(lb.size * text_width_ratio, 4)
        # KiCad's font size is (width height). The stock stroke font is far wider
        # per glyph than the Arial Bold the printed panels use, so the width is
        # condensed to keep "OUT-L"/"OUT-R" inside the 12.17 mm jack pitch and
        # the branding clear of the mounting slots.
        o.append(f'\t(gr_text "{lb.text}" (at {_f(lb.x)} {_f(lb.y)}) (layer "F.SilkS") (tstamp {ids.next()})')
        o.append(f"\t\t(effects (font (size {_f(lb.size)} {_f(w)}) (thickness {_f(th)}) bold))")
        o.append("\t)")

    for dot in src.dots:
        # A filled circle is a zero-length line with a round cap of the right
        # width - simpler than a gr_circle with a fill, and it plots identically.
        o.append(f"\t(gr_line (start {_f(dot.x)} {_f(dot.y)}) (end {_f(dot.x)} {_f(dot.y)})")
        o.append(f"\t\t(stroke (width {_f(dot.d)}) (type solid))")
        o.append(f'\t\t(layer "F.SilkS") (tstamp {ids.next()}))')

    if jlc_marker is not None:
        # JLCPCB stamps its order number on every board. Left to itself it lands
        # somewhere on the front, which is not what you want on a faceplate. The
        # literal string "JLCJLCJLCJLC" tells them where to put it instead — here,
        # on the back, in the empty band between the OLED and the top jack row.
        mx, my = jlc_marker
        o.append(f'\t(gr_text "JLCJLCJLCJLC" (at {_f(mx)} {_f(my)}) (layer "B.SilkS") (tstamp {ids.next()})')
        o.append("\t\t(effects (font (size 1 1) (thickness 0.15)) (justify mirror))")
        o.append("\t)")
    o.append("")

    # --- One footprint carrying every non-plated hole ---------------------
    o.append(f'\t(footprint "faceplate:holes" (layer "F.Cu") (tstamp {ids.next()})')
    o.append("\t\t(at 0 0)")
    o.append('\t\t(attr through_hole exclude_from_pos_files exclude_from_bom)')
    o.append(f'\t\t(fp_text reference "PANEL" (at 0 -2) (layer "F.SilkS") hide (tstamp {ids.next()})')
    o.append("\t\t\t(effects (font (size 1 1) (thickness 0.15)))")
    o.append("\t\t)")
    o.append(f'\t\t(fp_text value "{title}" (at 0 -4) (layer "F.Fab") hide (tstamp {ids.next()})')
    o.append("\t\t\t(effects (font (size 1 1) (thickness 0.15)))")
    o.append("\t\t)")
    # With washer pads the hole gets Electrosmith's 0.1 mm copper ring and a
    # matching mask opening (a thin bare ring, hidden under the jack/pot washer).
    # Without, pad size == drill size, so both the ring and the mask opening
    # collapse onto the hole edge and the black mask runs right up to it.
    ring_grow = 0.2 if washer_pads else 0.0
    layers = '"*.Cu" "*.Mask"'
    for h in sorted(src.holes, key=lambda h: (h.y, h.x)):
        if h.shape == "circle":
            d = h.drill[0]
            ring = d + ring_grow
            o.append(
                f'\t\t(pad "" np_thru_hole circle (at {_f(h.x)} {_f(h.y)}) '
                f"(size {_f(ring)} {_f(ring)}) (drill {_f(d)}) "
                f"(layers {layers}) (tstamp {ids.next()}))"
            )
        else:
            major, minor = h.drill
            o.append(
                f'\t\t(pad "" np_thru_hole oval (at {_f(h.x)} {_f(h.y)}) '
                f"(size {_f(major + ring_grow * 2)} {_f(minor + ring_grow * 2)}) "
                f"(drill oval {_f(major)} {_f(minor)}) "
                f"(layers {layers}) (tstamp {ids.next()}))"
            )
    o.append("\t)")
    o.append(")")
    return "\n".join(o) + "\n"


_PRO = {
    "board": {"3dviewports": [], "design_settings": {}, "layer_presets": [], "viewports": []},
    "boards": [],
    "cvpcb": {"equivalence_files": []},
    "libraries": {"pinned_footprint_libs": [], "pinned_symbol_libs": []},
    "meta": {"filename": "panel.kicad_pro", "version": 1},
    "net_settings": {
        "classes": [
            {
                "bus_width": 12,
                "clearance": 0.2,
                "diff_pair_gap": 0.25,
                "diff_pair_width": 0.2,
                "line_style": 0,
                "microvia_diameter": 0.3,
                "microvia_drill": 0.1,
                "name": "Default",
                "pcb_color": "rgba(0, 0, 0, 0.000)",
                "schematic_color": "rgba(0, 0, 0, 0.000)",
                "track_width": 0.2,
                "via_diameter": 0.6,
                "via_drill": 0.3,
                "wire_width": 6,
            }
        ],
        "meta": {"version": 3},
        "net_colors": None,
    },
    "pcbnew": {"last_paths": {}, "page_layout_descr_file": ""},
    "sheets": [],
    "text_variables": {},
}


# ---------------------------------------------------------------------------
# Fabrication output
# ---------------------------------------------------------------------------

# The set JLCPCB wants. A faceplate has no traces, but the copper and mask
# layers still have to be there or the order form rejects the upload.
_FAB_LAYERS = "F.Cu,B.Cu,F.SilkS,B.SilkS,F.Mask,B.Mask,Edge.Cuts"


def write_preview(
    src: PanelSource,
    labels: list[Label],
    out: Path,
    *,
    px_wide: int = 1150,
) -> None:
    """Draw the board as a PNG, straight from the hole and label tables.

    The README used to say "export an SVG with kicad-cli, then rasterise over
    black", which needs a rasteriser this machine does not have and produced a
    picture of a plot rather than of the board. Drawing it here needs only PIL,
    and it draws the same tables the gerbers come from, so the preview cannot
    drift from what gets fabricated.

    Colours follow the convention the existing joy_10hp/preview.png set: the
    mask-free copper rings red, silkscreen pale yellow, the outline grey, on
    black. It is still a picture of the plot - the real panel is black mask with
    white lettering.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:  # pragma: no cover - preview is optional
        print("  preview skipped: PIL not installed")
        return

    SS = 3  # supersample, then downsample once at the end for clean edges
    scale = px_wide * SS / src.panel_w
    W = int(round(src.panel_w * scale))
    H = int(round(src.panel_h * scale))
    mm = lambda v: v * scale  # noqa: E731

    BG, COPPER, SILK, EDGE = "#000000", "#c83434", "#f2eda1", "#8a8d88"
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    d.rectangle([0, 0, W - 1, H - 1], outline=EDGE, width=max(1, int(mm(0.25))))

    for h in src.holes:
        major, minor = h.drill
        ring = mm(0.9)  # the mask-free annulus around each hole
        if h.shape == "oval" and abs(major - minor) > 1e-6:
            half = mm((major - minor) / 2)
            r = mm(minor / 2)
            for rr, col in ((r + ring, COPPER), (r, BG)):
                d.rounded_rectangle(
                    [mm(h.x) - half - rr, mm(h.y) - rr, mm(h.x) + half + rr, mm(h.y) + rr],
                    radius=rr, fill=col,
                )
        else:
            r = mm(major / 2)
            d.ellipse([mm(h.x) - r - ring, mm(h.y) - r - ring,
                       mm(h.x) + r + ring, mm(h.y) + r + ring], fill=COPPER)
            d.ellipse([mm(h.x) - r, mm(h.y) - r, mm(h.x) + r, mm(h.y) + r], fill=BG)

    for c in src.cutouts:
        d.rectangle([mm(c.x), mm(c.y), mm(c.x + c.w), mm(c.y + c.h)],
                    fill=BG, outline=EDGE, width=max(1, int(mm(0.25))))

    for dot in src.dots:
        r = mm(dot.d / 2)
        d.ellipse([mm(dot.x) - r, mm(dot.y) - r, mm(dot.x) + r, mm(dot.y) + r], fill=SILK)

    font_path = next((f for f in (
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ) if Path(f).exists()), None)
    for lb in labels:
        size = max(6, int(mm(lb.size)))
        try:
            font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        except OSError:
            font = ImageFont.load_default()
        d.text((mm(lb.x), mm(lb.y)), lb.text, fill=SILK, font=font, anchor="mm")

    im.resize((px_wide, int(round(px_wide * src.panel_h / src.panel_w))),
              Image.LANCZOS).save(out)
    print(f"wrote {out}")


def plot_gerbers(pcb: Path, gerber_dir: Path, zip_name: str) -> None:
    """Plot gerbers + Excellon drill with ``kicad-cli`` and zip them up."""
    import shutil
    import subprocess

    cli = shutil.which("kicad-cli")
    if cli is None:
        print("\nkicad-cli not found - open the board in KiCad and use")
        print("  File > Fabrication Outputs > Gerbers... / Drill Files...")
        return

    gerber_dir.mkdir(parents=True, exist_ok=True)  # kicad-cli won't create it
    subprocess.run(
        [cli, "pcb", "export", "gerbers", "-o", f"{gerber_dir}/", "--layers", _FAB_LAYERS, str(pcb)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            cli, "pcb", "export", "drill",
            "-o", f"{gerber_dir}/",
            "--format", "excellon",
            "--excellon-separate-th",
            "--drill-origin", "absolute",
            "--excellon-units", "mm",
            str(pcb),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    zpath = gerber_dir.parent / zip_name
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(gerber_dir.iterdir()):
            if f.is_file() and f.name != zip_name:
                z.write(f, f.name)
    print(f"\nwrote {gerber_dir}/ ({len(list(gerber_dir.iterdir()))} files)")
    print(f"wrote {zpath}  <- upload this to JLCPCB")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(src: PanelSource, labels: list[Label]) -> str:
    lines = [
        f"panel        {src.panel_w} x {src.panel_h} mm "
        f"({src.panel_w / 5.08:.0f}HP)",
        f"holes        {len(src.holes)}",
    ]
    by_drill: dict[str, int] = {}
    for h in src.holes:
        key = f"{h.drill[0]:g}" if h.shape == "circle" else f"{h.drill[0]:g}x{h.drill[1]:g} slot"
        by_drill[key] = by_drill.get(key, 0) + 1
    for key in sorted(by_drill, key=lambda k: (len(k), k)):
        lines.append(f"  {by_drill[key]:>2} x {key} mm")
    lines.append(f"cutouts      {len(src.cutouts)}")
    for r in src.cutouts:
        lines.append(f"  {r.w:g} x {r.h:g} mm at ({r.x:g}, {r.y:g})")
    lines.append(f"silkscreen   {len(labels)} strings")
    for lb in sorted(labels, key=lambda l: (l.y, l.x)):
        lines.append(f"  {lb.text:<12} at ({lb.x:7.3f}, {lb.y:7.3f})  {lb.size} mm")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a KiCad PCB faceplate from a build123d panel script."
    )
    ap.add_argument("panel", type=Path, help="e.g. panels/daisy_braids.py")
    ap.add_argument("--outdir", type=Path, required=True, help="directory to write the project into")
    ap.add_argument("--name", default=None, help="project stem (default: panel script stem)")
    ap.add_argument("--title", default=None, help="title block text (default: --name)")
    ap.add_argument("--label-size", type=float, default=2.6, help="silkscreen label height, mm")
    ap.add_argument("--brand-size", type=float, default=3.0, help="silkscreen branding height, mm")
    ap.add_argument(
        "--text-thickness-ratio",
        type=float,
        default=0.17,
        help="stroke width as a fraction of text height (KiCad bold is ~0.15)",
    )
    ap.add_argument(
        "--text-width-ratio",
        type=float,
        default=0.72,
        help="glyph width as a fraction of height; condenses KiCad's wide stroke font",
    )
    ap.add_argument(
        "--brand-margin",
        type=float,
        default=None,
        help="branding distance from the top/bottom edge, mm "
        "(default 4.01, matching the printed panels — note that is inside the rack-rail zone)",
    )
    ap.add_argument("--board-thickness", type=float, default=1.6, help="PCB thickness, mm")
    ap.add_argument("--edge-width", type=float, default=0.1, help="Edge.Cuts line width, mm")
    ap.add_argument(
        "--no-washer-pads",
        action="store_true",
        help="drop Electrosmith's 0.1 mm copper ring / mask opening around each hole, "
        "so the solder mask runs right up to the hole edge",
    )
    ap.add_argument(
        "--no-jlc-marker",
        action="store_true",
        help='omit the "JLCJLCJLCJLC" back-silkscreen marker that tells JLCPCB where '
        "to stamp its order number",
    )
    ap.add_argument(
        "--gerbers",
        action="store_true",
        help="also plot gerbers + drill via kicad-cli and zip them for JLCPCB",
    )
    ap.add_argument(
        "--no-preview",
        action="store_true",
        help="skip preview.png (drawn from the hole and label tables, needs PIL)",
    )
    args = ap.parse_args()

    src = load_panel(args.panel)
    if args.brand_margin is not None:
        src.brand_margin = args.brand_margin
    name = args.name or src.stem
    title = args.title or name
    labels = build_labels(src, label_size=args.label_size, brand_size=args.brand_size)

    args.outdir.mkdir(parents=True, exist_ok=True)
    pcb = args.outdir / f"{name}.kicad_pcb"
    pcb.write_text(
        render_pcb(
            src,
            labels,
            thickness=args.board_thickness,
            edge_width=args.edge_width,
            text_thickness_ratio=args.text_thickness_ratio,
            text_width_ratio=args.text_width_ratio,
            washer_pads=not args.no_washer_pads,
            jlc_marker=None if args.no_jlc_marker else (src.panel_w / 2, 75.0),
            title=title,
        )
    )
    pro = dict(_PRO)
    pro["meta"] = {"filename": f"{name}.kicad_pro", "version": 1}
    (args.outdir / f"{name}.kicad_pro").write_text(json.dumps(pro, indent=2) + "\n")

    print(report(src, labels))
    print(f"\nwrote {pcb}")
    print(f"wrote {args.outdir / f'{name}.kicad_pro'}")

    if not args.no_preview:
        write_preview(src, labels, args.outdir / "preview.png")

    if args.gerbers:
        plot_gerbers(pcb, args.outdir / "gerbers", f"{name}-gerbers.zip")


if __name__ == "__main__":
    main()
