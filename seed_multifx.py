"""N8Synth 6HP 2xN faceplate recreated in build123d.

This file mirrors the geometry/parameters from the provided OpenSCAD source:
- Base panel with optional Eurorack mounting slots/holes
- 2 columns of control holes
- Optional screen cutout + (bottom) screen mounting holes
- Raised text labels (separate solid for 2-color printing)

Run:
  ./.venv/bin/python n8synth_faceplate_build123d.py

Export STLs:
  ./.venv/bin/python n8synth_faceplate_build123d.py --export-mode base --stl base.stl
  ./.venv/bin/python n8synth_faceplate_build123d.py --export-mode labels --stl labels.stl

Notes:
- Coordinates follow the OpenSCAD model: XY origin at panel bottom-left, Z up.
- Text uses OCP fonts; "Arial" is the default and should work on macOS.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
TextMode = Literal["emboss", "deboss", "inlay"]


@dataclass(frozen=True)
class FaceplateParams:
    # Panel
    panel_w: float = 30.00248
    panel_h: float = 128.50114
    thickness: float = 2.0

    # Control holes
    hole_d: float = 6.90183
    hole_oversize: float = 0.0

    # Global vertical offset applied to content features (not mounting holes)
    content_y_offset: float = 1.0

    # Hole positions (mm from bottom-left)
    col_x: tuple[float, float] = (6.752515, 23.26259)
    row_y: tuple[float, ...] = (17.001699, 34.271711, 51.541635, 68.811655, 86.08158)

    # Eurorack mounting
    add_mount_holes: bool = True
    mount_hole_d: float = 3.2
    mount_x_from_edge: float = 3.0
    mount_y_from_edge: float = 3.0
    mount_both_sides: bool = True
    mount_slot: bool = True
    mount_slot_len: float = 4.0

    # Labels
    label_height: float = 0.2
    label_font: str = "Arial"
    label_font_style: FontStyle = FontStyle.BOLD
    label_size: float = 3.2
    labels_y_offset: float = 0.0

    # Secondary (shift/alt) labels under the main label
    secondary_label_size: float = 2.1
    secondary_label_height: float = 0.2
    secondary_label_style: FontStyle = FontStyle.BOLD
    # Offset from the main label position (dx, dy) in mm
    secondary_label_offset_from_main: tuple[float, float] = (0.0, -3.0)

    # Inverse label style (a raised rounded rectangle with the text cut out)
    # This is useful for e.g. printing a black plaque over a white base,
    # leaving "white text" by exposing the base color.
    inverse_label_enable: bool = True
    # Per-label masks (must match len(row_y)); True means use inverse style for that label.
    inverse_left: tuple[bool, ...] = (False, False, False, False, False)
    inverse_right: tuple[bool, ...] = (True, True, False, False, False)
    inverse_corner_r: float = 0.8
    # Padding around estimated text bounds (mm)
    inverse_pad_x: float = 0.6
    inverse_pad_y: float = 0.6
    # Minimum plaque size (mm)
    inverse_min_w: float = 0.0
    inverse_min_h: float = 0.0

    brand_size: float = 2.6
    brand_height: float = 0.2

    # Text rendering mode
    # - emboss: raised text (current behaviour)
    # - deboss: engraved text (subtracted from base)
    # - inlay: recessed pocket in base + separate inlay solid (for 2-color printing)
    text_mode: TextMode = "emboss"
    # Depth used for deboss/inlay (mm). With 0.1 mm layers, 0.2 mm = 2 layers.
    inlay_depth: float = 0.2

    base_color: tuple[float, float, float] = (0.86, 0.86, 0.86)
    label_color: tuple[float, float, float] = (0.10, 0.10, 0.10)

    brand_text: str = "84aW"
    model_text: str = "MultiFX"
    brand_margin: float = 4.01

    # Screen
    screen_enable: bool = True
    screen_w: float = 24.8
    screen_h: float = 13.6
    screen_y_bias: float = 3.0
    screen_corner_r: float = 0.0
    screen_mount_hole_d: float = 3.0
    screen_top_hole_offset: float = 2.2
    screen_bottom_hole_offset: float = 7.6

    # Per-hole labels (must match len(row_y))
    labels_left: tuple[str, ...] = ("IN-R", "IN-L", "CV1", "CV1", "MIX")
    labels_right: tuple[str, ...] = ("OUT-R", "OUT-L", "CV2", "CV2", "SELECT")
    # Optional secondary labels (set to "" for none)
    secondary_left: tuple[str, ...] = ("", "", "", "", "")
    secondary_right: tuple[str, ...] =  ("", "", "CLK", "", "MENU")
    label_offset: tuple[float, float] = (0.0, -7.0)

    def screen_origin(self) -> tuple[float, float]:
        """Bottom-left of screen cutout, matching the OpenSCAD formula."""
        sx = (self.panel_w - self.screen_w) / 2

        top_control_y = self.row_y[-1] + self.labels_y_offset
        top_mount_y = self.panel_h - self.mount_y_from_edge
        center_between = (top_control_y + top_mount_y) / 2

        sy = center_between - self.screen_h / 2 + self.screen_y_bias
        return (sx, sy)

    def brand_bottom_pos(self) -> tuple[float, float]:
        return (self.panel_w / 2, self.brand_margin)

    def brand_top_pos(self) -> tuple[float, float]:
        return (self.panel_w / 2, self.panel_h - self.brand_margin)


def _slot_or_hole_2d(params: FaceplateParams) -> None:
    """Create a mounting slot/hole profile at the sketch origin."""
    if params.mount_slot:
        SlotOverall(params.mount_slot_len, params.mount_hole_d)
    else:
        Circle(params.mount_hole_d / 2)


def build_base(params: FaceplateParams) -> "object":
    """Return the base panel solid with all cutouts subtracted."""

    cut_z0 = -0.2
    cut_h = params.thickness + 0.2

    with BuildPart() as p:
        Box(
            params.panel_w,
            params.panel_h,
            params.thickness,
            align=(Align.MIN, Align.MIN, Align.MIN),
            mode=Mode.ADD,
        )

        # All cutouts are built as a single sketch and extruded through.
        with BuildSketch(Plane.XY.offset(cut_z0)):
            # Mounting features (not affected by content_y_offset)
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

            # Content features shifted by content_y_offset
            dy = params.content_y_offset

            # 2xN control holes
            control_points: list[tuple[float, float]] = []
            for cx in params.col_x:
                for y in params.row_y:
                    control_points.append((cx, y + params.labels_y_offset + dy))

            with Locations(*control_points):
                d = params.hole_d + params.hole_oversize
                Circle(d / 2)

            # Screen features
            if params.screen_enable:
                sx, sy = params.screen_origin()
                sx += 0.0
                sy += dy

                with Locations((sx, sy)):
                    if params.screen_corner_r > 0:
                        RectangleRounded(
                            params.screen_w,
                            params.screen_h,
                            params.screen_corner_r,
                            align=(Align.MIN, Align.MIN),
                        )
                    else:
                        Rectangle(
                            params.screen_w,
                            params.screen_h,
                            align=(Align.MIN, Align.MIN),
                        )

                xL = sx
                xR = sx + params.screen_w
                yB = sy
                yT = sy + params.screen_h

                # Match the OpenSCAD: bottom holes only (top holes commented out there)
                screen_mount_pts = [
                    (xL, yB - params.screen_bottom_hole_offset),
                    (xR, yB - params.screen_bottom_hole_offset),
                ]
                with Locations(*screen_mount_pts):
                    Circle(params.screen_mount_hole_d / 2)

        extrude(amount=cut_h, mode=Mode.SUBTRACT)

        # Optional recessed text pocket (for deboss/inlay modes)
        if params.text_mode in ("deboss", "inlay"):
            pocket_depth = max(0.0, float(params.inlay_depth))
            if pocket_depth > 0:
                # Build the pocket sketch at the pocket bottom and cut upward.
                z0 = params.thickness - pocket_depth

                def add_text_to_sketch(txt: str, *, font_size: float, style: FontStyle, at_xy: tuple[float, float]):
                    with Locations(at_xy):
                        Text(txt, font_size=font_size, font=params.label_font, font_style=style)

                with BuildSketch(Plane.XY.offset(z0)) as pocket_sk:
                    # Brand labels (no global vertical offset)
                    add_text_to_sketch(
                        params.brand_text,
                        font_size=params.brand_size,
                        style=params.label_font_style,
                        at_xy=params.brand_bottom_pos(),
                    )
                    add_text_to_sketch(
                        params.model_text,
                        font_size=params.brand_size,
                        style=params.label_font_style,
                        at_xy=params.brand_top_pos(),
                    )

                    # Per-hole labels shifted by the global offset
                    dy = params.content_y_offset
                    dx_off, dy_off = params.label_offset
                    sec_dx, sec_dy = params.secondary_label_offset_from_main

                    for i, y in enumerate(params.row_y):
                        y0_txt = y + params.labels_y_offset + dy + dy_off
                        left_xy = (params.col_x[0] + dx_off, y0_txt)
                        right_xy = (params.col_x[1] + dx_off, y0_txt)

                        if params.labels_left[i]:
                            add_text_to_sketch(
                                params.labels_left[i],
                                font_size=params.label_size,
                                style=params.label_font_style,
                                at_xy=left_xy,
                            )
                        if params.labels_right[i]:
                            add_text_to_sketch(
                                params.labels_right[i],
                                font_size=params.label_size,
                                style=params.label_font_style,
                                at_xy=right_xy,
                            )

                        sec_left = params.secondary_left[i].strip()
                        if sec_left:
                            add_text_to_sketch(
                                sec_left,
                                font_size=params.secondary_label_size,
                                style=params.secondary_label_style,
                                at_xy=(left_xy[0] + sec_dx, left_xy[1] + sec_dy),
                            )

                        sec_right = params.secondary_right[i].strip()
                        if sec_right:
                            add_text_to_sketch(
                                sec_right,
                                font_size=params.secondary_label_size,
                                style=params.secondary_label_style,
                                at_xy=(right_xy[0] + sec_dx, right_xy[1] + sec_dy),
                            )

                extrude(to_extrude=pocket_sk.sketch, amount=pocket_depth + 0.1, mode=Mode.SUBTRACT)

    return p.part


def build_labels(params: FaceplateParams) -> "object":
    """Return either raised labels (emboss) or an inlay solid (inlay/deboss)."""

    n_rows = len(params.row_y)
    if len(params.labels_left) != n_rows or len(params.labels_right) != n_rows:
        raise ValueError("labels_left/labels_right must match the number of row_y entries")
    if len(params.secondary_left) != n_rows or len(params.secondary_right) != n_rows:
        raise ValueError("secondary_left/secondary_right must match the number of row_y entries")
    if len(params.inverse_left) != n_rows or len(params.inverse_right) != n_rows:
        raise ValueError("inverse_left/inverse_right must match the number of row_y entries")

    # In deboss mode, there is no separate print part, but returning the inlay
    # volume is still useful for preview/debugging.
    if params.text_mode in ("deboss", "inlay"):
        depth = max(0.0, float(params.inlay_depth))
        z0 = params.thickness - depth

        with BuildPart() as p:
            if depth <= 0:
                return p.part

            def add_text(txt: str, *, font_size: float, style: FontStyle, at_xy: tuple[float, float]):
                with BuildSketch(Plane.XY.offset(z0)) as sk:
                    with Locations(at_xy):
                        Text(txt, font_size=font_size, font=params.label_font, font_style=style)
                extrude(to_extrude=sk.sketch, amount=depth, mode=Mode.ADD)

            # Brand
            add_text(
                params.brand_text,
                font_size=params.brand_size,
                style=params.label_font_style,
                at_xy=params.brand_bottom_pos(),
            )
            add_text(
                params.model_text,
                font_size=params.brand_size,
                style=params.label_font_style,
                at_xy=params.brand_top_pos(),
            )

            dy = params.content_y_offset
            dx_off, dy_off = params.label_offset
            sec_dx, sec_dy = params.secondary_label_offset_from_main

            for i, y in enumerate(params.row_y):
                y0_txt = y + params.labels_y_offset + dy + dy_off
                left_xy = (params.col_x[0] + dx_off, y0_txt)
                right_xy = (params.col_x[1] + dx_off, y0_txt)

                if params.labels_left[i]:
                    add_text(
                        params.labels_left[i],
                        font_size=params.label_size,
                        style=params.label_font_style,
                        at_xy=left_xy,
                    )
                if params.labels_right[i]:
                    add_text(
                        params.labels_right[i],
                        font_size=params.label_size,
                        style=params.label_font_style,
                        at_xy=right_xy,
                    )

                sec_left = params.secondary_left[i].strip()
                if sec_left:
                    add_text(
                        sec_left,
                        font_size=params.secondary_label_size,
                        style=params.secondary_label_style,
                        at_xy=(left_xy[0] + sec_dx, left_xy[1] + sec_dy),
                    )

                sec_right = params.secondary_right[i].strip()
                if sec_right:
                    add_text(
                        sec_right,
                        font_size=params.secondary_label_size,
                        style=params.secondary_label_style,
                        at_xy=(right_xy[0] + sec_dx, right_xy[1] + sec_dy),
                    )

        return p.part

    # Default: emboss mode (existing behaviour)
    def make_text_sketch(txt: str, *, font_size: float, style: FontStyle, at_xy: tuple[float, float]):
        with BuildSketch(Plane.XY.offset(params.thickness)) as sk:
            with Locations(at_xy):
                Text(txt, font_size=font_size, font=params.label_font, font_style=style)
        return sk.sketch

    def plaque_size_from_text_sketch(text_sketch) -> tuple[float, float]:
        bb_size = text_sketch.bounding_box().size
        w = max(params.inverse_min_w, bb_size.X + 2 * params.inverse_pad_x)
        h = max(params.inverse_min_h, bb_size.Y + 2 * params.inverse_pad_y)
        return (w, h)

    def add_inverse_label(
        txt_main: str,
        *,
        at_xy: tuple[float, float],
        main_font_size: float,
        main_style: FontStyle,
        height: float,
    ) -> None:
        # Build the text sketch once, use its actual bounds to size the plaque,
        # then subtract the same sketch from the plaque to create the inverse label.
        text_sk = make_text_sketch(txt_main, font_size=main_font_size, style=main_style, at_xy=at_xy)
        w, h = plaque_size_from_text_sketch(text_sk)
        with BuildSketch(Plane.XY.offset(params.thickness)) as plaque_sk:
            with Locations(at_xy):
                RectangleRounded(w, h, params.inverse_corner_r)
        extrude(to_extrude=plaque_sk.sketch, amount=height, mode=Mode.ADD)

        # Cut the text out of the plaque so the base color shows through.
        extrude(to_extrude=text_sk, amount=height + 0.1, mode=Mode.SUBTRACT)

    with BuildPart() as p:
        # Brand top/bottom (no global vertical offset)
        brand_bottom_sk = make_text_sketch(
            params.brand_text,
            font_size=params.brand_size,
            style=params.label_font_style,
            at_xy=params.brand_bottom_pos(),
        )
        brand_top_sk = make_text_sketch(
            params.model_text,
            font_size=params.brand_size,
            style=params.label_font_style,
            at_xy=params.brand_top_pos(),
        )
        extrude(to_extrude=brand_bottom_sk, amount=params.brand_height, mode=Mode.ADD)
        extrude(to_extrude=brand_top_sk, amount=params.brand_height, mode=Mode.ADD)

        # Per-hole labels shifted by the global offset
        dy = params.content_y_offset
        dx_off, dy_off = params.label_offset

        sec_dx, sec_dy = params.secondary_label_offset_from_main

        for i, y in enumerate(params.row_y):
            y0 = y + params.labels_y_offset + dy + dy_off

            left_xy = (params.col_x[0] + dx_off, y0)
            right_xy = (params.col_x[1] + dx_off, y0)

            # Main labels
            if params.inverse_label_enable and params.inverse_left[i] and params.labels_left[i]:
                add_inverse_label(
                    params.labels_left[i],
                    at_xy=left_xy,
                    main_font_size=params.label_size,
                    main_style=params.label_font_style,
                    height=params.label_height,
                )
            elif params.labels_left[i]:
                left_sk = make_text_sketch(
                    params.labels_left[i],
                    font_size=params.label_size,
                    style=params.label_font_style,
                    at_xy=left_xy,
                )
                extrude(to_extrude=left_sk, amount=params.label_height, mode=Mode.ADD)

            if params.inverse_label_enable and params.inverse_right[i] and params.labels_right[i]:
                add_inverse_label(
                    params.labels_right[i],
                    at_xy=right_xy,
                    main_font_size=params.label_size,
                    main_style=params.label_font_style,
                    height=params.label_height,
                )
            elif params.labels_right[i]:
                right_sk = make_text_sketch(
                    params.labels_right[i],
                    font_size=params.label_size,
                    style=params.label_font_style,
                    at_xy=right_xy,
                )
                extrude(to_extrude=right_sk, amount=params.label_height, mode=Mode.ADD)

            # Secondary labels (small) under the main label
            sec_left = params.secondary_left[i].strip()
            if sec_left:
                sec_left_sk = make_text_sketch(
                    sec_left,
                    font_size=params.secondary_label_size,
                    style=params.secondary_label_style,
                    at_xy=(left_xy[0] + sec_dx, left_xy[1] + sec_dy),
                )
                extrude(to_extrude=sec_left_sk, amount=params.secondary_label_height, mode=Mode.ADD)

            sec_right = params.secondary_right[i].strip()
            if sec_right:
                sec_right_sk = make_text_sketch(
                    sec_right,
                    font_size=params.secondary_label_size,
                    style=params.secondary_label_style,
                    at_xy=(right_xy[0] + sec_dx, right_xy[1] + sec_dy),
                )
                extrude(to_extrude=sec_right_sk, amount=params.secondary_label_height, mode=Mode.ADD)

    return p.part


def build_faceplate(params: FaceplateParams, export_mode: ExportMode = "combined") -> tuple[object | None, object | None]:
    base = build_base(params) if export_mode in ("combined", "base") else None
    labels = build_labels(params) if export_mode in ("combined", "labels") else None
    return base, labels


def export_print_template(params: FaceplateParams, *, svg: Path | None, dxf: Path | None) -> None:
    """Export a 1:1 2D template for paper printing/checking alignment.

    The output is intended to be printed at 100% (no scaling). It includes:
    - Panel outline
    - All cutouts (mount holes/slots, control holes, screen cutout)
    - Label outlines (including inverse plaques if enabled)
    - A 10 mm calibration square
    """

    def make_layer_outline():
        with BuildSketch(Plane.XY) as sk:
            Rectangle(params.panel_w, params.panel_h, align=(Align.MIN, Align.MIN))
        return sk.sketch

    def make_layer_cutouts():
        with BuildSketch(Plane.XY) as sk:
            # Mounting features (not affected by content_y_offset)
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

            dy = params.content_y_offset

            # 2xN control holes
            control_points: list[tuple[float, float]] = []
            for cx in params.col_x:
                for y in params.row_y:
                    control_points.append((cx, y + params.labels_y_offset + dy))
            with Locations(*control_points):
                d = params.hole_d + params.hole_oversize
                Circle(d / 2)

            # Screen cutout + mount holes
            if params.screen_enable:
                sx, sy = params.screen_origin()
                sx += 0.0
                sy += dy

                with Locations((sx, sy)):
                    if params.screen_corner_r > 0:
                        RectangleRounded(
                            params.screen_w,
                            params.screen_h,
                            params.screen_corner_r,
                            align=(Align.MIN, Align.MIN),
                        )
                    else:
                        Rectangle(params.screen_w, params.screen_h, align=(Align.MIN, Align.MIN))

                xL = sx
                xR = sx + params.screen_w
                yB = sy

                with Locations(
                    (xL, yB - params.screen_bottom_hole_offset),
                    (xR, yB - params.screen_bottom_hole_offset),
                ):
                    Circle(params.screen_mount_hole_d / 2)

        return sk.sketch

    def text_sketch(txt: str, *, font_size: float, style: FontStyle, at_xy: tuple[float, float]):
        with BuildSketch(Plane.XY) as sk:
            with Locations(at_xy):
                Text(txt, font_size=font_size, font=params.label_font, font_style=style)
        return sk.sketch

    def plaque_sketch(*, w: float, h: float, r: float, at_xy: tuple[float, float]):
        with BuildSketch(Plane.XY) as sk:
            with Locations(at_xy):
                RectangleRounded(w, h, r)
        return sk.sketch

    def make_layer_labels():
        # Brand labels (no content offset)
        brand = [
            text_sketch(
                params.brand_text,
                font_size=params.brand_size,
                style=params.label_font_style,
                at_xy=params.brand_bottom_pos(),
            ),
            text_sketch(
                params.model_text,
                font_size=params.brand_size,
                style=params.label_font_style,
                at_xy=params.brand_top_pos(),
            ),
        ]

        dy = params.content_y_offset
        dx_off, dy_off = params.label_offset
        sec_dx, sec_dy = params.secondary_label_offset_from_main

        main_text: list[object] = []
        secondary_text: list[object] = []
        plaques: list[object] = []

        for i, y in enumerate(params.row_y):
            y0 = y + params.labels_y_offset + dy + dy_off
            left_xy = (params.col_x[0] + dx_off, y0)
            right_xy = (params.col_x[1] + dx_off, y0)

            # Left main
            if params.labels_left[i]:
                tsk = text_sketch(
                    params.labels_left[i],
                    font_size=params.label_size,
                    style=params.label_font_style,
                    at_xy=left_xy,
                )
                main_text.append(tsk)
                if params.inverse_label_enable and params.inverse_left[i]:
                    bb = tsk.bounding_box().size
                    w = max(params.inverse_min_w, bb.X + 2 * params.inverse_pad_x)
                    h = max(params.inverse_min_h, bb.Y + 2 * params.inverse_pad_y)
                    plaques.append(plaque_sketch(w=w, h=h, r=params.inverse_corner_r, at_xy=left_xy))

            # Right main
            if params.labels_right[i]:
                tsk = text_sketch(
                    params.labels_right[i],
                    font_size=params.label_size,
                    style=params.label_font_style,
                    at_xy=right_xy,
                )
                main_text.append(tsk)
                if params.inverse_label_enable and params.inverse_right[i]:
                    bb = tsk.bounding_box().size
                    w = max(params.inverse_min_w, bb.X + 2 * params.inverse_pad_x)
                    h = max(params.inverse_min_h, bb.Y + 2 * params.inverse_pad_y)
                    plaques.append(plaque_sketch(w=w, h=h, r=params.inverse_corner_r, at_xy=right_xy))

            # Secondary labels
            sec_left = params.secondary_left[i].strip()
            if sec_left:
                secondary_text.append(
                    text_sketch(
                        sec_left,
                        font_size=params.secondary_label_size,
                        style=params.secondary_label_style,
                        at_xy=(left_xy[0] + sec_dx, left_xy[1] + sec_dy),
                    )
                )
            sec_right = params.secondary_right[i].strip()
            if sec_right:
                secondary_text.append(
                    text_sketch(
                        sec_right,
                        font_size=params.secondary_label_size,
                        style=params.secondary_label_style,
                        at_xy=(right_xy[0] + sec_dx, right_xy[1] + sec_dy),
                    )
                )

        return {
            "brand": brand,
            "main": main_text,
            "secondary": secondary_text,
            "plaques": plaques,
        }

    def make_layer_calibration():
        # 10 mm square placed to the right of the panel
        x0 = params.panel_w + 6
        y0 = 6
        with BuildSketch(Plane.XY) as sk:
            with Locations((x0, y0)):
                Rectangle(10, 10, align=(Align.MIN, Align.MIN))
        return sk.sketch

    outline = make_layer_outline()
    cutouts = make_layer_cutouts()
    labels_layers = make_layer_labels()
    calib = make_layer_calibration()

    if svg is not None:
        exp = ExportSVG(margin=5, line_weight=0.18)
        exp.add_layer("outline")
        exp.add_layer("cutouts")
        exp.add_layer("plaques")
        exp.add_layer("labels")
        exp.add_layer("secondary")
        exp.add_layer("calibration")
        exp.add_shape(outline, layer="outline")
        exp.add_shape(cutouts, layer="cutouts")
        exp.add_shape(labels_layers["plaques"], layer="plaques")
        exp.add_shape(labels_layers["brand"], layer="labels")
        exp.add_shape(labels_layers["main"], layer="labels")
        exp.add_shape(labels_layers["secondary"], layer="secondary")
        exp.add_shape(calib, layer="calibration")
        exp.write(svg)

    if dxf is not None:
        exp = ExportDXF()
        exp.add_layer("outline")
        exp.add_layer("cutouts")
        exp.add_layer("plaques")
        exp.add_layer("labels")
        exp.add_layer("secondary")
        exp.add_layer("calibration")
        exp.add_shape(outline, layer="outline")
        exp.add_shape(cutouts, layer="cutouts")
        exp.add_shape(labels_layers["plaques"], layer="plaques")
        exp.add_shape(labels_layers["brand"], layer="labels")
        exp.add_shape(labels_layers["main"], layer="labels")
        exp.add_shape(labels_layers["secondary"], layer="secondary")
        exp.add_shape(calib, layer="calibration")
        exp.write(dxf)


def main() -> None:
    import argparse
    from dataclasses import replace

    parser = argparse.ArgumentParser(description="Rebuild N8Synth faceplate in build123d")
    parser.add_argument(
        "--export-mode",
        choices=("combined", "base", "labels"),
        default="combined",
        help="Which solids to generate (matches the OpenSCAD export_mode)",
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
        default=0.2,
        help="Depth in mm used for deboss/inlay modes (e.g. 0.2 = 2 layers at 0.1 mm)",
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
    args = parser.parse_args()

    params = replace(FaceplateParams(), text_mode=args.text_mode, inlay_depth=float(args.inlay_depth))

    # In non-emboss modes, inverse plaques aren't meaningful for 2-color inlay/deboss.
    if params.text_mode in ("deboss", "inlay") and params.inverse_label_enable:
        params = replace(params, inverse_label_enable=False)

    if args.template_svg is not None or args.template_dxf is not None:
        export_print_template(params, svg=args.template_svg, dxf=args.template_dxf)

    base, labels = build_faceplate(params, export_mode=args.export_mode)

    # Export
    if args.stl is not None:
        if args.export_mode == "combined":
            # For combined mode, export the fused solid.
            # (Viewer/printing typically prefers separate parts, but this matches the SCAD notion.)
            from build123d import Compound

            export_stl(Compound([o for o in (base, labels) if o is not None]), args.stl)
        elif args.export_mode == "base" and base is not None:
            export_stl(base, args.stl)
        elif args.export_mode == "labels" and labels is not None:
            export_stl(labels, args.stl)

    # Convenience exports (can be used alongside --export-mode)
    if args.stl_base is not None and base is not None:
        export_stl(base, args.stl_base)
    if args.stl_labels is not None and labels is not None:
        export_stl(labels, args.stl_labels)

    if args.step is not None:
        if args.export_mode == "combined":
            from build123d import Compound

            export_step(Compound([o for o in (base, labels) if o is not None]), args.step)
        elif args.export_mode == "base" and base is not None:
            export_step(base, args.step)
        elif args.export_mode == "labels" and labels is not None:
            export_step(labels, args.step)

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
