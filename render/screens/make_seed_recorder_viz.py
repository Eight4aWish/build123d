"""Compose the Seed Recorder visualisation: the Retrospective macOS menu-bar
app (mocked from mac-app/Retrospective/MenuBarContent.swift) above the 1U
Eurorack panel render. Output is portrait 1200x3000 to match the other
*_flat.png renders.

Output: render/out/seed_recorder/seed_recorder_flat.png
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 3000
HERE = Path(__file__).resolve().parents[1]
PANEL = HERE / "out" / "seed_recorder_panel" / "seed_recorder_panel_flat.png"
OUT = HERE / "out" / "seed_recorder" / "seed_recorder_flat.png"

# Fonts
_REG = ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf"]
_BOLD = ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/System/Library/Fonts/Helvetica.ttc"]


def font(paths, size):
    for p in paths:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


F_TITLE = font(_BOLD, 78)
F_SUB = font(_REG, 34)
F_HEAD = font(_BOLD, 34)
F_ITEM = font(_REG, 31)
F_SC = font(_REG, 29)
F_CAP = font(_REG, 32)


OUTER = (90, 92, 95)     # light-grey background (matches the *_flat renders)
INNER = (20, 21, 24)     # near-black backing rectangle
MARGIN = 36              # grey border thickness


def main() -> None:
    # Outer light-grey rectangle, then the near-black inner rectangle — same
    # framing as teensy_chaos_flat etc. The menu and panel sit on the black.
    img = Image.new("RGB", (W, H), OUTER)
    d = ImageDraw.Draw(img)
    d.rectangle([MARGIN, MARGIN, W - MARGIN, H - MARGIN], fill=INNER)

    def ctext(x, y, s, fnt, fill, anchor="la"):
        d.text((x, y), s, font=fnt, fill=fill, anchor=anchor)

    # --- dropdown menu (Retrospective -> Quit only) ---
    mw = 760
    mx0, mtop = (W - mw) // 2, 470
    pad = 22
    rows = [
        ("head", "Retrospective", ""),
        ("div", "", ""),
        ("text", "State: Idle", ""),
        ("text", "Capturing: 8 ch @ 48000 Hz", ""),
        ("text", "Frames/sec: 48000", ""),
        ("text", "Last save: 8 ch (2 silent skipped)", ""),
        ("div", "", ""),
        ("act", "Capture Now", "⌘C"),
        ("div", "", ""),
        ("act", "Settings…", "⌘,"),
        ("act", "Reveal Scratch in Finder", ""),
        ("hi", "Reveal Captures in Finder", ""),
        ("act", "Refresh Devices", "⌘R"),
        ("div", "", ""),
        ("act", "Quit", "⌘Q"),
    ]
    ROW, DIVH = 58, 26
    mh = pad * 2 + sum(DIVH if r[0] == "div" else ROW for r in rows)
    d.rounded_rectangle([mx0, mtop, mx0 + mw, mtop + mh], 18, fill=(40, 40, 45), outline=(70, 70, 78), width=2)

    y = mtop + pad
    tx = mx0 + 30
    sx = mx0 + mw - 30
    for kind, label, sc in rows:
        if kind == "div":
            d.line([(mx0 + 14, y + DIVH // 2), (mx0 + mw - 14, y + DIVH // 2)], fill=(72, 72, 80), width=2)
            y += DIVH
            continue
        if kind == "hi":
            d.rounded_rectangle([mx0 + 8, y + 4, mx0 + mw - 8, y + ROW - 4], 9, fill=(10, 122, 255))
            d.text((tx, y + ROW / 2), label, font=F_ITEM, fill=(255, 255, 255), anchor="lm")
            if sc:
                d.text((sx, y + ROW / 2), sc, font=F_SC, fill=(235, 240, 255), anchor="rm")
        else:
            fill = (245, 245, 248) if kind in ("head", "act") else (175, 178, 186)
            fnt = F_HEAD if kind == "head" else F_ITEM
            d.text((tx, y + ROW / 2), label, font=fnt, fill=fill, anchor="lm")
            if sc:
                d.text((sx, y + ROW / 2), sc, font=F_SC, fill=(140, 142, 150), anchor="rm")
        y += ROW

    # --- 1U panel (transparent render, cropped to its bounds) on the black ---
    if PANEL.exists():
        panel = Image.open(PANEL).convert("RGBA")
        bbox = panel.getbbox()
        if bbox:
            panel = panel.crop(bbox)
        target_w = 980
        scale = target_w / panel.width
        panel = panel.resize((target_w, int(panel.height * scale)), Image.LANCZOS)
        py = mtop + mh + 250
        px = (W - panel.width) // 2
        img.paste(panel, (px, py), panel)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f"wrote {OUT}  ({W}x{H})")


if __name__ == "__main__":
    main()
