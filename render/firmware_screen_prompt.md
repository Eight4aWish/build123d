# Prompt: extract the OLED screen content from a module's firmware

Paste the following into Claude Code (or Claude) **inside the firmware repo**
for a module that has a screen. It produces a self-contained PIL script that
regenerates the module's default/idle screen as a PNG, which the panel
renderer maps onto the OLED as an emissive texture.

Save the resulting script's output as
`render/out/<module>/<module>_screen.png` in the build123d repo.

---

You are looking at the firmware for a Eurorack module that drives a small
monochrome OLED (most likely SSD1306/SH1106, 128×64 or 128×32). I need a
faithful picture of what the display shows on power-up / its idle/home screen,
so I can composite it onto a 3D render of the front panel.

Please:

1. Find the display driver and its resolution. Confirm width×height in pixels
   and whether pixels are white/blue-on-black (typical OLED).
2. Locate the code that draws the **default / idle / home** screen (the screen
   shown when the module is powered on and untouched). If there are several
   modes, pick the boot/idle one and note the others.
3. Transcribe exactly what is drawn: every text string (with its font size /
   x,y position as best you can infer), and every graphic element (lines,
   rectangles, progress/level bars, waveforms, icons, inverted/highlighted
   regions, the cursor/selection state).
4. Write a **single self-contained Python script using Pillow (PIL)** that
   reproduces that screen and saves a PNG:
   - Canvas = native OLED resolution (e.g. 128×64), then upscale ×8 with
     **nearest-neighbour** so pixels stay crisp.
   - Background black `(0,0,0)`; lit pixels the OLED colour (white
     `(235,245,255)` for a typical blue-white OLED — note the real tint if you
     can tell).
   - Use a default PIL bitmap font if a specific font isn't easily available;
     approximate sizes/positions are fine, legibility matters most.
   - No external assets or network; only Pillow + stdlib.
   - Save to `<module>_screen.png`.
5. Print a short summary of the elements you reproduced and any assumptions.

Output: the spec summary + the complete runnable PIL script in one block.
