"""Minimal build123d + ocp_vscode demo.

Run with the workspace venv interpreter:
  ./.venv/bin/python show_build123d_demo.py

Then open the VS Code extension "OCP CAD Viewer" (if installed) to see the model.
"""

from __future__ import annotations

from build123d import BuildPart, Box, Mode
from ocp_vscode import Camera, show


def main() -> None:
    # Create a simple 3D part.
    with BuildPart() as part:
        Box(40, 30, 10, mode=Mode.ADD)

    # Display it in the OCP CAD Viewer.
    show(part.part, reset_camera=Camera.RESET, grid=True)


if __name__ == "__main__":
    main()
