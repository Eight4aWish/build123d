"""Minimal build123d + ocp_vscode demo.

Run with the workspace venv interpreter:
  ./.venv/bin/python show_build123d_demo.py

Then open the VS Code extension "OCP CAD Viewer" (if installed) to see the model.
"""

from __future__ import annotations

import os

from build123d import BuildPart, Box, Mode
from ocp_vscode import Camera, show


def _ocp_port(default: int = 3939) -> int:
    try:
        raw = os.environ.get("OCP_VSCODE_PORT") or os.environ.get("OCP_PORT") or str(default)
        return int(raw)
    except ValueError:
        return default


def main() -> None:
    # Create a simple 3D part.
    with BuildPart() as part:
        Box(40, 30, 10, mode=Mode.ADD)

    # Display it in the OCP CAD Viewer.
    try:
        show(part.part, reset_camera=Camera.RESET, grid=True, port=_ocp_port())
    except RuntimeError as ex:
        print("\nOCP viewer is not reachable.")
        print("- If you're using the VS Code extension: open 'OCP CAD Viewer' and ensure the backend is running.")
        print("- Or start the standalone viewer with: ./.venv/bin/python -m ocp_vscode --port 3939")
        print(f"\nDetails: {ex}")


if __name__ == "__main__":
    main()
