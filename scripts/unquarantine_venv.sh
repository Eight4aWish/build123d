#!/usr/bin/env bash
set -euo pipefail

# Clears macOS Gatekeeper quarantine flags from native extensions inside the venv.
# This fixes repeated “Not Opened (possible malware)” dialogs for .so/.dylib files.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VENV_DIR="$ROOT_DIR/.venv"
if [[ -d "$ROOT_DIR/.venv.nosync" ]]; then
  VENV_DIR="$ROOT_DIR/.venv.nosync"
fi

SITE_PACKAGES_DIR="$VENV_DIR/lib"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "No venv found at: $VENV_DIR" >&2
  echo "Create it first: python3 -m venv .venv" >&2
  echo "Tip (iCloud-friendly): mv .venv .venv.nosync" >&2
  exit 1
fi

# Find the active site-packages under .venv/lib/pythonX.Y/site-packages
SITE_PACKAGES_PATH="$(find "$VENV_DIR/lib" -maxdepth 3 -type d -path '*/site-packages' | head -n 1 || true)"

if [[ -z "${SITE_PACKAGES_PATH}" || ! -d "${SITE_PACKAGES_PATH}" ]]; then
  echo "Could not locate site-packages under: $SITE_PACKAGES_DIR" >&2
  exit 1
fi

echo "Clearing quarantine flags under: $SITE_PACKAGES_PATH"
# -r: recursive, -d: delete attribute
xattr -dr com.apple.quarantine "$SITE_PACKAGES_PATH" || true


echo "Done. If VS Code still errors, restart VS Code and retry."