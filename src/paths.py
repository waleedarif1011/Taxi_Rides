"""Shared filesystem paths with backward-compatible data lookup."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
ARTIFACTS_DIR = ROOT / "artifacts"
NOTEBOOKS_DIR = ROOT / "notebooks"

DEFAULT_CSV_NAME = "nyc_taxis.csv"


def resolve_data_path(explicit: str | Path | None = None) -> Path:
    """Locate ``nyc_taxis.csv`` under ``data/`` or the repo root (legacy)."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    candidates = [
        DATA_DIR / DEFAULT_CSV_NAME,
        ROOT / DEFAULT_CSV_NAME,
        NOTEBOOKS_DIR / DEFAULT_CSV_NAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find {DEFAULT_CSV_NAME}. Expected at {DATA_DIR / DEFAULT_CSV_NAME}."
    )
