"""Plot artifact helpers for diagnostic-only report outputs."""

from __future__ import annotations

from pathlib import Path


def diagnostic_plot_path(reports_root: Path, name: str) -> Path:
    """Return a normalized path for diagnostic plots.

    Plot paths are report artifacts only; they must not feed routing or feature
    construction.
    """

    safe_name = str(name).replace("/", "_").replace("\\", "_")
    if not safe_name.endswith(".png"):
        safe_name = f"{safe_name}.png"
    return Path(reports_root) / "plots" / safe_name
