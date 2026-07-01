"""Optional SAIL adapter.

SAIL is a reference for MIDOG++ real-feature diagnostic semantics in this repo,
not a required runtime dependency for the independent gate package.
"""

from __future__ import annotations

from pathlib import Path


def compare_sail_semantics(repo_root: Path) -> dict[str, object]:
    """Compare split/artifact semantics against the current SAIL diagnostics.

    This avoids a runtime dependency while checking that the reference files are
    present where this repo currently defines MIDOG++ real-feature diagnostics.
    """
    root = Path(repo_root)
    references = {
        "multiaxis_module": root / "sail" / "src" / "sail" / "midogpp_multiaxis.py",
        "signal_controls_module": root / "sail" / "src" / "sail" / "midogpp_signal_controls.py",
        "multiaxis_config": root / "sail" / "configs" / "midogpp_virchow2_real_feature_multiaxis_baseline.yaml",
        "signal_controls_config": root / "sail" / "configs" / "midogpp_virchow2_real_feature_signal_controls.yaml",
    }
    return {
        "status": "PASS" if all(path.exists() for path in references.values()) else "MISSING_REFERENCE",
        "references": {key: str(path) for key, path in references.items()},
        "missing": [key for key, path in references.items() if not path.exists()],
        "runtime_dependency": False,
    }
