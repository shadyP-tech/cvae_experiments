"""Thin orchestration shim for diagnostic all-candidate downstream runs."""

from __future__ import annotations

from pathlib import Path

from ..artifacts import assert_frozen_snapshot_exists
from . import assert_diagnostic_matrix_path


def validate_run_preconditions(*, output_matrix: Path, frozen_snapshot: Path) -> None:
    """Validate filesystem-level firewalls before heavy downstream evaluation."""

    assert_diagnostic_matrix_path(output_matrix)
    assert_frozen_snapshot_exists(frozen_snapshot)
