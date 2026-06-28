"""Diagnostic-only oracle wrappers for all-candidate utility matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..downstream import CandidateDownstreamRow, OracleScore, compute_single_expert_oracles
from . import assert_diagnostic_matrix_path


def compute_diagnostic_oracles(
    rows: Sequence[CandidateDownstreamRow],
) -> dict[tuple[int, str, str, int, int, int], OracleScore]:
    return compute_single_expert_oracles(rows)


def validate_diagnostic_output_path(path: Path) -> None:
    assert_diagnostic_matrix_path(path)
