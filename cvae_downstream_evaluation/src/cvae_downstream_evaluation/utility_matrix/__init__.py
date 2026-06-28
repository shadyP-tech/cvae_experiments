"""Diagnostic downstream utility matrix firewall."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..downstream import (
    CandidateDownstreamRow,
    read_candidate_downstream_matrix,
    write_candidate_downstream_matrix,
)
from ..protocol import ProtocolError

DIAGNOSTIC_MATRIX_BASENAME = "diagnostic_downstream_utility"


def assert_diagnostic_matrix_path(path: Path) -> None:
    """Ensure all-candidate utility matrices stay in diagnostic-only paths."""

    normalized = Path(path)
    if DIAGNOSTIC_MATRIX_BASENAME not in normalized.stem:
        raise ProtocolError(
            "All-candidate downstream utility matrices must be named "
            f"{DIAGNOSTIC_MATRIX_BASENAME}.*"
        )
    if "selections" in normalized.parts or "features" in normalized.parts:
        raise ProtocolError(f"Diagnostic downstream matrix cannot live under deployable path: {path}")


def assert_selection_does_not_read_matrix(path: Path) -> None:
    """Block deployable selection code from consuming diagnostic oracle matrices."""

    if DIAGNOSTIC_MATRIX_BASENAME in Path(path).name:
        raise ProtocolError(
            "Deployable selection cannot read diagnostic downstream utility matrices."
        )


def diagnostic_matrix_path(artifacts_root: Path, *, suffix: str = ".csv") -> Path:
    """Return the canonical quarantined diagnostic utility matrix path."""

    return Path(artifacts_root) / "tables" / f"{DIAGNOSTIC_MATRIX_BASENAME}{suffix}"


def write_diagnostic_downstream_matrix(
    path: Path,
    rows: Sequence[CandidateDownstreamRow],
) -> None:
    """Write all-candidate downstream utility with diagnostic-only naming."""

    assert_diagnostic_matrix_path(path)
    write_candidate_downstream_matrix(path, rows)


def read_diagnostic_downstream_matrix(path: Path) -> list[CandidateDownstreamRow]:
    """Read a diagnostic downstream matrix after checking quarantine naming."""

    assert_diagnostic_matrix_path(path)
    return read_candidate_downstream_matrix(path)
