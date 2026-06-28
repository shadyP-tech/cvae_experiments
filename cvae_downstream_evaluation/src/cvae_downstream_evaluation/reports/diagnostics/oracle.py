"""Diagnostic-only downstream oracle helpers.

This module intentionally lives outside deployable baseline and selection
packages. It may quantify oracle gaps, but it must not be imported by adoption-
eligible routing code.
"""

from __future__ import annotations

from typing import Sequence

from ...downstream import CandidateDownstreamRow, OracleScore, compute_single_expert_oracles


def diagnostic_single_expert_oracles(
    rows: Sequence[CandidateDownstreamRow],
) -> dict[tuple[int, str, str, int, int, int], OracleScore]:
    return compute_single_expert_oracles(rows)
