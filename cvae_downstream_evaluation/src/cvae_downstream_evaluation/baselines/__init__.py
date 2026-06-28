"""Deployable baseline candidate-pool checks."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..protocol import ProtocolError
from ..schemas import DIAGNOSTIC_ONLY, SELECTION_ELIGIBLE


def assert_deployable_candidate_pool(
    *,
    heldout_target: str,
    candidate_rows: Sequence[Mapping[str, object]],
) -> None:
    """Reject target-expert leakage in deployable candidates and baselines."""

    deployable = [row for row in candidate_rows if str(row.get("eligibility")) == SELECTION_ELIGIBLE]
    if not deployable:
        raise ProtocolError("Deployable candidate pool is empty.")
    for row in deployable:
        source_domain = str(row.get("source_domain", ""))
        expert_identity = str(row.get("expert_checkpoint_id", ""))
        if source_domain == str(heldout_target) or expert_identity == str(heldout_target):
            raise ProtocolError(
                f"Target expert leakage in deployable candidate pool for target={heldout_target}: {row}"
            )


def assert_oracle_rows_diagnostic_only(candidate_rows: Sequence[Mapping[str, object]]) -> None:
    for row in candidate_rows:
        role = str(row.get("role", ""))
        if "oracle" in role and str(row.get("eligibility")) != DIAGNOSTIC_ONLY:
            raise ProtocolError(f"Oracle candidate rows must be diagnostic_only: {row}")
