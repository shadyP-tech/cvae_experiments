"""Frozen candidate-only, observed-maximum and cyclic controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .canonical_probabilities import canonical_hash
from .eligibility import ActionCandidate, assess_action
from .policy_prefixes import PrefixCandidate, PrefixSelection, select_prefix
from .posterior_expected_utility import FavorableUtility
from .utility_calibration import UtilityReplay


CANDIDATE_ONLY_METHOD_ID = "CBPUPR_UNIFIED_CANDIDATE_ONLY"
OBSERVED_MAX_METHOD_ID = "CBPUPR_OBSERVED_MAX_PREFIX_CONTROL"
CYCLIC_METHOD_ID = "CBPUPR_CYCLIC_FINGERPRINT_PREFIX_CONTROL"


@dataclass(frozen=True)
class ControlPolicy:
    method_id: str
    selected_candidate_hashes: tuple[str, ...]
    aggregate_utility: FavorableUtility
    authorized: bool
    source_hashes: tuple[str, ...]
    policy_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.method_id
            not in (CANDIDATE_ONLY_METHOD_ID, OBSERVED_MAX_METHOD_ID, CYCLIC_METHOD_ID)
            or len(set(self.selected_candidate_hashes))
            != len(self.selected_candidate_hashes)
            or self.authorized != bool(self.selected_candidate_hashes)
        ):
            raise ProtocolError("CBPUPR control policy contract drifted.")
        object.__setattr__(
            self,
            "policy_hash",
            canonical_hash(
                {
                    "schema_version": "cbpupr_control_policy_v1",
                    "method_id": self.method_id,
                    "selected_candidate_hashes": list(self.selected_candidate_hashes),
                    "aggregate_utility": self.aggregate_utility.to_payload(),
                    "authorized": self.authorized,
                    "source_hashes": list(self.source_hashes),
                }
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ControlPolicy":
        row = cls(
            str(payload["method_id"]),
            tuple(str(value) for value in payload["selected_candidate_hashes"]),  # type: ignore[index]
            FavorableUtility.from_payload(payload["aggregate_utility"]),  # type: ignore[arg-type]
            bool(payload["authorized"]),
            tuple(str(value) for value in payload["source_hashes"]),  # type: ignore[index]
        )
        if "policy_hash" in payload and str(payload["policy_hash"]) != row.policy_hash:
            raise ProtocolError("CBPUPR control policy hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "selected_candidate_hashes": list(self.selected_candidate_hashes),
            "aggregate_utility": self.aggregate_utility.to_payload(),
            "authorized": self.authorized,
            "source_hashes": list(self.source_hashes),
            "policy_hash": self.policy_hash,
        }


def candidate_only_control(candidates: Sequence[ActionCandidate]) -> ControlPolicy:
    """Apply every independently eligible case action with no policy correction."""

    rows = tuple(sorted(candidates, key=lambda row: (row.case_id, row.action_hash)))
    if (
        len({row.case_id for row in rows}) != len(rows)
        or any(not assess_action(row).eligible for row in rows)
    ):
        raise ProtocolError("CBPUPR candidate-only control repeats a case.")
    aggregate = FavorableUtility.zeros()
    for row in rows:
        aggregate = aggregate + row.estimate.utility
    return ControlPolicy(
        CANDIDATE_ONLY_METHOD_ID,
        tuple(row.action_hash for row in rows),
        aggregate,
        bool(rows),
        tuple(row.estimate.estimate_hash for row in rows),
    )


def observed_maximum_bias(replays: Sequence[UtilityReplay]) -> FavorableUtility:
    rows = tuple(replays)
    if not rows:
        raise ProtocolError("CBPUPR observed-maximum control lacks replays.")
    matrix = np.asarray([row.overprediction.as_tuple() for row in rows], dtype=np.float64)
    values = np.max(matrix, axis=0)
    return FavorableUtility(*(float(value) for value in values))


def observed_maximum_prefix_control(
    candidates: Sequence[ActionCandidate],
    replays: Sequence[UtilityReplay],
) -> tuple[PrefixSelection, ControlPolicy]:
    """Reproduce the deliberately conservative global-maximum sensitivity."""

    margin = observed_maximum_bias(replays)
    source_hash = canonical_hash(
        {
            "schema_version": "cbpupr_observed_maximum_bias_v1",
            "margin": margin.to_payload(),
            "replay_hashes": sorted(row.replay_hash for row in replays),
        }
    )
    prefix_candidates = tuple(
        PrefixCandidate(row, row.estimate.utility - margin, source_hash)
        for row in candidates
    )
    selection = select_prefix(prefix_candidates)
    selected = selection.selected_candidates
    aggregate = selection.evaluations[selection.selected_k].aggregate_utility
    control = ControlPolicy(
        OBSERVED_MAX_METHOD_ID,
        tuple(row.candidate.action_hash for row in selected),
        aggregate,
        selection.authorized,
        (source_hash, selection.selection_hash),
    )
    return selection, control


def cyclically_shift_within_case(
    values: Sequence[float],
    *,
    shift: int = 1,
) -> tuple[float, ...]:
    """Deterministic nonzero complete-case cyclic fingerprint control."""

    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ProtocolError("CBPUPR cyclic control input drifted.")
    if len(array) == 1:
        return (float(array[0]),)
    offset = int(shift) % len(array)
    if offset == 0:
        raise ProtocolError("CBPUPR cyclic control shift must be a derangement offset.")
    return tuple(float(value) for value in np.roll(array, offset))


def cyclic_control_policy(selection: PrefixSelection) -> ControlPolicy:
    selected = selection.selected_candidates
    aggregate = selection.evaluations[selection.selected_k].aggregate_utility
    return ControlPolicy(
        CYCLIC_METHOD_ID,
        tuple(row.candidate.action_hash for row in selected),
        aggregate,
        selection.authorized,
        (selection.selection_hash,),
    )


__all__ = (
    "CANDIDATE_ONLY_METHOD_ID",
    "CYCLIC_METHOD_ID",
    "ControlPolicy",
    "OBSERVED_MAX_METHOD_ID",
    "candidate_only_control",
    "cyclic_control_policy",
    "cyclically_shift_within_case",
    "observed_maximum_bias",
    "observed_maximum_prefix_control",
)
