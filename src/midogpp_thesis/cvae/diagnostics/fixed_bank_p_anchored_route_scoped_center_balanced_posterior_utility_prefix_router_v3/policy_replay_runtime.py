"""Leave-J pseudo-policy reconstruction and replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .canonical_probabilities import canonical_hash
from .candidate_runtime import CandidateRuntimeResult
from .policy_calibration import PolicyReplay
from .policy_prefixes import PrefixCandidate, PrefixSelection, select_prefix
from .posterior_expected_utility import FavorableUtility
from .utility_calibration import CenterBalancedUtilityCalibration


@dataclass(frozen=True)
class PolicyReplayRuntimeResult:
    selection: PrefixSelection
    replay: PolicyReplay
    candidate_calibration_hash: str
    candidate_runtime_hashes: tuple[str, ...]
    runtime_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.replay.policy_hash != self.selection.selection_hash
            or not self.candidate_runtime_hashes
            or len(set(self.candidate_runtime_hashes))
            != len(self.candidate_runtime_hashes)
        ):
            raise ProtocolError("CBPUPR replay policy and prefix selection disagree.")
        object.__setattr__(
            self,
            "runtime_hash",
            canonical_hash(
                {
                    "schema_version": "cbpupr_policy_replay_runtime_v1",
                    "selection_hash": self.selection.selection_hash,
                    "replay_hash": self.replay.replay_hash,
                    "candidate_calibration_hash": self.candidate_calibration_hash,
                    "candidate_runtime_hashes": list(self.candidate_runtime_hashes),
                }
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "PolicyReplayRuntimeResult":
        row = cls(
            PrefixSelection.from_payload(payload["selection"]),  # type: ignore[arg-type]
            PolicyReplay.from_payload(payload["replay"]),  # type: ignore[arg-type]
            str(payload["candidate_calibration_hash"]),
            tuple(str(value) for value in payload["candidate_runtime_hashes"]),  # type: ignore[index]
        )
        if "runtime_hash" in payload and str(payload["runtime_hash"]) != row.runtime_hash:
            raise ProtocolError("CBPUPR policy replay runtime hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "selection": self.selection.to_payload(),
            "replay": self.replay.to_payload(),
            "candidate_calibration_hash": self.candidate_calibration_hash,
            "candidate_runtime_hashes": list(self.candidate_runtime_hashes),
            "runtime_hash": self.runtime_hash,
        }


def replay_pseudo_policy(
    candidate_results: Sequence[CandidateRuntimeResult],
    realized_utility_by_candidate_hash: Mapping[str, FavorableUtility],
    *,
    outer_center: str,
    donor_center: str,
    leave_j_candidate_calibration: CenterBalancedUtilityCalibration,
) -> PolicyReplayRuntimeResult:
    """Select pseudo-J with a calibration built only from K outside {H,J}."""

    runtime_rows = tuple(candidate_results)
    rows = tuple(
        result.selected_candidate
        for result in runtime_rows
        if result.selected_candidate is not None
    )
    excluded = set(leave_j_candidate_calibration.calibration_excluded_centers)
    if (
        str(outer_center) not in excluded
        or str(donor_center) not in excluded
        or any(row.center != str(donor_center) for row in rows)
        or any(result.outer_center != str(outer_center) for result in runtime_rows)
        or any(result.center != str(donor_center) for result in runtime_rows)
        or any(
            set(result.source_excluded_centers)
            != {str(outer_center), str(donor_center)}
            for result in runtime_rows
        )
        or any(len(result.endpoint_lineage_hash) != 64 for result in runtime_rows)
        or any(row.action_hash not in realized_utility_by_candidate_hash for row in rows)
        or any(center in excluded for center in leave_j_candidate_calibration.supported_donor_centers)
    ):
        raise ProtocolError("CBPUPR pseudo policy violates leave-J calibration.")
    prefix_candidates = tuple(
        PrefixCandidate(
            row,
            leave_j_candidate_calibration.correct(row.estimate.utility),
            leave_j_candidate_calibration.calibration_hash,
        )
        for row in rows
    )
    selection = select_prefix(prefix_candidates)
    selected = selection.selected_candidates
    predicted = selection.evaluations[selection.selected_k].aggregate_utility
    realized = FavorableUtility.zeros()
    for row in selected:
        realized = realized + realized_utility_by_candidate_hash[
            row.candidate.action_hash
        ]
    replay = PolicyReplay(
        str(outer_center),
        str(donor_center),
        selection.selected_k,
        tuple(row.candidate.action_hash for row in selected),
        predicted,
        realized,
        tuple(sorted(excluded)),
        selection.selection_hash,
    )
    return PolicyReplayRuntimeResult(
        selection,
        replay,
        leave_j_candidate_calibration.calibration_hash,
        tuple(sorted(result.runtime_hash for result in runtime_rows)),
    )


__all__ = ("PolicyReplayRuntimeResult", "replay_pseudo_policy")
