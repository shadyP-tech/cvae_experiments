"""Deterministic case ranking and aggregate prefix policy selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .canonical_probabilities import canonical_hash
from .eligibility import ActionCandidate, UTILITY_ZERO_TOLERANCE
from .posterior_expected_utility import FavorableUtility


@dataclass(frozen=True)
class PrefixCandidate:
    candidate: ActionCandidate
    corrected_utility: FavorableUtility
    calibration_hash: str
    policy_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.calibration_hash:
            raise ProtocolError("CBPUPR prefix candidate lacks calibration lineage.")
        object.__setattr__(
            self,
            "policy_hash",
            canonical_hash(
                {
                    "schema_version": "cbpupr_prefix_candidate_v1",
                    "center": self.candidate.center,
                    "case_id": self.candidate.case_id,
                    "action_hash": self.candidate.action_hash,
                    "corrected_utility": self.corrected_utility.to_payload(),
                    "calibration_hash": self.calibration_hash,
                }
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "PrefixCandidate":
        row = cls(
            ActionCandidate.from_payload(payload["candidate"]),  # type: ignore[arg-type]
            FavorableUtility.from_payload(payload["corrected_utility"]),  # type: ignore[arg-type]
            str(payload["calibration_hash"]),
        )
        if (
            "policy_hash" in payload
            and str(payload["policy_hash"]) != row.policy_hash
        ):
            raise ProtocolError("CBPUPR prefix policy hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_payload(),
            "corrected_utility": self.corrected_utility.to_payload(),
            "calibration_hash": self.calibration_hash,
            "policy_hash": self.policy_hash,
        }


@dataclass(frozen=True)
class PrefixEvaluation:
    k: int
    candidate_hashes: tuple[str, ...]
    aggregate_utility: FavorableUtility
    feasible: bool
    reason_codes: tuple[str, ...]
    prefix_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.k < 0 or self.k != len(self.candidate_hashes) or not self.reason_codes:
            raise ProtocolError("CBPUPR prefix evaluation topology drifted.")
        if len(set(self.candidate_hashes)) != len(self.candidate_hashes):
            raise ProtocolError("CBPUPR prefix repeats a candidate.")
        object.__setattr__(
            self,
            "prefix_hash",
            canonical_hash(
                {
                    "schema_version": "cbpupr_prefix_evaluation_v1",
                    "k": self.k,
                    "candidate_hashes": list(self.candidate_hashes),
                    "aggregate_utility": self.aggregate_utility.to_payload(),
                    "feasible": self.feasible,
                    "reason_codes": list(self.reason_codes),
                }
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "PrefixEvaluation":
        row = cls(
            int(payload["k"]),
            tuple(str(value) for value in payload["candidate_hashes"]),  # type: ignore[index]
            FavorableUtility.from_payload(payload["aggregate_utility"]),  # type: ignore[arg-type]
            bool(payload["feasible"]),
            tuple(str(value) for value in payload["reason_codes"]),  # type: ignore[index]
        )
        if "prefix_hash" in payload and str(payload["prefix_hash"]) != row.prefix_hash:
            raise ProtocolError("CBPUPR prefix evaluation hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "k": self.k,
            "candidate_hashes": list(self.candidate_hashes),
            "aggregate_utility": self.aggregate_utility.to_payload(),
            "feasible": self.feasible,
            "reason_codes": list(self.reason_codes),
            "prefix_hash": self.prefix_hash,
        }


@dataclass(frozen=True)
class PrefixSelection:
    ranked_candidates: tuple[PrefixCandidate, ...]
    evaluations: tuple[PrefixEvaluation, ...]
    selected_k: int
    selected_prefix_hash: str
    selection_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            len({row.candidate.case_id for row in self.ranked_candidates})
            != len(self.ranked_candidates)
            or tuple(row.k for row in self.evaluations)
            != tuple(range(len(self.ranked_candidates) + 1))
            or not 0 <= self.selected_k <= len(self.ranked_candidates)
            or self.evaluations[self.selected_k].prefix_hash != self.selected_prefix_hash
            or not self.evaluations[self.selected_k].feasible
        ):
            raise ProtocolError("CBPUPR prefix selection contract drifted.")
        object.__setattr__(
            self,
            "selection_hash",
            canonical_hash(
                {
                    "schema_version": "cbpupr_prefix_selection_v1",
                    "ranked_policy_hashes": [
                        row.policy_hash for row in self.ranked_candidates
                    ],
                    "evaluation_hashes": [row.prefix_hash for row in self.evaluations],
                    "selected_k": self.selected_k,
                    "selected_prefix_hash": self.selected_prefix_hash,
                }
            ),
        )

    @property
    def authorized(self) -> bool:
        return self.selected_k > 0

    @property
    def selected_candidates(self) -> tuple[PrefixCandidate, ...]:
        return self.ranked_candidates[: self.selected_k]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "PrefixSelection":
        row = cls(
            tuple(
                PrefixCandidate.from_payload(value)
                for value in payload["ranked_candidates"]  # type: ignore[index]
            ),
            tuple(
                PrefixEvaluation.from_payload(value)
                for value in payload["evaluations"]  # type: ignore[index]
            ),
            int(payload["selected_k"]),
            str(payload["selected_prefix_hash"]),
        )
        if "selection_hash" in payload and str(payload["selection_hash"]) != row.selection_hash:
            raise ProtocolError("CBPUPR prefix selection hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "ranked_candidates": [row.to_payload() for row in self.ranked_candidates],
            "evaluations": [row.to_payload() for row in self.evaluations],
            "selected_k": self.selected_k,
            "selected_prefix_hash": self.selected_prefix_hash,
            "selection_hash": self.selection_hash,
        }


def rank_prefix_candidates(
    candidates: Sequence[PrefixCandidate],
) -> tuple[PrefixCandidate, ...]:
    rows = tuple(candidates)
    if len({row.candidate.case_id for row in rows}) != len(rows):
        raise ProtocolError("CBPUPR policy has multiple actions for one case.")
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                -row.corrected_utility.bacc_gain,
                row.candidate.case_id,
                row.policy_hash,
            ),
        )
    )


def enumerate_prefixes(
    candidates: Sequence[PrefixCandidate],
    *,
    tolerance: float = UTILITY_ZERO_TOLERANCE,
) -> tuple[PrefixEvaluation, ...]:
    ranked = rank_prefix_candidates(candidates)
    result = [
        PrefixEvaluation(0, (), FavorableUtility.zeros(), True, ("EXACT_P_BASELINE",))
    ]
    aggregate = FavorableUtility.zeros()
    hashes: list[str] = []
    for index, row in enumerate(ranked, start=1):
        aggregate = aggregate + row.corrected_utility
        hashes.append(row.candidate.action_hash)
        reasons: list[str] = []
        if aggregate.bacc_gain <= float(tolerance):
            reasons.append("NONPOSITIVE_AGGREGATE_BACC")
        if aggregate.brier_gain < -float(tolerance):
            reasons.append("NEGATIVE_AGGREGATE_BRIER_GAIN")
        if aggregate.log_gain < -float(tolerance):
            reasons.append("NEGATIVE_AGGREGATE_LOG_GAIN")
        result.append(
            PrefixEvaluation(
                index,
                tuple(hashes),
                aggregate,
                not reasons,
                ("AGGREGATE_UTILITY_PASS",) if not reasons else tuple(reasons),
            )
        )
    return tuple(result)


def select_prefix(
    candidates: Sequence[PrefixCandidate],
    *,
    tolerance: float = UTILITY_ZERO_TOLERANCE,
) -> PrefixSelection:
    """Select the feasible prefix with maximal favourable BACC.

    Exact/tolerance utility ties resolve to smaller K and then prefix hash.
    """

    ranked = rank_prefix_candidates(candidates)
    evaluations = enumerate_prefixes(ranked, tolerance=tolerance)
    feasible = tuple(row for row in evaluations if row.feasible)
    maximum = max(row.aggregate_utility.bacc_gain for row in feasible)
    tied = tuple(
        row
        for row in feasible
        if abs(row.aggregate_utility.bacc_gain - maximum) <= float(tolerance)
    )
    selected = min(tied, key=lambda row: (row.k, row.prefix_hash))
    return PrefixSelection(ranked, evaluations, selected.k, selected.prefix_hash)


__all__ = (
    "PrefixCandidate",
    "PrefixEvaluation",
    "PrefixSelection",
    "enumerate_prefixes",
    "rank_prefix_candidates",
    "select_prefix",
)
