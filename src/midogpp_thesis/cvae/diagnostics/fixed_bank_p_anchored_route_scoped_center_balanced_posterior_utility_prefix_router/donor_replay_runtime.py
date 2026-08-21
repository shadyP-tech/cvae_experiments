"""Scoped-label pseudo-case utility replay for donor calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .canonical_probabilities import canonical_float32_probabilities, canonical_hash
from .eligibility import ActionCandidate
from .posterior_expected_utility import FavorableUtility, LOG_CLIP_EPSILON
from .utility_calibration import UtilityReplay


@dataclass(frozen=True)
class DonorReplayResult:
    replay: UtilityReplay
    label_scope: str
    label_count: int
    endpoint_lineage_hash: str
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        expected_scope = (
            f"PSEUDO_EVALUATION::H={self.replay.outer_center}::"
            f"J={self.replay.donor_center}::excluded_d={self.replay.case_id}"
        )
        if (
            self.label_scope != expected_scope
            or self.label_count <= 0
            or len(self.endpoint_lineage_hash) != 64
        ):
            raise ProtocolError("CBPUPR donor replay label capability drifted.")
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "cbpupr_donor_replay_result_v1",
                    "replay_hash": self.replay.replay_hash,
                    "label_scope": self.label_scope,
                    "label_count": self.label_count,
                    "endpoint_lineage_hash": self.endpoint_lineage_hash,
                    "raw_labels_persisted": False,
                }
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DonorReplayResult":
        row = cls(
            UtilityReplay.from_payload(payload["replay"]),  # type: ignore[arg-type]
            str(payload["label_scope"]),
            int(payload["label_count"]),
            str(payload["endpoint_lineage_hash"]),
        )
        if "result_hash" in payload and str(payload["result_hash"]) != row.result_hash:
            raise ProtocolError("CBPUPR donor replay result hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "replay": self.replay.to_payload(),
            "label_scope": self.label_scope,
            "label_count": self.label_count,
            "endpoint_lineage_hash": self.endpoint_lineage_hash,
            "raw_labels_persisted": False,
            "result_hash": self.result_hash,
        }


def realized_favorable_utility(
    portfolio_probabilities: object,
    candidate_probabilities: object,
    labels: Sequence[int],
    *,
    center_n_positive: int,
    center_n_negative: int,
    center_row_count: int,
    log_clip_epsilon: float = LOG_CLIP_EPSILON,
) -> FavorableUtility:
    """Realise one case contribution using fixed whole-center denominators."""

    p = canonical_float32_probabilities(portfolio_probabilities)
    a = canonical_float32_probabilities(candidate_probabilities, expected_length=len(p))
    y = np.asarray(tuple(labels), dtype=np.int8)
    if (
        y.shape != p.shape
        or bool(np.any((y != 0) & (y != 1)))
        or int(center_n_positive) <= 0
        or int(center_n_negative) <= 0
        or int(center_row_count) != int(center_n_positive) + int(center_n_negative)
        or int(np.sum(y)) > int(center_n_positive)
        or len(y) - int(np.sum(y)) > int(center_n_negative)
    ):
        raise ProtocolError("CBPUPR realised utility label/denominator drifted.")
    old_prediction = (p >= np.float32(0.5)).astype(np.float64)
    new_prediction = (a >= np.float32(0.5)).astype(np.float64)
    delta = new_prediction - old_prediction
    y64 = y.astype(np.float64)
    bacc = 0.5 * float(
        np.sum(
            delta
            * (
                y64 / int(center_n_positive)
                - (1.0 - y64) / int(center_n_negative)
            ),
            dtype=np.float64,
        )
    )
    p64 = p.astype(np.float64, copy=False)
    a64 = a.astype(np.float64, copy=False)
    brier = float(
        np.sum(p64 * p64 - a64 * a64 - 2.0 * y64 * (p64 - a64))
        / int(center_row_count)
    )
    epsilon = float(log_clip_epsilon)
    p_clip = np.clip(p64, epsilon, 1.0 - epsilon)
    a_clip = np.clip(a64, epsilon, 1.0 - epsilon)
    log_gain = float(
        np.sum(
            y64 * np.log(a_clip / p_clip)
            + (1.0 - y64)
            * np.log((1.0 - a_clip) / (1.0 - p_clip)),
            dtype=np.float64,
        )
        / int(center_row_count)
    )
    return FavorableUtility(bacc, brier, log_gain)


def replay_candidate(
    candidate: ActionCandidate,
    *,
    portfolio_probabilities: object,
    labels: Sequence[int],
    outer_center: str,
    donor_center: str,
    center_n_positive: int,
    center_n_negative: int,
    center_row_count: int,
    label_scope: str,
    source_excluded_centers: Sequence[str],
    endpoint_lineage_hash: str,
) -> DonorReplayResult:
    canonical_scope = (
        f"PSEUDO_EVALUATION::H={outer_center}::J={donor_center}::"
        f"excluded_d={candidate.case_id}"
    )
    if (
        candidate.center != str(donor_center)
        or str(outer_center) == str(donor_center)
        or label_scope != canonical_scope
    ):
        raise ProtocolError("CBPUPR donor replay route/label scope drifted.")
    excluded = tuple(sorted(set(str(value) for value in source_excluded_centers)))
    if set(excluded) != {str(outer_center), str(donor_center)}:
        raise ProtocolError("CBPUPR donor replay lineage must be exact H/J exclusion.")
    if len(str(endpoint_lineage_hash)) != 64:
        raise ProtocolError("CBPUPR donor replay endpoint lineage is unbound.")
    realized = realized_favorable_utility(
        portfolio_probabilities,
        candidate.probabilities.as_array(),
        labels,
        center_n_positive=center_n_positive,
        center_n_negative=center_n_negative,
        center_row_count=center_row_count,
    )
    replay = UtilityReplay(
        str(outer_center),
        str(donor_center),
        candidate.case_id,
        candidate.action_hash,
        candidate.estimate.utility,
        realized,
        excluded,
        candidate.control_id,
    )
    return DonorReplayResult(
        replay, str(label_scope), len(tuple(labels)), str(endpoint_lineage_hash)
    )


__all__ = (
    "DonorReplayResult",
    "realized_favorable_utility",
    "replay_candidate",
)
