"""Per-outer-center pseudo-only scientific admission gates."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ...metrics import spearman
from ...protocol import ProtocolError
from .contracts import FavorableUtility
from .identity import canonical_hash


@dataclass(frozen=True)
class PseudoPolicyEvidence:
    outer_center: str
    donor_center: str
    predicted: FavorableUtility
    realized: FavorableUtility
    routed: bool
    jointly_safe: bool
    endpoint_oracle_bacc_gain: float
    absolute_oracle_regret: float
    legacy_realized: FavorableUtility
    legacy_routed: bool
    legacy_jointly_safe: bool
    legacy_absolute_oracle_regret: float
    evidence_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.endpoint_oracle_bacc_gain,
                self.absolute_oracle_regret,
                self.legacy_absolute_oracle_regret,
            ],
            dtype=np.float64,
        )
        if (
            not self.outer_center
            or not self.donor_center
            or self.outer_center == self.donor_center
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
        ):
            raise ProtocolError("P-DCAPS pseudo admission evidence drifted.")
        object.__setattr__(
            self,
            "evidence_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_pseudo_policy_evidence_v1",
                    "outer_center": self.outer_center,
                    "donor_center": self.donor_center,
                    "predicted": self.predicted.to_payload(),
                    "realized": self.realized.to_payload(),
                    "routed": self.routed,
                    "jointly_safe": self.jointly_safe,
                    "endpoint_oracle_bacc_gain": self.endpoint_oracle_bacc_gain,
                    "absolute_oracle_regret": self.absolute_oracle_regret,
                    "legacy_realized": self.legacy_realized.to_payload(),
                    "legacy_routed": self.legacy_routed,
                    "legacy_jointly_safe": self.legacy_jointly_safe,
                    "legacy_absolute_oracle_regret": self.legacy_absolute_oracle_regret,
                    "target_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True)
class OuterAdmission:
    outer_center: str
    donor_centers: tuple[str, ...]
    passed: bool
    reasons: tuple[str, ...]
    statistics: tuple[tuple[str, float], ...]
    evidence_hashes: tuple[str, ...]
    target_labels_opened: bool = False
    admission_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.target_labels_opened:
            raise ProtocolError("P-DCAPS admission consumed target labels.")
        object.__setattr__(
            self,
            "admission_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_outer_admission_v1",
                    "outer_center": self.outer_center,
                    "donor_centers": self.donor_centers,
                    "passed": self.passed,
                    "reasons": self.reasons,
                    "statistics": self.statistics,
                    "evidence_hashes": self.evidence_hashes,
                    "target_labels_opened": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_center": self.outer_center,
            "donor_centers": list(self.donor_centers),
            "passed": self.passed,
            "reasons": list(self.reasons),
            "statistics": dict(self.statistics),
            "evidence_hashes": list(self.evidence_hashes),
            "target_labels_opened": False,
            "admission_hash": self.admission_hash,
        }


def build_outer_admission(
    outer_center: str,
    evidence: Sequence[PseudoPolicyEvidence],
) -> OuterAdmission:
    rows = tuple(sorted(evidence, key=lambda row: row.donor_center))
    if (
        len(rows) < 6
        or len({row.donor_center for row in rows}) != len(rows)
        or {row.outer_center for row in rows} != {str(outer_center)}
    ):
        raise ProtocolError("P-DCAPS Admission_H donor inventory drifted.")
    predicted = [row.predicted.bacc_gain for row in rows]
    realized = [row.realized.bacc_gain for row in rows]
    brier_predicted = [row.predicted.brier_gain for row in rows]
    brier_realized = [row.realized.brier_gain for row in rows]
    log_predicted = [row.predicted.log_gain for row in rows]
    log_realized = [row.realized.log_gain for row in rows]
    correlations = (
        float(spearman(predicted, realized)),
        float(spearman(brier_predicted, brier_realized)),
        float(spearman(log_predicted, log_realized)),
    )
    routed = tuple(row for row in rows if row.routed)
    legacy_routed = tuple(row for row in rows if row.legacy_routed)
    safe_rate = (
        sum(row.jointly_safe for row in routed) / len(routed) if routed else 0.0
    )
    legacy_safe_rate = (
        sum(row.legacy_jointly_safe for row in legacy_routed) / len(legacy_routed)
        if legacy_routed
        else 0.0
    )
    denominators_valid = all(
        math.isfinite(row.endpoint_oracle_bacc_gain)
        and row.endpoint_oracle_bacc_gain > 0.0
        for row in rows
    )
    normalized_gap = (
        float(
            np.mean(
                [
                    row.absolute_oracle_regret / row.endpoint_oracle_bacc_gain
                    for row in rows
                ],
                dtype=np.float64,
            )
        )
        if denominators_valid
        else math.inf
    )
    legacy_normalized_gap = (
        float(
            np.mean(
                [
                    row.legacy_absolute_oracle_regret
                    / row.endpoint_oracle_bacc_gain
                    for row in rows
                ],
                dtype=np.float64,
            )
        )
        if denominators_valid
        else math.inf
    )
    statistics = {
        "routed_policy_count": float(len(routed)),
        "bacc_spearman": correlations[0],
        "brier_spearman": correlations[1],
        "log_spearman": correlations[2],
        "equal_center_realized_bacc": float(np.mean(realized, dtype=np.float64)),
        "joint_safe_routed_rate": safe_rate,
        "legacy_joint_safe_routed_rate": legacy_safe_rate,
        "absolute_oracle_regret": float(
            np.mean([row.absolute_oracle_regret for row in rows], dtype=np.float64)
        ),
        "legacy_absolute_oracle_regret": float(
            np.mean(
                [row.legacy_absolute_oracle_regret for row in rows],
                dtype=np.float64,
            )
        ),
        "normalized_oracle_gap": normalized_gap,
        "legacy_normalized_oracle_gap": legacy_normalized_gap,
    }
    reasons: list[str] = []
    gates = (
        (len(routed) > 0, "NO_NONTRIVIAL_PSEUDO_ROUTING"),
        (math.isfinite(correlations[0]) and correlations[0] > 0.0, "NONPOSITIVE_BACC_SPEARMAN"),
        (math.isfinite(correlations[1]) and correlations[1] >= 0.0, "NEGATIVE_BRIER_SPEARMAN"),
        (math.isfinite(correlations[2]) and correlations[2] >= 0.0, "NEGATIVE_LOG_SPEARMAN"),
        (statistics["equal_center_realized_bacc"] > 0.0, "NONPOSITIVE_REALIZED_BACC"),
        (safe_rate >= legacy_safe_rate, "JOINT_SAFE_RATE_BELOW_LEGACY"),
        (
            statistics["absolute_oracle_regret"]
            <= statistics["legacy_absolute_oracle_regret"],
            "ABSOLUTE_ORACLE_REGRET_ABOVE_LEGACY",
        ),
        (denominators_valid, "INVALID_NORMALIZED_ORACLE_DENOMINATOR"),
        (normalized_gap <= legacy_normalized_gap, "NORMALIZED_ORACLE_GAP_ABOVE_LEGACY"),
    )
    reasons.extend(reason for passed, reason in gates if not passed)
    return OuterAdmission(
        str(outer_center),
        tuple(row.donor_center for row in rows),
        not reasons,
        tuple(reasons) if reasons else ("PSEUDO_ONLY_ADMISSION_PASS",),
        tuple(sorted((key, float(value)) for key, value in statistics.items())),
        tuple(row.evidence_hash for row in rows),
    )


__all__ = ("OuterAdmission", "PseudoPolicyEvidence", "build_outer_admission")
