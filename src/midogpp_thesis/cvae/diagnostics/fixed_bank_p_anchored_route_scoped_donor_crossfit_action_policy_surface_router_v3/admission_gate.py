"""Pure, label-free Admission-H computation for P-DCAPS v3."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from ...metrics import spearman
from ...protocol import ProtocolError
from .identity import canonical_hash, require_sha256
from .nullable_statistics import (
    ADMISSION_STATISTIC_NAMES,
    CONSTANT_RANK_UNDEFINED_REASON,
    DENOMINATOR_UNDEFINED_REASON,
    NullableStatistic,
    strict_bool,
)
from .pseudo_evidence import PseudoPolicyEvidence


OUTER_ADMISSION_SCHEMA = "pdcaps_v3_outer_admission_v1"


@dataclass(frozen=True)
class OuterAdmission:
    """Canonical result of one outer-center pseudo-only admission gate."""

    outer_center: str
    donor_centers: tuple[str, ...]
    passed: bool
    reasons: tuple[str, ...]
    statistics: tuple[NullableStatistic, ...]
    evidence_hashes: tuple[str, ...]
    target_labels_opened: bool = False
    admission_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = str(self.outer_center)
        donors = tuple(str(value) for value in self.donor_centers)
        passed = strict_bool(self.passed, "outer admission pass flag")
        reasons = tuple(str(value) for value in self.reasons)
        statistics = tuple(self.statistics)
        hashes = tuple(
            require_sha256(value, "pseudo evidence hash")
            for value in self.evidence_hashes
        )
        if (
            not outer
            or len(donors) < 6
            or tuple(sorted(donors)) != donors
            or len(set(donors)) != len(donors)
            or outer in donors
            or len(hashes) != len(donors)
            or len(set(hashes)) != len(hashes)
            or type(self.target_labels_opened) is not bool
            or self.target_labels_opened
            or any(not isinstance(row, NullableStatistic) for row in statistics)
            or tuple(row.name for row in statistics) != ADMISSION_STATISTIC_NAMES
            or not reasons
            or (passed and reasons != ("PSEUDO_ONLY_ADMISSION_PASS",))
            or (not passed and "PSEUDO_ONLY_ADMISSION_PASS" in reasons)
        ):
            raise ProtocolError("P-DCAPS v3 outer admission contract drifted.")
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(self, "donor_centers", donors)
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "statistics", statistics)
        object.__setattr__(self, "evidence_hashes", hashes)
        object.__setattr__(
            self,
            "admission_hash",
            canonical_hash(self._payload_without_hash()),
        )

    @property
    def statistics_by_name(self) -> dict[str, NullableStatistic]:
        return {row.name: row for row in self.statistics}

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": OUTER_ADMISSION_SCHEMA,
            "outer_center": self.outer_center,
            "donor_centers": list(self.donor_centers),
            "passed": self.passed,
            "reasons": list(self.reasons),
            "statistics": [row.to_payload() for row in self.statistics],
            "evidence_hashes": list(self.evidence_hashes),
            "target_labels_opened": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "admission_hash": self.admission_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "OuterAdmission":
        expected = {
            "schema_version",
            "outer_center",
            "donor_centers",
            "passed",
            "reasons",
            "statistics",
            "evidence_hashes",
            "target_labels_opened",
            "admission_hash",
        }
        if (
            not isinstance(payload, Mapping)
            or set(payload) != expected
            or payload.get("schema_version") != OUTER_ADMISSION_SCHEMA
            or payload.get("target_labels_opened") is not False
        ):
            raise ProtocolError("P-DCAPS v3 persisted admission schema drifted.")
        donors = payload["donor_centers"]
        reasons = payload["reasons"]
        statistics = payload["statistics"]
        hashes = payload["evidence_hashes"]
        if not all(
            isinstance(value, (list, tuple))
            for value in (donors, reasons, statistics, hashes)
        ):
            raise ProtocolError("P-DCAPS v3 persisted admission rows drifted.")
        result = cls(
            str(payload["outer_center"]),
            tuple(str(value) for value in donors),
            strict_bool(payload["passed"], "outer admission pass flag"),
            tuple(str(value) for value in reasons),
            tuple(
                NullableStatistic.from_payload(row)
                for row in statistics
                if isinstance(row, Mapping)
            ),
            tuple(str(value) for value in hashes),
            False,
        )
        if len(result.statistics) != len(statistics):
            raise ProtocolError("P-DCAPS v3 persisted statistic row drifted.")
        if result.admission_hash != require_sha256(
            payload["admission_hash"], "persisted outer admission hash"
        ):
            raise ProtocolError("P-DCAPS v3 persisted admission hash drifted.")
        return result


def _rank_statistic(
    name: str,
    predicted: Sequence[float],
    realized: Sequence[float],
) -> NullableStatistic:
    predicted_array = np.asarray(predicted, dtype=np.float64)
    realized_array = np.asarray(realized, dtype=np.float64)
    if (
        predicted_array.shape != realized_array.shape
        or predicted_array.ndim != 1
        or len(predicted_array) < 2
        or not np.isfinite(predicted_array).all()
        or not np.isfinite(realized_array).all()
    ):
        raise ProtocolError("P-DCAPS v3 correlation input drifted.")
    predicted_constant = bool(np.all(predicted_array == predicted_array[0]))
    realized_constant = bool(np.all(realized_array == realized_array[0]))
    if predicted_constant or realized_constant:
        return NullableStatistic.undefined(name, CONSTANT_RANK_UNDEFINED_REASON)
    value = float(spearman(predicted_array, realized_array))
    if not math.isfinite(value):
        raise ProtocolError("P-DCAPS v3 correlation kernel returned nonfinite.")
    return NullableStatistic.finite(name, value)


def build_outer_admission(
    outer_center: str,
    evidence: Sequence[PseudoPolicyEvidence],
) -> OuterAdmission:
    """Build label-free Admission-H, failing closed on every undefined value."""

    rows = tuple(sorted(tuple(evidence), key=lambda row: row.donor_center))
    outer = str(outer_center)
    if (
        len(rows) < 6
        or any(not isinstance(row, PseudoPolicyEvidence) for row in rows)
        or len({row.donor_center for row in rows}) != len(rows)
        or {row.outer_center for row in rows} != {outer}
    ):
        raise ProtocolError("P-DCAPS v3 Admission_H donor inventory drifted.")

    predicted = {
        "bacc": [row.predicted.bacc_gain for row in rows],
        "brier": [row.predicted.brier_gain for row in rows],
        "log": [row.predicted.log_gain for row in rows],
    }
    realized = {
        "bacc": [row.realized.bacc_gain for row in rows],
        "brier": [row.realized.brier_gain for row in rows],
        "log": [row.realized.log_gain for row in rows],
    }
    routed = tuple(row for row in rows if row.routed)
    legacy_routed = tuple(row for row in rows if row.legacy_routed)
    safe_rate = (
        sum(row.jointly_safe for row in routed) / len(routed) if routed else 0.0
    )
    legacy_safe_rate = (
        sum(row.legacy_jointly_safe for row in legacy_routed)
        / len(legacy_routed)
        if legacy_routed
        else 0.0
    )
    absolute_regret = float(
        np.mean([row.absolute_oracle_regret for row in rows], dtype=np.float64)
    )
    legacy_absolute_regret = float(
        np.mean(
            [row.legacy_absolute_oracle_regret for row in rows],
            dtype=np.float64,
        )
    )
    denominators_valid = all(
        row.endpoint_oracle_bacc_gain > 0.0 for row in rows
    )
    if denominators_valid:
        normalized = NullableStatistic.finite(
            "normalized_oracle_gap",
            np.mean(
                [
                    row.absolute_oracle_regret / row.endpoint_oracle_bacc_gain
                    for row in rows
                ],
                dtype=np.float64,
            ),
        )
        legacy_normalized = NullableStatistic.finite(
            "legacy_normalized_oracle_gap",
            np.mean(
                [
                    row.legacy_absolute_oracle_regret
                    / row.endpoint_oracle_bacc_gain
                    for row in rows
                ],
                dtype=np.float64,
            ),
        )
    else:
        normalized = NullableStatistic.undefined(
            "normalized_oracle_gap", DENOMINATOR_UNDEFINED_REASON
        )
        legacy_normalized = NullableStatistic.undefined(
            "legacy_normalized_oracle_gap", DENOMINATOR_UNDEFINED_REASON
        )

    statistics = (
        NullableStatistic.finite("routed_policy_count", len(routed)),
        _rank_statistic("bacc_spearman", predicted["bacc"], realized["bacc"]),
        _rank_statistic("brier_spearman", predicted["brier"], realized["brier"]),
        _rank_statistic("log_spearman", predicted["log"], realized["log"]),
        NullableStatistic.finite(
            "equal_center_realized_bacc",
            np.mean(realized["bacc"], dtype=np.float64),
        ),
        NullableStatistic.finite("joint_safe_routed_rate", safe_rate),
        NullableStatistic.finite(
            "legacy_joint_safe_routed_rate", legacy_safe_rate
        ),
        NullableStatistic.finite("absolute_oracle_regret", absolute_regret),
        NullableStatistic.finite(
            "legacy_absolute_oracle_regret", legacy_absolute_regret
        ),
        normalized,
        legacy_normalized,
    )
    by_name = {row.name: row for row in statistics}
    reasons: list[str] = []
    if not routed:
        reasons.append("NO_NONTRIVIAL_PSEUDO_ROUTING")
    for metric, threshold, strict, failure in (
        ("bacc", 0.0, True, "NONPOSITIVE_BACC_SPEARMAN"),
        ("brier", 0.0, False, "NEGATIVE_BRIER_SPEARMAN"),
        ("log", 0.0, False, "NEGATIVE_LOG_SPEARMAN"),
    ):
        statistic = by_name[f"{metric}_spearman"]
        if not statistic.defined:
            reasons.append(
                f"UNDEFINED_{metric.upper()}_SPEARMAN::"
                f"{statistic.undefined_reason}"
            )
        else:
            assert statistic.value is not None
            passed = (
                statistic.value > threshold
                if strict
                else statistic.value >= threshold
            )
            if not passed:
                reasons.append(failure)
    realized_bacc = by_name["equal_center_realized_bacc"].value
    assert realized_bacc is not None
    if realized_bacc <= 0.0:
        reasons.append("NONPOSITIVE_REALIZED_BACC")
    if safe_rate < legacy_safe_rate:
        reasons.append("JOINT_SAFE_RATE_BELOW_LEGACY")
    if absolute_regret > legacy_absolute_regret:
        reasons.append("ABSOLUTE_ORACLE_REGRET_ABOVE_LEGACY")
    if not denominators_valid:
        reasons.append("INVALID_NORMALIZED_ORACLE_DENOMINATOR")
    else:
        assert normalized.value is not None
        assert legacy_normalized.value is not None
        if normalized.value > legacy_normalized.value:
            reasons.append("NORMALIZED_ORACLE_GAP_ABOVE_LEGACY")

    return OuterAdmission(
        outer,
        tuple(row.donor_center for row in rows),
        not reasons,
        tuple(reasons) if reasons else ("PSEUDO_ONLY_ADMISSION_PASS",),
        statistics,
        tuple(row.evidence_hash for row in rows),
    )


__all__ = ("OUTER_ADMISSION_SCHEMA", "OuterAdmission", "build_outer_admission")
