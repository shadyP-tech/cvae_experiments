"""Terminal-only oracle and metric products; never accepted by routing APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    DIRECTION_IDS,
    EXACT_TIE_TOLERANCE,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    PRE_TERMINAL_METHOD_IDS,
    TERMINAL_ORACLE_IDS,
    candidate_sources,
)
from .hashing import canonical_hash


@dataclass(frozen=True, order=True)
class OracleCandidateUtility:
    source: str | None
    numerator: int
    denominator: int

    @property
    def exact(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def to_payload(self) -> dict[str, object]:
        return {
            "source": self.source,
            "numerator": self.exact.numerator,
            "denominator": self.exact.denominator,
            "value": float(self.exact),
        }


@dataclass(frozen=True, order=True)
class DirectionalOracleDecision:
    method_id: str
    target_center: str
    case_id: str
    direction: str
    candidate_utilities: tuple[OracleCandidateUtility, ...]
    selected_source: str | None
    oracle_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        rows = tuple(self.candidate_utilities)
        if (
            self.method_id not in TERMINAL_ORACLE_IDS
            or self.direction not in DIRECTION_IDS
            or tuple(row.source for row in rows) != (None, *candidate_sources(self.target_center))
            or rows[0].exact != 0
        ):
            raise ProtocolError("OGDE terminal directional oracle topology drifted.")
        maximum = max(row.exact for row in rows)
        eligible = tuple(row.source for row in rows if maximum - row.exact <= EXACT_TIE_TOLERANCE)
        expected = min(eligible, key=lambda source: -1 if source is None else int(source))
        if self.selected_source != expected:
            raise ProtocolError("OGDE terminal oracle violates exact OFF-first selection.")
        object.__setattr__(self, "candidate_utilities", rows)
        object.__setattr__(self, "oracle_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.direction

    def utility_for(self, source: str | None) -> Fraction:
        return next(row.exact for row in self.candidate_utilities if row.source == source)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_terminal_directional_oracle_v1",
            "method_id": self.method_id,
            "target_center": self.target_center,
            "case_id": self.case_id,
            "direction": self.direction,
            "candidate_utilities": [row.to_payload() for row in self.candidate_utilities],
            "selected_source": self.selected_source,
            "terminal_labels_used": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "oracle_hash": self.oracle_hash}


@dataclass(frozen=True, order=True)
class IdentificationMetrics:
    method_id: str
    route_direction_count: int
    oracle_off_count: int
    predicted_off_count: int
    off_precision: float
    off_recall: float
    off_balanced_accuracy: float
    exact_action_agreement: float
    active_source_top1: float
    active_source_count: int
    macro_spearman: float
    mean_oracle_gap: float
    mean_normalized_oracle_gap: float
    metrics_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        values = (
            self.off_precision, self.off_recall, self.off_balanced_accuracy,
            self.exact_action_agreement, self.active_source_top1, self.macro_spearman,
            self.mean_oracle_gap, self.mean_normalized_oracle_gap,
        )
        count = int(self.route_direction_count)
        bounded = values[:5]
        if (
            self.method_id not in {"I_OPPORTUNITY_GATED", "I_FEATURE_BLOCK_PERMUTED"}
            or count != EXPECTED_TOTAL_CASE_COUNT * len(DIRECTION_IDS)
            or not 0 <= int(self.oracle_off_count) <= count
            or not 0 <= int(self.predicted_off_count) <= count
            or not 0 <= int(self.active_source_count) <= count
            or not all(math.isfinite(float(value)) for value in values)
            or any(not 0.0 <= float(value) <= 1.0 for value in bounded)
            or not -1.0 <= float(self.macro_spearman) <= 1.0
            or float(self.mean_oracle_gap) < 0.0
            or not 0.0 <= float(self.mean_normalized_oracle_gap) <= 1.0
        ):
            raise ProtocolError("OGDE identification metrics drifted.")
        object.__setattr__(self, "metrics_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_identification_metrics_v1",
            **{name: value for name, value in self.__dict__.items() if name != "metrics_hash"},
            "terminal_labels_used": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "metrics_hash": self.metrics_hash}


@dataclass(frozen=True, order=True)
class ProbabilityMetrics:
    method_id: str
    sample_count: int
    brier: float
    log_loss: float
    calibration_intercept: float
    calibration_slope: float
    equal_center_bacc: float
    center_bacc: tuple[tuple[str, float], ...]
    metrics_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        values = (
            self.brier, self.log_loss, self.calibration_intercept,
            self.calibration_slope, self.equal_center_bacc,
        )
        center_rows = tuple((str(center), float(value)) for center, value in self.center_bacc)
        if (
            self.method_id not in PRE_TERMINAL_METHOD_IDS
            or int(self.sample_count) != EXPECTED_TEST_ROW_COUNT
            or not all(math.isfinite(float(value)) for value in values)
            or not 0.0 <= float(self.brier) <= 1.0
            or float(self.log_loss) < 0.0
            or not 0.0 <= float(self.equal_center_bacc) <= 1.0
            or tuple(center for center, _ in center_rows) != CENTERS
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for _, value in center_rows)
        ):
            raise ProtocolError("OGDE probability metrics drifted.")
        object.__setattr__(self, "center_bacc", center_rows)
        object.__setattr__(self, "metrics_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_probability_metrics_v1",
            "method_id": self.method_id,
            "sample_count": self.sample_count,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "calibration_intercept": self.calibration_intercept,
            "calibration_slope": self.calibration_slope,
            "equal_center_bacc": self.equal_center_bacc,
            "center_bacc": [[center, value] for center, value in self.center_bacc],
            "terminal_labels_used": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "metrics_hash": self.metrics_hash}


__all__ = (
    "DirectionalOracleDecision",
    "IdentificationMetrics",
    "OracleCandidateUtility",
    "ProbabilityMetrics",
)
