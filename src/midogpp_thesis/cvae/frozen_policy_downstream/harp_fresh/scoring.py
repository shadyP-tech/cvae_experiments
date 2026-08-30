"""Center- and case-aware descriptive scoring after the fresh HARP seal."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...runtime.harp_probability_menu.hashing import canonical_sha256
from .label_access import HarpFreshEvaluationCapability
from .metric_primitives import (
    binary_log_loss,
    case_equal_balanced_accuracy,
    case_equal_mean,
)
from .oracle_diagnostics import (
    HarpFreshOracleDiagnosticResult,
    build_harp_fresh_oracle_diagnostics,
)
from .sealing import HarpFreshPrelabelSeal


TWO_SIDED_95_T_DF8 = 2.306004135204166
ONE_SIDED_95_T_DF8 = 1.8595480375228424
_TIE_TOLERANCE = 1.0e-15


@dataclass(frozen=True, kw_only=True)
class HarpFreshCaseMetrics:
    center: str
    case_id: str
    row_count: int
    truth_classes: tuple[int, ...]
    baseline_accuracy: float
    routed_accuracy: float
    baseline_brier: float
    routed_brier: float
    baseline_log_loss: float
    routed_log_loss: float
    route_rate: float
    descriptive_only: bool = True


@dataclass(frozen=True, kw_only=True)
class HarpFreshCenterMetrics:
    center: str
    row_count: int
    case_count: int
    baseline_balanced_accuracy: float
    routed_balanced_accuracy: float
    balanced_accuracy_delta: float
    baseline_brier: float
    routed_brier: float
    brier_delta: float
    baseline_log_loss: float
    routed_log_loss: float
    log_loss_delta: float
    route_rate: float
    equal_case_weighting: bool = True
    descriptive_only: bool = True


@dataclass(frozen=True, kw_only=True)
class HarpFreshCenterInference:
    endpoint: str
    positive_delta_favors_routed: bool
    center_deltas: tuple[float, ...]
    mean_delta: float
    sample_standard_deviation: float
    standard_error: float
    two_sided_95_interval_low: float
    two_sided_95_interval_high: float
    one_sided_95_lower_bound: float
    wins: int
    ties: int
    losses: int
    inference_unit: str = "target_center"
    inference_unit_count: int = 9
    degrees_of_freedom: int = 8
    seed_cells_are_inference_units: bool = False

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.center_deltas)
        numeric = (
            *values,
            self.mean_delta,
            self.sample_standard_deviation,
            self.standard_error,
            self.two_sided_95_interval_low,
            self.two_sided_95_interval_high,
            self.one_sided_95_lower_bound,
        )
        if (
            not self.endpoint
            or self.positive_delta_favors_routed is not True
            or len(values) != len(CENTERS)
            or any(not math.isfinite(float(value)) for value in numeric)
            or self.sample_standard_deviation < 0.0
            or self.standard_error < 0.0
            or self.wins + self.ties + self.losses != len(CENTERS)
            or self.inference_unit != "target_center"
            or self.inference_unit_count != len(CENTERS)
            or self.degrees_of_freedom != len(CENTERS) - 1
            or self.seed_cells_are_inference_units is not False
        ):
            raise ProtocolError("Fresh HARP target-center inference drifted.")
        object.__setattr__(self, "center_deltas", values)


@dataclass(frozen=True, kw_only=True)
class HarpFreshDescriptiveResult:
    prediction_seal_hash: str
    case_metrics: tuple[HarpFreshCaseMetrics, ...]
    center_metrics: tuple[HarpFreshCenterMetrics, ...]
    equal_center_metrics: HarpFreshCenterMetrics
    center_inference: tuple[HarpFreshCenterInference, ...]
    oracle_diagnostics: HarpFreshOracleDiagnosticResult
    labels_used_for_scoring_only: bool = True
    center_level_inference_allowed: bool = True
    row_or_case_level_inference_allowed: bool = False
    fresh_claim_requires_completed_eligible_reservation_and_bundle: bool = True
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            tuple(row.center for row in self.center_metrics) != CENTERS
            or self.equal_center_metrics.center != "ALL_CENTERS_EQUAL_WEIGHT"
            or tuple(row.endpoint for row in self.center_inference)
            != (
                "balanced_accuracy_improvement",
                "brier_improvement",
                "log_loss_improvement",
            )
            or self.labels_used_for_scoring_only is not True
            or self.center_level_inference_allowed is not True
            or self.row_or_case_level_inference_allowed is not False
            or self.fresh_claim_requires_completed_eligible_reservation_and_bundle
            is not True
            or not isinstance(
                self.oracle_diagnostics, HarpFreshOracleDiagnosticResult
            )
            or self.oracle_diagnostics.prelabel_seal_hash
            != self.prediction_seal_hash
        ):
            raise ProtocolError("Fresh HARP descriptive result coverage drifted.")
        payload = {
            "schema_version": "midogpp_harp_fresh_descriptive_result_v1",
            "prediction_seal_hash": self.prediction_seal_hash,
            "case_metrics": [row.__dict__ for row in self.case_metrics],
            "center_metrics": [row.__dict__ for row in self.center_metrics],
            "equal_center_metrics": self.equal_center_metrics.__dict__,
            "center_inference": [row.__dict__ for row in self.center_inference],
            "oracle_diagnostics_result_hash": self.oracle_diagnostics.result_hash,
            "labels_used_for_scoring_only": True,
            "equal_case_weighting": True,
            "equal_center_weighting": True,
            "center_inference_unit": "target_center",
            "center_inference_unit_count": len(CENTERS),
            "seed_cells_are_inference_units": False,
            "center_level_inference_allowed": True,
            "row_or_case_level_inference_allowed": False,
            "fresh_claim_requires_completed_eligible_reservation_and_bundle": True,
        }
        object.__setattr__(self, "result_hash", canonical_sha256(payload))


def _center_metrics(
    *,
    center: str,
    truth: np.ndarray,
    baseline: np.ndarray,
    routed: np.ndarray,
    case_ids: np.ndarray,
    route_flags: np.ndarray,
) -> HarpFreshCenterMetrics:
    baseline_bacc = case_equal_balanced_accuracy(truth, baseline, case_ids)
    routed_bacc = case_equal_balanced_accuracy(truth, routed, case_ids)
    baseline_brier = case_equal_mean((baseline - truth) ** 2, case_ids)
    routed_brier = case_equal_mean((routed - truth) ** 2, case_ids)
    baseline_loss = case_equal_mean(binary_log_loss(truth, baseline), case_ids)
    routed_loss = case_equal_mean(binary_log_loss(truth, routed), case_ids)
    route_rate = case_equal_mean(route_flags.astype(np.float64), case_ids)
    return HarpFreshCenterMetrics(
        center=center,
        row_count=len(truth),
        case_count=len(set(case_ids.tolist())),
        baseline_balanced_accuracy=baseline_bacc,
        routed_balanced_accuracy=routed_bacc,
        balanced_accuracy_delta=routed_bacc - baseline_bacc,
        baseline_brier=baseline_brier,
        routed_brier=routed_brier,
        brier_delta=routed_brier - baseline_brier,
        baseline_log_loss=baseline_loss,
        routed_log_loss=routed_loss,
        log_loss_delta=routed_loss - baseline_loss,
        route_rate=route_rate,
    )


def _center_inference(
    endpoint: str, center_deltas: Sequence[float]
) -> HarpFreshCenterInference:
    values = np.asarray(tuple(center_deltas), dtype=np.float64)
    if values.shape != (len(CENTERS),) or not np.isfinite(values).all():
        raise ProtocolError("Fresh HARP inference requires exactly nine center deltas.")
    mean = float(np.mean(values, dtype=np.float64))
    sample_sd = float(np.std(values, ddof=1, dtype=np.float64))
    standard_error = sample_sd / math.sqrt(len(CENTERS))
    two_sided_radius = TWO_SIDED_95_T_DF8 * standard_error
    one_sided_radius = ONE_SIDED_95_T_DF8 * standard_error
    wins = int(np.sum(values > _TIE_TOLERANCE))
    losses = int(np.sum(values < -_TIE_TOLERANCE))
    ties = len(CENTERS) - wins - losses
    return HarpFreshCenterInference(
        endpoint=endpoint,
        positive_delta_favors_routed=True,
        center_deltas=tuple(float(value) for value in values),
        mean_delta=mean,
        sample_standard_deviation=sample_sd,
        standard_error=standard_error,
        two_sided_95_interval_low=mean - two_sided_radius,
        two_sided_95_interval_high=mean + two_sided_radius,
        one_sided_95_lower_bound=mean - one_sided_radius,
        wins=wins,
        ties=ties,
        losses=losses,
    )


def score_harp_fresh_routes(
    seal: HarpFreshPrelabelSeal,
    capability: HarpFreshEvaluationCapability,
) -> HarpFreshDescriptiveResult:
    if not isinstance(seal, HarpFreshPrelabelSeal) or not isinstance(
        capability, HarpFreshEvaluationCapability
    ):
        raise ProtocolError("Fresh HARP scoring requires a seal and one-shot capability.")
    labels = capability.consume(seal)
    case_rows: list[HarpFreshCaseMetrics] = []
    center_rows: list[HarpFreshCenterMetrics] = []
    for center, vector in zip(CENTERS, seal.routed_vectors, strict=True):
        case_ids = np.asarray(vector.case_ids, dtype=object)
        truth = np.asarray(
            [labels[(center, case_id, row_id)] for row_id, case_id in zip(vector.row_ids, vector.case_ids, strict=True)],
            dtype=np.int64,
        )
        baseline = np.asarray(vector.baseline_probabilities, dtype=np.float64)
        routed = np.asarray(vector.routed_probabilities, dtype=np.float64)
        route_flags = np.asarray([row.eligible for row in vector.decisions], dtype=bool)
        center_rows.append(
            _center_metrics(
                center=center,
                truth=truth,
                baseline=baseline,
                routed=routed,
                case_ids=case_ids,
                route_flags=route_flags,
            )
        )
        for case in sorted(set(vector.case_ids)):
            selected = case_ids == case
            case_truth = truth[selected]
            case_baseline = baseline[selected]
            case_routed = routed[selected]
            case_route = route_flags[selected]
            case_rows.append(
                HarpFreshCaseMetrics(
                    center=center,
                    case_id=case,
                    row_count=int(selected.sum()),
                    truth_classes=tuple(sorted(set(int(value) for value in case_truth))),
                    baseline_accuracy=float(
                        np.mean((case_baseline >= 0.5) == case_truth)
                    ),
                    routed_accuracy=float(np.mean((case_routed >= 0.5) == case_truth)),
                    baseline_brier=float(np.mean((case_baseline - case_truth) ** 2)),
                    routed_brier=float(np.mean((case_routed - case_truth) ** 2)),
                    baseline_log_loss=float(
                        np.mean(binary_log_loss(case_truth, case_baseline))
                    ),
                    routed_log_loss=float(
                        np.mean(binary_log_loss(case_truth, case_routed))
                    ),
                    route_rate=float(np.mean(case_route)),
                )
            )

    fields = (
        "baseline_balanced_accuracy",
        "routed_balanced_accuracy",
        "balanced_accuracy_delta",
        "baseline_brier",
        "routed_brier",
        "brier_delta",
        "baseline_log_loss",
        "routed_log_loss",
        "log_loss_delta",
        "route_rate",
    )
    averages = {
        name: float(np.mean([getattr(row, name) for row in center_rows], dtype=np.float64))
        for name in fields
    }
    if any(not math.isfinite(value) for value in averages.values()):
        raise ProtocolError("Fresh HARP descriptive metrics are nonfinite.")
    global_row = HarpFreshCenterMetrics(
        center="ALL_CENTERS_EQUAL_WEIGHT",
        row_count=sum(row.row_count for row in center_rows),
        case_count=sum(row.case_count for row in center_rows),
        **averages,
    )
    inference = (
        _center_inference(
            "balanced_accuracy_improvement",
            [row.balanced_accuracy_delta for row in center_rows],
        ),
        _center_inference(
            "brier_improvement",
            [-row.brier_delta for row in center_rows],
        ),
        _center_inference(
            "log_loss_improvement",
            [-row.log_loss_delta for row in center_rows],
        ),
    )
    oracle = build_harp_fresh_oracle_diagnostics(seal, labels)
    return HarpFreshDescriptiveResult(
        prediction_seal_hash=seal.seal_hash,
        case_metrics=tuple(case_rows),
        center_metrics=tuple(center_rows),
        equal_center_metrics=global_row,
        center_inference=inference,
        oracle_diagnostics=oracle,
    )


__all__ = (
    "HarpFreshCaseMetrics",
    "HarpFreshCenterMetrics",
    "HarpFreshCenterInference",
    "HarpFreshDescriptiveResult",
    "score_harp_fresh_routes",
)
