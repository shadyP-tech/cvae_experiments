"""Complete all-case and nested held-center policy replay for HARP v8."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import CasePrediction, SourceActionOutcome
from .effective_menu import EffectiveMenu
from .hashing import canonical_hash
from .model import NestedPolicyFold, _SourceCase, _filter_surface, _source_cases


DISABLED_CERTIFICATE_CONFIDENCE_THRESHOLD = 2.0


@dataclass(frozen=True, slots=True)
class RiskCoverageConfig:
    certificate_confidence_thresholds: tuple[float, ...] = (
        0.25,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    )
    rank_margin_thresholds: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01, 0.02)
    min_case_equal_bacc_gain: float = 0.0
    min_delete_center_bacc_gain: float = -0.005
    max_routed_harm_rate: float = 0.25
    max_case_equal_brier_delta: float = 0.002
    max_case_equal_log_delta: float = 0.005
    min_coverage: float = 0.02
    min_routed_cases: int = 3

    def __post_init__(self) -> None:
        if (
            not self.certificate_confidence_thresholds
            or not self.rank_margin_thresholds
            or tuple(sorted(set(self.certificate_confidence_thresholds)))
            != self.certificate_confidence_thresholds
            or tuple(sorted(set(self.rank_margin_thresholds))) != self.rank_margin_thresholds
            or any(
                not 0.0 <= value <= 1.0
                for value in self.certificate_confidence_thresholds
            )
            or any(not math.isfinite(value) or value < 0.0 for value in self.rank_margin_thresholds)
            or not 0.0 <= self.max_routed_harm_rate <= 1.0
            or not 0.0 <= self.min_coverage <= 1.0
            or int(self.min_routed_cases) < 1
            or any(
                not math.isfinite(value)
                for value in (
                    self.min_case_equal_bacc_gain,
                    self.min_delete_center_bacc_gain,
                    self.max_case_equal_brier_delta,
                    self.max_case_equal_log_delta,
                )
            )
        ):
            raise ProtocolError("HARP v8 risk-coverage configuration is malformed.")


@dataclass(frozen=True, slots=True)
class PolicyReplay:
    certificate_confidence_threshold: float
    rank_margin_threshold: float
    routed_cases: int
    case_count: int
    coverage: float
    case_equal_bacc_gain: float
    min_delete_center_bacc_gain: float
    case_equal_brier_delta: float
    case_equal_log_delta: float
    routed_harm_rate: float
    safe: bool
    replay_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.routed_cases <= self.case_count or self.case_count < 1:
            raise ProtocolError("HARP v8 policy replay counts are malformed.")
        object.__setattr__(
            self,
            "replay_hash",
            canonical_hash(
                {
                    "schema_version": "baseline_inclusive_policy_replay_v8",
                    "certificate_confidence_threshold": (
                        self.certificate_confidence_threshold
                    ),
                    "rank_margin_threshold": self.rank_margin_threshold,
                    "routed_cases": self.routed_cases,
                    "case_count": self.case_count,
                    "coverage": self.coverage,
                    "case_equal_bacc_gain": self.case_equal_bacc_gain,
                    "min_delete_center_bacc_gain": self.min_delete_center_bacc_gain,
                    "case_equal_brier_delta": self.case_equal_brier_delta,
                    "case_equal_log_delta": self.case_equal_log_delta,
                    "routed_harm_rate": self.routed_harm_rate,
                    "safe": self.safe,
                    "safe_set_before_top1": True,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class SelectiveCalibration:
    outer_target_id: str
    calibrated: bool
    certificate_confidence_threshold: float
    rank_margin_threshold: float
    selected_replay: PolicyReplay
    nested_replay: PolicyReplay
    heldout_thresholds: tuple[tuple[str, float, float], ...]
    frontier: tuple[PolicyReplay, ...]
    config: RiskCoverageConfig
    calibration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.calibrated != (self.selected_replay.safe and self.nested_replay.safe):
            raise ProtocolError("HARP v8 policy calibration/replay safety disagree.")
        object.__setattr__(
            self,
            "calibration_hash",
            canonical_hash(
                {
                    "schema_version": "baseline_inclusive_selective_calibration_v8",
                    "outer_target_id": self.outer_target_id,
                    "calibrated": self.calibrated,
                    "certificate_confidence_threshold": (
                        self.certificate_confidence_threshold
                    ),
                    "certificate_confidence_threshold_semantics": (
                        "MIN_ONE_MINUS_HARM_UCB"
                    ),
                    "rank_margin_threshold": self.rank_margin_threshold,
                    "selected_replay_hash": self.selected_replay.replay_hash,
                    "nested_replay_hash": self.nested_replay.replay_hash,
                    "heldout_thresholds": self.heldout_thresholds,
                    "frontier_hashes": tuple(row.replay_hash for row in self.frontier),
                    "config": self.config,
                    "fit_surface": "STRICT_HELD_SOURCE_POLICY_REPLAY_ONLY",
                    "target_evaluation_labels_used": False,
                }
            ),
        )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _decision_rows(
    predictions: Sequence[CasePrediction],
    cases: Sequence[_SourceCase],
    *,
    outer_target_id: str,
    confidence_threshold: float,
    rank_margin_threshold: float,
) -> dict[tuple[str, str], tuple[float, float, float, bool]]:
    by_key = {(row.query_center_id, row.case_id): row for row in predictions}
    expected = {(case.menu.query_center_id, case.menu.case_id) for case in cases}
    if set(by_key) != expected:
        raise ProtocolError("HARP v8 policy predictions do not cover their case inventory.")
    output: dict[tuple[str, str], tuple[float, float, float, bool]] = {}
    for case in cases:
        key = (case.menu.query_center_id, case.menu.case_id)
        prediction = by_key[key]
        if (
            prediction.outer_target_id != outer_target_id
            or prediction.query_center_id in prediction.training_center_ids
            or prediction.query_center_id in prediction.training_candidate_ids
            or prediction.menu_hash != case.menu.menu_hash
        ):
            raise ProtocolError("HARP v8 policy replay received leaked/menu-drifted rows.")
        selected = None
        selected_id = prediction.top_action_id
        if selected_id is not None and prediction.passes_rank_margin(rank_margin_threshold):
            certificate = next(row for row in prediction.action_certificates if row.action_id == selected_id)
            if 1.0 - certificate.harm_probability_ucb >= confidence_threshold:
                selected = next(row for row in case.outcomes if row.action.action_id == selected_id)
        output[key] = (
            (0.0, 0.0, 0.0, False)
            if selected is None
            else (selected.bacc_gain, selected.brier_delta, selected.log_delta, True)
        )
    return output


def _replay_from_decisions(
    values: dict[tuple[str, str], tuple[float, float, float, bool]],
    *,
    confidence_threshold: float,
    rank_margin_threshold: float,
    config: RiskCoverageConfig,
) -> PolicyReplay:
    by_center: dict[str, list[tuple[float, float, float, bool]]] = defaultdict(list)
    routed_gains: list[float] = []
    for (center, _case), row in values.items():
        by_center[center].append(row)
        if row[3]:
            routed_gains.append(row[0])
    centers = tuple(sorted(by_center))
    center_bacc = {center: _mean([row[0] for row in by_center[center]]) for center in centers}
    center_brier = {center: _mean([row[1] for row in by_center[center]]) for center in centers}
    center_log = {center: _mean([row[2] for row in by_center[center]]) for center in centers}
    bacc = _mean(tuple(center_bacc.values()))
    brier = _mean(tuple(center_brier.values()))
    log_delta = _mean(tuple(center_log.values()))
    delete = tuple(
        _mean([center_bacc[center] for center in centers if center != removed])
        for removed in centers
    )
    min_delete = min(delete) if len(centers) > 1 else bacc
    routed = len(routed_gains)
    case_count = len(values)
    coverage = routed / case_count
    harm_rate = sum(value < 0.0 for value in routed_gains) / routed if routed else 0.0
    safe = bool(
        routed >= config.min_routed_cases
        and coverage >= config.min_coverage
        and bacc > config.min_case_equal_bacc_gain
        and min_delete >= config.min_delete_center_bacc_gain
        and harm_rate <= config.max_routed_harm_rate
        and brier <= config.max_case_equal_brier_delta
        and log_delta <= config.max_case_equal_log_delta
    )
    return PolicyReplay(
        confidence_threshold,
        rank_margin_threshold,
        routed,
        case_count,
        coverage,
        bacc,
        min_delete,
        brier,
        log_delta,
        harm_rate,
        safe,
    )


def _frontier(
    predictions: Sequence[CasePrediction],
    cases: Sequence[_SourceCase],
    *,
    outer_target_id: str,
    config: RiskCoverageConfig,
) -> tuple[PolicyReplay, ...]:
    return tuple(
        _replay_from_decisions(
            _decision_rows(
                predictions,
                cases,
                outer_target_id=outer_target_id,
                confidence_threshold=confidence,
                rank_margin_threshold=margin,
            ),
            confidence_threshold=confidence,
            rank_margin_threshold=margin,
            config=config,
        )
        for confidence in config.certificate_confidence_thresholds
        for margin in config.rank_margin_thresholds
    )


def _select(frontier: Sequence[PolicyReplay]) -> PolicyReplay | None:
    feasible = tuple(row for row in frontier if row.safe)
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda row: (
            row.coverage,
            row.case_equal_bacc_gain,
            row.min_delete_center_bacc_gain,
            -row.certificate_confidence_threshold,
            -row.rank_margin_threshold,
        ),
    )


def calibrate_policy_risk_coverage(
    predictions: Sequence[CasePrediction],
    observations: Sequence[SourceActionOutcome],
    *,
    config: RiskCoverageConfig = RiskCoverageConfig(),
    effective_menus: Sequence[EffectiveMenu] | None = None,
    nested_policy_folds: Sequence[NestedPolicyFold] | None = None,
) -> SelectiveCalibration:
    """Calibrate only after per-action safe sets exist, then replay all cases."""

    rows = tuple(observations)
    menus = None if effective_menus is None else tuple(effective_menus)
    cases = _source_cases(rows, menus, min_centers=2)
    outer = cases[0].menu.outer_target_id
    centers = tuple(sorted({case.menu.query_center_id for case in cases}))
    folds = {row.heldout_center_id: row for row in (nested_policy_folds or ())}
    if set(folds) != set(centers):
        raise ProtocolError("HARP v8 calibration requires every nested held-source fold.")
    frontier = _frontier(predictions, cases, outer_target_id=outer, config=config)
    selected = _select(frontier)
    if selected is None:
        selected = _replay_from_decisions(
            _decision_rows(
                predictions,
                cases,
                outer_target_id=outer,
                confidence_threshold=DISABLED_CERTIFICATE_CONFIDENCE_THRESHOLD,
                rank_margin_threshold=0.0,
            ),
            confidence_threshold=DISABLED_CERTIFICATE_CONFIDENCE_THRESHOLD,
            rank_margin_threshold=0.0,
            config=config,
        )

    heldout_thresholds: list[tuple[str, float, float]] = []
    nested_values: dict[tuple[str, str], tuple[float, float, float, bool]] = {}
    for heldout in centers:
        fold = folds[heldout]
        excluded = frozenset((outer, heldout))
        training_rows, training_menus = _filter_surface(rows, menus, excluded_center_ids=excluded)
        training_cases = _source_cases(training_rows, training_menus, min_centers=2)
        inner_frontier = _frontier(
            fold.predictions,
            training_cases,
            outer_target_id=outer,
            config=config,
        )
        inner_selected = _select(inner_frontier)
        if inner_selected is None:
            confidence, margin = DISABLED_CERTIFICATE_CONFIDENCE_THRESHOLD, 0.0
        else:
            confidence = inner_selected.certificate_confidence_threshold
            margin = inner_selected.rank_margin_threshold
        heldout_thresholds.append((heldout, confidence, margin))
        heldout_cases = tuple(case for case in cases if case.menu.query_center_id == heldout)
        nested_values.update(
            _decision_rows(
                fold.heldout_predictions,
                heldout_cases,
                outer_target_id=outer,
                confidence_threshold=confidence,
                rank_margin_threshold=margin,
            )
        )
    nested_replay = _replay_from_decisions(
        nested_values,
        confidence_threshold=-1.0,
        rank_margin_threshold=-1.0,
        config=config,
    )
    calibrated = bool(selected.safe and nested_replay.safe)
    return SelectiveCalibration(
        outer_target_id=outer,
        calibrated=calibrated,
        certificate_confidence_threshold=(
            selected.certificate_confidence_threshold
        ),
        rank_margin_threshold=selected.rank_margin_threshold,
        selected_replay=selected,
        nested_replay=nested_replay,
        heldout_thresholds=tuple(heldout_thresholds),
        frontier=frontier,
        config=config,
    )


__all__ = (
    "DISABLED_CERTIFICATE_CONFIDENCE_THRESHOLD",
    "PolicyReplay",
    "RiskCoverageConfig",
    "SelectiveCalibration",
    "calibrate_policy_risk_coverage",
)
