"""Whole-policy held-source replay and selective risk--coverage calibration."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import CasePrediction, SourceActionOutcome
from .hashing import canonical_hash
from .effective_menu import EffectiveMenu
from .model import NestedPolicyFold, _SourceCase, _source_cases


# A finite, predeclared sentinel that can never be reached by a probability in
# [0, 1].  It represents an explicitly disabled fold without deriving any
# threshold from the held source center.
DISABLED_OPPORTUNITY_THRESHOLD = 2.0


@dataclass(frozen=True, slots=True)
class RiskCoverageConfig:
    opportunity_thresholds: tuple[float, ...] = (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
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
            not self.opportunity_thresholds
            or not self.rank_margin_thresholds
            or tuple(sorted(set(self.opportunity_thresholds))) != self.opportunity_thresholds
            or tuple(sorted(set(self.rank_margin_thresholds))) != self.rank_margin_thresholds
            or any(not 0.0 <= value <= 1.0 for value in self.opportunity_thresholds)
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
            raise ProtocolError("Risk--coverage configuration is malformed.")


@dataclass(frozen=True, slots=True)
class PolicyReplay:
    opportunity_threshold: float
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
            raise ProtocolError("Policy replay counts are malformed.")
        object.__setattr__(
            self,
            "replay_hash",
            canonical_hash(
                {
                    "schema_version": "source_active_policy_replay_v7",
                    "opportunity_threshold": self.opportunity_threshold,
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
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class SelectiveCalibration:
    outer_target_id: str
    calibrated: bool
    opportunity_threshold: float
    rank_margin_threshold: float
    selected_replay: PolicyReplay
    nested_replay: PolicyReplay
    heldout_thresholds: tuple[tuple[str, float, float], ...]
    frontier: tuple[PolicyReplay, ...]
    config: RiskCoverageConfig
    calibration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.calibrated != (self.selected_replay.safe and self.nested_replay.safe):
            raise ProtocolError("Selective calibration and replay safety disagree.")
        object.__setattr__(
            self,
            "calibration_hash",
            canonical_hash(
                {
                    "schema_version": "source_active_selective_calibration_v7",
                    "outer_target_id": self.outer_target_id,
                    "calibrated": self.calibrated,
                    "opportunity_threshold": self.opportunity_threshold,
                    "rank_margin_threshold": self.rank_margin_threshold,
                    "selected_replay_hash": self.selected_replay.replay_hash,
                    "nested_replay_hash": self.nested_replay.replay_hash,
                    "heldout_thresholds": self.heldout_thresholds,
                    "frontier_hashes": tuple(row.replay_hash for row in self.frontier),
                    "config": self.config,
                    "fit_surface": "HELD_SOURCE_OOF_ONLY",
                    "target_evaluation_labels_used": False,
                }
            ),
        )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def calibrate_policy_risk_coverage(
    predictions: Sequence[CasePrediction],
    observations: Sequence[SourceActionOutcome],
    *,
    config: RiskCoverageConfig = RiskCoverageConfig(),
    effective_menus: Sequence[EffectiveMenu] | None = None,
    nested_policy_folds: Sequence[NestedPolicyFold] | None = None,
) -> SelectiveCalibration:
    """Calibrate abstention on complete held-source top-1 decisions.

    Endpoint constraints apply to the deployed policy replay, not separately to
    every action.  This avoids the maximum-of-center-maxima failure mode while
    retaining delete-one-center stability and a routed-case harm budget.
    """

    cases = _source_cases(observations, effective_menus)
    by_key = {(row.query_center_id, row.case_id): row for row in predictions}
    expected = {(case.menu.query_center_id, case.menu.case_id) for case in cases}
    if set(by_key) != expected:
        raise ProtocolError("Calibration OOF predictions do not cover the source inventory.")
    outer = cases[0].menu.outer_target_id
    centers = tuple(sorted({case.menu.query_center_id for case in cases}))
    nested_by_heldout = {
        fold.heldout_center_id: fold for fold in (nested_policy_folds or ())
    }
    if set(nested_by_heldout) != set(centers):
        raise ProtocolError(
            "Calibration requires one leakage-free nested policy fold per source center."
        )

    def decisions(
        opportunity_threshold: float,
        rank_margin_threshold: float,
        selected_centers: Sequence[str],
        prediction_inventory: dict[tuple[str, str], CasePrediction] = by_key,
        case_inventory: Sequence[_SourceCase] = cases,
    ) -> dict[tuple[str, str], tuple[float, float, float, bool]]:
        allowed = set(selected_centers)
        output: dict[tuple[str, str], tuple[float, float, float, bool]] = {}
        for case in case_inventory:
            center = case.menu.query_center_id
            if center not in allowed:
                continue
            prediction = prediction_inventory[(center, case.menu.case_id)]
            if (
                prediction.outer_target_id != outer
                or prediction.query_center_id in prediction.training_center_ids
                or prediction.menu_hash != case.menu.menu_hash
            ):
                raise ProtocolError("Calibration received non-OOF or menu-drifted predictions.")
            selected = None
            if (
                prediction.top_action_id is not None
                and prediction.opportunity_probability >= opportunity_threshold
                and prediction.passes_rank_margin(rank_margin_threshold)
            ):
                selected = next(
                    row
                    for row in case.outcomes
                    if row.action.action_id == prediction.top_action_id
                )
            output[(center, case.menu.case_id)] = (
                (0.0, 0.0, 0.0, False)
                if selected is None
                else (
                    selected.bacc_gain,
                    selected.brier_delta,
                    selected.log_delta,
                    True,
                )
            )
        return output

    def replay_from_decisions(
        values: dict[tuple[str, str], tuple[float, float, float, bool]],
        opportunity_threshold: float,
        rank_margin_threshold: float,
    ) -> PolicyReplay:
        endpoint_by_center: dict[str, list[tuple[float, float, float, bool]]] = defaultdict(list)
        routed_gains: list[float] = []
        routed = 0
        for (center, _case), row in values.items():
            endpoint_by_center[center].append(row)
            if row[3]:
                routed += 1
                routed_gains.append(row[0])

        replay_centers = tuple(sorted(endpoint_by_center))
        center_bacc = {
            center: _mean([row[0] for row in endpoint_by_center[center]])
            for center in replay_centers
        }
        center_brier = {
            center: _mean([row[1] for row in endpoint_by_center[center]])
            for center in replay_centers
        }
        center_log = {
            center: _mean([row[2] for row in endpoint_by_center[center]])
            for center in replay_centers
        }
        bacc = _mean(tuple(center_bacc.values()))
        brier = _mean(tuple(center_brier.values()))
        log_delta = _mean(tuple(center_log.values()))
        delete_gains = tuple(
            _mean([center_bacc[center] for center in replay_centers if center != deleted])
            for deleted in replay_centers
        )
        min_delete = min(delete_gains) if len(replay_centers) > 1 else bacc
        case_count = len(values)
        coverage = routed / case_count
        harm_rate = (
            sum(value < 0.0 for value in routed_gains) / routed if routed else 0.0
        )
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
            opportunity_threshold=float(opportunity_threshold),
            rank_margin_threshold=float(rank_margin_threshold),
            routed_cases=routed,
            case_count=case_count,
            coverage=coverage,
            case_equal_bacc_gain=bacc,
            min_delete_center_bacc_gain=min_delete,
            case_equal_brier_delta=brier,
            case_equal_log_delta=log_delta,
            routed_harm_rate=harm_rate,
            safe=safe,
        )

    def replay(
        opportunity_threshold: float,
        rank_margin_threshold: float,
        selected_centers: Sequence[str] = centers,
    ) -> PolicyReplay:
        return replay_from_decisions(
            decisions(opportunity_threshold, rank_margin_threshold, selected_centers),
            opportunity_threshold,
            rank_margin_threshold,
        )

    frontier = tuple(
        replay(opportunity, margin)
        for opportunity in config.opportunity_thresholds
        for margin in config.rank_margin_thresholds
    )
    feasible = tuple(row for row in frontier if row.safe)
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                row.coverage,
                row.case_equal_bacc_gain,
                row.min_delete_center_bacc_gain,
                -row.opportunity_threshold,
                -row.rank_margin_threshold,
            ),
        )
    else:
        # Persist an explicit exact-B replay rather than fabricating a route.
        selected = replay(DISABLED_OPPORTUNITY_THRESHOLD, 0.0)
    # Nested center-held-out replay: each held center is evaluated at thresholds
    # chosen exclusively from the other source centers' OOF outcomes.
    heldout_thresholds: list[tuple[str, float, float]] = []
    nested_decisions: dict[tuple[str, str], tuple[float, float, float, bool]] = {}
    for heldout in centers:
        inner_centers = tuple(center for center in centers if center != heldout)
        fold = nested_by_heldout[heldout]
        if tuple(sorted(fold.training_center_ids)) != inner_centers:
            raise ProtocolError("Nested calibration fold training inventory drifted.")
        inner_predictions = {
            (row.query_center_id, row.case_id): row for row in fold.predictions
        }
        inner_cases = tuple(
            case for case in cases if case.menu.query_center_id != heldout
        )

        def inner_replay(opportunity: float, margin: float) -> PolicyReplay:
            return replay_from_decisions(
                decisions(
                    opportunity,
                    margin,
                    inner_centers,
                    prediction_inventory=inner_predictions,
                    case_inventory=inner_cases,
                ),
                opportunity,
                margin,
            )

        inner_frontier = tuple(
            inner_replay(opportunity, margin)
            for opportunity in config.opportunity_thresholds
            for margin in config.rank_margin_thresholds
        )
        inner_feasible = tuple(row for row in inner_frontier if row.safe)
        if inner_feasible:
            inner_selected = max(
                inner_feasible,
                key=lambda row: (
                    row.coverage,
                    row.case_equal_bacc_gain,
                    row.min_delete_center_bacc_gain,
                    -row.opportunity_threshold,
                    -row.rank_margin_threshold,
                ),
            )
            opportunity = inner_selected.opportunity_threshold
            margin = inner_selected.rank_margin_threshold
        else:
            # This disabled-fold sentinel is predeclared and independent of
            # every held-source prediction, including the current q.
            opportunity = DISABLED_OPPORTUNITY_THRESHOLD
            margin = 0.0
        heldout_thresholds.append((heldout, opportunity, margin))
        nested_decisions.update(decisions(opportunity, margin, (heldout,)))
    nested_replay = replay_from_decisions(nested_decisions, -1.0, -1.0)
    calibrated = bool(selected.safe and nested_replay.safe)
    return SelectiveCalibration(
        outer_target_id=outer,
        calibrated=calibrated,
        opportunity_threshold=selected.opportunity_threshold,
        rank_margin_threshold=selected.rank_margin_threshold,
        selected_replay=selected,
        nested_replay=nested_replay,
        heldout_thresholds=tuple(heldout_thresholds),
        frontier=frontier,
        config=config,
    )


__all__ = (
    "DISABLED_OPPORTUNITY_THRESHOLD",
    "PolicyReplay",
    "RiskCoverageConfig",
    "SelectiveCalibration",
    "calibrate_policy_risk_coverage",
)
