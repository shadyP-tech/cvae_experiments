"""Nested source-center whole-policy risk/coverage calibration for HARP v12."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .hashing import canonical_hash
from .model import NestedPolicyFold
from .outcome_inventory import CaseKey, CaseOutcomeInventory, SourceOutcomeUniverse
from .outcome_replay import PolicyDecisionOutcome, replay_policy_decisions


DISABLED_ACCEPTANCE_THRESHOLD = 2.0


@dataclass(frozen=True, slots=True)
class PolicyRiskConfig:
    acceptance_thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    fixed_rank_margin_threshold: float = 0.0
    min_case_equal_bacc_gain: float = 0.0
    min_delete_center_bacc_gain: float = -0.005
    max_routed_harm_rate: float = 0.25
    max_case_equal_brier_delta: float = 0.002
    max_case_equal_log_delta: float = 0.005
    min_coverage: float = 0.02
    min_routed_cases: int = 3

    def __post_init__(self) -> None:
        if (
            not self.acceptance_thresholds
            or tuple(sorted(set(self.acceptance_thresholds))) != self.acceptance_thresholds
            or any(not 0.0 <= value <= 1.0 for value in self.acceptance_thresholds)
            or not math.isfinite(self.fixed_rank_margin_threshold)
            or self.fixed_rank_margin_threshold < 0.0
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
            raise ProtocolError("HARP v12 policy-risk configuration is malformed.")


@dataclass(frozen=True, slots=True)
class PolicyReplay:
    acceptance_threshold: float
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
            raise ProtocolError("HARP v12 policy replay counts are malformed.")
        if any(
            not math.isfinite(value)
            for value in (
                self.acceptance_threshold,
                self.rank_margin_threshold,
                self.coverage,
                self.case_equal_bacc_gain,
                self.min_delete_center_bacc_gain,
                self.case_equal_brier_delta,
                self.case_equal_log_delta,
                self.routed_harm_rate,
            )
        ):
            raise ProtocolError("HARP v12 policy replay contains non-finite values.")
        object.__setattr__(
            self,
            "replay_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_policy_replay_v12",
                    "acceptance_threshold": self.acceptance_threshold,
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
                    "rank_all_then_accept_selected": True,
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "acceptance_threshold": self.acceptance_threshold,
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
            "replay_hash": self.replay_hash,
        }


@dataclass(frozen=True, slots=True)
class PolicyCalibration:
    outer_target_id: str
    calibrated: bool
    acceptance_threshold: float
    rank_margin_threshold: float
    selected_replay: PolicyReplay
    nested_replay: PolicyReplay
    heldout_thresholds: tuple[tuple[str, float, float], ...]
    frontier: tuple[PolicyReplay, ...]
    config: PolicyRiskConfig
    calibration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.calibrated != (self.selected_replay.safe and self.nested_replay.safe):
            raise ProtocolError("HARP v12 calibration and nested policy replay disagree.")
        if not self.frontier or len({row[0] for row in self.heldout_thresholds}) != len(
            self.heldout_thresholds
        ):
            raise ProtocolError("HARP v12 policy-calibration inventory is malformed.")
        object.__setattr__(
            self,
            "calibration_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_selected_action_policy_v12",
                    "outer_target_id": self.outer_target_id,
                    "calibrated": self.calibrated,
                    "acceptance_threshold": self.acceptance_threshold,
                    "rank_margin_threshold": self.rank_margin_threshold,
                    "selected_replay_hash": self.selected_replay.replay_hash,
                    "nested_replay_hash": self.nested_replay.replay_hash,
                    "heldout_thresholds": self.heldout_thresholds,
                    "frontier_hashes": tuple(row.replay_hash for row in self.frontier),
                    "config": self.config,
                    "policy_level_source_lodo_risk_control": True,
                    "formal_target_conformal_guarantee": False,
                    "target_evaluation_labels_used": False,
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "outer_target_id": self.outer_target_id,
            "calibrated": self.calibrated,
            "acceptance_threshold": self.acceptance_threshold,
            "rank_margin_threshold": self.rank_margin_threshold,
            "selected_replay": self.selected_replay.public_payload(),
            "nested_replay": self.nested_replay.public_payload(),
            "heldout_thresholds": [
                {
                    "heldout_center_id": center,
                    "acceptance_threshold": threshold,
                    "rank_margin_threshold": margin,
                }
                for center, threshold, margin in self.heldout_thresholds
            ],
            "frontier": [row.public_payload() for row in self.frontier],
            "calibration_hash": self.calibration_hash,
        }


def _decision_rows(
    inventory: CaseOutcomeInventory,
    *,
    acceptance_threshold: float,
    rank_margin_threshold: float,
) -> dict[CaseKey, PolicyDecisionOutcome]:
    return replay_policy_decisions(
        inventory,
        acceptance_threshold=acceptance_threshold,
        rank_margin_threshold=rank_margin_threshold,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _replay_from_decisions(
    values: Mapping[CaseKey, PolicyDecisionOutcome],
    *,
    acceptance_threshold: float,
    rank_margin_threshold: float,
    config: PolicyRiskConfig,
) -> PolicyReplay:
    by_center: dict[str, list[tuple[float, float, float, bool]]] = defaultdict(list)
    routed_gains: list[float] = []
    for (_outer, center, _case), decision in values.items():
        row = decision.numeric()
        by_center[center].append(row)
        if decision.routed:
            routed_gains.append(decision.bacc_gain)
    centers = tuple(sorted(by_center))
    if not centers:
        raise ProtocolError("HARP v12 policy replay has no centers.")
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
    count = len(values)
    coverage = routed / count
    harm = sum(value < 0.0 for value in routed_gains) / routed if routed else 0.0
    safe = bool(
        routed >= config.min_routed_cases
        and coverage >= config.min_coverage
        and bacc > config.min_case_equal_bacc_gain
        and min_delete >= config.min_delete_center_bacc_gain
        and harm <= config.max_routed_harm_rate
        and brier <= config.max_case_equal_brier_delta
        and log_delta <= config.max_case_equal_log_delta
    )
    return PolicyReplay(
        acceptance_threshold=acceptance_threshold,
        rank_margin_threshold=rank_margin_threshold,
        routed_cases=routed,
        case_count=count,
        coverage=coverage,
        case_equal_bacc_gain=bacc,
        min_delete_center_bacc_gain=min_delete,
        case_equal_brier_delta=brier,
        case_equal_log_delta=log_delta,
        routed_harm_rate=harm,
        safe=safe,
    )


def _frontier(
    inventory: CaseOutcomeInventory,
    config: PolicyRiskConfig,
) -> tuple[PolicyReplay, ...]:
    return tuple(
        _replay_from_decisions(
            _decision_rows(
                inventory,
                acceptance_threshold=threshold,
                rank_margin_threshold=margin,
            ),
            acceptance_threshold=threshold,
            rank_margin_threshold=margin,
            config=config,
        )
        for threshold in config.acceptance_thresholds
        for margin in (config.fixed_rank_margin_threshold,)
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
            row.acceptance_threshold,
            row.rank_margin_threshold,
        ),
    )


def _disabled(
    inventory: CaseOutcomeInventory,
    config: PolicyRiskConfig,
) -> PolicyReplay:
    return _replay_from_decisions(
        _decision_rows(
            inventory,
            acceptance_threshold=DISABLED_ACCEPTANCE_THRESHOLD,
            rank_margin_threshold=0.0,
        ),
        acceptance_threshold=DISABLED_ACCEPTANCE_THRESHOLD,
        rank_margin_threshold=0.0,
        config=config,
    )


def calibrate_selected_policy(
    inventory: CaseOutcomeInventory,
    *,
    config: PolicyRiskConfig = PolicyRiskConfig(),
    outcome_universe: SourceOutcomeUniverse,
    nested_policy_folds: Sequence[NestedPolicyFold] | None = None,
) -> PolicyCalibration:
    """Select one policy threshold and verify it by nested held-center replay."""

    if not isinstance(inventory, CaseOutcomeInventory) or not isinstance(
        outcome_universe, SourceOutcomeUniverse
    ):
        raise ProtocolError("HARP v12 calibration requires typed outcome inventories.")
    outers = {row.key[0] for row in inventory.contexts}
    if len(outers) != 1:
        raise ProtocolError("HARP v12 calibration crossed outer targets.")
    outer = next(iter(outers))
    centers = tuple(sorted({row.key[1] for row in inventory.contexts}))
    folds = {row.heldout_center_id: row for row in (nested_policy_folds or ())}
    if set(folds) != set(centers):
        raise ProtocolError("HARP v12 calibration requires every nested held-source fold.")
    frontier = _frontier(inventory, config)
    selected = _select(frontier)
    selected_replay = selected or _disabled(inventory, config)
    nested_decisions: dict[CaseKey, PolicyDecisionOutcome] = {}
    heldout_thresholds: list[tuple[str, float, float]] = []
    for center in centers:
        fold = folds[center]
        training_inventory = outcome_universe.bind_predictions(fold.predictions)
        training_frontier = _frontier(training_inventory, config)
        training_selected = _select(training_frontier)
        threshold = (
            DISABLED_ACCEPTANCE_THRESHOLD
            if training_selected is None
            else training_selected.acceptance_threshold
        )
        margin = 0.0 if training_selected is None else training_selected.rank_margin_threshold
        heldout_thresholds.append((center, threshold, margin))
        heldout_inventory = outcome_universe.bind_predictions(
            fold.heldout_predictions
        )
        decisions = _decision_rows(
            heldout_inventory,
            acceptance_threshold=threshold,
            rank_margin_threshold=margin,
        )
        if set(nested_decisions) & set(decisions):
            raise ProtocolError("HARP v12 nested held-source decisions overlap.")
        nested_decisions.update(decisions)
    if set(nested_decisions) != {row.key for row in inventory.contexts}:
        raise ProtocolError("HARP v12 nested held-source inventory is incomplete.")
    nested_replay = _replay_from_decisions(
        nested_decisions,
        acceptance_threshold=(
            selected_replay.acceptance_threshold
            if selected is not None
            else DISABLED_ACCEPTANCE_THRESHOLD
        ),
        rank_margin_threshold=selected_replay.rank_margin_threshold,
        config=config,
    )
    return PolicyCalibration(
        outer_target_id=outer,
        calibrated=bool(selected is not None and nested_replay.safe),
        acceptance_threshold=selected_replay.acceptance_threshold,
        rank_margin_threshold=selected_replay.rank_margin_threshold,
        selected_replay=selected_replay,
        nested_replay=nested_replay,
        heldout_thresholds=tuple(heldout_thresholds),
        frontier=frontier,
        config=config,
    )


__all__ = (
    "DISABLED_ACCEPTANCE_THRESHOLD",
    "PolicyCalibration",
    "PolicyReplay",
    "PolicyRiskConfig",
    "calibrate_selected_policy",
)
