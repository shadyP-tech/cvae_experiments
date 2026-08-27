"""Terminal-only scoring of the already sealed SCEPTRE route policy."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError

from ..fixed_bank_sceptre_router.hashing import canonical_hash, require_sha256
from ..fixed_bank_sceptre_router.outcome_surface import (
    EXACT_B_CANDIDATE,
    ConfusionCounts,
)
from ..fixed_bank_sceptre_router.route_policy import FrozenRoutePolicy
from ..fixed_bank_sceptre_router.seals import EXPECTED_DECISION_KEYS
from ..fixed_bank_sceptre_router.uncertainty import (
    SEED_CELL_COUNT,
    RolePredictionSurface,
)


@dataclass(frozen=True, slots=True)
class ActionAggregate:
    confusion: ConfusionCounts
    brier_sum: float
    log_loss_sum: float
    observation_count: int

    @property
    def bacc(self) -> float:
        return self.confusion.bacc

    @property
    def brier(self) -> float:
        return self.brier_sum / (self.observation_count * SEED_CELL_COUNT)

    @property
    def log_loss(self) -> float:
        return self.log_loss_sum / (self.observation_count * SEED_CELL_COUNT)

    def to_payload(self) -> dict[str, object]:
        return {
            "confusion": {
                "tn": self.confusion.tn,
                "fp": self.confusion.fp,
                "fn": self.confusion.fn,
                "tp": self.confusion.tp,
            },
            "observation_count": self.observation_count,
            "seed_cell_count": SEED_CELL_COUNT,
            "bacc": self.bacc,
            "brier_sum": self.brier_sum,
            "brier": self.brier,
            "log_loss_sum": self.log_loss_sum,
            "log_loss": self.log_loss,
        }


@dataclass(frozen=True, slots=True)
class TerminalFoldMetric:
    target_center: str
    fold_ordinal: int
    fold_hash: str
    evaluation_case_set_hash: str
    route: str
    route_aggregate: ActionAggregate
    exact_b_aggregate: ActionAggregate
    oracle_action: str
    oracle_bacc: float
    case_count: int
    fold_metric_hash: str = ""

    def __post_init__(self) -> None:
        if (
            (self.target_center, self.fold_ordinal) not in set(EXPECTED_DECISION_KEYS)
            or self.case_count <= 0
            or self.route_aggregate.observation_count
            != self.exact_b_aggregate.observation_count
            or not math.isfinite(self.oracle_bacc)
        ):
            raise ProtocolError("SCEPTRE v3 terminal fold metric drifted.")
        require_sha256(self.fold_hash, "terminal fold")
        require_sha256(self.evaluation_case_set_hash, "terminal case set")
        expected = canonical_hash(self._payload_without_hash())
        if self.fold_metric_hash and self.fold_metric_hash != expected:
            raise ProtocolError("SCEPTRE v3 terminal fold metric hash drifted.")
        object.__setattr__(self, "fold_metric_hash", expected)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_v3_terminal_fold_metric_v1",
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "fold_hash": self.fold_hash,
            "evaluation_case_set_hash": self.evaluation_case_set_hash,
            "route": self.route,
            "route_metrics": self.route_aggregate.to_payload(),
            "exact_b_metrics": self.exact_b_aggregate.to_payload(),
            "route_minus_exact_b": {
                "bacc": self.route_aggregate.bacc - self.exact_b_aggregate.bacc,
                "brier": self.route_aggregate.brier - self.exact_b_aggregate.brier,
                "log_loss": (
                    self.route_aggregate.log_loss - self.exact_b_aggregate.log_loss
                ),
            },
            "oracle_action_descriptive_only": self.oracle_action,
            "oracle_bacc_descriptive_only": self.oracle_bacc,
            "route_matches_descriptive_oracle": self.route == self.oracle_action,
            "case_count": self.case_count,
            "raw_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "fold_metric_hash": self.fold_metric_hash}


@dataclass(frozen=True, slots=True)
class TerminalEvaluationResult:
    route_policy_hash: str
    prediction_store_hash: str
    terminal_capability_hash: str
    folds: tuple[TerminalFoldMetric, ...]
    result_hash: str = ""
    _summary: Mapping[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        rows = tuple(self.folds)
        if tuple((row.target_center, row.fold_ordinal) for row in rows) != (
            EXPECTED_DECISION_KEYS
        ):
            raise ProtocolError("SCEPTRE v3 terminal result lacks 45 folds.")
        for value, role in (
            (self.route_policy_hash, "terminal route policy"),
            (self.prediction_store_hash, "terminal prediction store"),
            (self.terminal_capability_hash, "terminal capability"),
        ):
            require_sha256(value, role)
        summary = _summary_payload(rows)
        body = {
            "schema_version": "sceptre_v3_terminal_evaluation_result_v1",
            "route_policy_hash": self.route_policy_hash,
            "prediction_store_hash": self.prediction_store_hash,
            "terminal_capability_hash": self.terminal_capability_hash,
            "fold_metric_hashes": [row.fold_metric_hash for row in rows],
            "summary": summary,
            "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
            "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
            "fresh_evidence": False,
            "routing_success_claimed": False,
            "nelbo_compatibility_claimed": False,
            "raw_labels_persisted": False,
        }
        expected = canonical_hash(body)
        if self.result_hash and self.result_hash != expected:
            raise ProtocolError("SCEPTRE v3 terminal result hash drifted.")
        object.__setattr__(self, "folds", rows)
        object.__setattr__(self, "_summary", summary)
        object.__setattr__(self, "result_hash", expected)

    @property
    def summary(self) -> Mapping[str, object]:
        return self._summary

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_v3_terminal_evaluation_result_v1",
            "route_policy_hash": self.route_policy_hash,
            "prediction_store_hash": self.prediction_store_hash,
            "terminal_capability_hash": self.terminal_capability_hash,
            "folds": [row.to_payload() for row in self.folds],
            "summary": dict(self.summary),
            "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
            "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
            "fresh_evidence": False,
            "routing_success_claimed": False,
            "nelbo_compatibility_claimed": False,
            "raw_labels_persisted": False,
            "result_hash": self.result_hash,
        }


def evaluate_terminal_surfaces(
    route_policy: FrozenRoutePolicy,
    surfaces: Sequence[RolePredictionSurface],
    *,
    prediction_store_hash: str,
    terminal_capability_hash: str,
) -> TerminalEvaluationResult:
    """Score without changing a single sealed route."""

    if not isinstance(route_policy, FrozenRoutePolicy):
        raise ProtocolError("SCEPTRE v3 terminal evaluation requires its route policy.")
    by_key = {
        (surface.target_center, surface.fold.fold_ordinal): surface
        for surface in surfaces
        if isinstance(surface, RolePredictionSurface)
    }
    if set(by_key) != set(EXPECTED_DECISION_KEYS) or len(by_key) != len(surfaces):
        raise ProtocolError("SCEPTRE v3 terminal surfaces lack exact fold coverage.")
    folds: list[TerminalFoldMetric] = []
    for key in EXPECTED_DECISION_KEYS:
        surface = by_key[key]
        if (
            surface.role != "EVALUATION"
            or surface.partition_hash != route_policy.partition_hash
            or surface.phase_capability.route_policy_hash
            != route_policy.policy_artifact_hash
            or surface.phase_capability.capability_hash != terminal_capability_hash
        ):
            raise ProtocolError("SCEPTRE v3 evaluation surface lineage drifted.")
        route = route_policy.route_for(*key)
        aggregates = {
            action.action_id: _aggregate_action(surface, action.action_id)
            for action in surface.actions
        }
        oracle_bacc = max(row.bacc for row in aggregates.values())
        oracle = next(
            action.action_id
            for action in surface.actions
            if aggregates[action.action_id].bacc == oracle_bacc
        )
        folds.append(
            TerminalFoldMetric(
                target_center=key[0],
                fold_ordinal=key[1],
                fold_hash=surface.fold.fold_hash,
                evaluation_case_set_hash=surface.fold.case_set_hash("EVALUATION"),
                route=route,
                route_aggregate=aggregates[route],
                exact_b_aggregate=aggregates[EXACT_B_CANDIDATE],
                oracle_action=oracle,
                oracle_bacc=oracle_bacc,
                case_count=len(surface.whole_case_ids),
            )
        )
    return TerminalEvaluationResult(
        route_policy_hash=route_policy.policy_artifact_hash,
        prediction_store_hash=require_sha256(
            prediction_store_hash, "terminal prediction store"
        ),
        terminal_capability_hash=require_sha256(
            terminal_capability_hash, "terminal capability"
        ),
        folds=tuple(folds),
    )


def _aggregate_action(
    surface: RolePredictionSurface, action_id: str
) -> ActionAggregate:
    action = next(row for row in surface.actions if row.action_id == action_id)
    probabilities = np.asarray(
        [cell.probabilities for cell in action.seed_cells], dtype=np.float64
    )
    labels = np.asarray(surface.labels, dtype=np.int8)
    predicted = probabilities >= 0.5
    truth = labels.astype(bool)[None, :]
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    target = labels.astype(np.float64)[None, :]
    return ActionAggregate(
        confusion=ConfusionCounts(
            tn=int(np.sum((~truth) & (~predicted))),
            fp=int(np.sum((~truth) & predicted)),
            fn=int(np.sum(truth & (~predicted))),
            tp=int(np.sum(truth & predicted)),
        ),
        brier_sum=float(
            np.sum(np.square(probabilities - target), dtype=np.float64)
        ),
        log_loss_sum=float(
            np.sum(
                -(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped)),
                dtype=np.float64,
            )
        ),
        observation_count=len(labels),
    )


def _pool(rows: Sequence[ActionAggregate]) -> ActionAggregate:
    confusion = ConfusionCounts(0, 0, 0, 0)
    for row in rows:
        confusion = confusion + row.confusion
    return ActionAggregate(
        confusion=confusion,
        brier_sum=sum(row.brier_sum for row in rows),
        log_loss_sum=sum(row.log_loss_sum for row in rows),
        observation_count=sum(row.observation_count for row in rows),
    )


def _summary_payload(rows: Sequence[TerminalFoldMetric]) -> dict[str, object]:
    centers = {}
    for center in CENTERS:
        subset = tuple(row for row in rows if row.target_center == center)
        route = _pool(tuple(row.route_aggregate for row in subset))
        baseline = _pool(tuple(row.exact_b_aggregate for row in subset))
        centers[center] = _comparison_payload(route, baseline)
    route = _pool(tuple(row.route_aggregate for row in rows))
    baseline = _pool(tuple(row.exact_b_aggregate for row in rows))
    return {
        "global": _comparison_payload(route, baseline),
        "by_center": centers,
        "route_counts": {
            action: sum(row.route == action for row in rows)
            for action in (*CENTERS, EXACT_B_CANDIDATE)
        },
        "descriptive_oracle_top1_agreement": (
            sum(row.route == row.oracle_action for row in rows) / len(rows)
        ),
        "fold_count": len(rows),
        "evaluation_cases_exactly_once": True,
        "seed_cells_are_nuisance_replications": True,
        "p_values_or_confidence_intervals_reported": False,
    }


def _comparison_payload(
    route: ActionAggregate, baseline: ActionAggregate
) -> dict[str, object]:
    return {
        "route": route.to_payload(),
        "exact_b": baseline.to_payload(),
        "route_minus_exact_b": {
            "bacc": route.bacc - baseline.bacc,
            "brier": route.brier - baseline.brier,
            "log_loss": route.log_loss - baseline.log_loss,
        },
    }


__all__ = (
    "ActionAggregate",
    "TerminalEvaluationResult",
    "TerminalFoldMetric",
    "evaluate_terminal_surfaces",
)
