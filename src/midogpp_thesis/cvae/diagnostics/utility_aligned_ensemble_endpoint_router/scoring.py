"""Terminal target scoring for exact-nine probability ensembles."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...metrics import spearman
from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned.ensemble_endpoint import (
    score_nine_seed_probability_ensemble,
)
from ...routing.utility_aligned.ensemble_endpoint_contracts import (
    ENSEMBLE_SEED_PAIR_COUNT,
    ProbabilityEnsembleEndpoint,
    SeedProbabilityVector,
)
from .actions import FrozenEnsembleEndpointActionLibrary
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    EXPECTED_FROZEN_TARGET_ACTION_COUNT,
    ROUTED_ENSEMBLE_ACTION_ID,
    candidate_sources,
    expected_target_action_ids,
    h_x_e_action_id,
)
from .diagnostic_plan import Stage90EnsembleDiagnosticPlanSet


@dataclass(frozen=True)
class TargetEnsembleEndpointScore:
    target_id: str
    action_id: str
    action_hash: str
    router_plan_hash: str
    support_partition_hash: str
    evaluation_partition_hash: str
    prediction_seal_hash: str
    target_probe_seal_hash: str
    evaluation_case_count: int
    endpoint: ProbabilityEnsembleEndpoint
    score_hash: str = field(init=False)
    support_eval_disjoint: bool = True
    global_target_seal_verified_before_labels: bool = True
    terminal_target_labels_used_for_plan: bool = False
    may_update_policy: bool = False
    diagnostic_only: bool = True

    def __post_init__(self) -> None:
        target = str(self.target_id)
        action = str(self.action_id)
        if (
            target not in CENTERS
            or action not in expected_target_action_ids(target)
            or not isinstance(self.endpoint, ProbabilityEnsembleEndpoint)
            or self.evaluation_case_count <= 0
            or any(
                not _is_hash(value)
                for value in (
                    self.action_hash,
                    self.router_plan_hash,
                    self.support_partition_hash,
                    self.evaluation_partition_hash,
                    self.prediction_seal_hash,
                    self.target_probe_seal_hash,
                )
            )
            or self.support_partition_hash == self.evaluation_partition_hash
            or self.support_eval_disjoint is not True
            or self.global_target_seal_verified_before_labels is not True
            or self.terminal_target_labels_used_for_plan is not False
            or self.may_update_policy is not False
            or self.diagnostic_only is not True
        ):
            raise ProtocolError("Target ensemble endpoint score boundary drifted.")
        object.__setattr__(self, "target_id", target)
        object.__setattr__(self, "action_id", action)
        object.__setattr__(self, "score_hash", canonical_sha256(self.to_payload()))

    @property
    def balanced_accuracy(self) -> float:
        return self.endpoint.balanced_accuracy

    @property
    def seed_pair_count(self) -> int:
        return len(self.endpoint.seed_keys)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_stage90_ensemble_target_score_v1",
            "target_center": self.target_id,
            "action_id": self.action_id,
            "action_hash": self.action_hash,
            "router_plan_hash": self.router_plan_hash,
            "support_partition_hash": self.support_partition_hash,
            "evaluation_partition_hash": self.evaluation_partition_hash,
            "prediction_seal_hash": self.prediction_seal_hash,
            "target_probe_seal_hash": self.target_probe_seal_hash,
            "evaluation_row_identity_hash": self.endpoint.row_identity_hash,
            "evaluation_label_hash": self.endpoint.label_hash,
            "endpoint_hash": self.endpoint.endpoint_hash,
            "component_vector_hashes": list(self.endpoint.component_vector_hashes),
            "ensemble_probability_hash": _array_hash(
                self.endpoint.mean_positive_probabilities
            ),
            "ensemble_prediction_hash": _array_hash(self.endpoint.predictions),
            "seed_pair_count": self.seed_pair_count,
            "row_count": self.endpoint.row_count,
            "case_count": self.evaluation_case_count,
            "balanced_accuracy": self.balanced_accuracy,
            "primary_endpoint": "all_nine_seed_probability_ensemble_bacc",
            "support_eval_disjoint": True,
            "global_target_seal_verified_before_labels": True,
            "terminal_target_labels_used_for_plan": False,
            "may_update_policy": False,
            "inference_unit": "target_center",
            "diagnostic_only": True,
        }


@dataclass(frozen=True)
class TargetEnsembleEndpointScoreSet:
    rows: tuple[TargetEnsembleEndpointScore, ...]
    prediction_seal_hash: str
    action_library_hash: str
    score_set_hash: str

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        keys = tuple((row.target_id, row.action_id) for row in rows)
        expected_keys = tuple(
            (target, action_id)
            for target in CENTERS
            for action_id in expected_target_action_ids(target)
        )
        if (
            len(rows) != EXPECTED_FROZEN_TARGET_ACTION_COUNT
            or keys != expected_keys
            or len(set(keys)) != len(keys)
            or any(row.prediction_seal_hash != self.prediction_seal_hash for row in rows)
        ):
            raise ProtocolError("Target ensemble endpoint score-set coverage drifted.")
        expected = canonical_sha256(
            _score_set_payload(
                rows,
                prediction_seal_hash=self.prediction_seal_hash,
                action_library_hash=self.action_library_hash,
            )
        )
        if self.score_set_hash != expected:
            raise ProtocolError("Target ensemble endpoint score-set hash drifted.")

    @property
    def by_key(self) -> Mapping[tuple[str, str], TargetEnsembleEndpointScore]:
        return MappingProxyType(
            {(row.target_id, row.action_id): row for row in self.rows}
        )

    def to_payload(self) -> dict[str, object]:
        return {
            **_score_set_payload(
                self.rows,
                prediction_seal_hash=self.prediction_seal_hash,
                action_library_hash=self.action_library_hash,
            ),
            "score_set_hash": self.score_set_hash,
        }


@dataclass(frozen=True)
class TerminalHxeOracleDiagnostic:
    target_id: str
    routed_selected_source: str
    oracle_source_ids: tuple[str, ...]
    routed_top1_exact_agreement: bool
    routed_top1_tie_agreement: bool
    predicted_gain_utility_spearman: float | None
    routed_hxe_bacc: float
    oracle_hxe_bacc: float
    base_bacc: float
    normalized_oracle_gap: float
    plan_hash: str
    target_score_set_hash: str
    diagnostic_hash: str
    may_update_policy: bool = False
    diagnostic_only: bool = True

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_stage90_ensemble_hxe_oracle_v1",
            "target_center": self.target_id,
            "R2E_selected_source": self.routed_selected_source,
            "oracle_source_ids": list(self.oracle_source_ids),
            "R2E_top1_exact_agreement": self.routed_top1_exact_agreement,
            "R2E_top1_tie_agreement": self.routed_top1_tie_agreement,
            "predicted_gain_utility_spearman": self.predicted_gain_utility_spearman,
            "R2E_Hxe_bacc": self.routed_hxe_bacc,
            "oracle_Hxe_bacc": self.oracle_hxe_bacc,
            "base_bacc": self.base_bacc,
            "normalized_oracle_gap": self.normalized_oracle_gap,
            "plan_hash": self.plan_hash,
            "target_score_set_hash": self.target_score_set_hash,
            "row_role": "terminal_oracle_diagnostic",
            "Hxe_may_feed_plan": False,
            "may_update_policy": False,
            "diagnostic_only": True,
        }


def score_target_action_ensemble_endpoint(
    *,
    target_id: str,
    action_id: str,
    vectors: Sequence[SeedProbabilityVector],
    labels: Sequence[int] | np.ndarray,
    action_hash: str,
    router_plan_hash: str,
    support_partition_hash: str,
    evaluation_partition_hash: str,
    prediction_seal_hash: str,
    target_probe_seal_hash: str,
    evaluation_case_count: int,
    global_target_seal_verified: bool,
) -> TargetEnsembleEndpointScore:
    """Score one action only after the globally complete target seal exists."""

    if global_target_seal_verified is not True:
        raise ProtocolError("Terminal labels require a verified global target seal.")
    endpoint = score_nine_seed_probability_ensemble(vectors, labels)
    return TargetEnsembleEndpointScore(
        target_id=target_id,
        action_id=action_id,
        action_hash=action_hash,
        router_plan_hash=router_plan_hash,
        support_partition_hash=support_partition_hash,
        evaluation_partition_hash=evaluation_partition_hash,
        prediction_seal_hash=prediction_seal_hash,
        target_probe_seal_hash=target_probe_seal_hash,
        evaluation_case_count=evaluation_case_count,
        endpoint=endpoint,
    )


def validate_target_ensemble_endpoint_scores(
    rows: Sequence[TargetEnsembleEndpointScore],
    actions: FrozenEnsembleEndpointActionLibrary,
) -> TargetEnsembleEndpointScoreSet:
    """Validate complete 9x13 scores against actions and one global seal."""

    if not isinstance(actions, FrozenEnsembleEndpointActionLibrary):
        raise ProtocolError("Target score validation requires the frozen action library.")
    ordered = tuple(sorted(rows, key=lambda row: (CENTERS.index(row.target_id), expected_target_action_ids(row.target_id).index(row.action_id))))
    if any(not isinstance(row, TargetEnsembleEndpointScore) for row in ordered):
        raise ProtocolError("Target score rows must use their typed contract.")
    if len({row.prediction_seal_hash for row in ordered}) != 1:
        raise ProtocolError("All terminal target scores require one global prediction seal.")
    for row in ordered:
        action = actions.action(row.target_id, row.action_id)
        if (
            row.action_hash != action.action_hash
            or row.router_plan_hash != action.router_plan_hash
        ):
            raise ProtocolError("Target endpoint score/action binding drifted.")
    seal_hash = ordered[0].prediction_seal_hash
    payload = _score_set_payload(
        ordered,
        prediction_seal_hash=seal_hash,
        action_library_hash=actions.action_library_hash,
    )
    return TargetEnsembleEndpointScoreSet(
        rows=ordered,
        prediction_seal_hash=seal_hash,
        action_library_hash=actions.action_library_hash,
        score_set_hash=canonical_sha256(payload),
    )


def build_terminal_hxe_oracle_diagnostics(
    plans: Stage90EnsembleDiagnosticPlanSet,
    scores: TargetEnsembleEndpointScoreSet,
) -> tuple[TerminalHxeOracleDiagnostic, ...]:
    """Open Hxe results terminally; they cannot alter the frozen plan."""

    if (
        not isinstance(plans, Stage90EnsembleDiagnosticPlanSet)
        or not isinstance(scores, TargetEnsembleEndpointScoreSet)
    ):
        raise ProtocolError("Hxe oracle diagnostics require typed sealed inputs.")
    output: list[TerminalHxeOracleDiagnostic] = []
    metric = scores.by_key
    for target in CENTERS:
        plan = plans.by_target[target]
        sources = candidate_sources(target)
        selected = plan.proposed_source_by_router[ROUTED_ENSEMBLE_ACTION_ID]
        utilities = {
            source: metric[(target, h_x_e_action_id(source))].balanced_accuracy
            for source in sources
        }
        maximum = max(utilities.values())
        minimum = min(utilities.values())
        oracle_sources = tuple(
            source
            for source in sources
            if math.isclose(utilities[source], maximum, abs_tol=1.0e-15)
        )
        denominator = maximum - minimum
        gap = 0.0 if denominator == 0.0 else (maximum - utilities[selected]) / denominator
        predicted = plan.prediction_by_router_source[ROUTED_ENSEMBLE_ACTION_ID]
        rho = spearman(
            [predicted[source] for source in sources],
            [utilities[source] for source in sources],
        )
        unhashed = {
            "schema_version": "midogpp_utility_aligned_stage90_ensemble_hxe_oracle_v1",
            "target_center": target,
            "R2E_selected_source": selected,
            "oracle_source_ids": list(oracle_sources),
            "R2E_top1_exact_agreement": selected == min(oracle_sources),
            "R2E_top1_tie_agreement": selected in oracle_sources,
            "predicted_gain_utility_spearman": rho,
            "R2E_Hxe_bacc": utilities[selected],
            "oracle_Hxe_bacc": maximum,
            "base_bacc": metric[(target, BASE_ACTION_ID)].balanced_accuracy,
            "normalized_oracle_gap": gap,
            "plan_hash": plan.plan_hash,
            "target_score_set_hash": scores.score_set_hash,
            "row_role": "terminal_oracle_diagnostic",
            "Hxe_may_feed_plan": False,
            "may_update_policy": False,
            "diagnostic_only": True,
        }
        output.append(
            TerminalHxeOracleDiagnostic(
                target_id=target,
                routed_selected_source=selected,
                oracle_source_ids=oracle_sources,
                routed_top1_exact_agreement=selected == min(oracle_sources),
                routed_top1_tie_agreement=selected in oracle_sources,
                predicted_gain_utility_spearman=rho,
                routed_hxe_bacc=utilities[selected],
                oracle_hxe_bacc=maximum,
                base_bacc=metric[(target, BASE_ACTION_ID)].balanced_accuracy,
                normalized_oracle_gap=gap,
                plan_hash=plan.plan_hash,
                target_score_set_hash=scores.score_set_hash,
                diagnostic_hash=canonical_sha256(unhashed),
            )
        )
    return tuple(output)


def _score_set_payload(
    rows: Sequence[TargetEnsembleEndpointScore],
    *,
    prediction_seal_hash: str,
    action_library_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_stage90_ensemble_target_score_set_v1",
        "centers": list(CENTERS),
        "score_hashes": [row.score_hash for row in rows],
        "prediction_seal_hash": prediction_seal_hash,
        "target_probe_seal_hashes": sorted({row.target_probe_seal_hash for row in rows}),
        "action_library_hash": action_library_hash,
        "row_count": len(rows),
        "primary_endpoint": "all_nine_seed_probability_ensemble_bacc",
        "inference_unit": "target_center",
        "technical_seed_cells_are_independent_units": False,
        "global_target_seal_verified_before_labels": True,
        "diagnostic_only": True,
    }


def _array_hash(value: np.ndarray) -> str:
    from ...routing.residual_topup.hashing import array_sha256

    return array_sha256(value)


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 16 and value.strip() == value


__all__ = (
    "TargetEnsembleEndpointScore",
    "TargetEnsembleEndpointScoreSet",
    "TerminalHxeOracleDiagnostic",
    "build_terminal_hxe_oracle_diagnostics",
    "score_target_action_ensemble_endpoint",
    "validate_target_ensemble_endpoint_scores",
)
