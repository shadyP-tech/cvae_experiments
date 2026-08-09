"""Low-capacity model and source-inner transfer contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..local_marginal_utility.ridge import ClusterWeightedRidgeModel


ROUTING_TUNING_ENDPOINT = "mean_normalized_oracle_regret"


@dataclass(frozen=True)
class EnsembleCapacityReport:
    context: str
    independent_query_count: int
    observation_count: int
    predictor_column_count: int
    design_column_count: int
    design_rank: int
    constant_columns: tuple[str, ...]
    sandwich_rank_ceiling: int
    observed_sandwich_rank: int | None
    gate_passed: bool
    failures: tuple[str, ...]
    report_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_ensemble_capacity_report_v1",
            "context": self.context,
            "independent_query_count": self.independent_query_count,
            "observation_count": self.observation_count,
            "predictor_column_count": self.predictor_column_count,
            "design_column_count": self.design_column_count,
            "design_rank": self.design_rank,
            "constant_columns": list(self.constant_columns),
            "sandwich_rank_ceiling": self.sandwich_rank_ceiling,
            "observed_sandwich_rank": self.observed_sandwich_rank,
            "gate_passed": self.gate_passed,
            "failures": list(self.failures),
            "report_hash": self.report_hash,
        }


@dataclass(frozen=True)
class EnsembleFoldAudit:
    predicted_row_key: tuple[str, str, str]
    excluded_domain_ids: tuple[str, ...]
    training_query_ids: tuple[str, ...]
    training_source_ids: tuple[str, ...]
    selected_alpha: float
    capacity_report_hash: str
    strict_h_q_e_exclusion: bool = True


@dataclass(frozen=True)
class EnsembleUtilityModel:
    """Low-capacity candidate-specific models and strict crossfit evidence."""

    outer_target_id: str
    feature_names: tuple[str, ...]
    selected_alpha: float
    routing_tuning_endpoint: str
    routing_loss_by_alpha: Mapping[float, float]
    selected_alpha_by_heldout_query: Mapping[str, float]
    candidate_models: Mapping[str, ClusterWeightedRidgeModel]
    candidate_capacity_reports: Mapping[str, EnsembleCapacityReport]
    crossfit_predictions: np.ndarray
    crossfit_row_keys: tuple[tuple[str, str, str], ...]
    fold_audits: tuple[EnsembleFoldAudit, ...]
    feature_surface_hash: str
    utility_surface_hash: str
    permutation_seed: int | None
    model_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "routing_loss_by_alpha",
            MappingProxyType(
                {float(key): float(value) for key, value in self.routing_loss_by_alpha.items()}
            ),
        )
        object.__setattr__(
            self,
            "selected_alpha_by_heldout_query",
            MappingProxyType(
                {
                    str(key): float(value)
                    for key, value in self.selected_alpha_by_heldout_query.items()
                }
            ),
        )
        object.__setattr__(self, "candidate_models", MappingProxyType(dict(self.candidate_models)))
        object.__setattr__(
            self,
            "candidate_capacity_reports",
            MappingProxyType(dict(self.candidate_capacity_reports)),
        )


@dataclass(frozen=True)
class EnsembleCardinalityTransferResult:
    outer_target_id: str
    independent_query_count: int
    source_inner_candidate_count: int
    deployment_candidate_count: int
    metrics_by_role: Mapping[str, Mapping[str, float]]
    bootstrap_bounds_by_role: Mapping[str, Mapping[str, float]]
    paired_improvement_bounds: Mapping[str, Mapping[str, float]]
    capacity_report_hashes_by_role: Mapping[str, tuple[str, ...]]
    query_bootstrap_seed: int
    query_bootstrap_draw_count: int
    query_bootstrap_indices_hash: str
    all_capacity_gates_passed: bool
    authorized_for_target_policy: bool
    authorization_failures: tuple[str, ...]
    transfer_hash: str

    def __post_init__(self) -> None:
        metrics = {
            str(role): MappingProxyType(
                {str(name): float(value) for name, value in values.items()}
            )
            for role, values in self.metrics_by_role.items()
        }
        capacity = {
            str(role): tuple(str(value) for value in values)
            for role, values in self.capacity_report_hashes_by_role.items()
        }
        bootstrap_bounds = {
            str(role): MappingProxyType(
                {str(name): float(value) for name, value in values.items()}
            )
            for role, values in self.bootstrap_bounds_by_role.items()
        }
        paired_bounds = {
            str(comparator): MappingProxyType(
                {str(name): float(value) for name, value in values.items()}
            )
            for comparator, values in self.paired_improvement_bounds.items()
        }
        object.__setattr__(self, "metrics_by_role", MappingProxyType(metrics))
        object.__setattr__(
            self, "bootstrap_bounds_by_role", MappingProxyType(bootstrap_bounds)
        )
        object.__setattr__(
            self, "paired_improvement_bounds", MappingProxyType(paired_bounds)
        )
        object.__setattr__(
            self, "capacity_report_hashes_by_role", MappingProxyType(capacity)
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                "midogpp_utility_aligned_ensemble_cardinality_transfer_v1"
            ),
            "outer_target_id": self.outer_target_id,
            "independent_query_count": self.independent_query_count,
            "source_inner_candidate_count": self.source_inner_candidate_count,
            "deployment_candidate_count": self.deployment_candidate_count,
            "metrics_by_role": {
                role: dict(values) for role, values in self.metrics_by_role.items()
            },
            "bootstrap_bounds_by_role": {
                role: dict(values)
                for role, values in self.bootstrap_bounds_by_role.items()
            },
            "paired_improvement_bounds": {
                comparator: dict(values)
                for comparator, values in self.paired_improvement_bounds.items()
            },
            "capacity_report_hashes_by_role": {
                role: list(values)
                for role, values in self.capacity_report_hashes_by_role.items()
            },
            "all_capacity_gates_passed": self.all_capacity_gates_passed,
            "query_bootstrap_seed": self.query_bootstrap_seed,
            "query_bootstrap_draw_count": self.query_bootstrap_draw_count,
            "query_bootstrap_indices_hash": self.query_bootstrap_indices_hash,
            "authorized_for_target_policy": self.authorized_for_target_policy,
            "authorization_failures": list(self.authorization_failures),
            "claim_role": "source_inner_ensemble_eligibility_only_not_target_utility",
            "transfer_hash": self.transfer_hash,
        }




__all__ = (
    "ROUTING_TUNING_ENDPOINT",
    "EnsembleCapacityReport",
    "EnsembleCardinalityTransferResult",
    "EnsembleFoldAudit",
    "EnsembleUtilityModel",
)

