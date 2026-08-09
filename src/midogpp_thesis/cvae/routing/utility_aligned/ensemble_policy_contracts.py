"""Policy roles, bootstrap constants, and immutable policy contract."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


ENSEMBLE_QUERY_BOOTSTRAP_SEED = 20_260_809
ENSEMBLE_QUERY_BOOTSTRAP_DRAWS = 10_000
ENSEMBLE_GAIN_LCB_MULTIPLIER = 1.96
ENSEMBLE_BASE_ROLE = "B"
ENSEMBLE_GLOBAL_ROLE = "G"
ENSEMBLE_ROUTED_ROLE = "R"
ENSEMBLE_PERMUTATION_ROLE = "P"
ENSEMBLE_AUTHORIZATION_UNCERTAINTY_COMPONENTS = (
    "model_covariance_and_residual",
    "independent_whole_case_bootstrap",
)
ENSEMBLE_TARGET_SEED_SPREAD_ROLE = "descriptive_only_non_decision"


@dataclass(frozen=True)
class EnsembleUtilityPolicy:
    target_id: str
    selected_action_role: str
    selected_source: str | None
    exact_b_fallback: bool
    fallback_reason: str | None
    role_selected_source: Mapping[str, str | None]
    role_selected_action: Mapping[str, str]
    role_prediction_by_source: Mapping[str, Mapping[str, float]]
    role_model_standard_error_by_source: Mapping[str, Mapping[str, float]]
    role_bootstrap_standard_deviation_by_source: Mapping[str, Mapping[str, float]]
    role_target_scalar_seed_standard_deviation_by_source: Mapping[
        str, Mapping[str, float]
    ]
    role_combined_standard_error_by_source: Mapping[str, Mapping[str, float]]
    role_lower_confidence_bound_by_source: Mapping[str, Mapping[str, float]]
    routed_bootstrap_mean_by_source: Mapping[str, float]
    routed_bootstrap_standard_deviation_by_source: Mapping[str, float]
    point_feature_surface_hash: str
    bootstrap_feature_surface_hashes: tuple[str, ...]
    cardinality_transfer_hash: str
    policy_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "role_selected_source",
            MappingProxyType(
                {
                    str(key): (None if value is None else str(value))
                    for key, value in self.role_selected_source.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "role_selected_action",
            MappingProxyType(
                {str(key): str(value) for key, value in self.role_selected_action.items()}
            ),
        )
        nested = {
            str(role): MappingProxyType(
                {str(source): float(value) for source, value in values.items()}
            )
            for role, values in self.role_prediction_by_source.items()
        }
        object.__setattr__(self, "role_prediction_by_source", MappingProxyType(nested))
        for field_name in (
            "role_model_standard_error_by_source",
            "role_bootstrap_standard_deviation_by_source",
            "role_target_scalar_seed_standard_deviation_by_source",
            "role_combined_standard_error_by_source",
            "role_lower_confidence_bound_by_source",
        ):
            values = getattr(self, field_name)
            normalized = {
                str(role): MappingProxyType(
                    {str(source): float(value) for source, value in by_source.items()}
                )
                for role, by_source in values.items()
            }
            object.__setattr__(self, field_name, MappingProxyType(normalized))
        object.__setattr__(
            self,
            "routed_bootstrap_mean_by_source",
            MappingProxyType(
                {
                    str(key): float(value)
                    for key, value in self.routed_bootstrap_mean_by_source.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "routed_bootstrap_standard_deviation_by_source",
            MappingProxyType(
                {
                    str(key): float(value)
                    for key, value in self.routed_bootstrap_standard_deviation_by_source.items()
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_ensemble_policy_v1",
            "target_id": self.target_id,
            "selected_action_role": self.selected_action_role,
            "selected_source": self.selected_source,
            "exact_b_fallback": self.exact_b_fallback,
            "fallback_reason": self.fallback_reason,
            "role_selected_source": dict(self.role_selected_source),
            "role_selected_action": dict(self.role_selected_action),
            "role_prediction_by_source": {
                role: dict(values)
                for role, values in self.role_prediction_by_source.items()
            },
            "role_model_standard_error_by_source": {
                role: dict(values)
                for role, values in self.role_model_standard_error_by_source.items()
            },
            "role_bootstrap_standard_deviation_by_source": {
                role: dict(values)
                for role, values in self.role_bootstrap_standard_deviation_by_source.items()
            },
            "role_target_scalar_seed_standard_deviation_by_source": {
                role: dict(values)
                for role, values in self.role_target_scalar_seed_standard_deviation_by_source.items()
            },
            "role_combined_standard_error_by_source": {
                role: dict(values)
                for role, values in self.role_combined_standard_error_by_source.items()
            },
            "role_lower_confidence_bound_by_source": {
                role: dict(values)
                for role, values in self.role_lower_confidence_bound_by_source.items()
            },
            "gain_lcb_multiplier": ENSEMBLE_GAIN_LCB_MULTIPLIER,
            "bootstrap_dispersion_divided_by_seed_repeat_sqrt": False,
            "authorization_uncertainty_components": list(
                ENSEMBLE_AUTHORIZATION_UNCERTAINTY_COMPONENTS
            ),
            "target_scalar_seed_spread_role": ENSEMBLE_TARGET_SEED_SPREAD_ROLE,
            "target_scalar_seed_spread_enters_combined_standard_error": False,
            "routed_bootstrap_mean_by_source": dict(
                self.routed_bootstrap_mean_by_source
            ),
            "routed_bootstrap_standard_deviation_by_source": dict(
                self.routed_bootstrap_standard_deviation_by_source
            ),
            "point_feature_surface_hash": self.point_feature_surface_hash,
            "bootstrap_feature_surface_hashes": list(
                self.bootstrap_feature_surface_hashes
            ),
            "cardinality_transfer_hash": self.cardinality_transfer_hash,
            "policy_hash": self.policy_hash,
        }




__all__ = (
    "ENSEMBLE_BASE_ROLE",
    "ENSEMBLE_AUTHORIZATION_UNCERTAINTY_COMPONENTS",
    "ENSEMBLE_GAIN_LCB_MULTIPLIER",
    "ENSEMBLE_GLOBAL_ROLE",
    "ENSEMBLE_PERMUTATION_ROLE",
    "ENSEMBLE_QUERY_BOOTSTRAP_DRAWS",
    "ENSEMBLE_QUERY_BOOTSTRAP_SEED",
    "ENSEMBLE_ROUTED_ROLE",
    "ENSEMBLE_TARGET_SEED_SPREAD_ROLE",
    "EnsembleUtilityPolicy",
)
