"""Immutable deployment policy contract and frozen action identifiers."""

from __future__ import annotations

from dataclasses import dataclass


BASE_ACTION_ID = "B"
UNIFORM_ACTION_ID = "U"
GLOBAL_ACTION_ID = "G_delta"
ROUTED_ACTION_ID = "R"
PERMUTATION_ACTION_ID = "P"

ABSTENTION_SEMANTICS = "exact_base_when_best_gain_lower_confidence_bound_nonpositive"


@dataclass(frozen=True)
class UtilityAlignedPolicy:
    target_id: str
    candidate_sources: tuple[str, ...]
    router_kind: str
    proposed_action_id: str
    action_id: str
    proposed_source: str
    selected_source: str | None
    predicted_gain: float
    standard_error: float
    lower_confidence_bound: float
    confidence_multiplier: float
    minimum_gain: float
    support_case_count: int
    minimum_support_case_count: int
    seed_pair_count: int
    replicate_standard_deviation: float
    support_bootstrap_replicates: int
    minimum_support_bootstrap_replicates: int
    support_bootstrap_standard_deviation: float
    support_bootstrap_surface_hashes: tuple[str, ...]
    case_bootstrap_replicate_hashes: tuple[str, ...]
    used_exact_base_fallback: bool
    fallback_reason: str | None
    global_only: bool
    permutation_seed: int | None
    model_hash: str
    feature_surface_hash: str
    cardinality_eligibility_hash: str
    case_bootstrap_plan_hash: str | None
    target_support_labels_used: bool
    target_evaluation_used: bool
    seed_selection_performed: bool
    abstention_semantics: str
    policy_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_policy_v1",
            "target_id": self.target_id,
            "candidate_sources": list(self.candidate_sources),
            "router_kind": self.router_kind,
            "proposed_action_id": self.proposed_action_id,
            "action_id": self.action_id,
            "proposed_source": self.proposed_source,
            "selected_source": self.selected_source,
            "predicted_gain": self.predicted_gain,
            "standard_error": self.standard_error,
            "lower_confidence_bound": self.lower_confidence_bound,
            "confidence_multiplier": self.confidence_multiplier,
            "minimum_gain": self.minimum_gain,
            "support_case_count": self.support_case_count,
            "minimum_support_case_count": self.minimum_support_case_count,
            "seed_pair_count": self.seed_pair_count,
            "replicate_standard_deviation": self.replicate_standard_deviation,
            "support_bootstrap_replicates": self.support_bootstrap_replicates,
            "minimum_support_bootstrap_replicates": (
                self.minimum_support_bootstrap_replicates
            ),
            "support_bootstrap_standard_deviation": (
                self.support_bootstrap_standard_deviation
            ),
            "support_bootstrap_surface_hashes": list(
                self.support_bootstrap_surface_hashes
            ),
            "case_bootstrap_replicate_hashes": list(
                self.case_bootstrap_replicate_hashes
            ),
            "used_exact_base_fallback": self.used_exact_base_fallback,
            "fallback_reason": self.fallback_reason,
            "global_only": self.global_only,
            "permutation_seed": self.permutation_seed,
            "model_hash": self.model_hash,
            "feature_surface_hash": self.feature_surface_hash,
            "cardinality_eligibility_hash": self.cardinality_eligibility_hash,
            "case_bootstrap_plan_hash": self.case_bootstrap_plan_hash,
            "target_support_labels_used": self.target_support_labels_used,
            "target_evaluation_used": self.target_evaluation_used,
            "seed_selection_performed": self.seed_selection_performed,
            "abstention_semantics": self.abstention_semantics,
            "policy_hash": self.policy_hash,
        }


__all__ = (
    "ABSTENTION_SEMANTICS",
    "BASE_ACTION_ID",
    "GLOBAL_ACTION_ID",
    "PERMUTATION_ACTION_ID",
    "ROUTED_ACTION_ID",
    "UNIFORM_ACTION_ID",
    "UtilityAlignedPolicy",
)
