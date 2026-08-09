"""Backward-compatible facade for candidate-level ensemble contracts.

Definitions live in cohesive endpoint/support, feature/target, model/transfer,
and policy contract modules. Re-exporting the original class objects keeps
legacy imports and identity stable without duplicating implementations.
"""

from .ensemble_endpoint_contracts import (
    ENSEMBLE_ENDPOINT_SEMANTICS,
    ENSEMBLE_SEED_KEYS,
    ENSEMBLE_SEED_PAIR_COUNT,
    ENSEMBLE_THRESHOLD,
    ENSEMBLE_UTILITY_SEMANTICS,
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
    SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS,
    ProbabilityEnsembleEndpoint,
    SeedProbabilityVector,
    SupportActionProbabilityShift,
)
from .ensemble_utility_contracts import (
    EnsembleUtilityResponse,
    EnsembleUtilitySurface,
    ScoredEnsembleUtilityResponse,
)
from .ensemble_feature_contracts import (
    AGGREGATED_FEATURE_NAMES,
    GLOBAL_SOURCE_CONTROL_NAME,
    GLOBAL_SOURCE_CONTROL_SEMANTICS,
    EnsembleCandidateFeatureRow,
    EnsembleFeatureSurface,
    GlobalSourceControl,
    TargetEnsembleFeatureProduction,
    TargetSupportActionShiftCase,
)
from .ensemble_model_contracts import (
    ROUTING_TUNING_ENDPOINT,
    EnsembleCapacityReport,
    EnsembleCardinalityTransferResult,
    EnsembleFoldAudit,
    EnsembleUtilityModel,
)
from .ensemble_policy_contracts import (
    ENSEMBLE_BASE_ROLE,
    ENSEMBLE_GAIN_LCB_MULTIPLIER,
    ENSEMBLE_GLOBAL_ROLE,
    ENSEMBLE_PERMUTATION_ROLE,
    ENSEMBLE_QUERY_BOOTSTRAP_DRAWS,
    ENSEMBLE_QUERY_BOOTSTRAP_SEED,
    ENSEMBLE_ROUTED_ROLE,
    EnsembleUtilityPolicy,
)


__all__ = (
    "AGGREGATED_FEATURE_NAMES",
    "ENSEMBLE_BASE_ROLE",
    "ENSEMBLE_ENDPOINT_SEMANTICS",
    "ENSEMBLE_GAIN_LCB_MULTIPLIER",
    "ENSEMBLE_GLOBAL_ROLE",
    "ENSEMBLE_PERMUTATION_ROLE",
    "ENSEMBLE_QUERY_BOOTSTRAP_DRAWS",
    "ENSEMBLE_QUERY_BOOTSTRAP_SEED",
    "ENSEMBLE_ROUTED_ROLE",
    "ENSEMBLE_SEED_KEYS",
    "ENSEMBLE_SEED_PAIR_COUNT",
    "ENSEMBLE_THRESHOLD",
    "ENSEMBLE_UTILITY_SEMANTICS",
    "GLOBAL_SOURCE_CONTROL_NAME",
    "GLOBAL_SOURCE_CONTROL_SEMANTICS",
    "ROUTING_TUNING_ENDPOINT",
    "SUPPORT_ACTION_PROBABILITY_SHIFT_NAME",
    "SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA",
    "SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS",
    "SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS",
    "EnsembleCandidateFeatureRow",
    "EnsembleCapacityReport",
    "EnsembleCardinalityTransferResult",
    "EnsembleFeatureSurface",
    "EnsembleFoldAudit",
    "EnsembleUtilityModel",
    "EnsembleUtilityPolicy",
    "EnsembleUtilityResponse",
    "EnsembleUtilitySurface",
    "GlobalSourceControl",
    "ProbabilityEnsembleEndpoint",
    "ScoredEnsembleUtilityResponse",
    "SeedProbabilityVector",
    "SupportActionProbabilityShift",
    "TargetEnsembleFeatureProduction",
    "TargetSupportActionShiftCase",
)
