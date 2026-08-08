"""Utility-aligned exact additive-tail routing core.

Public APIs are pure and stage-neutral.  Artifact materialization belongs in
the Stage-50/60 producer and Stage-70 evaluator packages; consumed diagnostics
must never be imported here as training inputs.
"""

from .action_adapter import build_utility_aligned_action
from .policy_contracts import (
    ABSTENTION_SEMANTICS,
    BASE_ACTION_ID,
    GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID,
    ROUTED_ACTION_ID,
    UNIFORM_ACTION_ID,
    UtilityAlignedPolicy,
)
from .result_contracts import (
    CARDINALITY_CLAIM_ROLE,
    MAX_AUTHORIZED_NORMALIZED_ORACLE_GAP,
    MODEL_SEMANTICS,
    SOURCE_INNER_TOP1_CHANCE,
    CardinalityTransferResult,
    CrossfitResult,
    FoldAudit,
    RankingMetrics,
    UtilityAlignedModels,
)
from .row_contracts import (
    DEFAULT_CASE_BOOTSTRAP_SEED,
    EXACT_TAIL_UTILITY_SEMANTICS,
    INNER_CANDIDATE_COUNT,
    INNER_ROLE,
    MIN_SUPPORT_BOOTSTRAP_REPLICATES,
    MIN_TARGET_SUPPORT_CASES,
    SEED_PAIR_COUNT,
    TARGET_CANDIDATE_COUNT,
    TARGET_ROLE,
    TRAIN_CANDIDATE_COUNT_AFTER_STRICT_EXCLUSION,
    CaseBootstrapPlan,
    CaseBootstrapReplicate,
    ExactTailUtilityRow,
    build_case_bootstrap_plan,
)
from .surface_contracts import (
    FEATURE_SEMANTICS,
    CandidateFeatureRow,
    ExactTailUtilitySurface,
    FeatureSurface,
    PairwisePreference,
)
from .features import (
    NUMERIC_INTERACTION_FEATURE_NAMES,
    SOURCE_INDICATOR_PREFIX,
    build_distributional_feature_surface,
    permute_interaction_features,
)
from .modeling import (
    fit_utility_aligned_models,
    nested_cardinality_transfer_evaluation,
)
from .policy import build_utility_aligned_policy
from .utility_surface import (
    build_pairwise_preferences,
    validate_exact_tail_utility_rows,
)
from .target_features import (
    TargetCandidateComponents,
    TargetFeatureProduction,
    build_target_feature_production,
    target_feature_production_from_payload,
)


__all__ = (
    "ABSTENTION_SEMANTICS",
    "BASE_ACTION_ID",
    "CARDINALITY_CLAIM_ROLE",
    "DEFAULT_CASE_BOOTSTRAP_SEED",
    "CardinalityTransferResult",
    "CaseBootstrapPlan",
    "CaseBootstrapReplicate",
    "CandidateFeatureRow",
    "CrossfitResult",
    "EXACT_TAIL_UTILITY_SEMANTICS",
    "ExactTailUtilityRow",
    "ExactTailUtilitySurface",
    "FEATURE_SEMANTICS",
    "FeatureSurface",
    "FoldAudit",
    "GLOBAL_ACTION_ID",
    "INNER_CANDIDATE_COUNT",
    "INNER_ROLE",
    "MAX_AUTHORIZED_NORMALIZED_ORACLE_GAP",
    "MIN_TARGET_SUPPORT_CASES",
    "MIN_SUPPORT_BOOTSTRAP_REPLICATES",
    "MODEL_SEMANTICS",
    "NUMERIC_INTERACTION_FEATURE_NAMES",
    "PERMUTATION_ACTION_ID",
    "PairwisePreference",
    "ROUTED_ACTION_ID",
    "RankingMetrics",
    "SEED_PAIR_COUNT",
    "SOURCE_INDICATOR_PREFIX",
    "SOURCE_INNER_TOP1_CHANCE",
    "TARGET_CANDIDATE_COUNT",
    "TARGET_ROLE",
    "TargetCandidateComponents",
    "TargetFeatureProduction",
    "TRAIN_CANDIDATE_COUNT_AFTER_STRICT_EXCLUSION",
    "UNIFORM_ACTION_ID",
    "UtilityAlignedModels",
    "UtilityAlignedPolicy",
    "build_distributional_feature_surface",
    "build_case_bootstrap_plan",
    "build_pairwise_preferences",
    "build_target_feature_production",
    "target_feature_production_from_payload",
    "build_utility_aligned_action",
    "build_utility_aligned_policy",
    "fit_utility_aligned_models",
    "nested_cardinality_transfer_evaluation",
    "permute_interaction_features",
    "validate_exact_tail_utility_rows",
)
