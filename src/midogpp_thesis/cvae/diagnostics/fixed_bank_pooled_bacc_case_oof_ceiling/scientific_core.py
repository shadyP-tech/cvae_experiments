"""Stable facade for the terminal pooled exact-BACC v2 scientific core."""

from .case_partitions import CaseFold, CaseOOFPartition, build_case_oof_partition
from .core_contracts import (
    AggregatedProbabilityRow,
    BinaryLabelRow,
    CaseActionSufficientStatistics,
    CaseIdentityRow,
    SealedProbabilitySurface,
    SeedProbabilityRow,
    SufficientStatisticSurface,
    make_statistics_surface,
)
from .decisions import (
    DecisionConfig,
    DecisionSeal,
    FoldDecision,
    make_fold_decision,
    seal_fold_decisions,
)
from .permutation_controls import (
    BlockedSupportPermutation,
    build_blocked_support_permutation,
    permute_fold_support_statistics,
    permute_fold_support_utilities,
)
from .permutation_plan import (
    NULL_DERANGEMENT_ALGORITHM,
    PERMUTATION_DECISION_TIE_BREAK,
    PermutationDecisionPlan,
    build_permutation_decision_plan,
)
from .pooled_evaluation import (
    ActionSelectionMetricRow,
    CeilingEvaluationResult,
    CenterEvaluationMetric,
    EqualCenterInferenceRow,
    FoldEvaluationMetric,
    PermutationNullSummaryRow,
    PooledCeilingEvaluationResult,
    evaluate_decision_seal,
    evaluate_statistics_seal,
)
from .pooled_metrics import (
    PairedClusterContrast,
    PooledExactBacc,
    action_rows,
    binary_balanced_accuracy,
    paired_pooled_difference,
    paired_whole_case_cluster_contrast,
    pooled_exact_bacc,
    score_evaluation_statistics_after_decision_seal,
    score_evaluation_statistics_after_preevaluation_seals,
    score_fold_support_statistics,
    score_loco_prior_statistics,
)
from .pooled_posterior import (
    CandidatePosteriorEstimate,
    FoldPosterior,
    PooledFoldPosterior,
    PosteriorConfig,
    fit_fold_local_posterior,
    fit_pooled_fold_posterior,
)
from .pooled_prior import (
    CandidateGlobalEstimate,
    PairwisePriorEstimate,
    PooledLocoPrior,
    PriorConfig,
    fit_label_derived_loco_global_prior,
    fit_pooled_loco_prior,
)
from .probability_surface import aggregate_exact_nine_probabilities
from .scientific_constants import (
    BASELINE_ACTION_ID,
    DEFAULT_CONFIDENCE_MULTIPLIER,
    DEFAULT_MINIMUM_GAIN,
    DEFAULT_PERMUTATION_COUNT,
    DEFAULT_TIE_TOLERANCE,
    DEFAULT_VARIANCE_FLOOR,
    EXPECTED_ACTION_COUNT,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_FOLD_COUNT,
    EXPECTED_FOLD_DECISION_COUNT,
    EXPECTED_SEED_PAIR_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    MIDOGPP_CENTERS,
    TERMINAL_DECISION,
    UNCERTAINTY_UNIT,
    UTILITY_ID,
    action_ids,
    candidate_actions,
    legal_donor_centers,
    routing_challengers,
)
from .scientific_payloads import (
    decision_seal_from_payload,
    evaluation_result_from_payload,
    fold_decision_from_payload,
    partition_from_payload,
    permutation_plan_from_payload,
    pooled_fold_posterior_from_payload,
    pooled_loco_prior_from_payload,
    probability_surface_from_payload,
    statistics_surface_from_payload,
)


__all__ = tuple(name for name in globals() if not name.startswith("_"))
