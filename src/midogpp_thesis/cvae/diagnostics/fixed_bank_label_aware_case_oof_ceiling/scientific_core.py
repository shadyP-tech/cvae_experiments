"""Stable facade for the terminal label-aware case-OOF scientific core."""

from .core_contracts import (
    AggregatedProbabilityRow,
    BinaryLabelRow,
    CaseActionUtility,
    CaseIdentityRow,
    CaseUtilitySurface,
    SealedProbabilitySurface,
    SeedProbabilityRow,
)
from .decisions import (
    DecisionConfig,
    DecisionSeal,
    FoldDecision,
    make_fold_decision,
    seal_fold_decisions,
)
from .evaluation import (
    ActionSelectionMetricRow,
    CaseEvaluationMetric,
    CeilingEvaluationResult,
    CenterEvaluationMetric,
    CenterSmoothMetric,
    PermutationNullSummaryRow,
    SmoothDescriptiveResult,
    evaluate_decision_seal,
    evaluate_smooth_descriptive,
)
from .global_prior import (
    CandidatePriorEstimate,
    LocoGlobalPrior,
    PriorConfig,
    fit_label_derived_loco_global_prior,
)
from .partitions import CaseFold, CaseOOFPartition, build_case_oof_partition
from .permutation_controls import (
    BlockedSupportPermutation,
    build_blocked_support_permutation,
    permute_fold_support_utilities,
)
from .permutation_plan import (
    PERMUTATION_DECISION_TIE_BREAK,
    PermutationDecisionPlan,
    build_permutation_decision_plan,
)
from .posterior import (
    CandidatePosteriorEstimate,
    FoldPosterior,
    PosteriorConfig,
    fit_fold_local_posterior,
)
from .probabilities import aggregate_exact_nine_probabilities
from .scientific_constants import (
    BASELINE_ACTION_ID,
    EXPECTED_ACTION_COUNT,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_FOLD_COUNT,
    EXPECTED_SEED_PAIR_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    MIDOGPP_CENTERS,
    action_ids,
    candidate_actions,
)
from .utilities import (
    binary_balanced_accuracy,
    replace_smooth_descriptive,
    score_evaluation_utilities_after_decision_seal,
    score_fold_support_utilities,
    score_loco_prior_utilities,
    soft_binary_balanced_accuracy,
)


__all__ = tuple(name for name in globals() if not name.startswith("_"))
