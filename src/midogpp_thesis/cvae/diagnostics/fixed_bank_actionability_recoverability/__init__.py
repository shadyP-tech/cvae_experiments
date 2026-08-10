"""Consumed-test fixed-bank actionability/recoverability diagnostic.

The public symbols below are pure scientific primitives.  Runtime, persistence,
label capabilities, bundle validation, and CLI registration live in separate
modules so importing this package cannot open data or start computation.
"""

from .actions import ActionSpec, actions_for_target, build_action_library
from .aggregation import (
    AggregatedProbabilityRow,
    ExactNineProbabilitySurface,
    SeedProbabilityRow,
    aggregate_exact_nine_probabilities,
)
from .constants import (
    B_ACTION_ID,
    CASE_ACTION_FEATURE_NAMES,
    GEOMETRY_IDS,
    MIDOGPP_CENTERS,
    RIDGE_ALPHA,
    U_ACTION_ID,
    candidate_sources,
    geometry_action_id,
)
from .contracts import (
    ActionScoreRow,
    BinaryLabelRow,
    BinaryPredictionRow,
    CaseActionFeatureRow,
    CaseConfusionCounts,
    MethodDecision,
    PooledBacc,
    RidgeActionModel,
    TerminalOracleResult,
    UtilityTargetRow,
)
from .decisions import build_pre_support_decisions, build_support_static_decisions
from .evaluation import (
    DiagnosticEvaluationResult,
    GeometryEvaluationResult,
    MethodEvaluationResult,
)
from .features import (
    build_label_free_case_action_features,
    matched_blocked_feature_permutation,
    restrict_feature_context,
)
from .metrics import (
    PairwiseComplementarity,
    RankStabilityResult,
    complementarity_metrics,
    normalized_oracle_gap,
    pooled_exact_bacc,
    rank_stability,
    score_case_confusions,
    terminal_oracles,
)
from .models import (
    fit_all_model_families,
    fit_fixed_alpha_ridge_models,
    predict_action_scores,
)
from .targets import build_class_balanced_proper_loss_targets


def __getattr__(name: str) -> object:
    """Keep config, runner, and workstation imports lazy at package import."""

    if name in {
        "FixedBankActionabilityRecoverabilityConfig",
        "load_fixed_bank_actionability_recoverability_config",
    }:
        from .config import (
            FixedBankActionabilityRecoverabilityConfig,
            load_fixed_bank_actionability_recoverability_config,
        )

        return {
            "FixedBankActionabilityRecoverabilityConfig": (
                FixedBankActionabilityRecoverabilityConfig
            ),
            "load_fixed_bank_actionability_recoverability_config": (
                load_fixed_bank_actionability_recoverability_config
            ),
        }[name]
    if name in {
        "FixedBankActionabilityRecoverabilityDependencies",
        "run_fixed_bank_actionability_recoverability",
    }:
        from .runner import (
            FixedBankActionabilityRecoverabilityDependencies,
            run_fixed_bank_actionability_recoverability,
        )

        return {
            "FixedBankActionabilityRecoverabilityDependencies": (
                FixedBankActionabilityRecoverabilityDependencies
            ),
            "run_fixed_bank_actionability_recoverability": (
                run_fixed_bank_actionability_recoverability
            ),
        }[name]
    raise AttributeError(name)


__all__ = (
    "ActionScoreRow",
    "ActionSpec",
    "AggregatedProbabilityRow",
    "B_ACTION_ID",
    "BinaryLabelRow",
    "BinaryPredictionRow",
    "CASE_ACTION_FEATURE_NAMES",
    "CaseActionFeatureRow",
    "CaseConfusionCounts",
    "DiagnosticEvaluationResult",
    "ExactNineProbabilitySurface",
    "FixedBankActionabilityRecoverabilityConfig",
    "FixedBankActionabilityRecoverabilityDependencies",
    "GEOMETRY_IDS",
    "GeometryEvaluationResult",
    "MIDOGPP_CENTERS",
    "MethodDecision",
    "MethodEvaluationResult",
    "PairwiseComplementarity",
    "PooledBacc",
    "RIDGE_ALPHA",
    "RankStabilityResult",
    "RidgeActionModel",
    "SeedProbabilityRow",
    "TerminalOracleResult",
    "U_ACTION_ID",
    "UtilityTargetRow",
    "actions_for_target",
    "aggregate_exact_nine_probabilities",
    "build_action_library",
    "build_label_free_case_action_features",
    "build_class_balanced_proper_loss_targets",
    "build_pre_support_decisions",
    "build_support_static_decisions",
    "candidate_sources",
    "complementarity_metrics",
    "fit_all_model_families",
    "fit_fixed_alpha_ridge_models",
    "geometry_action_id",
    "matched_blocked_feature_permutation",
    "normalized_oracle_gap",
    "pooled_exact_bacc",
    "predict_action_scores",
    "rank_stability",
    "restrict_feature_context",
    "score_case_confusions",
    "terminal_oracles",
    "load_fixed_bank_actionability_recoverability_config",
    "run_fixed_bank_actionability_recoverability",
)
