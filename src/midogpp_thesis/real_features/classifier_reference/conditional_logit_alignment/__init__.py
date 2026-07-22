"""Stage-10 conditional-logit alignment diagnostic."""

from .config import (
    AlignmentOptimizerConfig,
    ConditionalLogitAlignmentConfig,
    EXPECTED_VERSIONS,
    GAMMA_GRID,
    canonical_classifier_spec,
    load_conditional_logit_alignment_config,
)
from .estimator import (
    AlignmentFitResult,
    ConditionalObjectiveTerms,
    PreparedConditionalLogit,
    conditional_logit_objective_and_gradient,
    conditional_logit_objective_terms,
    fit_conditional_logit,
    fit_prepared_conditional_logit,
    prepare_conditional_logit,
)
from .folds import (
    ConditionalLogitFold,
    FoldRowIdentity,
    OuterLodoFold,
    SourceInnerLodoFold,
    make_inner_fold,
    make_outer_fold,
)
from .penalty import ConditionalPenaltyOperator, build_conditional_penalty
from .selection import (
    GammaFoldScore,
    GammaSelection,
    GammaSummary,
    OuterEvaluationPlan,
    plan_outer_evaluation,
    select_gamma_source_inner,
    summarize_gamma_scores,
)


def run_conditional_logit_alignment(*args: object, **kwargs: object) -> object:
    """Lazy public runner facade, keeping numerical imports reusable in tests."""

    from .runner import run_conditional_logit_alignment as _run

    return _run(*args, **kwargs)


def assert_conditional_logit_alignment_artifacts(
    *args: object, **kwargs: object
) -> object:
    """Lazy public validation facade."""

    from .validation import assert_conditional_logit_alignment_artifacts as _assert

    return _assert(*args, **kwargs)


__all__ = [
    "AlignmentFitResult",
    "AlignmentOptimizerConfig",
    "ConditionalLogitAlignmentConfig",
    "ConditionalLogitFold",
    "ConditionalObjectiveTerms",
    "ConditionalPenaltyOperator",
    "EXPECTED_VERSIONS",
    "FoldRowIdentity",
    "GAMMA_GRID",
    "GammaFoldScore",
    "GammaSelection",
    "GammaSummary",
    "OuterLodoFold",
    "OuterEvaluationPlan",
    "PreparedConditionalLogit",
    "SourceInnerLodoFold",
    "build_conditional_penalty",
    "assert_conditional_logit_alignment_artifacts",
    "canonical_classifier_spec",
    "conditional_logit_objective_and_gradient",
    "conditional_logit_objective_terms",
    "fit_conditional_logit",
    "fit_prepared_conditional_logit",
    "load_conditional_logit_alignment_config",
    "make_inner_fold",
    "make_outer_fold",
    "prepare_conditional_logit",
    "plan_outer_evaluation",
    "select_gamma_source_inner",
    "run_conditional_logit_alignment",
    "summarize_gamma_scores",
]
