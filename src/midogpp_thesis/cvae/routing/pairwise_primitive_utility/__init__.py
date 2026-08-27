"""Stage-neutral center-jackknife analytic pairwise routing primitives."""

from .admission import evaluate_source_only_admission, seal_admission_decision
from .clustered_uncertainty import (
    ONE_SIDED_ALPHA,
    apply_calibrated_bound,
    calibrate_clustered_uncertainty,
)
from .contracts import *  # noqa: F403
from .contracts import __all__ as _contract_exports
from .opportunity import build_opportunity_set
from .pairwise_ranker import (
    PAIRWISE_ALPHA_GRID,
    fit_pairwise_ranker,
    predict_action_score,
    predict_pairwise_contrast,
    rank_action_queries,
)
from .primitive_utility import (
    build_expected_denominators,
    expected_additive_utility,
    normalize_expected_utility,
    sum_primitives,
)
from .row_posterior import (
    assert_label_free_feature_names,
    crossfit_source_row_posterior,
    fit_final_source_row_posterior,
    fit_source_row_posterior,
    predict_source_row_posterior,
)
from .selection import assemble_action_selection_evidence, select_fail_closed_action


__all__ = (
    *_contract_exports,
    "ONE_SIDED_ALPHA",
    "PAIRWISE_ALPHA_GRID",
    "apply_calibrated_bound",
    "assemble_action_selection_evidence",
    "assert_label_free_feature_names",
    "build_expected_denominators",
    "build_opportunity_set",
    "calibrate_clustered_uncertainty",
    "crossfit_source_row_posterior",
    "evaluate_source_only_admission",
    "expected_additive_utility",
    "fit_final_source_row_posterior",
    "fit_pairwise_ranker",
    "fit_source_row_posterior",
    "normalize_expected_utility",
    "predict_action_score",
    "predict_pairwise_contrast",
    "predict_source_row_posterior",
    "rank_action_queries",
    "select_fail_closed_action",
    "seal_admission_decision",
    "sum_primitives",
)
