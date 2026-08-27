"""Stable public facade for the modular source-only row posterior."""

from .row_posterior_crossfit import crossfit_source_row_posterior, predict_source_row_posterior
from .row_posterior_features import ROW_POSTERIOR_MAX_FEATURES, assert_label_free_feature_names
from .row_posterior_fit import (
    ROW_POSTERIOR_MAX_ITERATIONS,
    ROW_POSTERIOR_PROBABILITY_FLOOR,
    ROW_POSTERIOR_RIDGE_ALPHA,
    fit_final_source_row_posterior,
    fit_source_row_posterior,
)


__all__ = (
    "ROW_POSTERIOR_MAX_FEATURES",
    "ROW_POSTERIOR_MAX_ITERATIONS",
    "ROW_POSTERIOR_PROBABILITY_FLOOR",
    "ROW_POSTERIOR_RIDGE_ALPHA",
    "assert_label_free_feature_names",
    "crossfit_source_row_posterior",
    "fit_final_source_row_posterior",
    "fit_source_row_posterior",
    "predict_source_row_posterior",
)
