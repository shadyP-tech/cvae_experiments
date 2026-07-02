"""Schema constants for downstream classifier source-inner tuning artifacts."""

from __future__ import annotations

SOURCE_INNER_CLASSIFIER_TUNING_SCHEMA_VERSION = "source_inner_classifier_tuning_v1"

SOURCE_INNER_CLASSIFIER_TUNING_COLUMNS = (
    "schema_version",
    "experiment_seed",
    "classifier_seed",
    "outer_target_center",
    "selector_centers",
    "inner_lodo_centers",
    "classifier_grid_hash",
    "selected_classifier_spec",
    "selected_classifier_config_hash",
    "source_inner_lodo_center_bacc_vector",
    "source_inner_lodo_center_macro_f1_vector",
    "selection_metric",
    "selection_source",
    "selected_by_source_inner_lodo",
    "aggregate_selection_score",
    "tie_breaker",
    "convergence_by_center",
    "n_iter_by_center",
    "status_by_center",
    "selection_used_target_labels",
    "fit_used_target_center",
    "target_eval_labels_used_for_scoring",
)
