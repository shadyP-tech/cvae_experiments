"""Closed-world file inventory and table schemas for fresh Stage 70."""

REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/policy_binding.json",
    "manifests/evaluation_plan.json",
    "manifests/prediction_seal.json",
    "manifests/content_index.json",
    "checkpoints/source/source_cache.json",
    "checkpoints/predictions/prediction_cache.json",
    "tables/prediction_index.csv",
    "tables/seed_cell_metrics.csv",
    "tables/ensemble_metrics.csv",
    "tables/center_contrasts.csv",
    "tables/contrast_inference.csv",
    "tables/oracle_diagnostics.csv",
    "reports/label_access_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/run_state.json",
    "reports/validation_report.json",
)
CONTENT_INDEX_EXCLUSIONS = frozenset(
    {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
)
STATIC_CONTENT_INDEX_MEMBERS = tuple(
    member for member in REQUIRED_FILES if member not in CONTENT_INDEX_EXCLUSIONS
)

SEED_METRIC_COLUMNS = (
    "target_center",
    "training_seed",
    "generation_seed",
    "action_id",
    "bacc",
    "macro_f1",
    "evaluation_row_count",
    "prediction_seal_hash",
    "endpoint_role",
    "descriptive_only",
)
ENSEMBLE_METRIC_COLUMNS = (
    "target_center",
    "action_id",
    "bacc",
    "macro_f1",
    "evaluation_row_count",
    "seed_cell_count",
    "prediction_seal_hash",
    "endpoint",
    "primary_endpoint",
    "probability_aggregation",
)
CENTER_CONTRAST_COLUMNS = (
    "contrast_id",
    "target_center",
    "left_action_id",
    "right_action_id",
    "probability_ensemble_bacc_delta",
    "descriptive_seed_cell_mean_bacc_delta",
    "contrast_role",
    "primary_endpoint",
    "inference_unit",
)
INFERENCE_COLUMNS = (
    "contrast_id",
    "mean_probability_ensemble_bacc_delta",
    "two_sided_95_ci_low",
    "two_sided_95_ci_high",
    "one_sided_95_lcb",
    "wins",
    "ties",
    "losses",
    "center_count",
    "contrast_role",
    "primary_endpoint",
    "inference_unit",
)
ORACLE_COLUMNS = (
    "target_center",
    "source_count",
    "support_score_utility_spearman",
    "spearman_defined",
    "top1_agreement",
    "oracle_headroom_bacc",
    "normalized_oracle_gap",
    "oracle_utility_range_bacc",
    "prediction_seal_hash",
    "diagnostic_only",
    "may_update_frozen_policy",
)


__all__ = (
    "CENTER_CONTRAST_COLUMNS",
    "CONTENT_INDEX_EXCLUSIONS",
    "ENSEMBLE_METRIC_COLUMNS",
    "INFERENCE_COLUMNS",
    "ORACLE_COLUMNS",
    "REQUIRED_FILES",
    "SEED_METRIC_COLUMNS",
    "STATIC_CONTENT_INDEX_MEMBERS",
)
