"""Schema constants, row roles, and gate criteria for the MIDOG++ gate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


SCHEMA_VERSION = "midogpp_real_feature_transfer_ceiling_v1"
SOURCE_INNER_RELIABILITY_SCHEMA_VERSION = "midogpp_source_inner_reliability_v1"
DATASET = "midogpp"
POSITIVE_LABEL = 1
POSITIVE_LABEL_NAME = "mitotic"
ELIGIBLE_CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
QUARANTINE_CENTERS = ("4",)


class RowRole(StrEnum):
    SOURCE_ONLY_TRANSFER = "source_only_transfer"
    SOURCE_INNER_WEIGHTED_ENSEMBLE = "source_inner_weighted_ensemble"
    UNIFORM_DENSE_ENSEMBLE = "uniform_dense_ensemble"
    POOLED_DIAGNOSTIC_CEILING = "pooled_diagnostic_ceiling"
    SOURCE_ORACLE_DIAGNOSTIC = "source_oracle_diagnostic"


class ClaimRole(StrEnum):
    TRANSFER_BASELINE = "real_feature_transfer_baseline"
    SOURCE_INNER_ENSEMBLE = "real_feature_source_inner_ensemble"
    DIAGNOSTIC_CEILING = "diagnostic_ceiling"
    DIAGNOSTIC_ORACLE = "diagnostic_oracle"
    QUARANTINE_ONLY = "quarantine_only"


REQUIRED_MATRIX_COLUMNS = (
    "schema_version",
    "dataset",
    "domain_regime",
    "fold_unit",
    "heldout_center",
    "heldout_tumor_domain",
    "source_scope",
    "fit_domains",
    "eval_domain",
    "method",
    "row_role",
    "claim_role",
    "adoption_eligible",
    "diagnostic_only",
    "selection_source",
    "fit_used_target_center",
    "selection_used_target_labels",
    "target_eval_labels_used_for_scoring_only",
    "support_labels_used",
    "threshold_policy",
    "calibration_policy",
    "model_seed",
    "status",
    "invalid_reason",
    "manifest_hash",
    "cache_hash",
    "config_hash",
    "protocol_hash",
    "prediction_hash",
    "n_eval",
    "n_eval_pos",
    "n_eval_neg",
    "target_prevalence",
    "predicted_positive_rate",
    "tp",
    "fp",
    "tn",
    "fn",
    "sensitivity",
    "specificity",
    "precision",
    "macro_f1",
    "balanced_accuracy",
    "auroc",
    "pr_auc",
    "pr_auc_baseline",
)

REQUIRED_RELIABILITY_RESULT_COLUMNS = (
    "schema_version",
    "dataset",
    "domain_regime",
    "fold_unit",
    "heldout_center",
    "source_scope",
    "method",
    "row_role",
    "claim_role",
    "adoption_eligible",
    "diagnostic_only",
    "selection_source",
    "utility_family",
    "fit_used_target_center",
    "selection_used_target_labels",
    "target_eval_labels_used_for_scoring_only",
    "support_labels_used",
    "target_expert_excluded",
    "candidate_experts",
    "n_candidate_experts",
    "weight_policy",
    "fallback_reason",
    "tau",
    "cap_min",
    "cap_max",
    "shrinkage",
    "model_seed",
    "status",
    "invalid_reason",
    "manifest_hash",
    "cache_hash",
    "config_hash",
    "protocol_hash",
    "prediction_hash",
    "n_eval",
    "n_eval_pos",
    "n_eval_neg",
    "target_prevalence",
    "predicted_positive_rate",
    "tp",
    "fp",
    "tn",
    "fn",
    "sensitivity",
    "specificity",
    "precision",
    "macro_f1",
    "balanced_accuracy",
    "auroc",
    "pr_auc",
    "pr_auc_baseline",
)

REQUIRED_RELIABILITY_SCORE_COLUMNS = (
    "schema_version",
    "heldout_center",
    "pseudo_target_center",
    "expert_center",
    "eligible",
    "utility_family",
    "utility_value",
    "z_iq",
    "fold_mean_utility",
    "fold_std_utility",
    "fallback_reason",
    "fit_used_pseudo_target_center",
    "selection_used_target_labels",
)

REQUIRED_ENSEMBLE_WEIGHT_COLUMNS = (
    "schema_version",
    "heldout_center",
    "method",
    "row_role",
    "expert_center",
    "eligible",
    "utility_family",
    "n_eligible_pseudo_target_folds",
    "s_i",
    "weight_raw",
    "w_i_utility",
    "w_i_preservation",
    "fallback_reason",
    "tau",
    "cap_min",
    "cap_max",
    "shrinkage",
    "selection_source",
    "target_expert_excluded",
    "selection_used_target_labels",
    "fit_used_target_center",
)


@dataclass(frozen=True)
class GateCriteria:
    min_valid_eligible_fold_fraction: float = 0.70
    min_source_only_bacc: float = 0.60
    min_source_only_auroc: float = 0.60
    worst_center_collapse_bacc: float = 0.50
    min_headroom_delta: float = 0.05
    min_headroom_centers: int = 2
    negative_control_bacc_low: float = 0.45
    negative_control_bacc_high: float = 0.55


DEFAULT_GATE_CRITERIA = GateCriteria()
