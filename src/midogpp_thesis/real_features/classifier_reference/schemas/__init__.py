"""Shared constants and row contracts for downstream evaluation.

The constants in this module intentionally encode the locked v1 protocol.
They are used by config validation, report builders, and tests so stale
template choices cannot silently drift back into the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

EXPERIMENT_NAME = "direct_support_nelbo_selected_synthetic_downstream_v1"
DATASET_NAME = "camelyon17"
DOMAIN_KEY = "center"
CAMELYON17_CENTERS = ("0", "1", "2", "3", "4")

EXPERIMENT_SEEDS = (42, 43, 44)
SUPPORT_SEEDS = (17, 23, 31)
SUPPORT_SIZES = (4, 8, 16, 32)
GENERATION_SEEDS = (17, 23, 31)
CLASSIFIER_SEEDS = (17, 23, 31)

PRIMARY_BUDGET_PER_CLASS = 128
DIAGNOSTIC_BUDGETS_PER_CLASS = (64, 128, 256, 512)

PRIMARY_GENERATION_MODE = "class_stratified_reference_posterior_resampling"
NEGATIVE_CONTROL_GENERATION_MODE = "unconditional_prior_sampling_assigned_label_negative_control"
GENERATION_MODES = (
    PRIMARY_GENERATION_MODE,
    NEGATIVE_CONTROL_GENERATION_MODE,
)

SUPPORT_NELBO_METHOD = "support_set_nelbo_top1"
METADATA_METHOD = "support_metadata_routing"
SOURCE_GLOBAL_METHOD = "source_global_prior_routing"
RANDOM_METHOD = "random_candidate_expert"
ENSEMBLE_METHOD = "naive_all_expert_ensemble"
DOWNSTREAM_ORACLE_METHOD = "single_expert_downstream_oracle_diagnostic_only"
SOURCE_GLOBAL_GATED_METHOD_PREFIX = "source_global_gated_support_nelbo"

SINGLE_EXPERT_METHODS = (
    SUPPORT_NELBO_METHOD,
    METADATA_METHOD,
    SOURCE_GLOBAL_METHOD,
    RANDOM_METHOD,
)

ADOPTION_ELIGIBLE_METHODS = (
    SUPPORT_NELBO_METHOD,
    METADATA_METHOD,
    SOURCE_GLOBAL_METHOD,
    RANDOM_METHOD,
    ENSEMBLE_METHOD,
)

METHODS_WITH_FULL_RANKING = (SUPPORT_NELBO_METHOD,)

BASELINE_METHODS = (
    METADATA_METHOD,
    RANDOM_METHOD,
    SOURCE_GLOBAL_METHOD,
    ENSEMBLE_METHOD,
)

SINGLE_EXPERT_ROW_TYPE = "single_expert"
METHOD_BASELINE_ROW_TYPE = "method_baseline"
ENSEMBLE_EXPERT_ID = "__ensemble__"
SINGLE_EXPERT_HASH = "__single_expert__"

MATRIX_SCHEMA_VERSION = "all_expert_downstream_matrix_v1"
ALL_EXPERT_DOWNSTREAM_PRIMARY_KEY = (
    "experiment_seed",
    "heldout_center",
    "candidate_expert",
    "generation_mode",
    "budget_per_class",
    "generation_seed",
    "classifier_seed",
    "row_type",
    "candidate_experts_hash",
)

REQUIRED_ALIGNMENT_METRICS = (
    "top1_downstream_hit",
    "spearman_neg_nelbo_vs_bacc",
    "downstream_oracle_gap_bacc",
    "downstream_oracle_gap_macro_f1",
)

PRIMARY_DOWNSTREAM_METRICS = (
    "bacc",
    "macro_f1",
)

SECONDARY_DOWNSTREAM_METRICS = (
    "auroc",
    "auprc",
    "expected_calibration_error",
)

FIDELITY_DIAGNOSTIC_METRICS = (
    "rbf_mmd",
    "energy_distance",
    "frechet_embedding_distance",
    "mean_distance",
    "covariance_distance",
    "knn_precision",
    "knn_recall",
    "density",
    "coverage",
)

FORBIDDEN_ROUTER_INPUTS = (
    "target_evaluation_labels",
    "target_evaluation_nelbo",
    "downstream_oracle_expert",
    "target_test_metrics",
    "generation_hyperparameters_tuned_on_target_eval",
    "classifier_hyperparameters_tuned_on_target_eval",
)

ESSENTIAL_BASELINES = (
    METADATA_METHOD,
    SUPPORT_NELBO_METHOD,
    RANDOM_METHOD,
    SOURCE_GLOBAL_METHOD,
    ENSEMBLE_METHOD,
    DOWNSTREAM_ORACLE_METHOD,
)

ALL_EXPERT_DOWNSTREAM_COLUMNS = (
    "schema_version",
    "experiment_seed",
    "heldout_center",
    "candidate_expert",
    "generation_mode",
    "budget_per_class",
    "generation_seed",
    "classifier_seed",
    "bacc",
    "macro_f1",
    "auroc",
    "auprc",
    "row_type",
    "n_synthetic_train",
    "n_target_eval",
    "target_eval_pool_id",
    "candidate_experts_hash",
    "status",
    "error_message",
)

SUPPORT_SELECTION_COLUMNS = (
    "heldout_center",
    "experiment_seed",
    "support_size",
    "support_seed",
    "method",
    "selected_expert",
    "candidate_experts",
    "support_nelbo_by_expert_json",
    "target_expert_excluded",
    "support_eval_split_id",
)

ROUTING_ALIGNMENT_COLUMNS = (
    "heldout_center",
    "experiment_seed",
    "support_size",
    "support_seed",
    "generation_seed",
    "classifier_seed",
    "method",
    "selected_expert",
    "selected_bacc",
    "selected_macro_f1",
    "downstream_oracle_expert",
    "oracle_bacc",
    "oracle_macro_f1",
    "downstream_oracle_gap_bacc",
    "downstream_oracle_gap_macro_f1",
    "relative_downstream_oracle_gap_pct",
    "top1_downstream_hit",
    "spearman_neg_nelbo_vs_bacc",
    "metadata_bacc",
    "delta_vs_metadata",
)

SUPPORT_SIZE_SUMMARY_COLUMNS = (
    "support_size",
    "method",
    "mean_bacc",
    "mean_macro_f1",
    "mean_delta_bacc_vs_metadata",
    "mean_downstream_oracle_gap_bacc",
    "mean_spearman_neg_nelbo_vs_bacc",
    "center_pass_count",
)

BASELINE_COMPARISON_COLUMNS = (
    "method",
    "row_type",
    "mean_bacc",
    "mean_macro_f1",
    "mean_delta_bacc_vs_metadata",
    "mean_downstream_oracle_gap_bacc",
    "top1_downstream_hit_rate",
)

STABILITY_COLUMNS = (
    "method",
    "group",
    "mean_bacc",
    "std_bacc",
    "worst_center_bacc",
)

DECISION_CLASSIFICATIONS = (
    "PASS",
    "WEAK_PASS",
    "DIAGNOSTIC_ONLY",
    "FAIL",
)

SELECTION_ELIGIBLE = "selection_eligible"
DIAGNOSTIC_ONLY = "diagnostic_only"
ELIGIBILITY_FLAGS = (SELECTION_ELIGIBLE, DIAGNOSTIC_ONLY)

DIRECT_TARGET_IDENTITY_COLUMNS = (
    "target_domain",
    "target_domain_id",
    "heldout_target_identity",
    "direct_fold_identity",
    "fold_oracle_label",
    "oracle_label",
    "oracle_rank",
    "excluded_expert_identity",
    "downstream_oracle_expert",
)

FORBIDDEN_DEPLOYABLE_FEATURE_COLUMNS = DIRECT_TARGET_IDENTITY_COLUMNS + (
    "heldout_center",
    "candidate_expert",
    "bacc",
    "macro_f1",
    "target_eval_bacc",
    "target_eval_macro_f1",
    "target_eval_fidelity",
    "target_eval_embeddings",
    "target_eval_labels",
    "diagnostic_downstream_utility",
    "downstream_utility_matrix_path",
)

REQUIRED_LINEAGE_COLUMNS = (
    "fold_id",
    "experiment_seed",
    "target_domain",
    "support_split_id",
    "eval_split_id",
    "candidate_id",
    "expert_checkpoint_id",
    "expert_checkpoint_hash",
    "generation_mode",
    "generation_seed",
    "classifier_seed",
    "config_hash",
    "protocol_hash",
    "eligibility",
)

ADOPTION_ELIGIBLE_SELECTION_COLUMNS = REQUIRED_LINEAGE_COLUMNS + (
    "method",
    "predicted_primary_utility",
    "support_nelbo",
    "source_inner_stability",
    "selection_rank",
    "aggregation_weight",
)

LEARNED_UTILITY_ALIGNMENT_COLUMNS = (
    "fold_id",
    "experiment_seed",
    "target_domain",
    "support_split_id",
    "eval_split_id",
    "method",
    "candidate_id",
    "expert_checkpoint_id",
    "generation_mode",
    "generation_seed",
    "classifier_seed",
    "predicted_primary_utility",
    "selected_bacc",
    "selected_macro_f1",
    "downstream_oracle_candidate_id",
    "oracle_bacc",
    "oracle_macro_f1",
    "downstream_oracle_gap_bacc",
    "downstream_oracle_gap_macro_f1",
    "top1_downstream_oracle_hit",
    "eligibility",
)


@dataclass(frozen=True)
class CandidateManifestRow:
    """Protocol identity for one atomic downstream candidate."""

    candidate_id: str
    expert_checkpoint_id: str
    source_domain: str
    checkpoint_seed: int
    generation_mode: str
    latent_sampling_setting: str
    class_prior_rule: str
    synthetic_budget: int
    generation_seed: int
    aggregation_recipe: str
    classifier_seed: int
    expert_checkpoint_hash: str
    config_hash: str
    protocol_hash: str
    eligibility: str = SELECTION_ELIGIBLE

    def __post_init__(self) -> None:
        if self.eligibility not in ELIGIBILITY_FLAGS:
            raise ValueError(f"Unknown candidate eligibility: {self.eligibility!r}")

    def to_row(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "expert_checkpoint_id": self.expert_checkpoint_id,
            "source_domain": self.source_domain,
            "checkpoint_seed": self.checkpoint_seed,
            "generation_mode": self.generation_mode,
            "latent_sampling_setting": self.latent_sampling_setting,
            "class_prior_rule": self.class_prior_rule,
            "synthetic_budget": self.synthetic_budget,
            "generation_seed": self.generation_seed,
            "aggregation_recipe": self.aggregation_recipe,
            "classifier_seed": self.classifier_seed,
            "expert_checkpoint_hash": self.expert_checkpoint_hash,
            "config_hash": self.config_hash,
            "protocol_hash": self.protocol_hash,
            "eligibility": self.eligibility,
        }


@dataclass(frozen=True)
class ArtifactLineageKey:
    """Minimum join key required for deployable feature and selection rows."""

    fold_id: str
    experiment_seed: int
    target_domain: str
    support_split_id: str
    eval_split_id: str
    candidate_id: str
    expert_checkpoint_id: str
    expert_checkpoint_hash: str
    generation_mode: str
    generation_seed: int
    classifier_seed: int
    config_hash: str
    protocol_hash: str
    eligibility: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "ArtifactLineageKey":
        missing = [key for key in REQUIRED_LINEAGE_COLUMNS if key not in row]
        if missing:
            raise ValueError(f"Missing lineage columns: {missing}")
        return cls(
            fold_id=str(row["fold_id"]),
            experiment_seed=int(row["experiment_seed"]),
            target_domain=str(row["target_domain"]),
            support_split_id=str(row["support_split_id"]),
            eval_split_id=str(row["eval_split_id"]),
            candidate_id=str(row["candidate_id"]),
            expert_checkpoint_id=str(row["expert_checkpoint_id"]),
            expert_checkpoint_hash=str(row["expert_checkpoint_hash"]),
            generation_mode=str(row["generation_mode"]),
            generation_seed=int(row["generation_seed"]),
            classifier_seed=int(row["classifier_seed"]),
            config_hash=str(row["config_hash"]),
            protocol_hash=str(row["protocol_hash"]),
            eligibility=str(row["eligibility"]),
        )
