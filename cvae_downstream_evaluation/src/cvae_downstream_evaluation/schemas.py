"""Shared constants for downstream evaluation contracts.

The constants in this module intentionally encode the locked v1 protocol.
They are used by config validation, report builders, and tests so stale
template choices cannot silently drift back into the experiment.
"""

from __future__ import annotations

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
POSTERIOR_DECODER_MEAN_GENERATION_MODE = "posterior_sample_decoder_mean"
POSTERIOR_DECODER_NOISE_GENERATION_MODE = "posterior_sample_decoder_noise"
C42_POSTERIOR_REPLAY_GENERATION_MODE = "posterior_sample_decoder_mean_replayed"
C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE = "standard_prior_decoder_mean_replayed"
C42_LATENT_GMM_K1_GENERATION_MODE = "latent_gmm_k1_decoder_mean"
C42_LATENT_GMM_K2_GENERATION_MODE = "latent_gmm_k2_decoder_mean"
C42_LATENT_GMM_K4_GENERATION_MODE = "latent_gmm_k4_decoder_mean"
GENERATION_MODES = (
    PRIMARY_GENERATION_MODE,
    NEGATIVE_CONTROL_GENERATION_MODE,
    POSTERIOR_DECODER_MEAN_GENERATION_MODE,
    POSTERIOR_DECODER_NOISE_GENERATION_MODE,
    C42_POSTERIOR_REPLAY_GENERATION_MODE,
    C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE,
    C42_LATENT_GMM_K1_GENERATION_MODE,
    C42_LATENT_GMM_K2_GENERATION_MODE,
    C42_LATENT_GMM_K4_GENERATION_MODE,
)

LEGACY_GENERATOR_FAMILY = "legacy_locked_v1"
PLAIN_CLASS_CONDITIONAL_GENERATOR_FAMILY = "family_c_pca64_class_conditional_cvae_downstream_v1"
HETEROSCEDASTIC_GENERATOR_FAMILY = "family_c_pca64_class_conditional_heteroscedastic_cvae_downstream_v1"
LATENT_GMM_PRIOR_GENERATOR_FAMILY = "family_c_pca64_class_conditional_latent_gmm_prior_cvae_downstream_v1"

C41_ORACLE_ELIGIBLE_GENERATION_MODES = (
    PRIMARY_GENERATION_MODE,
    POSTERIOR_DECODER_MEAN_GENERATION_MODE,
    POSTERIOR_DECODER_NOISE_GENERATION_MODE,
    C42_POSTERIOR_REPLAY_GENERATION_MODE,
    C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE,
    C42_LATENT_GMM_K1_GENERATION_MODE,
    C42_LATENT_GMM_K2_GENERATION_MODE,
    C42_LATENT_GMM_K4_GENERATION_MODE,
)

BASELINE_ROUTING_FAMILY_USED = "plain_pca64_class_conditional_cvae"
BASELINE_SELECTED_EXPERT_IDS_SOURCE = "locked_plain_class_conditional_support_nelbo"

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

MATRIX_SCHEMA_VERSION = "all_expert_downstream_matrix_v3"
ALL_EXPERT_DOWNSTREAM_PRIMARY_KEY = (
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "candidate_expert",
    "generator_family",
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
    "support_size",
    "support_seed",
    "candidate_expert",
    "generator_family",
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
    "utility_context_key",
    "utility_depends_on_support",
    "selection_depends_on_support",
    "plain_baseline_source",
    "plain_baseline_artifact_path",
    "plain_baseline_training_profile",
    "plain_baseline_matches_locked_hparams",
    "routing_family_used",
    "routing_scores_recomputed_for_heteroscedastic",
    "selected_expert_ids_source",
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
    "generator_family",
    "generation_mode",
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
    "selection_depends_on_support",
)

C41_DELTA_SUMMARY_COLUMNS = (
    "heldout_center",
    "support_size",
    "generation_mode",
    "oracle_bacc_plain",
    "oracle_bacc_hetero_mean",
    "oracle_bacc_hetero_noise",
    "selected_bacc_plain_router_plain_generator",
    "selected_bacc_plain_router_hetero_mean_generator",
    "selected_bacc_plain_router_hetero_noise_generator",
    "oracle_bacc_delta_vs_plain_retrained",
    "selected_bacc_delta_vs_plain_retrained",
    "oracle_gap_delta_vs_plain_retrained",
    "generated_std_delta_vs_plain",
    "selected_expert_changed_across_modes",
    "oracle_expert_changed_vs_plain",
    "decision_label",
)

C42_DELTA_SUMMARY_COLUMNS = (
    "heldout_center",
    "support_size",
    "generation_mode",
    "latent_gmm_components_requested",
    "latent_gmm_components_effective",
    "oracle_bacc_plain",
    "oracle_bacc_posterior_replay",
    "oracle_bacc_standard_prior_replay",
    "oracle_bacc_latent_gmm",
    "selected_bacc_locked_c41_router_plain_generator",
    "selected_bacc_locked_c41_router_posterior_replay_generator",
    "selected_bacc_locked_c41_router_standard_prior_replay_generator",
    "selected_bacc_locked_c41_router_latent_gmm_generator",
    "oracle_bacc_delta_vs_plain_retrained",
    "selected_bacc_delta_vs_plain_retrained",
    "oracle_gap_delta_vs_plain_retrained",
    "plain_replay_bacc_delta_vs_c41_stored",
    "plain_replay_matches_c41_within_tolerance",
    "oracle_expert_changed_vs_plain",
    "oracle_top1_stability_across_generation_seeds",
    "selected_expert_changed_across_modes",
    "decision_label",
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
