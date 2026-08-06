from __future__ import annotations

from pathlib import Path

import yaml

from midogpp_thesis.cvae.diagnostics.dense_residual_router.bundle import (
    REQUIRED_FILES as BUNDLE_REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.dense_residual_router.config import (
    load_dense_residual_diagnostic_config,
)
from midogpp_thesis.workspace.runtime import ArtifactEntry, MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_validation_dense_residual_router_v1.yaml"
)
EXPERIMENT_ID = (
    "midogpp.oracle."
    "uniform_b_v2_consumed_validation_dense_residual_router.v1"
)
OUTPUT_ID = (
    "midogpp_output_uniform_b_v2_consumed_validation_dense_residual_router_v1"
)
EXPERT_BANK_ID = "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
GENERATION_LOCK_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
CACHE_ALIAS_ID = "midogpp_stage90_dense_residual_router_validation_cache_v1"
MANIFEST_ALIAS_ID = "midogpp_stage90_dense_residual_router_validation_manifest_v1"
ORIGINAL_CACHE_ID = "midogpp_virchow2_uniform_b_v2_routing_validation_cache_seed42"
ORIGINAL_MANIFEST_ID = "midogpp_source_inner_validation_manifest_v1"

NON_DIAGNOSTIC_REUSE = {
    "real_feature_reference_evidence",
    "cvae_preservation_evidence",
    "expert_bank_evidence",
    "generation_evidence",
    "all_candidate_utility_diagnostic",
    "synthetic_downstream_utility_evidence",
    "routing_evidence",
    "expert_selection_evidence",
    "nelbo_compatibility_evidence",
}
REQUIRED_OUTPUT_FILES = {
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/support_partition_lock.json",
    "manifests/compatibility_index.json",
    "manifests/development_prediction_seals.json",
    "manifests/all_action_target_prediction_seal.json",
    "manifests/diagnostic_decision_seals.json",
    "manifests/target_prediction_seals.json",
    "manifests/content_index.json",
    "arrays/development_predictions.npz",
    "arrays/target_predictions.npz",
    "tables/support_partitions.csv",
    "tables/compatibility_case_energy.csv",
    "tables/compatibility_scores.csv",
    "tables/development_prediction_index.csv",
    "tables/development_metrics.csv",
    "tables/action_summaries.csv",
    "tables/diagnostic_selections.csv",
    "tables/target_weight_plans.csv",
    "tables/target_assignments.csv",
    "tables/target_prediction_index.csv",
    "tables/target_metrics.csv",
    "tables/paired_deltas.csv",
    "reports/phase_01_support_and_compatibility_complete.json",
    "reports/phase_02_development_complete.json",
    "reports/phase_03_target_predictions_complete.json",
    "reports/phase_04_scoring_complete.json",
    "reports/label_access_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/run_state.json",
    "reports/validation_report.json",
}


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    return workspace


def _hashes(artifact: ArtifactEntry) -> dict[str, tuple[str, str]]:
    return {
        relative: (expectation.algorithm, expectation.digest)
        for relative, expectation in artifact.expected_file_hashes.items()
    }


def test_dense_residual_router_is_registered_as_stage90_diagnostic_only() -> None:
    workspace = _workspace()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == (
        EXPERT_BANK_ID,
        GENERATION_LOCK_ID,
        CACHE_ALIAS_ID,
        MANIFEST_ALIAS_ID,
    )
    assert set(experiment.input_claim_scope_exceptions) == {GENERATION_LOCK_ID}
    assert experiment.runner_argv == (
        "{python}",
        "-m",
        "midogpp_thesis",
        "cvae-diagnostics",
        "dense-residual-router-diagnostic",
        "--config",
        "{resolved_config}",
        "--artifact-root",
        f"output://{OUTPUT_ID}",
    )
    assert experiment.runner_env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    assert output.stage == "90_oracles_and_diagnostics"
    assert output.canonical_path == (
        "artifacts/midogpp/90_oracles_and_diagnostics/"
        "uniform_b_v2_consumed_validation_dense_residual_router/v1"
    )
    assert output.evidence_label == "EXPLORATORY_CONSUMED_DATA_ONLY"
    assert output.claim_scope == "diagnostic_only"
    assert output.required_files == BUNDLE_REQUIRED_FILES
    assert set(output.required_files) == REQUIRED_OUTPUT_FILES
    assert set(output.forbidden_reuse) == NON_DIAGNOSTIC_REUSE
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert output.semantic_identities["fresh_confirmation"] == "false"
    assert output.semantic_identities["routing_quality_claimed"] == "false"
    assert output.semantic_identities["may_feed_stage60"] == "false"
    assert output.semantic_identities["may_feed_stage70"] == "false"


def test_consumed_validation_aliases_copy_exact_bytes_and_fence_consumers() -> None:
    workspace = _workspace()
    pairs = (
        (CACHE_ALIAS_ID, ORIGINAL_CACHE_ID),
        (MANIFEST_ALIAS_ID, ORIGINAL_MANIFEST_ID),
    )

    for alias_id, original_id in pairs:
        alias = workspace.artifacts[alias_id]
        original = workspace.artifacts[original_id]
        assert alias.physical_path == original.physical_path
        assert alias.canonical_path == original.canonical_path
        assert alias.required_files == original.required_files
        assert _hashes(alias) == _hashes(original)
        assert alias.semantic_identities["alias_of_artifact_id"] == original_id
        assert alias.semantic_identities["consumption_status"] == (
            "CONSUMED_FOR_STAGE90_DIAGNOSTIC_ROUTER_PROTOTYPING"
        )
        assert alias.semantic_identities["fresh_evidence"] == "false"
        assert alias.semantic_identities["authorized_consumer_experiment_ids"] == (
            EXPERIMENT_ID
        )
        assert set(alias.forbidden_reuse) == NON_DIAGNOSTIC_REUSE
        assert "oracle_and_diagnostic_evidence" not in alias.forbidden_reuse
        assert alias.may_feed_recipe_selection is False
        assert alias.may_feed_deployable_selection is False

        consumers = {
            experiment.experiment_id
            for experiment in workspace.experiments.values()
            if alias_id in experiment.input_artifact_ids
        }
        assert consumers == {EXPERIMENT_ID}

    original_cache = workspace.artifacts[ORIGINAL_CACHE_ID]
    original_manifest = workspace.artifacts[ORIGINAL_MANIFEST_ID]
    assert "oracle_and_diagnostic_evidence" in original_cache.forbidden_reuse
    assert "oracle_and_diagnostic_evidence" in original_manifest.forbidden_reuse
    assert original_cache.may_feed_deployable_selection is True
    assert original_manifest.may_feed_deployable_selection is True


def test_dense_residual_config_freezes_scientific_and_publication_contracts() -> None:
    config = load_dense_residual_diagnostic_config(CONFIG_PATH)
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config.input_artifact_ids == (
        EXPERT_BANK_ID,
        GENERATION_LOCK_ID,
        CACHE_ALIAS_ID,
        MANIFEST_ALIAS_ID,
    )
    assert payload["inputs"]["validation_cache_root"] == f"artifact://{CACHE_ALIAS_ID}"
    assert payload["inputs"]["validation_manifest_path"] == (
        f"artifact://{MANIFEST_ALIAS_ID}/manifest.csv"
    )
    assert ORIGINAL_CACHE_ID not in str(payload["inputs"])
    assert ORIGINAL_MANIFEST_ID not in str(payload["inputs"])

    protocol = config.protocol
    assert protocol["support_split_seed"] == 20260806
    assert protocol["support_case_count_per_center"] == 2
    assert protocol["training_seeds"] == [17, 42, 101]
    assert protocol["generation_seeds"] == [17, 42, 101]
    assert protocol["support_evaluation_case_disjoint"] is True
    assert protocol["support_evaluation_sample_disjoint"] is True
    assert all(
        query != outer
        for outer, queries in protocol["development_queries_by_outer_target"].items()
        for query in queries
    )

    compatibility = config.compatibility
    assert compatibility["class_prior"] == [0.5, 0.5]
    assert compatibility["reconstruction_term"] == (
        "common_3840_inverse_frame_mse_mean"
    )
    assert compatibility["own_source_calibration_location"] == "case_equal_median"
    assert compatibility["own_source_calibration_scale_floor_value"] == 1.0e-6
    assert compatibility["replica_aggregation"] == (
        "arithmetic_mean_across_all_three_training_seed_scores"
    )
    assert compatibility["exact_nelbo_claimed"] is False

    router = config.router
    assert router["rhos"] == [0.0, 0.25, 0.5]
    assert router["temperature"] == 1.0
    assert router["absolute_max_source_weight"] == 0.25
    assert router["minimum_effective_source_count"] == 6.0
    assert router["minimum_integer_allocation_per_source"] == 1
    assert router["development_total_generated_samples_per_class"] == 1008
    assert router["target_total_generated_samples_per_class"] == 1024
    assert all(
        action["minimum_integer_allocation_per_source"] == 1
        for action in router["actions"]
    )

    assert config.classifier.C == 0.01
    assert config.classifier.penalty == "l2"
    assert config.classifier.solver == "lbfgs"
    assert config.classifier.max_iter == 3000
    assert config.classifier.random_state == 23
    assert config.classifier.scaler_fit == "synthetic_train_only"

    runtime = payload["runtime"]
    assert runtime["expected_development_classifier_fit_count"] == 1944
    assert runtime["expected_target_unique_classifier_fit_count"] == 243
    assert runtime["maximum_total_classifier_fit_count"] == 2187
    assert runtime["maximum_resident_generated_source_blocks"] == 9
    assert runtime["maximum_resident_generated_embedding_bytes"] == 283115520
    assert runtime["control_alias_reuses_rho0_fit"] is True

    selection = config.selection
    assert selection["objective"] == (
        "mean_regret_plus_0.5_upper_quartile_cvar_regret_plus_"
        "0.01_mean_squared_l2_distance_from_uniform"
    )
    assert selection["upper_quartile_cvar_definition"] == (
        "mean_of_largest_ceil_25_percent_regrets"
    )
    assert selection["aggregation"] == (
        "equal_weight_over_q_not_H_and_all_nine_seed_cells"
    )
    assert selection["nonuniform_pass_rule"] == (
        "strictly_positive_mean_paired_bacc_delta_vs_rho0"
    )
    assert selection["fallback_action_id"] == "rho_0.00"
    assert selection["tie_break"] == "smallest_rho_then_lexicographic_action_id"

    boundary = config.claim_boundary
    assert boundary["publication_status"] == "EXPLORATORY_CONSUMED_DATA_ONLY"
    assert boundary["diagnostic_only"] is True
    assert boundary["routing_quality_claimed"] is False
    assert boundary["fresh_confirmation"] is False
    assert boundary["may_feed_stage60"] is False
    assert boundary["may_feed_stage70"] is False
    assert boundary["may_feed_recipe_selection"] is False
    assert boundary["may_feed_deployable_selection"] is False
