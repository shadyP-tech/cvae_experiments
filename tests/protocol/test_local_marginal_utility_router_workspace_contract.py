from __future__ import annotations

from pathlib import Path

import yaml

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.bundle import (
    REQUIRED_FILES as BUNDLE_REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.config import (
    load_local_marginal_utility_router_config,
)
from midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.contracts import (
    EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    EXPECTED_MARGINAL_UTILITY_ROW_COUNT,
    PERTURBATION_LIBRARY_HASH,
)
from midogpp_thesis.workspace.runtime import ArtifactEntry, MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_validation_local_marginal_utility_router_v1.yaml"
)
EXPERIMENT_ID = (
    "midogpp.oracle."
    "uniform_b_v2_consumed_validation_local_marginal_utility_router.v1"
)
OUTPUT_ID = (
    "midogpp_output_uniform_b_v2_consumed_validation_"
    "local_marginal_utility_router_v1"
)
EXPERT_BANK_ID = "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
GENERATION_LOCK_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
CACHE_ALIAS_ID = "midogpp_stage90_local_marginal_utility_router_validation_cache_v1"
MANIFEST_ALIAS_ID = (
    "midogpp_stage90_local_marginal_utility_router_validation_manifest_v1"
)
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
    "manifests/perturbation_library.json",
    "manifests/support_partition_lock.json",
    "manifests/compatibility_index.json",
    "manifests/global_development_prediction_seal.json",
    "manifests/content_index.json",
    "arrays/development_predictions.npz",
    "tables/support_partitions.csv",
    "tables/compatibility_case_energy.csv",
    "tables/compatibility_scores.csv",
    "tables/development_prediction_index.csv",
    "tables/development_metrics.csv",
    "tables/marginal_utilities.csv",
    "tables/loqdo_predictions.csv",
    "tables/loqdo_summary.csv",
    "tables/model_fits.csv",
    "tables/target_plans.csv",
    "reports/phase_01_support_and_compatibility_complete.json",
    "reports/phase_02_global_predictions_sealed.json",
    "reports/phase_03_utility_surface_complete.json",
    "reports/phase_04_model_and_plans_complete.json",
    "reports/label_access_report.json",
    "reports/leakage_report.json",
    "reports/learnability_report.json",
    "reports/optimizer_report.json",
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


def test_local_marginal_utility_router_is_stage90_diagnostic_only() -> None:
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
        "local-marginal-utility-router-diagnostic",
        "--config",
        "{resolved_config}",
        "--artifact-root",
        f"output://{OUTPUT_ID}",
    )

    assert output.stage == "90_oracles_and_diagnostics"
    assert output.canonical_path == (
        "artifacts/midogpp/90_oracles_and_diagnostics/"
        "uniform_b_v2_consumed_validation_local_marginal_utility_router/v1"
    )
    assert output.evidence_label == "EXPLORATORY_CONSUMED_DATA_ONLY"
    assert output.claim_scope == "diagnostic_only"
    assert output.required_files == BUNDLE_REQUIRED_FILES
    assert set(output.required_files) == REQUIRED_OUTPUT_FILES
    assert set(output.forbidden_reuse) == NON_DIAGNOSTIC_REUSE
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert output.semantic_identities["routing_quality_claimed"] == "false"
    assert output.semantic_identities["target_performance_claimed"] == "false"
    assert output.semantic_identities["target_labels_opened_for_target_plans"] == "false"
    assert output.semantic_identities["target_plans_unscored"] == "true"
    assert output.semantic_identities["may_feed_stage60"] == "false"
    assert output.semantic_identities["may_feed_stage70"] == "false"


def test_local_utility_validation_aliases_are_byte_exact_and_experiment_fenced() -> None:
    workspace = _workspace()
    for alias_id, original_id in (
        (CACHE_ALIAS_ID, ORIGINAL_CACHE_ID),
        (MANIFEST_ALIAS_ID, ORIGINAL_MANIFEST_ID),
    ):
        alias = workspace.artifacts[alias_id]
        original = workspace.artifacts[original_id]
        assert alias.physical_path == original.physical_path
        assert alias.canonical_path == original.canonical_path
        assert alias.required_files == original.required_files
        assert _hashes(alias) == _hashes(original)
        assert alias.semantic_identities["alias_of_artifact_id"] == original_id
        assert alias.semantic_identities["fresh_evidence"] == "false"
        assert alias.semantic_identities["authorized_consumer_experiment_ids"] == (
            EXPERIMENT_ID
        )
        assert set(alias.forbidden_reuse) == NON_DIAGNOSTIC_REUSE
        assert alias.may_feed_recipe_selection is False
        assert alias.may_feed_deployable_selection is False

        consumers = {
            experiment.experiment_id
            for experiment in workspace.experiments.values()
            if alias_id in experiment.input_artifact_ids
        }
        assert consumers == {EXPERIMENT_ID}


def test_local_marginal_utility_config_freezes_surface_and_firewall() -> None:
    config = load_local_marginal_utility_router_config(CONFIG_PATH)
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
    assert protocol["support_case_count_per_center"] == 2
    assert protocol["support_labels_used"] is False
    assert protocol["support_evaluation_case_disjoint"] is True
    assert protocol["support_evaluation_sample_disjoint"] is True
    assert protocol[
        "global_development_predictions_sealed_before_any_development_label_access"
    ] is True
    assert protocol["target_H_labels_used_for_target_plan"] is False
    assert protocol["target_predictions_materialized"] is False
    assert protocol["target_labels_opened"] is False
    for outer, queries in protocol["development_queries_by_outer_target"].items():
        assert outer not in queries
        for query in queries:
            sources = protocol["development_sources_by_outer_target_and_query"][outer][query]
            assert len(sources) == 7
            assert outer not in sources
            assert query not in sources

    perturbations = config.perturbations
    assert perturbations["epsilon"] == 0.125
    assert perturbations["control_action_id"] == "control"
    assert perturbations["boost_action_prefix"] == "boost_source_"
    assert perturbations["control_allocation_per_source_per_class"] == 144
    assert perturbations["boosted_source_allocation_per_class"] == 252
    assert perturbations["nonboosted_source_allocation_per_class"] == 126
    assert perturbations["maximum_source_weight"] == 0.25
    assert perturbations["perturbed_effective_source_count"] == 6.4
    assert perturbations["perturbation_library_hash"] == PERTURBATION_LIBRARY_HASH

    assert config.model["alpha_selection"] == (
        "nested_loqdo_equal_query_cluster_mse_strict_domain_role_exclusion"
    )
    assert config.model["primary_learnability_metric"] == (
        "top1_utility_oracle_agreement"
    )
    assert config.model["outer_evaluation"] == "leave_one_domain_out"
    assert config.model["outer_fold_domain_exclusion"] == (
        "heldout_domain_excluded_from_both_query_center_and_source_roles"
    )
    assert config.model["inner_alpha_fold_domain_exclusion"] == (
        "heldout_inner_domain_excluded_from_both_query_center_and_source_roles"
    )
    assert config.model["primary_learnability_metric"] == (
        "top1_utility_oracle_agreement"
    )
    assert config.model["rmse_may_override_primary_metrics"] is False
    assert config.model["feature_label_status"] == "label_free"
    assert config.model["target_H_labels_used"] is False
    assert config.optimizer["kappa"] == 1.0
    assert config.optimizer["max_source_weight"] == 0.25
    assert config.optimizer["min_effective_sources"] == 6.0
    assert config.optimizer["target_labels_used"] is False

    assert config.runtime["expected_development_classifier_fit_count"] == (
        EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT
    )
    assert config.runtime["expected_marginal_utility_row_count"] == (
        EXPECTED_MARGINAL_UTILITY_ROW_COUNT
    )
    assert config.runtime["maximum_total_classifier_fit_count"] == 5184
    assert config.runtime["control_fit_reused_within_outer_query_seed_cell"] is True

    boundary = config.claim_boundary
    assert boundary["publication_status"] == "EXPLORATORY_CONSUMED_DATA_ONLY"
    assert boundary["diagnostic_only"] is True
    assert boundary["routing_quality_claimed"] is False
    assert boundary["target_performance_claimed"] is False
    assert boundary["fresh_confirmation"] is False
    assert boundary["may_feed_stage60"] is False
    assert boundary["may_feed_stage70"] is False
    assert boundary["may_feed_recipe_selection"] is False
    assert boundary["may_feed_deployable_selection"] is False


def test_local_marginal_utility_cli_surface_is_registered_lazily() -> None:
    args = build_parser().parse_args(
        [
            "local-marginal-utility-router-diagnostic",
            "--config",
            str(CONFIG_PATH),
            "--artifact-root",
            f"output://{OUTPUT_ID}",
        ]
    )
    assert args.surface == "local-marginal-utility-router-diagnostic"
