from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_proxy_information_audit.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_proxy_information_audit.config import (
    load_utility_aligned_ensemble_endpoint_proxy_information_audit_config,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_proxy_information_audit.contracts import (
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_proxy_information_audit.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import ArtifactEntry, MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_validation_utility_aligned_ensemble_endpoint_proxy_information_audit_v1.yaml"
)
ORIGINAL_CACHE_ID = "midogpp_virchow2_uniform_b_v2_routing_validation_cache_seed42"
ORIGINAL_MANIFEST_ID = "midogpp_source_inner_validation_manifest_v1"
PRIOR_STAGE90_OUTPUT_ID = (
    "midogpp_output_uniform_b_v2_consumed_validation_utility_aligned_"
    "ensemble_endpoint_router_v1"
)
FORBIDDEN_REUSE = {
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


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    return workspace


def _hashes(artifact: ArtifactEntry) -> dict[str, tuple[str, str]]:
    return {
        member: (expectation.algorithm, expectation.digest)
        for member, expectation in artifact.expected_file_hashes.items()
    }


def test_config_freezes_independent_proxy_information_audit_contract() -> None:
    config = load_utility_aligned_ensemble_endpoint_proxy_information_audit_config(
        CONFIG
    )

    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 5
    assert PRIOR_STAGE90_OUTPUT_ID not in config.input_artifact_ids
    assert config.protocol["primary_development_response_count"] == 504
    assert config.protocol["descriptive_per_seed_utility_row_count"] == 4536
    assert config.protocol["descriptive_per_seed_rows_may_feed_model"] is False
    assert config.protocol["strict_H_q_e_exclusion_in_fit_scaling_and_prediction"] is True
    assert config.protocol["cross_fit_mode"] == (
        "strict_all_role_H_q_e_domain_holdout"
    )
    assert config.protocol["strict_crossfit_training_row_count"] == 120
    assert config.protocol["fixed_support_case_count_per_center"] == 2
    assert config.protocol["support_labels_used"] is False
    assert config.protocol["target_actions_built"] is False
    assert config.protocol["target_labels_opened"] is False
    assert config.protocol["stage50_outputs_used"] is False
    assert config.protocol["stage60_outputs_used"] is False
    assert config.protocol["stage70_outputs_used"] is False
    assert config.protocol["previous_stage90_outputs_used"] is False
    assert config.proxy_features["feature_row_count"] == 504
    assert config.proxy_features["primitive_names"] == [
        "metadata_similarity",
        "absolute_ensemble_shift",
        "reconstruction_mean_within_query_z",
        "kl_mean_within_query_z",
        "log_distribution_mmd_within_query_z",
        "signed_margin_projection",
        "threshold_flip_rate",
        "mean_entropy_change",
    ]
    assert config.proxy_features[
        "within_query_standardization_uses_only_current_label_free_candidate_list"
    ] is True
    assert config.proxy_features[
        "within_query_standardization_uses_utility_or_evaluation_labels"
    ] is False
    assert config.proxy_features["cyclic_directional_permutation_seed"] == 90902026
    assert config.proxy_features["cyclic_directional_permutation_shift"] == 1
    assert config.model["family"] == (
        "fixed_alpha_cluster_weighted_ridge_proxy_information_v1"
    )
    assert config.model["ridge_alpha"] == 1.0
    assert config.model["hyperparameter_selection"] == (
        "none_predeclared_before_labels"
    )
    assert config.model["maximum_predictors_per_family"] == 3
    assert config.model["scaling_fit_on_training_fold_only"] is True
    assert config.model["ridge_cluster_unit"] == "outer_target_query"
    assert config.model["family_predictors"]["hybrid_compact"] == [
        "metadata_similarity",
        "log_distribution_mmd_within_query_z",
        "signed_margin_projection",
    ]
    assert config.evaluation["outer_inference_unit_count"] == 9
    assert config.evaluation["query_metric_row_count"] == 72
    assert config.evaluation[
        "query_metrics_are_descriptive_nested_within_centers"
    ] is True
    assert config.evaluation["screening_candidate_family_ids"] == [
        "rich_distributional_compact",
        "directional_action_compact",
        "hybrid_compact",
    ]
    gate = config.evaluation["screening_gate"]
    assert gate["outer_center_mean_spearman_ci95_lower_strictly_above"] == 0.0
    assert gate[
        "outer_center_pairwise_accuracy_ci95_lower_strictly_above"
    ] == 0.5
    assert gate[
        "outer_center_normalized_regret_ci95_upper_strictly_below"
    ] == 0.5
    assert gate["mean_regret_strictly_below_each_control_family"] is True
    assert config.evaluation["screening_gate_may_authorize_policy"] is False
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["source_stream_count"] == 81
    assert config.runtime["development_coarse_task_count"] == 648
    assert config.runtime["development_classifier_fit_count"] == 5184
    assert config.runtime["maximum_total_classifier_fit_count"] == 5184
    assert config.runtime["target_task_count"] == 0
    assert config.runtime["target_action_count"] == 0
    assert config.runtime["target_classifier_fit_count"] == 0
    assert config.claim_boundary["publication_status"] == (
        "EXPLORATORY_CONSUMED_DATA_ONLY"
    )
    assert config.claim_boundary[
        "fixed_two_case_support_is_insufficient_for_policy"
    ] is True
    assert config.claim_boundary["screening_gate_may_authorize_policy"] is False
    assert config.claim_boundary["policy_update_authorized"] is False
    assert config.claim_boundary["may_update_policy"] is False
    assert config.claim_boundary["action_selection_authorized"] is False
    assert config.claim_boundary["promotion_eligible"] is False
    assert config.claim_boundary["may_feed_stage60"] is False
    assert config.claim_boundary["may_feed_stage70"] is False
    assert config.claim_boundary["may_feed_another_stage90_experiment"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "validation_cache_root",
            Path(
                "/tmp/artifacts/midogpp/90_oracles_and_diagnostics/"
                "uniform_b_v2_consumed_validation_utility_aligned_"
                "ensemble_endpoint_router/v1"
            ),
        ),
        (
            "validation_cache_root",
            Path("/tmp/artifacts/midogpp/50_all_candidate_utility_matrix/oracle"),
        ),
        (
            "validation_cache_root",
            Path("/tmp/artifacts/midogpp/60_routing_and_composition/policy"),
        ),
        (
            "validation_cache_root",
            Path("/tmp/artifacts/midogpp/70_frozen_policy_downstream/target"),
        ),
        ("validation_cache_root", Path("/tmp/historical/cache")),
        ("validation_cache_root", Path("/tmp/quarantine/cache")),
    ],
)
def test_input_fence_rejects_prior_or_unauthorized_surfaces(
    field: str, value: Path
) -> None:
    config = load_utility_aligned_ensemble_endpoint_proxy_information_audit_config(
        CONFIG
    )
    assert_input_fence(config)
    with pytest.raises(ProtocolError):
        assert_input_fence(replace(config, **{field: value}))


def test_input_fence_rejects_prior_stage90_output_identity() -> None:
    config = load_utility_aligned_ensemble_endpoint_proxy_information_audit_config(
        CONFIG
    )
    mutated_ids = (
        *config.input_artifact_ids[:2],
        PRIOR_STAGE90_OUTPUT_ID,
        *config.input_artifact_ids[3:],
    )

    class MutatedConfig:
        experiment_id = config.experiment_id
        output_artifact_id = config.output_artifact_id
        input_artifact_ids = mutated_ids
        expert_bank_root = config.expert_bank_root
        generation_lock_root = config.generation_lock_root
        validation_cache_root = config.validation_cache_root
        validation_manifest_path = config.validation_manifest_path
        metadata_profile_root = config.metadata_profile_root

    with pytest.raises(ProtocolError):
        assert_input_fence(MutatedConfig())


def test_cli_parser_and_lazy_dispatch_use_dedicated_audit(monkeypatch, capsys) -> None:
    import midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_proxy_information_audit as surface

    parsed = cli.build_parser().parse_args(
        (
            "utility-aligned-ensemble-endpoint-proxy-information-audit",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/ensemble-endpoint-proxy-information-audit",
        )
    )
    assert parsed.surface == (
        "utility-aligned-ensemble-endpoint-proxy-information-audit"
    )

    sentinel_config = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        surface,
        "load_utility_aligned_ensemble_endpoint_proxy_information_audit_config",
        lambda _path: sentinel_config,
    )

    def _run(config: object, *, artifact_root: Path) -> Path:
        calls.append((config, artifact_root))
        return Path("/tmp/ensemble-endpoint-proxy-information-result")

    monkeypatch.setattr(
        surface,
        "run_utility_aligned_ensemble_endpoint_proxy_information_audit",
        _run,
    )
    result = cli.main(
        [
            "utility-aligned-ensemble-endpoint-proxy-information-audit",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/ensemble-endpoint-proxy-information-audit",
        ]
    )

    assert result == 0
    assert calls == [
        (sentinel_config, Path("/tmp/ensemble-endpoint-proxy-information-audit"))
    ]
    assert capsys.readouterr().out.strip() == (
        "/tmp/ensemble-endpoint-proxy-information-result"
    )


def test_registry_and_output_are_independent_terminal_and_closed_world() -> None:
    workspace = _workspace()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert PRIOR_STAGE90_OUTPUT_ID not in experiment.input_artifact_ids
    assert not {
        "50_all_candidate_utility_matrix",
        "60_routing_and_composition",
        "70_frozen_policy_downstream",
        "90_oracles_and_diagnostics",
    }.intersection(
        workspace.artifacts[artifact_id].stage
        for artifact_id in experiment.input_artifact_ids
    )
    assert set(experiment.input_claim_scope_exceptions) == {
        "midogpp_output_uniform_b_v2_generation_lock_v1",
        "midogpp_routing_metadata_profiles_v1",
    }
    assert experiment.runner_argv == (
        "{python}",
        "-m",
        "midogpp_thesis",
        "cvae-diagnostics",
        "utility-aligned-ensemble-endpoint-proxy-information-audit",
        "--config",
        "{resolved_config}",
        "--artifact-root",
        f"output://{OUTPUT_ARTIFACT_ID}",
    )
    assert experiment.runner_env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert experiment.runner_env["OMP_NUM_THREADS"] == "1"

    assert output.stage == "90_oracles_and_diagnostics"
    assert output.claim_scope == "diagnostic_only"
    assert output.required_files == REQUIRED_FILES
    assert set(output.forbidden_reuse) == FORBIDDEN_REUSE.union(
        {"oracle_and_diagnostic_evidence"}
    )
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert output.semantic_identities["primary_response_row_count"] == "504"
    assert output.semantic_identities["descriptive_seed_row_count"] == "4536"
    assert output.semantic_identities["seed_rows_are_model_observations"] == "false"
    assert output.semantic_identities["query_metric_row_count"] == "72"
    assert output.semantic_identities[
        "strict_H_q_e_exclusion_in_fit_scaling_and_prediction"
    ] == "true"
    for key in (
        "target_actions_built",
        "target_predictions_materialized",
        "target_labels_opened",
        "stage50_outputs_used",
        "stage60_outputs_used",
        "stage70_outputs_used",
        "previous_stage90_outputs_used",
        "historical_or_quarantined_inputs_used",
        "screening_gate_may_authorize_policy",
        "policy_update_authorized",
        "may_update_policy",
        "action_selection_authorized",
        "fresh_evidence",
        "routing_quality_claimed",
        "target_performance_claimed",
        "promotion_eligible",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_another_stage90_experiment",
    ):
        assert output.semantic_identities[key] == "false"


def test_consumed_validation_aliases_are_byte_exact_and_single_consumer() -> None:
    workspace = _workspace()
    lock_hashes: set[str] = set()
    for alias_id, original_id in (
        (VALIDATION_CACHE_ARTIFACT_ID, ORIGINAL_CACHE_ID),
        (VALIDATION_MANIFEST_ARTIFACT_ID, ORIGINAL_MANIFEST_ID),
    ):
        alias = workspace.artifacts[alias_id]
        original = workspace.artifacts[original_id]
        assert alias.physical_path == original.physical_path
        assert alias.canonical_path == original.canonical_path
        assert alias.required_files == original.required_files
        assert _hashes(alias) == _hashes(original)
        assert alias.semantic_identities["alias_of_artifact_id"] == original_id
        assert alias.semantic_identities["fresh_evidence"] == "false"
        assert alias.semantic_identities[
            "authorized_consumer_experiment_ids"
        ] == EXPERIMENT_ID
        lock_hashes.add(alias.semantic_identities["policy_consumption_lock_hash"])
        assert set(alias.forbidden_reuse) == FORBIDDEN_REUSE
        assert alias.may_feed_recipe_selection is False
        assert alias.may_feed_deployable_selection is False

        consumers = {
            candidate.experiment_id
            for candidate in workspace.experiments.values()
            if alias_id in candidate.input_artifact_ids
        }
        assert consumers == {EXPERIMENT_ID}
    assert lock_hashes == {"b2df56d4f95e51f2"}
