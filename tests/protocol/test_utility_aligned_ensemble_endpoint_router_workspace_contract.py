from __future__ import annotations

from pathlib import Path

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.config import (
    load_utility_aligned_ensemble_endpoint_router_config,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.contracts import (
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.workspace.runtime import ArtifactEntry, MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_validation_utility_aligned_ensemble_endpoint_router_v1.yaml"
)
ORIGINAL_CACHE_ID = "midogpp_virchow2_uniform_b_v2_routing_validation_cache_seed42"
ORIGINAL_MANIFEST_ID = "midogpp_source_inner_validation_manifest_v1"
PRIOR_STAGE90_OUTPUT_ID = (
    "midogpp_output_uniform_b_v2_consumed_validation_utility_aligned_"
    "exact_tail_router_v1"
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


def test_config_freezes_ensemble_endpoint_units_and_workstation_contract() -> None:
    config = load_utility_aligned_ensemble_endpoint_router_config(CONFIG)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 5
    assert PRIOR_STAGE90_OUTPUT_ID not in config.input_artifact_ids
    assert config.protocol["primary_development_response_count"] == 504
    assert config.protocol["descriptive_per_seed_utility_row_count"] == 4536
    assert config.protocol["descriptive_per_seed_rows_may_feed_model"] is False
    assert config.protocol["probabilities_averaged_before_single_threshold"] is True
    assert config.protocol["per_seed_support_shifts_are_descriptive_only"] is True
    assert config.protocol["fixed_support_case_count_per_center"] == 2
    assert config.protocol["fresh_policy_minimum_support_case_count"] == 8
    assert config.protocol["low_support_status"] == "INSUFFICIENT_SUPPORT_FOR_POLICY"
    assert config.protocol["previous_stage90_outputs_used"] is False
    assert config.protocol["stage60_policy_or_surface_inputs_used"] is False
    assert config.protocol["stage70_target_or_scoring_inputs_used"] is False
    assert config.model["response_row_count"] == 504
    assert config.model["per_seed_utility_row_count"] == 4536
    assert config.model["per_seed_utility_rows_may_feed_model"] is False
    assert config.model["exact_nine_seed_cells_collapsed_before_model_fit"] is True
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["generation_workers_per_device"] == 1
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["tf32_enabled"] is False
    assert config.runtime["amp_enabled"] is False
    assert config.runtime["source_stream_count"] == 81
    assert config.runtime["target_action_identity_count"] == 1053
    assert config.runtime["target_unique_classifier_fit_count"] == 810
    assert config.claim_boundary["fixed_two_case_R2E_is_insufficient_for_policy"] is True
    assert config.claim_boundary["terminal_stage90_diagnostic"] is True
    assert config.claim_boundary["fresh_evidence"] is False
    assert config.claim_boundary["may_update_policy"] is False
    assert config.claim_boundary["may_feed_stage60"] is False
    assert config.claim_boundary["may_feed_stage70"] is False


def test_cli_parser_and_lazy_dispatch_use_dedicated_surface(monkeypatch, capsys) -> None:
    import midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router as surface

    parsed = cli.build_parser().parse_args(
        (
            "utility-aligned-ensemble-endpoint-router-diagnostic",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/utility-aligned-ensemble-endpoint-router",
        )
    )
    assert parsed.surface == "utility-aligned-ensemble-endpoint-router-diagnostic"

    sentinel_config = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        surface,
        "load_utility_aligned_ensemble_endpoint_router_config",
        lambda _path: sentinel_config,
    )

    def _run(config: object, *, artifact_root: Path) -> Path:
        calls.append((config, artifact_root))
        return Path("/tmp/utility-aligned-ensemble-endpoint-result")

    monkeypatch.setattr(
        surface,
        "run_utility_aligned_ensemble_endpoint_router_diagnostic",
        _run,
    )
    result = cli.main(
        [
            "utility-aligned-ensemble-endpoint-router-diagnostic",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/utility-aligned-ensemble-endpoint-router",
        ]
    )

    assert result == 0
    assert calls == [
        (sentinel_config, Path("/tmp/utility-aligned-ensemble-endpoint-router"))
    ]
    assert capsys.readouterr().out.strip() == (
        "/tmp/utility-aligned-ensemble-endpoint-result"
    )


def test_registry_and_output_are_active_terminal_and_closed_world() -> None:
    workspace = _workspace()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert PRIOR_STAGE90_OUTPUT_ID not in experiment.input_artifact_ids
    assert not {
        "60_routing_and_composition",
        "70_frozen_policy_downstream",
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
        "utility-aligned-ensemble-endpoint-router-diagnostic",
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
    assert output.semantic_identities["frozen_target_action_count"] == "117"
    assert output.semantic_identities["target_action_seed_identity_count"] == "1053"
    assert output.semantic_identities[
        "target_action_seed_identities_are_descriptive_only"
    ] == "true"
    assert output.semantic_identities["routing_status"] == (
        "INSUFFICIENT_SUPPORT_FOR_POLICY"
    )
    for key in (
        "previous_stage90_outputs_used",
        "stage60_outputs_used",
        "stage70_outputs_used",
        "fresh_evidence",
        "routing_quality_claimed",
        "target_performance_claimed",
        "target_specific_router_success_claimed",
        "promotion_eligible",
        "oracle_eligible",
        "may_feed_stage60",
        "may_feed_stage70",
    ):
        assert output.semantic_identities[key] == "false"


def test_consumed_validation_aliases_are_byte_exact_and_single_consumer() -> None:
    workspace = _workspace()
    aliases = (
        (VALIDATION_CACHE_ARTIFACT_ID, ORIGINAL_CACHE_ID),
        (VALIDATION_MANIFEST_ARTIFACT_ID, ORIGINAL_MANIFEST_ID),
    )
    lock_hashes: set[str] = set()
    for alias_id, original_id in aliases:
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
    assert len(lock_hashes) == 1
