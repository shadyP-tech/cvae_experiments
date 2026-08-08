from __future__ import annotations

from pathlib import Path

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof.bundle import (
    REQUIRED_FILES as BUNDLE_REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof.config import (
    load_residual_topup_case_oof_config,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof.contracts import (
    EXPECTED_ACTION_COUNT_PER_TARGET,
    EXPECTED_CASE_OOF_FOLD_COUNT,
    EXPECTED_FROZEN_ACTION_COUNT,
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
    / "uniform_b_v2_consumed_validation_residual_topup_b_u_g_s_case_oof_v1.yaml"
)
ORIGINAL_CACHE_ID = (
    "midogpp_virchow2_uniform_b_v2_routing_validation_cache_seed42"
)
ORIGINAL_MANIFEST_ID = "midogpp_source_inner_validation_manifest_v1"
OLD_RESIDUAL_OUTPUT = (
    "midogpp_output_uniform_b_v2_consumed_validation_residual_topup_router_v1"
)
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


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    return workspace


def _hashes(artifact: ArtifactEntry) -> dict[str, tuple[str, str]]:
    return {
        member: (expectation.algorithm, expectation.digest)
        for member, expectation in artifact.expected_file_hashes.items()
    }


def test_case_oof_config_freezes_protocol_actions_and_workstation() -> None:
    config = load_residual_topup_case_oof_config(CONFIG)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert config.protocol["total_case_count"] == 44
    assert config.protocol["fixed_support_case_count_per_center"] == 2
    assert config.protocol["expected_case_oof_fold_count"] == (
        EXPECTED_CASE_OOF_FOLD_COUNT
    )
    assert config.protocol["cross_fitted_fixed_support_diagnostic"] is True
    assert config.protocol["cross_fitted_transductive_diagnostic"] is False
    assert config.protocol["heldout_case_excluded_from_own_route"] is True
    assert config.protocol["other_evaluation_embeddings_available_to_router"] is False
    assert config.protocol["global_proxy_excludes_outer_target_H_and_query_q"] is True
    assert config.protocol["target_support_proxy_uses_fixed_S_H_only"] is True
    assert config.protocol["support_labels_used"] is False
    assert config.protocol["source_expert_updated"] is False
    assert config.protocol[
        "all_actions_predictions_globally_sealed_before_any_label_access"
    ] is True
    assert config.protocol["previous_stage90_router_or_utility_inputs_used"] is False
    assert config.actions["action_count_per_target"] == (
        EXPECTED_ACTION_COUNT_PER_TARGET
    )
    assert config.actions["frozen_action_count"] == EXPECTED_FROZEN_ACTION_COUNT
    assert config.actions["replica_aggregation"] == (
        "mean_three_training_replicas_before_each_case_ballot"
    )
    assert config.actions["borda_direction_semantics"] == (
        "explicit_one_minus_mean_normalized_midrank"
    )
    assert config.actions["no_selector_or_fallback_gate"] is True
    assert config.evaluation["primary_contrasts"] == ["S-U", "S-G"]
    assert config.evaluation["inference_unit"] == "target_center"
    assert config.evaluation["inference_center_count"] == 9
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["maximum_unique_classifier_fit_count"] == 1053
    assert config.claim_boundary["terminal_stage90_diagnostic"] is True
    assert config.claim_boundary["fresh_evidence"] is False
    assert config.claim_boundary["routing_quality_claimed"] is False
    assert config.claim_boundary["promotion_eligible"] is False
    assert config.claim_boundary["may_feed_stage60"] is False
    assert config.claim_boundary["may_feed_stage70"] is False

    parsed = cli.build_parser().parse_args(
        (
            "residual-topup-case-oof-diagnostic",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/residual-topup-case-oof",
        )
    )
    assert parsed.surface == "residual-topup-case-oof-diagnostic"


def test_case_oof_cli_dispatches_to_dedicated_runner(monkeypatch, capsys) -> None:
    import midogpp_thesis.cvae.diagnostics.residual_topup_case_oof as surface

    sentinel_config = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        surface,
        "load_residual_topup_case_oof_config",
        lambda path: sentinel_config,
    )

    def _run(config: object, *, artifact_root: Path) -> Path:
        calls.append((config, artifact_root))
        return Path("/tmp/residual-topup-case-oof-result")

    monkeypatch.setattr(surface, "run_residual_topup_case_oof_diagnostic", _run)
    result = cli.main(
        [
            "residual-topup-case-oof-diagnostic",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/residual-topup-case-oof",
        ]
    )

    assert result == 0
    assert calls == [(sentinel_config, Path("/tmp/residual-topup-case-oof"))]
    assert capsys.readouterr().out.strip() == "/tmp/residual-topup-case-oof-result"


def test_registry_and_output_are_terminal_and_prior_stage90_free() -> None:
    workspace = _workspace()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert OLD_RESIDUAL_OUTPUT not in experiment.input_artifact_ids
    assert set(experiment.input_claim_scope_exceptions) == {
        "midogpp_output_uniform_b_v2_generation_lock_v1"
    }
    assert experiment.runner_argv == (
        "{python}",
        "-m",
        "midogpp_thesis",
        "cvae-diagnostics",
        "residual-topup-case-oof-diagnostic",
        "--config",
        "{resolved_config}",
        "--artifact-root",
        f"output://{OUTPUT_ARTIFACT_ID}",
    )
    assert experiment.runner_env["CUDA_VISIBLE_DEVICES"] == "0,1"

    assert output.stage == "90_oracles_and_diagnostics"
    assert output.claim_scope == "diagnostic_only"
    assert output.required_files == BUNDLE_REQUIRED_FILES
    assert set(output.forbidden_reuse) == NON_DIAGNOSTIC_REUSE.union(
        {"oracle_and_diagnostic_evidence"}
    )
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    for key in (
        "routing_quality_claimed",
        "target_performance_claimed",
        "target_specific_router_success_claimed",
        "promotion_eligible",
        "oracle_eligible",
        "may_feed_stage60",
        "may_feed_stage70",
    ):
        assert output.semantic_identities[key] == "false"
    assert output.semantic_identities["cross_fitted_fixed_support_diagnostic"] == "true"
    assert output.semantic_identities["cross_fitted_transductive_diagnostic"] == "false"


def test_validation_aliases_are_byte_exact_and_single_consumer_fenced() -> None:
    workspace = _workspace()
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
        assert set(alias.forbidden_reuse) == NON_DIAGNOSTIC_REUSE
        assert alias.may_feed_recipe_selection is False
        assert alias.may_feed_deployable_selection is False

        consumers = {
            candidate.experiment_id
            for candidate in workspace.experiments.values()
            if alias_id in candidate.input_artifact_ids
        }
        assert consumers == {EXPERIMENT_ID}
