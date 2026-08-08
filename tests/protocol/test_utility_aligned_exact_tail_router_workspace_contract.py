from __future__ import annotations

from pathlib import Path

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.config import (
    load_utility_aligned_exact_tail_router_config,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.contracts import (
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
    / "uniform_b_v2_consumed_validation_utility_aligned_exact_tail_router_v1.yaml"
)
ORIGINAL_CACHE_ID = "midogpp_virchow2_uniform_b_v2_routing_validation_cache_seed42"
ORIGINAL_MANIFEST_ID = "midogpp_source_inner_validation_manifest_v1"
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


def test_config_freezes_consumed_crossfit_and_workstation_contract() -> None:
    config = load_utility_aligned_exact_tail_router_config(CONFIG)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert config.protocol["fixed_support_case_count_per_center"] == 2
    assert config.protocol["fresh_policy_minimum_support_case_count"] == 8
    assert config.protocol["low_support_status"] == "INSUFFICIENT_SUPPORT_FOR_POLICY"
    assert config.protocol[
        "outer_target_H_excluded_from_development_query_and_source_roles"
    ] is True
    assert config.protocol["pseudoquery_q_excluded_from_candidate_source_role"] is True
    assert config.protocol["development_predictions_sealed_before_development_labels"] is True
    assert config.protocol[
        "development_label_phase_contains_all_centers_but_outer_H_rows_excluded_per_model"
    ] is True
    assert config.protocol[
        "target_predictions_globally_sealed_before_terminal_target_scoring"
    ] is True
    assert config.actions["inner_matched_total_per_class"] == 1134
    assert config.actions["target_matched_total_per_class"] == 1152
    assert config.actions["source_cache_prefix_per_class"] == 270
    assert config.evaluation["primary_contrasts"] == [
        "R2-B",
        "R2-G_delta",
        "R2-U",
        "R2-P",
    ]
    assert config.evaluation[
        "target_scoring_capability_requires_global_target_seal"
    ] is True
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["source_prefix_rows_per_class"] == 270
    assert config.runtime["development_classifier_fit_count"] == 5184
    assert config.claim_boundary["fixed_two_case_R2_is_insufficient_for_policy"] is True
    assert config.claim_boundary["fresh_evidence"] is False
    assert config.claim_boundary["routing_quality_claimed"] is False
    assert config.claim_boundary["may_feed_stage60"] is False
    assert config.claim_boundary["may_feed_stage70"] is False

    parsed = cli.build_parser().parse_args(
        (
            "utility-aligned-exact-tail-router-diagnostic",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/utility-aligned-exact-tail-router",
        )
    )
    assert parsed.surface == "utility-aligned-exact-tail-router-diagnostic"


def test_cli_dispatches_to_dedicated_runner(monkeypatch, capsys) -> None:
    import midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router as surface

    sentinel_config = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        surface,
        "load_utility_aligned_exact_tail_router_config",
        lambda _path: sentinel_config,
    )

    def _run(config: object, *, artifact_root: Path) -> Path:
        calls.append((config, artifact_root))
        return Path("/tmp/utility-aligned-exact-tail-result")

    monkeypatch.setattr(
        surface, "run_utility_aligned_exact_tail_router_diagnostic", _run
    )
    result = cli.main(
        [
            "utility-aligned-exact-tail-router-diagnostic",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/utility-aligned-exact-tail-router",
        ]
    )

    assert result == 0
    assert calls == [
        (sentinel_config, Path("/tmp/utility-aligned-exact-tail-router"))
    ]
    assert capsys.readouterr().out.strip() == "/tmp/utility-aligned-exact-tail-result"


def test_registry_and_output_are_terminal_and_closed_world() -> None:
    workspace = _workspace()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert set(experiment.input_claim_scope_exceptions) == {
        "midogpp_output_uniform_b_v2_generation_lock_v1",
        "midogpp_routing_metadata_profiles_v1",
    }
    assert experiment.runner_argv == (
        "{python}",
        "-m",
        "midogpp_thesis",
        "cvae-diagnostics",
        "utility-aligned-exact-tail-router-diagnostic",
        "--config",
        "{resolved_config}",
        "--artifact-root",
        f"output://{OUTPUT_ARTIFACT_ID}",
    )
    assert experiment.runner_env["CUDA_VISIBLE_DEVICES"] == "0,1"

    assert output.stage == "90_oracles_and_diagnostics"
    assert output.claim_scope == "diagnostic_only"
    assert output.required_files == REQUIRED_FILES
    assert set(output.forbidden_reuse) == FORBIDDEN_REUSE.union(
        {"oracle_and_diagnostic_evidence"}
    )
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    for key in (
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
    assert output.semantic_identities["routing_status"] == (
        "INSUFFICIENT_SUPPORT_FOR_POLICY"
    )
    assert output.semantic_identities[
        "development_crossfit_labels_opened_before_target_action_lock"
    ] == "true"


def test_consumed_validation_aliases_are_byte_exact_and_single_consumer() -> None:
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
        assert set(alias.forbidden_reuse) == FORBIDDEN_REUSE
        assert alias.may_feed_recipe_selection is False
        assert alias.may_feed_deployable_selection is False

        consumers = {
            candidate.experiment_id
            for candidate in workspace.experiments.values()
            if alias_id in candidate.input_artifact_ids
        }
        assert consumers == {EXPERIMENT_ID}
