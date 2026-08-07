from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.residual_topup_router import config as config_module
from midogpp_thesis.cvae.diagnostics.residual_topup_router.bundle import (
    REQUIRED_FILES as BUNDLE_REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_router.config import (
    load_residual_topup_config,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_router.contracts import (
    EXPERIMENT_ID,
    FORBIDDEN_ROUTER_INPUT_ARTIFACT_IDS,
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
    / "uniform_b_v2_consumed_validation_residual_topup_router_v1.yaml"
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


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    return workspace


def _hashes(artifact: ArtifactEntry) -> dict[str, tuple[str, str]]:
    return {
        member: (expectation.algorithm, expectation.digest)
        for member, expectation in artifact.expected_file_hashes.items()
    }


def test_residual_topup_config_and_workstation_surface_are_frozen() -> None:
    config = load_residual_topup_config(CONFIG)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert config.protocol["target_expert_excluded"] is True
    assert config.protocol["inner_outer_target_excluded"] is True
    assert config.protocol["inner_query_expert_excluded"] is True
    assert config.protocol[
        "all_actions_predictions_globally_sealed_before_any_label_access"
    ] is True
    assert config.protocol[
        "evaluation_labels_available_before_global_prediction_seal"
    ] is False
    assert config.protocol["evaluation_labels_available_to_action_prediction"] is False
    assert config.protocol[
        "inner_query_evaluation_labels_used_after_global_seal_for_outer_H_calibration"
    ] is True
    assert config.protocol[
        "target_H_evaluation_labels_used_for_own_selection"
    ] is False
    assert config.protocol["previous_stage90_router_or_utility_inputs_used"] is False
    assert config.actions["family"] == (
        "immutable_equal_union_backbone_with_residual_topup_v1"
    )
    assert config.actions["target_base_total_per_class"] == 1024
    assert config.actions["target_topup_total_per_class"] == 128
    assert config.actions["development_base_total_per_class"] == 1008
    assert config.actions["development_topup_total_per_class"] == 126
    assert config.actions["primary_control_action_id"] == "uniform_topup"
    assert config.actions["base_only_role"] == (
        "separate_budget_reference_not_primary_control"
    )
    assert config.selection["nested_hyperparameter_selection"] is False
    assert config.selection["target_H_labels_used_for_selection"] is False
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["maximum_unique_classifier_fit_count"] == 1539
    assert config.claim_boundary["terminal_stage90_diagnostic"] is True
    assert config.claim_boundary["routing_quality_claimed"] is False
    assert config.claim_boundary["promotion_eligible"] is False
    assert config.claim_boundary["may_feed_stage60"] is False
    assert config.claim_boundary["may_feed_stage70"] is False

    parsed = cli.build_parser().parse_args(
        (
            "residual-topup-router-diagnostic",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/residual-topup-router",
        )
    )
    assert parsed.surface == "residual-topup-router-diagnostic"


def test_residual_topup_cli_dispatches_to_dedicated_runner(
    monkeypatch,
    capsys,
) -> None:
    sentinel_config = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        config_module,
        "load_residual_topup_config",
        lambda path: sentinel_config,
    )
    runner_module = ModuleType(
        "midogpp_thesis.cvae.diagnostics.residual_topup_router.runner"
    )

    def _run(config: object, *, artifact_root: Path) -> Path:
        calls.append((config, artifact_root))
        return Path("/tmp/residual-topup-result")

    runner_module.run_residual_topup_router_diagnostic = _run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)

    result = cli.main(
        [
            "residual-topup-router-diagnostic",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/residual-topup-router",
        ]
    )

    assert result == 0
    assert calls == [(sentinel_config, Path("/tmp/residual-topup-router"))]
    assert capsys.readouterr().out.strip() == "/tmp/residual-topup-result"


def test_registry_and_catalog_are_terminal_and_previous_stage90_free() -> None:
    workspace = _workspace()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert not FORBIDDEN_ROUTER_INPUT_ARTIFACT_IDS.intersection(
        experiment.input_artifact_ids
    )
    assert set(experiment.input_claim_scope_exceptions) == {
        "midogpp_output_uniform_b_v2_generation_lock_v1"
    }
    assert experiment.runner_argv == (
        "{python}",
        "-m",
        "midogpp_thesis",
        "cvae-diagnostics",
        "residual-topup-router-diagnostic",
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
        "promotion_eligible",
        "oracle_eligible",
        "may_feed_stage60",
        "may_feed_stage70",
    ):
        assert output.semantic_identities[key] == "false"


def test_validation_aliases_are_byte_exact_and_experiment_fenced() -> None:
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


def test_config_uses_only_fenced_validation_aliases() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    inputs = payload["inputs"]
    assert inputs["validation_cache_root"] == (
        f"artifact://{VALIDATION_CACHE_ARTIFACT_ID}"
    )
    assert inputs["validation_manifest_path"] == (
        f"artifact://{VALIDATION_MANIFEST_ARTIFACT_ID}/manifest.csv"
    )
    assert ORIGINAL_CACHE_ID not in str(inputs)
    assert ORIGINAL_MANIFEST_ID not in str(inputs)
    assert not any(
        artifact_id in str(inputs)
        for artifact_id in FORBIDDEN_ROUTER_INPUT_ARTIFACT_IDS
    )
