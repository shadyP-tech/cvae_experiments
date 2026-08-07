from __future__ import annotations

from pathlib import Path

import yaml

from midogpp_thesis.cvae.diagnostics.mmd_kmm_router.bundle import REQUIRED_FILES
from midogpp_thesis.cvae.diagnostics.mmd_kmm_router.config import (
    load_mmd_kmm_router_config,
)
from midogpp_thesis.cvae.diagnostics.mmd_kmm_router.contracts import (
    CLASSIFIER_THREADS_PER_WORKER,
    CLASSIFIER_WORKERS,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    KERNEL_DEVICES,
    MAX_SOURCE_PREFIX_PER_CLASS,
    MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT,
    OUTPUT_ARTIFACT_ID,
    ROUTER_PREFIX_PER_CLASS,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.workspace.runtime import MidogppWorkspace


REPO = Path(__file__).resolve().parents[2]
CONFIG = (
    REPO
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_validation_mmd_kmm_router_v1.yaml"
)


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(REPO)
    workspace.validate()
    return workspace


def test_mmd_kmm_router_is_registered_as_terminal_stage90_diagnostic() -> None:
    workspace = _workspace()
    experiment = workspace.experiments[EXPERIMENT_ID]
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert experiment.runner_argv[-2:] == (
        "--artifact-root",
        f"output://{OUTPUT_ARTIFACT_ID}",
    )
    assert output.required_files == REQUIRED_FILES
    assert output.semantic_identities["fresh_evidence"] == "false"
    assert output.semantic_identities["routing_quality_claimed"] == "false"
    assert output.semantic_identities["promotion_eligible"] == "false"
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert "oracle_and_diagnostic_evidence" in output.forbidden_reuse


def test_mmd_kmm_validation_aliases_are_byte_exact_and_single_consumer() -> None:
    workspace = _workspace()
    pairs = (
        (
            VALIDATION_CACHE_ARTIFACT_ID,
            "midogpp_virchow2_uniform_b_v2_routing_validation_cache_seed42",
        ),
        (
            VALIDATION_MANIFEST_ARTIFACT_ID,
            "midogpp_source_inner_validation_manifest_v1",
        ),
    )
    for alias_id, original_id in pairs:
        alias = workspace.artifacts[alias_id]
        original = workspace.artifacts[original_id]
        assert alias.physical_path == original.physical_path
        assert alias.canonical_path == original.canonical_path
        assert alias.required_files == original.required_files
        assert alias.expected_file_hashes == original.expected_file_hashes
        assert alias.semantic_identities["alias_of_artifact_id"] == original_id
        assert alias.semantic_identities["fresh_evidence"] == "false"
        assert alias.semantic_identities["authorized_consumer_experiment_ids"] == EXPERIMENT_ID
        consumers = {
            experiment.experiment_id
            for experiment in workspace.experiments.values()
            if alias_id in experiment.input_artifact_ids
        }
        assert consumers == {EXPERIMENT_ID}


def test_mmd_kmm_config_freezes_workstation_budget_and_claim_firewall() -> None:
    config = load_mmd_kmm_router_config(CONFIG)
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert raw["proxy"]["source_prefix_per_class"] == MAX_SOURCE_PREFIX_PER_CLASS == 256
    assert raw["proxy"]["router_fit_prefix_per_class"] == ROUTER_PREFIX_PER_CLASS == 32
    assert tuple(raw["runtime"]["kernel_devices"]) == KERNEL_DEVICES
    assert raw["runtime"]["classifier_workers"] == CLASSIFIER_WORKERS == 4
    assert raw["runtime"]["classifier_threads_per_worker"] == CLASSIFIER_THREADS_PER_WORKER == 3
    assert raw["runtime"]["maximum_unique_classifier_fit_count"] == MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT == 162
    assert raw["protocol"]["support_labels_used"] is False
    assert raw["protocol"]["evaluation_embeddings_available_to_router"] is False
    assert raw["protocol"]["global_target_predictions_sealed_before_any_label_access"] is True
    assert raw["protocol"]["previous_stage90_router_or_utility_inputs_used"] is False
    for key in (
        "fresh_evidence",
        "fresh_confirmation",
        "routing_quality_claimed",
        "promotion_eligible",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_recipe_selection",
        "may_feed_deployable_selection",
    ):
        assert raw["claim_boundary"][key] is False


def test_mmd_kmm_cli_is_lazy_and_registered() -> None:
    args = build_parser().parse_args(
        [
            "mmd-kmm-router-diagnostic",
            "--config",
            "config.yaml",
            "--artifact-root",
            "output",
        ]
    )
    assert args.surface == "mmd-kmm-router-diagnostic"
