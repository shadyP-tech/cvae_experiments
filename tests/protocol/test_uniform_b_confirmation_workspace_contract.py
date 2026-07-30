from __future__ import annotations

from pathlib import Path

from midogpp_thesis.real_features.classifier_reference.uniform_b_confirmation.workspace_binding import (
    EXPERIMENT_ID,
    INPUT_IDS,
    OUTPUT_ID,
    TEST_B_ID,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


def test_uniform_b_confirmation_is_stage90_hash_promoted_and_firewalled() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ID]
    test_cache = workspace.artifacts[TEST_B_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == INPUT_IDS
    assert output.canonical_path == (
        "artifacts/midogpp/90_oracles_and_diagnostics/"
        "uniform_b_v3_prospective_test_confirmation_v1/seed42"
    )
    assert output.evidence_label == "DIAGNOSTIC ONLY"
    assert set(output.required_files).issubset(output.expected_file_hashes)
    assert test_cache.evidence_label == "AUDIT_ONLY"
    assert set(test_cache.required_files).issubset(test_cache.expected_file_hashes)
    assert {
        "cvae_preservation_evidence",
        "routing_evidence",
        "synthetic_downstream_utility_evidence",
    }.issubset(output.forbidden_reuse)
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False


def test_uniform_b_confirmation_declares_only_a_b_and_frozen_discovery_inputs() -> None:
    workspace = MidogppWorkspace.load()
    experiment = workspace.get_experiment(EXPERIMENT_ID)

    assert all("11520" not in artifact_id for artifact_id in experiment.input_artifact_ids)
    assert experiment.runner_argv[-3:] == (
        "uniform-b-v3-confirmation",
        "--config",
        "{resolved_config}",
    )
    assert Path(experiment.config_path or "").name == (
        "uniform_b_v3_prospective_test_confirmation_v1.yaml"
    )
