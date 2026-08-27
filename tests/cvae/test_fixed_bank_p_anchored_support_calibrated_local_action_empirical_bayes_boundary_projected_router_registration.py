from __future__ import annotations

import hashlib
import json
from pathlib import Path

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.config import (
    CONFIG_TOP_LEVEL,
    load_support_calibrated_local_action_empirical_bayes_boundary_projected_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.experiment_contracts import (
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.identity import (
    CLI_SURFACE,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.source_seal import (
    validate_source_seal,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
STEM = (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_support_calibrated_"
    "local_action_empirical_bayes_boundary_projected_router"
)
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / f"{STEM}_v1.yaml"
)
AMENDMENT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / f"{STEM}_ledger_amendment_v1.json"
)
SCOPED_ARTIFACT_IDS = {
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
}


def test_scale_bp_is_exact_six_planned_and_non_authorized() -> None:
    config = (
        load_support_calibrated_local_action_empirical_bayes_boundary_projected_router_config(
            CONFIG
        )
    )
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    source_seal = validate_source_seal()

    assert len(INPUT_ARTIFACT_IDS) == len(set(INPUT_ARTIFACT_IDS)) == 6
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert config.execution_authorized is False
    assert config.protocol["execution_authorized"] is False
    assert config.runtime["execution_authorized"] is False
    assert config.claim_boundary["execution_authorized"] is False
    assert output.availability == "planned_execution_not_authorized"
    assert output.evidence_label == "NEEDS_EVIDENCE_EXECUTION_NOT_AUTHORIZED"
    assert output.semantic_identities["execution_authorized"] == "false"
    assert output.semantic_identities["consumed_test_reuse_authorized"] == "false"
    assert output.semantic_identities["target_terminal_labels_may_open"] == "false"
    assert output.semantic_identities["may_feed_another_experiment"] == "false"
    assert output.semantic_identities["nelbo_compatibility_claimed"] == "false"
    assert output.semantic_identities["config_contract_hash"] == config.contract_hash
    assert output.semantic_identities["protocol_contract_hash"] == (
        config.protocol["protocol_hash"]
    )
    assert output.semantic_identities["source_manifest_member"] == (
        source_seal.manifest_member
    )
    assert output.semantic_identities["source_manifest_sha256"] == (
        source_seal.manifest_sha256
    )
    assert output.semantic_identities["source_tree_sha256"] == (
        source_seal.tree_sha256
    )
    assert output.semantic_identities["source_member_count"] == str(
        source_seal.member_count
    )
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False


def test_scale_bp_config_is_closed_world_and_amendment_is_canonical() -> None:
    config = (
        load_support_calibrated_local_action_empirical_bayes_boundary_projected_router_config(
            CONFIG
        )
    )
    raw = AMENDMENT.read_text(encoding="utf-8")
    amendment = json.loads(raw)

    assert len(CONFIG_TOP_LEVEL) == 15
    assert config.protocol["route_local_support"] == "H_minus_c"
    assert config.protocol["support_fold_count"] == 4
    assert config.protocol["held_case_c_excluded_from_all_preterminal_fits"] is True
    assert config.protocol["support_labels_must_not_update_global_models"] is True
    assert config.protocol["may_feed_another_experiment"] is False
    assert config.protocol["nelbo_compatibility_claimed"] is False
    assert raw == json.dumps(amendment, indent=2, sort_keys=True) + "\n"
    assert hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() == (
        EXPECTED_LEDGER_AMENDMENT_SHA256
    )
    assert amendment["authorized_consumer_experiment_ids"] == []
    assert amendment["registered_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert tuple(amendment["direct_input_artifact_ids"]) == INPUT_ARTIFACT_IDS
    assert amendment["execution_authorized"] is False
    assert amendment["target_terminal_labels_may_open"] is False
    assert amendment["may_feed_another_experiment"] is False


def test_scale_bp_catalog_aliases_are_resolution_only() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    scoped = {
        artifact_id
        for artifact_id in workspace.artifacts
        if "support_calibrated_local_action_empirical_bayes_boundary_projected" in artifact_id
    }
    assert SCOPED_ARTIFACT_IDS.issubset(scoped)
    for artifact_id in SCOPED_ARTIFACT_IDS - {OUTPUT_ARTIFACT_ID}:
        artifact = workspace.artifacts[artifact_id]
        identities = artifact.semantic_identities
        assert identities["registered_consumer_experiment_ids"] == EXPERIMENT_ID
        assert identities["consumer_resolution_fence_only"] == "true"
        assert identities["execution_authorized"] == "false"
        assert identities["consumed_test_reuse_authorized"] == "false"
        assert "authorized_consumer_experiment_ids" not in identities
        assert artifact.may_feed_recipe_selection is False
        assert artifact.may_feed_deployable_selection is False


def test_scale_bp_cli_surface_is_registered() -> None:
    args = build_parser().parse_args(
        [CLI_SURFACE, "--config", "config.yaml", "--artifact-root", "output"]
    )
    assert args.surface == CLI_SURFACE
