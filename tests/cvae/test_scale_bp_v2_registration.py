from __future__ import annotations

import hashlib
import json
from pathlib import Path

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.config import (
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.experiment_contracts import (
    validate_authorization_amendment,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    CLI_SURFACE,
    DIRECT_INPUT_ARTIFACT_IDS,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.source_snapshot import (
    build_source_snapshot_payload,
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
    / f"{STEM}_v2.yaml"
)
AMENDMENT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / f"{STEM}_ledger_amendment_v2.json"
)
SOURCE_MANIFEST = (
    ROOT
    / "src/midogpp_thesis/cvae/diagnostics/"
    "fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_"
    "boundary_projected_router_v2/source_manifest_v2.json"
)
V2_SCOPED_ARTIFACT_IDS = {
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
}


def test_v2_registration_is_exact_six_authorized_and_terminal_only() -> None:
    config = load_config(CONFIG)
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.status == "diagnostic"
    assert experiment.input_artifact_ids == DIRECT_INPUT_ARTIFACT_IDS
    assert len(DIRECT_INPUT_ARTIFACT_IDS) == len(set(DIRECT_INPUT_ARTIFACT_IDS)) == 6
    assert config.input_artifact_ids == DIRECT_INPUT_ARTIFACT_IDS
    assert config.execution_authorized is True
    assert config.consumed_test_reuse_authorized is True
    assert output.availability == "generated_on_run"
    assert output.evidence_label == (
        "AUTHORIZED_TERMINAL_CONSUMED_TEST_DIAGNOSTIC_PENDING_RUN"
    )
    assert output.semantic_identities["config_contract_hash"] == config.contract_hash
    assert output.semantic_identities["protocol_contract_hash"] == (
        config.protocol["protocol_hash"]
    )
    assert output.semantic_identities["fresh_evidence"] == "false"
    assert output.semantic_identities["routing_success_claimed"] == "false"
    assert output.semantic_identities["may_feed_another_experiment"] == "false"
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False


def test_v2_authorization_and_source_snapshot_are_canonical() -> None:
    config = load_config(CONFIG)
    source = build_source_snapshot_payload()
    checked_in_source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    amendment_text = AMENDMENT.read_text(encoding="utf-8")
    amendment = json.loads(amendment_text)

    assert checked_in_source == source
    assert amendment_text == json.dumps(amendment, indent=2, sort_keys=True) + "\n"
    assert hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() == (
        config.expected_authorization_amendment_sha256
    )
    validate_authorization_amendment(
        amendment,
        expected_source_manifest_sha256=source["manifest_sha256"],
        expected_source_tree_sha256=source["tree_sha256"],
        expected_source_member_count=source["member_count"],
    )
    assert amendment["authorization_exhausted"] is False
    assert amendment["implementation_did_not_launch_execution"] is True
    assert amendment["fresh_evidence"] is False
    assert amendment["may_feed_another_experiment"] is False


def test_v2_catalog_aliases_are_single_consumer_and_non_promotable() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    for artifact_id in V2_SCOPED_ARTIFACT_IDS - {OUTPUT_ARTIFACT_ID}:
        artifact = workspace.artifacts[artifact_id]
        identities = artifact.semantic_identities
        assert identities["authorized_consumer_experiment_ids"] == EXPERIMENT_ID
        assert identities["execution_authorized"] == "true"
        assert identities["consumed_test_reuse_authorized"] == "true"
        assert identities["fresh_evidence"] == "false"
        assert artifact.may_feed_recipe_selection is False
        assert artifact.may_feed_deployable_selection is False


def test_v2_cli_surface_has_mutation_free_dry_run_mode() -> None:
    args = build_parser().parse_args(
        [
            CLI_SURFACE,
            "--config",
            "config.yaml",
            "--artifact-root",
            "output",
            "--dry-run",
        ]
    )
    assert args.surface == CLI_SURFACE
    assert args.dry_run is True
