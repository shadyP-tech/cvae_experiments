from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.config import (
    CONFIG_TOP_LEVEL,
    EXPERT_BANK_ARTIFACT_ID,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    NULLABLE_ADMISSION_STATISTICS_SCHEMA,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.identity import (
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    V2_EXECUTION_STATUS,
    V2_EXPERIMENT_ID,
    V2_OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.runner import (
    run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.source_seal import (
    EXPECTED_COMBINED_SOURCE_SEAL_SHA256,
    EXPECTED_V2_SOURCE_MANIFEST_SHA256,
    EXPECTED_V2_SOURCE_MEMBER_COUNT,
    EXPECTED_V2_SOURCE_TREE_SHA256,
    EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256,
    EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT,
    EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256,
    v2_base_source_root,
    v3_repair_source_root,
    validate_v2_base_source_seal,
    validate_v3_repair_source_seal,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


ROOT = Path(__file__).resolve().parents[2]
STEM = (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "donor_crossfit_action_policy_surface_router"
)
SURFACE = (
    "fixed-bank-p-anchored-route-scoped-donor-crossfit-"
    "action-policy-surface-router-v3"
)
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / f"{STEM}_v3.yaml"
)
AMENDMENT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / f"{STEM}_ledger_amendment_v3.json"
)
EXPECTED_OUTPUT_MEMBERS = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "reports/run_state.json",
)
SCOPED_ARTIFACT_IDS = {
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
}


def test_v3_is_exact_six_planned_non_authorized_and_v2_is_exhausted() -> None:
    config = (
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_config(
            CONFIG
        )
    )
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    predecessor = workspace.get_experiment(V2_EXPERIMENT_ID)

    assert CONFIG_TOP_LEVEL == {
        "experiment",
        "inputs",
        "protocol",
        "runtime",
        "claim_boundary",
    }
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(INPUT_ARTIFACT_IDS) == len(set(INPUT_ARTIFACT_IDS)) == 6
    assert INPUT_ARTIFACT_IDS[:2] == (
        EXPERT_BANK_ARTIFACT_ID,
        GENERATION_LOCK_ARTIFACT_ID,
    )
    assert EXPERT_BANK_ARTIFACT_ID == (
        "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
    )
    assert GENERATION_LOCK_ARTIFACT_ID == (
        "midogpp_output_uniform_b_v2_generation_lock_v1"
    )
    assert V2_OUTPUT_ARTIFACT_ID not in INPUT_ARTIFACT_IDS
    assert all(value.endswith("_v3") for value in INPUT_ARTIFACT_IDS[2:])
    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert predecessor.status == "failed"
    assert predecessor.runnable is False
    assert config.execution_authorized is False
    assert config.protocol["execution_authorized"] is False
    assert config.runtime["execution_authorized"] is False
    assert config.claim_boundary["execution_authorized"] is False
    assert config.protocol["v2_execution_status"] == V2_EXECUTION_STATUS
    assert config.protocol["v2_authorization_exhausted"] is True
    assert config.protocol["v2_retry_forbidden"] is True
    assert config.protocol["scientific_thresholds_changed_from_v2"] is False
    assert config.protocol["scientific_ordering_changed_from_v2"] is False
    assert config.protocol["nullable_statistic_schema"] == (
        NULLABLE_ADMISSION_STATISTICS_SCHEMA
    )
    assert output.availability == "planned_execution_not_authorized"
    assert output.evidence_label == "NEEDS_EVIDENCE_EXECUTION_NOT_AUTHORIZED"
    assert output.required_files == EXPECTED_OUTPUT_MEMBERS
    assert output.semantic_identities["config_contract_hash"] == (
        config.contract_hash
    )
    assert output.semantic_identities["protocol_contract_hash"] == (
        config.protocol["protocol_hash"]
    )
    assert output.semantic_identities["execution_authorized"] == "false"
    assert output.semantic_identities["implementation_authorizes_execution"] == (
        "false"
    )
    assert output.semantic_identities["consumed_test_reuse_authorized"] == (
        "false"
    )
    assert output.semantic_identities["v2_execution_status"] == (
        V2_EXECUTION_STATUS
    )
    assert output.semantic_identities["v2_output_used"] == "false"
    assert output.semantic_identities["routing_success_claimed"] == "false"
    assert output.semantic_identities["may_feed_another_experiment"] == "false"
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False


def test_v3_amendment_is_canonical_hash_pinned_and_non_authorizing() -> None:
    raw = AMENDMENT.read_text(encoding="utf-8")
    payload = json.loads(raw)
    identities = MidogppWorkspace.load(ROOT).artifacts[
        LEDGER_AMENDMENT_ARTIFACT_ID
    ].semantic_identities

    assert raw == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() == (
        EXPECTED_LEDGER_AMENDMENT_SHA256
    )
    assert identities["amendment_sha256"] == EXPECTED_LEDGER_AMENDMENT_SHA256
    assert payload["execution_authorized"] is False
    assert payload["consumed_test_reuse_authorized"] is False
    assert payload["authorized_consumer_experiment_ids"] == []
    assert payload["registered_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert tuple(payload["direct_input_artifact_ids"]) == INPUT_ARTIFACT_IDS
    assert payload["direct_input_count"] == 6
    assert payload["v2_execution_status"] == V2_EXECUTION_STATUS
    assert payload["v2_authorization_reused"] is False
    assert payload["v2_output_used"] is False
    assert payload["v2_amendment_used"] is False
    assert payload["v2_run_state_used"] is False
    assert payload["v2_label_capability_history_used"] is False
    assert payload["v2_scratch_or_checkpoint_used"] is False
    assert payload["mechanical_repair_only"] is True
    assert payload["scientific_protocol_unchanged_from_v2"] is True
    assert payload["scientific_method_changed_from_v2"] is False
    assert payload["nullable_admission_statistics_schema"] == (
        NULLABLE_ADMISSION_STATISTICS_SCHEMA
    )
    assert payload["undefined_statistic_gate_result"] == "FAIL_CLOSED_EXACT_P"
    assert payload["target_terminal_labels_may_open"] is False
    assert payload["may_feed_another_experiment"] is False


def test_v3_scoped_catalog_aliases_are_resolution_only() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    scoped = {
        artifact_id
        for artifact_id in workspace.artifacts
        if "donor_crossfit_action_policy_surface_router" in artifact_id
        and artifact_id.endswith("_v3")
    }
    assert scoped == SCOPED_ARTIFACT_IDS
    for artifact_id in scoped - {OUTPUT_ARTIFACT_ID}:
        artifact = workspace.artifacts[artifact_id]
        identities = artifact.semantic_identities
        assert identities["registered_consumer_experiment_ids"] == EXPERIMENT_ID
        assert identities["consumer_resolution_fence_only"] == "true"
        assert identities["execution_authorized"] == "false"
        assert identities["consumed_test_reuse_authorized"] == "false"
        assert "authorized_consumer_experiment_ids" not in identities
        assert artifact.may_feed_recipe_selection is False
        assert artifact.may_feed_deployable_selection is False

    assert workspace.artifacts[TEST_CACHE_ARTIFACT_ID].semantic_identities[
        "alias_of_artifact_id"
    ] == "midogpp_virchow2_uniform_b_v2_descriptive_test_cache_seed42"
    assert workspace.artifacts[TEST_MANIFEST_ARTIFACT_ID].semantic_identities[
        "alias_of_artifact_id"
    ] == "midogpp_dataset_contract_annotation_patch_v1"
    assert workspace.artifacts[
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID
    ].semantic_identities["alias_of_artifact_id"] == (
        "midogpp_uniform_b_test_consumption_ledger_v1"
    )


def test_v3_disjoint_source_seals_bind_unchanged_v2_and_complete_repair() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    base = validate_v2_base_source_seal()
    repair = validate_v3_repair_source_seal()

    assert v3_repair_source_root().parent == v2_base_source_root().parent
    assert v3_repair_source_root() != v2_base_source_root()
    assert not v3_repair_source_root().is_relative_to(v2_base_source_root())
    assert base["v2_base_source_snapshot_manifest_sha256"] == (
        EXPECTED_V2_SOURCE_MANIFEST_SHA256
    )
    assert base["v2_base_source_snapshot_tree_sha256"] == (
        EXPECTED_V2_SOURCE_TREE_SHA256
    )
    assert base["v2_base_source_snapshot_member_count"] == (
        EXPECTED_V2_SOURCE_MEMBER_COUNT
    )
    assert repair["v3_repair_source_snapshot_manifest_sha256"] == (
        EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256
    )
    assert repair["v3_repair_source_snapshot_tree_sha256"] == (
        EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256
    )
    assert repair["v3_repair_source_snapshot_member_count"] == (
        EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT
    )
    assert output.semantic_identities[
        "inherited_v2_base_source_manifest_sha256"
    ] == EXPECTED_V2_SOURCE_MANIFEST_SHA256
    assert output.semantic_identities[
        "inherited_v2_base_source_tree_sha256"
    ] == EXPECTED_V2_SOURCE_TREE_SHA256
    assert int(
        output.semantic_identities["inherited_v2_base_source_member_count"]
    ) == EXPECTED_V2_SOURCE_MEMBER_COUNT
    assert output.semantic_identities["v3_repair_source_manifest_sha256"] == (
        EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256
    )
    assert output.semantic_identities["v3_repair_source_tree_sha256"] == (
        EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256
    )
    assert int(output.semantic_identities["v3_repair_source_member_count"]) == (
        EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT
    )
    assert output.semantic_identities["combined_source_seal_hash"] == (
        EXPECTED_COMBINED_SOURCE_SEAL_SHA256
    )


def test_v3_config_is_strict_and_path_independent(tmp_path: Path) -> None:
    canonical = (
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_config(
            CONFIG
        )
    )
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["experiment"]["artifact_root"] = "relative/output"
    for key in (
        "expert_bank_root",
        "generation_lock_root",
        "test_cache_root",
        "test_manifest_path",
        "test_consumption_ledger_path",
        "ledger_amendment_path",
    ):
        payload["inputs"][key] = f"relative/{key}"
    relocated = tmp_path / "elsewhere" / "config.yaml"
    relocated.parent.mkdir(parents=True)
    relocated.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    loaded = (
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_config(
            relocated
        )
    )
    assert loaded.contract_hash == canonical.contract_hash

    payload["runtime"]["outer_cpu_worker_count"] = 8
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="config section drifted: runtime"):
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_config(
            drifted
        )


def test_v3_workspace_cli_and_runner_refuse_before_mutation(
    tmp_path: Path,
) -> None:
    workspace = MidogppWorkspace.load(ROOT)
    with pytest.raises(WorkspaceError, match="status='planned'"):
        workspace._render_run(  # noqa: SLF001 - exact pre-run refusal seam
            EXPERIMENT_ID,
            require_inputs=False,
            validate_workspace=True,
            include_all_declared_inputs=True,
        )

    parsed = cli.build_parser().parse_args(
        [SURFACE, "--config", str(CONFIG), "--artifact-root", "/tmp/pdcaps-v3"]
    )
    assert parsed.surface == SURFACE
    config = (
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_config(
            CONFIG
        )
    )
    direct_output = tmp_path / "direct" / "output"
    direct_scratch = tmp_path / "direct" / "scratch"
    with pytest.raises(ProtocolError, match="execution is not authorized"):
        run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3(
            config,
            artifact_root=direct_output,
            scratch_root=direct_scratch,
        )
    assert not (tmp_path / "direct").exists()

    cli_output = tmp_path / "cli" / "output"
    with pytest.raises(ProtocolError, match="execution is not authorized"):
        cli.main(
            [
                SURFACE,
                "--config",
                str(CONFIG),
                "--artifact-root",
                str(cli_output),
            ]
        )
    assert not (tmp_path / "cli").exists()


def test_v3_output_has_no_consumer() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    consumers = [
        experiment.experiment_id
        for experiment in workspace.experiments.values()
        if OUTPUT_ARTIFACT_ID in experiment.input_artifact_ids
    ]
    assert consumers == []
