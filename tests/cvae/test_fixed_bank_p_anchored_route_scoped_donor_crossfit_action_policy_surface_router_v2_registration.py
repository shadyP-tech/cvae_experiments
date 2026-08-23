from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.config import (
    CONFIG_TOP_LEVEL,
    load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.experiment_contracts import (
    EXTERNAL_NEUTRAL_MODULE_SOURCE_POLICY,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256,
    EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT,
    EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    SOURCE_SNAPSHOT_SCOPE,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    V1_OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.identity import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.input_contracts import (
    source_snapshot_identity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.protocol import (
    V2_METHODOLOGICAL_DELTA_ROLE,
    V2_METHODOLOGICAL_DELTAS,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
STEM = (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "donor_crossfit_action_policy_surface_router"
)
SURFACE = (
    "fixed-bank-p-anchored-route-scoped-donor-crossfit-"
    "action-policy-surface-router-v2"
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
V1_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router.v1"
)
FAILED_RUNNER = (
    "{python}",
    "-c",
    "raise SystemExit('P-DCAPS v2 failed preterminally and cannot be "
    "recovered or rerun')",
)
EXPECTED_OUTPUT_MEMBERS = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "arrays/composed_probabilities.npz",
    "arrays/composed_probabilities.npz.manifest.json",
    "tables/preterminal_science.json",
    "manifests/preterminal_content_index.json",
    "reports/preterminal_fresh_process_attestation.json",
    "tables/terminal_result.json",
    "manifests/final_content_index.json",
    "reports/workstation_preflight.json",
    "reports/diagnostic_summary.json",
    "reports/label_capability_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/runtime_summary.json",
    "reports/final_fresh_process_attestation.json",
    "reports/validation_report.json",
    "reports/run_state.json",
)


def test_v2_is_failed_exhausted_exact_six_and_v1_stays_planned() -> None:
    config = (
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config(
            CONFIG
        )
    )
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    v1 = workspace.get_experiment(V1_EXPERIMENT_ID)

    assert len(CONFIG_TOP_LEVEL) == 9
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(INPUT_ARTIFACT_IDS) == len(set(INPUT_ARTIFACT_IDS)) == 6
    assert V1_OUTPUT_ARTIFACT_ID not in INPUT_ARTIFACT_IDS
    assert not any(value.endswith("_v1") for value in INPUT_ARTIFACT_IDS[2:])
    assert experiment.status == "failed"
    assert experiment.runnable is False
    assert experiment.runner_env == {}
    assert experiment.runner_argv == FAILED_RUNNER
    assert v1.status == "planned"
    assert v1.runnable is False

    assert config.execution_authorized is True
    assert config.protocol["execution_authorized"] is True
    assert config.runtime["execution_authorized"] is True
    assert config.claim_boundary["execution_authorized"] is True
    assert config.protocol["scientific_protocol_unchanged_from_v1"] is False
    assert config.protocol["scientific_method_changed_from_v1"] is True
    assert tuple(config.protocol["methodological_deltas"]) == (
        V2_METHODOLOGICAL_DELTAS
    )
    assert config.protocol["methodological_delta_role"] == (
        V2_METHODOLOGICAL_DELTA_ROLE
    )
    assert config.protocol[
        "methodological_deltas_are_terminal_consumed_test_only"
    ] is True
    assert config.protocol["methodological_deltas_create_fresh_evidence"] is False
    assert config.protocol["methodological_deltas_are_promotable"] is False
    assert config.claim_boundary["scientific_method_changed_from_v1"] is True
    assert tuple(config.claim_boundary["methodological_deltas"]) == (
        V2_METHODOLOGICAL_DELTAS
    )
    assert config.runtime[
        "worker_results_are_manifest_hashes_and_compact_offsets_only"
    ] is False
    assert config.runtime["worker_results_are_plain_pickle_safe_science_DTOs"] is True
    assert config.runtime[
        "outer_task_handles_both_posterior_controls_sequentially"
    ] is True
    assert config.runtime["nested_process_pools_forbidden"] is True
    assert config.protocol["response_denominators"] == (
        "derived_inside_lifecycle_from_support_plus_held"
    )
    assert config.action_library["endpoint_donor_prior_policy"] == (
        "ZERO_VECTOR_NO_FITTED_PRIOR"
    )
    assert config.action_library["minimum_effective_sample_size_per_class"] == 5.0
    assert output.availability == "workstation_failed_preterminal"
    assert output.required_files == EXPECTED_OUTPUT_MEMBERS
    assert output.evidence_label == "REJECTED"
    assert output.semantic_identities["config_contract_hash"] == config.contract_hash
    assert output.semantic_identities["protocol_contract_hash"] == (
        config.protocol["protocol_hash"]
    )
    assert output.semantic_identities["execution_authorized"] == "true"
    assert output.semantic_identities["original_execution_authorized"] == "true"
    assert output.semantic_identities["further_execution_authorized"] == "false"
    assert output.semantic_identities["authorization_exhausted"] == "true"
    assert output.semantic_identities["single_use_authorization_consumed"] == (
        "true"
    )
    assert output.semantic_identities["run_state_status"] == "FAILED"
    assert output.semantic_identities["failure_phase"] == (
        "FOUR_SPAWN_OUTER_H_WORKERS"
    )
    assert output.semantic_identities["run_state_hash"] == (
        "2c10b41ee3eb03c1b2ace3f9efb84d3fc2241e355f1271b450254ae8084c11a4"
    )
    assert output.semantic_identities["diagnostic_result_valid"] == "false"
    assert output.semantic_identities[
        "route_surfaces_and_pseudo_responses_complete"
    ] == "true"
    assert output.semantic_identities["pseudo_response_capabilities_opened"] == (
        "true"
    )
    assert output.semantic_identities["target_terminal_capability_opened"] == (
        "false"
    )
    assert output.semantic_identities["terminal_metrics_computed"] == "false"
    assert output.semantic_identities["consumed_test_reuse_authorized"] == "true"
    assert output.semantic_identities["execution_authorization_basis"] == (
        AUTHORIZATION_BASIS
    )
    assert output.semantic_identities["authorization_scope"] == AUTHORIZATION_SCOPE
    assert output.semantic_identities["fresh_evidence"] == "false"
    assert output.semantic_identities["routing_success_claimed"] == "false"
    assert output.semantic_identities["may_feed_another_experiment"] == "false"
    assert output.semantic_identities["scientific_protocol_unchanged_from_v1"] == (
        "false"
    )
    assert output.semantic_identities["scientific_method_changed_from_v1"] == "true"
    assert output.semantic_identities["methodological_delta_role"] == (
        V2_METHODOLOGICAL_DELTA_ROLE
    )
    assert output.semantic_identities["methodological_deltas"] == "|".join(
        V2_METHODOLOGICAL_DELTAS
    )
    assert output.semantic_identities[
        "methodological_deltas_create_fresh_evidence"
    ] == "false"
    assert output.semantic_identities["methodological_deltas_are_promotable"] == (
        "false"
    )
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False


def test_v2_amendment_is_hash_pinned_single_consumer_and_terminal_only() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    identities = MidogppWorkspace.load(ROOT).artifacts[
        LEDGER_AMENDMENT_ARTIFACT_ID
    ].semantic_identities

    assert AMENDMENT.read_text(encoding="utf-8") == (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    assert hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() == (
        EXPECTED_LEDGER_AMENDMENT_SHA256
    )
    assert payload["authorized_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert "registered_consumer_experiment_ids" not in payload
    assert payload["authorization_basis"] == AUTHORIZATION_BASIS
    assert payload["authorization_scope"] == AUTHORIZATION_SCOPE
    assert payload["execution_authorized"] is True
    assert payload["consumed_test_reuse_authorized"] is True
    assert payload["single_use_execution_identity"] is True
    assert payload["authorization_exhausted"] is False
    assert tuple(payload["direct_input_artifact_ids"]) == INPUT_ARTIFACT_IDS
    assert payload["direct_input_count"] == 6
    assert payload["v1_output_used"] is False
    assert payload["v1_amendment_used"] is False
    assert payload["v1_scratch_or_checkpoint_used"] is False
    assert payload["scientific_protocol_unchanged_from_v1"] is False
    assert payload["scientific_method_changed_from_v1"] is True
    assert tuple(payload["methodological_deltas"]) == V2_METHODOLOGICAL_DELTAS
    assert payload["methodological_delta_role"] == V2_METHODOLOGICAL_DELTA_ROLE
    assert payload["methodological_deltas_are_terminal_consumed_test_only"] is True
    assert payload["methodological_deltas_create_fresh_evidence"] is False
    assert payload["methodological_deltas_are_promotable"] is False
    assert identities["scientific_protocol_unchanged_from_v1"] == "false"
    assert identities["scientific_method_changed_from_v1"] == "true"
    assert identities["methodological_delta_role"] == V2_METHODOLOGICAL_DELTA_ROLE
    assert identities["methodological_deltas"] == "|".join(
        V2_METHODOLOGICAL_DELTAS
    )
    assert payload["previous_stage90_outputs_used"] is False
    assert payload["may_feed_another_experiment"] is False
    assert payload["terminal_decision"] == (
        "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
    )


def test_v2_fresh_aliases_use_authorized_consumer_fences() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    scoped = {
        artifact_id
        for artifact_id in workspace.artifacts
        if "donor_crossfit_action_policy_surface_router" in artifact_id
        and artifact_id.endswith("_v2")
    }
    assert scoped == {
        TEST_CACHE_ARTIFACT_ID,
        TEST_MANIFEST_ARTIFACT_ID,
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
        LEDGER_AMENDMENT_ARTIFACT_ID,
        OUTPUT_ARTIFACT_ID,
    }
    for artifact_id in scoped - {OUTPUT_ARTIFACT_ID}:
        identities = workspace.artifacts[artifact_id].semantic_identities
        assert identities["authorized_consumer_experiment_ids"] == EXPERIMENT_ID
        assert "registered_consumer_experiment_ids" not in identities

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


def test_v2_source_snapshot_is_exactly_bound() -> None:
    config = (
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config(
            CONFIG
        )
    )
    workspace = MidogppWorkspace.load(ROOT)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    amendment_artifact = workspace.artifacts[LEDGER_AMENDMENT_ARTIFACT_ID]
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    identities = output.semantic_identities
    amendment_identities = amendment_artifact.semantic_identities
    observed = source_snapshot_identity()

    assert EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT > 0
    assert len(EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256) == 64
    assert len(EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256) == 64
    assert identities["source_snapshot_manifest_sha256"] == (
        EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256
    )
    assert identities["source_snapshot_tree_sha256"] == (
        EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256
    )
    assert int(identities["source_snapshot_member_count"]) == (
        EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT
    )
    assert observed["source_snapshot_manifest_sha256"] == (
        EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256
    )
    assert observed["source_snapshot_tree_sha256"] == (
        EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256
    )
    assert observed["source_snapshot_member_count"] == (
        EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT
    )
    for payload in (amendment, amendment_identities):
        assert payload["source_snapshot_manifest_sha256"] == (
            EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256
        )
        assert payload["source_snapshot_tree_sha256"] == (
            EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256
        )
        assert int(payload["source_snapshot_member_count"]) == (
            EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT
        )
    assert config.protocol["source_snapshot_binding_required"] is True
    assert config.protocol["source_snapshot_excludes_pyc_and_cache"] is True
    assert config.protocol["source_snapshot_scope"] == SOURCE_SNAPSHOT_SCOPE
    assert config.protocol["external_neutral_module_source_policy"] == (
        EXTERNAL_NEUTRAL_MODULE_SOURCE_POLICY
    )
    assert identities["source_snapshot_scope"] == SOURCE_SNAPSHOT_SCOPE
    assert identities["external_neutral_module_source_policy"] == (
        EXTERNAL_NEUTRAL_MODULE_SOURCE_POLICY
    )
    assert amendment["source_snapshot_scope"] == SOURCE_SNAPSHOT_SCOPE
    assert amendment["external_neutral_module_source_policy"] == (
        EXTERNAL_NEUTRAL_MODULE_SOURCE_POLICY
    )


def test_v2_config_is_strict_and_path_independent(tmp_path: Path) -> None:
    canonical = (
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config(
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
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config(
            relocated
        )
    )
    assert loaded.contract_hash == canonical.contract_hash

    payload["policy_menu"]["action_response_model"]["alpha"] = 0.5
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="config section drifted"):
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config(
            drifted
        )


def test_v2_cli_surface_is_unique_and_rejects_exhausted_identity() -> None:
    parsed = cli.build_parser().parse_args(
        [SURFACE, "--config", str(CONFIG), "--artifact-root", "/tmp/pdcaps-v2"]
    )
    assert parsed.surface == SURFACE

    with pytest.raises(ProtocolError, match="authorization is exhausted"):
        cli.main(
            [
                SURFACE,
                "--config",
                str(CONFIG),
                "--artifact-root",
                "/tmp/pdcaps-v2",
            ]
        )
