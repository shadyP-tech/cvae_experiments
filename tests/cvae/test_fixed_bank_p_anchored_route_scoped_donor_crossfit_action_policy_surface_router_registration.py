from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.config import (
    CONFIG_TOP_LEVEL,
    load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.execution_admission import (
    assert_execution_authorized,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.experiment_contracts import (
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.identity import (
    EXPERIMENT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.validation.protocol import (
    validate_no_sibling_imports,
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
    "action-policy-surface-router"
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


def test_registration_is_exact_six_planned_and_not_runnable() -> None:
    config = (
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config(
            CONFIG
        )
    )
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert len(CONFIG_TOP_LEVEL) == 9
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(INPUT_ARTIFACT_IDS) == len(set(INPUT_ARTIFACT_IDS)) == 6
    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert output.availability == "planned_execution_not_authorized"
    assert output.semantic_identities["config_contract_hash"] == config.contract_hash
    assert output.semantic_identities["protocol_contract_hash"] == (
        config.protocol["protocol_hash"]
    )
    assert output.semantic_identities["execution_authorized"] == "false"
    assert output.semantic_identities["consumed_test_reuse_authorized"] == "false"
    assert output.semantic_identities["may_feed_another_experiment"] == "false"
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert config.execution_authorized is False
    assert config.protocol["execution_authorized"] is False
    assert config.runtime["execution_authorized"] is False
    assert config.claim_boundary["execution_authorized"] is False
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


def test_config_is_strict_and_path_independent(tmp_path: Path) -> None:
    canonical = (
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config(
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
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config(
            relocated
        )
    )
    assert loaded.contract_hash == canonical.contract_hash

    payload["policy_menu"]["action_response_model"]["alpha"] = 0.5
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="config section drifted"):
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config(
            drifted
        )


def test_amendment_is_hash_pinned_non_authorizing_and_exact_six() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))

    assert hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() == (
        EXPECTED_LEDGER_AMENDMENT_SHA256
    )
    assert payload["execution_authorized"] is False
    assert payload["consumed_test_reuse_authorized"] is False
    assert payload["authorized_consumer_experiment_ids"] == []
    assert payload["registered_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert tuple(payload["direct_input_artifact_ids"]) == INPUT_ARTIFACT_IDS
    assert payload["direct_input_count"] == 6
    assert len(set(payload["direct_input_artifact_ids"])) == 6
    assert payload["previous_stage90_outputs_used"] is False
    assert payload["previous_stage90_amendments_used"] is False
    assert payload["may_feed_another_experiment"] is False


def test_all_five_scoped_catalog_entries_are_non_authorizing() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    scoped = {
        artifact_id
        for artifact_id in workspace.artifacts
        if "donor_crossfit_action_policy_surface_router" in artifact_id
        and artifact_id.endswith("_v1")
    }

    assert scoped == SCOPED_ARTIFACT_IDS
    for artifact_id in scoped:
        artifact = workspace.artifacts[artifact_id]
        assert artifact.semantic_identities["execution_authorized"] == "false"
        assert "authorized_consumer_experiment_ids" not in artifact.semantic_identities
        if artifact_id == OUTPUT_ARTIFACT_ID:
            assert "registered_consumer_experiment_ids" not in artifact.semantic_identities
        else:
            assert artifact.semantic_identities[
                "registered_consumer_experiment_ids"
            ] == EXPERIMENT_ID
            assert artifact.semantic_identities[
                "consumer_resolution_fence_only"
            ] == "true"
            assert artifact.semantic_identities[
                "consumed_test_reuse_authorized"
            ] == "false"
        assert artifact.may_feed_recipe_selection is False
        assert artifact.may_feed_deployable_selection is False


def test_direct_admission_rejects_before_output_or_scratch_write(
    tmp_path: Path,
) -> None:
    config = (
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config(
            CONFIG
        )
    )
    output = tmp_path / "must-not-exist" / "output"
    scratch = tmp_path / "must-not-exist" / "scratch"

    with pytest.raises(ProtocolError, match="execution is not authorized"):
        assert_execution_authorized(
            config,
            artifact_root=output,
            scratch_root=scratch,
        )
    assert not output.exists()
    assert not scratch.exists()
    assert not (tmp_path / "must-not-exist").exists()


def test_workspace_refuses_planned_run_before_runner() -> None:
    workspace = MidogppWorkspace.load(ROOT)

    with pytest.raises(WorkspaceError, match="status='planned'"):
        workspace._render_run(  # noqa: SLF001 - exact pre-run refusal seam
            EXPERIMENT_ID,
            require_inputs=False,
            validate_workspace=True,
            include_all_declared_inputs=True,
        )


def test_cli_surface_parses_and_remains_import_light() -> None:
    parsed = build_parser().parse_args(
        [SURFACE, "--config", str(CONFIG), "--artifact-root", "/tmp/pdcaps"]
    )
    assert parsed.surface == SURFACE


def test_output_has_no_registered_consumer() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    consumers = [
        experiment.experiment_id
        for experiment in workspace.experiments.values()
        if OUTPUT_ARTIFACT_ID in experiment.input_artifact_ids
    ]
    assert consumers == []


def test_package_imports_no_diagnostic_sibling() -> None:
    report = validate_no_sibling_imports()
    assert report["status"] == "PASS"
    assert report["diagnostic_sibling_import_count"] == 0
