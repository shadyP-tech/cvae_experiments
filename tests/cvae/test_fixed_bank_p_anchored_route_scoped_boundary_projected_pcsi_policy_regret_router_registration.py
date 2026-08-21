from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.bundle import (
    REQUIRED_FILES,
    assert_closed_world,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.config import (
    load_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.experiment_contracts import (
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.constants import (
    QUARANTINED_V1_EXPERIMENT_ID,
    QUARANTINED_V1_OUTPUT_ARTIFACT_ID,
    REPAIR_CODE_COMMIT,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.protocol import (
    build_frozen_protocol,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.workspace_inputs import (
    validate_active_workspace_binding,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file
from midogpp_thesis.workspace.runtime import MidogppWorkspace
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.workstation import (
    assert_runtime,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "boundary_projected_pcsi_policy_regret_router_v2.yaml"
)
AMENDMENT = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "boundary_projected_pcsi_policy_regret_router_ledger_amendment_v2.json"
)
PACKAGE = ROOT / (
    "src/midogpp_thesis/cvae/diagnostics/"
    "fixed_bank_p_anchored_route_scoped_boundary_projected_"
    "pcsi_policy_regret_router"
)


def test_registration_is_fresh_fenced_runnable_and_terminal_only() -> None:
    config = load_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router_config(
        CONFIG
    )
    protocol = build_frozen_protocol()
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert config.contract_hash == "aa1fd1d4b63b2404"
    assert protocol.protocol_hash == (
        "e9da22f3909cd68d8e2bc1cfda727de5167ea93e6ca7aa2e6d466dc9e7f2b85a"
    )
    assert sha256_file(AMENDMENT) == EXPECTED_LEDGER_AMENDMENT_SHA256
    assert experiment.status == "diagnostic"
    assert experiment.runnable
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(INPUT_ARTIFACT_IDS) == len(set(INPUT_ARTIFACT_IDS)) == 6
    assert output.required_files == REQUIRED_FILES
    assert len(REQUIRED_FILES) == len(set(REQUIRED_FILES)) == 67
    assert output.semantic_identities["execution_authorized"] == "true"
    assert output.semantic_identities["mechanical_repair_only"] == "true"
    assert output.semantic_identities["quarantined_v1_output_used"] == "false"
    assert output.semantic_identities[
        "transport_identity_level_route_noninterference_proven"
    ] == "true"
    assert output.semantic_identities["may_feed_another_experiment"] == "false"
    assert "oracle_and_diagnostic_evidence" in output.forbidden_reuse
    assert config.claim_boundary["fresh_evidence"] is False
    assert config.claim_boundary["mechanical_repair_only"] is True
    assert config.claim_boundary["scientific_method_changed_from_v1"] is False
    assert config.claim_boundary["quarantined_v1_output_used"] is False
    assert config.claim_boundary["routing_success_claimed"] is False
    assert config.claim_boundary["may_feed_another_experiment"] is False
    assert_runtime(config.runtime)
    assert validate_active_workspace_binding(config)["status"] == "PASS"

    failed_v1_id = EXPERIMENT_ID.removesuffix(".v2") + ".v1"
    failed_v1 = workspace.get_experiment(failed_v1_id)
    failed_output = workspace.artifacts[
        OUTPUT_ARTIFACT_ID.removesuffix("_v2") + "_v1"
    ]
    assert failed_v1.status == "failed"
    assert failed_v1.runnable is False
    assert failed_output.evidence_label == "REJECTED"
    assert failed_output.semantic_identities["quarantined"] == "true"
    assert failed_output.semantic_identities["recoverable"] == "false"


def test_v2_amendment_binds_repair_and_rejects_v1_state() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))

    assert payload["authorized_repair_code_commit"] == REPAIR_CODE_COMMIT
    assert payload["quarantined_v1_experiment_id"] == QUARANTINED_V1_EXPERIMENT_ID
    assert (
        payload["quarantined_v1_output_artifact_id"]
        == QUARANTINED_V1_OUTPUT_ARTIFACT_ID
    )
    assert payload["scientific_protocol_unchanged_from_v1"] is True
    assert payload["scientific_method_changed_from_v1"] is False
    assert payload["quarantined_v1_output_used"] is False
    assert payload["quarantined_v1_scratch_or_checkpoint_used"] is False
    assert payload["prior_partial_label_capability_state_used"] is False
    assert QUARANTINED_V1_OUTPUT_ARTIFACT_ID not in INPUT_ARTIFACT_IDS
    assert all(value.endswith("_v2") for value in INPUT_ARTIFACT_IDS[2:])


def test_registered_bundle_inventory_is_closed_world(tmp_path: Path) -> None:
    for member in REQUIRED_FILES:
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert_closed_world(tmp_path)
    (tmp_path / "foreign.bin").touch()
    with pytest.raises(ProtocolError, match="closed-world drifted"):
        assert_closed_world(tmp_path)


def test_cli_surface_is_registered_lazily() -> None:
    surface = (
        "fixed-bank-p-anchored-route-scoped-boundary-projected-"
        "pcsi-policy-regret-router"
    )
    parsed = build_parser().parse_args(
        [surface, "--config", str(CONFIG), "--artifact-root", "/tmp/pcsi-racr"]
    )
    assert parsed.surface == surface


def test_config_rejects_scientific_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["protocol"]["envelope"] = "median_only"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="config section drifted"):
        load_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router_config(
            path
        )


def test_input_fence_rejects_blocked_predecessor() -> None:
    config = load_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router_config(
        CONFIG
    )
    assert_input_fence(config)
    poisoned = replace(
        config,
        test_cache_root=Path(
            "/tmp/fixed_bank_p_anchored_boundary_projected_"
            "pcsi_policy_regret_router/cache"
        ),
    )
    with pytest.raises(ProtocolError, match="predecessor diagnostic input"):
        assert_input_fence(poisoned)


def test_scientific_package_imports_no_diagnostic_sibling() -> None:
    forbidden = "midogpp_thesis.cvae.diagnostics."
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert not str(node.module or "").startswith(forbidden)
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(forbidden) for alias in node.names)
