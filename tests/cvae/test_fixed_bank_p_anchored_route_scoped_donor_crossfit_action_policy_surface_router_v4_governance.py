from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.execution_admission import (
    assert_v4_execution_authorized,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.experiment_contracts import (
    AUTHORIZED_INPUT_ROLES,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_SCHEMA_VERSION,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.identity import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.source_seal import (
    build_combined_source_seal_payload,
    validate_combined_source_seal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.workspace_manifest import (
    WORKSPACE_INPUT_MANIFEST_SCHEMA,
    validate_workspace_manifest_header,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "donor_crossfit_action_policy_surface_router_v4.yaml"
)
AMENDMENT = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "donor_crossfit_action_policy_surface_router_ledger_amendment_v4.json"
)
PACKAGE = ROOT / (
    "src/midogpp_thesis/cvae/diagnostics/"
    "fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_"
    "surface_router_v4"
)


def test_v4_registration_is_exact_six_authorized_terminal_only() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    registry = yaml.safe_load(
        (ROOT / "experiments/midogpp/registry.yaml").read_text(encoding="utf-8")
    )
    catalog = yaml.safe_load(
        (ROOT / "experiments/midogpp/artifact_catalog.yaml").read_text(
            encoding="utf-8"
        )
    )
    experiment = next(
        row for row in registry["experiments"] if row["experiment_id"] == EXPERIMENT_ID
    )
    output = next(
        row for row in catalog["artifacts"] if row["artifact_id"] == OUTPUT_ARTIFACT_ID
    )
    assert experiment["status"] == "diagnostic"
    assert tuple(experiment["input_artifact_ids"]) == INPUT_ARTIFACT_IDS
    assert len(set(experiment["input_artifact_ids"])) == 6
    assert config["experiment"]["execution_authorized"] is True
    assert config["protocol"]["consumed_test_reuse_authorized"] is True
    assert config["protocol"]["fresh_evidence"] is False
    assert config["protocol"]["may_feed_another_experiment"] is False
    assert config["claim_boundary"]["terminal_decision"] == (
        "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
    )
    assert output["availability"] == "generated_on_run"
    assert output["semantic_identities"]["fresh_evidence"] == "false"


def test_v4_accepts_the_actual_workspace_input_manifest_transport_schema() -> None:
    workspace = MidogppWorkspace.load()
    rendered = workspace._render_run(  # noqa: SLF001 - regression seam
        EXPERIMENT_ID,
        require_inputs=False,
        validate_workspace=True,
        include_all_declared_inputs=True,
    )
    payload = rendered.input_manifest

    assert payload["schema_version"] == WORKSPACE_INPUT_MANIFEST_SCHEMA
    assert WORKSPACE_INPUT_MANIFEST_SCHEMA == "midogpp_input_artifacts_v2"
    assert tuple(row["artifact_id"] for row in payload["input_artifacts"]) == tuple(
        sorted(INPUT_ARTIFACT_IDS)
    )
    validate_workspace_manifest_header(payload)


def test_v4_ledger_is_single_consumer_and_does_not_reuse_predecessor_state() -> None:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    assert amendment["schema_version"] == LEDGER_AMENDMENT_SCHEMA_VERSION
    assert amendment["authorized_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert amendment["direct_input_artifact_ids"] == list(INPUT_ARTIFACT_IDS)
    assert amendment["authorized_input_roles"] == list(AUTHORIZED_INPUT_ROLES)
    assert amendment["authorization_basis"] == AUTHORIZATION_BASIS
    assert amendment["authorization_scope"] == AUTHORIZATION_SCOPE
    assert amendment["execution_authorized"] is True
    assert amendment["consumed_test_reuse_authorized"] is True
    assert amendment["single_use_execution_identity"] is True
    assert amendment["authorization_exhausted"] is False
    assert amendment["v1_output_used"] is False
    assert amendment["prior_v2_output_used"] is False
    assert amendment["v3_output_used"] is False
    assert amendment["cross_run_recovery_used"] is False
    assert amendment["may_feed_another_experiment"] is False


def test_v4_import_boundary_forbids_exhausted_v2_and_v3_authority_modules() -> None:
    allowed_v3 = {
        "admission",
        "method_controls",
        "protocol",
        "routing",
        "source_seal",
    }
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            assert ".v2" not in node.module
            if "surface_router_v3" in node.module:
                assert node.module.rsplit(".", 1)[-1] in allowed_v3


def test_v4_combined_source_payload_keeps_three_disjoint_scopes() -> None:
    seal = validate_combined_source_seal()
    payload = build_combined_source_seal_payload()
    assert seal.v4_member_count == 41
    assert payload["source_scopes_are_disjoint"] is True
    assert payload["v2_base"]["member_count"] == 105
    assert payload["v3_nullable_admission_repair"]["member_count"] == 13
    assert payload["v4_executable_orchestration"]["member_count"] == 41


def test_v4_cli_surface_is_registered() -> None:
    args = build_parser().parse_args(
        [
            "fixed-bank-p-anchored-route-scoped-donor-crossfit-"
            "action-policy-surface-router-v4",
            "--config",
            "config.yaml",
            "--artifact-root",
            "output",
        ]
    )
    assert args.surface.endswith("action-policy-surface-router-v4")


def test_v4_admission_rejects_noncanonical_config_before_path_mutation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist"
    with pytest.raises(ProtocolError, match="exact canonical config loader"):
        assert_v4_execution_authorized(
            {"experiment_id": EXPERIMENT_ID, "artifact_root": output}
        )
    assert not output.exists()
