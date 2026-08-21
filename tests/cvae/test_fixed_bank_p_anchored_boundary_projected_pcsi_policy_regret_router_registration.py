from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.bundle import (
    REQUIRED_FILES,
    assert_closed_world,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.config import (
    load_p_anchored_boundary_projected_pcsi_policy_regret_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.experiment_contracts import (
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.protocol import (
    build_frozen_protocol,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.workspace_inputs import (
    validate_active_workspace_binding,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_boundary_projected_"
    "pcsi_policy_regret_router_v1.yaml"
)
AMENDMENT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_boundary_projected_"
    "pcsi_policy_regret_router_ledger_amendment_v1.json"
)


def test_registration_is_exact_six_input_terminal_only() -> None:
    config = load_p_anchored_boundary_projected_pcsi_policy_regret_router_config(
        CONFIG
    )
    protocol = build_frozen_protocol()
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert config.contract_hash == "82f65ec3ad9f3d3d"
    assert protocol.protocol_hash == (
        "222a4ec08039330a94b35a43c7c73039ad4a17b1a10f0646d032c18938e607ab"
    )
    assert sha256_file(AMENDMENT) == EXPECTED_LEDGER_AMENDMENT_SHA256
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    assert amendment["transport_identity_level_route_noninterference_required"] is True
    assert amendment["transport_identity_level_route_noninterference_proven"] is False
    assert amendment["transport_authorization_valid"] is False
    assert amendment["execution_authorized"] is False
    assert amendment["transport_protocol_status"] == (
        "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK"
    )
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(INPUT_ARTIFACT_IDS) == len(set(INPUT_ARTIFACT_IDS)) == 6
    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert experiment.claim_scope == "diagnostic_only"
    assert config.claim_boundary["may_feed_another_experiment"] is False
    assert config.claim_boundary["routing_success_claimed"] is False
    assert "oracle_and_diagnostic_evidence" in output.forbidden_reuse
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert output.required_files == REQUIRED_FILES
    assert len(REQUIRED_FILES) == len(set(REQUIRED_FILES))
    expected_transport = {
        "transport_semantics": "support_conditioned_endpoint_reconstructed_P_B_I_R",
        "transport_endpoint_support_scope": "endpoint_target_T_minus_held_case_c",
        "transport_actual_source_prior_scope": "q_not_in_endpoint_target_T_or_source_e",
        "transport_donor_source_prior_scope": (
            "q_not_in_outer_H_or_endpoint_target_T_or_source_e"
        ),
        "transport_source_prior_labels_used_upstream": True,
        "transport_route_local_support_labels_used_upstream": True,
        "transport_held_case_evaluation_capability_used_directly": False,
        "transport_pseudo_evaluation_capability_used_directly": False,
        "transport_terminal_evaluation_capability_used_directly": False,
        "transport_label_free_claim": False,
        "transport_uses_pre_equivalence_endpoint_crossing_rates": True,
        "transport_screens_sealed_before_pseudo_evaluation_capability_open": True,
        "transport_screens_sealed_before_terminal_evaluation_capability_open": True,
        "transport_identity_level_route_noninterference_required": True,
        "transport_identity_level_route_noninterference_proven": False,
        "transport_authorization_valid": False,
        "transport_protocol_status": "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK",
    }
    assert {
        key: protocol.payload[key] for key in expected_transport
    } == expected_transport
    assert "transport_uses_labels" not in protocol.payload
    assert "transport_uses_pre_equivalence_physical_crossing_rates" not in (
        protocol.payload
    )
    assert config.policy_menu["transport"]["semantics"] == expected_transport[
        "transport_semantics"
    ]
    assert config.policy_menu["transport"]["label_free_claim"] is False
    assert config.policy_menu["transport"]["authorization_valid"] is False
    assert config.claim_boundary["transport_label_free_claim"] is False
    assert config.claim_boundary["execution_authorized"] is False
    expected_catalog_transport = {
        key: str(value).lower() if isinstance(value, bool) else value
        for key, value in expected_transport.items()
    }
    assert {
        key: output.semantic_identities[key] for key in expected_catalog_transport
    } == expected_catalog_transport


def test_registered_bundle_inventory_is_closed_world(tmp_path: Path) -> None:
    for member in REQUIRED_FILES:
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert_closed_world(tmp_path)

    (tmp_path / "unregistered.txt").touch()
    with pytest.raises(ProtocolError, match="closed-world drifted"):
        assert_closed_world(tmp_path)


def test_cli_surface_is_registered_lazily() -> None:
    surface = (
        "fixed-bank-p-anchored-boundary-projected-pcsi-policy-regret-router"
    )
    parsed = build_parser().parse_args(
        [surface, "--config", str(CONFIG), "--artifact-root", "/tmp/pcsi-parc"]
    )
    assert parsed.surface == surface


def test_config_rejects_scientific_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["protocol"]["policy_regret_correction"] = "scalarized"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError, match="config section drifted"):
        load_p_anchored_boundary_projected_pcsi_policy_regret_router_config(path)


def test_config_rejects_legacy_label_free_physical_transport_claim(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["protocol"]["transport_uses_labels"] = False
    payload["protocol"][
        "transport_uses_pre_equivalence_physical_crossing_rates"
    ] = True
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError, match="config section drifted"):
        load_p_anchored_boundary_projected_pcsi_policy_regret_router_config(path)


def test_planned_workspace_refuses_before_runner() -> None:
    workspace = MidogppWorkspace.load(ROOT)

    with pytest.raises(WorkspaceError, match="status='planned'"):
        workspace._render_run(  # noqa: SLF001 - exact pre-run refusal seam
            EXPERIMENT_ID,
            require_inputs=False,
            validate_workspace=True,
            include_all_declared_inputs=True,
        )


def test_direct_runner_workspace_binding_is_protocol_blocked() -> None:
    config = load_p_anchored_boundary_projected_pcsi_policy_regret_router_config(
        CONFIG
    )

    with pytest.raises(
        ProtocolError, match="identity-level route noninterference is unproved"
    ):
        validate_active_workspace_binding(config)


def test_exact_workspace_resolved_config_is_loadable(tmp_path: Path) -> None:
    workspace = MidogppWorkspace.load(ROOT)
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    resolved_payload = workspace.resolve_value(
        payload,
        require_inputs=False,
        used_inputs=set(),
    )
    path = tmp_path / "config.resolved.yaml"
    path.write_text(yaml.safe_dump(resolved_payload, sort_keys=False), encoding="utf-8")

    resolved = load_p_anchored_boundary_projected_pcsi_policy_regret_router_config(
        path
    )
    assert resolved.artifact_root.as_posix().endswith(
        "artifacts/midogpp/90_oracles_and_diagnostics/"
        "uniform_b_v2_consumed_test_fixed_bank_p_anchored_boundary_projected_"
        "pcsi_policy_regret_router/v1"
    )
    assert resolved.contract_hash == "82f65ec3ad9f3d3d"


@pytest.mark.parametrize(
    "fragment",
    (
        "fixed_bank_p_anchored_crossfit_sample_influence_router",
        "fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router",
        "fixed_bank_loo_nested_donor_endpoint_regret_router",
    ),
)
def test_input_fence_rejects_predecessor_diagnostic_paths(fragment: str) -> None:
    config = load_p_anchored_boundary_projected_pcsi_policy_regret_router_config(
        CONFIG
    )
    assert_input_fence(config)
    poisoned = replace(
        config,
        test_cache_root=Path(f"/tmp/{fragment}/cache"),
    )
    with pytest.raises(ProtocolError, match="predecessor diagnostic input"):
        assert_input_fence(poisoned)
