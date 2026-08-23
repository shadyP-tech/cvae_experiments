from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.identity import (
    ACTION_STRATA,
    DIRECT_INPUT_ROLES,
    METHOD_MENU,
    PRIMARY_METHOD_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.bundle import (
    PRETERMINAL_INDEX_MEMBER,
    build_closed_world_index,
    verify_closed_world_index,
    verify_index_payload_members,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.experiment_contracts import (
    INPUT_ARTIFACT_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.identity import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2 import (
    fresh_process_validation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.lineage import (
    build_six_input_binding,
    reconstruct_persisted_six_input_binding,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.lifecycle import (
    DurablePreterminalAttestation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.inventory import (
    ExpectedRouteInventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.outer_chunks import (
    persist_and_verify_outer_chunks,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.persistence import (
    FINAL_INDEXED_MEMBERS,
    FINAL_REPORT_MEMBERS,
    PRETERMINAL_ATTESTATION_MEMBER,
    PRETERMINAL_REQUIRED_MEMBERS,
    TERMINAL_RESULT_MEMBER,
    WORKSTATION_PREFLIGHT_MEMBER,
    persist_terminal_bundle,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.reports import (
    diagnostic_summary_payload,
    label_capability_report_payload,
    leakage_report_payload,
    publication_decision_payload,
    runtime_summary_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.run_state import (
    PHASE_ORDER,
    RUN_STATE_MEMBER,
    write_run_state,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.scratch import (
    OUTER_DIRECTORY,
    ScratchLease,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.terminal.contracts import (
    TerminalEvaluationResult,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.terminal.inference import (
    exact_shared_center_max_sign_flip,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.validation import (
    _validate_preterminal_lifecycle_audit,
    _validate_terminal_claim_boundary,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2 import (
    workspace_inputs,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.workspace_inputs import (
    validate_workspace_provenance,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64


def _provenance(root: str) -> dict[str, Mapping[str, object]]:
    return {
        artifact_id: {
            "artifact_id": artifact_id,
            "resolved_path": f"{root}/{artifact_id}",
            "exists": True,
            "semantic_identities": {
                "semantic_id": f"input-{ordinal}",
                "version": "frozen",
            },
            "file_integrity": {
                "schema_version": "fixture_integrity_v1",
                "member_sha256": canonical_hash((artifact_id, ordinal)),
            },
        }
        for ordinal, artifact_id in enumerate(INPUT_ARTIFACT_IDS)
    }


def _terminal_result() -> TerminalEvaluationResult:
    return TerminalEvaluationResult(
        method_rows=(
            {
                "method_id": "P_PROTECTED",
                "equal_center_bacc": 0.5,
            },
        ),
        center_rows=(
            {
                "method_id": "P_PROTECTED",
                "center": "0",
                "center_bacc": 0.5,
            },
        ),
        case_diagnostic_rows=(),
        selection_control={"selected_method_id": "P_PROTECTED"},
        router_diagnostics={"joint_safe_routed_rate": 0.0},
        preterminal_seal_hash=_HASH_A,
        label_identity_hash=_HASH_B,
    )


def _source_snapshot() -> dict[str, object]:
    return {
        "schema_version": "pdcaps_v2_source_snapshot_manifest_v1",
        "manifest_sha256": _HASH_A,
        "tree_sha256": _HASH_B,
        "member_count": 1,
    }


def _final_reports(
    result: TerminalEvaluationResult,
) -> dict[str, Mapping[str, object]]:
    lifecycle = {
        "lifecycle_hash": _HASH_C,
        "phase": "TERMINAL",
        "durable_preterminal_attestation_hash": _HASH_B,
        "target_labels_can_change_preterminal_decisions": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "raw_labels_persisted": False,
    }
    return {
        "reports/diagnostic_summary.json": diagnostic_summary_payload(result),
        "reports/label_capability_report.json": (
            label_capability_report_payload(lifecycle)
        ),
        "reports/leakage_report.json": leakage_report_payload(
            input_binding={"binding_hash": _HASH_C},
            pre_gpu_firewall={"status": "PASS"},
            lifecycle_audit=lifecycle,
            source_snapshot=_source_snapshot(),
        ),
        "reports/publication_decision.json": publication_decision_payload(
            result
        ),
        "reports/runtime_summary.json": runtime_summary_payload(
            preflight={
                "status": "PASS",
                "schema_version": "pdcaps_v2_workstation_preflight_v1",
            },
            physical_surface_hash=_HASH_A,
            route_runtime_hash=_HASH_B,
            pseudo_runtime_hash=_HASH_C,
            outer_execution={
                "execution_mode": "spawn",
                "worker_count": 4,
                "runtime_hash": _HASH_A,
                "science_hash": _HASH_B,
            },
            outer_chunk_manifest={
                "manifest_hash": _HASH_C,
                "written_atomically": True,
                "verified_after_write": True,
            },
        ),
    }


def _seed_preterminal_members(root: Path) -> None:
    for member in PRETERMINAL_REQUIRED_MEMBERS:
        target = root / member
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".npz":
            target.write_bytes(b"pdcaps-v2-test-npz\n")
        else:
            atomic_json(target, {"schema_version": "fixture_v1"})
    atomic_json(
        root / PRETERMINAL_INDEX_MEMBER,
        {"schema_version": "fixture_preterminal_index_v1"},
    )
    atomic_json(
        root / PRETERMINAL_ATTESTATION_MEMBER,
        {"schema_version": "fixture_attestation_v1"},
    )
    atomic_json(
        root / WORKSTATION_PREFLIGHT_MEMBER,
        {"schema_version": "fixture_preflight_v1", "status": "PASS"},
    )


def _assert_self_hash(payload: Mapping[str, object], hash_key: str) -> None:
    body = {key: value for key, value in payload.items() if key != hash_key}
    assert payload[hash_key] == canonical_hash(body)


def test_v2_six_input_binding_is_path_free_and_ordered() -> None:
    config = SimpleNamespace(
        input_artifact_ids=INPUT_ARTIFACT_IDS,
        protocol={"protocol_hash": _HASH_A},
    )
    workstation = build_six_input_binding(
        config, _provenance("/home/stud/spark/workspace")
    )
    local = build_six_input_binding(
        config, _provenance("/Users/reviewer/local-workspace")
    )

    assert workstation == local
    assert tuple(role for role, _ in workstation.artifact_ids_by_role) == (
        DIRECT_INPUT_ROLES
    )
    assert len(workstation.artifact_ids_by_role) == 6
    persisted = json.dumps(workstation.to_payload(), sort_keys=True)
    assert "/home/" not in persisted
    assert "/Users/" not in persisted
    assert "resolved_path" not in persisted
    _assert_self_hash(workstation.to_payload(), "binding_hash")


def test_v2_persisted_input_poison_changes_reconstructed_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _provenance("/validated/workspace")
    payload = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": (
            "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
            "route_scoped_donor_crossfit_action_policy_surface_router.v2"
        ),
        "stage": "90_oracles_and_diagnostics",
        "claim_scope": "diagnostic_only",
        "selection_used_target_eval_artifacts": False,
        "input_artifacts": [rows[key] for key in sorted(rows)],
    }
    paths = {
        artifact_id: Path(str(row["resolved_path"]))
        for artifact_id, row in rows.items()
    }
    config = SimpleNamespace(
        input_artifact_ids=INPUT_ARTIFACT_IDS,
        protocol={"protocol_hash": _HASH_A},
        expert_bank_root=paths[INPUT_ARTIFACT_IDS[0]],
        generation_lock_root=paths[INPUT_ARTIFACT_IDS[1]],
        test_cache_root=paths[INPUT_ARTIFACT_IDS[2]],
        test_manifest_path=paths[INPUT_ARTIFACT_IDS[3]] / "manifest.csv",
        test_consumption_ledger_path=(
            paths[INPUT_ARTIFACT_IDS[4]]
            / "reports/test_consumption_ledger.json"
        ),
        ledger_amendment_path=(
            paths[INPUT_ARTIFACT_IDS[5]] / "amendment.json"
        ),
    )

    class _WorkspaceFixture:
        def validate(self) -> None:
            return None

        def _render_run(self, *_args, **_kwargs):
            return SimpleNamespace(input_manifest=payload)

    monkeypatch.setattr(
        workspace_inputs.MidogppWorkspace,
        "load",
        lambda *_args, **_kwargs: _WorkspaceFixture(),
    )
    path = tmp_path / "provenance/input_artifacts.json"
    atomic_json(path, payload)
    before = reconstruct_persisted_six_input_binding(tmp_path, config)
    validate_workspace_provenance(tmp_path, config)

    poisoned = json.loads(json.dumps(payload))
    poisoned["input_artifacts"][0]["semantic_identities"]["version"] = (
        "poisoned"
    )
    atomic_json(path, poisoned)
    after = reconstruct_persisted_six_input_binding(tmp_path, config)
    assert after.binding_hash != before.binding_hash
    with pytest.raises(ProtocolError, match="provenance replay differs"):
        validate_workspace_provenance(tmp_path, config)


@pytest.mark.parametrize(
    ("status", "phase", "exhausted"),
    (
        ("RUNNING", "BEGIN", False),
        ("COMPLETE", "COMPLETE", True),
        ("FAILED", "BEGIN", True),
    ),
)
def test_v2_run_state_binds_hashes_and_exhausts_terminal_states(
    tmp_path: Path,
    status: str,
    phase: str,
    exhausted: bool,
) -> None:
    root = tmp_path / status.lower()
    payload = write_run_state(
        root,
        config_hash=_HASH_A,
        status=status,
        phase=phase,
        bound_hashes={"z_role": _HASH_C, "a_role": _HASH_B},
        error_class="FixtureError" if status == "FAILED" else None,
        error="fixture failure" if status == "FAILED" else None,
    )

    assert read_json(root / RUN_STATE_MEMBER) == payload
    assert payload["authorization_exhausted"] is exhausted
    assert payload["cross_run_recovery_allowed"] is False
    assert payload["terminal_recovery_allowed"] is False
    assert payload["scratch_recovery_used"] is False
    assert tuple(payload["bound_hashes"]) == ("a_role", "z_role")
    assert payload["bound_hashes"] == {
        "a_role": _HASH_B,
        "z_role": _HASH_C,
    }
    _assert_self_hash(payload, "state_hash")


def test_v2_run_state_is_monotonic_and_terminal_is_irreversible(
    tmp_path: Path,
) -> None:
    root = tmp_path / "monotonic"
    write_run_state(
        root,
        config_hash=_HASH_A,
        status="RUNNING",
        phase=PHASE_ORDER[0],
    )
    write_run_state(
        root,
        config_hash=_HASH_A,
        status="RUNNING",
        phase=PHASE_ORDER[1],
    )
    with pytest.raises(ProtocolError, match="not authenticated"):
        write_run_state(
            root,
            config_hash=_HASH_A,
            status="RUNNING",
            phase=PHASE_ORDER[0],
        )
    write_run_state(
        root,
        config_hash=_HASH_A,
        status="FAILED",
        phase=PHASE_ORDER[1],
    )
    with pytest.raises(ProtocolError, match="not authenticated"):
        write_run_state(
            root,
            config_hash=_HASH_A,
            status="FAILED",
            phase=PHASE_ORDER[1],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("experiment_id", "poisoned-experiment"),
        ("config_hash", _HASH_C),
        ("process_id", -1),
        ("authorization_scope", "poisoned-scope"),
        ("unexpected_recovery_token", True),
    ),
)
def test_v2_run_state_authenticates_fixed_attempt_identity(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = tmp_path / field
    write_run_state(
        root,
        config_hash=_HASH_A,
        status="RUNNING",
        phase=PHASE_ORDER[0],
        bound_hashes={"input": _HASH_B},
    )
    path = root / RUN_STATE_MEMBER
    poisoned = read_json(path)
    poisoned[field] = value
    poisoned_base = {
        key: item for key, item in poisoned.items() if key != "state_hash"
    }
    poisoned["state_hash"] = canonical_hash(poisoned_base)
    atomic_json(path, poisoned)

    with pytest.raises(ProtocolError, match="not authenticated"):
        write_run_state(
            root,
            config_hash=_HASH_A,
            status="RUNNING",
            phase=PHASE_ORDER[1],
            bound_hashes={"input": _HASH_B},
        )


def test_v2_run_state_rejects_hash_poison_and_requires_append_only_bindings(
    tmp_path: Path,
) -> None:
    poisoned_root = tmp_path / "state-hash"
    write_run_state(
        poisoned_root,
        config_hash=_HASH_A,
        status="RUNNING",
        phase=PHASE_ORDER[0],
        bound_hashes={"input": _HASH_B},
    )
    path = poisoned_root / RUN_STATE_MEMBER
    poisoned = read_json(path)
    poisoned["state_hash"] = "0" * 64
    atomic_json(path, poisoned)
    with pytest.raises(ProtocolError, match="not authenticated"):
        write_run_state(
            poisoned_root,
            config_hash=_HASH_A,
            status="RUNNING",
            phase=PHASE_ORDER[1],
            bound_hashes={"input": _HASH_B},
        )

    append_only_root = tmp_path / "append-only"
    write_run_state(
        append_only_root,
        config_hash=_HASH_A,
        status="RUNNING",
        phase=PHASE_ORDER[0],
        bound_hashes={"input": _HASH_B},
    )
    write_run_state(
        append_only_root,
        config_hash=_HASH_A,
        status="RUNNING",
        phase=PHASE_ORDER[1],
        bound_hashes={"input": _HASH_B, "preflight": _HASH_C},
    )
    with pytest.raises(ProtocolError, match="not append-only"):
        write_run_state(
            append_only_root,
            config_hash=_HASH_A,
            status="RUNNING",
            phase=PHASE_ORDER[2],
            bound_hashes={"preflight": _HASH_C},
        )


def test_v2_reports_bind_source_anchors_and_forbid_promotion() -> None:
    result = _terminal_result()
    summary = diagnostic_summary_payload(result)
    publication = publication_decision_payload(result)
    leakage = leakage_report_payload(
        input_binding={"binding_hash": _HASH_C},
        pre_gpu_firewall={"status": "PASS"},
        lifecycle_audit={"lifecycle_hash": _HASH_C},
        source_snapshot=_source_snapshot(),
    )

    assert leakage["source_snapshot_manifest_sha256"] == _HASH_A
    assert leakage["source_snapshot_tree_sha256"] == _HASH_B
    assert leakage["exact_six_inputs"] is True
    assert leakage["target_labels_can_change_preterminal_decisions"] is False
    assert summary["publication_status"] == PUBLICATION_STATUS
    assert summary["terminal_decision"] == TERMINAL_DECISION
    assert summary["routing_success_claimed"] is False
    assert summary["promotion_allowed"] is False
    assert publication["fresh_evidence"] is False
    assert publication["routing_success_claimed"] is False
    assert publication["downstream_utility_claimed"] is False
    assert publication["promotion_eligible"] is False
    assert publication["may_feed_another_experiment"] is False
    assert publication["may_feed_stage50"] is False
    assert publication["may_feed_stage60"] is False
    assert publication["may_feed_stage70"] is False
    assert publication["may_feed_recipe_selection"] is False
    for report in (summary, publication, leakage):
        _assert_self_hash(report, "report_hash")


def test_v2_terminal_claim_boundary_reconstructs_selection_and_rejects_poison() -> None:
    center_rows = [
        {
            "method_id": method,
            "target_center": center,
            "center_bacc": (
                0.5
                if method == "P_PROTECTED"
                else 0.5 + 0.001 * (METHOD_MENU.index(method) + 1)
            ),
            "center_bacc_delta_vs_P": (
                0.0
                if method == "P_PROTECTED"
                else 0.001 * (METHOD_MENU.index(method) + 1)
            ),
            "center_brier_delta_vs_P": (
                0.1
                if method == PRIMARY_METHOD_ID and center == CENTERS[1]
                else 0.0
            ),
            "center_log_loss_delta_vs_P": 0.0,
        }
        for method in METHOD_MENU
        for center in CENTERS
    ]
    center_metrics = {
        method: {
            center: next(
                row
                for row in center_rows
                if row["method_id"] == method
                and row["target_center"] == center
            )
            for center in CENTERS
        }
        for method in METHOD_MENU
    }
    case_rows = [
        {"method_id": PRIMARY_METHOD_ID, "case_harmed_vs_P": False},
        {"method_id": PRIMARY_METHOD_ID, "case_harmed_vs_P": True},
    ]
    payload = {
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "selection_control": exact_shared_center_max_sign_flip(center_metrics),
        "router_diagnostics": {
            "schema_version": "pdcaps_v2_terminal_router_diagnostics_v1",
            "terminal_labels_changed_preterminal_decisions": False,
            "nonzero_route_count_is_not_success": True,
            "descriptive_only": True,
            "formal_claim_authorized": False,
            "routed_center_count": 2,
            "joint_safe_routed_center_count": 1,
            "joint_safe_routed_policy_rate": 0.5,
            "action_expected_vs_realized_midrank_spearman_by_stratum": [
                {
                    "family": family,
                    "direction": direction,
                    "pair_count": 0,
                    "midrank_spearman": None,
                }
                for family, direction in ACTION_STRATA
            ],
            "policy_expected_vs_realized_midrank_spearman": None,
            "policy_expected_vs_realized_pair_count": len(CENTERS),
            "normalized_endpoint_oracle_gap_definition": (
                "best_sealed_case_action_minus_primary_over_best_minus_worst_"
                "sealed_case_action"
            ),
            "normalized_endpoint_oracle_gap_defined_case_count": 0,
            "mean_normalized_endpoint_oracle_gap": None,
            "center_action_frequencies": [
                {
                    "family": family,
                    "direction": direction,
                    "selected_count": int(index == 0),
                }
                for index, (family, direction) in enumerate(ACTION_STRATA)
            ],
            "primary_case_harm_count": 1,
            "primary_case_harm_rate": 0.5,
        },
    }
    _validate_terminal_claim_boundary(
        payload,
        center_rows=center_rows,
        case_rows=case_rows,
        primary_routed_centers=CENTERS[:2],
        primary_selected_action_count=1,
    )

    poisons = (
        ("publication_status", "FRESH_CONFIRMATORY"),
        ("terminal_decision", "PROMOTE"),
        ("selection_control.formal_claim_authorized", True),
        ("router_diagnostics.formal_claim_authorized", True),
        ("router_diagnostics.routing_success_claimed", True),
        ("router_diagnostics.routed_center_count", 1),
        ("router_diagnostics.primary_case_harm_count", 0),
    )
    for dotted, value in poisons:
        poisoned = deepcopy(payload)
        if "." in dotted:
            parent, child = dotted.split(".", 1)
            poisoned[parent][child] = value
        else:
            poisoned[dotted] = value
        with pytest.raises(ProtocolError, match="claim boundary drifted"):
            _validate_terminal_claim_boundary(
                poisoned,
                center_rows=center_rows,
                case_rows=case_rows,
                primary_routed_centers=CENTERS[:2],
                primary_selected_action_count=1,
            )


def test_v2_preterminal_lifecycle_audit_rejects_rehashed_poison() -> None:
    inventory = ExpectedRouteInventory.focused_fixture(
        (
            ("0", "case-a", "sample-a"),
            ("1", "case-b", "sample-b"),
        )
    )
    surface = {
        "schema_version": "pdcaps_action_surface_set_v1",
        "expected_inventory_hash": inventory.inventory_hash,
        "physical_surface_hash": _HASH_A,
        "control_surface_seals": [
            ["IDENTITY", _HASH_A],
            ["WITHIN_CASE_CYCLIC_SHIFT", _HASH_B],
        ],
        "route_inventory_seal_hashes": [
            ["IDENTITY", _HASH_B],
            ["WITHIN_CASE_CYCLIC_SHIFT", _HASH_C],
        ],
        "pseudo_labels_used": False,
        "target_labels_used": False,
        "surface_set_seal_hash": _HASH_C,
    }
    lifecycle_base = {
        "schema_version": "pdcaps_label_lifecycle_v2",
        "phase": "PRETERMINAL_ATTESTED",
        "protocol_hash": _HASH_C,
        "expected_outer_centers": list(inventory.centers),
        "expected_inventory": inventory.to_payload(),
        "action_surface_set": surface,
        "action_surface_seal_hash": _HASH_A,
        "pseudo_response_surface_count": 2 * inventory.pseudo_route_count,
        "preterminal_seal_hash": _HASH_B,
        "terminal_centers_opened": [],
        "firewall_hash": _HASH_C,
        "target_labels_can_change_preterminal_decisions": False,
        "support_class_count_scope_count": inventory.case_count,
        "response_denominators_derived_inside_label_lifecycle": True,
        "durable_terminal_attestation_required": True,
        "durable_preterminal_attestation_hash": None,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "raw_labels_persisted": False,
    }
    lifecycle = {
        **lifecycle_base,
        "lifecycle_hash": canonical_hash(lifecycle_base),
    }
    arguments = {
        "config": SimpleNamespace(protocol={"protocol_hash": _HASH_C}),
        "canonical_inventory": inventory,
        "surface_set": surface,
        "preterminal_seal_hash": _HASH_B,
    }
    _validate_preterminal_lifecycle_audit(
        {"lifecycle_audit": lifecycle}, **arguments
    )

    poisons = (
        ("phase", "PSEUDO_RESPONSE"),
        ("pseudo_response_surface_count", 0),
        ("support_class_count_scope_count", 0),
        ("response_denominators_derived_inside_label_lifecycle", False),
        ("durable_terminal_attestation_required", False),
        ("durable_preterminal_attestation_hash", _HASH_A),
        ("raw_labels_persisted", True),
    )
    for key, value in poisons:
        poisoned_base = {**lifecycle_base, key: value}
        poisoned = {
            **poisoned_base,
            "lifecycle_hash": canonical_hash(poisoned_base),
        }
        with pytest.raises(ProtocolError, match="label lifecycle drifted"):
            _validate_preterminal_lifecycle_audit(
                {"lifecycle_audit": poisoned}, **arguments
            )


def test_v2_durable_attestation_requires_two_distinct_validator_results() -> None:
    with pytest.raises(ProtocolError, match="two fresh validators"):
        DurablePreterminalAttestation(
            preterminal_seal_hash=_HASH_A,
            validator_process_ids=(1001, 1002),
            validator_result_hashes=(_HASH_B, _HASH_B),
            durable_bundle_hash=_HASH_C,
        )


def test_v2_fresh_validation_requires_two_sequential_cuda_free_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], Mapping[str, str]]] = []
    process_ids = iter((41001, 41002, 42001, 42002))

    def fake_run(command, **kwargs):
        phase = command[command.index("--phase") + 1]
        process_id = next(process_ids)
        checks = (
            {
                "status": "PASS",
                "preterminal_seal_hash": _HASH_A,
                "content_index_hash": _HASH_B,
            }
            if phase == "preterminal"
            else {"status": "PASS", "terminal_result_hash": _HASH_C}
        )
        payload = {
            "process_id": process_id,
            "validation_phase": phase,
            "checks": checks,
        }
        calls.append((list(command), dict(kwargs["env"])))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            stderr="",
        )

    monkeypatch.setattr(fresh_process_validation.subprocess, "run", fake_run)
    durable = (
        fresh_process_validation.require_two_fresh_preterminal_validations(
            tmp_path
        )
    )
    final = fresh_process_validation.require_two_fresh_final_validations(
        tmp_path
    )

    assert durable.validator_process_ids == (41001, 41002)
    assert len(set(durable.validator_result_hashes)) == 2
    assert final["validator_process_ids"] == [42001, 42002]
    assert len(calls) == 4
    for command, environment in calls:
        assert "--worker" in command
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        assert all(
            environment[key] == value
            for key, value in fresh_process_validation.THREAD_ENVIRONMENT.items()
        )


@dataclass(frozen=True)
class _OuterChunkFixture:
    outer_center: str
    ordinal: int

    @property
    def result_hash(self) -> str:
        return canonical_hash((self.outer_center, self.ordinal))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v2_outer_runtime_result_fixture_v1",
            "outer_center": self.outer_center,
            "ordinal": self.ordinal,
            "result_hash": self.result_hash,
        }


def test_v2_outer_chunks_roundtrip_atomically_and_refuse_recovery(
    tmp_path: Path,
) -> None:
    scratch_root = (tmp_path / "scratch").resolve()
    scratch_root.mkdir()
    lease = ScratchLease(scratch_root, "artifact_parent")
    rows = (
        _OuterChunkFixture("0", 0),
        _OuterChunkFixture("1", 1),
    )

    manifest = persist_and_verify_outer_chunks(lease, rows)  # type: ignore[arg-type]
    directory = scratch_root / OUTER_DIRECTORY
    members = tuple(sorted(directory.glob("*.json")))

    assert manifest["chunk_count"] == 2
    assert manifest["written_atomically"] is True
    assert manifest["verified_after_write"] is True
    assert manifest["scratch_only"] is True
    assert manifest["cross_run_recovery_allowed"] is False
    assert len(members) == 2
    assert not tuple(directory.glob("*.tmp"))
    for member, row in zip(members, rows, strict=True):
        chunk = read_json(member)
        assert chunk["outer_result"] == row.to_payload()
        assert chunk["outer_result_hash"] == row.result_hash
        _assert_self_hash(chunk, "chunk_hash")
    original_bytes = tuple(member.read_bytes() for member in members)

    with pytest.raises(ProtocolError, match="refuses outer chunk recovery"):
        persist_and_verify_outer_chunks(lease, rows)  # type: ignore[arg-type]
    assert tuple(member.read_bytes() for member in members) == original_bytes


def test_v2_terminal_bundle_requires_and_indexes_exact_report_inventory(
    tmp_path: Path,
) -> None:
    result = _terminal_result()
    reports = _final_reports(result)
    assert set(reports) == set(FINAL_REPORT_MEMBERS)

    missing_root = tmp_path / "missing-report"
    _seed_preterminal_members(missing_root)
    missing = dict(reports)
    missing.pop(FINAL_REPORT_MEMBERS[-1])
    with pytest.raises(ProtocolError, match="final report inventory drifted"):
        persist_terminal_bundle(missing_root, result, final_reports=missing)
    assert not (missing_root / TERMINAL_RESULT_MEMBER).exists()

    poisoned_root = tmp_path / "poisoned-report"
    _seed_preterminal_members(poisoned_root)
    poisoned = {key: dict(value) for key, value in reports.items()}
    publication = dict(poisoned["reports/publication_decision.json"])
    publication["promotion_eligible"] = True
    publication_base = {
        key: value
        for key, value in publication.items()
        if key != "report_hash"
    }
    publication["report_hash"] = canonical_hash(publication_base)
    poisoned["reports/publication_decision.json"] = publication
    with pytest.raises(ProtocolError, match="terminal-only report contract"):
        persist_terminal_bundle(
            poisoned_root, result, final_reports=poisoned
        )
    assert not (poisoned_root / TERMINAL_RESULT_MEMBER).exists()

    extra_claim_root = tmp_path / "extra-promotion-claim"
    _seed_preterminal_members(extra_claim_root)
    extra_claim = {key: dict(value) for key, value in reports.items()}
    publication = dict(extra_claim["reports/publication_decision.json"])
    publication["alternate_promotion_claim"] = True
    publication_base = {
        key: value for key, value in publication.items() if key != "report_hash"
    }
    publication["report_hash"] = canonical_hash(publication_base)
    extra_claim["reports/publication_decision.json"] = publication
    with pytest.raises(ProtocolError, match="final report hash drifted"):
        persist_terminal_bundle(
            extra_claim_root, result, final_reports=extra_claim
        )
    assert not (extra_claim_root / TERMINAL_RESULT_MEMBER).exists()

    inconsistent_root = tmp_path / "inconsistent-summary"
    _seed_preterminal_members(inconsistent_root)
    inconsistent = {key: dict(value) for key, value in reports.items()}
    summary = dict(inconsistent["reports/diagnostic_summary.json"])
    summary["method_rows"] = [
        {**dict(summary["method_rows"][0]), "equal_center_bacc": 0.75}
    ]
    summary_base = {
        key: value for key, value in summary.items() if key != "report_hash"
    }
    summary["report_hash"] = canonical_hash(summary_base)
    inconsistent["reports/diagnostic_summary.json"] = summary
    with pytest.raises(ProtocolError, match="terminal-only report contract"):
        persist_terminal_bundle(
            inconsistent_root, result, final_reports=inconsistent
        )
    assert not (inconsistent_root / TERMINAL_RESULT_MEMBER).exists()

    root = tmp_path / "complete"
    _seed_preterminal_members(root)
    receipt = persist_terminal_bundle(root, result, final_reports=reports)
    index = verify_closed_world_index(
        root, phase="final", expected_members=FINAL_INDEXED_MEMBERS
    )
    indexed = {str(row["member"]) for row in index["members"]}
    expected = {
        *PRETERMINAL_REQUIRED_MEMBERS,
        PRETERMINAL_INDEX_MEMBER,
        PRETERMINAL_ATTESTATION_MEMBER,
        WORKSTATION_PREFLIGHT_MEMBER,
        TERMINAL_RESULT_MEMBER,
        *FINAL_REPORT_MEMBERS,
    }

    assert indexed == expected
    assert index["member_count"] == len(expected)
    assert receipt["terminal_result_hash"] == result.result_hash
    assert receipt["final_content_index_hash"] == index["content_index_hash"]
    assert read_json(root / TERMINAL_RESULT_MEMBER) == result.to_payload()
    for member in FINAL_REPORT_MEMBERS:
        assert read_json(root / member) == reports[member]


def test_v2_preterminal_index_is_exact_but_allows_post_index_attestation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "preterminal-index"
    _seed_preterminal_members(root)
    index = build_closed_world_index(
        root,
        required_members=PRETERMINAL_REQUIRED_MEMBERS,
        phase="preterminal",
    )
    verified = verify_closed_world_index(
        root,
        phase="preterminal",
        expected_members=PRETERMINAL_REQUIRED_MEMBERS,
    )
    assert verified == index

    omitted = dict(index)
    omitted["members"] = list(index["members"][:-1])
    omitted["member_count"] = len(omitted["members"])
    omitted["observed_controlled_members"] = [
        row["member"]
        for row in omitted["members"]
        if "/" in str(row["member"])
    ]
    omitted_base = {
        key: value for key, value in omitted.items() if key != "content_index_hash"
    }
    omitted["content_index_hash"] = canonical_hash(omitted_base)
    with pytest.raises(ProtocolError, match="indexed member inventory drifted"):
        verify_index_payload_members(
            root,
            omitted,
            phase="preterminal",
            expected_members=PRETERMINAL_REQUIRED_MEMBERS,
        )

    unexpected = root / "tables/unexpected.json"
    atomic_json(unexpected, {"schema_version": "poison_v1"})
    with pytest.raises(ProtocolError, match="unexpected controlled member"):
        verify_closed_world_index(
            root,
            phase="preterminal",
            expected_members=PRETERMINAL_REQUIRED_MEMBERS,
        )
