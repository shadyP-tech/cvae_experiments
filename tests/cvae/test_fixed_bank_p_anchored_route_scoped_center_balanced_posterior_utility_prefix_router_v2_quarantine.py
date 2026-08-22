from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router import (
    v2_quarantine_audit as quarantine_audit_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.bundle import (
    PRETERMINAL_SCIENTIFIC_MEMBERS,
    write_content_index,
    write_preterminal_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.config import (
    load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.config_payloads import (
    canonical_action_library_payload,
    canonical_claim_boundary_payload,
    canonical_evaluation_payload,
    canonical_policy_menu_payload,
    canonical_runtime_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.constants import (
    EXPECTED_OUTER_PLAN_COUNT,
    EXPECTED_PSEUDO_ROUTE_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.fresh_process_validation import (
    THREAD_ENVIRONMENT,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.preterminal_gate import (
    persist_preterminal_validation_seal,
    preterminal_validation_checks_payload,
    preterminal_validation_report_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.protocol import (
    FROZEN_PROTOCOL_HASH,
    frozen_protocol_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.reports import (
    protocol_manifest_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.terminal_access_journal import (
    persist_terminal_label_access_intent,
    persist_terminal_label_access_opened_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.v2_terminal_failure_quarantine import (
    V2_FINAL_PERSISTENCE_ORDER,
    V2_FINAL_PHASE,
    V2_TERMINAL_FAILURE_SCRATCH_DIRECTORIES,
    V2_TERMINAL_PERSISTENCE_ORDER,
    V2_TERMINAL_PHASE,
    audit_failed_v2_terminal_or_final_for_quarantine,
    quarantine_failed_v2_terminal_or_final,
)
from midogpp_thesis.cvae.protocol import ProtocolError


_CONFIG_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_center_"
    "balanced_posterior_utility_prefix_router_v2.yaml"
)
_TIMESTAMP = "20260822T235959Z"
_SOURCE_BYTES = {
    "arrays/frozen_source_streams.npy": b"v2-source-array\x00\x01",
    "manifests/frozen_source_stream_index.json": b'{"index":"v2"}\n',
    "manifests/frozen_source_stream_lock.json": b'{"lock":"v2"}\n',
}
_SCRATCH_TO_ARTIFACT = {
    "source_generation/arrays/frozen_source_streams.npy": (
        "arrays/frozen_source_streams.npy"
    ),
    "source_generation/manifests/frozen_source_stream_index.json": (
        "manifests/frozen_source_stream_index.json"
    ),
    "source_generation/manifests/frozen_source_stream_lock.json": (
        "manifests/frozen_source_stream_lock.json"
    ),
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_config(root: Path) -> object:
    raw = yaml.safe_load(_CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    raw["experiment"]["artifact_root"] = str(root)
    raw["protocol"] = frozen_protocol_payload()
    raw["action_library"] = canonical_action_library_payload()
    raw["policy_menu"] = canonical_policy_menu_payload()
    raw["evaluation"] = canonical_evaluation_payload()
    raw["runtime"] = canonical_runtime_payload()
    raw["claim_boundary"] = canonical_claim_boundary_payload()
    for key in (
        "expert_bank_root",
        "generation_lock_root",
        "test_cache_root",
        "test_manifest_path",
        "test_consumption_ledger_path",
        "ledger_amendment_path",
    ):
        raw["inputs"][key] = str(root.parent / "inputs" / key)
    path = root / "config.resolved.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config(
        path
    )


def _preterminal_capability(plan_hash: str) -> dict[str, object]:
    event_count = (
        9 * 8
        + 9 * 8 * 7
        + EXPECTED_OUTER_PLAN_COUNT
        + EXPECTED_PSEUDO_ROUTE_COUNT
    )
    events = [
        {"role": f"sealed_preterminal_{ordinal}", "raw_labels_persisted": False}
        for ordinal in range(event_count)
    ]
    return {
        "schema_version": "fixed_bank_cbpupr_label_access_audit_v1",
        "plan_seal_hash": plan_hash,
        "event_count": len(events),
        "events": events,
        "target_candidate_seal_complete": True,
        "pre_evaluation_seal_complete": True,
        "pseudo_evaluation_route_count": EXPECTED_PSEUDO_ROUTE_COUNT,
        "calibration_seal_complete": True,
        "decision_count": 4 * EXPECTED_OUTER_PLAN_COUNT,
        "aggregate_seal_complete": True,
        "terminal_opened": False,
        "raw_labels_persisted": False,
        "audit_hash": canonical_hash(events),
    }


def _preterminal_attestation(checks: dict[str, object]) -> dict[str, object]:
    children = []
    for ordinal, process_id in enumerate((1001, 1002), start=1):
        worker = {
            "process_id": process_id,
            "validation_phase": "preterminal",
            "checks": checks,
        }
        children.append(
            {
                "ordinal": ordinal,
                "process_id": process_id,
                "exit_code": 0,
                "result_hash": canonical_hash(worker),
            }
        )
    payload = {
        "schema_version": (
            "fixed_bank_cbpupr_preterminal_fresh_process_attestation_v1"
        ),
        "status": "PASS",
        "validation_phase": "preterminal",
        "fresh_python_process_count": 2,
        "independent_fresh_python_processes": True,
        "process_launches_sequential": True,
        "cuda_visible_devices": "",
        "worker_thread_environment": THREAD_ENVIRONMENT,
        "parent_process_id": 1000,
        "child_process_ids": [1001, 1002],
        "child_process_results": children,
        "reconstructed_checks_exactly_equal": True,
        "reconstructed_checks_hash": canonical_hash(checks),
        "validator_entrypoint": "validate_preterminal_bundle",
        "terminal_opened": False,
    }
    return {**payload, "attestation_hash": canonical_hash(payload)}


def _failed_state(phase: str) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_cbpupr_run_state_v1",
        "status": "FAILED",
        "phase": phase,
        "error": "injected terminal persistence failure",
        "error_class": "RuntimeError",
        "updated_at_utc": "2026-08-22T23:59:59+00:00",
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
    }


def _write_bundle(
    root: Path,
    *,
    phase: str,
    terminal_prefix_length: int,
    final_prefix_length: int = 0,
) -> None:
    config = _write_config(root)
    for relative in PRETERMINAL_SCIENTIFIC_MEMBERS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative != "config.resolved.yaml":
            path.write_bytes(
                _SOURCE_BYTES.get(relative, f"sealed::{relative}\n".encode())
            )
    provenance = {
        artifact_id: {"artifact_id": artifact_id}
        for artifact_id in config.input_artifact_ids
    }
    _write_json(
        root / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            config,
            protocol_hash=FROZEN_PROTOCOL_HASH,
            provenance=provenance,
            cache_binding_hash="b" * 64,
            pre_gpu_firewall={
                "status": "PASS",
                "repair_source_manifest_validated": True,
                "repair_source_manifest_sha256": config.protocol[
                    "repair_source_manifest_sha256"
                ],
                "repair_source_tree_sha256": config.protocol[
                    "repair_source_tree_sha256"
                ],
                "repair_source_member_count": config.protocol[
                    "repair_source_member_count"
                ],
            },
        ),
    )
    plan_hash = "a" * 64
    _write_json(root / "manifests/outer_plan_seal.json", {"seal_hash": plan_hash})
    _write_json(
        root / "reports/preterminal_label_capability_report.json",
        _preterminal_capability(plan_hash),
    )
    _write_json(root / "reports/run_state.json", _failed_state(phase))
    (root / ".run.lock").write_bytes(b"pid=5432\n")
    content = write_preterminal_content_index(root)
    checks = preterminal_validation_checks_payload(
        config_contract_hash=config.contract_hash,
        protocol_contract_hash=FROZEN_PROTOCOL_HASH,
        content_index_hash=str(content["content_index_hash"]),
        outer_route_count=EXPECTED_OUTER_PLAN_COUNT,
        target_posterior_model_fit_count=(
            EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
        ),
        pseudo_posterior_reference_count=2 * EXPECTED_PSEUDO_ROUTE_COUNT,
        preterminal_hash="c" * 64,
    )
    attestation = _preterminal_attestation(checks)
    _write_json(
        root / "reports/preterminal_fresh_process_attestation.json", attestation
    )
    report = preterminal_validation_report_payload(checks, attestation)
    _write_json(root / "reports/preterminal_validation_report.json", report)
    persist_preterminal_validation_seal(
        root, checks=checks, attestation=attestation, report=report
    )
    _write_json(root / "reports/run_state.json", _failed_state(phase))

    terminal_prefix = V2_TERMINAL_PERSISTENCE_ORDER[:terminal_prefix_length]
    if terminal_prefix:
        intent = persist_terminal_label_access_intent(
            root, expected_checks=checks
        )
    if len(terminal_prefix) >= 2:
        persist_terminal_label_access_opened_receipt(
            root, intent=intent, labels=(None,) * 9_928
        )
    for relative in terminal_prefix[2:]:
        _write_json(root / relative, {"sealed": relative})
    if final_prefix_length:
        write_content_index(root)
        for relative in V2_FINAL_PERSISTENCE_ORDER[1:final_prefix_length]:
            _write_json(root / relative, {"sealed": relative})


def _write_scratch(scratch: Path) -> None:
    for relative in V2_TERMINAL_FAILURE_SCRATCH_DIRECTORIES:
        (scratch / relative).mkdir(parents=True, exist_ok=True)
    for relative, artifact_relative in _SCRATCH_TO_ARTIFACT.items():
        (scratch / relative).write_bytes(_SOURCE_BYTES[artifact_relative])


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    artifact_parent = tmp_path / "artifacts"
    scratch_parent = tmp_path / "scratch"
    artifact_parent.mkdir()
    scratch_parent.mkdir()
    root = artifact_parent / "cbpupr-v2"
    scratch = scratch_parent / "cbpupr-v2-scratch"
    artifact_destination = root.with_name(
        root.name + f".quarantine-v2-terminal-failure-{_TIMESTAMP}"
    )
    scratch_destination = scratch.with_name(
        scratch.name + f".quarantine-v2-terminal-failure-{_TIMESTAMP}"
    )
    return root, scratch, artifact_destination, scratch_destination


@pytest.mark.parametrize("terminal_prefix_length", (0, 1, 2, 3, 6, 11))
def test_v2_terminal_audit_accepts_only_ordered_prefixes_and_missing_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_prefix_length: int,
) -> None:
    root, scratch, _, _ = _paths(tmp_path)
    _write_bundle(
        root,
        phase=V2_TERMINAL_PHASE,
        terminal_prefix_length=terminal_prefix_length,
    )
    _write_scratch(scratch)
    monkeypatch.setattr(quarantine_audit_module, "SCRATCH_ROOT", str(scratch))

    audit = audit_failed_v2_terminal_or_final_for_quarantine(root)

    assert audit["terminal_persistence_prefix_length"] == terminal_prefix_length
    assert audit["terminal_capability_report_persisted"] is (
        terminal_prefix_length >= 3
    )
    assert audit["terminal_access_journal_status"] == (
        "NOT_OPENED"
        if terminal_prefix_length == 0
        else "UNKNOWN_CONSERVATIVELY_CONSUMED"
        if terminal_prefix_length == 1
        else "OPENED"
    )
    assert audit["durable_preterminal_gate_revalidated"] is True
    assert audit["v2_rerun_authorized"] is False


def test_v2_audit_rejects_phase_prefix_gap_and_early_absent_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, _, _ = _paths(tmp_path)
    _write_bundle(root, phase=V2_TERMINAL_PHASE, terminal_prefix_length=1)
    _write_scratch(scratch)
    monkeypatch.setattr(quarantine_audit_module, "SCRATCH_ROOT", str(scratch))
    gap = root / V2_TERMINAL_PERSISTENCE_ORDER[2]
    _write_json(gap, {"out_of_order": True})
    with pytest.raises(ProtocolError, match="phase-aware artifact inventory"):
        audit_failed_v2_terminal_or_final_for_quarantine(root)
    gap.unlink()
    os.rename(scratch, scratch.with_name("removed-scratch"))
    with pytest.raises(ProtocolError, match="scratch is absent before cleanup edge"):
        audit_failed_v2_terminal_or_final_for_quarantine(root)


def test_v2_quarantine_moves_scratch_first_and_replays_hashed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, artifact_destination, scratch_destination = _paths(tmp_path)
    _write_bundle(root, phase=V2_TERMINAL_PHASE, terminal_prefix_length=0)
    _write_scratch(scratch)
    monkeypatch.setattr(quarantine_audit_module, "SCRATCH_ROOT", str(scratch))
    real_rename = os.rename
    renames: list[tuple[Path, Path]] = []

    def recording_rename(source: object, destination: object) -> None:
        renames.append((Path(source), Path(destination)))
        real_rename(source, destination)

    monkeypatch.setattr(
        "midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.v2_quarantine_move.os.rename",
        recording_rename,
    )
    receipt = quarantine_failed_v2_terminal_or_final(
        root,
        artifact_destination=artifact_destination,
        scratch_destination=scratch_destination,
    )
    assert renames == [
        (scratch, scratch_destination),
        (root, artifact_destination),
    ]
    assert receipt["move_order"] == ["scratch", "artifact"]
    assert receipt["quarantined_bytes_may_feed_rerun"] is False
    unhashed = {
        key: value
        for key, value in receipt.items()
        if key != "quarantine_receipt_hash"
    }
    assert receipt["quarantine_receipt_hash"] == canonical_hash(unhashed)
    assert Path(receipt["receipt_path"]).is_file()
    renames.clear()
    replay = quarantine_failed_v2_terminal_or_final(
        root,
        artifact_destination=artifact_destination,
        scratch_destination=scratch_destination,
    )
    assert replay == receipt
    assert renames == []
    receipt_path = Path(receipt["receipt_path"])
    receipt_path.unlink()
    recreated = quarantine_failed_v2_terminal_or_final(
        root,
        artifact_destination=artifact_destination,
        scratch_destination=scratch_destination,
    )
    assert recreated == receipt
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_v2_final_full_prefix_accepts_cleanup_complete_absent_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, artifact_destination, scratch_destination = _paths(tmp_path)
    _write_bundle(
        root,
        phase=V2_FINAL_PHASE,
        terminal_prefix_length=len(V2_TERMINAL_PERSISTENCE_ORDER),
        final_prefix_length=len(V2_FINAL_PERSISTENCE_ORDER),
    )
    monkeypatch.setattr(quarantine_audit_module, "SCRATCH_ROOT", str(scratch))

    audit = audit_failed_v2_terminal_or_final_for_quarantine(root)
    assert audit["final_persistence_prefix"] == list(V2_FINAL_PERSISTENCE_ORDER)
    assert audit["scratch_state"] == "ABSENT_AFTER_FINAL_REPORT"
    receipt = quarantine_failed_v2_terminal_or_final(
        root,
        artifact_destination=artifact_destination,
        scratch_destination=scratch_destination,
    )
    assert receipt["move_order"] == ["artifact"]
    assert receipt["scratch_absent_before_quarantine"] is True
    assert artifact_destination.is_dir()
    assert not scratch_destination.exists()
