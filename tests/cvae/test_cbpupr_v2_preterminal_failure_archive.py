from __future__ import annotations

from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import shutil

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import (
    fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v2_archive as archive,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v2_archive import (
    audit as audit_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v2_archive import (
    contracts as contract_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v2_archive import (
    move as move_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v2_archive.contracts import (
    ArchiveContract,
    CLAIM_ROLE,
    CLAIM_SCOPE,
    CONFIG_CONTRACT_HASH,
    EXPERIMENT_ID,
    EXPECTED_AMENDMENT_SHA256,
    EXPECTED_LEDGER_SHA256,
    EXPECTED_MANIFEST_SHA256,
    FAILED_ERROR,
    FAILED_PHASE,
    MemberDigest,
    OUTPUT_ARTIFACT_ID,
    PROTOCOL_CONTRACT_HASH,
    PUBLICATION_STATUS,
    REPAIR_SOURCE_MANIFEST_SHA256,
    REPAIR_SOURCE_MEMBER_COUNT,
    REPAIR_SOURCE_TREE_SHA256,
    STAGE_ID,
    TERMINAL_DECISION,
    V2_PRETERMINAL_ARTIFACT_DIRECTORIES,
    V2_PRETERMINAL_ARTIFACT_FILES,
    V2_PRETERMINAL_SCRATCH_DIRECTORIES,
    V2_PRETERMINAL_SCRATCH_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v2_archive.hashing import (
    canonical_hash,
    short_hash,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file


_TAG = "20260822T235959Z"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    artifact_parent = tmp_path / "artifacts"
    scratch_parent = tmp_path / "scratch"
    artifact_parent.mkdir()
    scratch_parent.mkdir()
    root = artifact_parent / "cbpupr-v2"
    scratch = scratch_parent / "cbpupr-v2-scratch"
    artifact_destination = root.with_name(
        root.name + f".quarantine-v2-preterminal-endpoint-lineage-{_TAG}"
    )
    scratch_destination = scratch.with_name(
        scratch.name + f".quarantine-v2-preterminal-endpoint-lineage-{_TAG}"
    )
    return root, scratch, artifact_destination, scratch_destination


def _config_payload(root: Path, scratch: Path) -> dict[str, object]:
    false_flags = {key: False for key in audit_module._FORBIDDEN_CLAIM_FLAGS}
    action_library = {
        "schema_version": "fixed_bank_cbpupr_action_library_v1",
        "target_expert_excluded": True,
    }
    policy_menu = {
        "schema_version": "fixed_bank_cbpupr_policy_menu_v1",
        "protected_fallback": "P_PROTECTED",
    }
    return {
        "experiment": {
            "id": EXPERIMENT_ID,
            "artifact_root": str(root),
            "claim_scope": CLAIM_SCOPE,
            "status": PUBLICATION_STATUS,
        },
        "inputs": {
            "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "expected_test_consumption_ledger_sha256": EXPECTED_LEDGER_SHA256,
            "expected_ledger_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
            "expected_ledger_amendment_parent_sha256": EXPECTED_LEDGER_SHA256,
            "ledger_amendment_authorized_experiment_id": EXPERIMENT_ID,
        },
        "protocol": {
            "schema_version": "fixed_bank_cbpupr_protocol_v2",
            "experiment_id": EXPERIMENT_ID,
            "execution_revision": "v2_canonical_row_order_mechanical_repair",
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "claim_scope": CLAIM_SCOPE,
            "claim_role": CLAIM_ROLE,
            "split": "test",
            "split_previously_consumed": True,
            "fresh_evidence": False,
            "repair_source_manifest_sha256": REPAIR_SOURCE_MANIFEST_SHA256,
            "repair_source_tree_sha256": REPAIR_SOURCE_TREE_SHA256,
            "repair_source_member_count": REPAIR_SOURCE_MEMBER_COUNT,
            "previous_stage90_outputs_used": False,
            "previous_stage90_scratch_or_checkpoints_used": False,
            "routing_success_claimed": False,
            "promotion_eligible": False,
            "may_feed_another_experiment": False,
        },
        "runtime": {
            "schema_version": "fixed_bank_cbpupr_workstation_runtime_v2",
            "scratch_preference": [str(scratch), "artifact_parent"],
            "cross_run_recovery_allowed": False,
            "terminal_recovery_allowed": False,
            "owned_task_checkpoint_replay_allowed": False,
            "foreign_checkpoint_reuse_forbidden": True,
            "previous_stage90_scratch_reuse_forbidden": True,
            "repair_source_manifest_sha256": REPAIR_SOURCE_MANIFEST_SHA256,
            "repair_source_tree_sha256": REPAIR_SOURCE_TREE_SHA256,
            "repair_source_member_count": REPAIR_SOURCE_MEMBER_COUNT,
        },
        "claim_boundary": {
            "schema_version": "fixed_bank_cbpupr_claim_boundary_v2",
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "claim_role": CLAIM_ROLE,
            "consumed_test_data": True,
            "terminal_stage90_diagnostic": True,
            **false_flags,
        },
        "action_library": action_library,
        "policy_menu": policy_menu,
    }


def _member_rows(root: Path, files: frozenset[str]) -> tuple[MemberDigest, ...]:
    return tuple(
        MemberDigest(relative, (root / relative).stat().st_size, sha256_file(root / relative))
        for relative in sorted(files)
    )


def _write_bundle(
    root: Path,
    scratch: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ArchiveContract:
    for directory in V2_PRETERMINAL_ARTIFACT_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)
    for relative in V2_PRETERMINAL_ARTIFACT_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"placeholder::{relative}\n".encode())
    config = _config_payload(root, scratch)
    (root / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _write_json(root / "manifests/action_library.json", config["action_library"])
    _write_json(root / "manifests/policy_menu.json", config["policy_menu"])

    input_ids = sorted(dict(contract_module.INPUT_ARTIFACT_HASHES))
    provenance_rows = [
        {
            "artifact_id": artifact_id,
            "exists": True,
            "semantic_identities": {"archive_fixture": "label_free"},
            "file_integrity": {"status": "SEALED"},
        }
        for artifact_id in input_ids
    ]
    provenance = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": EXPERIMENT_ID,
        "stage": STAGE_ID,
        "claim_scope": CLAIM_SCOPE,
        "selection_used_target_eval_artifacts": False,
        "input_artifacts": provenance_rows,
    }
    _write_json(root / "provenance/input_artifacts.json", provenance)
    input_hashes = tuple(
        (str(row["artifact_id"]), canonical_hash(row)) for row in provenance_rows
    )
    protocol_unhashed = {
        "schema_version": "fixed_bank_cbpupr_protocol_manifest_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "config_contract_hash": CONFIG_CONTRACT_HASH,
        "protocol_contract_hash": PROTOCOL_CONTRACT_HASH,
        "stage": STAGE_ID,
        "claim_scope": CLAIM_SCOPE,
        "claim_role": CLAIM_ROLE,
        "input_artifact_hashes": dict(input_hashes),
        "cache_binding_hash": "b" * 64,
        "pre_gpu_firewall": {
            "status": "PASS",
            "repair_source_manifest_validated": True,
            "repair_source_manifest_sha256": REPAIR_SOURCE_MANIFEST_SHA256,
            "repair_source_tree_sha256": REPAIR_SOURCE_TREE_SHA256,
            "repair_source_member_count": REPAIR_SOURCE_MEMBER_COUNT,
            "target_labels_opened": False,
            "target_expert_used": False,
            "predecessor_stage90_artifact_prediction_checkpoint_or_scratch_consumed": False,
        },
        "exact_six_original_inputs": True,
        "previous_stage90_output_or_checkpoint_used": False,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "publication_status": PUBLICATION_STATUS,
    }
    _write_json(
        root / "manifests/protocol_manifest.json",
        {
            **protocol_unhashed,
            "protocol_manifest_hash": canonical_hash(protocol_unhashed),
        },
    )
    _write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "fixed_bank_cbpupr_run_state_v1",
            "status": "FAILED",
            "phase": FAILED_PHASE,
            "error": FAILED_ERROR,
            "error_class": "ProtocolError",
            "updated_at_utc": "2026-08-22T19:21:56.008346+00:00",
            "cross_run_recovery_allowed": False,
            "terminal_recovery_allowed": False,
        },
    )
    _write_json(
        root / "reports/workstation_preflight.json",
        {
            "schema_version": "fixed_bank_cbpupr_workstation_preflight_v1",
            "status": "PASS",
            "outer_route_count": 218,
            "target_probability_cell_count": 810,
            "expected_target_posterior_model_fit_count": 436,
            "scratch_absent_at_launch": True,
            "owned_task_checkpoint_replay_allowed": False,
            "foreign_checkpoint_reuse_forbidden": True,
            "cross_run_recovery_allowed": False,
            "terminal_recovery_allowed": False,
        },
    )

    (root / "arrays/frozen_source_streams.npy").write_bytes(b"source-array")
    source_index_unhashed = {
        "schema_version": "midogpp_frozen_source_stream_index_v1",
        "records": [],
        "stream_count": 81,
        "labels_consumed": False,
    }
    source_index = {
        **source_index_unhashed,
        "source_stream_index_hash": short_hash(source_index_unhashed),
    }
    _write_json(root / "manifests/frozen_source_stream_index.json", source_index)
    source_lock_unhashed = {
        "schema_version": "midogpp_frozen_source_stream_lock_v1",
        "config_contract_hash": CONFIG_CONTRACT_HASH,
        "source_array_sha256": sha256_file(root / "arrays/frozen_source_streams.npy"),
        "source_stream_index_sha256": sha256_file(
            root / "manifests/frozen_source_stream_index.json"
        ),
        "source_stream_index_hash": source_index["source_stream_index_hash"],
        "stream_count": 81,
        "labels_consumed": False,
        "source_experts_updated": False,
    }
    source_lock = {
        **source_lock_unhashed,
        "source_stream_lock_hash": short_hash(source_lock_unhashed),
    }
    _write_json(root / "manifests/frozen_source_stream_lock.json", source_lock)

    (root / "arrays/fixed_bank_a1_action_probabilities.npz").write_bytes(
        b"prediction-array"
    )
    prediction_fields = {
        "config_contract_hash": CONFIG_CONTRACT_HASH,
        "partition_hash": "c" * 64,
        "source_stream_lock_hash": source_lock["source_stream_lock_hash"],
        "action_library_hash": "d" * 16,
        "target_cache_binding_hash": "e" * 64,
        "store_hash": "f" * 16,
    }
    _write_json(
        root / "manifests/fixed_bank_a1_prediction_index.json",
        {
            "schema_version": "fixed_bank_a1_prediction_index_v1",
            **prediction_fields,
            "cells": [],
        },
    )
    prediction_unhashed = {
        "schema_version": "fixed_bank_a1_prediction_seal_v1",
        **prediction_fields,
        "arrays_sha256": sha256_file(
            root / "arrays/fixed_bank_a1_action_probabilities.npz"
        ),
        "index_sha256": sha256_file(
            root / "manifests/fixed_bank_a1_prediction_index.json"
        ),
        "cell_count": 810,
        "task_count": 81,
        "labels_opened": False,
        "target_expert_used": False,
    }
    prediction = {
        **prediction_unhashed,
        "global_prediction_seal_hash": short_hash(prediction_unhashed),
    }
    _write_json(root / "manifests/fixed_bank_a1_prediction_seal.json", prediction)
    rows = [{"ordinal": ordinal, "labels_used": False} for ordinal in range(90)]
    _write_json(
        root / "tables/exact_nine_probability_index.json",
        {
            "schema_version": "fixed_bank_cbpupr_exact_nine_probability_index_v1",
            "row_count": 90,
            "rows": rows,
        },
    )
    physical_unhashed = {
        "schema_version": "fixed_bank_cbpupr_physical_surface_seal_v1",
        "surface_hash": "1" * 64,
        "probability_store_hash": prediction["store_hash"],
        "source_stream_lock_hash": source_lock["source_stream_lock_hash"],
        "global_prediction_seal_hash": prediction["global_prediction_seal_hash"],
        "probability_index_hash": canonical_hash(rows),
        "target_probability_cell_count": 810,
        "labels_used": False,
    }
    _write_json(
        root / "manifests/physical_surface_seal.json",
        {
            **physical_unhashed,
            "physical_surface_seal_hash": canonical_hash(physical_unhashed),
        },
    )
    (root / ".run.lock").write_bytes(b"pid=1234\n")

    for directory in V2_PRETERMINAL_SCRATCH_DIRECTORIES:
        (scratch / directory).mkdir(parents=True, exist_ok=True)
    duplicate_pairs = {
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
    for target, source in duplicate_pairs.items():
        shutil.copyfile(root / source, scratch / target)

    contract = ArchiveContract(
        _member_rows(root, V2_PRETERMINAL_ARTIFACT_FILES),
        _member_rows(scratch, V2_PRETERMINAL_SCRATCH_FILES),
        input_hashes,
    )
    monkeypatch.setattr(contract_module, "CANONICAL_ARCHIVE_CONTRACT", contract)
    return contract


def _refresh_artifact_contract(
    root: Path,
    scratch: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = contract_module.CANONICAL_ARCHIVE_CONTRACT
    monkeypatch.setattr(
        contract_module,
        "CANONICAL_ARCHIVE_CONTRACT",
        replace(
            old,
            artifact_members=_member_rows(root, V2_PRETERMINAL_ARTIFACT_FILES),
            scratch_members=_member_rows(scratch, V2_PRETERMINAL_SCRATCH_FILES),
        ),
    )


def test_audit_certifies_exact_failed_prefix_and_full_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, _, _ = _paths(tmp_path)
    contract = _write_bundle(root, scratch, monkeypatch)

    result = archive.audit_failed_v2_preterminal_for_archive(
        root, scratch_root=scratch
    )

    assert result["archive_contract_hash"] == contract.contract_hash
    assert result["source_run_phase"] == FAILED_PHASE
    assert result["scratch_state"] == "EXACT_OBSERVED_FULL_SCRATCH"
    assert result["terminal_access_journal_status"] == "ABSENT_NOT_OPENED"
    assert result["terminal_labels_opened"] is False
    assert result["v2_rerun_authorized"] is False
    assert result["v3_input_reuse_authorized"] is False
    assert result["promotion_eligible"] is False


def test_explicit_no_scratch_is_distinct_from_missing_supplied_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, _, _ = _paths(tmp_path)
    _write_bundle(root, scratch, monkeypatch)
    with pytest.raises(ProtocolError, match="explicit no-scratch state"):
        archive.audit_failed_v2_preterminal_for_archive(root, scratch_root=None)
    shutil.rmtree(scratch)

    no_scratch = archive.audit_failed_v2_preterminal_for_archive(
        root, scratch_root=None
    )
    assert no_scratch["scratch_state"] == "EXPLICIT_NO_SCRATCH_API_STATE"
    assert no_scratch["scratch_verified"] is False
    with pytest.raises(ProtocolError, match="supplied scratch is absent"):
        archive.audit_failed_v2_preterminal_for_archive(root, scratch_root=scratch)


@pytest.mark.parametrize("mutation", ("extra", "missing", "journal", "hash"))
def test_audit_rejects_inventory_journal_and_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root, scratch, _, _ = _paths(tmp_path)
    _write_bundle(root, scratch, monkeypatch)
    if mutation == "extra":
        (root / "reports/foreign.json").write_text("{}\n", encoding="utf-8")
    elif mutation == "missing":
        (root / "manifests/policy_menu.json").unlink()
    elif mutation == "journal":
        _write_json(root / "manifests/terminal_label_access_intent.json", {})
    else:
        with (root / "arrays/frozen_source_streams.npy").open("ab") as handle:
            handle.write(b"drift")
    with pytest.raises(ProtocolError):
        archive.audit_failed_v2_preterminal_for_archive(root, scratch_root=scratch)


def test_audit_rejects_symlinks_active_lock_and_nonexact_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, _, _ = _paths(tmp_path)
    _write_bundle(root, scratch, monkeypatch)
    lock_path = root / ".run.lock"
    with lock_path.open("r+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ProtocolError, match="diagnostic is active"):
            archive.audit_failed_v2_preterminal_for_archive(
                root, scratch_root=scratch
            )
    source = root / "manifests/policy_menu.json"
    source.unlink()
    source.symlink_to(root / "manifests/action_library.json")
    with pytest.raises(ProtocolError, match="unsafe member"):
        archive.audit_failed_v2_preterminal_for_archive(root, scratch_root=scratch)
    source.unlink()
    config = yaml.safe_load((root / "config.resolved.yaml").read_text("utf-8"))
    _write_json(source, config["policy_menu"])
    shutil.rmtree(scratch)
    scratch.mkdir()
    with pytest.raises(ProtocolError, match="scratch inventory drifted"):
        archive.audit_failed_v2_preterminal_for_archive(root, scratch_root=scratch)


@pytest.mark.parametrize("drift", ("phase", "error", "recovery", "promotion"))
def test_audit_rejects_semantic_failure_or_claim_drift_even_if_resigned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    root, scratch, _, _ = _paths(tmp_path)
    _write_bundle(root, scratch, monkeypatch)
    if drift in {"phase", "error", "recovery"}:
        path = root / "reports/run_state.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if drift == "phase":
            payload["phase"] = "DURABLE_PRETERMINAL_BARRIER"
        elif drift == "error":
            payload["error"] = "another failure"
        else:
            payload["cross_run_recovery_allowed"] = True
        _write_json(path, payload)
    else:
        path = root / "config.resolved.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload["claim_boundary"]["promotion_eligible"] = True
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _refresh_artifact_contract(root, scratch, monkeypatch)
    with pytest.raises(ProtocolError):
        archive.audit_failed_v2_preterminal_for_archive(root, scratch_root=scratch)


def test_quarantine_moves_scratch_first_and_replays_immutable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, artifact_destination, scratch_destination = _paths(tmp_path)
    _write_bundle(root, scratch, monkeypatch)
    real_rename = os.rename
    moves: list[tuple[Path, Path]] = []

    def recording_rename(source: object, destination: object) -> None:
        moves.append((Path(source), Path(destination)))
        real_rename(source, destination)

    monkeypatch.setattr(move_module.os, "rename", recording_rename)
    receipt = archive.quarantine_failed_v2_preterminal_for_archive(
        root,
        artifact_destination=artifact_destination,
        scratch_root=scratch,
        scratch_destination=scratch_destination,
    )
    receipt_path = Path(f"{artifact_destination}.receipt.json")
    receipt_bytes = receipt_path.read_bytes()

    assert moves[:2] == [
        (scratch.resolve(), scratch_destination.resolve()),
        (root.resolve(), artifact_destination.resolve()),
    ]
    assert not root.exists() and not scratch.exists()
    assert artifact_destination.is_dir() and scratch_destination.is_dir()
    assert receipt["move_order"] == ["scratch", "artifact"]
    assert receipt["quarantined_bytes_may_feed_successor"] is False
    assert receipt["pre_move_audit"] == receipt["post_move_audit"]
    replay = archive.quarantine_failed_v2_preterminal_for_archive(
        root,
        artifact_destination=artifact_destination,
        scratch_root=scratch,
        scratch_destination=scratch_destination,
    )
    assert replay == receipt
    assert receipt_path.read_bytes() == receipt_bytes


def test_quarantine_retries_after_scratch_move_and_rejects_receipt_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, artifact_destination, scratch_destination = _paths(tmp_path)
    _write_bundle(root, scratch, monkeypatch)
    os.rename(scratch, scratch_destination)

    receipt = archive.quarantine_failed_v2_preterminal_for_archive(
        root,
        artifact_destination=artifact_destination,
        scratch_root=scratch,
        scratch_destination=scratch_destination,
    )
    receipt_path = Path(str(receipt["receipt_path"]))
    receipt_path.chmod(0o644)
    receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="immutable receipt drifted"):
        archive.quarantine_failed_v2_preterminal_for_archive(
            root,
            artifact_destination=artifact_destination,
            scratch_root=scratch,
            scratch_destination=scratch_destination,
        )


def test_quarantine_supports_only_explicit_no_scratch_and_safe_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, artifact_destination, scratch_destination = _paths(tmp_path)
    _write_bundle(root, scratch, monkeypatch)
    shutil.rmtree(scratch)
    with pytest.raises(ProtocolError, match="supplied together"):
        archive.quarantine_failed_v2_preterminal_for_archive(
            root,
            artifact_destination=artifact_destination,
            scratch_root=None,
            scratch_destination=scratch_destination,
        )
    wrong_parent = tmp_path / "other"
    wrong_parent.mkdir()
    with pytest.raises(ProtocolError, match="not same-parent"):
        archive.quarantine_failed_v2_preterminal_for_archive(
            root,
            artifact_destination=wrong_parent / artifact_destination.name,
            scratch_root=None,
            scratch_destination=None,
        )

    receipt = archive.quarantine_failed_v2_preterminal_for_archive(
        root,
        artifact_destination=artifact_destination,
        scratch_root=None,
        scratch_destination=None,
    )
    assert receipt["move_order"] == ["artifact"]
    assert receipt["explicit_no_scratch_api_state"] is True
    assert receipt["whole_scratch_move_completed"] is False
