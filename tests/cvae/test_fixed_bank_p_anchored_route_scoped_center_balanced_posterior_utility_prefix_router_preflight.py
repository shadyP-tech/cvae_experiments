from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router import (
    preflight as cbpupr_preflight,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router import (
    run_admission as run_admission_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router import (
    terminal_failure_quarantine as terminal_quarantine_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router import (
    scratch as scratch_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.config_payloads import (
    canonical_runtime_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.constants import (
    QUARANTINED_V1_EXPERIMENT_ID,
    RUN_RECOVERY_POLICY,
    SCRATCH_ROOT,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.bundle import (
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.run_admission import (
    FAILED_TERMINAL_LINEAGE_FILES,
    FAILED_TERMINAL_LINEAGE_FINAL_MEMBERS,
    FAILED_TERMINAL_LINEAGE_SCRATCH_DIRECTORIES,
    FAILED_TERMINAL_LINEAGE_SCRATCH_FILES,
    FAILED_WORKSTATION_PREFLIGHT_FILES,
    audit_failed_terminal_lineage_for_quarantine,
    audit_failed_workstation_preflight_for_quarantine,
    quarantine_failed_terminal_lineage,
    quarantine_failed_workstation_preflight,
    reject_quarantined_v1_execution,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.runner import (
    run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.workstation import (
    assert_runtime as assert_cbpupr_runtime,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime import frozen_source_streams
from midogpp_thesis.cvae.runtime import preflight as shared_preflight
from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_contracts import (
    assert_runtime as assert_prediction_runtime,
)


def _mock_workstation_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in shared_preflight.REQUIRED_THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.delitem(shared_preflight.sys.modules, "torch", raising=False)
    monkeypatch.setattr(
        shared_preflight.mp, "get_all_start_methods", lambda: ("spawn",)
    )
    monkeypatch.setattr(shared_preflight, "_available_cpu_count", lambda: 24)
    monkeypatch.setattr(
        shared_preflight, "_physical_ram_bytes", lambda: 128 * 1024**3
    )
    monkeypatch.setattr(
        shared_preflight.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=100 * 1024**3),
    )
    monkeypatch.setattr(
        shared_preflight,
        "_package_versions",
        lambda: {
            name: "test" for name in shared_preflight.REQUIRED_DISTRIBUTIONS
        },
    )
    monkeypatch.setattr(
        shared_preflight,
        "_nvidia_smi_rows",
        lambda: tuple(
            {
                "index": index,
                "name": "NVIDIA RTX A5000",
                "memory_total_mib": 24_576,
                "memory_free_mib": 20_000,
            }
            for index in (0, 1)
        ),
    )


def _assert_shared_preflight_rejects(runtime: dict[str, object]) -> None:
    with pytest.raises(ProtocolError, match="Label-free workstation topology drifted"):
        shared_preflight.run_label_free_workstation_preflight(
            Path("/unused-before-resource-probes"),
            runtime=runtime,
            expected_scratch_root=SCRATCH_ROOT,
            expected_target_action_identity_count=90,
            expected_target_probability_cell_count=810,
            expected_unique_classifier_fit_count=810,
            expected_resume_policy=RUN_RECOVERY_POLICY,
        )


def test_canonical_runtime_satisfies_shared_physical_runtime_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = canonical_runtime_payload()
    assert_cbpupr_runtime(runtime)
    frozen_source_streams._assert_runtime(runtime)
    assert_prediction_runtime(runtime)
    _mock_workstation_resources(monkeypatch)
    isolated_scratch = tmp_path / "dedicated-scratch"
    runtime = {
        **runtime,
        "scratch_preference": [str(isolated_scratch), "artifact_parent"],
    }
    monkeypatch.setattr(cbpupr_preflight, "SCRATCH_ROOT", str(isolated_scratch))
    monkeypatch.setattr(scratch_module, "SCRATCH_ROOT", str(isolated_scratch))

    admitted = cbpupr_preflight.run_workstation_preflight(
        tmp_path / "bundle", runtime=runtime
    )

    assert admitted["status"] == "PASS"
    assert admitted["schema_version"] == cbpupr_preflight.SCHEMA
    assert admitted["target_action_identity_count"] == 90
    assert admitted["target_probability_cell_count"] == 810
    assert admitted["target_unique_classifier_fit_count"] == 810


def test_physical_runtime_contracts_reject_missing_source_and_target_fields() -> None:
    poisoned_source = canonical_runtime_payload()
    poisoned_source.pop("source_prefix_rows_per_class")
    with pytest.raises(ProtocolError, match="CBPUPR workstation runtime contract"):
        assert_cbpupr_runtime(poisoned_source)
    with pytest.raises(ProtocolError, match="Frozen source generation requires"):
        frozen_source_streams._assert_runtime(poisoned_source)
    _assert_shared_preflight_rejects(poisoned_source)

    poisoned_target = canonical_runtime_payload()
    poisoned_target.pop("target_task_count")
    with pytest.raises(ProtocolError, match="CBPUPR workstation runtime contract"):
        assert_cbpupr_runtime(poisoned_target)
    with pytest.raises(ProtocolError, match="A1 prediction workstation contract"):
        assert_prediction_runtime(poisoned_target)
    _assert_shared_preflight_rejects(poisoned_target)


def test_quarantined_v1_execution_is_blocked_before_any_run_state_write(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "must-not-be-resolved-or-created"
    config = SimpleNamespace(
        experiment_id=QUARANTINED_V1_EXPERIMENT_ID,
        artifact_root=sentinel,
    )
    with pytest.raises(ProtocolError, match="separately authorized v2"):
        run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router(
            config, artifact_root=sentinel
        )
    assert not sentinel.exists()
    assert not (sentinel / "config.resolved.yaml").exists()
    assert not (sentinel / "reports/run_state.json").exists()

    with pytest.raises(ProtocolError, match="separately authorized v2"):
        reject_quarantined_v1_execution(config)

    reject_quarantined_v1_execution(
        SimpleNamespace(
            experiment_id=(
                "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_"
                "anchored_route_scoped_center_balanced_posterior_utility_"
                "prefix_router.v2"
            )
        )
    )


def test_failed_preflight_quarantine_audit_requires_zero_capability_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "failed"
    for relative in FAILED_WORKSTATION_PREFLIGHT_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (root / ".run.lock").write_text("pid=123\n", encoding="ascii")
    (root / "reports/run_state.json").write_text(
        json.dumps(
            {
                "schema_version": "fixed_bank_cbpupr_run_state_v1",
                "status": "FAILED",
                "phase": "WORKSTATION_PREFLIGHT",
                "error": "Label-free workstation topology drifted.",
                "error_class": "ProtocolError",
                "updated_at_utc": "2026-08-21T23:19:06+00:00",
                "cross_run_recovery_allowed": False,
                "terminal_recovery_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    scratch = tmp_path / "absent-scratch"
    monkeypatch.setattr(
        run_admission_module, "QUARANTINED_V1_SCRATCH_ROOT", str(scratch)
    )

    lock_path = root / ".run.lock"
    lock_bytes = lock_path.read_bytes()
    observed_lock_flags: list[int] = []
    real_os_open = os.open

    def recording_os_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if Path(path) == lock_path:
            observed_lock_flags.append(flags)
        return real_os_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(run_admission_module.os, "open", recording_os_open)

    certificate = audit_failed_workstation_preflight_for_quarantine(root)
    assert certificate["status"] == "PASS"
    assert certificate["label_capability_opened"] is False
    assert certificate["physical_generation_started"] is False
    assert certificate["quarantined_bytes_may_feed_rerun"] is False
    assert certificate["eligible_next_action"] == (
        "MOVE_WHOLE_FAILED_ROOT_TO_QUARANTINE_ONLY"
    )
    assert observed_lock_flags
    assert all(
        flags & os.O_ACCMODE == os.O_RDWR for flags in observed_lock_flags
    )
    assert all(flags & os.O_CREAT == 0 for flags in observed_lock_flags)
    assert all(flags & os.O_TRUNC == 0 for flags in observed_lock_flags)
    expected_safety_flags = getattr(os, "O_NOFOLLOW", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    assert all(
        flags & expected_safety_flags == expected_safety_flags
        for flags in observed_lock_flags
    )
    assert lock_path.read_bytes() == lock_bytes

    capability = root / "reports/label_capability_report.json"
    capability.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="zero-capability inventory"):
        audit_failed_workstation_preflight_for_quarantine(root)
    capability.unlink()
    scratch.mkdir()
    with pytest.raises(ProtocolError, match="scratch is not absent"):
        audit_failed_workstation_preflight_for_quarantine(root)
    scratch.rmdir()

    root_link = tmp_path / "failed-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ProtocolError, match="root is a symlink"):
        audit_failed_workstation_preflight_for_quarantine(root_link)
    root_link.unlink()

    with (root / ".run.lock").open("r+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ProtocolError, match="diagnostic is active"):
            audit_failed_workstation_preflight_for_quarantine(root)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    quarantine = root.with_name(
        root.name + ".quarantine-failed-preflight-20260821T231906Z"
    )
    quarantine.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ProtocolError, match="quarantine path is a symlink"):
        quarantine_failed_workstation_preflight(
            root, destination=quarantine
        )
    quarantine.unlink()
    receipt = quarantine_failed_workstation_preflight(
        root, destination=quarantine
    )
    assert receipt["status"] == "PASS"
    assert receipt["whole_root_move_completed"] is True
    assert receipt["quarantined_bytes_may_feed_rerun"] is False
    assert not root.exists()
    assert quarantine.is_dir()


_TERMINAL_TIMESTAMP = "20260822T144828Z"
_SOURCE_BYTES = {
    "arrays/frozen_source_streams.npy": b"sealed-source-array\x00\x01",
    "manifests/frozen_source_stream_index.json": b'{"index":"sealed"}\n',
    "manifests/frozen_source_stream_lock.json": b'{"lock":"sealed"}\n',
}
_SCRATCH_SOURCE_TO_ARTIFACT = {
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


def _write_failed_terminal_bundle(root: Path) -> None:
    deferred = {
        "manifests/content_index.json",
        "reports/label_capability_report.json",
        "reports/run_state.json",
    }
    for relative in sorted(FAILED_TERMINAL_LINEAGE_FILES - deferred):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_SOURCE_BYTES.get(relative, f"{relative}\n".encode("utf-8")))
    protocol = {
        "schema_version": "fixed_bank_cbpupr_protocol_manifest_v1",
        "experiment_id": QUARANTINED_V1_EXPERIMENT_ID,
        "output_artifact_id": (
            "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
            "route_scoped_center_balanced_posterior_utility_prefix_router_v1"
        ),
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "previous_stage90_output_or_checkpoint_used": False,
    }
    protocol["protocol_manifest_hash"] = canonical_hash(protocol)
    (root / "manifests/protocol_manifest.json").write_text(
        json.dumps(protocol), encoding="utf-8"
    )
    outer_plan_seal = {"seal_hash": "a" * 64}
    (root / "manifests/outer_plan_seal.json").write_text(
        json.dumps(outer_plan_seal), encoding="utf-8"
    )
    events = [
        {"role": f"scoped_capability_{index}", "raw_labels_persisted": False}
        for index in range(2538)
    ] + [
        {
            "role": "target_terminal_after_aggregate_seal",
            "raw_labels_persisted": False,
        }
    ]
    capability = {
        "schema_version": "fixed_bank_cbpupr_label_access_audit_v1",
        "plan_seal_hash": "a" * 64,
        "event_count": len(events),
        "events": events,
        "target_candidate_seal_complete": True,
        "pre_evaluation_seal_complete": True,
        "pseudo_evaluation_route_count": 1744,
        "calibration_seal_complete": True,
        "decision_count": 872,
        "aggregate_seal_complete": True,
        "terminal_opened": True,
        "raw_labels_persisted": False,
        "audit_hash": canonical_hash(events),
    }
    (root / "reports/label_capability_report.json").write_text(
        json.dumps(capability), encoding="utf-8"
    )
    state = {
        "schema_version": "fixed_bank_cbpupr_run_state_v1",
        "status": "FAILED",
        "phase": "CONTENT_AND_TWO_FRESH_PROCESS_VALIDATION",
        "error": "CBPUPR persisted posterior prediction/model lineage drifted.",
        "error_class": "ProtocolError",
        "updated_at_utc": "2026-08-22T14:48:28.753245+00:00",
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
    }
    (root / "reports/run_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    write_content_index(root)
    (root / ".run.lock").write_bytes(b"pid=4321\n")


def _write_failed_terminal_scratch(scratch: Path) -> None:
    for relative in FAILED_TERMINAL_LINEAGE_SCRATCH_DIRECTORIES:
        (scratch / relative).mkdir(parents=True, exist_ok=True)
    for relative, artifact_relative in _SCRATCH_SOURCE_TO_ARTIFACT.items():
        (scratch / relative).write_bytes(_SOURCE_BYTES[artifact_relative])


def _terminal_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    artifact_parent = tmp_path / "artifact-parent"
    scratch_parent = tmp_path / "scratch-parent"
    artifact_parent.mkdir()
    scratch_parent.mkdir()
    root = artifact_parent / "cbpupr-v1"
    scratch = scratch_parent / "cbpupr-v1-scratch"
    artifact_destination = root.with_name(
        root.name + f".quarantine-terminal-lineage-{_TERMINAL_TIMESTAMP}"
    )
    scratch_destination = scratch.with_name(
        scratch.name + f".quarantine-terminal-lineage-{_TERMINAL_TIMESTAMP}"
    )
    return root, scratch, artifact_destination, scratch_destination


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _prepare_failed_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path]:
    paths = _terminal_paths(tmp_path)
    root, scratch, _, _ = paths
    _write_failed_terminal_bundle(root)
    _write_failed_terminal_scratch(scratch)
    monkeypatch.setattr(
        terminal_quarantine_module,
        "QUARANTINED_V1_SCRATCH_ROOT",
        str(scratch),
    )
    return paths


def test_terminal_lineage_quarantine_preserves_both_trees_and_replays_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, artifact_destination, scratch_destination = (
        _prepare_failed_terminal_state(tmp_path, monkeypatch)
    )
    artifact_bytes = _tree_bytes(root)
    scratch_bytes = _tree_bytes(scratch)
    audit = audit_failed_terminal_lineage_for_quarantine(root)
    assert audit["status"] == "PASS"
    assert audit["terminal_capability_opened"] is True
    assert audit["fresh_process_attestation_present"] is False
    assert audit["final_validation_report_present"] is False
    assert audit["fresh_v2_execution_identity_required"] is True
    assert len(audit["artifact_members"]) == len(FAILED_TERMINAL_LINEAGE_FILES) + 1
    assert {row["path"] for row in audit["scratch_members"]} == set(
        FAILED_TERMINAL_LINEAGE_SCRATCH_FILES
    )

    real_rename = os.rename
    renames: list[tuple[Path, Path]] = []

    def recording_rename(source: object, destination: object) -> None:
        renames.append((Path(source), Path(destination)))
        real_rename(source, destination)

    monkeypatch.setattr(terminal_quarantine_module.os, "rename", recording_rename)
    receipt = quarantine_failed_terminal_lineage(
        root,
        artifact_destination=artifact_destination,
        scratch_destination=scratch_destination,
    )
    assert renames == [
        (scratch, scratch_destination),
        (root, artifact_destination),
    ]
    assert not root.exists()
    assert not scratch.exists()
    assert _tree_bytes(artifact_destination) == artifact_bytes
    assert _tree_bytes(scratch_destination) == scratch_bytes
    assert receipt["quarantined_bytes_may_feed_rerun"] is False
    assert receipt["quarantined_v1_results_may_be_promoted"] is False
    assert receipt["fresh_v2_execution_identity_required"] is True
    unhashed = {
        key: value
        for key, value in receipt.items()
        if key != "quarantine_receipt_hash"
    }
    assert receipt["quarantine_receipt_hash"] == canonical_hash(unhashed)

    renames.clear()
    replay = quarantine_failed_terminal_lineage(
        root,
        artifact_destination=artifact_destination,
        scratch_destination=scratch_destination,
    )
    assert replay == receipt
    assert renames == []
    assert _tree_bytes(artifact_destination) == artifact_bytes
    assert _tree_bytes(scratch_destination) == scratch_bytes


def test_terminal_lineage_quarantine_recovers_only_scratch_first_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, artifact_destination, scratch_destination = (
        _prepare_failed_terminal_state(tmp_path, monkeypatch)
    )
    scratch_bytes = _tree_bytes(scratch)
    os.rename(scratch, scratch_destination)

    receipt = quarantine_failed_terminal_lineage(
        root,
        artifact_destination=artifact_destination,
        scratch_destination=scratch_destination,
    )
    assert receipt["status"] == "PASS"
    assert not root.exists()
    assert not scratch.exists()
    assert artifact_destination.is_dir()
    assert _tree_bytes(scratch_destination) == scratch_bytes


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("status", "RUNNING"),
        ("phase", "TERMINAL_LABELS_METRICS_AND_CONTROLS"),
        ("error", "different"),
        ("error_class", "RuntimeError"),
        ("cross_run_recovery_allowed", True),
        ("terminal_recovery_allowed", True),
    ),
)
def test_terminal_lineage_quarantine_rejects_state_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: object,
) -> None:
    root, _, _, _ = _prepare_failed_terminal_state(tmp_path, monkeypatch)
    state_path = root / "reports/run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state[key] = value
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ProtocolError, match="failed state drifted"):
        audit_failed_terminal_lineage_for_quarantine(root)


def test_terminal_lineage_quarantine_rejects_inventory_capability_and_scratch_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, _, _ = _prepare_failed_terminal_state(tmp_path, monkeypatch)

    final = root / next(iter(FAILED_TERMINAL_LINEAGE_FINAL_MEMBERS))
    final.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="artifact inventory drifted"):
        audit_failed_terminal_lineage_for_quarantine(root)
    final.unlink()

    required = root / "tables/outer_plans.json"
    required_bytes = required.read_bytes()
    required.unlink()
    with pytest.raises(ProtocolError, match="artifact inventory drifted"):
        audit_failed_terminal_lineage_for_quarantine(root)
    required.write_bytes(required_bytes)

    foreign_artifact = root / "reports/foreign.json"
    foreign_artifact.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="artifact inventory drifted"):
        audit_failed_terminal_lineage_for_quarantine(root)
    foreign_artifact.unlink()

    foreign_directory = root / "reports/foreign-directory"
    foreign_directory.mkdir()
    with pytest.raises(ProtocolError, match="artifact inventory drifted"):
        audit_failed_terminal_lineage_for_quarantine(root)
    foreign_directory.rmdir()

    capability_path = root / "reports/label_capability_report.json"
    capability = json.loads(capability_path.read_text(encoding="utf-8"))
    capability["terminal_opened"] = False
    capability_path.write_text(json.dumps(capability), encoding="utf-8")
    with pytest.raises(ProtocolError, match="capability state drifted"):
        audit_failed_terminal_lineage_for_quarantine(root)
    capability["terminal_opened"] = True
    capability_path.write_text(json.dumps(capability), encoding="utf-8")

    foreign = scratch / "prediction_cache/foreign.bin"
    foreign.write_bytes(b"foreign")
    with pytest.raises(ProtocolError, match="scratch inventory drifted"):
        audit_failed_terminal_lineage_for_quarantine(root)
    foreign.unlink()

    source_copy = scratch / "source_generation/arrays/frozen_source_streams.npy"
    source_copy.write_bytes(b"poisoned")
    with pytest.raises(ProtocolError, match="bytes differ"):
        audit_failed_terminal_lineage_for_quarantine(root)


def test_terminal_lineage_quarantine_rejects_symlinks_lock_and_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, artifact_destination, scratch_destination = (
        _prepare_failed_terminal_state(tmp_path, monkeypatch)
    )

    unsafe = scratch / "prediction_cache/unsafe"
    unsafe.symlink_to(root / "config.resolved.yaml")
    with pytest.raises(ProtocolError, match="unsafe member"):
        audit_failed_terminal_lineage_for_quarantine(root)
    unsafe.unlink()

    root_link = root.with_name("root-link")
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ProtocolError, match="root is a symlink"):
        audit_failed_terminal_lineage_for_quarantine(root_link)
    root_link.unlink()

    with (root / ".run.lock").open("r+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ProtocolError, match="diagnostic is active"):
            audit_failed_terminal_lineage_for_quarantine(root)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    lock = root / ".run.lock"
    lock_bytes = lock.read_bytes()
    lock.unlink()
    lock.symlink_to(root / "config.resolved.yaml")
    with pytest.raises(ProtocolError, match="run lock is absent or unsafe"):
        audit_failed_terminal_lineage_for_quarantine(root)
    lock.unlink()
    lock.write_bytes(lock_bytes)

    bad_parent = tmp_path / "different-parent"
    bad_parent.mkdir()
    with pytest.raises(ProtocolError, match="not same-parent"):
        quarantine_failed_terminal_lineage(
            root,
            artifact_destination=bad_parent / artifact_destination.name,
            scratch_destination=scratch_destination,
        )
    with pytest.raises(ProtocolError, match="name is unsafe"):
        quarantine_failed_terminal_lineage(
            root,
            artifact_destination=root.with_name(root.name + ".wrong"),
            scratch_destination=scratch_destination,
        )
    mismatched_scratch = scratch.with_name(
        scratch.name + ".quarantine-terminal-lineage-20260822T144829Z"
    )
    with pytest.raises(ProtocolError, match="timestamps differ"):
        quarantine_failed_terminal_lineage(
            root,
            artifact_destination=artifact_destination,
            scratch_destination=mismatched_scratch,
        )
    invalid_timestamp_root = root.with_name(
        root.name + ".quarantine-terminal-lineage-20269999T999999Z"
    )
    invalid_timestamp_scratch = scratch.with_name(
        scratch.name + ".quarantine-terminal-lineage-20269999T999999Z"
    )
    with pytest.raises(ProtocolError, match="timestamp is invalid"):
        quarantine_failed_terminal_lineage(
            root,
            artifact_destination=invalid_timestamp_root,
            scratch_destination=invalid_timestamp_scratch,
        )
    artifact_destination.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ProtocolError, match="quarantine path is a symlink"):
        quarantine_failed_terminal_lineage(
            root,
            artifact_destination=artifact_destination,
            scratch_destination=scratch_destination,
        )
    artifact_destination.unlink()
    artifact_destination.mkdir()
    with pytest.raises(ProtocolError, match="quarantine state is unsafe"):
        quarantine_failed_terminal_lineage(
            root,
            artifact_destination=artifact_destination,
            scratch_destination=scratch_destination,
        )


def test_terminal_lineage_quarantine_rejects_root_first_and_post_move_foreign_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, artifact_destination, scratch_destination = (
        _prepare_failed_terminal_state(tmp_path, monkeypatch)
    )
    os.rename(root, artifact_destination)
    with pytest.raises(ProtocolError, match="quarantine state is unsafe"):
        quarantine_failed_terminal_lineage(
            root,
            artifact_destination=artifact_destination,
            scratch_destination=scratch_destination,
        )
    os.rename(artifact_destination, root)

    real_rename = os.rename

    def rename_with_foreign_member(source: object, destination: object) -> None:
        real_rename(source, destination)
        if Path(source) == root:
            foreign = Path(destination) / "reports/foreign.json"
            foreign.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        terminal_quarantine_module.os, "rename", rename_with_foreign_member
    )
    with pytest.raises(ProtocolError, match="artifact inventory drifted"):
        quarantine_failed_terminal_lineage(
            root,
            artifact_destination=artifact_destination,
            scratch_destination=scratch_destination,
        )
    assert artifact_destination.is_dir()
    assert scratch_destination.is_dir()
    assert not root.exists()
    assert not scratch.exists()


def test_terminal_lineage_quarantine_rehashes_same_size_post_move_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, artifact_destination, scratch_destination = (
        _prepare_failed_terminal_state(tmp_path, monkeypatch)
    )
    real_rename = os.rename

    def rename_then_mutate_scratch(source: object, destination: object) -> None:
        real_rename(source, destination)
        if Path(source) == root:
            moved = (
                scratch_destination
                / "source_generation/manifests/frozen_source_stream_lock.json"
            )
            original = moved.read_bytes()
            moved.write_bytes(b"X" + original[1:])

    monkeypatch.setattr(
        run_admission_module.os, "rename", rename_then_mutate_scratch
    )
    with pytest.raises(ProtocolError, match="bytes differ"):
        quarantine_failed_terminal_lineage(
            root,
            artifact_destination=artifact_destination,
            scratch_destination=scratch_destination,
        )
    assert artifact_destination.is_dir()
    assert scratch_destination.is_dir()
    assert not root.exists()
    assert not scratch.exists()
