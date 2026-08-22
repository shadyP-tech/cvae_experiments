from __future__ import annotations

import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router import (
    preflight as cbpupr_preflight,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router import (
    run_admission as run_admission_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.config_payloads import (
    canonical_runtime_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.constants import (
    RUN_RECOVERY_POLICY,
    SCRATCH_ROOT,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.run_admission import (
    FAILED_WORKSTATION_PREFLIGHT_FILES,
    audit_failed_workstation_preflight_for_quarantine,
    quarantine_failed_workstation_preflight,
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
    monkeypatch.setattr(run_admission_module, "SCRATCH_ROOT", str(scratch))

    certificate = audit_failed_workstation_preflight_for_quarantine(root)
    assert certificate["status"] == "PASS"
    assert certificate["label_capability_opened"] is False
    assert certificate["physical_generation_started"] is False
    assert certificate["quarantined_bytes_may_feed_rerun"] is False
    assert certificate["eligible_next_action"] == (
        "MOVE_WHOLE_FAILED_ROOT_TO_QUARANTINE_ONLY"
    )

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

    with (root / ".run.lock").open("rb") as handle:
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
