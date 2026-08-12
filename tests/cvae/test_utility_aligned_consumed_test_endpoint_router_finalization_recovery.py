from __future__ import annotations

import json
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router import (
    finalization_recovery,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.recovery import (
    EXACT_EXISTING_SNAPSHOT_UTILITY_ALIGNED_CONSUMED_TEST_ENDPOINT_ROUTER_V1,
    detect_registered_exact_recovery,
)


def _write_failed_finalization_inventory(root: Path) -> None:
    assert len(finalization_recovery.FINALIZATION_RECOVERY_FILES) == 41
    for relative in finalization_recovery.FINALIZATION_RECOVERY_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "reports/run_state.json":
            state = {
                **finalization_recovery.FAILED_FEATURE_PARTITION_BINDING_STATE,
                "updated_at_utc": "2026-08-12T13:19:07.472578+00:00",
            }
            path.write_text(json.dumps(state), encoding="utf-8")
        else:
            path.write_bytes(b"sealed")


def _write_complete_inventory(root: Path) -> None:
    assert len(finalization_recovery.COMPLETE_REVALIDATION_FILES) == 42
    for relative in finalization_recovery.COMPLETE_REVALIDATION_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "reports/run_state.json":
            state = {
                **finalization_recovery.COMPLETE_ENDPOINT_ROUTER_STATE,
                "updated_at_utc": "2026-08-12T13:51:56.541023+00:00",
            }
            path.write_text(json.dumps(state), encoding="utf-8")
        else:
            path.write_bytes(b"sealed")


def test_finalization_recovery_accepts_only_exact_pending_validation_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _write_failed_finalization_inventory(root)
    (root / ".run.lock").write_bytes(b"owned transient lock")

    assert finalization_recovery.detect_feature_partition_binding_finalization_recovery(
        root
    )


@pytest.mark.parametrize("drift", ("missing", "extra"))
def test_finalization_recovery_rejects_inventory_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    root = tmp_path / "artifact"
    _write_failed_finalization_inventory(root)
    if drift == "missing":
        (root / "tables/aggregate_contrasts.csv").unlink()
    else:
        unexpected = root / "checkpoints/target_predictions/unexpected.json"
        unexpected.parent.mkdir(parents=True)
        unexpected.write_bytes(b"unsafe")

    with pytest.raises(ProtocolError, match="finalization recovery boundary drifted"):
        finalization_recovery.detect_feature_partition_binding_finalization_recovery(
            root
        )


def test_finalization_recovery_rejects_symlinked_member(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_failed_finalization_inventory(root)
    member = root / "tables/aggregate_contrasts.csv"
    member.unlink()
    external = tmp_path / "external.csv"
    external.write_bytes(b"sealed")
    member.symlink_to(external)

    with pytest.raises(ProtocolError, match="artifact tree contains symlinks"):
        finalization_recovery.detect_feature_partition_binding_finalization_recovery(
            root
        )


def test_workspace_dispatches_to_composite_endpoint_recovery(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_failed_finalization_inventory(root)

    assert detect_registered_exact_recovery(
        EXACT_EXISTING_SNAPSHOT_UTILITY_ALIGNED_CONSUMED_TEST_ENDPOINT_ROUTER_V1,
        root,
    )


def test_complete_revalidation_accepts_only_exact_terminal_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _write_complete_inventory(root)
    (root / ".run.lock").write_bytes(b"owned transient lock")

    assert finalization_recovery.detect_complete_endpoint_router_revalidation(root)
    assert detect_registered_exact_recovery(
        EXACT_EXISTING_SNAPSHOT_UTILITY_ALIGNED_CONSUMED_TEST_ENDPOINT_ROUTER_V1,
        root,
    )


@pytest.mark.parametrize("drift", ("missing", "extra", "state"))
def test_complete_revalidation_rejects_terminal_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    root = tmp_path / "artifact"
    _write_complete_inventory(root)
    if drift == "missing":
        (root / "reports/validation_report.json").unlink()
    elif drift == "extra":
        unexpected = root / "checkpoints/target_predictions/unexpected.json"
        unexpected.parent.mkdir(parents=True)
        unexpected.write_bytes(b"unsafe")
    else:
        state_path = root / "reports/run_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["promotion_eligible"] = True
        state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ProtocolError, match="complete revalidation boundary drifted"):
        finalization_recovery.detect_complete_endpoint_router_revalidation(root)
