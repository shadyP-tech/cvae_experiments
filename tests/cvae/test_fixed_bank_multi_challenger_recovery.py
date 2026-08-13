from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.recovery import (
    FAILED_MAPPINGPROXY_STATE,
    FINALIZATION_RECOVERABLE_INVENTORY,
    RECOVERABLE_INVENTORY,
    detect_registered_multi_challenger_recovery,
    failed_finalization_schema_state,
    recovery_capability,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json


def test_detector_accepts_only_exact_fold_plan_sealed_snapshot(
    tmp_path: Path,
) -> None:
    root = _write_recoverable_snapshot(tmp_path / "bundle")
    (root / ".run.lock").write_text("operational\n", encoding="utf-8")

    assert detect_registered_multi_challenger_recovery(root) is True


def test_detector_accepts_exact_validation_only_finalization_snapshot(
    tmp_path: Path,
) -> None:
    root = _write_finalization_snapshot(tmp_path / "bundle")
    (root / ".run.lock").write_text("operational\n", encoding="utf-8")

    capability = recovery_capability(root)

    assert capability is not None
    assert capability.mode == "FINALIZATION_VALIDATION"
    assert capability.state_phase == "FINALIZATION"
    assert capability.validation_only is True
    assert capability.labels_may_be_reopened_for_validation is True
    assert capability.scientific_products_may_be_recomputed is False
    assert capability.scientific_products_may_be_persisted is False
    assert detect_registered_multi_challenger_recovery(root) is True


def test_finalization_detector_requires_its_root_relative_error_path(
    tmp_path: Path,
) -> None:
    root = _write_finalization_snapshot(tmp_path / "bundle")
    atomic_json(
        root / "reports/run_state.json",
        failed_finalization_schema_state(tmp_path / "different-bundle"),
    )

    assert recovery_capability(root) is None
    assert detect_registered_multi_challenger_recovery(root) is False


def test_finalization_detector_rejects_state_or_inventory_drift(
    tmp_path: Path,
) -> None:
    root = _write_finalization_snapshot(tmp_path / "state-drift")
    atomic_json(
        root / "reports/run_state.json",
        {
            **failed_finalization_schema_state(root),
            "unregistered": True,
        },
    )
    with pytest.raises(ProtocolError, match="finalization recovery state drifted"):
        recovery_capability(root)

    root = _write_finalization_snapshot(tmp_path / "missing")
    (root / "tables/terminal_case_confusions.csv").unlink()
    with pytest.raises(ProtocolError, match="finalization recovery inventory drifted"):
        recovery_capability(root)

    root = _write_finalization_snapshot(tmp_path / "extra")
    (root / "reports/foreign.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(ProtocolError, match="finalization recovery inventory drifted"):
        recovery_capability(root)


def test_finalization_detector_accepts_durable_validation_report_retry(
    tmp_path: Path,
) -> None:
    root = _write_finalization_snapshot(tmp_path / "bundle")
    (root / "reports/validation_report.json").write_text(
        '{"status":"PASS"}\n', encoding="utf-8"
    )

    capability = recovery_capability(root)

    assert capability is not None
    assert capability.mode == "FINALIZATION_VALIDATION"

    (root / "reports/validation_report.json.123.tmp").write_text(
        '{"status":"PASS"}\n', encoding="utf-8"
    )
    assert recovery_capability(root) is not None

    root = _write_finalization_snapshot(tmp_path / "empty")
    (root / "tables/terminal_case_confusions.csv").write_bytes(b"")
    with pytest.raises(ProtocolError, match="empty durable member"):
        recovery_capability(root)


def test_finalization_detector_allows_only_redundant_owned_atomic_temp(
    tmp_path: Path,
) -> None:
    root = _write_finalization_snapshot(tmp_path / "bundle")
    temporary = root / "tables/terminal_case_confusions.csv.123.tmp"
    temporary.write_text("redundant\n", encoding="utf-8")

    assert recovery_capability(root) is not None

    (root / "tables/terminal_case_confusions.csv").unlink()
    with pytest.raises(ProtocolError, match="partial_atomic_bases"):
        recovery_capability(root)


def test_finalization_detector_rejects_symlinked_member(tmp_path: Path) -> None:
    root = _write_finalization_snapshot(tmp_path / "bundle")
    member = root / "tables/terminal_case_confusions.csv"
    outside = tmp_path / "outside.csv"
    outside.write_text("outside\n", encoding="utf-8")
    member.unlink()
    member.symlink_to(outside)

    with pytest.raises(ProtocolError, match="symlink"):
        recovery_capability(root)


def test_detector_returns_false_for_a_different_failure(tmp_path: Path) -> None:
    root = _write_recoverable_snapshot(tmp_path / "bundle")
    atomic_json(
        root / "reports/run_state.json",
        {**FAILED_MAPPINGPROXY_STATE, "error": "another failure"},
    )

    assert detect_registered_multi_challenger_recovery(root) is False


def test_detector_rejects_exact_failure_state_with_extra_metadata(
    tmp_path: Path,
) -> None:
    root = _write_recoverable_snapshot(tmp_path / "bundle")
    atomic_json(
        root / "reports/run_state.json",
        {**FAILED_MAPPINGPROXY_STATE, "updated_at_utc": "not-part-of-this-state"},
    )

    with pytest.raises(ProtocolError, match="state_matches=False"):
        detect_registered_multi_challenger_recovery(root)


@pytest.mark.parametrize(
    "mutation",
    ("missing", "extra", "empty"),
)
def test_detector_rejects_missing_extra_or_empty_durable_members(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _write_recoverable_snapshot(tmp_path / "bundle")
    if mutation == "missing":
        (root / "manifests/fold_plan_seals.json").unlink()
        match = "inventory drifted"
    elif mutation == "extra":
        (root / "tables/model_fits.csv").write_text("partial donor\n", encoding="utf-8")
        match = "inventory drifted"
    else:
        (root / "manifests/fold_plan_seals.json").write_bytes(b"")
        match = "empty durable member"

    with pytest.raises(ProtocolError, match=match):
        detect_registered_multi_challenger_recovery(root)


def test_detector_allows_only_redundant_package_owned_atomic_remnants(
    tmp_path: Path,
) -> None:
    root = _write_recoverable_snapshot(tmp_path / "bundle")
    owned = root / "manifests/fold_plan_seals.json.123.tmp"
    owned.write_text("interrupted duplicate write\n", encoding="utf-8")

    assert detect_registered_multi_challenger_recovery(root) is True

    (root / "manifests/fold_plan_seals.json").unlink()
    with pytest.raises(ProtocolError, match="partial_atomic_bases"):
        detect_registered_multi_challenger_recovery(root)


def test_detector_rejects_foreign_or_future_atomic_remnants(tmp_path: Path) -> None:
    root = _write_recoverable_snapshot(tmp_path / "bundle")
    future = root / "tables/model_fits.csv.123.tmp"
    future.write_text("partial donor\n", encoding="utf-8")

    with pytest.raises(ProtocolError, match="unknown atomic remnant"):
        detect_registered_multi_challenger_recovery(root)


@pytest.mark.parametrize("symlink_role", ("root", "file", "directory"))
def test_detector_rejects_symlinks(
    tmp_path: Path,
    symlink_role: str,
) -> None:
    real = _write_recoverable_snapshot(tmp_path / "real")
    if symlink_role == "root":
        root = tmp_path / "bundle"
        root.symlink_to(real, target_is_directory=True)
    else:
        root = real
        target = tmp_path / "outside"
        target.write_text("outside\n", encoding="utf-8")
        if symlink_role == "file":
            member = root / "manifests/fold_plan_seals.json"
            member.unlink()
            member.symlink_to(target)
        else:
            extra = root / "foreign"
            extra.symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(ProtocolError, match="symlink"):
        detect_registered_multi_challenger_recovery(root)


def test_only_root_operational_lock_is_excluded(tmp_path: Path) -> None:
    root = _write_recoverable_snapshot(tmp_path / "bundle")
    nested = root / "manifests/.run.lock"
    nested.write_text("not operational\n", encoding="utf-8")

    with pytest.raises(ProtocolError, match="inventory drifted"):
        detect_registered_multi_challenger_recovery(root)


def _write_recoverable_snapshot(root: Path) -> Path:
    for relative in RECOVERABLE_INVENTORY:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "reports/run_state.json":
            atomic_json(path, FAILED_MAPPINGPROXY_STATE)
        else:
            path.write_bytes(f"fixture:{relative}\n".encode("utf-8"))
    return root


def _write_finalization_snapshot(root: Path) -> Path:
    for relative in FINALIZATION_RECOVERABLE_INVENTORY:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "reports/run_state.json":
            atomic_json(path, failed_finalization_schema_state(root))
        else:
            path.write_bytes(f"fixture:{relative}\n".encode("utf-8"))
    return root
