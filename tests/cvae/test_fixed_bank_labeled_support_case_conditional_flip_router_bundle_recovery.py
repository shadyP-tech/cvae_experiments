from __future__ import annotations

import json
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router import (
    recovery,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.bundle import (
    REQUIRED_FILES,
    assert_closed_world,
    relative_files,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.recovery import (
    RUN_STATE_SCHEMA,
    detect_registered_flip_router_recovery,
    recovery_capability,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.recovery import (
    EXACT_EXISTING_SNAPSHOT_FIXED_BANK_LABELED_SUPPORT_CASE_CONDITIONAL_FLIP_ROUTER_V1,
    detect_registered_exact_recovery,
)


def _touch(root: Path, member: str, content: bytes = b"sealed") -> Path:
    path = root / member
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _state(*, phase: str, status: str = "FAILED") -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": RUN_STATE_SCHEMA,
        "phase": phase,
        "status": status,
        "terminal_consumed_test_diagnostic_only": True,
        "automatic_resume_requires_hash_validation": True,
    }
    if status == "FAILED":
        payload.update(
            error="injected interruption",
            error_class="RuntimeError",
        )
    return payload


def _write_phase_prefix(root: Path, phase: str) -> None:
    _touch(root, "config.resolved.yaml")
    _touch(root, "provenance/input_artifacts.json")
    state_path = _touch(root, "reports/run_state.json")
    state_path.write_text(json.dumps(_state(phase=phase)), encoding="utf-8")
    ordinal = recovery._PRELABEL_INDEX[phase]
    for _, group in recovery._PRELABEL_PHASES[:ordinal]:
        for member in group:
            if member != "reports/run_state.json":
                _touch(root, member)


def test_incomplete_closed_world_accepts_only_owned_exact_checkpoint_names(
    tmp_path: Path,
) -> None:
    _touch(tmp_path, "config.resolved.yaml")
    _touch(
        tmp_path,
        "checkpoints/fixed_bank_a1_action_predictions/"
        "tasks/target_0_train_17_generation_42.json",
    )
    assert_closed_world(tmp_path, allow_incomplete=True)

    _touch(
        tmp_path,
        "checkpoints/fixed_bank_a1_action_predictions/"
        "tasks/target_4_train_17_generation_42.json",
    )
    with pytest.raises(ProtocolError, match="closed-world inventory drifted"):
        assert_closed_world(tmp_path, allow_incomplete=True)


def test_complete_closed_world_is_exact_and_rejects_extras(tmp_path: Path) -> None:
    for member in REQUIRED_FILES:
        _touch(tmp_path, member)
    assert set(relative_files(tmp_path)) == set(REQUIRED_FILES)
    assert_closed_world(tmp_path, allow_incomplete=False)

    _touch(tmp_path, "reports/unregistered.json")
    with pytest.raises(ProtocolError, match="closed-world inventory drifted"):
        assert_closed_world(tmp_path, allow_incomplete=False)


def test_bundle_rejects_symlinked_files_and_directories(tmp_path: Path) -> None:
    external = tmp_path.parent / f"{tmp_path.name}-external"
    external.mkdir()
    (external / "payload.json").write_bytes(b"outside")
    (tmp_path / "manifests").symlink_to(external, target_is_directory=True)

    with pytest.raises(ProtocolError, match="contains a symlink"):
        relative_files(tmp_path)


def test_registered_recovery_accepts_exact_phase_prefix_and_workspace_dispatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _write_phase_prefix(root, "ADMISSION")
    (root / ".run.lock").write_bytes(b"owned transient lock")

    assert detect_registered_flip_router_recovery(root)
    capability = recovery_capability(root)
    assert capability is not None
    assert capability.mode == "PRELABEL_REPLAY"
    assert capability.resume_phase == "ADMISSION"
    assert capability.labels_may_be_reopened_for_validation is False
    assert capability.labels_may_be_opened_for_deterministic_policy_construction is True
    assert capability.labels_may_update_frozen_policy_contract is False
    assert detect_registered_exact_recovery(
        EXACT_EXISTING_SNAPSHOT_FIXED_BANK_LABELED_SUPPORT_CASE_CONDITIONAL_FLIP_ROUTER_V1,
        root,
    )


@pytest.mark.parametrize("drift", ("unknown_phase", "extra", "symlink"))
def test_registered_recovery_rejects_boundary_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    root = tmp_path / "artifact"
    _write_phase_prefix(root, "ADMISSION")
    if drift == "unknown_phase":
        (root / "reports/run_state.json").write_text(
            json.dumps(_state(phase="UNREGISTERED")),
            encoding="utf-8",
        )
    elif drift == "extra":
        _touch(root, "reports/unregistered.json")
    else:
        member = root / "config.resolved.yaml"
        member.unlink()
        external = tmp_path / "external.yaml"
        external.write_bytes(b"outside")
        member.symlink_to(external)

    with pytest.raises(ProtocolError):
        detect_registered_flip_router_recovery(root)


def test_registered_recovery_rejects_partial_checkpoint_pair(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_phase_prefix(root, "SOURCE_GENERATION")
    _touch(root, "checkpoints/frozen_source_streams/source_0_train_17.json")

    with pytest.raises(ProtocolError, match="partial checkpoint pair"):
        detect_registered_flip_router_recovery(root)


def test_postlabel_recovery_allows_label_reread_only_for_reconstructive_validation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    for member in recovery._POSTLABEL_PREFIX | recovery._TERMINAL_GROUP:
        _touch(root, member)
    (root / "reports/run_state.json").write_text(
        json.dumps(_state(phase="TERMINAL_EVALUATION")),
        encoding="utf-8",
    )

    capability = recovery_capability(root)
    assert capability is not None
    assert capability.mode == "TERMINAL_FINALIZATION"
    assert capability.resume_phase == "FINALIZATION"
    assert capability.validation_only is True
    assert capability.labels_may_be_reopened_for_validation is True
    assert capability.labels_may_be_opened_for_deterministic_policy_construction is False
    assert capability.labels_may_update_frozen_policy_contract is False


@pytest.mark.parametrize(
    ("phase", "prior", "sequence"),
    (
        ("DONOR_MODEL_FITTING", recovery._PRELABEL_COMPLETE, recovery._DONOR_SEQUENCE),
        (
            "FOLD_DECISION_SEALING",
            recovery._PRELABEL_COMPLETE | recovery._DONOR_GROUP,
            recovery._DECISION_SEQUENCE,
        ),
    ),
)
def test_label_aware_recovery_accepts_only_exact_deterministic_phase_prefixes(
    tmp_path: Path,
    phase: str,
    prior: frozenset[str],
    sequence: tuple[str, ...],
) -> None:
    root = tmp_path / "artifact"
    for member in prior | frozenset(sequence[: len(sequence) // 2]):
        _touch(root, member)
    (root / "reports/run_state.json").write_text(
        json.dumps(_state(phase=phase)), encoding="utf-8"
    )

    capability = recovery_capability(root)
    assert capability is not None
    assert capability.mode == "LABEL_AWARE_REPLAY"
    assert capability.validation_only is False
    assert capability.labels_may_be_opened_for_deterministic_policy_construction
    assert capability.labels_may_be_reopened_for_validation is False
    assert capability.labels_may_update_frozen_policy_contract is False

    _touch(root, "reports/foreign.json")
    with pytest.raises(ProtocolError, match="exact phase prefix"):
        recovery_capability(root)


def test_complete_recovery_requires_the_exact_terminal_inventory(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    for member in REQUIRED_FILES:
        _touch(root, member)
    (root / "reports/run_state.json").write_text(
        json.dumps(_state(phase="COMPLETE", status="COMPLETE")),
        encoding="utf-8",
    )
    assert detect_registered_flip_router_recovery(root)

    _touch(root, "checkpoints/unowned.json")
    with pytest.raises(ProtocolError, match="COMPLETE inventory drifted"):
        detect_registered_flip_router_recovery(root)
