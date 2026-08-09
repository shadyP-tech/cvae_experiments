from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    assert_closed_world,
    validate_content_index,
    write_content_index,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _materialize_index_members(root: Path) -> None:
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{member}\n", encoding="utf-8")


def test_bundle_contains_fitted_fold_provenance_and_no_target_surface() -> None:
    assert "tables/crossfit_fold_audits.csv" in REQUIRED_FILES
    assert "tables/crossfit_fold_audits.csv" in CONTENT_INDEX_MEMBERS
    assert not any(
        "target_prediction" in member or "target_action" in member
        for member in REQUIRED_FILES
    )


def test_content_index_resume_rejects_tamper_without_repair(tmp_path: Path) -> None:
    _materialize_index_members(tmp_path)
    write_content_index(tmp_path, config_contract_hash="a" * 64)
    validate_content_index(tmp_path, config_contract_hash="a" * 64)
    path = tmp_path / "manifests/content_index.json"
    tampered = path.read_bytes() + b"tamper"
    path.write_bytes(tampered)

    with pytest.raises(ProtocolError):
        write_content_index(tmp_path, config_contract_hash="a" * 64)
    assert path.read_bytes() == tampered


def test_strict_closed_world_rejects_orphan_task_checkpoint(tmp_path: Path) -> None:
    for member in REQUIRED_FILES:
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoints/orphan-task.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="orphan-task"):
        assert_closed_world(tmp_path, allow_incomplete=False)


def test_validator_fails_content_bytes_before_scientific_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit import (
        validation,
    )

    config = SimpleNamespace(
        contract_hash="a" * 64,
        artifact_root=tmp_path.resolve(),
        input_artifact_ids=("input",),
    )
    reached_reconstruction: list[bool] = []
    monkeypatch.setattr(validation, "assert_closed_world", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        validation,
        "load_utility_aligned_case_aware_proxy_information_audit_config",
        lambda _path: config,
    )
    monkeypatch.setattr(
        validation,
        "validate_content_index",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ProtocolError("content byte tamper")
        ),
    )
    monkeypatch.setattr(
        validation,
        "assert_input_fence",
        lambda _config: reached_reconstruction.append(True),
    )
    with pytest.raises(ProtocolError, match="content byte tamper"):
        validation.validate_case_aware_proxy_information_audit_bundle(
            tmp_path, config=config
        )
    assert reached_reconstruction == []
