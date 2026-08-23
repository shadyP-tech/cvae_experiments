from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4 import (
    unadmitted_quarantine,
    workspace_inputs,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.experiment_contracts import (
    INPUT_ARTIFACT_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.identity import (
    EXPERIMENT_ID,
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.lineage import (
    reconstruct_persisted_six_input_binding,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.unadmitted_quarantine import (
    quarantine_unadmitted_workspace_skeleton,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.workspace_inputs import (
    validate_workspace_provenance,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json


_HASH = "a" * 64


def _provenance_rows(root: Path) -> dict[str, dict[str, object]]:
    return {
        artifact_id: {
            "artifact_id": artifact_id,
            "resolved_path": str(root / artifact_id),
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


def _provenance_payload(rows: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": EXPERIMENT_ID,
        "stage": "90_oracles_and_diagnostics",
        "claim_scope": "diagnostic_only",
        "selection_used_target_eval_artifacts": False,
        "repository_revision": "fixture-revision",
        "repository_dirty": False,
        "repository_status_hash": _HASH,
        "input_artifacts": [rows[key] for key in sorted(rows)],
    }


def _config(rows: dict[str, dict[str, object]]) -> SimpleNamespace:
    paths = {
        artifact_id: Path(str(row["resolved_path"]))
        for artifact_id, row in rows.items()
    }
    return SimpleNamespace(
        input_artifact_ids=INPUT_ARTIFACT_IDS,
        protocol={"protocol_hash": _HASH},
        expert_bank_root=paths[INPUT_ARTIFACT_IDS[0]],
        generation_lock_root=paths[INPUT_ARTIFACT_IDS[1]],
        test_cache_root=paths[INPUT_ARTIFACT_IDS[2]],
        test_manifest_path=paths[INPUT_ARTIFACT_IDS[3]] / "manifest.csv",
        test_consumption_ledger_path=(
            paths[INPUT_ARTIFACT_IDS[4]] / "reports/test_consumption_ledger.json"
        ),
        ledger_amendment_path=paths[INPUT_ARTIFACT_IDS[5]] / "amendment.json",
    )


def _write_skeleton(root: Path) -> None:
    root.mkdir()
    for relative in ("manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir()
    (root / "config.resolved.yaml").write_text("experiment: {}\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")


def _bind_quarantine_paths(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    dedicated_scratch: Path,
) -> None:
    monkeypatch.setattr(
        unadmitted_quarantine,
        "_canonical_output_root",
        lambda: root.resolve(),
    )
    monkeypatch.setattr(
        unadmitted_quarantine,
        "CANONICAL_SCRATCH_ROOT",
        str(dedicated_scratch),
    )


def test_v4_transport_schema_passes_both_consumers_and_poison_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _provenance_rows(tmp_path / "inputs")
    payload = _provenance_payload(rows)
    config = _config(rows)

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
    poisoned["input_artifacts"][0]["semantic_identities"]["version"] = "poisoned"
    atomic_json(path, poisoned)
    after = reconstruct_persisted_six_input_binding(tmp_path, config)
    assert after.binding_hash != before.binding_hash
    with pytest.raises(ProtocolError, match="provenance replay differs"):
        validate_workspace_provenance(tmp_path, config)

    repository_poison = json.loads(json.dumps(payload))
    repository_poison["repository_status_hash"] = "b" * 64
    atomic_json(path, repository_poison)
    with pytest.raises(ProtocolError, match="provenance replay differs"):
        validate_workspace_provenance(tmp_path, config)

    poisoned["schema_version"] = "midogpp_input_artifacts_v4"
    atomic_json(path, poisoned)
    with pytest.raises(ProtocolError, match="provenance header drifted"):
        reconstruct_persisted_six_input_binding(tmp_path, config)
    with pytest.raises(ProtocolError, match="provenance header drifted"):
        validate_workspace_provenance(tmp_path, config)


def test_v4_quarantines_only_the_exact_unadmitted_skeleton(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    _write_skeleton(root)
    _bind_quarantine_paths(monkeypatch, root, tmp_path / "dedicated-scratch")
    destination = root.with_name(
        root.name + ".quarantine-unadmitted-provenance-header-20260823T201603Z"
    )

    receipt = quarantine_unadmitted_workspace_skeleton(
        root,
        destination=destination,
    )

    assert receipt["authorization_consumed"] is False
    assert receipt["run_lock_present"] is False
    assert receipt["run_state_present"] is False
    assert receipt["scratch_present"] is False
    assert not root.exists()
    assert destination.is_dir()
    persisted = read_json(Path(str(destination) + ".receipt.json"))
    assert persisted == receipt
    assert quarantine_unadmitted_workspace_skeleton(
        root,
        destination=destination,
    ) == receipt


@pytest.mark.parametrize(
    "relative",
    (
        ".run.lock",
        "reports/run_state.json",
        "tables/science.json",
        "manifests/nested/foreign.json",
    ),
)
def test_v4_quarantine_rejects_any_attempt_or_foreign_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    root = tmp_path / "artifact"
    _write_skeleton(root)
    member = root / relative
    member.parent.mkdir(parents=True, exist_ok=True)
    member.write_text("{}\n", encoding="utf-8")
    _bind_quarantine_paths(monkeypatch, root, tmp_path / "dedicated-scratch")
    destination = root.with_name(
        root.name + ".quarantine-unadmitted-provenance-header-20260823T201603Z"
    )

    with pytest.raises(ProtocolError, match="partial, foreign, or prior-run"):
        quarantine_unadmitted_workspace_skeleton(root, destination=destination)
    assert root.is_dir()
    assert not destination.exists()


@pytest.mark.parametrize("role", ("dedicated", "fallback"))
def test_v4_quarantine_rejects_either_scratch_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    root = tmp_path / "artifact"
    _write_skeleton(root)
    dedicated = tmp_path / "dedicated-scratch"
    _bind_quarantine_paths(monkeypatch, root, dedicated)
    scratch = (
        dedicated
        if role == "dedicated"
        else root.parent / f".{root.name}.pdcaps-v4-scratch"
    )
    scratch.mkdir()
    destination = root.with_name(
        root.name + ".quarantine-unadmitted-provenance-header-20260823T201603Z"
    )

    with pytest.raises(ProtocolError, match="scratch exists"):
        quarantine_unadmitted_workspace_skeleton(root, destination=destination)
    assert root.is_dir()
    assert not destination.exists()


def test_v4_quarantine_rejects_aliases_and_non_sibling_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    _write_skeleton(root)
    _bind_quarantine_paths(monkeypatch, root, tmp_path / "dedicated-scratch")
    alias = tmp_path / "artifact-link"
    alias.symlink_to(root, target_is_directory=True)
    safe_name = (
        root.name + ".quarantine-unadmitted-provenance-header-20260823T201603Z"
    )

    with pytest.raises(ProtocolError, match="not absolute and real"):
        quarantine_unadmitted_workspace_skeleton(
            alias,
            destination=tmp_path / safe_name,
        )
    foreign_parent = tmp_path / "foreign"
    foreign_parent.mkdir()
    with pytest.raises(ProtocolError, match="not a safe sibling"):
        quarantine_unadmitted_workspace_skeleton(
            root,
            destination=foreign_parent / safe_name,
        )
    wrong = tmp_path / "wrong-artifact"
    _write_skeleton(wrong)
    wrong_destination = wrong.with_name(
        wrong.name + ".quarantine-unadmitted-provenance-header-20260823T201603Z"
    )
    with pytest.raises(ProtocolError, match="root identity drifted"):
        quarantine_unadmitted_workspace_skeleton(
            wrong,
            destination=wrong_destination,
        )
    assert root.is_dir()


def test_v4_quarantine_resumes_after_move_before_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    _write_skeleton(root)
    _bind_quarantine_paths(monkeypatch, root, tmp_path / "dedicated-scratch")
    destination = root.with_name(
        root.name + ".quarantine-unadmitted-provenance-header-20260823T201603Z"
    )
    original = unadmitted_quarantine._finalize_moved_transition

    def interrupt_after_move(*_args, **_kwargs):
        raise ProtocolError("injected receipt interruption")

    monkeypatch.setattr(
        unadmitted_quarantine,
        "_finalize_moved_transition",
        interrupt_after_move,
    )
    with pytest.raises(ProtocolError, match="injected receipt interruption"):
        quarantine_unadmitted_workspace_skeleton(root, destination=destination)
    assert not root.exists()
    assert destination.is_dir()
    assert Path(str(destination) + ".receipt.pending.json").is_file()
    assert not Path(str(destination) + ".receipt.json").exists()

    monkeypatch.setattr(
        unadmitted_quarantine,
        "_finalize_moved_transition",
        original,
    )
    receipt = quarantine_unadmitted_workspace_skeleton(
        root,
        destination=destination,
    )
    assert receipt["status"] == "PASS"
    assert not Path(str(destination) + ".receipt.pending.json").exists()


def test_v4_quarantine_wraps_and_resumes_a_move_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    _write_skeleton(root)
    _bind_quarantine_paths(monkeypatch, root, tmp_path / "dedicated-scratch")
    destination = root.with_name(
        root.name + ".quarantine-unadmitted-provenance-header-20260823T201603Z"
    )
    original = unadmitted_quarantine.os.rename

    def fail_move(*_args, **_kwargs):
        raise OSError("injected move failure")

    monkeypatch.setattr(unadmitted_quarantine.os, "rename", fail_move)
    with pytest.raises(ProtocolError, match="skeleton move failed"):
        quarantine_unadmitted_workspace_skeleton(root, destination=destination)
    assert root.is_dir()
    assert Path(str(destination) + ".receipt.pending.json").is_file()

    monkeypatch.setattr(unadmitted_quarantine.os, "rename", original)
    receipt = quarantine_unadmitted_workspace_skeleton(
        root,
        destination=destination,
    )
    assert receipt["status"] == "PASS"
