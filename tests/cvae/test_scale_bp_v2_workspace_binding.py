from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2 import execution_admission
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.identity import (
    CLAIM_SCOPE,
    DIRECT_INPUT_ARTIFACT_IDS,
    EXPERIMENT_ID,
    GovernanceError,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.workspace_manifest import (
    WORKSPACE_INPUT_MEMBERS,
    validate_workspace_input_bindings,
)


def _manifest_and_paths(tmp_path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    rows: list[dict[str, object]] = []
    paths: dict[str, Path] = {}
    for index, artifact_id in enumerate(sorted(DIRECT_INPUT_ARTIFACT_IDS)):
        root = tmp_path / f"input-{index}"
        root.mkdir()
        member = WORKSPACE_INPUT_MEMBERS[artifact_id]
        target = root if not member else root / member
        if member:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"{artifact_id}\n", encoding="utf-8")
        rows.append(
            {
                "artifact_id": artifact_id,
                "resolved_path": str(root),
                "exists": True,
                "semantic_identities": {},
                "file_integrity": {},
            }
        )
        paths[artifact_id] = target
    payload = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": EXPERIMENT_ID,
        "stage": "90_oracles_and_diagnostics",
        "claim_scope": CLAIM_SCOPE,
        "selection_used_target_eval_artifacts": False,
        "repository_revision": "test-revision",
        "repository_dirty": False,
        "repository_status_hash": "test-status-hash",
        "input_artifacts": rows,
    }
    return payload, paths


def test_exact_workspace_input_bindings_pass(tmp_path: Path) -> None:
    manifest, paths = _manifest_and_paths(tmp_path)
    observed = validate_workspace_input_bindings(
        manifest,
        catalog_roots_by_artifact_id={
            artifact_id: row["resolved_path"]
            for artifact_id in DIRECT_INPUT_ARTIFACT_IDS
            for row in manifest["input_artifacts"]
            if row["artifact_id"] == artifact_id
        },
        resolved_paths_by_artifact_id=paths,
    )
    assert tuple(observed) == DIRECT_INPUT_ARTIFACT_IDS


def test_workspace_input_binding_rejects_role_swap(tmp_path: Path) -> None:
    manifest, paths = _manifest_and_paths(tmp_path)
    first, second = DIRECT_INPUT_ARTIFACT_IDS[:2]
    paths[first], paths[second] = paths[second], paths[first]
    with pytest.raises(GovernanceError, match="input binding drifted"):
        validate_workspace_input_bindings(
            manifest,
            catalog_roots_by_artifact_id={
                str(row["artifact_id"]): str(row["resolved_path"])
                for row in manifest["input_artifacts"]
            },
            resolved_paths_by_artifact_id=paths,
        )


def test_workspace_input_binding_rejects_identical_copy_outside_root(
    tmp_path: Path,
) -> None:
    manifest, paths = _manifest_and_paths(tmp_path)
    artifact_id = DIRECT_INPUT_ARTIFACT_IDS[-1]
    copy = tmp_path / "outside" / paths[artifact_id].name
    copy.parent.mkdir()
    copy.write_bytes(paths[artifact_id].read_bytes())
    paths[artifact_id] = copy
    with pytest.raises(GovernanceError, match="input binding drifted"):
        validate_workspace_input_bindings(
            manifest,
            catalog_roots_by_artifact_id={
                str(row["artifact_id"]): str(row["resolved_path"])
                for row in manifest["input_artifacts"]
            },
            resolved_paths_by_artifact_id=paths,
        )


def test_workspace_input_binding_rejects_coordinated_manifest_and_config_edit(
    tmp_path: Path,
) -> None:
    manifest, paths = _manifest_and_paths(tmp_path)
    catalog_roots = {
        str(row["artifact_id"]): str(row["resolved_path"])
        for row in manifest["input_artifacts"]
    }
    artifact_id = DIRECT_INPUT_ARTIFACT_IDS[-1]
    forged_root = tmp_path / "forged-root"
    forged_root.mkdir()
    member = WORKSPACE_INPUT_MEMBERS[artifact_id]
    forged = forged_root / member
    forged.write_bytes(paths[artifact_id].read_bytes())
    for row in manifest["input_artifacts"]:
        if row["artifact_id"] == artifact_id:
            row["resolved_path"] = str(forged_root)
    paths[artifact_id] = forged

    with pytest.raises(GovernanceError, match="catalog binding drifted"):
        validate_workspace_input_bindings(
            manifest,
            catalog_roots_by_artifact_id=catalog_roots,
            resolved_paths_by_artifact_id=paths,
        )


def test_workspace_binding_precedes_authorization_lease_check() -> None:
    source = inspect.getsource(execution_admission.admit_single_use_execution)
    assert source.index("validate_workspace_input_bindings") < source.index(
        "assert_authorization_unclaimed"
    )
