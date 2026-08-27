"""Pure workspace-manifest contract for SCALE-BP v2 launch provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import yaml

from .experiment_contracts import LEDGER_AMENDMENT_FILENAME, validate_exact_input_fence
from .identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    CLAIM_SCOPE,
    CANONICAL_OUTPUT_RELATIVE_ROOT,
    DIRECT_INPUT_ARTIFACT_IDS,
    EXPERIMENT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    GovernanceError,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)


WORKSPACE_INPUT_MANIFEST_SCHEMA = "midogpp_input_artifacts_v2"
WORKSPACE_DATASET_ID = "midogpp"
WORKSPACE_STAGE_ID = "90_oracles_and_diagnostics"
WORKSPACE_REPLAY_FIELDS = (
    "schema_version",
    "dataset_id",
    "experiment_id",
    "stage",
    "claim_scope",
    "selection_used_target_eval_artifacts",
    "repository_revision",
    "repository_dirty",
    "repository_status_hash",
)
WORKSPACE_INPUT_MEMBERS = {
    EXPERT_BANK_ARTIFACT_ID: "",
    GENERATION_LOCK_ARTIFACT_ID: "",
    TEST_CACHE_ARTIFACT_ID: "",
    TEST_MANIFEST_ARTIFACT_ID: "manifest.csv",
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID: "reports/test_consumption_ledger.json",
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID: LEDGER_AMENDMENT_FILENAME,
}


def validate_workspace_manifest(payload: Mapping[str, object]) -> None:
    """Validate transport identity and exact-six provenance rows."""

    rows = payload.get("input_artifacts")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise GovernanceError("SCALE-BP v2 workspace input rows are malformed.")
    artifact_ids: Sequence[object] = tuple(
        row.get("artifact_id") for row in rows if isinstance(row, Mapping)
    )
    resolved_paths = tuple(
        str(row.get("resolved_path", "")) for row in rows if isinstance(row, Mapping)
    )
    validate_exact_input_fence(DIRECT_INPUT_ARTIFACT_IDS, resolved_paths=resolved_paths)
    if (
        tuple(artifact_ids) != tuple(sorted(DIRECT_INPUT_ARTIFACT_IDS))
        or len(set(str(value) for value in artifact_ids)) != len(artifact_ids)
        or payload.get("schema_version") != WORKSPACE_INPUT_MANIFEST_SCHEMA
        or payload.get("dataset_id") != WORKSPACE_DATASET_ID
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("stage") != WORKSPACE_STAGE_ID
        or payload.get("claim_scope") != CLAIM_SCOPE
        or payload.get("selection_used_target_eval_artifacts") is not False
        or not isinstance(payload.get("repository_revision"), str)
        or not payload.get("repository_revision")
        or payload.get("repository_dirty") not in {True, False, None}
        or not isinstance(payload.get("repository_status_hash"), str)
        or not payload.get("repository_status_hash")
    ):
        raise GovernanceError("SCALE-BP v2 workspace provenance header drifted.")
    for row in rows:
        assert isinstance(row, Mapping)
        if (
            row.get("exists") is not True
            or not isinstance(row.get("resolved_path"), str)
            or not row.get("resolved_path")
            or not isinstance(row.get("semantic_identities"), Mapping)
            or not isinstance(row.get("file_integrity"), Mapping)
        ):
            raise GovernanceError("SCALE-BP v2 workspace provenance row drifted.")


def validate_workspace_input_bindings(
    payload: Mapping[str, object],
    *,
    catalog_roots_by_artifact_id: Mapping[str, str | Path],
    resolved_paths_by_artifact_id: Mapping[str, str | Path],
) -> dict[str, str]:
    """Bind each resolved config role to its exact registered root and member."""

    validate_workspace_manifest(payload)
    if (
        set(catalog_roots_by_artifact_id) != set(DIRECT_INPUT_ARTIFACT_IDS)
        or set(resolved_paths_by_artifact_id) != set(DIRECT_INPUT_ARTIFACT_IDS)
    ):
        raise GovernanceError("SCALE-BP v2 workspace input binding keys drifted.")
    rows = payload.get("input_artifacts")
    assert isinstance(rows, list)
    rows_by_id = {
        str(row["artifact_id"]): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("artifact_id"), str)
    }
    if set(rows_by_id) != set(DIRECT_INPUT_ARTIFACT_IDS):
        raise GovernanceError("SCALE-BP v2 workspace input rows drifted.")

    canonical: dict[str, str] = {}
    for artifact_id in DIRECT_INPUT_ARTIFACT_IDS:
        row = rows_by_id[artifact_id]
        raw_root = Path(str(row.get("resolved_path", "")))
        if (
            not raw_root.is_absolute()
            or raw_root == Path(raw_root.anchor)
            or raw_root.is_symlink()
        ):
            raise GovernanceError("SCALE-BP v2 workspace artifact root is unsafe.")
        try:
            registered_root = raw_root.resolve(strict=True)
            catalog_root = Path(
                catalog_roots_by_artifact_id[artifact_id]
            ).resolve(strict=True)
        except OSError as exc:
            raise GovernanceError(
                "SCALE-BP v2 workspace artifact root is absent."
            ) from exc
        if not registered_root.is_dir():
            raise GovernanceError("SCALE-BP v2 workspace artifact root is not a directory.")
        if registered_root != catalog_root:
            raise GovernanceError(
                f"SCALE-BP v2 workspace catalog binding drifted: {artifact_id}."
            )

        member = WORKSPACE_INPUT_MEMBERS[artifact_id]
        registered_input = registered_root if not member else registered_root / member
        supplied_input = Path(resolved_paths_by_artifact_id[artifact_id])
        if (
            not supplied_input.is_absolute()
            or supplied_input == Path(supplied_input.anchor)
            or supplied_input.is_symlink()
            or registered_input.is_symlink()
        ):
            raise GovernanceError("SCALE-BP v2 workspace input path is unsafe.")
        try:
            expected = registered_input.resolve(strict=True)
            observed = supplied_input.resolve(strict=True)
        except OSError as exc:
            raise GovernanceError("SCALE-BP v2 workspace input path is absent.") from exc
        try:
            expected.relative_to(registered_root)
        except ValueError as exc:
            raise GovernanceError(
                "SCALE-BP v2 workspace input member escapes its artifact root."
            ) from exc
        if observed != expected:
            raise GovernanceError(
                f"SCALE-BP v2 workspace input binding drifted: {artifact_id}."
            )
        canonical[artifact_id] = str(observed)
    return canonical


def load_workspace_catalog_input_roots(
    artifact_root: str | Path,
) -> dict[str, Path]:
    """Replay the six manifest roots against the active workspace catalog."""

    output = Path(artifact_root)
    if not output.is_absolute() or output == Path(output.anchor):
        raise GovernanceError("SCALE-BP v2 workspace output root is unsafe.")
    relative_output = Path(CANONICAL_OUTPUT_RELATIVE_ROOT)
    workspace_root = output.resolve(strict=True)
    for _ in relative_output.parts:
        workspace_root = workspace_root.parent
    if (workspace_root / relative_output).resolve(strict=True) != output.resolve(
        strict=True
    ):
        raise GovernanceError("SCALE-BP v2 workspace root binding drifted.")

    catalog_path = workspace_root / "experiments/midogpp/artifact_catalog.yaml"
    try:
        catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise GovernanceError("SCALE-BP v2 workspace catalog is unreadable.") from exc
    if not isinstance(catalog, Mapping) or catalog.get("path_base") != "repository_root":
        raise GovernanceError("SCALE-BP v2 workspace catalog header drifted.")
    entries = catalog.get("artifacts")
    if not isinstance(entries, list):
        raise GovernanceError("SCALE-BP v2 workspace catalog rows are malformed.")

    roots: dict[str, Path] = {}
    for artifact_id in DIRECT_INPUT_ARTIFACT_IDS:
        matches = [
            row
            for row in entries
            if isinstance(row, Mapping) and row.get("artifact_id") == artifact_id
        ]
        if len(matches) != 1:
            raise GovernanceError(
                f"SCALE-BP v2 workspace catalog identity drifted: {artifact_id}."
            )
        row = matches[0]
        locations = [
            str(row[key])
            for key in ("physical_path", "canonical_path")
            if isinstance(row.get(key), str) and row.get(key)
        ]
        if len(locations) != 1:
            raise GovernanceError(
                f"SCALE-BP v2 workspace catalog location drifted: {artifact_id}."
            )
        raw = Path(locations[0])
        if raw.is_absolute() or ".." in raw.parts:
            raise GovernanceError("SCALE-BP v2 workspace catalog path is unsafe.")
        try:
            root = (workspace_root / raw).resolve(strict=True)
        except OSError as exc:
            raise GovernanceError(
                f"SCALE-BP v2 workspace catalog root is absent: {artifact_id}."
            ) from exc
        if not root.is_dir() or root.is_symlink():
            raise GovernanceError("SCALE-BP v2 workspace catalog root is unsafe.")
        roots[artifact_id] = root
    return roots


__all__ = (
    "WORKSPACE_DATASET_ID",
    "WORKSPACE_INPUT_MANIFEST_SCHEMA",
    "WORKSPACE_INPUT_MEMBERS",
    "WORKSPACE_REPLAY_FIELDS",
    "WORKSPACE_STAGE_ID",
    "load_workspace_catalog_input_roots",
    "validate_workspace_input_bindings",
    "validate_workspace_manifest",
)
