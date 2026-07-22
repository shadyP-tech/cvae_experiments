"""Fail-closed binding of complete CLA runs to the canonical workspace.

The workspace creates ``config.resolved.yaml`` and
``provenance/input_artifacts.json`` before launching the scientific runner.
This module verifies those snapshots, the live registry/catalog declaration,
and the two frozen input bytes before any model fit is allowed to start.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError

from ..artifacts import stable_hash
from ..protocol import ProtocolError
from .schema import (
    CLA_CANONICAL_OUTPUT_PATH,
    CLA_CLAIM_SCOPE,
    CLA_COMPLETE_REQUIRED_OUTPUTS,
    CLA_EXPECTED_FEATURE_CACHE_SHA256,
    CLA_EXPECTED_MANIFEST_SHA256,
    CLA_EXPERIMENT_ID,
    CLA_INPUT_ARTIFACT_IDS,
    CLA_OUTPUT_ARTIFACT_ID,
    CLA_STAGE,
    WORKSPACE_BINDING_SCHEMA_VERSION,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static analysis
    from .config import ConditionalLogitAlignmentConfig


_MANIFEST_ARTIFACT_ID = CLA_INPUT_ARTIFACT_IDS[0]
_FEATURE_CACHE_ARTIFACT_ID = CLA_INPUT_ARTIFACT_IDS[1]
_MANIFEST_MEMBER = "manifest.csv"
_FEATURE_CACHE_MEMBER = "embeddings/train.pt"

_INPUT_METADATA: Mapping[str, Mapping[str, object]] = {
    _MANIFEST_ARTIFACT_ID: {
        "stage": "dataset_contract",
        "evidence_label": "AUDIT_ONLY",
        "claim_scope": "dataset_contract_and_split_provenance",
        "semantic_identities": {
            "contract_identity": "midogpp_annotation_patch_v1",
        },
        "member": _MANIFEST_MEMBER,
    },
    _FEATURE_CACHE_ARTIFACT_ID: {
        "stage": "derived_features",
        "evidence_label": "AUDIT_ONLY",
        "claim_scope": "feature_cache_provenance",
        "semantic_identities": {
            "feature_cache_hash": CLA_EXPECTED_FEATURE_CACHE_SHA256,
        },
        "member": _FEATURE_CACHE_MEMBER,
    },
}


def validate_production_workspace_binding(
    config: "ConditionalLogitAlignmentConfig",
    *,
    artifact_root_override: Path | None = None,
    _workspace: MidogppWorkspace | None = None,
) -> dict[str, object]:
    """Validate and return the immutable workspace/input binding payload."""

    if config.allow_partial_test_coverage:
        raise ProtocolError("Partial-test CLA configs do not have a production workspace binding.")
    if config.config_source_path is None:
        raise ProtocolError(
            "Complete CLA runs must load the workspace-owned config.resolved.yaml snapshot."
        )

    config_path = Path(config.config_source_path).resolve()
    try:
        workspace = _workspace or MidogppWorkspace.load(_repo_root_for(config_path))
        workspace.validate()
        experiment = workspace.get_experiment(CLA_EXPERIMENT_ID)
    except WorkspaceError as exc:
        raise ProtocolError(f"CLA workspace binding is invalid: {exc}") from exc

    if not experiment.runnable:
        raise ProtocolError(
            f"CLA experiment is status={experiment.status!r}; a complete run requires a "
            "runnable registered status."
        )
    if (
        experiment.stage != CLA_STAGE
        or experiment.claim_scope != CLA_CLAIM_SCOPE
        or experiment.output_artifact_id != CLA_OUTPUT_ARTIFACT_ID
        or tuple(experiment.input_artifact_ids) != CLA_INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("CLA registered experiment stage/claim/input/output contract drifted.")

    output = workspace.artifacts.get(CLA_OUTPUT_ARTIFACT_ID)
    if output is None:
        raise ProtocolError("CLA registered output artifact is absent from the catalog.")
    if (
        output.canonical_path != CLA_CANONICAL_OUTPUT_PATH
        or output.migration != "canonical_output"
        or output.stage != CLA_STAGE
        or output.claim_scope != CLA_CLAIM_SCOPE
        or tuple(output.required_files) != CLA_COMPLETE_REQUIRED_OUTPUTS
        or output.may_feed_recipe_selection is not False
        or output.may_feed_deployable_selection is not False
    ):
        raise ProtocolError("CLA canonical output catalog contract drifted.")

    try:
        canonical_root = workspace.resolve_artifact(
            CLA_OUTPUT_ARTIFACT_ID,
            for_output=True,
            require_exists=False,
        ).resolve()
    except WorkspaceError as exc:
        raise ProtocolError(f"CLA registered output resolution failed: {exc}") from exc

    configured_root = Path(config.artifact_root).resolve()
    effective_root = (
        configured_root
        if artifact_root_override is None
        else Path(artifact_root_override).resolve()
    )
    expected_config_path = canonical_root / "config.resolved.yaml"
    provenance_path = canonical_root / "provenance/input_artifacts.json"
    if (
        configured_root != canonical_root
        or effective_root != canonical_root
        or config_path != expected_config_path
    ):
        raise ProtocolError(
            "Complete CLA runs require the catalog root, its config.resolved.yaml, "
            "and no artifact-root override drift."
        )
    try:
        manifest_root = workspace.resolve_artifact(
            _MANIFEST_ARTIFACT_ID,
            require_exists=True,
        ).resolve()
        feature_root = workspace.resolve_artifact(
            _FEATURE_CACHE_ARTIFACT_ID,
            require_exists=True,
        ).resolve()
    except WorkspaceError as exc:
        raise ProtocolError(f"CLA registered input resolution failed: {exc}") from exc
    manifest_path = (manifest_root / _MANIFEST_MEMBER).resolve()
    feature_cache_path = (feature_root / _FEATURE_CACHE_MEMBER).resolve()
    if (
        Path(config.manifest_path).resolve() != manifest_path
        or Path(config.feature_cache_path).resolve() != feature_cache_path
    ):
        raise ProtocolError("CLA resolved config inputs differ from the registered artifacts.")
    for required in (config_path, provenance_path, manifest_path, feature_cache_path):
        if not required.is_file():
            raise ProtocolError(f"CLA workspace binding file is missing: {required}")

    _validate_catalog_input_hash_contract(workspace)
    manifest_sha256 = _file_sha256(manifest_path)
    feature_cache_sha256 = _file_sha256(feature_cache_path)
    _validate_frozen_input_hashes(manifest_sha256, feature_cache_sha256)
    provenance = _read_json(provenance_path)
    _validate_workspace_provenance(
        provenance,
        manifest_root=manifest_root,
        manifest_path=manifest_path,
        feature_root=feature_root,
        feature_cache_path=feature_cache_path,
        manifest_sha256=manifest_sha256,
        feature_cache_sha256=feature_cache_sha256,
    )

    payload: dict[str, object] = {
        "schema_version": WORKSPACE_BINDING_SCHEMA_VERSION,
        "experiment_id": CLA_EXPERIMENT_ID,
        "registry_status": experiment.status,
        "stage": CLA_STAGE,
        "claim_scope": CLA_CLAIM_SCOPE,
        "output_artifact_id": CLA_OUTPUT_ARTIFACT_ID,
        "input_artifact_ids": list(CLA_INPUT_ARTIFACT_IDS),
        "artifact_root": str(canonical_root),
        "config_resolved_path": str(config_path),
        "config_resolved_sha256": _file_sha256(config_path),
        "input_provenance_path": str(provenance_path),
        "input_provenance_sha256": _file_sha256(provenance_path),
        "manifest": {
            "artifact_id": _MANIFEST_ARTIFACT_ID,
            "resolved_root": str(manifest_root),
            "resolved_path": str(manifest_path),
            "sha256": manifest_sha256,
        },
        "feature_cache": {
            "artifact_id": _FEATURE_CACHE_ARTIFACT_ID,
            "resolved_root": str(feature_root),
            "resolved_path": str(feature_cache_path),
            "sha256": feature_cache_sha256,
        },
    }
    payload["workspace_binding_hash"] = stable_hash(payload)
    return payload


def validate_persisted_workspace_binding(
    artifact_root: Path,
    frozen_binding: object,
    protocol_binding: object,
) -> None:
    """Independently recompute complete-bundle workspace and input hashes."""

    frozen = _mapping(frozen_binding, "frozen workspace binding")
    protocol = _mapping(protocol_binding, "protocol workspace binding")
    if dict(frozen) != dict(protocol):
        raise ProtocolError("CLA frozen/protocol workspace bindings differ.")
    payload = dict(frozen)
    persisted_hash = payload.pop("workspace_binding_hash", None)
    if persisted_hash != stable_hash(payload):
        raise ProtocolError("CLA workspace binding hash mismatch.")

    root = Path(artifact_root).resolve()
    config_path = root / "config.resolved.yaml"
    provenance_path = root / "provenance/input_artifacts.json"
    expected_scalars = {
        "schema_version": WORKSPACE_BINDING_SCHEMA_VERSION,
        "experiment_id": CLA_EXPERIMENT_ID,
        "stage": CLA_STAGE,
        "claim_scope": CLA_CLAIM_SCOPE,
        "output_artifact_id": CLA_OUTPUT_ARTIFACT_ID,
        "input_artifact_ids": list(CLA_INPUT_ARTIFACT_IDS),
        "artifact_root": str(root),
        "config_resolved_path": str(config_path),
        "input_provenance_path": str(provenance_path),
    }
    for field, expected in expected_scalars.items():
        if frozen.get(field) != expected:
            raise ProtocolError(f"CLA persisted workspace binding field {field} drifted.")
    if frozen.get("registry_status") not in {"active", "diagnostic"}:
        raise ProtocolError("CLA persisted registry status was not runnable.")
    if not config_path.is_file() or not provenance_path.is_file():
        raise ProtocolError("CLA complete bundle lacks workspace config/provenance snapshots.")
    if (
        frozen.get("config_resolved_sha256") != _file_sha256(config_path)
        or frozen.get("input_provenance_sha256") != _file_sha256(provenance_path)
    ):
        raise ProtocolError("CLA workspace config/provenance SHA256 binding mismatch.")

    manifest = _mapping(frozen.get("manifest"), "manifest binding")
    feature_cache = _mapping(frozen.get("feature_cache"), "feature-cache binding")
    manifest_root = Path(str(manifest.get("resolved_root", ""))).resolve()
    manifest_path = Path(str(manifest.get("resolved_path", ""))).resolve()
    feature_root = Path(str(feature_cache.get("resolved_root", ""))).resolve()
    feature_cache_path = Path(str(feature_cache.get("resolved_path", ""))).resolve()
    if (
        manifest.get("artifact_id") != _MANIFEST_ARTIFACT_ID
        or feature_cache.get("artifact_id") != _FEATURE_CACHE_ARTIFACT_ID
        or manifest_path != (manifest_root / _MANIFEST_MEMBER).resolve()
        or feature_cache_path != (feature_root / _FEATURE_CACHE_MEMBER).resolve()
        or not manifest_path.is_file()
        or not feature_cache_path.is_file()
    ):
        raise ProtocolError("CLA persisted registered input paths are invalid.")
    manifest_sha256 = _file_sha256(manifest_path)
    feature_cache_sha256 = _file_sha256(feature_cache_path)
    _validate_frozen_input_hashes(manifest_sha256, feature_cache_sha256)
    if (
        manifest.get("sha256") != manifest_sha256
        or feature_cache.get("sha256") != feature_cache_sha256
    ):
        raise ProtocolError("CLA persisted input SHA256 binding mismatch.")
    _validate_workspace_provenance(
        _read_json(provenance_path),
        manifest_root=manifest_root,
        manifest_path=manifest_path,
        feature_root=feature_root,
        feature_cache_path=feature_cache_path,
        manifest_sha256=manifest_sha256,
        feature_cache_sha256=feature_cache_sha256,
    )


def _validate_catalog_input_hash_contract(workspace: MidogppWorkspace) -> None:
    expectations = (
        (_MANIFEST_ARTIFACT_ID, _MANIFEST_MEMBER, CLA_EXPECTED_MANIFEST_SHA256),
        (
            _FEATURE_CACHE_ARTIFACT_ID,
            _FEATURE_CACHE_MEMBER,
            CLA_EXPECTED_FEATURE_CACHE_SHA256,
        ),
    )
    for artifact_id, member, digest in expectations:
        artifact = workspace.artifacts.get(artifact_id)
        expectation = None if artifact is None else artifact.expected_file_hashes.get(member)
        if (
            artifact is None
            or expectation is None
            or expectation.algorithm != "sha256"
            or expectation.digest != digest
        ):
            raise ProtocolError(
                f"CLA catalog SHA256 expectation drifted for {artifact_id}/{member}."
            )


def _validate_frozen_input_hashes(manifest: str, feature_cache: str) -> None:
    if manifest != CLA_EXPECTED_MANIFEST_SHA256:
        raise ProtocolError(
            "CLA manifest SHA256 differs from the frozen annotation-patch contract."
        )
    if feature_cache != CLA_EXPECTED_FEATURE_CACHE_SHA256:
        raise ProtocolError(
            "CLA feature-cache SHA256 differs from the frozen corrected-xyxy cache."
        )


def _validate_workspace_provenance(
    provenance: Mapping[str, object],
    *,
    manifest_root: Path,
    manifest_path: Path,
    feature_root: Path,
    feature_cache_path: Path,
    manifest_sha256: str,
    feature_cache_sha256: str,
) -> None:
    for field, expected in {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": CLA_EXPERIMENT_ID,
        "stage": CLA_STAGE,
        "claim_scope": CLA_CLAIM_SCOPE,
        "selection_used_target_eval_artifacts": False,
    }.items():
        if provenance.get(field) != expected:
            raise ProtocolError(f"CLA workspace provenance field {field} drifted.")
    rows = provenance.get("input_artifacts")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ProtocolError("CLA workspace provenance must contain exactly two inputs.")
    by_id: dict[str, Mapping[str, object]] = {}
    for raw in rows:
        row = _mapping(raw, "input provenance row")
        artifact_id = str(row.get("artifact_id", ""))
        if artifact_id in by_id:
            raise ProtocolError("CLA workspace provenance contains duplicate input IDs.")
        by_id[artifact_id] = row
    if set(by_id) != set(CLA_INPUT_ARTIFACT_IDS):
        raise ProtocolError("CLA workspace provenance input IDs are unregistered or missing.")

    expected_paths = {
        _MANIFEST_ARTIFACT_ID: (manifest_root, manifest_path, manifest_sha256),
        _FEATURE_CACHE_ARTIFACT_ID: (
            feature_root,
            feature_cache_path,
            feature_cache_sha256,
        ),
    }
    for artifact_id in CLA_INPUT_ARTIFACT_IDS:
        row = by_id[artifact_id]
        metadata = _INPUT_METADATA[artifact_id]
        root, member_path, digest = expected_paths[artifact_id]
        for field in ("stage", "evidence_label", "claim_scope", "semantic_identities"):
            if row.get(field) != metadata[field]:
                raise ProtocolError(
                    f"CLA workspace provenance {artifact_id} field {field} drifted."
                )
        if (
            Path(str(row.get("resolved_path", ""))).resolve() != root
            or row.get("semantic_identities_are_file_hashes") is not False
            or row.get("exists") is not True
        ):
            raise ProtocolError(f"CLA workspace provenance root drifted for {artifact_id}.")
        integrity = _mapping(row.get("file_integrity"), "file_integrity")
        if (
            integrity.get("status") != "EXPECTED_FILE_HASHES_MATCH"
            or integrity.get("default_recording_algorithm") != "sha256"
        ):
            raise ProtocolError(f"CLA input integrity status is invalid for {artifact_id}.")
        files = integrity.get("files")
        if not isinstance(files, list):
            raise ProtocolError(f"CLA input integrity files are malformed for {artifact_id}.")
        relevant = [
            _mapping(item, "input integrity file")
            for item in files
            if isinstance(item, Mapping) and item.get("path") == metadata["member"]
        ]
        if len(relevant) != 1:
            raise ProtocolError(
                f"CLA relevant input provenance member is missing for {artifact_id}."
            )
        file_row = relevant[0]
        if (
            Path(str(file_row.get("resolved_path", ""))).resolve() != member_path
            or file_row.get("exists") is not True
            or file_row.get("expected")
            != {"algorithm": "sha256", "digest": digest}
            or _mapping(file_row.get("computed"), "computed hashes").get("sha256")
            != digest
            or file_row.get("verification") != "MATCH"
            or int(file_row.get("size_bytes", -1)) != member_path.stat().st_size
        ):
            raise ProtocolError(f"CLA relevant input hash provenance drifted for {artifact_id}.")


def _repo_root_for(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "experiments/midogpp/registry.yaml").is_file():
            return candidate
    raise ProtocolError(
        "CLA config.resolved.yaml is not inside a MIDOG++ workspace checkout."
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"CLA workspace provenance is unreadable: {path}") from exc
    return _mapping(payload, "workspace provenance")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"CLA {label} must be a mapping.")
    return value


__all__ = [
    "validate_persisted_workspace_binding",
    "validate_production_workspace_binding",
]
