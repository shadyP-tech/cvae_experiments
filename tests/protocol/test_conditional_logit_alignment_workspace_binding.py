from __future__ import annotations

from copy import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from midogpp_thesis.real_features.classifier_reference.artifacts import stable_hash
from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.config import (
    ConditionalLogitAlignmentConfig,
)
from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.schema import (
    CLA_CANONICAL_OUTPUT_PATH,
    CLA_CLAIM_SCOPE,
    CLA_EXPECTED_FEATURE_CACHE_SHA256,
    CLA_EXPECTED_MANIFEST_SHA256,
    CLA_EXPERIMENT_ID,
    CLA_INPUT_ARTIFACT_IDS,
    CLA_OUTPUT_ARTIFACT_ID,
    CLA_STAGE,
    WORKSPACE_BINDING_SCHEMA_VERSION,
)
from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment import (
    workspace_binding as binding_module,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


def _config(
    workspace: MidogppWorkspace,
    *,
    artifact_root: Path,
    config_source_path: Path,
) -> ConditionalLogitAlignmentConfig:
    return ConditionalLogitAlignmentConfig(
        name="conditional_logit_alignment_v1",
        artifact_root=artifact_root,
        manifest_path=workspace.repo_root / "unused-manifest.csv",
        feature_cache_path=workspace.repo_root / "unused-cache.pt",
        config_source_path=config_source_path,
    )


def _workspace_with_status(status: str) -> MidogppWorkspace:
    workspace = MidogppWorkspace.load()
    cloned = copy(workspace)
    cloned.experiments = dict(workspace.experiments)
    cloned.experiments[CLA_EXPERIMENT_ID] = replace(
        workspace.get_experiment(CLA_EXPERIMENT_ID),
        status=status,
    )
    return cloned


def test_production_binding_rejects_planned_registry_status(tmp_path: Path) -> None:
    workspace = _workspace_with_status("planned")
    config = _config(
        workspace,
        artifact_root=tmp_path / "output",
        config_source_path=tmp_path / "config.resolved.yaml",
    )

    with pytest.raises(ProtocolError, match="status='planned'"):
        binding_module.validate_production_workspace_binding(
            config,
            _workspace=workspace,
        )


def test_production_binding_rejects_unregistered_input_set(tmp_path: Path) -> None:
    workspace = _workspace_with_status("diagnostic")
    workspace.experiments[CLA_EXPERIMENT_ID] = replace(
        workspace.get_experiment(CLA_EXPERIMENT_ID),
        input_artifact_ids=(CLA_INPUT_ARTIFACT_IDS[0],),
    )
    config = _config(
        workspace,
        artifact_root=tmp_path / "output",
        config_source_path=tmp_path / "config.resolved.yaml",
    )

    with pytest.raises(ProtocolError, match="input/output contract drifted"):
        binding_module.validate_production_workspace_binding(
            config,
            _workspace=workspace,
        )


def test_production_binding_rejects_noncanonical_root_and_override(
    tmp_path: Path,
) -> None:
    workspace = _workspace_with_status("diagnostic")
    canonical = (workspace.repo_root / CLA_CANONICAL_OUTPUT_PATH).resolve()
    noncanonical = tmp_path / "other"
    config = _config(
        workspace,
        artifact_root=noncanonical,
        config_source_path=noncanonical / "config.resolved.yaml",
    )
    with pytest.raises(ProtocolError, match="catalog root"):
        binding_module.validate_production_workspace_binding(
            config,
            _workspace=workspace,
        )

    canonical_config = replace(
        config,
        artifact_root=canonical,
        config_source_path=canonical / "config.resolved.yaml",
    )
    with pytest.raises(ProtocolError, match="artifact-root override drift"):
        binding_module.validate_production_workspace_binding(
            canonical_config,
            artifact_root_override=noncanonical,
            _workspace=workspace,
        )


def test_frozen_input_hash_constants_match_registered_contract() -> None:
    workspace = MidogppWorkspace.load()
    manifest = workspace.artifacts[CLA_INPUT_ARTIFACT_IDS[0]].expected_file_hashes[
        "manifest.csv"
    ]
    cache = workspace.artifacts[CLA_INPUT_ARTIFACT_IDS[1]].expected_file_hashes[
        "embeddings/train.pt"
    ]
    assert (manifest.algorithm, manifest.digest) == (
        "sha256",
        CLA_EXPECTED_MANIFEST_SHA256,
    )
    assert (cache.algorithm, cache.digest) == (
        "sha256",
        CLA_EXPECTED_FEATURE_CACHE_SHA256,
    )


def test_persisted_binding_recomputes_workspace_snapshot_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    config_path = root / "config.resolved.yaml"
    provenance_path = root / "provenance/input_artifacts.json"
    manifest_root = tmp_path / "dataset"
    feature_root = tmp_path / "features"
    manifest_path = manifest_root / "manifest.csv"
    feature_path = feature_root / "embeddings/train.pt"
    for path in (config_path, manifest_path, feature_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("resolved: true\n", encoding="utf-8")
    manifest_path.write_bytes(b"manifest fixture\n")
    feature_path.write_bytes(b"feature fixture\n")
    manifest_hash = _sha256(manifest_path)
    feature_hash = _sha256(feature_path)

    monkeypatch.setattr(binding_module, "CLA_EXPECTED_MANIFEST_SHA256", manifest_hash)
    monkeypatch.setattr(binding_module, "CLA_EXPECTED_FEATURE_CACHE_SHA256", feature_hash)
    monkeypatch.setattr(
        binding_module,
        "_INPUT_METADATA",
        {
            CLA_INPUT_ARTIFACT_IDS[0]: {
                "stage": "dataset_contract",
                "evidence_label": "AUDIT_ONLY",
                "claim_scope": "dataset_contract_and_split_provenance",
                "semantic_identities": {
                    "contract_identity": "midogpp_annotation_patch_v1"
                },
                "member": "manifest.csv",
            },
            CLA_INPUT_ARTIFACT_IDS[1]: {
                "stage": "derived_features",
                "evidence_label": "AUDIT_ONLY",
                "claim_scope": "feature_cache_provenance",
                "semantic_identities": {"feature_cache_hash": feature_hash},
                "member": "embeddings/train.pt",
            },
        },
    )
    provenance = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": CLA_EXPERIMENT_ID,
        "stage": CLA_STAGE,
        "claim_scope": CLA_CLAIM_SCOPE,
        "selection_used_target_eval_artifacts": False,
        "input_artifacts": [
            _provenance_row(
                artifact_id=CLA_INPUT_ARTIFACT_IDS[0],
                root=manifest_root,
                member="manifest.csv",
                member_path=manifest_path,
                digest=manifest_hash,
                stage="dataset_contract",
                claim_scope="dataset_contract_and_split_provenance",
                semantic_identities={"contract_identity": "midogpp_annotation_patch_v1"},
            ),
            _provenance_row(
                artifact_id=CLA_INPUT_ARTIFACT_IDS[1],
                root=feature_root,
                member="embeddings/train.pt",
                member_path=feature_path,
                digest=feature_hash,
                stage="derived_features",
                claim_scope="feature_cache_provenance",
                semantic_identities={"feature_cache_hash": feature_hash},
            ),
        ],
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "schema_version": WORKSPACE_BINDING_SCHEMA_VERSION,
        "experiment_id": CLA_EXPERIMENT_ID,
        "registry_status": "diagnostic",
        "stage": CLA_STAGE,
        "claim_scope": CLA_CLAIM_SCOPE,
        "output_artifact_id": CLA_OUTPUT_ARTIFACT_ID,
        "input_artifact_ids": list(CLA_INPUT_ARTIFACT_IDS),
        "artifact_root": str(root.resolve()),
        "config_resolved_path": str(config_path.resolve()),
        "config_resolved_sha256": _sha256(config_path),
        "input_provenance_path": str(provenance_path.resolve()),
        "input_provenance_sha256": _sha256(provenance_path),
        "manifest": {
            "artifact_id": CLA_INPUT_ARTIFACT_IDS[0],
            "resolved_root": str(manifest_root.resolve()),
            "resolved_path": str(manifest_path.resolve()),
            "sha256": manifest_hash,
        },
        "feature_cache": {
            "artifact_id": CLA_INPUT_ARTIFACT_IDS[1],
            "resolved_root": str(feature_root.resolve()),
            "resolved_path": str(feature_path.resolve()),
            "sha256": feature_hash,
        },
    }
    payload["workspace_binding_hash"] = stable_hash(payload)

    binding_module.validate_persisted_workspace_binding(root, payload, payload)
    config_path.write_text("tampered: true\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="config/provenance SHA256"):
        binding_module.validate_persisted_workspace_binding(root, payload, payload)
    config_path.write_text("resolved: true\n", encoding="utf-8")
    provenance_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="config/provenance SHA256"):
        binding_module.validate_persisted_workspace_binding(root, payload, payload)


def _provenance_row(
    *,
    artifact_id: str,
    root: Path,
    member: str,
    member_path: Path,
    digest: str,
    stage: str,
    claim_scope: str,
    semantic_identities: dict[str, str],
) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "resolved_path": str(root.resolve()),
        "stage": stage,
        "evidence_label": "AUDIT_ONLY",
        "claim_scope": claim_scope,
        "semantic_identities": semantic_identities,
        "semantic_identities_are_file_hashes": False,
        "file_integrity": {
            "status": "EXPECTED_FILE_HASHES_MATCH",
            "default_recording_algorithm": "sha256",
            "files": [
                {
                    "path": member,
                    "resolved_path": str(member_path.resolve()),
                    "exists": True,
                    "expected": {"algorithm": "sha256", "digest": digest},
                    "size_bytes": member_path.stat().st_size,
                    "computed": {"sha256": digest},
                    "verification": "MATCH",
                }
            ],
        },
        "exists": True,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
