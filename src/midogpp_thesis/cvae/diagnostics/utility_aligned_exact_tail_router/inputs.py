"""Closed-world inputs for the consumed utility-aligned Stage-90 diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ....data.features.uniform_b_routing_validation import (
    load_unlabeled_validation_shard,
    validate_uniform_b_routing_validation_cache,
)
from ....data.features.uniform_b_routing_validation.config import (
    CACHE_NAME as VALIDATION_CACHE_SEMANTIC_ID,
    MANIFEST_SHA256 as EXPECTED_MANIFEST_SHA256,
    REPRESENTATION_ID as VALIDATION_CACHE_REPRESENTATION_ID,
)
from ....workspace.runtime import MidogppWorkspace
from ...expert_bank.uniform_b_v2_promotion import (
    load_promotion_config,
    validate_promoted_bank,
)
from ...generation import (
    load_generation_lock_config,
    read_generation_lock,
    validate_generation_bundle,
)
from ...generation.contracts import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    GenerationLock,
)
from ...protocol import ProtocolError
from ...routing import (
    load_equal_union_policy_config,
    read_policy_lock,
    validate_equal_union_policy_bundle,
)
from ...routing.contracts import (
    EXPECTED_POLICY_LOCK_HASH as EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
    EqualUnionPolicyLock,
)
from ...routing.metadata_compatibility import (
    derive_compatibility_scores,
    derive_metadata_profiles,
)
from ...routing.metadata_compatibility.contracts import (
    DOMAIN_MAPPING_MEMBER,
    DOMAIN_MAPPING_SHA256,
)
from .contracts import CENTERS
from .input_contracts import LabelFreeValidationFrame, ValidationRowIdentity


class DiagnosticInputConfig(Protocol):
    experiment_id: str
    output_artifact_id: str
    input_artifact_ids: Sequence[str]
    expert_bank_root: Path
    generation_lock_root: Path
    equal_union_policy_root: Path
    validation_cache_root: Path
    validation_manifest_path: Path
    metadata_profile_root: Path


@dataclass(frozen=True)
class ValidatedLocks:
    generation: GenerationLock
    equal_union: EqualUnionPolicyLock


def load_label_free_validation_frame(
    config: DiagnosticInputConfig,
) -> LabelFreeValidationFrame:
    """Admit the label-free cache without opening the scoring manifest."""

    checks = validate_uniform_b_routing_validation_cache(config.validation_cache_root)
    if checks.get("status") != "PASS" or checks.get("label_fields_absent") is not True:
        raise ProtocolError("Utility-aligned validation cache failed label-free checks.")
    arrays: list[np.ndarray] = []
    rows: list[ValidationRowIdentity] = []
    by_center: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    shard_hashes: dict[str, str] = {}
    ordinal = 0
    for center in CENTERS:
        shard = load_unlabeled_validation_shard(
            config.validation_cache_root / f"embeddings/by_center/center_{center}.pt",
            expected_center=center,
        )
        center_rows: list[ValidationRowIdentity] = []
        for metadata in shard.metadata:
            row = ValidationRowIdentity(
                row_ordinal=ordinal,
                manifest_row_index=int(metadata["manifest_row_index"]),
                sample_id=str(metadata["sample_id"]),
                case_id=str(metadata["case_id"]),
                center=center,
            )
            rows.append(row)
            center_rows.append(row)
            ordinal += 1
        arrays.append(np.asarray(shard.embeddings, dtype=np.float32))
        by_center[center] = tuple(center_rows)
        shard_hashes[center] = shard.cache_sha256
    protocol = _json(config.validation_cache_root / "manifests/frozen_build_protocol.json")
    content = _json(config.validation_cache_root / "manifests/content_index.json")
    input_hashes = protocol.get("input_hashes")
    if (
        protocol.get("cache_name") != VALIDATION_CACHE_SEMANTIC_ID
        or protocol.get("representation_id") != VALIDATION_CACHE_REPRESENTATION_ID
        or protocol.get("validation_split") != "val"
        or not isinstance(input_hashes, Mapping)
        or input_hashes.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
    ):
        raise ProtocolError("Utility-aligned validation-cache identity drifted.")
    binding = {
        "schema_version": "midogpp_utility_aligned_stage90_cache_binding_v1",
        "cache_artifact_id": config.input_artifact_ids[3],
        "cache_name": protocol.get("cache_name"),
        "representation_id": protocol.get("representation_id"),
        "validation_split": protocol.get("validation_split"),
        "manifest_sha256": input_hashes.get("manifest_sha256"),
        "feature_dim": 3840,
        "row_count": len(rows),
        "center_count": len(CENTERS),
        "cache_protocol_hash": protocol.get("frozen_build_protocol_hash"),
        "cache_content_hash": content.get("content_hash"),
        "shard_sha256_by_center": shard_hashes,
        "labels_persisted": False,
        "manifest_opened": False,
    }
    return LabelFreeValidationFrame(
        embeddings=np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32),
        rows=tuple(rows),
        rows_by_center=by_center,
        cache_binding=binding,
    )


def load_validated_locks(config: DiagnosticInputConfig) -> ValidatedLocks:
    generation_config = load_generation_lock_config(
        config.generation_lock_root / "config.resolved.yaml"
    )
    validate_generation_bundle(config.generation_lock_root, config=generation_config)
    generation = read_generation_lock(
        config.generation_lock_root / "manifests/generation_lock.json"
    )
    policy_config = load_equal_union_policy_config(
        config.equal_union_policy_root / "config.resolved.yaml"
    )
    validate_equal_union_policy_bundle(config.equal_union_policy_root, config=policy_config)
    policy = read_policy_lock(config.equal_union_policy_root / "manifests/policy_lock.json")
    if (
        generation.bank_lock_hash != EXPECTED_BANK_LOCK_HASH
        or generation.generation_lock_hash != EXPECTED_GENERATION_LOCK_HASH
        or policy.policy_lock_hash != EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH
        or policy.generation_lock_hash != generation.generation_lock_hash
    ):
        raise ProtocolError("Utility-aligned frozen lineage locks disagree.")
    return ValidatedLocks(generation=generation, equal_union=policy)


def load_metadata_similarity(
    config: DiagnosticInputConfig,
) -> Mapping[str, Mapping[str, float]]:
    profiles = derive_metadata_profiles(
        config.metadata_profile_root / DOMAIN_MAPPING_MEMBER,
        expected_sha256=DOMAIN_MAPPING_SHA256,
    )
    result: dict[str, dict[str, float]] = {center: {} for center in CENTERS}
    for score in derive_compatibility_scores(profiles):
        result[score.target_center][score.source_center] = (
            float(score.exact_match_count) / 3.0
        )
    if any(set(result[center]) != set(CENTERS).difference({center}) for center in CENTERS):
        raise ProtocolError("Utility-aligned metadata surface coverage drifted.")
    return MappingProxyType(
        {center: MappingProxyType(dict(result[center])) for center in CENTERS}
    )


def validate_pre_gpu_firewall(
    config: DiagnosticInputConfig,
    frame: LabelFreeValidationFrame,
) -> Mapping[str, object]:
    promotion_config = load_promotion_config(
        config.expert_bank_root / "config.resolved.yaml"
    )
    checks = validate_promoted_bank(
        config.expert_bank_root, config=promotion_config, allow_pending=False
    )
    bank_index = _json(config.expert_bank_root / "manifests/expert_bank_index.json")
    leakage = _json(config.expert_bank_root / "reports/leakage_report.json")
    source_evidence = _json(
        config.expert_bank_root / "manifests/source_evidence_lock.json"
    )
    records = bank_index.get("records")
    if (
        checks.get("status") != "PASS"
        or checks.get("all_experts_source_only") is not True
        or not isinstance(records, list)
        or len(records) != 27
        or any(
            not isinstance(row, Mapping)
            or row.get("fresh_source_only_training") is not True
            or row.get("parent_checkpoint_used") is not False
            for row in records
        )
        or leakage.get("status") != "PASS"
        or leakage.get("fresh_source_only_training") is not True
        or int(leakage.get("identity_overlap_failures", -1)) != 0
        or int(source_evidence.get("identity_overlap_failures", -1)) != 0
        or frame.cache_binding.get("validation_split") != "val"
        or frame.cache_binding.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or frame.cache_binding.get("labels_persisted") is not False
        or frame.cache_binding.get("manifest_opened") is not False
        or _sha256_file(config.validation_manifest_path) != EXPECTED_MANIFEST_SHA256
    ):
        raise ProtocolError("Utility-aligned pre-GPU firewall failed.")
    return {
        "status": "PASS",
        "bank_validation_status": "PASS",
        "bank_lock_hash": str(bank_index.get("bank_lock_hash")),
        "expert_count": len(records),
        "fresh_source_only_training": True,
        "parent_checkpoint_used": False,
        "bank_identity_overlap_failures": 0,
        "validation_split": "val",
        "validation_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "validation_cache_label_fields_absent": True,
        "source_vs_validation_overlap_basis": (
            "inherited_source_only_vs_hash_bound_validation_split_contract"
        ),
        "gpu_work_authorized": True,
    }


def validate_workspace_provenance(
    root: Path,
    config: DiagnosticInputConfig,
) -> dict[str, Mapping[str, object]]:
    payload = _json(root / "provenance/input_artifacts.json")
    if (
        payload.get("schema_version") != "midogpp_input_artifacts_v2"
        or payload.get("dataset_id") != "midogpp"
        or payload.get("experiment_id") != config.experiment_id
        or payload.get("stage") != "90_oracles_and_diagnostics"
        or payload.get("claim_scope") != "diagnostic_only"
    ):
        raise ProtocolError("Utility-aligned workspace provenance header drifted.")
    raw_rows = payload.get("input_artifacts")
    if not isinstance(raw_rows, list) or not all(isinstance(row, Mapping) for row in raw_rows):
        raise ProtocolError("Utility-aligned workspace provenance rows are malformed.")
    by_id = {str(row.get("artifact_id")): row for row in raw_rows}
    if len(by_id) != len(raw_rows) or tuple(by_id) != tuple(config.input_artifact_ids):
        raise ProtocolError("Utility-aligned workspace provenance order drifted.")
    expected_paths = (
        config.expert_bank_root,
        config.generation_lock_root,
        config.equal_union_policy_root,
        config.validation_cache_root,
        config.validation_manifest_path.parent,
        config.metadata_profile_root,
    )
    for artifact_id, expected_path in zip(config.input_artifact_ids, expected_paths, strict=True):
        row = by_id[artifact_id]
        if (
            Path(str(row.get("resolved_path", ""))).resolve() != expected_path.resolve()
            or row.get("exists") is not True
            or not isinstance(row.get("semantic_identities"), Mapping)
            or not isinstance(row.get("file_integrity"), Mapping)
        ):
            raise ProtocolError(f"Utility-aligned provenance drifted: {artifact_id}.")
    return {artifact_id: by_id[artifact_id] for artifact_id in config.input_artifact_ids}


def validate_active_diagnostic_workspace_binding(
    config: DiagnosticInputConfig,
) -> Mapping[str, object]:
    try:
        workspace = MidogppWorkspace.load()
        workspace.validate()
        experiment = workspace.get_experiment(config.experiment_id)
        output = workspace.artifacts[config.output_artifact_id]
    except (KeyError, ValueError, OSError) as exc:
        raise ProtocolError("Utility-aligned canonical workspace binding failed.") from exc
    if (
        experiment.status != "diagnostic"
        or experiment.stage != "90_oracles_and_diagnostics"
        or experiment.claim_scope != "diagnostic_only"
        or experiment.output_artifact_id != config.output_artifact_id
        or experiment.input_artifact_ids != tuple(config.input_artifact_ids)
        or output.stage != "90_oracles_and_diagnostics"
        or output.claim_scope != "diagnostic_only"
    ):
        raise ProtocolError("Utility-aligned experiment is not the active diagnostic binding.")
    return {
        "status": "PASS",
        "registry_status": experiment.status,
        "experiment_id": experiment.experiment_id,
        "output_artifact_id": experiment.output_artifact_id,
        "stage": experiment.stage,
        "claim_scope": experiment.claim_scope,
    }


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read utility-aligned JSON input: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"Utility-aligned JSON input must be an object: {path}.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash utility-aligned input: {path}.") from exc
    return digest.hexdigest()


__all__ = (
    "DiagnosticInputConfig",
    "ValidatedLocks",
    "load_label_free_validation_frame",
    "load_metadata_similarity",
    "load_validated_locks",
    "validate_active_diagnostic_workspace_binding",
    "validate_pre_gpu_firewall",
    "validate_workspace_provenance",
)
