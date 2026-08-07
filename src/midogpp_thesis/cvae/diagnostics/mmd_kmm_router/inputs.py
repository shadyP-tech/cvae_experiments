"""Validated label-free inputs and whole-case partitions for MMD/KMM."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ....data.features.uniform_b_routing_validation import (
    load_unlabeled_validation_shard,
    validate_uniform_b_routing_validation_cache,
)
from ...generation import (
    load_generation_lock_config,
    read_generation_lock,
    validate_generation_bundle,
)
from ...generation.contracts import GenerationLock
from ...protocol import ProtocolError
from ...routing import (
    load_equal_union_policy_config,
    read_policy_lock,
    validate_equal_union_policy_bundle,
)
from ...routing.contracts import EqualUnionPolicyLock
from ...routing.dense_residual_soft_router import deterministic_case_partitions
from .config import MMDKMMRouterDiagnosticConfig
from .contracts import (
    CENTERS,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    SUPPORT_CASE_COUNT,
    SUPPORT_PARTITION_NAMESPACE,
    SUPPORT_SPLIT_SEED,
    VALIDATION_CACHE_REPRESENTATION_ID,
    VALIDATION_CACHE_SEMANTIC_ID,
    ValidationRowIdentity,
    row_identity_hash,
)


SUPPORT_PARTITION_COLUMNS = (
    "schema_version",
    "row_ordinal",
    "manifest_row_index",
    "sample_id",
    "case_id",
    "center",
    "split",
    "partition_role",
    "center_partition_hash",
    "support_split_seed",
    "label_present",
)


@dataclass(frozen=True)
class LabelFreeValidationFrame:
    embeddings: np.ndarray
    rows: tuple[ValidationRowIdentity, ...]
    rows_by_center: Mapping[str, tuple[ValidationRowIdentity, ...]]
    cache_binding: Mapping[str, object]

    def __post_init__(self) -> None:
        values = np.asarray(self.embeddings)
        if (
            values.shape != (len(self.rows), 3840)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or set(self.rows_by_center) != set(CENTERS)
            or len({row.sample_id for row in self.rows}) != len(self.rows)
        ):
            raise ProtocolError("MMD/KMM label-free validation frame is malformed.")

    @property
    def cache_binding_hash(self) -> str:
        return stable_hash(dict(self.cache_binding))

    def embeddings_for(self, rows: Sequence[ValidationRowIdentity]) -> np.ndarray:
        ordinals = np.asarray([row.row_ordinal for row in rows], dtype=np.int64)
        if not len(ordinals) or np.any(ordinals < 0) or np.any(ordinals >= len(self.rows)):
            raise ProtocolError("MMD/KMM validation row slice is invalid.")
        if tuple(self.rows[int(index)].sample_id for index in ordinals) != tuple(
            row.sample_id for row in rows
        ):
            raise ProtocolError("MMD/KMM validation row identity drifted.")
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


@dataclass(frozen=True)
class PartitionSurface:
    support_rows_by_center: Mapping[str, tuple[ValidationRowIdentity, ...]]
    evaluation_rows_by_center: Mapping[str, tuple[ValidationRowIdentity, ...]]
    table_rows: tuple[Mapping[str, object], ...]
    lock_payload: Mapping[str, object]

    @property
    def lock_hash(self) -> str:
        return str(self.lock_payload["support_partition_lock_hash"])


@dataclass(frozen=True)
class ValidatedLocks:
    generation: GenerationLock
    equal_union: EqualUnionPolicyLock


def load_label_free_validation_frame(
    config: MMDKMMRouterDiagnosticConfig,
) -> LabelFreeValidationFrame:
    checks = validate_uniform_b_routing_validation_cache(config.validation_cache_root)
    if checks.get("status") != "PASS" or checks.get("label_fields_absent") is not True:
        raise ProtocolError("MMD/KMM validation cache did not pass label-free checks.")
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
            center_rows.append(row)
            rows.append(row)
            ordinal += 1
        arrays.append(np.asarray(shard.embeddings, dtype=np.float32))
        by_center[center] = tuple(center_rows)
        shard_hashes[center] = shard.cache_sha256
    protocol = _json(config.validation_cache_root / "manifests/frozen_build_protocol.json")
    content = _json(config.validation_cache_root / "manifests/content_index.json")
    if (
        protocol.get("cache_name") != VALIDATION_CACHE_SEMANTIC_ID
        or protocol.get("representation_id") != VALIDATION_CACHE_REPRESENTATION_ID
        or protocol.get("validation_split") != "val"
    ):
        raise ProtocolError("MMD/KMM validation cache semantic identity drifted.")
    binding = {
        "schema_version": "midogpp_mmd_kmm_validation_cache_binding_v1",
        "cache_artifact_id": config.input_artifact_ids[-2],
        "cache_name": protocol.get("cache_name"),
        "representation_id": protocol.get("representation_id"),
        "validation_split": protocol.get("validation_split"),
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


def build_partition_surface(
    frame: LabelFreeValidationFrame,
    *,
    config_contract_hash: str,
) -> PartitionSurface:
    support_by_center: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    evaluation_by_center: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    table: list[dict[str, object]] = []
    center_payloads: dict[str, object] = {}
    for center in CENTERS:
        original = frame.rows_by_center[center]
        partition = deterministic_case_partitions(
            [row.sample_id for row in original],
            [row.case_id for row in original],
            target_center=center,
            support_case_count=SUPPORT_CASE_COUNT,
            namespace=SUPPORT_PARTITION_NAMESPACE,
            split_seed=SUPPORT_SPLIT_SEED,
        )
        support_indices = set(partition.support_indices)
        support: list[ValidationRowIdentity] = []
        evaluation: list[ValidationRowIdentity] = []
        for local_index, original_row in enumerate(original):
            role = "support" if local_index in support_indices else "evaluation"
            row = ValidationRowIdentity(
                row_ordinal=original_row.row_ordinal,
                manifest_row_index=original_row.manifest_row_index,
                sample_id=original_row.sample_id,
                case_id=original_row.case_id,
                center=original_row.center,
                partition_role=role,
            )
            (support if role == "support" else evaluation).append(row)
            table.append(
                {
                    "schema_version": "midogpp_mmd_kmm_support_partition_row_v1",
                    "row_ordinal": row.row_ordinal,
                    "manifest_row_index": row.manifest_row_index,
                    "sample_id": row.sample_id,
                    "case_id": row.case_id,
                    "center": row.center,
                    "split": row.split,
                    "partition_role": role,
                    "center_partition_hash": partition.partition_hash,
                    "support_split_seed": SUPPORT_SPLIT_SEED,
                    "label_present": False,
                }
            )
        if len({row.case_id for row in support}) != SUPPORT_CASE_COUNT:
            raise ProtocolError("MMD/KMM support partition does not contain two cases.")
        if set(row.case_id for row in support).intersection(row.case_id for row in evaluation):
            raise ProtocolError("MMD/KMM support/evaluation cases overlap.")
        support_by_center[center] = tuple(support)
        evaluation_by_center[center] = tuple(evaluation)
        center_payloads[center] = {
            "partition_hash": partition.partition_hash,
            "support_cases": sorted({row.case_id for row in support}),
            "evaluation_cases": sorted({row.case_id for row in evaluation}),
            "support_row_identity_hash": row_identity_hash(support),
            "evaluation_row_identity_hash": row_identity_hash(evaluation),
            "support_row_count": len(support),
            "evaluation_row_count": len(evaluation),
        }
    unhashed = {
        "schema_version": "midogpp_mmd_kmm_support_partition_lock_v1",
        "status": "LOCKED_FROM_LABEL_FREE_CACHE_IDENTITIES",
        "config_contract_hash": config_contract_hash,
        "validation_cache_binding_hash": frame.cache_binding_hash,
        "support_case_count_per_center": SUPPORT_CASE_COUNT,
        "support_split_seed": SUPPORT_SPLIT_SEED,
        "support_partition_namespace": SUPPORT_PARTITION_NAMESPACE,
        "centers": center_payloads,
        "manifest_opened": False,
        "labels_used": False,
        "whole_case": True,
        "support_evaluation_case_disjoint": True,
        "support_evaluation_sample_disjoint": True,
    }
    lock = {**unhashed, "support_partition_lock_hash": stable_hash(unhashed)}
    return PartitionSurface(
        support_rows_by_center=support_by_center,
        evaluation_rows_by_center=evaluation_by_center,
        table_rows=tuple(table),
        lock_payload=lock,
    )


def load_validated_locks(config: MMDKMMRouterDiagnosticConfig) -> ValidatedLocks:
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
        raise ProtocolError("MMD/KMM frozen bank/generation/control locks disagree.")
    return ValidatedLocks(generation=generation, equal_union=policy)


def validate_workspace_provenance(
    root: Path,
    config: MMDKMMRouterDiagnosticConfig,
) -> dict[str, Mapping[str, object]]:
    payload = _json(root / "provenance/input_artifacts.json")
    if (
        payload.get("schema_version") != "midogpp_input_artifacts_v2"
        or payload.get("dataset_id") != "midogpp"
        or payload.get("experiment_id") != config.experiment_id
        or payload.get("stage") != "90_oracles_and_diagnostics"
        or payload.get("claim_scope") != "diagnostic_only"
    ):
        raise ProtocolError("MMD/KMM workspace provenance header drifted.")
    raw_rows = payload.get("input_artifacts")
    if not isinstance(raw_rows, list) or not all(isinstance(row, Mapping) for row in raw_rows):
        raise ProtocolError("MMD/KMM workspace provenance rows are malformed.")
    by_id = {str(row.get("artifact_id")): row for row in raw_rows}
    if len(by_id) != len(raw_rows) or set(by_id) != set(config.input_artifact_ids):
        raise ProtocolError("MMD/KMM workspace provenance input set drifted.")
    expected_paths = (
        config.expert_bank_root,
        config.generation_lock_root,
        config.equal_union_policy_root,
        config.validation_cache_root,
        config.validation_manifest_path.parent,
    )
    for artifact_id, expected_path in zip(
        config.input_artifact_ids, expected_paths, strict=True
    ):
        row = by_id[artifact_id]
        if (
            Path(str(row.get("resolved_path", ""))).resolve() != expected_path.resolve()
            or row.get("exists") is not True
            or not isinstance(row.get("semantic_identities"), Mapping)
            or not isinstance(row.get("file_integrity"), Mapping)
        ):
            raise ProtocolError(f"MMD/KMM provenance identity drifted: {artifact_id}.")
    return {
        artifact_id: by_id[artifact_id] for artifact_id in config.input_artifact_ids
    }


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read MMD/KMM JSON input: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"MMD/KMM JSON input must be an object: {path}.")
    return payload


__all__ = (
    "LabelFreeValidationFrame",
    "PartitionSurface",
    "SUPPORT_PARTITION_COLUMNS",
    "ValidatedLocks",
    "build_partition_surface",
    "load_label_free_validation_frame",
    "load_validated_locks",
    "validate_workspace_provenance",
)
