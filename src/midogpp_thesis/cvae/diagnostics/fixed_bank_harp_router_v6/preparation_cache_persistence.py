"""Persistence and independent validation of the HARP v6 label-free cache."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import atomic_json, atomic_npy, read_json, sha256_file
from .input_surfaces import (
    CACHE_INDEX,
    CACHE_ROWS,
    CONTENT_INDEX,
    DEVELOPMENT_ROLE,
    EVALUATION_ROLE,
    HarpCacheRow,
    V6_CACHE_IDENTITY,
    _load_cache_index_from_root,
)
from .preparation_contracts import (
    EXPECTED_ROW_COUNT,
    CanonicalLabelBlindFrame,
    HarpPreparationIdentity,
    V6_PREPARATION_IDENTITY,
)
from .preparation_durable_io import (
    write_cache_rows,
    write_content_index,
    write_final_content_index,
)


def persist_label_blind_cache(
    root: Path,
    *,
    frame: CanonicalLabelBlindFrame,
    partition: Mapping[tuple[str, str], str],
    partition_payload: Mapping[str, object],
    partition_hash: str,
    identity: HarpPreparationIdentity,
) -> tuple[HarpCacheRow, ...]:
    """Materialize the label-free cache and its durable pre-label barrier."""

    root.mkdir(parents=True, exist_ok=False)
    shard_rows: list[dict[str, object]] = []
    shard_hashes: dict[str, str] = {}
    embedding_offset: dict[tuple[str, str], int] = {}
    for center in CENTERS:
        relative = f"embeddings/by_center/center_{center}.npy"
        path = root / relative
        atomic_npy(path, frame.embeddings_by_center[center])
        digest = sha256_file(path)
        shard_hashes[relative] = digest
        shard_rows.append(
            {
                "relative_path": relative,
                "file_sha256": digest,
                "shape": list(frame.embeddings_by_center[center].shape),
                "dtype": "float32",
            }
        )
        for row in frame.rows_by_center[center]:
            embedding_offset[(center, row.sample_id)] = row.center_row_index
    prepared: list[HarpCacheRow] = []
    for role in (DEVELOPMENT_ROLE, EVALUATION_ROLE):
        for center in CENTERS:
            scoped = tuple(
                row
                for row in frame.rows_by_center[center]
                if partition[(center, row.case_id)] == role
            )
            for split_ordinal, row in enumerate(scoped):
                prepared.append(
                    HarpCacheRow(
                        center=center,
                        case_id=row.case_id,
                        sample_id=row.sample_id,
                        split_role=role,
                        split_row_index=split_ordinal,
                        embedding_file=f"embeddings/by_center/center_{center}.npy",
                        embedding_row_index=embedding_offset[(center, row.sample_id)],
                    )
                )
    if canonical_hash(partition_payload) != partition_hash:
        raise ProtocolError("HARP preparation partition payload drifted before write.")
    atomic_json(root / identity.case_partition, partition_payload)
    write_cache_rows(
        root / CACHE_ROWS,
        prepared,
        row_schema=identity.cache_identity.row_schema,
    )
    index_base: dict[str, object] = {
        "schema_version": identity.cache_identity.cache_schema,
        "artifact_id": identity.cache_identity.artifact_id,
        "dataset_family": "MIDOG++",
        "representation_id": "midogpp_virchow2_common_3840_float32_v1",
        "feature_dim": COMMON_OUTPUT_DIM,
        "dtype": "float32",
        "labels_stored": False,
        "split_roles": [DEVELOPMENT_ROLE, EVALUATION_ROLE],
        "row_index_member": CACHE_ROWS.as_posix(),
        "shards": shard_rows,
    }
    atomic_json(
        root / CACHE_INDEX,
        {**index_base, "cache_index_hash": canonical_hash(index_base)},
    )
    member_hashes = {
        CACHE_INDEX.as_posix(): sha256_file(root / CACHE_INDEX),
        CACHE_ROWS.as_posix(): sha256_file(root / CACHE_ROWS),
        identity.case_partition.as_posix(): sha256_file(root / identity.case_partition),
        **shard_hashes,
    }
    barrier_base: dict[str, object] = {
        "schema_version": identity.label_free_barrier_schema,
        "status": "DURABLE_LABEL_FREE_CACHE_BEFORE_SCORING_MANIFEST_OPEN",
        "canonical_cache_content_hash": frame.cache_content_hash,
        "canonical_cache_row_order_hash": frame.row_order_hash,
        "partition_hash": partition_hash,
        "cache_index_sha256": sha256_file(root / CACHE_INDEX),
        "row_index_sha256": sha256_file(root / CACHE_ROWS),
        "case_partition_sha256": sha256_file(root / identity.case_partition),
        "embedding_shard_sha256": dict(sorted(shard_hashes.items())),
        "row_count": len(prepared),
        "case_count": len(partition),
        "whole_case_disjoint": True,
        "labels_stored": False,
        "canonical_scoring_manifest_opened": False,
        "mixed_patch_labels_within_case_supported": True,
    }
    atomic_json(
        root / identity.label_free_barrier,
        {**barrier_base, "barrier_hash": canonical_hash(barrier_base)},
    )
    label_free_members = {
        **member_hashes,
        identity.label_free_barrier.as_posix(): sha256_file(
            root / identity.label_free_barrier
        ),
    }
    write_content_index(
        root / identity.label_free_content_index,
        label_free_members,
        content_schema=identity.cache_identity.content_schema,
    )
    write_content_index(
        root / CONTENT_INDEX,
        {
            **label_free_members,
            identity.label_free_content_index.as_posix(): sha256_file(
                root / identity.label_free_content_index
            ),
        },
        content_schema=identity.cache_identity.content_schema,
    )
    return tuple(prepared)


def independently_validate_label_blind_barrier(
    root: Path,
    *,
    expected_partition_hash: str,
    identity: HarpPreparationIdentity = V6_PREPARATION_IDENTITY,
):
    """Reconstruct and validate the label-free barrier from durable bytes."""

    content = read_json(root / CONTENT_INDEX)
    expected_content_hash = str(content.get("content_index_hash"))
    if identity.cache_identity != V6_CACHE_IDENTITY:
        raise ProtocolError("HARP v6 preparation cache identity drifted.")
    cache = _load_cache_index_from_root(
        root,
        expected_content_sha256=expected_content_hash,
    )
    barrier = read_json(root / identity.label_free_barrier)
    partition = read_json(root / identity.case_partition)
    label_free_content = read_json(root / identity.label_free_content_index)
    barrier_base = {key: value for key, value in barrier.items() if key != "barrier_hash"}
    label_free_base = {
        key: value
        for key, value in label_free_content.items()
        if key != "content_index_hash"
    }
    indexed_members = set(content["members"])
    expected_members = {
        CACHE_INDEX.as_posix(),
        CACHE_ROWS.as_posix(),
        identity.case_partition.as_posix(),
        identity.label_free_barrier.as_posix(),
        identity.label_free_content_index.as_posix(),
        *(f"embeddings/by_center/center_{center}.npy" for center in CENTERS),
    }
    label_free_expected = expected_members - {
        identity.label_free_content_index.as_posix()
    }
    assignments = partition.get("assignments")
    partition_roles = (
        {
            (str(row.get("center")), str(row.get("case_id"))): str(
                row.get("split_role")
            )
            for row in assignments
            if isinstance(row, Mapping)
        }
        if isinstance(assignments, list)
        else {}
    )
    cache_roles = {(row.center, row.case_id): row.split_role for row in cache.rows}
    if (
        barrier.get("barrier_hash") != canonical_hash(barrier_base)
        or barrier.get("partition_hash") != expected_partition_hash
        or canonical_hash(partition) != expected_partition_hash
        or barrier.get("canonical_scoring_manifest_opened") is not False
        or indexed_members != expected_members
        or label_free_content.get("content_index_hash")
        != canonical_hash(label_free_base)
        or set(label_free_content.get("members", {})) != label_free_expected
        or partition_roles != cache_roles
        or len(cache.rows) != EXPECTED_ROW_COUNT
    ):
        raise ProtocolError("HARP label-free preparation barrier failed validation.")
    return cache


def write_final_prepared_content_index(
    root: Path,
    *,
    identity: HarpPreparationIdentity = V6_PREPARATION_IDENTITY,
) -> None:
    write_final_content_index(
        root,
        content_schema=identity.cache_identity.content_schema,
    )


def validate_final_prepared_cache(
    root: Path,
    *,
    identity: HarpPreparationIdentity = V6_PREPARATION_IDENTITY,
):
    """Validate the final closed-world cache, including its receipt."""

    content = read_json(root / CONTENT_INDEX)
    if identity.cache_identity != V6_CACHE_IDENTITY:
        raise ProtocolError("HARP v6 preparation cache identity drifted.")
    cache = _load_cache_index_from_root(
        root,
        expected_content_sha256=str(content.get("content_index_hash")),
    )
    if identity.preparation_receipt.as_posix() not in cache.member_sha256:
        raise ProtocolError("HARP final prepared cache lacks its preparation receipt.")
    return cache


__all__ = (
    "persist_label_blind_cache",
    "independently_validate_label_blind_barrier",
    "write_final_prepared_content_index",
    "validate_final_prepared_cache",
)
