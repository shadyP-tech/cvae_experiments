"""Independent inventory validation and lock construction for source data."""

from __future__ import annotations

import hashlib
from itertools import product
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...generation.contracts import COMMON_OUTPUT_DIM, GenerationLock
from ...protocol import ProtocolError
from .input_contracts import row_identity_hash
from .source_cache_contracts import (
    COMPONENT_ARRAY_MEMBER,
    COMPONENT_INDEX_MEMBER,
    EXPECTED_COMPONENT_RECORD_COUNT,
    EXPECTED_SOURCE_STREAM_COUNT,
    EXPECTED_SOURCE_TASK_COUNT,
    GENERATION_DEVICES,
    SOURCE_ARRAY_MEMBER,
    SOURCE_CACHE_LOCK_MEMBER,
    SOURCE_INDEX_MEMBER,
    SOURCE_ROWS_PER_CLASS,
    SourceCache,
)
from .source_cache_store import read_json, sha256_file


class SourceCacheConfig(Protocol):
    contract_hash: str


def build_source_cache_lock(
    root: Path,
    *,
    config: SourceCacheConfig,
    generation_lock: GenerationLock,
    frame: object,
    partitions: object,
    source_cache: SourceCache,
) -> dict[str, object]:
    support_case_count = _support_case_count(partitions)
    expected_support_case_count = _support_case_count_from_config(config)
    if support_case_count != expected_support_case_count:
        raise ProtocolError("Stage-90 source-cache support geometry drifted.")
    validate_source_cache_inventory(
        source_cache, expected_support_case_count=support_case_count
    )
    support_by_center = getattr(partitions, "support_rows_by_center", {})
    support_hashes = {
        center: row_identity_hash(tuple(support_by_center[center])) for center in CENTERS
    }
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_stage90_utility_aligned_source_cache_lock_v1",
        "status": (
            "COMPLETE_LABEL_FREE_FIXED_TWO_CASE_SUPPORT_CACHE"
            if support_case_count == 2
            else "COMPLETE_LABEL_FREE_FIXED_SUPPORT_CACHE"
        ),
        "config_contract_hash": str(config.contract_hash),
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "validation_cache_binding_hash": str(
            getattr(frame, "cache_binding_hash", "")
        ),
        "support_partition_lock_hash": str(getattr(partitions, "lock_hash", "")),
        "support_partition_hash_by_center": support_hashes,
        "support_scratch_hash": source_cache.support_scratch_hash,
        "source_array_sha256": sha256_file(root / SOURCE_ARRAY_MEMBER),
        "component_array_sha256": sha256_file(root / COMPONENT_ARRAY_MEMBER),
        "source_index_sha256": sha256_file(root / SOURCE_INDEX_MEMBER),
        "component_index_sha256": sha256_file(root / COMPONENT_INDEX_MEMBER),
        "source_cache_hash": source_cache.source_cache_hash,
        "source_task_count": EXPECTED_SOURCE_TASK_COUNT,
        "source_stream_count": EXPECTED_SOURCE_STREAM_COUNT,
        "component_record_count": EXPECTED_COMPONENT_RECORD_COUNT,
        "rows_per_class": SOURCE_ROWS_PER_CLASS,
        "generation_devices": list(GENERATION_DEVICES),
        "one_spawned_persistent_worker_per_device": True,
        "parent_cuda_free": True,
        "tf32_disabled": True,
        "amp_disabled": True,
        "float32_memmaps": True,
        "fixed_support_case_count_per_center": support_case_count,
        "fresh_policy_eight_case_floor_satisfied": (
            support_case_count >= 8
            and bool(getattr(config, "fresh_evidence", False))
        ),
        "labels_consumed": False,
        "evaluation_embeddings_consumed": False,
        "source_experts_updated": False,
        "prior_stage90_cache_consumed": False,
    }
    return {**unhashed, "source_cache_lock_hash": stable_hash(unhashed)}


def validate_source_cache_lock(
    root: Path,
    *,
    config: SourceCacheConfig,
    generation_lock: GenerationLock,
    frame: object,
    partitions: object,
    source_cache: SourceCache,
) -> dict[str, object]:
    observed = dict(read_json(root / SOURCE_CACHE_LOCK_MEMBER))
    expected = build_source_cache_lock(
        root,
        config=config,
        generation_lock=generation_lock,
        frame=frame,
        partitions=partitions,
        source_cache=source_cache,
    )
    if observed != expected:
        raise ProtocolError("Stage-90 utility-aligned source-cache lock drifted.")
    return observed


def validate_source_cache_inventory(
    cache: SourceCache, *, expected_support_case_count: int | None = None
) -> None:
    try:
        sources = np.load(cache.source_array_path, mmap_mode="r", allow_pickle=False)
        components = np.load(
            cache.component_array_path, mmap_mode="r", allow_pickle=False
        )
    except (OSError, ValueError) as exc:
        raise ProtocolError("Stage-90 source cache is unreadable.") from exc
    if sources.shape != (
        EXPECTED_SOURCE_STREAM_COUNT,
        2 * SOURCE_ROWS_PER_CLASS,
        COMMON_OUTPUT_DIM,
    ) or sources.dtype != np.float32:
        raise ProtocolError("Stage-90 source-cache array geometry drifted.")
    if (
        components.ndim != 3
        or components.shape[:2] != (EXPECTED_SOURCE_TASK_COUNT, 4)
        or components.dtype != np.float32
        or components.shape[2] <= 0
        or not np.isfinite(components).all()
        or np.any(components < 0.0)
    ):
        raise ProtocolError("Stage-90 component-cache array geometry drifted.")

    expected_source_keys = tuple(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS))
    if (
        len(cache.source_records) != EXPECTED_SOURCE_STREAM_COUNT
        or tuple(record.key for record in cache.source_records) != expected_source_keys
    ):
        raise ProtocolError("Stage-90 source record coverage drifted.")
    labels = np.concatenate(
        (
            np.zeros(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
            np.ones(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
        )
    )
    for ordinal, record in enumerate(cache.source_records):
        if (
            record.block_ordinal != ordinal
            or record.rows_per_class != SOURCE_ROWS_PER_CLASS
            or record.row_count != 2 * SOURCE_ROWS_PER_CLASS
            or record.feature_dim != COMMON_OUTPUT_DIM
            or _array_bundle_sha256(sources[ordinal], labels) != record.output_sha256
        ):
            raise ProtocolError("Stage-90 source record binding drifted.")

    expected_component_keys = tuple(
        (query, source, training_seed)
        for source in CENTERS
        for training_seed in TRAINING_SEEDS
        for query in CENTERS
        if query != source
    )
    if (
        len(cache.component_records) != EXPECTED_COMPONENT_RECORD_COUNT
        or tuple(record.key for record in cache.component_records)
        != expected_component_keys
    ):
        raise ProtocolError("Stage-90 component record coverage drifted.")
    observed_support_case_counts = {
        record.support_case_count for record in cache.component_records
    }
    if (
        len(observed_support_case_counts) != 1
        or next(iter(observed_support_case_counts)) < 2
    ):
        raise ProtocolError("Stage-90 component support-case counts drifted.")
    observed_support_case_count = next(iter(observed_support_case_counts))
    if (
        expected_support_case_count is not None
        and observed_support_case_count != expected_support_case_count
    ):
        raise ProtocolError("Stage-90 component support-case count drifted.")
    offsets_by_query: dict[str, tuple[int, int, str]] = {}
    for ordinal, record in enumerate(cache.component_records):
        numeric = np.asarray(
            [record.case_equal_energy, *record.linear_kernel_mmd2_by_generation_seed.values()],
            dtype=np.float64,
        )
        offset = (
            record.support_start,
            record.support_stop,
            record.support_partition_hash,
        )
        previous = offsets_by_query.setdefault(record.query_center, offset)
        if (
            record.component_ordinal != ordinal
            or record.query_center not in CENTERS
            or record.source_center not in CENTERS
            or record.training_seed not in TRAINING_SEEDS
            or record.support_case_count != observed_support_case_count
            or record.support_row_count != record.support_stop - record.support_start
            or not 0 <= record.support_start < record.support_stop <= components.shape[2]
            or previous != offset
            or not np.isfinite(numeric).all()
            or np.any(numeric < 0.0)
        ):
            raise ProtocolError("Stage-90 component record binding drifted.")
    ordered_offsets = [offsets_by_query[center][:2] for center in CENTERS]
    cursor = 0
    for start, stop in ordered_offsets:
        if start != cursor:
            raise ProtocolError("Stage-90 support component offsets are not contiguous.")
        cursor = stop
    if cursor != components.shape[2]:
        raise ProtocolError("Stage-90 support component coverage drifted.")


def _support_case_count(partitions: object) -> int:
    by_center = getattr(partitions, "support_rows_by_center", None)
    if not isinstance(by_center, dict) and not hasattr(by_center, "items"):
        raise ProtocolError("Stage-90 source-cache support surface is absent.")
    counts = {
        len({str(row.case_id) for row in tuple(rows)})
        for rows in by_center.values()
    }
    if len(counts) != 1:
        raise ProtocolError("Stage-90 source-cache support counts differ by center.")
    count = next(iter(counts))
    if count < 2:
        raise ProtocolError("Stage-90 source-cache support count is too small.")
    return count


def _support_case_count_from_config(config: object) -> int:
    direct = getattr(config, "fixed_support_case_count_per_center", None)
    if direct is not None:
        return int(direct)
    protocol = getattr(config, "protocol", {})
    if isinstance(protocol, dict) or hasattr(protocol, "get"):
        return int(protocol.get("fixed_support_case_count_per_center", 2))
    return 2


def _array_bundle_sha256(embeddings: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in (embeddings, labels):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = (
    "build_source_cache_lock",
    "validate_source_cache_inventory",
    "validate_source_cache_lock",
)
