"""Independent validation and durable lock for source-cache products."""

from __future__ import annotations

import hashlib
from itertools import product
import json
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from ....common.hashing import stable_hash
from ...generation.contracts import COMMON_OUTPUT_DIM, GenerationLock
from ...protocol import ProtocolError
from .artifact_io import read_json, sha256_file
from .contracts import CENTERS, GENERATION_SEEDS, TRAINING_SEEDS
from .source_cache_contracts import (
    COMPATIBILITY_CASE_COLUMNS,
    COMPATIBILITY_CASE_MEMBER,
    EXPECTED_SOURCE_BLOCK_COUNT,
    EXPECTED_SOURCE_TASK_COUNT,
    GENERATION_DEVICES,
    SOURCE_BLOCK_ARRAY_MEMBER,
    SOURCE_BLOCK_INDEX_COLUMNS,
    SOURCE_BLOCK_INDEX_MEMBER,
    SOURCE_CACHE_LOCK_MEMBER,
    SourceCache,
)
from .source_cache_worker import MAX_SOURCE_PREFIX_PER_CLASS


class SourceCacheConfig(Protocol):
    contract_hash: str


def build_source_cache_lock(
    root: Path,
    *,
    config: SourceCacheConfig,
    generation_lock: GenerationLock,
    frame: object,
    crossfit: object,
    source_cache: SourceCache,
) -> dict[str, object]:
    validate_source_cache_inventory(source_cache)
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_case_oof_source_cache_lock_v1",
        "status": "COMPLETE_LABEL_FREE_FIXED_SUPPORT_SOURCE_CACHE",
        "config_contract_hash": config.contract_hash,
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "validation_cache_binding_hash": str(
            getattr(frame, "cache_binding_hash", "")
        ),
        "crossfit_fold_lock_hash": str(getattr(crossfit, "lock_hash", "")),
        "source_array_sha256": sha256_file(root / SOURCE_BLOCK_ARRAY_MEMBER),
        "source_index_sha256": sha256_file(root / SOURCE_BLOCK_INDEX_MEMBER),
        "compatibility_case_sha256": sha256_file(
            root / COMPATIBILITY_CASE_MEMBER
        ),
        "source_cache_hash": source_cache.source_cache_hash,
        "source_task_count": EXPECTED_SOURCE_TASK_COUNT,
        "source_block_count": EXPECTED_SOURCE_BLOCK_COUNT,
        "generation_devices": list(GENERATION_DEVICES),
        "one_persistent_worker_per_device": True,
        "tf32_disabled": True,
        "float32_memmap": True,
        "fixed_support_only": True,
        "support_labels_used": False,
        "evaluation_embeddings_used": False,
        "source_experts_updated": False,
    }
    return {**unhashed, "source_cache_lock_hash": stable_hash(unhashed)}


def validate_source_cache_lock(
    root: Path,
    *,
    config: SourceCacheConfig,
    generation_lock: GenerationLock,
    frame: object,
    crossfit: object,
    source_cache: SourceCache,
) -> Mapping[str, object]:
    observed = read_json(root / SOURCE_CACHE_LOCK_MEMBER)
    expected = build_source_cache_lock(
        root,
        config=config,
        generation_lock=generation_lock,
        frame=frame,
        crossfit=crossfit,
        source_cache=source_cache,
    )
    if observed != expected:
        raise ProtocolError("Case-OOF source-cache lock drifted.")
    return observed


def validate_source_cache_inventory(cache: SourceCache) -> None:
    try:
        array = np.load(cache.array_path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise ProtocolError("Case-OOF source cache is unreadable.") from exc
    expected_shape = (
        EXPECTED_SOURCE_BLOCK_COUNT,
        2 * MAX_SOURCE_PREFIX_PER_CLASS,
        COMMON_OUTPUT_DIM,
    )
    if array.shape != expected_shape or array.dtype != np.float32:
        raise ProtocolError("Case-OOF source-cache array geometry drifted.")
    expected_keys = tuple(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS))
    observed_keys: list[tuple[str, int, int]] = []
    if len(cache.index_rows) != EXPECTED_SOURCE_BLOCK_COUNT:
        raise ProtocolError("Case-OOF source-block index coverage drifted.")
    labels = _source_labels()
    for ordinal, row in enumerate(cache.index_rows):
        key = (
            str(row.get("source_center")),
            int(row.get("training_seed", -1)),
            int(row.get("generation_seed", -1)),
        )
        observed_keys.append(key)
        if (
            set(row) != set(SOURCE_BLOCK_INDEX_COLUMNS)
            or int(row.get("block_ordinal", -1)) != ordinal
            or int(row.get("samples_per_class", -1))
            != MAX_SOURCE_PREFIX_PER_CLASS
            or int(row.get("row_count", -1))
            != 2 * MAX_SOURCE_PREFIX_PER_CLASS
            or int(row.get("feature_dim", -1)) != COMMON_OUTPUT_DIM
            or _array_bundle_sha256(array[ordinal], labels)
            != str(row.get("output_sha256"))
        ):
            raise ProtocolError("Case-OOF source-block binding drifted.")
    if tuple(observed_keys) != expected_keys:
        raise ProtocolError("Case-OOF source-block key order drifted.")

    expected_replicas = set(product(CENTERS, TRAINING_SEEDS, CENTERS))
    case_ids_by_replica: dict[tuple[str, int, str], set[str]] = {}
    query_case_ids: dict[str, set[str]] = {}
    for row in cache.compatibility_case_rows:
        if set(row) != set(COMPATIBILITY_CASE_COLUMNS):
            raise ProtocolError("Case-OOF compatibility schema drifted.")
        key = (
            str(row["source_center"]),
            int(row["training_seed"]),
            str(row["query_center"]),
        )
        case_id = str(row["case_id"])
        cases = case_ids_by_replica.setdefault(key, set())
        if case_id in cases:
            raise ProtocolError("Case-OOF compatibility case is duplicated.")
        cases.add(case_id)
        query_case_ids.setdefault(key[2], set()).add(case_id)
        numeric = np.asarray(
            [
                float(row["marginal_variational_energy"]),
                float(row["class_0_energy"]),
                float(row["class_1_energy"]),
                float(row["class_0_common_reconstruction_mse"]),
                float(row["class_1_common_reconstruction_mse"]),
                float(row["class_0_normalized_ps_kl"]),
                float(row["class_1_normalized_ps_kl"]),
            ],
            dtype=np.float64,
        )
        if (
            not np.isfinite(numeric).all()
            or int(row["row_count"]) <= 0
            or str(row["query_partition_role"]) != "support"
            or str(row["class_prior_json"]) != "[0.5,0.5]"
            or _truthy(row["labels_used"])
            or _truthy(row["evaluation_embeddings_used"])
            or _truthy(row["source_experts_updated"])
            or _truthy(row["exact_nelbo_claimed"])
        ):
            raise ProtocolError("Case-OOF compatibility row drifted.")
    if set(case_ids_by_replica) != expected_replicas:
        raise ProtocolError("Case-OOF compatibility grid is incomplete.")
    for source, seed, query in expected_replicas:
        cases = case_ids_by_replica[(source, seed, query)]
        if len(cases) != 2 or cases != query_case_ids[query]:
            raise ProtocolError("Case-OOF fixed-support cases drifted.")


def _source_labels() -> np.ndarray:
    return np.concatenate(
        (
            np.zeros(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
            np.ones(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
        )
    )


def _array_bundle_sha256(embeddings: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in (embeddings, labels):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


__all__ = (
    "build_source_cache_lock",
    "validate_source_cache_inventory",
    "validate_source_cache_lock",
)
