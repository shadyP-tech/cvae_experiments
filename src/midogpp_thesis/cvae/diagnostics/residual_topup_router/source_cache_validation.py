"""Independent scientific validation of the persisted source cache."""

from __future__ import annotations

import hashlib
import json
from itertools import product
from typing import Mapping

import numpy as np

from ...generation.generation import source_generation_plan
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    COMMON_FEATURE_DIM,
    GENERATION_SEEDS,
    MAX_SOURCE_PREFIX_PER_CLASS,
    TRAINING_SEEDS,
)


def validate_source_cache_contents(
    source_cache: object,
    *,
    generation_lock: object,
    partitions: object,
) -> dict[str, object]:
    """Recompute every block hash and support-case binding from persisted bytes."""

    try:
        array = np.load(getattr(source_cache, "array_path"), mmap_mode="r")
        index_rows = tuple(getattr(source_cache, "index_rows"))
        energy_rows = tuple(getattr(source_cache, "compatibility_case_rows"))
    except (OSError, TypeError, ValueError) as exc:
        raise ProtocolError("Residual top-up source cache is unreadable.") from exc
    expected_shape = (
        len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS),
        2 * MAX_SOURCE_PREFIX_PER_CLASS,
        COMMON_FEATURE_DIM,
    )
    if array.shape != expected_shape or array.dtype != np.float32:
        raise ProtocolError("Residual top-up source-cache array geometry drifted.")

    generation_keys = {
        (key.source_center, key.training_seed, key.generation_seed): key
        for key in source_generation_plan(generation_lock)
    }
    expected_keys = tuple(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS))
    labels = np.concatenate(
        (
            np.zeros(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
            np.ones(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
        )
    )
    observed_keys: list[tuple[str, int, int]] = []
    for ordinal, row in enumerate(index_rows):
        key = (
            str(row.get("source_center")),
            _integer(row.get("training_seed")),
            _integer(row.get("generation_seed")),
        )
        observed_keys.append(key)
        generation_key = generation_keys.get(key)
        if (
            generation_key is None
            or _integer(row.get("block_ordinal")) != ordinal
            or str(row.get("stream_id")) != generation_key.stream_id
            or str(row.get("expert_lock_hash")) != generation_key.expert_lock_hash
            or _integer(row.get("samples_per_class"))
            != MAX_SOURCE_PREFIX_PER_CLASS
            or _integer(row.get("row_count"))
            != 2 * MAX_SOURCE_PREFIX_PER_CLASS
            or _integer(row.get("feature_dim")) != COMMON_FEATURE_DIM
            or _array_bundle_sha256(array[ordinal], labels)
            != str(row.get("output_sha256"))
        ):
            raise ProtocolError("Residual top-up source block binding drifted.")
    if tuple(observed_keys) != expected_keys:
        raise ProtocolError("Residual top-up source block key order drifted.")

    support_by_center = getattr(partitions, "support_rows_by_center", {})
    expected_cases = {
        center: {str(row.case_id) for row in support_by_center[center]}
        for center in CENTERS
    }
    cases_by_replica: dict[tuple[str, int, str], set[str]] = {}
    for row in energy_rows:
        key = (
            str(row.get("source_center")),
            _integer(row.get("training_seed")),
            str(row.get("query_center")),
        )
        case_id = str(row.get("case_id"))
        cases = cases_by_replica.setdefault(key, set())
        if case_id in cases:
            raise ProtocolError("Residual top-up compatibility case is duplicated.")
        cases.add(case_id)
        numeric = np.asarray(
            [
                float(row.get("marginal_variational_energy")),
                float(row.get("class_0_energy")),
                float(row.get("class_1_energy")),
                float(row.get("class_0_common_reconstruction_mse")),
                float(row.get("class_1_common_reconstruction_mse")),
                float(row.get("class_0_normalized_ps_kl")),
                float(row.get("class_1_normalized_ps_kl")),
            ],
            dtype=np.float64,
        )
        if (
            key[0] not in CENTERS
            or key[1] not in TRAINING_SEEDS
            or key[2] not in CENTERS
            or case_id not in expected_cases.get(key[2], set())
            or not np.isfinite(numeric).all()
            or _integer(row.get("row_count")) <= 0
            or str(row.get("query_partition_role")) != "support"
            or str(row.get("class_prior_json")) != "[0.5,0.5]"
            or _truthy(row.get("labels_used"))
            or _truthy(row.get("exact_nelbo_claimed"))
        ):
            raise ProtocolError("Residual top-up compatibility support binding drifted.")
    expected_replicas = set(product(CENTERS, TRAINING_SEEDS, CENTERS))
    if set(cases_by_replica) != expected_replicas or any(
        cases_by_replica[(source, seed, query)] != expected_cases[query]
        for source, seed, query in expected_replicas
    ):
        raise ProtocolError("Residual top-up compatibility grid is incomplete.")
    return {
        "source_block_count": len(index_rows),
        "compatibility_case_row_count": len(energy_rows),
        "all_source_block_hashes_verified": True,
        "support_case_binding_verified": True,
    }


def _array_bundle_sha256(embeddings: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in (embeddings, labels):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _integer(value: object) -> int:
    if isinstance(value, bool):
        raise ProtocolError("Residual top-up integer field is invalid.")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Residual top-up integer field is invalid.") from exc


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


__all__ = ("validate_source_cache_contents",)
