"""Typed contracts and semantic validation for resident expert streams.

This module owns the durable stream inventory independently of CUDA execution
and filesystem staging.  Keeping the schema here lets GPU workers, classifier
workers, and reconstruction code share one fail-closed contract without
importing the source-generation orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
from itertools import product
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ...routing.variational_compatibility import ENERGY_SEMANTICS


SOURCE_ROWS_PER_CLASS = 270
EXPECTED_TASK_COUNT = len(CENTERS) * len(TRAINING_SEEDS)
EXPECTED_STREAM_COUNT = EXPECTED_TASK_COUNT * len(GENERATION_SEEDS)

SOURCE_ARRAY_MEMBER = "arrays/resident_expert_streams.npy"
SOURCE_INDEX_MEMBER = "manifests/resident_expert_stream_index.json"
SOURCE_LOCK_MEMBER = "manifests/resident_expert_stream_lock.json"
COMPATIBILITY_MEMBER = "manifests/support_compatibility.json"
CHECKPOINT_DIRECTORY = "checkpoints/resident_expert_streams"


@dataclass(frozen=True)
class ResidentExpertStreamRecord:
    block_ordinal: int
    source_center: str
    training_seed: int
    generation_seed: int
    stream_id: str
    expert_lock_hash: str
    rows_per_class: int
    output_sha256: str

    @property
    def key(self) -> tuple[str, int, int]:
        return self.source_center, self.training_seed, self.generation_seed

    def to_payload(self) -> dict[str, object]:
        return {
            "block_ordinal": self.block_ordinal,
            "source_center": self.source_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "stream_id": self.stream_id,
            "expert_lock_hash": self.expert_lock_hash,
            "rows_per_class": self.rows_per_class,
            "row_count": 2 * self.rows_per_class,
            "feature_dim": COMMON_OUTPUT_DIM,
            "output_sha256": self.output_sha256,
        }


@dataclass(frozen=True)
class ResidentExpertStreamCache:
    root: Path
    source_array_path: Path
    records: tuple[ResidentExpertStreamRecord, ...]
    lock_payload: Mapping[str, object]
    compatibility_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        validate_resident_stream_cache(self)
        object.__setattr__(self, "lock_payload", MappingProxyType(dict(self.lock_payload)))
        object.__setattr__(
            self,
            "compatibility_payload",
            MappingProxyType(dict(self.compatibility_payload)),
        )

    @cached_property
    def by_key(self) -> Mapping[tuple[str, int, int], ResidentExpertStreamRecord]:
        return MappingProxyType({record.key: record for record in self.records})

    @property
    def lock_hash(self) -> str:
        return str(self.lock_payload["source_stream_lock_hash"])

    @property
    def compatibility_hash(self) -> str:
        return str(self.compatibility_payload["compatibility_hash"])

    def block(self, source: str, training_seed: int, generation_seed: int) -> np.ndarray:
        try:
            ordinal = self.by_key[
                (str(source), int(training_seed), int(generation_seed))
            ].block_ordinal
        except KeyError as exc:
            raise ProtocolError("HARP v7 resident expert stream is absent.") from exc
        values = np.load(self.source_array_path, mmap_mode="r", allow_pickle=False)
        return values[ordinal]


def validate_resident_stream_cache(cache: ResidentExpertStreamCache) -> None:
    """Validate semantic inventory and every persisted stream block."""

    expected_keys = tuple(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS))
    observed = tuple(record.key for record in cache.records)
    compatibility = cache.compatibility_payload
    replicas = compatibility.get("replicas")
    if (
        observed != expected_keys
        or [record.block_ordinal for record in cache.records]
        != list(range(EXPECTED_STREAM_COUNT))
        or any(record.rows_per_class != SOURCE_ROWS_PER_CLASS for record in cache.records)
        or cache.lock_payload.get("status")
        != "COMPLETE_LABEL_FREE_RESIDENT_EXPERT_STREAMS"
        or cache.lock_payload.get("stream_count") != EXPECTED_STREAM_COUNT
        or cache.lock_payload.get("labels_consumed") is not False
        or cache.lock_payload.get("source_experts_updated") is not False
        or compatibility.get("schema_version")
        != "midogpp_harp_v7_support_compatibility_surface_v1"
        or compatibility.get("training_seeds") != list(TRAINING_SEEDS)
        or compatibility.get("energy_semantics") != ENERGY_SEMANTICS
        or compatibility.get("all_replicas_used_without_selection") is not True
        or compatibility.get("computed_while_expert_resident") is not True
        or compatibility.get("exact_nelbo") is not False
        or compatibility.get("labels_consumed") is not False
        or compatibility.get("evaluation_rows_consumed") is not False
        or not isinstance(replicas, list)
        or len(replicas) != EXPECTED_TASK_COUNT
        or [
            (row.get("source_center"), row.get("training_seed"))
            for row in replicas
            if isinstance(row, Mapping)
        ]
        != list(product(CENTERS, TRAINING_SEEDS))
    ):
        raise ProtocolError("HARP v7 resident expert stream inventory drifted.")
    values = np.load(cache.source_array_path, mmap_mode="r", allow_pickle=False)
    if (
        values.shape
        != (EXPECTED_STREAM_COUNT, 2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM)
        or values.dtype != np.float32
    ):
        raise ProtocolError("HARP v7 resident expert stream array drifted.")
    for record in cache.records:
        if record.output_sha256 != _array_bundle_sha256(values[record.block_ordinal]):
            raise ProtocolError(
                "HARP v7 resident expert stream semantic output hash drifted."
            )


def _array_bundle_sha256(embeddings: np.ndarray) -> str:
    labels = np.concatenate(
        (
            np.zeros(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
            np.ones(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
        )
    )
    digest = hashlib.sha256()
    for values in (np.asarray(embeddings), labels):
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def source_block_sha256(embeddings: np.ndarray) -> str:
    """Return the GenerationLock-compatible semantic hash for one stream block."""

    values = np.asarray(embeddings)
    if (
        values.shape != (2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM)
        or values.dtype != np.float32
    ):
        raise ProtocolError("HARP v7 resident expert block geometry drifted.")
    return _array_bundle_sha256(values)


__all__ = (
    "CHECKPOINT_DIRECTORY",
    "COMPATIBILITY_MEMBER",
    "EXPECTED_STREAM_COUNT",
    "EXPECTED_TASK_COUNT",
    "ResidentExpertStreamCache",
    "ResidentExpertStreamRecord",
    "SOURCE_ARRAY_MEMBER",
    "SOURCE_INDEX_MEMBER",
    "SOURCE_LOCK_MEMBER",
    "SOURCE_ROWS_PER_CLASS",
    "source_block_sha256",
    "validate_resident_stream_cache",
)
