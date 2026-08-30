"""Immutable contracts for SCEPTRE v5 physical source streams."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import numpy as np

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.generation.contracts import (
    COMMON_OUTPUT_DIM,
    TOTAL_PER_CLASS,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SOURCE_ARRAY_MEMBER = "arrays/sceptre_v5_source_streams.npy"
SOURCE_INDEX_MEMBER = "manifests/sceptre_v5_source_stream_index.json"
SOURCE_RECEIPT_MEMBER = "manifests/sceptre_v5_source_stream_receipt.json"
CHECKPOINT_DIRECTORY = "checkpoints/sceptre_v5_source_streams"


class SourceRuntimeConfig(Protocol):
    """Minimum configuration surface accepted by the physical source phase."""

    runtime: Mapping[str, object]
    config_hash: str


@dataclass(frozen=True)
class SourceGeometry:
    """Complete geometry identity; non-production values require a test token."""

    centers: tuple[str, ...]
    training_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    rows_per_class: int
    feature_dim: int

    def __post_init__(self) -> None:
        if (
            not self.centers
            or len(set(self.centers)) != len(self.centers)
            or not self.training_seeds
            or len(set(self.training_seeds)) != len(self.training_seeds)
            or not self.generation_seeds
            or len(set(self.generation_seeds)) != len(self.generation_seeds)
            or self.rows_per_class <= 0
            or self.feature_dim <= 0
        ):
            raise ProtocolError("SCEPTRE v5 source geometry is malformed.")

    @property
    def task_count(self) -> int:
        return len(self.centers) * len(self.training_seeds)

    @property
    def stream_count(self) -> int:
        return self.task_count * len(self.generation_seeds)

    @property
    def array_shape(self) -> tuple[int, int, int]:
        return self.stream_count, 2 * self.rows_per_class, self.feature_dim

    def to_payload(self) -> dict[str, object]:
        return {
            "centers": list(self.centers),
            "training_seeds": list(self.training_seeds),
            "generation_seeds": list(self.generation_seeds),
            "rows_per_class": self.rows_per_class,
            "feature_dim": self.feature_dim,
            "task_count": self.task_count,
            "stream_count": self.stream_count,
            "array_shape": list(self.array_shape),
        }


PRODUCTION_SOURCE_GEOMETRY = SourceGeometry(
    centers=tuple(CENTERS),
    training_seeds=tuple(TRAINING_SEEDS),
    generation_seeds=tuple(GENERATION_SEEDS),
    rows_per_class=TOTAL_PER_CLASS,
    feature_dim=COMMON_OUTPUT_DIM,
)


@dataclass(frozen=True)
class SourceRuntimeTestMode:
    """Explicit, dependency-injected small-geometry seam for focused tests only."""

    geometry: SourceGeometry
    generation_keys: tuple[object, ...]
    generate_block: Callable[[object, int, str], np.ndarray]

    def __post_init__(self) -> None:
        if self.geometry == PRODUCTION_SOURCE_GEOMETRY:
            raise ProtocolError("Production source geometry cannot use the test seam.")
        if not callable(self.generate_block):
            raise ProtocolError("SCEPTRE v5 source test generator is not callable.")


@dataclass(frozen=True)
class SourceStreamRecord:
    block_ordinal: int
    source_center: str
    training_seed: int
    generation_seed: int
    stream_id: str
    expert_lock_hash: str
    rows_per_class: int
    feature_dim: int
    output_sha256: str
    array_sha256: str

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
            "feature_dim": self.feature_dim,
            "output_sha256": self.output_sha256,
            "array_sha256": self.array_sha256,
        }


@dataclass(frozen=True)
class SourceStreamStore:
    """Validated read-only view of the physical source-stream NPY store."""

    root: Path
    array_path: Path
    index_path: Path
    receipt_path: Path
    geometry: SourceGeometry
    records: tuple[SourceStreamRecord, ...]
    receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        expected_keys = tuple(
            product(
                self.geometry.centers,
                self.geometry.training_seeds,
                self.geometry.generation_seeds,
            )
        )
        if (
            tuple(record.key for record in self.records) != expected_keys
            or tuple(record.block_ordinal for record in self.records)
            != tuple(range(self.geometry.stream_count))
            or any(
                record.rows_per_class != self.geometry.rows_per_class
                or record.feature_dim != self.geometry.feature_dim
                for record in self.records
            )
        ):
            raise ProtocolError("SCEPTRE v5 source-stream inventory drifted.")
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    @cached_property
    def by_key(self) -> Mapping[tuple[str, int, int], SourceStreamRecord]:
        return MappingProxyType({record.key: record for record in self.records})

    @property
    def receipt_hash(self) -> str:
        return str(self.receipt["receipt_sha256"])

    @property
    def attempt_id(self) -> str:
        return str(self.receipt["attempt_id"])

    def block(
        self, source_center: str, training_seed: int, generation_seed: int
    ) -> np.ndarray:
        try:
            record = self.by_key[
                (str(source_center), int(training_seed), int(generation_seed))
            ]
        except KeyError as exc:
            raise ProtocolError("SCEPTRE v5 source stream is absent.") from exc
        values = np.load(self.array_path, mmap_mode="r", allow_pickle=False)
        block = values[record.block_ordinal]
        if block.flags.writeable:
            raise ProtocolError("SCEPTRE v5 source memmap unexpectedly became writable.")
        return block


__all__ = (
    "CHECKPOINT_DIRECTORY",
    "PRODUCTION_SOURCE_GEOMETRY",
    "SOURCE_ARRAY_MEMBER",
    "SOURCE_INDEX_MEMBER",
    "SOURCE_RECEIPT_MEMBER",
    "SourceGeometry",
    "SourceRuntimeConfig",
    "SourceRuntimeTestMode",
    "SourceStreamRecord",
    "SourceStreamStore",
)
