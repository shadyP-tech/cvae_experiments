"""Independent contracts for the consumed Stage-90 exact-tail source cache.

The cache is experiment-local.  It never reads an earlier Stage-90 product and
it carries generated embeddings and label-free CVAE components only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError


SOURCE_ROWS_PER_CLASS = 270
GENERATION_DEVICES = ("cuda:0", "cuda:1")
EXPECTED_SOURCE_TASK_COUNT = len(CENTERS) * len(TRAINING_SEEDS)
EXPECTED_SOURCE_STREAM_COUNT = (
    len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
)
# Only legal query/source pairs are scored.  q==source can never participate in
# either strict H/q/e development or target routing, so it is neither computed
# nor persisted as a feature component.
EXPECTED_COMPONENT_RECORD_COUNT = (
    len(CENTERS) * (len(CENTERS) - 1) * len(TRAINING_SEEDS)
)

SOURCE_ARRAY_MEMBER = "arrays/utility_aligned_source_prefixes.npy"
COMPONENT_ARRAY_MEMBER = "arrays/utility_aligned_support_components.npy"
SOURCE_INDEX_MEMBER = "tables/utility_aligned_source_streams.csv"
COMPONENT_INDEX_MEMBER = "tables/utility_aligned_support_components.csv"
SOURCE_CACHE_LOCK_MEMBER = "manifests/utility_aligned_source_cache_lock.json"

SOURCE_INDEX_COLUMNS = (
    "schema_version",
    "block_ordinal",
    "source_center",
    "training_seed",
    "generation_seed",
    "stream_id",
    "expert_lock_hash",
    "rows_per_class",
    "row_count",
    "feature_dim",
    "output_sha256",
)
COMPONENT_INDEX_COLUMNS = (
    "schema_version",
    "component_ordinal",
    "source_center",
    "training_seed",
    "query_center",
    "support_start",
    "support_stop",
    "support_row_count",
    "support_case_count",
    "support_partition_hash",
    "case_equal_energy",
    "linear_kernel_mmd2_by_generation_seed_json",
    "labels_consumed",
    "evaluation_embeddings_consumed",
    "exact_nelbo_claimed",
)


@dataclass(frozen=True)
class SourceBlockRecord:
    block_ordinal: int
    source_center: str
    training_seed: int
    generation_seed: int
    stream_id: str
    expert_lock_hash: str
    rows_per_class: int
    row_count: int
    feature_dim: int
    output_sha256: str

    @property
    def key(self) -> tuple[str, int, int]:
        return self.source_center, self.training_seed, self.generation_seed

    def to_row(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_utility_aligned_source_stream_v1",
            "block_ordinal": self.block_ordinal,
            "source_center": self.source_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "stream_id": self.stream_id,
            "expert_lock_hash": self.expert_lock_hash,
            "rows_per_class": self.rows_per_class,
            "row_count": self.row_count,
            "feature_dim": self.feature_dim,
            "output_sha256": self.output_sha256,
        }


@dataclass(frozen=True)
class LabelFreeComponentRecord:
    component_ordinal: int
    source_center: str
    training_seed: int
    query_center: str
    support_start: int
    support_stop: int
    support_row_count: int
    support_case_count: int
    support_partition_hash: str
    case_equal_energy: float
    linear_kernel_mmd2_by_generation_seed: Mapping[int, float]

    def __post_init__(self) -> None:
        values = {
            int(key): float(value)
            for key, value in self.linear_kernel_mmd2_by_generation_seed.items()
        }
        if set(values) != set(GENERATION_SEEDS):
            raise ProtocolError("Stage-90 component MMD seed coverage drifted.")
        object.__setattr__(
            self, "linear_kernel_mmd2_by_generation_seed", MappingProxyType(values)
        )

    @property
    def key(self) -> tuple[str, str, int]:
        return self.query_center, self.source_center, self.training_seed

    def to_row(self) -> dict[str, object]:
        import json

        return {
            "schema_version": "midogpp_stage90_utility_aligned_support_component_v1",
            "component_ordinal": self.component_ordinal,
            "source_center": self.source_center,
            "training_seed": self.training_seed,
            "query_center": self.query_center,
            "support_start": self.support_start,
            "support_stop": self.support_stop,
            "support_row_count": self.support_row_count,
            "support_case_count": self.support_case_count,
            "support_partition_hash": self.support_partition_hash,
            "case_equal_energy": self.case_equal_energy,
            "linear_kernel_mmd2_by_generation_seed_json": json.dumps(
                {str(key): value for key, value in self.linear_kernel_mmd2_by_generation_seed.items()},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "labels_consumed": False,
            "evaluation_embeddings_consumed": False,
            "exact_nelbo_claimed": False,
        }


@dataclass(frozen=True)
class SourceCache:
    root: Path
    source_array_path: Path
    component_array_path: Path
    source_records: tuple[SourceBlockRecord, ...]
    component_records: tuple[LabelFreeComponentRecord, ...]
    support_scratch_hash: str

    def __post_init__(self) -> None:
        from .source_cache_validation import validate_source_cache_inventory

        validate_source_cache_inventory(self)

    @cached_property
    def source_by_key(self) -> Mapping[tuple[str, int, int], SourceBlockRecord]:
        return MappingProxyType({record.key: record for record in self.source_records})

    @cached_property
    def component_by_key(
        self,
    ) -> Mapping[tuple[str, str, int], LabelFreeComponentRecord]:
        return MappingProxyType({record.key: record for record in self.component_records})

    @cached_property
    def source_cache_hash(self) -> str:
        return stable_hash(
            {
                "schema_version": "midogpp_stage90_utility_aligned_source_cache_v1",
                "source_rows": [record.to_row() for record in self.source_records],
                "component_rows": [record.to_row() for record in self.component_records],
                "support_scratch_hash": self.support_scratch_hash,
                "labels_consumed": False,
            }
        )

    def source_block(
        self, source_center: str, training_seed: int, generation_seed: int
    ) -> np.ndarray:
        try:
            ordinal = self.source_by_key[
                (str(source_center), int(training_seed), int(generation_seed))
            ].block_ordinal
        except KeyError as exc:
            raise ProtocolError("Stage-90 source stream is absent.") from exc
        values = np.load(self.source_array_path, mmap_mode="r", allow_pickle=False)
        return values[ordinal]

    def component_arrays(
        self, *, query_center: str, source_center: str, training_seed: int
    ) -> tuple[Mapping[int, np.ndarray], Mapping[int, np.ndarray]]:
        try:
            record = self.component_by_key[
                (str(query_center), str(source_center), int(training_seed))
            ]
        except KeyError as exc:
            raise ProtocolError("Stage-90 support component is absent.") from exc
        values = np.load(self.component_array_path, mmap_mode="r", allow_pickle=False)
        task_ordinal = CENTERS.index(record.source_center) * len(TRAINING_SEEDS) + TRAINING_SEEDS.index(record.training_seed)
        start, stop = record.support_start, record.support_stop
        return (
            {label: values[task_ordinal, label, start:stop] for label in (0, 1)},
            {label: values[task_ordinal, 2 + label, start:stop] for label in (0, 1)},
        )


__all__ = (
    "COMPONENT_ARRAY_MEMBER",
    "COMPONENT_INDEX_COLUMNS",
    "COMPONENT_INDEX_MEMBER",
    "EXPECTED_COMPONENT_RECORD_COUNT",
    "EXPECTED_SOURCE_STREAM_COUNT",
    "EXPECTED_SOURCE_TASK_COUNT",
    "GENERATION_DEVICES",
    "SOURCE_ARRAY_MEMBER",
    "SOURCE_CACHE_LOCK_MEMBER",
    "SOURCE_INDEX_COLUMNS",
    "SOURCE_INDEX_MEMBER",
    "SOURCE_ROWS_PER_CLASS",
    "LabelFreeComponentRecord",
    "SourceBlockRecord",
    "SourceCache",
)
