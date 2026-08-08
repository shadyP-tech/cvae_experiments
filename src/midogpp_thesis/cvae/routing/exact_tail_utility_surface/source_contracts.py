"""Immutable contracts for exact-tail source generation and feature caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...generation.contracts import SourceGenerationKey
from ...protocol import ProtocolError
from .contracts import CENTERS


SOURCE_CACHE_SCHEMA = "midogpp_exact_tail_source_cache_v1"
SOURCE_RECORD_SCHEMA = "midogpp_exact_tail_source_stream_v1"
COMPONENT_RECORD_SCHEMA = "midogpp_exact_tail_feature_components_v1"


class SourceGenerationConfigProtocol(Protocol):
    """Minimal configuration read by neutral source-feature generation."""

    expert_bank_root: Path
    generation_lock_root: Path
    classifier: ClassifierSpec


@dataclass(frozen=True)
class SourceGenerationConfig:
    """Concrete bridge for producers that do not own the exact-tail config."""

    expert_bank_root: Path
    generation_lock_root: Path
    classifier: ClassifierSpec


@dataclass(frozen=True)
class SourceFeatureInputs:
    """Label-free support inputs without a DevelopmentPartition dependency."""

    support_array_path_by_center: Mapping[str, Path]
    support_case_ids_by_center: Mapping[str, tuple[str, ...]]
    support_partition_hash_by_center: Mapping[str, str]

    def __post_init__(self) -> None:
        paths = {
            str(center): Path(path)
            for center, path in self.support_array_path_by_center.items()
        }
        case_ids = {
            str(center): tuple(str(value) for value in values)
            for center, values in self.support_case_ids_by_center.items()
        }
        partition_hashes = {
            str(center): str(value)
            for center, value in self.support_partition_hash_by_center.items()
        }
        if (
            tuple(paths) != CENTERS
            or tuple(case_ids) != CENTERS
            or tuple(partition_hashes) != CENTERS
            or any(not values for values in case_ids.values())
            or any(
                len(value) not in {16, 64}
                or any(character not in "0123456789abcdef" for character in value)
                for value in partition_hashes.values()
            )
        ):
            raise ProtocolError("Exact-tail source-feature input coverage drifted.")
        object.__setattr__(self, "support_array_path_by_center", MappingProxyType(paths))
        object.__setattr__(self, "support_case_ids_by_center", MappingProxyType(case_ids))
        object.__setattr__(
            self,
            "support_partition_hash_by_center",
            MappingProxyType(partition_hashes),
        )


@dataclass(frozen=True)
class SourceBlockRecord:
    source_center: str
    training_seed: int
    generation_seed: int
    stream_id: str
    expert_lock_hash: str
    relative_path: str
    file_sha256: str
    output_sha256: str
    rows_per_class: int
    feature_dim: int

    @property
    def key(self) -> tuple[str, int, int]:
        return self.source_center, self.training_seed, self.generation_seed

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": SOURCE_RECORD_SCHEMA,
            "source_center": self.source_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "stream_id": self.stream_id,
            "expert_lock_hash": self.expert_lock_hash,
            "relative_path": self.relative_path,
            "file_sha256": self.file_sha256,
            "output_sha256": self.output_sha256,
            "rows_per_class": self.rows_per_class,
            "feature_dim": self.feature_dim,
            "dtype": "float32",
            "class_row_order": "class_0_then_class_1",
        }


@dataclass(frozen=True)
class FeatureComponentRecord:
    query_center: str
    candidate_source: str
    training_seed: int
    relative_path: str
    file_sha256: str
    case_equal_energy: float
    linear_kernel_mmd2_by_generation_seed: Mapping[int, float]
    support_partition_hash: str

    @property
    def key(self) -> tuple[str, str, int]:
        return self.query_center, self.candidate_source, self.training_seed

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "linear_kernel_mmd2_by_generation_seed",
            MappingProxyType(
                {
                    int(key): float(value)
                    for key, value in self.linear_kernel_mmd2_by_generation_seed.items()
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": COMPONENT_RECORD_SCHEMA,
            "query_center": self.query_center,
            "candidate_source": self.candidate_source,
            "training_seed": self.training_seed,
            "relative_path": self.relative_path,
            "file_sha256": self.file_sha256,
            "case_equal_energy": self.case_equal_energy,
            "linear_kernel_mmd2_by_generation_seed": {
                str(key): value
                for key, value in self.linear_kernel_mmd2_by_generation_seed.items()
            },
            "distribution_feature_semantics": "linear_kernel_mmd_squared",
            "support_partition_hash": self.support_partition_hash,
            "labels_consumed": False,
        }


@dataclass(frozen=True)
class GeneratedDevelopmentCache:
    root: Path
    generation_lock_hash: str
    bank_lock_hash: str
    source_records: tuple[SourceBlockRecord, ...]
    component_records: tuple[FeatureComponentRecord, ...]
    cache_hash: str

    @property
    def source_by_key(self) -> Mapping[tuple[str, int, int], SourceBlockRecord]:
        return MappingProxyType({record.key: record for record in self.source_records})

    @property
    def component_by_key(
        self,
    ) -> Mapping[tuple[str, str, int], FeatureComponentRecord]:
        return MappingProxyType({record.key: record for record in self.component_records})


@dataclass(frozen=True)
class ExpertTask:
    source_center: str
    training_seed: int
    generation_keys: tuple[SourceGenerationKey, ...]
    existing_source_path_by_generation_seed: Mapping[int, str]
    query_centers: tuple[str, ...]
    support_array_path_by_center: Mapping[str, str]
    support_case_ids_by_center: Mapping[str, tuple[str, ...]]
    support_partition_hash_by_center: Mapping[str, str]
    device: str

    def __post_init__(self) -> None:
        for field_name in (
            "existing_source_path_by_generation_seed",
            "support_array_path_by_center",
            "support_case_ids_by_center",
            "support_partition_hash_by_center",
        ):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Rebuild immutable mappings after crossing a spawned-process queue."""

        return (
            type(self),
            (
                self.source_center,
                self.training_seed,
                self.generation_keys,
                dict(self.existing_source_path_by_generation_seed),
                self.query_centers,
                dict(self.support_array_path_by_center),
                dict(self.support_case_ids_by_center),
                dict(self.support_partition_hash_by_center),
                self.device,
            ),
        )


__all__ = (
    "COMPONENT_RECORD_SCHEMA",
    "SOURCE_CACHE_SCHEMA",
    "SOURCE_RECORD_SCHEMA",
    "ExpertTask",
    "FeatureComponentRecord",
    "GeneratedDevelopmentCache",
    "SourceFeatureInputs",
    "SourceGenerationConfig",
    "SourceGenerationConfigProtocol",
    "SourceBlockRecord",
)
