"""Domain model and durable serialization for the residual top-up source cache."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cached_property
import os
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...generation.contracts import GenerationLock, SourceGenerationKey
from ...protocol import ProtocolError
from ...routing.dense_residual_soft_router import (
    DEFAULT_SCALE_FLOOR,
    OwnSourceCalibration,
    ReplicaKey,
    calibrate_own_source_energies,
)
from ._source_worker import (
    _atomic_json,
    _atomic_save_npy,
    _json,
    _sha256_array,
    _sha256_file,
    canonical_compatibility_case_row as _canonical_compatibility_case_row,
    canonical_source_index_row as _canonical_source_index_row,
    validate_source_cache_inventory as _validate_source_cache_inventory,
)
from .contracts import (
    CENTERS,
    COMMON_FEATURE_DIM,
    EXPECTED_SOURCE_BLOCK_COUNT,
    EXPECTED_SOURCE_TASK_COUNT,
    GENERATION_DEVICES,
    GENERATION_SEEDS,
    MAX_SOURCE_PREFIX_PER_CLASS,
    TRAINING_SEEDS,
)
from .partitions import LabelFreeValidationFrame, PartitionSurface


SOURCE_BLOCK_ARRAY_MEMBER = "arrays/source_prefix_blocks.npy"
SOURCE_BLOCK_INDEX_MEMBER = "tables/source_block_index.csv"
COMPATIBILITY_CASE_MEMBER = "tables/compatibility_case_energy.csv"
SOURCE_CACHE_LOCK_MEMBER = "manifests/source_cache_lock.json"

SOURCE_BLOCK_INDEX_COLUMNS = (
    "schema_version",
    "block_ordinal",
    "source_center",
    "training_seed",
    "generation_seed",
    "stream_id",
    "expert_lock_hash",
    "samples_per_class",
    "row_count",
    "feature_dim",
    "output_sha256",
)
COMPATIBILITY_CASE_COLUMNS = (
    "schema_version",
    "source_center",
    "training_seed",
    "query_center",
    "case_id",
    "query_partition_role",
    "row_count",
    "marginal_variational_energy",
    "class_0_energy",
    "class_1_energy",
    "class_0_common_reconstruction_mse",
    "class_1_common_reconstruction_mse",
    "class_0_normalized_ps_kl",
    "class_1_normalized_ps_kl",
    "class_prior_json",
    "labels_used",
    "exact_nelbo_claimed",
)


class SourceCacheConfig(Protocol):
    contract_hash: str


@dataclass(frozen=True)
class CachedSourceKey:
    source_center: str
    training_seed: int
    generation_seed: int
    stream_id: str
    expert_lock_hash: str


@dataclass(frozen=True)
class CachedSourceBlock:
    key: CachedSourceKey
    embeddings: np.ndarray
    labels: np.ndarray
    output_sha256: str

    @property
    def source_center(self) -> str:
        return self.key.source_center

    @property
    def training_seed(self) -> int:
        return self.key.training_seed

    @property
    def generation_seed(self) -> int:
        return self.key.generation_seed

    @property
    def stream_id(self) -> str:
        return self.key.stream_id

    @property
    def expert_lock_hash(self) -> str:
        return self.key.expert_lock_hash


@dataclass(frozen=True)
class SourceCache:
    array_path: Path
    index_rows: tuple[Mapping[str, object], ...]
    compatibility_case_rows: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        _validate_source_cache_inventory(self)

    @cached_property
    def block_ordinal_by_key(self) -> Mapping[tuple[str, int, int], int]:
        return {
            (
                str(row["source_center"]),
                int(row["training_seed"]),
                int(row["generation_seed"]),
            ): int(row["block_ordinal"])
            for row in self.index_rows
        }

    @cached_property
    def cache_hash(self) -> str:
        return stable_hash(
            {
                "index_rows": [
                    _canonical_source_index_row(row) for row in self.index_rows
                ],
                "compatibility_case_rows": [
                    _canonical_compatibility_case_row(row)
                    for row in self.compatibility_case_rows
                ],
            }
        )

    @property
    def source_cache_hash(self) -> str:
        """Compatibility alias for phase/report code that spells out the role."""

        return self.cache_hash

    def block(
        self,
        source_center: str,
        training_seed: int,
        generation_seed: int,
    ) -> CachedSourceBlock:
        """Return one read-only 256-per-class stream from the shared memmap."""

        key_tuple = (str(source_center), int(training_seed), int(generation_seed))
        ordinal = self.block_ordinal_by_key.get(key_tuple)
        if ordinal is None:
            raise ProtocolError(f"Residual top-up source block is unknown: {key_tuple}.")
        row = self.index_rows[ordinal]
        array = np.load(self.array_path, mmap_mode="r")
        embeddings = array[ordinal]
        labels = np.concatenate(
            (
                np.zeros(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
                np.ones(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
            )
        )
        labels.setflags(write=False)
        return CachedSourceBlock(
            key=CachedSourceKey(
                source_center=key_tuple[0],
                training_seed=key_tuple[1],
                generation_seed=key_tuple[2],
                stream_id=str(row["stream_id"]),
                expert_lock_hash=str(row["expert_lock_hash"]),
            ),
            embeddings=embeddings,
            labels=labels,
            output_sha256=str(row["output_sha256"]),
        )

    def calibrated_energy_for(
        self,
        query_center: str,
        candidate_sources: Sequence[str],
    ) -> OwnSourceCalibration:
        """Calibrate only the exact candidate subset for one pseudo/real target."""

        query = str(query_center)
        candidates = tuple(str(value) for value in candidate_sources)
        if (
            query not in CENTERS
            or not candidates
            or len(candidates) != len(set(candidates))
            or any(source not in CENTERS for source in candidates)
            or query in candidates
            or candidates != tuple(source for source in CENTERS if source in candidates)
        ):
            raise ProtocolError(
                "Residual top-up calibration requires an exact canonical candidate set "
                "that excludes the query center."
            )

        energy_by_key: dict[tuple[str, int, str], dict[str, float]] = {}
        for row in self.compatibility_case_rows:
            key = (
                str(row["source_center"]),
                int(row["training_seed"]),
                str(row["query_center"]),
            )
            energy_by_key.setdefault(key, {})[str(row["case_id"])] = float(
                row["marginal_variational_energy"]
            )
        try:
            query_map = {
                ReplicaKey(source, seed): energy_by_key[(source, seed, query)]
                for source in candidates
                for seed in TRAINING_SEEDS
            }
            own_map = {
                ReplicaKey(source, seed): energy_by_key[(source, seed, source)]
                for source in candidates
                for seed in TRAINING_SEEDS
            }
        except KeyError as exc:
            raise ProtocolError(
                "Residual top-up calibration lacks a complete source-by-seed grid."
            ) from exc
        return calibrate_own_source_energies(
            query_map,
            own_map,
            candidate_sources=candidates,
            training_seeds=TRAINING_SEEDS,
            scale_floor=DEFAULT_SCALE_FLOOR,
        )


def materialize_source_array(
    array_path: Path,
    *,
    completed: Mapping[tuple[str, int], Mapping[str, object]],
    key_map: Mapping[tuple[str, int, int], SourceGenerationKey],
) -> list[dict[str, object]]:
    """Assemble checkpoint blocks in canonical order, then durably publish them."""

    array_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_array = array_path.with_name(array_path.name + f".{os.getpid()}.tmp")
    target = np.lib.format.open_memmap(
        temporary_array,
        mode="w+",
        dtype=np.float32,
        shape=(
            EXPECTED_SOURCE_BLOCK_COUNT,
            2 * MAX_SOURCE_PREFIX_PER_CLASS,
            COMMON_FEATURE_DIM,
        ),
    )
    index_rows: list[dict[str, object]] = []
    ordinal = 0
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            payload = completed[(source, training_seed)]
            source_array = np.load(Path(str(payload["array_path"])), mmap_mode="r")
            records = payload.get("blocks")
            if not isinstance(records, list):
                raise ProtocolError("Residual top-up source block inventory is absent.")
            for generation_index, generation_seed in enumerate(GENERATION_SEEDS):
                record = records[generation_index]
                if not isinstance(record, Mapping):
                    raise ProtocolError("Residual top-up source block row is malformed.")
                target[ordinal] = source_array[generation_index]
                key = key_map[(source, training_seed, generation_seed)]
                index_rows.append(
                    {
                        "schema_version": "midogpp_residual_topup_source_block_v1",
                        "block_ordinal": ordinal,
                        "source_center": source,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "stream_id": key.stream_id,
                        "expert_lock_hash": key.expert_lock_hash,
                        "samples_per_class": MAX_SOURCE_PREFIX_PER_CLASS,
                        "row_count": 2 * MAX_SOURCE_PREFIX_PER_CLASS,
                        "feature_dim": COMMON_FEATURE_DIM,
                        "output_sha256": str(record["output_sha256"]),
                    }
                )
                ordinal += 1
    target.flush()
    del target
    durable_replace(temporary_array, array_path)
    return index_rows


def write_support_scratch(
    array_path: Path,
    index_path: Path,
    *,
    frame: LabelFreeValidationFrame,
    partitions: PartitionSurface,
) -> Mapping[str, object]:
    rows = [
        row
        for center in CENTERS
        for row in partitions.support_rows_by_center[center]
    ]
    embeddings = frame.embeddings_for(rows)
    offsets: dict[str, object] = {}
    cursor = 0
    for center in CENTERS:
        center_rows = partitions.support_rows_by_center[center]
        stop = cursor + len(center_rows)
        offsets[center] = {
            "start": cursor,
            "stop": stop,
            "case_ids": [row.case_id for row in center_rows],
        }
        cursor = stop
    payload: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_support_scratch_v1",
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "offsets": offsets,
        "array_sha256": _sha256_array(embeddings),
    }
    payload["support_scratch_hash"] = stable_hash(payload)
    _atomic_save_npy(array_path, embeddings)
    _atomic_json(index_path, payload)
    return payload


def build_compatibility_case_rows(
    completed: Mapping[tuple[str, int], Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            raw_rows = completed[(source, training_seed)].get(
                "compatibility_case_records"
            )
            if not isinstance(raw_rows, list):
                raise ProtocolError("Residual top-up compatibility rows are absent.")
            for raw in raw_rows:
                if not isinstance(raw, Mapping):
                    raise ProtocolError("Residual top-up compatibility row is malformed.")
                rows.append(
                    {
                        "schema_version": "midogpp_residual_topup_compatibility_case_v1",
                        **dict(raw),
                        "query_partition_role": "support",
                        "class_prior_json": "[0.5,0.5]",
                        "labels_used": False,
                        "exact_nelbo_claimed": False,
                    }
                )
    rows.sort(
        key=lambda row: (
            str(row["source_center"]),
            int(row["training_seed"]),
            str(row["query_center"]),
            str(row["case_id"]),
        )
    )
    return rows


def load_source_cache(root: Path) -> SourceCache:
    return SourceCache(
        array_path=root / SOURCE_BLOCK_ARRAY_MEMBER,
        index_rows=tuple(read_csv(root / SOURCE_BLOCK_INDEX_MEMBER)),
        compatibility_case_rows=tuple(read_csv(root / COMPATIBILITY_CASE_MEMBER)),
    )


def build_source_cache_lock(
    root: Path,
    *,
    config: SourceCacheConfig,
    generation_lock: GenerationLock,
    frame: LabelFreeValidationFrame,
    partitions: PartitionSurface,
    source_cache: SourceCache,
) -> dict[str, object]:
    unhashed = {
        "schema_version": "midogpp_residual_topup_source_cache_lock_v1",
        "status": "COMPLETE",
        "config_contract_hash": config.contract_hash,
        "bank_lock_hash": generation_lock.bank_lock_hash,
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "validation_cache_binding_hash": frame.cache_binding_hash,
        "support_partition_lock_hash": partitions.lock_hash,
        "source_cache_hash": source_cache.cache_hash,
        "source_array_sha256": _sha256_file(root / SOURCE_BLOCK_ARRAY_MEMBER),
        "source_index_sha256": _sha256_file(root / SOURCE_BLOCK_INDEX_MEMBER),
        "compatibility_case_sha256": _sha256_file(root / COMPATIBILITY_CASE_MEMBER),
        "source_task_count": EXPECTED_SOURCE_TASK_COUNT,
        "source_block_count": EXPECTED_SOURCE_BLOCK_COUNT,
        "expert_load_count": EXPECTED_SOURCE_TASK_COUNT,
        "samples_per_source_class": MAX_SOURCE_PREFIX_PER_CLASS,
        "generation_devices": list(GENERATION_DEVICES),
        "labels_used": False,
        "evaluation_embeddings_used": False,
        "all_training_seeds_retained": True,
        "all_generation_seeds_retained": True,
    }
    return {**unhashed, "source_cache_lock_hash": stable_hash(unhashed)}


def validate_source_cache_lock(
    root: Path,
    *,
    config: SourceCacheConfig,
    generation_lock: GenerationLock,
    frame: LabelFreeValidationFrame,
    partitions: PartitionSurface,
    source_cache: SourceCache,
) -> Mapping[str, object]:
    observed = _json(root / SOURCE_CACHE_LOCK_MEMBER)
    expected = build_source_cache_lock(
        root,
        config=config,
        generation_lock=generation_lock,
        frame=frame,
        partitions=partitions,
        source_cache=source_cache,
    )
    if observed != expected:
        raise ProtocolError(
            "Residual top-up source cache is not bound to current inputs/config."
        )
    return observed


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise ProtocolError(f"Cannot read residual top-up table: {path}.") from exc


def atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            if set(row) != set(columns):
                raise ProtocolError("Residual top-up table schema drifted.")
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def durable_replace(temporary: Path, destination: Path) -> None:
    """Publish a complete memmap and persist the directory entry durably."""

    file_descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
    os.replace(temporary, destination)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


__all__ = (
    "COMPATIBILITY_CASE_COLUMNS",
    "COMPATIBILITY_CASE_MEMBER",
    "CachedSourceBlock",
    "CachedSourceKey",
    "SOURCE_BLOCK_ARRAY_MEMBER",
    "SOURCE_BLOCK_INDEX_COLUMNS",
    "SOURCE_BLOCK_INDEX_MEMBER",
    "SOURCE_CACHE_LOCK_MEMBER",
    "SourceCache",
    "atomic_write_csv",
    "build_compatibility_case_rows",
    "build_source_cache_lock",
    "durable_replace",
    "load_source_cache",
    "materialize_source_array",
    "read_csv",
    "validate_source_cache_lock",
    "write_support_scratch",
)
