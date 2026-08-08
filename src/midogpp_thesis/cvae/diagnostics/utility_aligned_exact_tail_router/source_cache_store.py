"""Atomic persistence and consolidation for the independent source cache."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from .source_cache_contracts import (
    COMPONENT_ARRAY_MEMBER,
    COMPONENT_INDEX_COLUMNS,
    COMPONENT_INDEX_MEMBER,
    EXPECTED_SOURCE_STREAM_COUNT,
    SOURCE_ARRAY_MEMBER,
    SOURCE_CACHE_LOCK_MEMBER,
    SOURCE_INDEX_COLUMNS,
    SOURCE_INDEX_MEMBER,
    SOURCE_ROWS_PER_CLASS,
    LabelFreeComponentRecord,
    SourceBlockRecord,
    SourceCache,
)


def materialize_source_products(
    root: Path,
    *,
    completed: Mapping[tuple[str, int], Mapping[str, object]],
    key_map: Mapping[tuple[str, int, int], object],
    support_scratch_hash: str,
    support_row_count: int,
) -> SourceCache:
    source_records = _materialize_source_array(
        root / SOURCE_ARRAY_MEMBER, completed=completed, key_map=key_map
    )
    component_records = _materialize_component_array(
        root / COMPONENT_ARRAY_MEMBER,
        completed=completed,
        support_row_count=support_row_count,
    )
    atomic_write_csv_rows(
        root / SOURCE_INDEX_MEMBER,
        [record.to_row() for record in source_records],
        columns=SOURCE_INDEX_COLUMNS,
    )
    atomic_write_csv_rows(
        root / COMPONENT_INDEX_MEMBER,
        [record.to_row() for record in component_records],
        columns=COMPONENT_INDEX_COLUMNS,
    )
    return SourceCache(
        root=root,
        source_array_path=root / SOURCE_ARRAY_MEMBER,
        component_array_path=root / COMPONENT_ARRAY_MEMBER,
        source_records=tuple(source_records),
        component_records=tuple(component_records),
        support_scratch_hash=str(support_scratch_hash),
    )


def load_source_cache(root: Path) -> SourceCache:
    lock = read_json(root / SOURCE_CACHE_LOCK_MEMBER)
    return SourceCache(
        root=root,
        source_array_path=root / SOURCE_ARRAY_MEMBER,
        component_array_path=root / COMPONENT_ARRAY_MEMBER,
        source_records=tuple(
            _source_record(row) for row in read_csv_rows(root / SOURCE_INDEX_MEMBER)
        ),
        component_records=tuple(
            _component_record(row)
            for row in read_csv_rows(root / COMPONENT_INDEX_MEMBER)
        ),
        support_scratch_hash=str(lock.get("support_scratch_hash", "")),
    )


def _materialize_source_array(
    path: Path,
    *,
    completed: Mapping[tuple[str, int], Mapping[str, object]],
    key_map: Mapping[tuple[str, int, int], object],
) -> list[SourceBlockRecord]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    target = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(
            EXPECTED_SOURCE_STREAM_COUNT,
            2 * SOURCE_ROWS_PER_CLASS,
            COMMON_OUTPUT_DIM,
        ),
    )
    records: list[SourceBlockRecord] = []
    ordinal = 0
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            payload = completed[(source, training_seed)]
            values = np.load(
                Path(str(payload["source_array_path"])),
                mmap_mode="r",
                allow_pickle=False,
            )
            raw_records = payload.get("source_records")
            if not isinstance(raw_records, list):
                raise ProtocolError("Stage-90 source records are absent.")
            for seed_index, generation_seed in enumerate(GENERATION_SEEDS):
                key = key_map[(source, training_seed, generation_seed)]
                raw = raw_records[seed_index]
                target[ordinal] = values[seed_index]
                records.append(
                    SourceBlockRecord(
                        block_ordinal=ordinal,
                        source_center=source,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        stream_id=str(getattr(key, "stream_id")),
                        expert_lock_hash=str(getattr(key, "expert_lock_hash")),
                        rows_per_class=SOURCE_ROWS_PER_CLASS,
                        row_count=2 * SOURCE_ROWS_PER_CLASS,
                        feature_dim=COMMON_OUTPUT_DIM,
                        output_sha256=str(raw["output_sha256"]),
                    )
                )
                ordinal += 1
    target.flush()
    del target
    os.replace(temporary, path)
    return records


def _materialize_component_array(
    path: Path,
    *,
    completed: Mapping[tuple[str, int], Mapping[str, object]],
    support_row_count: int,
) -> list[LabelFreeComponentRecord]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    target = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(CENTERS) * len(TRAINING_SEEDS), 4, int(support_row_count)),
    )
    records: list[LabelFreeComponentRecord] = []
    task_ordinal = 0
    component_ordinal = 0
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            payload = completed[(source, training_seed)]
            values = np.load(
                Path(str(payload["component_array_path"])),
                mmap_mode="r",
                allow_pickle=False,
            )
            target[task_ordinal] = values
            raw_records = payload.get("component_records")
            if not isinstance(raw_records, list):
                raise ProtocolError("Stage-90 component records are absent.")
            raw_by_query = {str(row["query_center"]): row for row in raw_records}
            for query in CENTERS:
                if query == source:
                    continue
                raw = raw_by_query[query]
                records.append(
                    LabelFreeComponentRecord(
                        component_ordinal=component_ordinal,
                        source_center=source,
                        training_seed=training_seed,
                        query_center=query,
                        support_start=int(raw["support_start"]),
                        support_stop=int(raw["support_stop"]),
                        support_row_count=int(raw["support_row_count"]),
                        support_case_count=int(raw["support_case_count"]),
                        support_partition_hash=str(raw["support_partition_hash"]),
                        case_equal_energy=float(raw["case_equal_energy"]),
                        linear_kernel_mmd2_by_generation_seed={
                            int(key): float(value)
                            for key, value in dict(
                                raw["linear_kernel_mmd2_by_generation_seed"]
                            ).items()
                        },
                    )
                )
                component_ordinal += 1
            task_ordinal += 1
    target.flush()
    del target
    os.replace(temporary, path)
    return records


def _source_record(row: Mapping[str, object]) -> SourceBlockRecord:
    if set(row) != set(SOURCE_INDEX_COLUMNS):
        raise ProtocolError("Stage-90 source index schema drifted.")
    return SourceBlockRecord(
        block_ordinal=int(row["block_ordinal"]),
        source_center=str(row["source_center"]),
        training_seed=int(row["training_seed"]),
        generation_seed=int(row["generation_seed"]),
        stream_id=str(row["stream_id"]),
        expert_lock_hash=str(row["expert_lock_hash"]),
        rows_per_class=int(row["rows_per_class"]),
        row_count=int(row["row_count"]),
        feature_dim=int(row["feature_dim"]),
        output_sha256=str(row["output_sha256"]),
    )


def _component_record(row: Mapping[str, object]) -> LabelFreeComponentRecord:
    if set(row) != set(COMPONENT_INDEX_COLUMNS):
        raise ProtocolError("Stage-90 component index schema drifted.")
    try:
        mmd = json.loads(str(row["linear_kernel_mmd2_by_generation_seed_json"]))
    except json.JSONDecodeError as exc:
        raise ProtocolError("Stage-90 component MMD JSON drifted.") from exc
    if (
        _truthy(row["labels_consumed"])
        or _truthy(row["evaluation_embeddings_consumed"])
        or _truthy(row["exact_nelbo_claimed"])
    ):
        raise ProtocolError("Stage-90 component index crossed its claim boundary.")
    return LabelFreeComponentRecord(
        component_ordinal=int(row["component_ordinal"]),
        source_center=str(row["source_center"]),
        training_seed=int(row["training_seed"]),
        query_center=str(row["query_center"]),
        support_start=int(row["support_start"]),
        support_stop=int(row["support_stop"]),
        support_row_count=int(row["support_row_count"]),
        support_case_count=int(row["support_case_count"]),
        support_partition_hash=str(row["support_partition_hash"]),
        case_equal_energy=float(row["case_equal_energy"]),
        linear_kernel_mmd2_by_generation_seed={
            int(key): float(value) for key, value in dict(mmd).items()
        },
    )


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(values))
        handle.flush()
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_csv_rows(
    path: Path, rows: Sequence[Mapping[str, object]], *, columns: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            if set(row) != set(columns):
                raise ProtocolError("Stage-90 CSV row schema drifted.")
            writer.writerow(row)
    os.replace(temporary, path)


def read_csv_rows(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))
    except OSError as exc:
        raise ProtocolError(f"Cannot read Stage-90 cache table: {path}.") from exc


def read_json(path: Path) -> Mapping[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read Stage-90 JSON: {path}.") from exc
    if not isinstance(raw, Mapping):
        raise ProtocolError("Stage-90 JSON payload must be an object.")
    return raw


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


__all__ = (
    "atomic_save_npy",
    "atomic_write_csv_rows",
    "atomic_write_json",
    "load_source_cache",
    "materialize_source_products",
    "read_csv_rows",
    "read_json",
    "sha256_array",
    "sha256_file",
)
