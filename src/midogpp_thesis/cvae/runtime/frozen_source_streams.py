"""Neutral frozen-bank source-stream materialization.

Only frozen expert and GenerationLock primitives are imported.  No diagnostic,
routing, support-utility, or label-access module is reachable from this file.
"""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cached_property
import gc
import hashlib
from itertools import product
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

import numpy as np

from ...common.hashing import stable_hash
from ..expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ..expert_bank.uniform_b_v2_promotion.serialization import (
    load_routing_authorized_expert,
)
from ..generation.contracts import COMMON_OUTPUT_DIM, GenerationLock, SourceGenerationKey
from ..generation.generation import generate_source_block, source_generation_plan
from ..protocol import ProtocolError
from .artifact_io import atomic_copy, atomic_json, read_json, sha256_array, sha256_file


SOURCE_ROWS_PER_CLASS = 270
GENERATION_DEVICES = ("cuda:0", "cuda:1")
EXPECTED_TASK_COUNT = len(CENTERS) * len(TRAINING_SEEDS)
EXPECTED_STREAM_COUNT = EXPECTED_TASK_COUNT * len(GENERATION_SEEDS)

SOURCE_ARRAY_MEMBER = "arrays/frozen_source_streams.npy"
SOURCE_INDEX_MEMBER = "manifests/frozen_source_stream_index.json"
SOURCE_LOCK_MEMBER = "manifests/frozen_source_stream_lock.json"
CHECKPOINT_DIRECTORY = "checkpoints/frozen_source_streams"


class FrozenSourceConfig(Protocol):
    contract_hash: str
    expert_bank_root: Path
    runtime: Mapping[str, object]


@dataclass(frozen=True)
class FrozenSourceStreamRecord:
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
class FrozenSourceStreamCache:
    root: Path
    source_array_path: Path
    records: tuple[FrozenSourceStreamRecord, ...]
    lock_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        _validate_cache(self)
        object.__setattr__(self, "lock_payload", MappingProxyType(dict(self.lock_payload)))

    @cached_property
    def by_key(self) -> Mapping[tuple[str, int, int], FrozenSourceStreamRecord]:
        return MappingProxyType({record.key: record for record in self.records})

    @property
    def lock_hash(self) -> str:
        return str(self.lock_payload["source_stream_lock_hash"])

    def block(self, source: str, training_seed: int, generation_seed: int) -> np.ndarray:
        try:
            ordinal = self.by_key[(str(source), int(training_seed), int(generation_seed))].block_ordinal
        except KeyError as exc:
            raise ProtocolError("Frozen source stream is absent.") from exc
        values = np.load(self.source_array_path, mmap_mode="r", allow_pickle=False)
        return values[ordinal]


def materialize_frozen_source_streams(
    config: FrozenSourceConfig,
    generation_lock: GenerationLock,
    *,
    root: Path,
) -> FrozenSourceStreamCache:
    """Generate all 81 streams through two persistent one-process GPU pools."""

    _assert_runtime(config.runtime)
    array_path = root / SOURCE_ARRAY_MEMBER
    index_path = root / SOURCE_INDEX_MEMBER
    lock_path = root / SOURCE_LOCK_MEMBER
    if array_path.is_file() and index_path.is_file() and lock_path.is_file():
        return load_frozen_source_streams(root, expected_config_hash=config.contract_hash,
                                          expected_generation_lock_hash=generation_lock.generation_lock_hash)

    checkpoint_root = root / CHECKPOINT_DIRECTORY
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    tasks = _build_tasks(config, generation_lock, checkpoint_root)
    completed: dict[tuple[str, int], Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        path = Path(str(task["checkpoint_path"]))
        if path.is_file():
            completed[_task_key(task)] = _load_checkpoint(path, task=task)
        else:
            pending.append(task)
    pending_by_key = {_task_key(task): task for task in pending}
    for payload in _execute_generation_tasks(pending):
        key = (str(payload["source_center"]), int(payload["training_seed"]))
        task = pending_by_key[key]
        verified = _load_checkpoint(Path(str(task["checkpoint_path"])), task=task)
        if verified.get("checkpoint_hash") != payload.get("checkpoint_hash"):
            raise ProtocolError("Frozen source checkpoint return drifted.")
        completed[key] = verified
        print(f"[label-aware-oof] source jobs {len(completed)}/{len(tasks)}", flush=True)
    if len(completed) != EXPECTED_TASK_COUNT:
        raise ProtocolError("Frozen source checkpoint coverage is incomplete.")

    records = _materialize_array(array_path, tasks=tasks, completed=completed)
    index_unhashed = {
        "schema_version": "midogpp_frozen_source_stream_index_v1",
        "config_contract_hash": config.contract_hash,
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "records": [record.to_payload() for record in records],
        "stream_count": len(records),
        "labels_consumed": False,
        "target_embeddings_consumed": False,
    }
    index = {**index_unhashed, "source_stream_index_hash": stable_hash(index_unhashed)}
    atomic_json(index_path, index)
    lock_unhashed = {
        "schema_version": "midogpp_frozen_source_stream_lock_v1",
        "status": "COMPLETE_LABEL_FREE_FROZEN_SOURCE_STREAMS",
        "config_contract_hash": config.contract_hash,
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "source_array_sha256": sha256_file(array_path),
        "source_stream_index_sha256": sha256_file(index_path),
        "source_stream_index_hash": index["source_stream_index_hash"],
        "stream_count": len(records),
        "rows_per_class": SOURCE_ROWS_PER_CLASS,
        "expert_bank_updated": False,
        "source_experts_updated": False,
        "labels_consumed": False,
        "tf32_disabled": True,
        "amp_disabled": True,
        "float32_store": True,
    }
    lock = {**lock_unhashed, "source_stream_lock_hash": stable_hash(lock_unhashed)}
    atomic_json(lock_path, lock)
    cache = load_frozen_source_streams(
        root,
        expected_config_hash=config.contract_hash,
        expected_generation_lock_hash=generation_lock.generation_lock_hash,
    )
    shutil.rmtree(checkpoint_root, ignore_errors=True)
    return cache


def load_frozen_source_streams(
    root: Path,
    *,
    expected_config_hash: str | None = None,
    expected_generation_lock_hash: str | None = None,
) -> FrozenSourceStreamCache:
    array_path = root / SOURCE_ARRAY_MEMBER
    index = read_json(root / SOURCE_INDEX_MEMBER)
    lock = read_json(root / SOURCE_LOCK_MEMBER)
    raw_records = index.get("records")
    if not isinstance(raw_records, list):
        raise ProtocolError("Frozen source index records are absent.")
    records = tuple(
        FrozenSourceStreamRecord(
            block_ordinal=int(row["block_ordinal"]),
            source_center=str(row["source_center"]),
            training_seed=int(row["training_seed"]),
            generation_seed=int(row["generation_seed"]),
            stream_id=str(row["stream_id"]),
            expert_lock_hash=str(row["expert_lock_hash"]),
            rows_per_class=int(row["rows_per_class"]),
            output_sha256=str(row["output_sha256"]),
        )
        for row in raw_records
        if isinstance(row, Mapping)
    )
    cache = FrozenSourceStreamCache(root=root, source_array_path=array_path, records=records, lock_payload=lock)
    index_unhashed = {key: value for key, value in index.items() if key != "source_stream_index_hash"}
    lock_unhashed = {key: value for key, value in lock.items() if key != "source_stream_lock_hash"}
    if (
        len(records) != len(raw_records)
        or index.get("source_stream_index_hash") != stable_hash(index_unhashed)
        or lock.get("source_stream_lock_hash") != stable_hash(lock_unhashed)
        or lock.get("source_stream_index_sha256") != sha256_file(root / SOURCE_INDEX_MEMBER)
        or lock.get("source_array_sha256") != sha256_file(array_path)
        or lock.get("source_stream_index_hash") != index.get("source_stream_index_hash")
        or (expected_config_hash is not None and lock.get("config_contract_hash") != expected_config_hash)
        or (
            expected_generation_lock_hash is not None
            and lock.get("generation_lock_hash") != expected_generation_lock_hash
        )
    ):
        raise ProtocolError("Frozen source stream lock failed validation.")
    return cache


def stage_frozen_source_streams(
    cache: FrozenSourceStreamCache,
    *,
    scratch_root: Path,
    canonical_root: Path,
    local_directory: str = "source_cache",
) -> FrozenSourceStreamCache:
    canonical = Path(canonical_root).resolve()
    if cache.root.resolve() != canonical:
        raise ProtocolError("Frozen source staging received another canonical root.")
    destination = Path(scratch_root).resolve() / local_directory
    if destination == canonical:
        return cache
    destination.mkdir(parents=True, exist_ok=True)
    members = (SOURCE_ARRAY_MEMBER, SOURCE_INDEX_MEMBER, SOURCE_LOCK_MEMBER)
    if all((destination / member).is_file() for member in members):
        try:
            staged = load_frozen_source_streams(
                destination,
                expected_config_hash=str(cache.lock_payload["config_contract_hash"]),
                expected_generation_lock_hash=str(cache.lock_payload["generation_lock_hash"]),
            )
            if dict(staged.lock_payload) == dict(cache.lock_payload):
                return staged
        except ProtocolError:
            pass
    expected = {
        SOURCE_ARRAY_MEMBER: str(cache.lock_payload["source_array_sha256"]),
        SOURCE_INDEX_MEMBER: str(cache.lock_payload["source_stream_index_sha256"]),
        SOURCE_LOCK_MEMBER: sha256_file(canonical / SOURCE_LOCK_MEMBER),
    }
    for member in members:
        atomic_copy(canonical / member, destination / member, expected_sha256=expected[member])
    staged = load_frozen_source_streams(
        destination,
        expected_config_hash=str(cache.lock_payload["config_contract_hash"]),
        expected_generation_lock_hash=str(cache.lock_payload["generation_lock_hash"]),
    )
    if dict(staged.lock_payload) != dict(cache.lock_payload):
        raise ProtocolError("Staged frozen source lock differs from canonical.")
    return staged


def _assert_runtime(runtime: Mapping[str, object]) -> None:
    if (
        tuple(runtime.get("generation_devices", ())) != GENERATION_DEVICES
        or int(runtime.get("source_workers_per_device", -1)) != 1
        or int(runtime.get("generation_workers_per_device", -1)) != 1
        or runtime.get("persistent_source_workers") is not True
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("parent_cuda_context_forbidden") is not True
        or runtime.get("tf32_enabled") is not False
        or runtime.get("amp_enabled") is not False
        or runtime.get("generated_cache_format") != "float32_npy_memmap"
        or int(runtime.get("source_prefix_rows_per_class", -1))
        != SOURCE_ROWS_PER_CLASS
    ):
        raise ProtocolError("Frozen source generation requires two exact float32 GPU streams.")
    torch_module = sys.modules.get("torch")
    if (
        torch_module is not None
        and getattr(torch_module, "cuda", None) is not None
        and torch_module.cuda.is_initialized()
    ):
        raise ProtocolError("Frozen source parent process must remain CUDA-free.")


def _build_tasks(
    config: FrozenSourceConfig, generation_lock: GenerationLock, checkpoint_root: Path
) -> tuple[Mapping[str, object], ...]:
    keys = tuple(source_generation_plan(generation_lock))
    by_key = {(key.source_center, key.training_seed, key.generation_seed): key for key in keys}
    if set(by_key) != set(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS)):
        raise ProtocolError("GenerationLock source grid drifted.")
    tasks: list[Mapping[str, object]] = []
    for ordinal, (source, training_seed) in enumerate(product(CENTERS, TRAINING_SEEDS)):
        stem = f"source_{source}_train_{training_seed}"
        task = {
            "schema_version": "midogpp_frozen_source_stream_task_v1",
            "task_ordinal": ordinal,
            "source_center": source,
            "training_seed": training_seed,
            "generation_keys": tuple(by_key[(source, training_seed, seed)] for seed in GENERATION_SEEDS),
            "device": GENERATION_DEVICES[ordinal % len(GENERATION_DEVICES)],
            "expert_bank_root": str(config.expert_bank_root),
            "checkpoint_path": str(checkpoint_root / f"{stem}.json"),
            "array_path": str(checkpoint_root / f"{stem}.npy"),
            "config_contract_hash": config.contract_hash,
            "generation_lock_hash": generation_lock.generation_lock_hash,
            "labels_available": False,
            "amp_enabled": False,
            "tf32_enabled": False,
        }
        tasks.append(task)
    return tuple(tasks)


def _execute_generation_tasks(tasks: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    if not tasks:
        return ()
    context = mp.get_context("spawn")
    executors = [ProcessPoolExecutor(max_workers=1, mp_context=context) for _ in GENERATION_DEVICES]
    futures: dict[Future[dict[str, object]], Mapping[str, object]] = {}
    try:
        for task in tasks:
            index = GENERATION_DEVICES.index(str(task["device"]))
            futures[executors[index].submit(_generate_task, task)] = task
        return tuple(future.result() for future in as_completed(futures))
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def _generate_task(task: Mapping[str, object]) -> dict[str, object]:
    keys = tuple(task["generation_keys"])
    device = str(task["device"])
    if (
        not all(isinstance(key, SourceGenerationKey) for key in keys)
        or task.get("labels_available") is not False
        or task.get("amp_enabled") is not False
        or task.get("tf32_enabled") is not False
    ):
        raise ProtocolError("Frozen source worker input drifted.")
    if device.startswith("cuda"):
        import torch

        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.set_num_threads(1)
    expert = load_routing_authorized_expert(
        Path(str(task["expert_bank_root"])),
        source_center=str(task["source_center"]),
        training_seed=int(task["training_seed"]),
        device=device,
    )
    try:
        blocks = [
            generate_source_block(expert, key, per_class=SOURCE_ROWS_PER_CLASS, device=device)
            for key in keys
        ]
        values = np.ascontiguousarray(np.stack([block.embeddings for block in blocks]), dtype=np.float32)
        array_path = Path(str(task["array_path"]))
        array_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = array_path.with_suffix(array_path.suffix + f".{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        os.replace(temporary, array_path)
        records = [
            {
                "generation_seed": block.key.generation_seed,
                "stream_id": block.key.stream_id,
                "expert_lock_hash": block.key.expert_lock_hash,
                "output_sha256": block.output_sha256,
                "array_sha256": sha256_array(block.embeddings),
            }
            for block in blocks
        ]
        unhashed = {
            "schema_version": "midogpp_frozen_source_stream_checkpoint_v1",
            "status": "COMPLETE",
            "config_contract_hash": task["config_contract_hash"],
            "generation_lock_hash": task["generation_lock_hash"],
            "task_ordinal": task["task_ordinal"],
            "source_center": task["source_center"],
            "training_seed": task["training_seed"],
            "device": device,
            "array_path": str(array_path),
            "array_file_sha256": sha256_file(array_path),
            "records": records,
            "labels_consumed": False,
            "source_expert_updated": False,
            "tf32_disabled": True,
            "amp_disabled": True,
            "float32_outputs": True,
        }
        payload = {**unhashed, "checkpoint_hash": stable_hash(unhashed)}
        atomic_json(Path(str(task["checkpoint_path"])), payload)
        return payload
    finally:
        del expert
        gc.collect()
        try:
            import torch

            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


def _load_checkpoint(path: Path, *, task: Mapping[str, object]) -> Mapping[str, object]:
    payload = read_json(path)
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    array_path = Path(str(payload.get("array_path", "")))
    records = payload.get("records")
    if (
        payload.get("checkpoint_hash") != stable_hash(unhashed)
        or payload.get("schema_version") != "midogpp_frozen_source_stream_checkpoint_v1"
        or payload.get("status") != "COMPLETE"
        or payload.get("config_contract_hash") != task["config_contract_hash"]
        or payload.get("generation_lock_hash") != task["generation_lock_hash"]
        or payload.get("task_ordinal") != task["task_ordinal"]
        or payload.get("source_center") != task["source_center"]
        or payload.get("training_seed") != task["training_seed"]
        or payload.get("device") != task["device"]
        or array_path != Path(str(task["array_path"]))
        or not array_path.is_file()
        or payload.get("array_file_sha256") != sha256_file(array_path)
        or not isinstance(records, list)
        or len(records) != len(GENERATION_SEEDS)
        or payload.get("labels_consumed") is not False
        or payload.get("source_expert_updated") is not False
        or payload.get("tf32_disabled") is not True
        or payload.get("amp_disabled") is not True
        or payload.get("float32_outputs") is not True
    ):
        raise ProtocolError("Frozen source checkpoint failed validation.")
    values = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if values.shape != (len(GENERATION_SEEDS), 2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM) or values.dtype != np.float32:
        raise ProtocolError("Frozen source checkpoint array geometry drifted.")
    for ordinal, (record, key) in enumerate(zip(records, task["generation_keys"], strict=True)):
        if (
            not isinstance(record, Mapping)
            or int(record.get("generation_seed", -1)) != key.generation_seed
            or record.get("stream_id") != key.stream_id
            or record.get("expert_lock_hash") != key.expert_lock_hash
            or record.get("array_sha256") != sha256_array(values[ordinal])
            or record.get("output_sha256")
            != _array_bundle_sha256(values[ordinal])
        ):
            raise ProtocolError("Frozen source checkpoint record drifted.")
    return payload


def _materialize_array(
    path: Path,
    *,
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[tuple[str, int], Mapping[str, object]],
) -> tuple[FrozenSourceStreamRecord, ...]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    values = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(EXPECTED_STREAM_COUNT, 2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM),
    )
    records: list[FrozenSourceStreamRecord] = []
    cursor = 0
    for task in tasks:
        result = completed[_task_key(task)]
        task_values = np.load(Path(str(result["array_path"])), mmap_mode="r", allow_pickle=False)
        for seed_ordinal, raw in enumerate(result["records"]):
            values[cursor] = task_values[seed_ordinal]
            records.append(
                FrozenSourceStreamRecord(
                    block_ordinal=cursor,
                    source_center=str(task["source_center"]),
                    training_seed=int(task["training_seed"]),
                    generation_seed=int(raw["generation_seed"]),
                    stream_id=str(raw["stream_id"]),
                    expert_lock_hash=str(raw["expert_lock_hash"]),
                    rows_per_class=SOURCE_ROWS_PER_CLASS,
                    output_sha256=str(raw["output_sha256"]),
                )
            )
            cursor += 1
    values.flush()
    del values
    os.replace(temporary, path)
    if cursor != EXPECTED_STREAM_COUNT:
        raise ProtocolError("Frozen source stream materialization coverage drifted.")
    return tuple(records)


def _validate_cache(cache: FrozenSourceStreamCache) -> None:
    expected_keys = tuple(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS))
    observed = tuple(record.key for record in cache.records)
    if (
        observed != expected_keys
        or [record.block_ordinal for record in cache.records] != list(range(EXPECTED_STREAM_COUNT))
        or any(record.rows_per_class != SOURCE_ROWS_PER_CLASS for record in cache.records)
        or cache.lock_payload.get("status") != "COMPLETE_LABEL_FREE_FROZEN_SOURCE_STREAMS"
        or cache.lock_payload.get("stream_count") != EXPECTED_STREAM_COUNT
        or cache.lock_payload.get("labels_consumed") is not False
        or cache.lock_payload.get("source_experts_updated") is not False
    ):
        raise ProtocolError("Frozen source stream inventory drifted.")
    values = np.load(cache.source_array_path, mmap_mode="r", allow_pickle=False)
    if values.shape != (EXPECTED_STREAM_COUNT, 2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM) or values.dtype != np.float32:
        raise ProtocolError("Frozen source stream array drifted.")
    for record in cache.records:
        if record.output_sha256 != _array_bundle_sha256(values[record.block_ordinal]):
            raise ProtocolError("Frozen source stream semantic output hash drifted.")


def _task_key(task: Mapping[str, object]) -> tuple[str, int]:
    return str(task["source_center"]), int(task["training_seed"])


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
    if values.shape != (2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM) or values.dtype != np.float32:
        raise ProtocolError("Frozen source block geometry drifted.")
    return _array_bundle_sha256(values)


__all__ = (
    "CHECKPOINT_DIRECTORY",
    "EXPECTED_STREAM_COUNT",
    "FrozenSourceStreamCache",
    "FrozenSourceStreamRecord",
    "SOURCE_ARRAY_MEMBER",
    "SOURCE_INDEX_MEMBER",
    "SOURCE_LOCK_MEMBER",
    "SOURCE_ROWS_PER_CLASS",
    "load_frozen_source_streams",
    "materialize_frozen_source_streams",
    "source_block_sha256",
    "stage_frozen_source_streams",
)
