"""Deterministic 256/class source streams for the fresh Stage-70 study."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion import load_routing_authorized_expert
from ...generation import (
    GeneratedBlock,
    GenerationLock,
    generate_source_block,
    source_generation_plan,
)
from ...generation.config import load_generation_lock_config
from ...generation.contracts import (
    COMMON_OUTPUT_DIM,
    EXPECTED_GENERATION_LOCK_HASH,
    SourceGenerationKey,
)
from ...generation.runner import read_generation_lock
from ...generation.validation import validate_generation_bundle
from ...protocol import ProtocolError
from .config import SOURCE_BLOCK_PER_CLASS, ResidualTopupFreshConfig
from .contracts import CENTERS, GENERATION_SEEDS, TRAINING_SEEDS
from .workstation import publish_validated_scratch_file


SOURCE_CACHE_SCHEMA = "midogpp_residual_topup_fresh_source_cache_v1"
SOURCE_BLOCK_SCHEMA = "midogpp_residual_topup_fresh_source_block_v1"
EXPECTED_SOURCE_BLOCK_COUNT = (
    len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
)
EXPECTED_EXPERT_TASK_COUNT = len(CENTERS) * len(TRAINING_SEEDS)
GENERATION_DEVICES = ("cuda:0", "cuda:1")


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
            "schema_version": SOURCE_BLOCK_SCHEMA,
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
class FreshSourceCache:
    root: Path
    generation_lock_hash: str
    bank_lock_hash: str
    records: tuple[SourceBlockRecord, ...]
    cache_hash: str

    def __post_init__(self) -> None:
        if (
            len(self.records) != EXPECTED_SOURCE_BLOCK_COUNT
            or len({record.key for record in self.records})
            != EXPECTED_SOURCE_BLOCK_COUNT
        ):
            raise ProtocolError("Fresh source-cache coverage drifted.")

    @cached_property
    def record_by_key(self) -> Mapping[tuple[str, int, int], SourceBlockRecord]:
        return MappingProxyType({record.key: record for record in self.records})

    def block(
        self, source_center: str, training_seed: int, generation_seed: int
    ) -> GeneratedBlock:
        key_tuple = (str(source_center), int(training_seed), int(generation_seed))
        record = self.record_by_key.get(key_tuple)
        if record is None:
            raise ProtocolError("Fresh source-cache key is unknown.")
        path = _safe_member(self.root, record.relative_path)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        _validate_array(
            array,
            rows_per_class=record.rows_per_class,
            feature_dim=record.feature_dim,
        )
        labels = np.concatenate(
            (
                np.zeros(record.rows_per_class, dtype=np.int64),
                np.ones(record.rows_per_class, dtype=np.int64),
            )
        )
        labels.setflags(write=False)
        generation_key = SourceGenerationKey(
            source_center=record.source_center,
            training_seed=record.training_seed,
            generation_seed=record.generation_seed,
            expert_lock_hash=record.expert_lock_hash,
            stream_id=record.stream_id,
            class_seed_by_label={},
            max_samples_per_class=record.rows_per_class,
            equal_union_prefix_per_class=128,
        )
        return GeneratedBlock(
            key=generation_key,
            embeddings=array,
            labels=labels,
            output_sha256=record.output_sha256,
        )


@dataclass(frozen=True)
class SourceExpertTask:
    source_center: str
    training_seed: int
    keys: tuple[SourceGenerationKey, ...]
    device: str


SourceTaskExecutor = Callable[
    [Sequence[SourceExpertTask], Path, Path, int], Sequence[Mapping[str, object]]
]


def load_validated_generation_lock(
    config: ResidualTopupFreshConfig,
) -> GenerationLock:
    """Validate both the routing-authorized bank and its GenerationLock."""

    generation_config_path = config.generation_lock_root / "config.resolved.yaml"
    lock_path = config.generation_lock_root / "manifests/generation_lock.json"
    if not generation_config_path.is_file() or not lock_path.is_file():
        raise ProtocolError("Fresh Stage-70 validated GenerationLock is absent.")
    generation_config = load_generation_lock_config(generation_config_path)
    if generation_config.bank_root.resolve() != config.expert_bank_root.resolve():
        raise ProtocolError("Fresh Stage-70 bank/GenerationLock roots disagree.")
    validate_generation_bundle(
        config.generation_lock_root,
        config=generation_config,
    )
    lock = read_generation_lock(lock_path)
    if (
        lock.generation_lock_hash != EXPECTED_GENERATION_LOCK_HASH
        or lock.generation_lock_hash != config.expected_generation_lock_hash
        or lock.bank_lock_hash != config.expected_bank_lock_hash
        or generation_config.classifier != config.classifier
    ):
        raise ProtocolError("Fresh Stage-70 GenerationLock identity drifted.")
    return lock


def materialize_source_cache(
    config: ResidualTopupFreshConfig,
    generation_lock: GenerationLock,
    *,
    root: Path,
    scratch_root: Path | None = None,
    executor: SourceTaskExecutor | None = None,
) -> FreshSourceCache:
    """Generate/resume all 81 streams with two persistent GPU workers."""

    root.mkdir(parents=True, exist_ok=True)
    keys = tuple(source_generation_plan(generation_lock))
    if (
        len(keys) != EXPECTED_SOURCE_BLOCK_COUNT
        or {
            (key.source_center, key.training_seed, key.generation_seed)
            for key in keys
        }
        != {
            (center, training_seed, generation_seed)
            for center in CENTERS
            for training_seed in TRAINING_SEEDS
            for generation_seed in GENERATION_SEEDS
        }
        or any(key.max_samples_per_class < SOURCE_BLOCK_PER_CLASS for key in keys)
    ):
        raise ProtocolError("Fresh source generation plan drifted.")

    records: dict[tuple[str, int, int], SourceBlockRecord] = {}
    pending_keys: list[SourceGenerationKey] = []
    for key in keys:
        record = _load_resumed_record(root, key)
        if record is None:
            pending_keys.append(key)
        else:
            records[record.key] = record

    tasks = _expert_tasks(pending_keys)
    if tasks:
        work_root = root if scratch_root is None else scratch_root.resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        active_executor = executor or _spawn_source_tasks
        raw_records = active_executor(
            tasks,
            config.expert_bank_root,
            work_root,
            SOURCE_BLOCK_PER_CLASS,
        )
        if len(raw_records) != len(pending_keys):
            raise ProtocolError("Fresh source worker result coverage drifted.")
        for raw in raw_records:
            record = _publish_worker_record(
                raw,
                canonical_root=root,
                scratch_root=scratch_root,
            )
            if record.key in records:
                raise ProtocolError("Fresh source worker duplicated a cache key.")
            records[record.key] = record

    ordered = tuple(
        records[(center, training_seed, generation_seed)]
        for center in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )
    unhashed = {
        "schema_version": SOURCE_CACHE_SCHEMA,
        "status": "COMPLETE",
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "bank_lock_hash": generation_lock.bank_lock_hash,
        "source_block_count": len(ordered),
        "expert_task_count": EXPECTED_EXPERT_TASK_COUNT,
        "generation_devices": list(GENERATION_DEVICES),
        "persistent_worker_count": 2,
        "one_expert_per_gpu_at_a_time": True,
        "rows_per_source_class": SOURCE_BLOCK_PER_CLASS,
        "dtype": "float32",
        "labels_persisted": False,
        "scratch_authoritative": False,
        "records": [record.to_payload() for record in ordered],
    }
    lock = {**unhashed, "source_cache_hash": stable_hash(unhashed)}
    _atomic_json(root / "source_cache.json", lock)
    return FreshSourceCache(
        root=root,
        generation_lock_hash=generation_lock.generation_lock_hash,
        bank_lock_hash=generation_lock.bank_lock_hash,
        records=ordered,
        cache_hash=str(lock["source_cache_hash"]),
    )


def load_source_cache(root: Path) -> FreshSourceCache:
    payload = _json(root / "source_cache.json")
    observed_hash = payload.get("source_cache_hash")
    unhashed = {
        key: value for key, value in payload.items() if key != "source_cache_hash"
    }
    if (
        observed_hash != stable_hash(unhashed)
        or payload.get("schema_version") != SOURCE_CACHE_SCHEMA
        or payload.get("status") != "COMPLETE"
        or payload.get("source_block_count") != EXPECTED_SOURCE_BLOCK_COUNT
        or payload.get("rows_per_source_class") != SOURCE_BLOCK_PER_CLASS
        or payload.get("labels_persisted") is not False
        or payload.get("scratch_authoritative") is not False
    ):
        raise ProtocolError("Fresh source-cache lock drifted.")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ProtocolError("Fresh source-cache inventory is absent.")
    records = tuple(_record_from_payload(raw) for raw in raw_records)
    cache = FreshSourceCache(
        root=root,
        generation_lock_hash=str(payload.get("generation_lock_hash", "")),
        bank_lock_hash=str(payload.get("bank_lock_hash", "")),
        records=records,
        cache_hash=str(observed_hash),
    )
    for record in cache.records:
        path = _safe_member(root, record.relative_path)
        if not path.is_file() or _sha256_file(path) != record.file_sha256:
            raise ProtocolError("Fresh source-cache member hash drifted.")
        _validate_array(
            np.load(path, mmap_mode="r", allow_pickle=False),
            rows_per_class=record.rows_per_class,
            feature_dim=record.feature_dim,
        )
    return cache


def _expert_tasks(keys: Sequence[SourceGenerationKey]) -> tuple[SourceExpertTask, ...]:
    grouped: dict[tuple[str, int], list[SourceGenerationKey]] = {}
    for key in keys:
        grouped.setdefault((key.source_center, key.training_seed), []).append(key)
    tasks: list[SourceExpertTask] = []
    for ordinal, pair in enumerate(
        (center, seed) for center in CENTERS for seed in TRAINING_SEEDS
    ):
        task_keys = grouped.get(pair)
        if not task_keys:
            continue
        ordered = tuple(sorted(task_keys, key=lambda key: key.generation_seed))
        tasks.append(
            SourceExpertTask(
                source_center=pair[0],
                training_seed=pair[1],
                keys=ordered,
                device=GENERATION_DEVICES[ordinal % len(GENERATION_DEVICES)],
            )
        )
    return tuple(tasks)


def _spawn_source_tasks(
    tasks: Sequence[SourceExpertTask],
    bank_root: Path,
    output_root: Path,
    per_class: int,
) -> Sequence[Mapping[str, object]]:
    context = mp.get_context("spawn")
    queues = [context.Queue(), context.Queue()]
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_source_worker_main,
            args=(queues[index], result_queue, str(bank_root), str(output_root), per_class),
            name=f"fresh-stage70-source-gpu-{index}",
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for task in tasks:
        device_index = GENERATION_DEVICES.index(task.device)
        queues[device_index].put(task)
    for queue in queues:
        queue.put(None)
    results: list[Mapping[str, object]] = []
    expected = sum(len(task.keys) for task in tasks)
    while len(results) < expected:
        payload = result_queue.get()
        if not isinstance(payload, Mapping):
            raise ProtocolError("Fresh source worker returned an invalid payload.")
        if payload.get("error"):
            for process in processes:
                process.terminate()
            raise ProtocolError(f"Fresh source worker failed: {payload['error']}.")
        results.append(payload)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise ProtocolError("Fresh source worker exited unsuccessfully.")
    return tuple(results)


def _source_worker_main(
    task_queue: object,
    result_queue: object,
    bank_root: str,
    output_root: str,
    per_class: int,
) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    while True:
        task = task_queue.get()  # type: ignore[attr-defined]
        if task is None:
            break
        try:
            assert isinstance(task, SourceExpertTask)
            torch.cuda.set_device(task.device)
            expert = load_routing_authorized_expert(
                bank_root,
                source_center=task.source_center,
                training_seed=task.training_seed,
                device=task.device,
            )
            for key in task.keys:
                block = generate_source_block(
                    expert,
                    key,
                    per_class=per_class,
                    device=task.device,
                )
                path = _worker_array_path(Path(output_root), key.stream_id)
                _atomic_save_npy(path, block.embeddings)
                result_queue.put(  # type: ignore[attr-defined]
                    {
                        "source_center": key.source_center,
                        "training_seed": key.training_seed,
                        "generation_seed": key.generation_seed,
                        "stream_id": key.stream_id,
                        "expert_lock_hash": key.expert_lock_hash,
                        "path": str(path),
                        "file_sha256": _sha256_file(path),
                        "output_sha256": block.output_sha256,
                        "rows_per_class": per_class,
                        "feature_dim": int(block.embeddings.shape[1]),
                    }
                )
            del expert
            torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - exercised on workstation
            result_queue.put({"error": f"{type(exc).__name__}: {exc}"})  # type: ignore[attr-defined]
            return


def _publish_worker_record(
    raw: Mapping[str, object],
    *,
    canonical_root: Path,
    scratch_root: Path | None,
) -> SourceBlockRecord:
    stream_id = str(raw.get("stream_id", ""))
    source_path = Path(str(raw.get("path", ""))).resolve()
    destination = _worker_array_path(canonical_root, stream_id)
    digest = str(raw.get("file_sha256", ""))
    if scratch_root is None:
        if source_path != destination.resolve() or _sha256_file(source_path) != digest:
            raise ProtocolError("Fresh source worker canonical output drifted.")
    else:
        publish_validated_scratch_file(
            source_path,
            destination,
            expected_sha256=digest,
            scratch_root=scratch_root,
        )
    record = SourceBlockRecord(
        source_center=str(raw.get("source_center", "")),
        training_seed=int(raw.get("training_seed", -1)),
        generation_seed=int(raw.get("generation_seed", -1)),
        stream_id=stream_id,
        expert_lock_hash=str(raw.get("expert_lock_hash", "")),
        relative_path=str(destination.relative_to(canonical_root)),
        file_sha256=digest,
        output_sha256=str(raw.get("output_sha256", "")),
        rows_per_class=int(raw.get("rows_per_class", -1)),
        feature_dim=int(raw.get("feature_dim", -1)),
    )
    _validate_record(record)
    _atomic_json(
        canonical_root / f"metadata/{record.stream_id}.json",
        record.to_payload(),
    )
    return record


def _load_resumed_record(
    root: Path, key: SourceGenerationKey
) -> SourceBlockRecord | None:
    metadata = root / f"metadata/{key.stream_id}.json"
    if not metadata.is_file():
        return None
    try:
        record = _record_from_payload(_json(metadata))
        if (
            record.key != (key.source_center, key.training_seed, key.generation_seed)
            or record.expert_lock_hash != key.expert_lock_hash
            or record.stream_id != key.stream_id
            or record.rows_per_class != SOURCE_BLOCK_PER_CLASS
            or record.feature_dim != COMMON_OUTPUT_DIM
        ):
            return None
        path = _safe_member(root, record.relative_path)
        if not path.is_file() or _sha256_file(path) != record.file_sha256:
            return None
        _validate_array(
            np.load(path, mmap_mode="r", allow_pickle=False),
            rows_per_class=record.rows_per_class,
            feature_dim=record.feature_dim,
        )
        return record
    except (ProtocolError, OSError, ValueError):
        return None


def _record_from_payload(raw: object) -> SourceBlockRecord:
    if not isinstance(raw, Mapping) or raw.get("schema_version") != SOURCE_BLOCK_SCHEMA:
        raise ProtocolError("Fresh source-cache record is malformed.")
    record = SourceBlockRecord(
        source_center=str(raw.get("source_center", "")),
        training_seed=int(raw.get("training_seed", -1)),
        generation_seed=int(raw.get("generation_seed", -1)),
        stream_id=str(raw.get("stream_id", "")),
        expert_lock_hash=str(raw.get("expert_lock_hash", "")),
        relative_path=str(raw.get("relative_path", "")),
        file_sha256=str(raw.get("file_sha256", "")),
        output_sha256=str(raw.get("output_sha256", "")),
        rows_per_class=int(raw.get("rows_per_class", -1)),
        feature_dim=int(raw.get("feature_dim", -1)),
    )
    _validate_record(record)
    return record


def _validate_record(record: SourceBlockRecord) -> None:
    if (
        record.source_center not in CENTERS
        or record.training_seed not in TRAINING_SEEDS
        or record.generation_seed not in GENERATION_SEEDS
        or not record.stream_id
        or not record.expert_lock_hash
        or record.rows_per_class != SOURCE_BLOCK_PER_CLASS
        or record.feature_dim != COMMON_OUTPUT_DIM
        or len(record.file_sha256) != 64
        or len(record.output_sha256) != 64
    ):
        raise ProtocolError("Fresh source-cache record identity drifted.")


def _worker_array_path(root: Path, stream_id: str) -> Path:
    return root / f"arrays/source_blocks/{stream_id}.npy"


def _validate_array(
    array: np.ndarray, *, rows_per_class: int, feature_dim: int
) -> None:
    if (
        array.dtype != np.float32
        or array.shape != (2 * rows_per_class, feature_dim)
        or not np.isfinite(array).all()
    ):
        raise ProtocolError("Fresh source-cache array geometry drifted.")


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(array, dtype=np.float32), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read fresh source-cache JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Fresh source-cache JSON must be a mapping.")
    return payload


def _safe_member(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ProtocolError("Fresh source-cache member escapes its root.")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "EXPECTED_EXPERT_TASK_COUNT",
    "EXPECTED_SOURCE_BLOCK_COUNT",
    "FreshSourceCache",
    "SOURCE_BLOCK_SCHEMA",
    "SOURCE_CACHE_SCHEMA",
    "SourceBlockRecord",
    "SourceExpertTask",
    "SourceTaskExecutor",
    "load_source_cache",
    "load_validated_generation_lock",
    "materialize_source_cache",
)
