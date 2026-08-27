"""Label-free SCEPTRE source-stream materialization from the frozen bank.

This module has one scientific responsibility: materialize the complete
GenerationLock source grid at the full 1,024 rows per class.  It never opens a
target cache, manifest, or outcome surface.  Production work is dispatched to
two persistent spawn pools with one worker bound to each physical GPU.  The
parent process only transports paths and immutable identities and must remain
CUDA-context free.

The small-geometry seam exists solely for focused tests.  Production callers
that omit :class:`SourceRuntimeTestMode` are always checked against the exact
81 x 2,048 x 3,840 geometry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
from typing import Protocol

import numpy as np

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.generation.contracts import (
    COMMON_OUTPUT_DIM,
    SOURCE_BUDGET_PER_CLASS,
    TOTAL_PER_CLASS,
    GenerationLock,
    SourceGenerationKey,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import (
    atomic_json,
    read_json,
    sha256_array,
    sha256_file,
)


GPU_DEVICES = ("cuda:0", "cuda:1")
SOURCE_ARRAY_MEMBER = "arrays/sceptre_source_streams.npy"
SOURCE_INDEX_MEMBER = "manifests/sceptre_source_stream_index.json"
SOURCE_RECEIPT_MEMBER = "manifests/sceptre_source_stream_receipt.json"
CHECKPOINT_DIRECTORY = "checkpoints/sceptre_source_streams"


class SourceRuntimeConfig(Protocol):
    """Minimum configuration surface accepted by the physical source phase."""

    expert_bank_root: Path
    runtime: Mapping[str, object]


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
            raise ProtocolError("SCEPTRE source geometry is malformed.")

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
            raise ProtocolError("SCEPTRE source test generator is not callable.")


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
            raise ProtocolError("SCEPTRE source-stream inventory drifted.")
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    @cached_property
    def by_key(self) -> Mapping[tuple[str, int, int], SourceStreamRecord]:
        return MappingProxyType({record.key: record for record in self.records})

    @property
    def receipt_hash(self) -> str:
        return str(self.receipt["receipt_sha256"])

    def block(
        self, source_center: str, training_seed: int, generation_seed: int
    ) -> np.ndarray:
        try:
            record = self.by_key[
                (str(source_center), int(training_seed), int(generation_seed))
            ]
        except KeyError as exc:
            raise ProtocolError("SCEPTRE source stream is absent.") from exc
        values = np.load(self.array_path, mmap_mode="r", allow_pickle=False)
        block = values[record.block_ordinal]
        if block.flags.writeable:
            raise ProtocolError("SCEPTRE source memmap unexpectedly became writable.")
        return block


def materialize_source_streams(
    config: SourceRuntimeConfig,
    generation_lock: GenerationLock,
    *,
    root: Path,
    test_mode: SourceRuntimeTestMode | None = None,
) -> SourceStreamStore:
    """Materialize every frozen source stream and return a read-only store.

    In production this submits exactly 27 jobs to two persistent one-process
    spawn pools.  Each job loads one source/training expert and emits all three
    generation seeds, so an expert is loaded only once per physical task.
    """

    geometry = _geometry(test_mode)
    config_hash = _config_hash(config)
    destination = Path(root)
    _assert_owned_root(destination)
    _assert_parent_cuda_free()
    if test_mode is None:
        _assert_production_runtime(config.runtime)
    keys = _generation_keys(generation_lock, test_mode=test_mode)
    _validate_generation_grid(keys, generation_lock, geometry=geometry, test_mode=test_mode)
    _assert_parent_cuda_free()

    final_paths = _final_paths(destination)
    present = tuple(path.is_file() for path in final_paths)
    if any(path.is_symlink() for path in final_paths):
        raise ProtocolError("SCEPTRE source final store contains a symlink.")
    if all(present):
        return load_source_streams(
            destination,
            expected_config_hash=config_hash,
            expected_generation_lock_hash=generation_lock.generation_lock_hash,
            test_mode=test_mode,
        )
    if any(present):
        raise ProtocolError("SCEPTRE source final store is an unsafe partial state.")

    checkpoint_root = destination / CHECKPOINT_DIRECTORY
    _validate_checkpoint_directory(checkpoint_root, geometry)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    tasks = _build_tasks(
        config,
        generation_lock,
        keys=keys,
        geometry=geometry,
        checkpoint_root=checkpoint_root,
    )
    completed: dict[tuple[str, int], Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        checkpoint = _load_checkpoint_if_complete(task, geometry=geometry)
        if checkpoint is None:
            pending.append(task)
        else:
            completed[_task_key(task)] = checkpoint

    if pending:
        if test_mode is None:
            results = _execute_gpu_tasks(pending)
        else:
            results = tuple(
                _execute_injected_task(
                    task,
                    geometry=geometry,
                    generate_block=test_mode.generate_block,
                )
                for task in pending
            )
        _assert_parent_cuda_free()
        for result in results:
            key = (str(result["source_center"]), int(result["training_seed"]))
            task = next(task for task in pending if _task_key(task) == key)
            loaded = _load_checkpoint_if_complete(task, geometry=geometry)
            if loaded is None or loaded.get("checkpoint_sha256") != result.get(
                "checkpoint_sha256"
            ):
                raise ProtocolError("SCEPTRE source worker checkpoint return drifted.")
            completed[key] = loaded

    if len(completed) != geometry.task_count:
        raise ProtocolError("SCEPTRE source checkpoint coverage is incomplete.")
    records = _publish_source_array(
        destination / SOURCE_ARRAY_MEMBER,
        tasks=tasks,
        completed=completed,
        geometry=geometry,
    )
    index_unhashed = {
        "schema_version": "midogpp_sceptre_v2_source_stream_index_v1",
        "config_hash": config_hash,
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "geometry": geometry.to_payload(),
        "records": [record.to_payload() for record in records],
        "record_count": len(records),
        "source_streams_only": True,
        "target_cache_opened": False,
        "manifest_opened": False,
        "outcomes_available": False,
        "seed_selection_performed": False,
    }
    index = {**index_unhashed, "index_sha256": _canonical_sha256(index_unhashed)}
    _persist_exact_json(destination / SOURCE_INDEX_MEMBER, index)
    array_path = destination / SOURCE_ARRAY_MEMBER
    index_path = destination / SOURCE_INDEX_MEMBER
    receipt_unhashed = {
        "schema_version": "midogpp_sceptre_v2_source_stream_receipt_v1",
        "status": "COMPLETE_LABEL_FREE_FULL_SOURCE_STREAMS",
        "config_hash": config_hash,
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "geometry": geometry.to_payload(),
        "source_array_sha256": sha256_file(array_path),
        "source_index_file_sha256": sha256_file(index_path),
        "source_index_sha256": index["index_sha256"],
        "record_count": len(records),
        "dtype": "float32",
        "npy_memmap_mode": "read_only",
        "two_persistent_gpu_workers": test_mode is None,
        "gpu_devices": list(GPU_DEVICES) if test_mode is None else [],
        "parent_cuda_context_created": False,
        "target_cache_opened": False,
        "manifest_opened": False,
        "outcomes_available": False,
        "expert_bank_updated": False,
        "seed_selection_performed": False,
        "synthetic_test_mode": test_mode is not None,
    }
    receipt = {
        **receipt_unhashed,
        "receipt_sha256": _canonical_sha256(receipt_unhashed),
    }
    _persist_exact_json(destination / SOURCE_RECEIPT_MEMBER, receipt)
    array_path.chmod(0o444)
    store = load_source_streams(
        destination,
        expected_config_hash=config_hash,
        expected_generation_lock_hash=generation_lock.generation_lock_hash,
        test_mode=test_mode,
    )
    _validate_checkpoint_directory(checkpoint_root, geometry)
    shutil.rmtree(checkpoint_root)
    return store


def load_source_streams(
    root: Path,
    *,
    expected_config_hash: str | None = None,
    expected_generation_lock_hash: str | None = None,
    test_mode: SourceRuntimeTestMode | None = None,
) -> SourceStreamStore:
    """Load and fully validate a completed source-stream store."""

    geometry = _geometry(test_mode)
    destination = Path(root)
    array_path, index_path, receipt_path = _final_paths(destination)
    if any(path.is_symlink() or not path.is_file() for path in (array_path, index_path, receipt_path)):
        raise ProtocolError("SCEPTRE source final store is absent or unsafe.")
    index = read_json(index_path)
    receipt = read_json(receipt_path)
    raw_records = index.get("records")
    if not isinstance(raw_records, list):
        raise ProtocolError("SCEPTRE source index records are absent.")
    try:
        records = tuple(
            SourceStreamRecord(
                block_ordinal=int(raw["block_ordinal"]),
                source_center=str(raw["source_center"]),
                training_seed=int(raw["training_seed"]),
                generation_seed=int(raw["generation_seed"]),
                stream_id=str(raw["stream_id"]),
                expert_lock_hash=str(raw["expert_lock_hash"]),
                rows_per_class=int(raw["rows_per_class"]),
                feature_dim=int(raw["feature_dim"]),
                output_sha256=str(raw["output_sha256"]),
                array_sha256=str(raw["array_sha256"]),
            )
            for raw in raw_records
            if isinstance(raw, Mapping)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("SCEPTRE source index record is malformed.") from exc
    index_unhashed = {key: value for key, value in index.items() if key != "index_sha256"}
    receipt_unhashed = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if (
        len(records) != len(raw_records)
        or index.get("schema_version")
        != "midogpp_sceptre_v2_source_stream_index_v1"
        or receipt.get("schema_version")
        != "midogpp_sceptre_v2_source_stream_receipt_v1"
        or receipt.get("status") != "COMPLETE_LABEL_FREE_FULL_SOURCE_STREAMS"
        or index.get("geometry") != geometry.to_payload()
        or receipt.get("geometry") != geometry.to_payload()
        or index.get("index_sha256") != _canonical_sha256(index_unhashed)
        or receipt.get("receipt_sha256") != _canonical_sha256(receipt_unhashed)
        or receipt.get("source_array_sha256") != sha256_file(array_path)
        or receipt.get("source_index_file_sha256") != sha256_file(index_path)
        or receipt.get("source_index_sha256") != index.get("index_sha256")
        or receipt.get("record_count") != geometry.stream_count
        or index.get("record_count") != geometry.stream_count
        or receipt.get("dtype") != "float32"
        or receipt.get("npy_memmap_mode") != "read_only"
        or receipt.get("target_cache_opened") is not False
        or receipt.get("manifest_opened") is not False
        or receipt.get("outcomes_available") is not False
        or receipt.get("expert_bank_updated") is not False
        or receipt.get("seed_selection_performed") is not False
        or receipt.get("synthetic_test_mode") is not (test_mode is not None)
        or (
            expected_config_hash is not None
            and receipt.get("config_hash") != expected_config_hash
        )
        or (
            expected_generation_lock_hash is not None
            and receipt.get("generation_lock_hash")
            != expected_generation_lock_hash
        )
    ):
        raise ProtocolError("SCEPTRE source receipt failed validation.")
    values = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if (
        values.shape != geometry.array_shape
        or values.dtype != np.float32
        or values.flags.writeable
        or not np.isfinite(values).all()
    ):
        raise ProtocolError("SCEPTRE source array geometry or values drifted.")
    store = SourceStreamStore(
        root=destination,
        array_path=array_path,
        index_path=index_path,
        receipt_path=receipt_path,
        geometry=geometry,
        records=records,
        receipt=receipt,
    )
    for record in records:
        block = values[record.block_ordinal]
        if (
            sha256_array(block) != record.array_sha256
            or _block_bundle_sha256(block, geometry.rows_per_class)
            != record.output_sha256
        ):
            raise ProtocolError("SCEPTRE source block bytes drifted.")
    return store


def _geometry(test_mode: SourceRuntimeTestMode | None) -> SourceGeometry:
    return PRODUCTION_SOURCE_GEOMETRY if test_mode is None else test_mode.geometry


def _config_hash(config: object) -> str:
    value = getattr(config, "config_hash", getattr(config, "contract_hash", ""))
    text = str(value)
    if not text:
        raise ProtocolError("SCEPTRE source config hash is absent.")
    return text


def _assert_owned_root(root: Path) -> None:
    if root.is_symlink():
        raise ProtocolError("SCEPTRE source root is a symlink.")
    if root.exists() and not root.is_dir():
        raise ProtocolError("SCEPTRE source root is not a directory.")
    root.mkdir(parents=True, exist_ok=True)


def _final_paths(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / SOURCE_ARRAY_MEMBER,
        root / SOURCE_INDEX_MEMBER,
        root / SOURCE_RECEIPT_MEMBER,
    )


def _assert_parent_cuda_free() -> None:
    torch_module = sys.modules.get("torch")
    cuda = getattr(torch_module, "cuda", None) if torch_module is not None else None
    if cuda is not None and bool(cuda.is_initialized()):
        raise ProtocolError("SCEPTRE source parent process must remain CUDA-free.")


def _assert_production_runtime(runtime: Mapping[str, object]) -> None:
    if (
        tuple(runtime.get("gpu_devices", ())) != GPU_DEVICES
        or int(runtime.get("persistent_gpu_generation_workers", -1)) != 2
        or runtime.get("one_persistent_worker_per_physical_gpu") is not True
        or int(runtime.get("generated_source_family_streams", -1)) != 81
        or int(runtime.get("full_source_rows_per_class", -1)) != TOTAL_PER_CLASS
        or runtime.get("prediction_store_dtype") != "float32"
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("top_level_spawn_pool_only") is not True
        or runtime.get("nested_pools_allowed") is not False
    ):
        raise ProtocolError("SCEPTRE source workstation topology drifted.")


def _generation_keys(
    generation_lock: GenerationLock,
    *,
    test_mode: SourceRuntimeTestMode | None,
) -> tuple[object, ...]:
    if test_mode is not None:
        return tuple(test_mode.generation_keys)
    from midogpp_thesis.cvae.generation.generation import source_generation_plan

    return tuple(source_generation_plan(generation_lock))


def _validate_generation_grid(
    keys: Sequence[object],
    generation_lock: GenerationLock,
    *,
    geometry: SourceGeometry,
    test_mode: SourceRuntimeTestMode | None,
) -> None:
    observed: dict[tuple[str, int, int], object] = {}
    for key in keys:
        try:
            identity = (
                str(getattr(key, "source_center")),
                int(getattr(key, "training_seed")),
                int(getattr(key, "generation_seed")),
            )
            stream_id = str(getattr(key, "stream_id"))
            expert_hash = str(getattr(key, "expert_lock_hash"))
        except (TypeError, ValueError) as exc:
            raise ProtocolError("SCEPTRE source generation key is malformed.") from exc
        if not stream_id or not expert_hash or identity in observed:
            raise ProtocolError("SCEPTRE source generation key identity drifted.")
        if test_mode is None and (
            not isinstance(key, SourceGenerationKey)
            or int(key.max_samples_per_class) != TOTAL_PER_CLASS
            or int(key.equal_union_prefix_per_class) != SOURCE_BUDGET_PER_CLASS
        ):
            raise ProtocolError("SCEPTRE GenerationLock source budget drifted.")
        observed[identity] = key
    expected = set(
        product(
            geometry.centers,
            geometry.training_seeds,
            geometry.generation_seeds,
        )
    )
    if set(observed) != expected or len(keys) != geometry.stream_count:
        raise ProtocolError("SCEPTRE GenerationLock source grid drifted.")
    if not str(getattr(generation_lock, "generation_lock_hash", "")):
        raise ProtocolError("SCEPTRE GenerationLock hash is absent.")


def _build_tasks(
    config: SourceRuntimeConfig,
    generation_lock: GenerationLock,
    *,
    keys: Sequence[object],
    geometry: SourceGeometry,
    checkpoint_root: Path,
) -> tuple[Mapping[str, object], ...]:
    by_key = {
        (
            str(getattr(key, "source_center")),
            int(getattr(key, "training_seed")),
            int(getattr(key, "generation_seed")),
        ): key
        for key in keys
    }
    tasks: list[Mapping[str, object]] = []
    for ordinal, (source, training_seed) in enumerate(
        product(geometry.centers, geometry.training_seeds)
    ):
        stem = f"source_{source}_train_{training_seed}"
        generation_keys = tuple(
            by_key[(source, training_seed, seed)]
            for seed in geometry.generation_seeds
        )
        task_identity = {
            "schema_version": "midogpp_sceptre_v2_source_task_v1",
            "task_ordinal": ordinal,
            "source_center": source,
            "training_seed": training_seed,
            "generation_seeds": list(geometry.generation_seeds),
            "stream_ids": [str(getattr(key, "stream_id")) for key in generation_keys],
            "expert_lock_hashes": [
                str(getattr(key, "expert_lock_hash")) for key in generation_keys
            ],
            "device": GPU_DEVICES[ordinal % 2],
            "config_hash": _config_hash(config),
            "generation_lock_hash": generation_lock.generation_lock_hash,
            "geometry": geometry.to_payload(),
            "expert_bank_root": str(Path(config.expert_bank_root).resolve()),
            "target_cache_available": False,
            "manifest_available": False,
            "outcomes_available": False,
            "amp_enabled": False,
            "tf32_enabled": False,
        }
        tasks.append(
            {
                **task_identity,
                "task_sha256": _canonical_sha256(task_identity),
                "generation_keys": generation_keys,
                "checkpoint_array_path": str(checkpoint_root / f"{stem}.npy"),
                "checkpoint_json_path": str(checkpoint_root / f"{stem}.json"),
            }
        )
    if len(tasks) != geometry.task_count:
        raise ProtocolError("SCEPTRE source task coverage drifted.")
    return tuple(tasks)


def _task_identity(task: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in task.items()
        if key
        not in {
            "task_sha256",
            "generation_keys",
            "checkpoint_array_path",
            "checkpoint_json_path",
        }
    }


def _execute_gpu_tasks(
    tasks: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    if not tasks:
        return ()
    context = mp.get_context("spawn")
    executors = tuple(
        ProcessPoolExecutor(max_workers=1, mp_context=context) for _ in GPU_DEVICES
    )
    futures: dict[Future[Mapping[str, object]], Mapping[str, object]] = {}
    try:
        for task in tasks:
            device_index = GPU_DEVICES.index(str(task["device"]))
            futures[executors[device_index].submit(_production_generation_worker, task)] = task
        return tuple(future.result() for future in as_completed(futures))
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def _production_generation_worker(task: Mapping[str, object]) -> Mapping[str, object]:
    keys = tuple(task.get("generation_keys", ()))
    if (
        not keys
        or not all(isinstance(key, SourceGenerationKey) for key in keys)
        or task.get("target_cache_available") is not False
        or task.get("manifest_available") is not False
        or task.get("outcomes_available") is not False
        or task.get("amp_enabled") is not False
        or task.get("tf32_enabled") is not False
        or task.get("task_sha256") != _canonical_sha256(_task_identity(task))
    ):
        raise ProtocolError("SCEPTRE source worker boundary drifted.")
    device = str(task["device"])
    if device not in GPU_DEVICES:
        raise ProtocolError("SCEPTRE source worker device drifted.")
    import torch

    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    # The workstation contract budgets one native intra-op and one inter-op
    # thread per persistent GPU worker.  Setting only the intra-op pool leaves
    # a separate Torch dispatch pool unconstrained.
    torch.set_num_interop_threads(1)
    from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.serialization import (
        load_routing_authorized_expert,
    )
    from midogpp_thesis.cvae.generation.generation import generate_source_block

    expert = load_routing_authorized_expert(
        Path(str(task["expert_bank_root"])),
        source_center=str(task["source_center"]),
        training_seed=int(task["training_seed"]),
        device=device,
    )
    try:
        blocks = []
        for key in keys:
            generated = generate_source_block(
                expert,
                key,
                per_class=TOTAL_PER_CLASS,
                device=device,
            )
            values = np.ascontiguousarray(generated.embeddings, dtype=np.float32)
            if generated.output_sha256 != _block_bundle_sha256(values, TOTAL_PER_CLASS):
                raise ProtocolError("SCEPTRE generated source semantic hash drifted.")
            blocks.append(values)
        return _publish_checkpoint(task, blocks=blocks, geometry=PRODUCTION_SOURCE_GEOMETRY)
    finally:
        del expert
        gc.collect()
        torch.cuda.empty_cache()


def _execute_injected_task(
    task: Mapping[str, object],
    *,
    geometry: SourceGeometry,
    generate_block: Callable[[object, int, str], np.ndarray],
) -> Mapping[str, object]:
    if task.get("task_sha256") != _canonical_sha256(_task_identity(task)):
        raise ProtocolError("SCEPTRE injected source task identity drifted.")
    blocks = tuple(
        np.asarray(generate_block(key, geometry.rows_per_class, str(task["device"])))
        for key in task["generation_keys"]
    )
    return _publish_checkpoint(task, blocks=blocks, geometry=geometry)


def _publish_checkpoint(
    task: Mapping[str, object],
    *,
    blocks: Sequence[np.ndarray],
    geometry: SourceGeometry,
) -> Mapping[str, object]:
    if len(blocks) != len(geometry.generation_seeds):
        raise ProtocolError("SCEPTRE source worker generation-seed coverage drifted.")
    values = np.ascontiguousarray(np.stack(blocks), dtype=np.float32)
    expected_shape = (
        len(geometry.generation_seeds),
        2 * geometry.rows_per_class,
        geometry.feature_dim,
    )
    if (
        values.shape != expected_shape
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ProtocolError("SCEPTRE source worker emitted invalid values.")
    array_path = Path(str(task["checkpoint_array_path"]))
    json_path = Path(str(task["checkpoint_json_path"]))
    _persist_exact_npy(array_path, values)
    records = []
    for ordinal, key in enumerate(task["generation_keys"]):
        block = values[ordinal]
        records.append(
            {
                "generation_seed": int(getattr(key, "generation_seed")),
                "stream_id": str(getattr(key, "stream_id")),
                "expert_lock_hash": str(getattr(key, "expert_lock_hash")),
                "output_sha256": _block_bundle_sha256(
                    block, geometry.rows_per_class
                ),
                "array_sha256": sha256_array(block),
            }
        )
    checkpoint_unhashed = {
        "schema_version": "midogpp_sceptre_v2_source_checkpoint_v1",
        "status": "COMPLETE",
        "task_sha256": task["task_sha256"],
        "source_center": task["source_center"],
        "training_seed": task["training_seed"],
        "device": task["device"],
        "array_file_sha256": sha256_file(array_path),
        "array_shape": list(values.shape),
        "array_dtype": str(values.dtype),
        "records": records,
        "target_cache_opened": False,
        "manifest_opened": False,
        "outcomes_available": False,
        "expert_updated": False,
        "float32_outputs": True,
    }
    checkpoint = {
        **checkpoint_unhashed,
        "checkpoint_sha256": _canonical_sha256(checkpoint_unhashed),
    }
    _persist_exact_json(json_path, checkpoint)
    return checkpoint


def _load_checkpoint_if_complete(
    task: Mapping[str, object], *, geometry: SourceGeometry
) -> Mapping[str, object] | None:
    array_path = Path(str(task["checkpoint_array_path"]))
    json_path = Path(str(task["checkpoint_json_path"]))
    present = (array_path.is_file(), json_path.is_file())
    if any(path.is_symlink() for path in (array_path, json_path)):
        raise ProtocolError("SCEPTRE source checkpoint contains a symlink.")
    if present == (False, False):
        return None
    if present != (True, True):
        raise ProtocolError("SCEPTRE source checkpoint is partial; refusing refit.")
    payload = read_json(json_path)
    unhashed = {
        key: value for key, value in payload.items() if key != "checkpoint_sha256"
    }
    records = payload.get("records")
    values = np.load(array_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (
        len(geometry.generation_seeds),
        2 * geometry.rows_per_class,
        geometry.feature_dim,
    )
    if (
        payload.get("checkpoint_sha256") != _canonical_sha256(unhashed)
        or payload.get("schema_version")
        != "midogpp_sceptre_v2_source_checkpoint_v1"
        or payload.get("status") != "COMPLETE"
        or payload.get("task_sha256") != task["task_sha256"]
        or payload.get("source_center") != task["source_center"]
        or int(payload.get("training_seed", -1)) != task["training_seed"]
        or payload.get("device") != task["device"]
        or payload.get("array_file_sha256") != sha256_file(array_path)
        or payload.get("array_shape") != list(expected_shape)
        or payload.get("array_dtype") != "float32"
        or values.shape != expected_shape
        or values.dtype != np.float32
        or not np.isfinite(values).all()
        or not isinstance(records, list)
        or len(records) != len(geometry.generation_seeds)
        or payload.get("target_cache_opened") is not False
        or payload.get("manifest_opened") is not False
        or payload.get("outcomes_available") is not False
        or payload.get("expert_updated") is not False
    ):
        raise ProtocolError("SCEPTRE source checkpoint failed validation.")
    for ordinal, (raw, key) in enumerate(zip(records, task["generation_keys"], strict=True)):
        if (
            not isinstance(raw, Mapping)
            or int(raw.get("generation_seed", -1))
            != int(getattr(key, "generation_seed"))
            or raw.get("stream_id") != str(getattr(key, "stream_id"))
            or raw.get("expert_lock_hash") != str(getattr(key, "expert_lock_hash"))
            or raw.get("array_sha256") != sha256_array(values[ordinal])
            or raw.get("output_sha256")
            != _block_bundle_sha256(values[ordinal], geometry.rows_per_class)
        ):
            raise ProtocolError("SCEPTRE source checkpoint record drifted.")
    return payload


def _publish_source_array(
    path: Path,
    *,
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[tuple[str, int], Mapping[str, object]],
    geometry: SourceGeometry,
) -> tuple[SourceStreamRecord, ...]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    records: list[SourceStreamRecord] = []
    cursor = 0
    try:
        target = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=geometry.array_shape,
        )
        for task in tasks:
            checkpoint = completed[_task_key(task)]
            values = np.load(
                Path(str(task["checkpoint_array_path"])),
                mmap_mode="r",
                allow_pickle=False,
            )
            for seed_ordinal, raw in enumerate(checkpoint["records"]):
                target[cursor] = values[seed_ordinal]
                records.append(
                    SourceStreamRecord(
                        block_ordinal=cursor,
                        source_center=str(task["source_center"]),
                        training_seed=int(task["training_seed"]),
                        generation_seed=int(raw["generation_seed"]),
                        stream_id=str(raw["stream_id"]),
                        expert_lock_hash=str(raw["expert_lock_hash"]),
                        rows_per_class=geometry.rows_per_class,
                        feature_dim=geometry.feature_dim,
                        output_sha256=str(raw["output_sha256"]),
                        array_sha256=str(raw["array_sha256"]),
                    )
                )
                cursor += 1
        target.flush()
        del target
        if cursor != geometry.stream_count:
            raise ProtocolError("SCEPTRE source final array coverage drifted.")
        if path.exists():
            raise ProtocolError("SCEPTRE source final array appeared during publication.")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return tuple(records)


def _persist_exact_npy(path: Path, values: np.ndarray) -> None:
    if path.is_symlink():
        raise ProtocolError("SCEPTRE source checkpoint array is a symlink.")
    if path.exists():
        if not path.is_file():
            raise ProtocolError("SCEPTRE source checkpoint array is unsafe.")
        observed = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            observed.shape != values.shape
            or observed.dtype != values.dtype
            or sha256_array(observed) != sha256_array(values)
        ):
            raise ProtocolError(
                "SCEPTRE source checkpoint differs; refusing regeneration."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(values), allow_pickle=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _persist_exact_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise ProtocolError("SCEPTRE source JSON member is a symlink.")
    if path.exists():
        if not path.is_file() or read_json(path) != dict(payload):
            raise ProtocolError("SCEPTRE source JSON differs; refusing overwrite.")
        return
    atomic_json(path, payload)


def _validate_checkpoint_directory(directory: Path, geometry: SourceGeometry) -> None:
    if not directory.exists():
        if directory.is_symlink():
            raise ProtocolError("SCEPTRE source checkpoint root is a dangling symlink.")
        return
    if directory.is_symlink() or not directory.is_dir():
        raise ProtocolError("SCEPTRE source checkpoint root is unsafe.")
    expected = {
        f"source_{source}_train_{seed}.{suffix}"
        for source, seed in product(geometry.centers, geometry.training_seeds)
        for suffix in ("json", "npy")
    }
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file() or path.name not in expected:
            raise ProtocolError("SCEPTRE source checkpoint tree has an unknown member.")


def _task_key(task: Mapping[str, object]) -> tuple[str, int]:
    return str(task["source_center"]), int(task["training_seed"])


def _block_bundle_sha256(block: np.ndarray, rows_per_class: int) -> str:
    values = np.ascontiguousarray(block, dtype=np.float32)
    truth = np.concatenate(
        (
            np.zeros(rows_per_class, dtype=np.int64),
            np.ones(rows_per_class, dtype=np.int64),
        )
    )
    digest = hashlib.sha256()
    for array in (values, truth):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = (
    "CHECKPOINT_DIRECTORY",
    "GPU_DEVICES",
    "PRODUCTION_SOURCE_GEOMETRY",
    "SOURCE_ARRAY_MEMBER",
    "SOURCE_INDEX_MEMBER",
    "SOURCE_RECEIPT_MEMBER",
    "SourceGeometry",
    "SourceRuntimeConfig",
    "SourceRuntimeTestMode",
    "SourceStreamRecord",
    "SourceStreamStore",
    "load_source_streams",
    "materialize_source_streams",
)
