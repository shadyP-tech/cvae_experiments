"""Deterministic workstation plan for the exact-tail development surface.

Runtime choices are intentionally separate from the scientific contract.  The
workstation schedule generates each 270-row/class source stream once on one of
two persistent GPU workers, then executes 648 coarse CPU jobs.  Each coarse job
loads its memmaps once and fits the exact base plus seven tails sequentially.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
from types import MappingProxyType
from typing import Callable, Mapping, Sequence, TypeVar

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    EXPECTED_COARSE_TASK_COUNT,
    EXPECTED_PREDICTION_CELL_COUNT,
    EXPECTED_SOURCE_STREAM_COUNT,
    GENERATION_SEEDS,
    SOURCE_PREFIX_ROWS_PER_CLASS,
    TRAINING_SEEDS,
    action_library_for,
    expected_coarse_task_keys,
    legal_sources,
)


WORKSTATION_PROFILE = "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"
GENERATION_DEVICES = ("cuda:0", "cuda:1")
GPU_WORKERS_PER_DEVICE = 1
CLASSIFIER_WORKERS = 4
CLASSIFIER_THREADS_PER_WORKER = 3
MINIMUM_LOGICAL_CPU_COUNT = 12
MINIMUM_RAM_GIB = 100
MINIMUM_GPU_MEMORY_MIB = 24_000
MINIMUM_GPU_FREE_MIB = 18_000
MINIMUM_ARTIFACT_DISK_FREE_GIB = 8
BLAS_THREAD_ENV_VALUE = "1"


@dataclass(frozen=True)
class WorkstationRuntimePlan:
    workstation_profile: str = WORKSTATION_PROFILE
    generation_devices: tuple[str, str] = GENERATION_DEVICES
    generation_workers_per_device: int = GPU_WORKERS_PER_DEVICE
    classifier_workers: int = CLASSIFIER_WORKERS
    classifier_threads_per_worker: int = CLASSIFIER_THREADS_PER_WORKER
    source_prefix_rows_per_class: int = SOURCE_PREFIX_ROWS_PER_CLASS
    parent_cuda_context_forbidden: bool = True
    multiprocessing_start_method: str = "spawn"
    generated_cache_storage: str = "float32_memmap"
    coarse_checkpoint_scope: str = "B_plus_seven_tails_atomic"
    tf32_enabled: bool = False
    amp_enabled: bool = False
    launch_blas_threads: int = 1
    scratch_preference: tuple[str, ...] = ("/data/local", "artifact_root")

    def __post_init__(self) -> None:
        if (
            self.workstation_profile != WORKSTATION_PROFILE
            or tuple(self.generation_devices) != GENERATION_DEVICES
            or self.generation_workers_per_device != 1
            or self.classifier_workers != 4
            or self.classifier_threads_per_worker != 3
            or self.source_prefix_rows_per_class != 270
            or self.parent_cuda_context_forbidden is not True
            or self.multiprocessing_start_method != "spawn"
            or self.generated_cache_storage != "float32_memmap"
            or self.coarse_checkpoint_scope != "B_plus_seven_tails_atomic"
            or self.tf32_enabled is not False
            or self.amp_enabled is not False
            or self.launch_blas_threads != 1
        ):
            raise ProtocolError("Exact-tail workstation runtime contract drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            "workstation_profile": self.workstation_profile,
            "generation_devices": list(self.generation_devices),
            "generation_workers_per_device": self.generation_workers_per_device,
            "classifier_workers": self.classifier_workers,
            "classifier_threads_per_worker": self.classifier_threads_per_worker,
            "source_prefix_rows_per_class": self.source_prefix_rows_per_class,
            "parent_cuda_context_forbidden": self.parent_cuda_context_forbidden,
            "multiprocessing_start_method": self.multiprocessing_start_method,
            "generated_cache_storage": self.generated_cache_storage,
            "coarse_checkpoint_scope": self.coarse_checkpoint_scope,
            "tf32_enabled": self.tf32_enabled,
            "amp_enabled": self.amp_enabled,
            "launch_blas_threads": self.launch_blas_threads,
            "scratch_preference": list(self.scratch_preference),
            "source_stream_count": EXPECTED_SOURCE_STREAM_COUNT,
            "coarse_task_count": EXPECTED_COARSE_TASK_COUNT,
            "classifier_fit_count": EXPECTED_PREDICTION_CELL_COUNT,
        }


@dataclass(frozen=True)
class WorkstationSnapshot:
    logical_cpu_count: int
    ram_gib: float
    gpu_names: tuple[str, ...]
    gpu_total_memory_mib: tuple[int, ...]
    gpu_free_memory_mib: tuple[int, ...]
    artifact_disk_free_gib: float
    parent_cuda_initialized: bool


def validate_workstation_snapshot(snapshot: WorkstationSnapshot) -> None:
    """Fail closed before spawning expensive workers on the wrong topology."""

    if snapshot.logical_cpu_count < MINIMUM_LOGICAL_CPU_COUNT:
        raise ProtocolError("Exact-tail workstation exposes fewer than 12 CPUs.")
    if snapshot.ram_gib < MINIMUM_RAM_GIB:
        raise ProtocolError("Exact-tail workstation exposes less than 100 GiB RAM.")
    if (
        len(snapshot.gpu_names) != 2
        or len(snapshot.gpu_total_memory_mib) != 2
        or len(snapshot.gpu_free_memory_mib) != 2
    ):
        raise ProtocolError("Exact-tail workstation requires exactly two GPUs.")
    if any("RTX A5000" not in name for name in snapshot.gpu_names) or any(
        memory < MINIMUM_GPU_MEMORY_MIB
        for memory in snapshot.gpu_total_memory_mib
    ):
        raise ProtocolError("Exact-tail workstation GPU profile is not dual A5000.")
    if any(memory < MINIMUM_GPU_FREE_MIB for memory in snapshot.gpu_free_memory_mib):
        raise ProtocolError("Exact-tail workstation GPU free-memory reserve is too low.")
    if snapshot.artifact_disk_free_gib < MINIMUM_ARTIFACT_DISK_FREE_GIB:
        raise ProtocolError("Exact-tail workstation lacks artifact-disk reserve.")
    if snapshot.parent_cuda_initialized:
        raise ProtocolError("Exact-tail parent CUDA context exists before spawn.")


@dataclass(frozen=True)
class SourceStreamTask:
    source_center: str
    training_seed: int
    generation_seed: int
    rows_per_class: int
    task_hash: str

    @property
    def key(self) -> tuple[str, int, int]:
        return self.source_center, self.training_seed, self.generation_seed


@dataclass(frozen=True)
class CoarsePredictionTask:
    """One checkpoint unit that amortizes loads over eight classifier fits."""

    outer_target: str
    pseudo_query: str
    training_seed: int
    generation_seed: int
    candidate_sources: tuple[str, ...]
    action_ids: tuple[str, ...]
    task_hash: str

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (
            self.outer_target,
            self.pseudo_query,
            self.training_seed,
            self.generation_seed,
        )


def source_stream_tasks() -> tuple[SourceStreamTask, ...]:
    tasks: list[SourceStreamTask] = []
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                payload = {
                    "source_center": source,
                    "training_seed": training_seed,
                    "generation_seed": generation_seed,
                    "rows_per_class": SOURCE_PREFIX_ROWS_PER_CLASS,
                    "storage": "float32_memmap",
                }
                tasks.append(
                    SourceStreamTask(
                        source_center=source,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        rows_per_class=SOURCE_PREFIX_ROWS_PER_CLASS,
                        task_hash=stable_hash(payload),
                    )
                )
    if len(tasks) != EXPECTED_SOURCE_STREAM_COUNT:
        raise ProtocolError("Exact-tail source-stream task coverage drifted.")
    return tuple(tasks)


def coarse_prediction_tasks() -> tuple[CoarsePredictionTask, ...]:
    tasks: list[CoarsePredictionTask] = []
    for outer, query, training_seed, generation_seed in expected_coarse_task_keys():
        sources = legal_sources(outer_target=outer, pseudo_query=query)
        actions = action_library_for(outer_target=outer, pseudo_query=query)
        payload = {
            "outer_target": outer,
            "pseudo_query": query,
            "training_seed": training_seed,
            "generation_seed": generation_seed,
            "candidate_sources": list(sources),
            "action_hashes": [action.action_hash for action in actions],
            "checkpoint_scope": "B_plus_seven_tails_atomic",
        }
        tasks.append(
            CoarsePredictionTask(
                outer_target=outer,
                pseudo_query=query,
                training_seed=training_seed,
                generation_seed=generation_seed,
                candidate_sources=sources,
                action_ids=tuple(action.action_id for action in actions),
                task_hash=stable_hash(payload),
            )
        )
    if len(tasks) != EXPECTED_COARSE_TASK_COUNT:
        raise ProtocolError("Exact-tail coarse-task coverage drifted.")
    return tuple(tasks)


@dataclass(frozen=True)
class CoarseTaskCheckpoint:
    task_key: tuple[str, str, int, int]
    task_hash: str
    action_prediction_sha256: Mapping[str, str]
    action_probability_sha256: Mapping[str, str]
    action_support_probability_sha256: Mapping[str, str]
    checkpoint_hash: str

    def __post_init__(self) -> None:
        task_by_key = {task.key: task for task in coarse_prediction_tasks()}
        expected = task_by_key.get(tuple(self.task_key))
        if expected is None or self.task_hash != expected.task_hash:
            raise ProtocolError("Exact-tail checkpoint task binding drifted.")
        predictions = _hash_mapping(
            self.action_prediction_sha256, expected.action_ids, "prediction"
        )
        probabilities = _hash_mapping(
            self.action_probability_sha256, expected.action_ids, "probability"
        )
        support_probabilities = _hash_mapping(
            self.action_support_probability_sha256,
            expected.action_ids,
            "support probability",
        )
        object.__setattr__(
            self, "action_prediction_sha256", MappingProxyType(predictions)
        )
        object.__setattr__(
            self, "action_probability_sha256", MappingProxyType(probabilities)
        )
        object.__setattr__(
            self,
            "action_support_probability_sha256",
            MappingProxyType(support_probabilities),
        )
        expected_hash = stable_hash(self._unhashed_payload())
        if self.checkpoint_hash != expected_hash:
            raise ProtocolError("Exact-tail coarse checkpoint hash drifted.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_exact_tail_coarse_checkpoint_v2",
            "task_key": list(self.task_key),
            "task_hash": self.task_hash,
            "action_prediction_sha256": dict(self.action_prediction_sha256),
            "action_probability_sha256": dict(self.action_probability_sha256),
            "action_support_probability_sha256": dict(
                self.action_support_probability_sha256
            ),
            "all_eight_actions_materialized": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "checkpoint_hash": self.checkpoint_hash}


def build_coarse_task_checkpoint(
    *,
    task: CoarsePredictionTask,
    action_prediction_sha256: Mapping[str, str],
    action_probability_sha256: Mapping[str, str],
    action_support_probability_sha256: Mapping[str, str],
) -> CoarseTaskCheckpoint:
    values: dict[str, object] = {
        "task_key": task.key,
        "task_hash": task.task_hash,
        "action_prediction_sha256": dict(action_prediction_sha256),
        "action_probability_sha256": dict(action_probability_sha256),
        "action_support_probability_sha256": dict(
            action_support_probability_sha256
        ),
        "checkpoint_hash": "",
    }
    provisional = CoarseTaskCheckpoint.__new__(CoarseTaskCheckpoint)
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    values["checkpoint_hash"] = stable_hash(provisional._unhashed_payload())
    return CoarseTaskCheckpoint(**values)  # type: ignore[arg-type]


def validate_complete_checkpoint_set(
    checkpoints: Sequence[CoarseTaskCheckpoint],
) -> tuple[CoarseTaskCheckpoint, ...]:
    expected = coarse_prediction_tasks()
    by_key: dict[tuple[str, str, int, int], CoarseTaskCheckpoint] = {}
    for checkpoint in checkpoints:
        if checkpoint.task_key in by_key:
            raise ProtocolError("Exact-tail checkpoint set duplicates a coarse task.")
        by_key[checkpoint.task_key] = checkpoint
    if set(by_key) != {task.key for task in expected}:
        raise ProtocolError("Exact-tail checkpoint set is incomplete.")
    ordered = tuple(by_key[task.key] for task in expected)
    if any(checkpoint.task_hash != task.task_hash for checkpoint, task in zip(
        ordered, expected, strict=True
    )):
        raise ProtocolError("Exact-tail checkpoint task plan drifted.")
    return ordered


T = TypeVar("T")


def execute_coarse_tasks(
    tasks: Sequence[CoarsePredictionTask],
    worker: Callable[[CoarsePredictionTask], T],
    *,
    max_workers: int = CLASSIFIER_WORKERS,
) -> tuple[T, ...]:
    """Execute independent coarse tasks with deterministic result ordering.

    The worker must fit all eight actions and publish one atomic checkpoint.
    GPU generation is intentionally not permitted in this CPU phase.
    """

    planned = tuple(tasks)
    if not planned or len({task.key for task in planned}) != len(planned):
        raise ProtocolError("Exact-tail execution tasks are empty or duplicated.")
    if max_workers != CLASSIFIER_WORKERS:
        raise ProtocolError("Exact-tail CPU pool must use four workers.")
    context = mp.get_context("spawn")
    completed: dict[tuple[str, str, int, int], T] = {}
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as pool:
        futures = {pool.submit(worker, task): task.key for task in planned}
        for future in as_completed(futures):
            key = futures[future]
            if key in completed:
                raise ProtocolError("Exact-tail CPU worker duplicated a result.")
            completed[key] = future.result()
    if set(completed) != {task.key for task in planned}:
        raise ProtocolError("Exact-tail CPU worker result coverage drifted.")
    return tuple(completed[task.key] for task in planned)


def _hash_mapping(
    raw: Mapping[str, str], expected_keys: Sequence[str], role: str
) -> dict[str, str]:
    values = {str(key): str(value) for key, value in raw.items()}
    if tuple(values) != tuple(expected_keys):
        raise ProtocolError(f"Exact-tail {role} hash action coverage drifted.")
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in values.values()
    ):
        raise ProtocolError(f"Exact-tail {role} SHA-256 is malformed.")
    return values


__all__ = (
    "BLAS_THREAD_ENV_VALUE",
    "CLASSIFIER_THREADS_PER_WORKER",
    "CLASSIFIER_WORKERS",
    "GENERATION_DEVICES",
    "GPU_WORKERS_PER_DEVICE",
    "MINIMUM_ARTIFACT_DISK_FREE_GIB",
    "MINIMUM_GPU_MEMORY_MIB",
    "MINIMUM_GPU_FREE_MIB",
    "MINIMUM_LOGICAL_CPU_COUNT",
    "MINIMUM_RAM_GIB",
    "WORKSTATION_PROFILE",
    "CoarsePredictionTask",
    "CoarseTaskCheckpoint",
    "SourceStreamTask",
    "WorkstationRuntimePlan",
    "WorkstationSnapshot",
    "build_coarse_task_checkpoint",
    "coarse_prediction_tasks",
    "execute_coarse_tasks",
    "source_stream_tasks",
    "validate_complete_checkpoint_set",
    "validate_workstation_snapshot",
)
