"""Deterministic spawn pools with sealed callback and result-byte admission."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import multiprocessing as mp
import os
import time

from ..protocol import ProtocolError
from .batch_receipts import ExecutionBatchResult, build_execution_batch_result
from .callbacks import resolve_sealed_callback, seal_callback_descriptor
from .dtos import (
    OuterFoldTaskDTO,
    PredictionTaskDTO,
    SealedCallbackDescriptorDTO,
    WorkerExecutionDTO,
    WorkerResultDTO,
    assert_pickle_safe_label_free_dto,
)
from .file_evidence import hash_read_only_regular_file
from .workstation import (
    assert_coordinator_process,
    capture_threadpool_evidence,
    initialize_cpu_outer_worker,
    initialize_persistent_gpu_worker,
    validate_worker_environment,
)


PredictionCallback = Callable[[PredictionTaskDTO], WorkerResultDTO]
OuterFoldCallback = Callable[[OuterFoldTaskDTO], WorkerResultDTO]
WorkerExecutionReceipt = WorkerExecutionDTO
_CALLBACK_LOGICAL_DEVICE_ENV = "MIDOGPP_OE_PPUR_CALLBACK_LOGICAL_DEVICE"
_CPU_STARTUP_BARRIER: object | None = None


def run_persistent_gpu_prediction_pool(
    tasks: Sequence[PredictionTaskDTO],
    *,
    callback: PredictionCallback | SealedCallbackDescriptorDTO,
    result_evidence_mode: str = "auto",
) -> ExecutionBatchResult:
    """Run canonical tasks without transporting a callable to either worker."""

    assert_coordinator_process()
    descriptor = _coerce_callback_descriptor(
        callback,
        role="gpu_prediction",
        result_evidence_mode=result_evidence_mode,
    )
    _assert_v1_test_only_launch(descriptor)
    resolve_sealed_callback(descriptor, expected_role="gpu_prediction")
    ordered = tuple(sorted(tasks, key=lambda row: (row.task_id, row.task_hash)))
    if (
        len(ordered) < 2
        or any(not isinstance(row, PredictionTaskDTO) for row in ordered)
        or len({row.task_id for row in ordered}) != len(ordered)
        or len({row.task_hash for row in ordered}) != len(ordered)
        or any(row.device_index != ordinal % 2 for ordinal, row in enumerate(ordered))
    ):
        raise ProtocolError("OE-PPUR GPU task distribution is not canonical round-robin.")
    for task in ordered:
        assert_pickle_safe_label_free_dto(task)

    futures: list[Future[WorkerExecutionDTO]] = []
    with ExitStack() as stack:
        executors = tuple(
            stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=mp.get_context("spawn"),
                    initializer=initialize_persistent_gpu_worker,
                    initargs=(device_index,),
                )
            )
            for device_index in (0, 1)
        )
        roster = tuple(
            executor.submit(_observe_worker_pid, "gpu").result()
            for executor in executors
        )
        for ordinal, task in enumerate(ordered):
            futures.append(
                executors[task.device_index].submit(
                    _execute_gpu_prediction_task,
                    descriptor,
                    ordinal,
                    task,
                )
            )
        transported = tuple(future.result() for future in futures)
    receipts = tuple(
        admit_worker_execution_transport(row, descriptor=descriptor, task=task)
        for row, task in zip(transported, ordered, strict=True)
    )
    return build_execution_batch_result(
        role="gpu_prediction",
        configured_worker_count=2,
        configured_device_indices=(0, 1),
        ordered_task_hashes=tuple(row.task_hash for row in ordered),
        receipts=receipts,
        worker_roster=roster,
    )


def run_cpu_outer_pool(
    tasks: Sequence[OuterFoldTaskDTO],
    *,
    callback: OuterFoldCallback | SealedCallbackDescriptorDTO,
    candidate_surface: object,
    result_evidence_mode: str = "auto",
) -> ExecutionBatchResult:
    """Run one CUDA-hidden four-worker pool with descriptor-only callbacks."""

    assert_coordinator_process()
    descriptor = _coerce_callback_descriptor(
        callback,
        role="cpu_outer",
        result_evidence_mode=result_evidence_mode,
    )
    _assert_v1_test_only_launch(descriptor)
    resolve_sealed_callback(descriptor, expected_role="cpu_outer")
    from .surfaces import validate_candidate_probability_surface_receipt

    surface = validate_candidate_probability_surface_receipt(candidate_surface)
    ordered = tuple(
        sorted(
            tasks,
            key=lambda row: (row.H, row.J, row.K, row.L, row.d, row.task_hash),
        )
    )
    if (
        len(ordered) < 4
        or any(not isinstance(row, OuterFoldTaskDTO) for row in ordered)
        or len({row.task_hash for row in ordered}) != len(ordered)
        or any(row.row_index_sha256 != surface.row_index_sha256 for row in ordered)
        or any(
            row.candidate_probability_surface_sha256
            != surface.candidate_probability_surface_sha256
            for row in ordered
        )
        or any(
            row.probability_sha256 not in surface.output_file_hashes
            for row in ordered
        )
    ):
        raise ProtocolError("OE-PPUR CPU outer-task inventory is invalid.")
    for task in ordered:
        assert_pickle_safe_label_free_dto(task)

    context = mp.get_context("spawn")
    startup_barrier = context.Barrier(4)
    with ProcessPoolExecutor(
        max_workers=4,
        mp_context=context,
        initializer=_initialize_cpu_outer_worker_with_barrier,
        initargs=(startup_barrier,),
    ) as executor:
        roster_futures = tuple(
            executor.submit(_observe_cpu_worker_pid_at_barrier) for _ in range(4)
        )
        roster = tuple(sorted(future.result() for future in roster_futures))
        if len(roster) != 4 or len(set(roster)) != 4:
            raise ProtocolError(
                "OE-PPUR CPU pool did not expose four distinct workers."
            )
        futures = tuple(
            executor.submit(
                _execute_cpu_outer_task,
                descriptor,
                ordinal,
                task,
            )
            for ordinal, task in enumerate(ordered)
        )
        transported = tuple(future.result() for future in futures)
    receipts = tuple(
        admit_worker_execution_transport(row, descriptor=descriptor, task=task)
        for row, task in zip(transported, ordered, strict=True)
    )
    return build_execution_batch_result(
        role="cpu_outer",
        configured_worker_count=4,
        configured_device_indices=(),
        ordered_task_hashes=tuple(row.task_hash for row in ordered),
        receipts=receipts,
        worker_roster=roster,
    )


def admit_worker_execution_transport(
    value: WorkerExecutionDTO,
    *,
    descriptor: SealedCallbackDescriptorDTO,
    task: PredictionTaskDTO | OuterFoldTaskDTO,
) -> WorkerExecutionDTO:
    """Revalidate primitive transport and claimed result bytes in coordinator."""

    if not isinstance(value, WorkerExecutionDTO):
        raise ProtocolError("OE-PPUR worker returned a non-transport result.")
    assert_pickle_safe_label_free_dto(value)
    assert_pickle_safe_label_free_dto(descriptor)
    if (
        value.task_hash != task.task_hash
        or value.callback_descriptor_hash != descriptor.descriptor_hash
        or value.role != descriptor.callback_role
        or value.result_evidence_mode != descriptor.result_evidence_mode
        or value.row_index_sha256 != task.row_index_sha256
        or value.source_surface_sha256 != _task_source_surface(task)
        or value.verified_input_content_hashes != _task_input_hashes(task)
    ):
        raise ProtocolError("OE-PPUR worker transport lineage drifted.")
    if descriptor.result_evidence_mode == "regular_file":
        _reopen_and_hash_result_files(value)
    else:
        _validate_strict_test_result_fixture(value)
    return value


def _execute_gpu_prediction_task(
    descriptor: SealedCallbackDescriptorDTO,
    ordinal: int,
    task: PredictionTaskDTO,
) -> WorkerExecutionDTO:
    assert_pickle_safe_label_free_dto(descriptor)
    assert_pickle_safe_label_free_dto(task)
    validate_worker_environment(os.environ, role="gpu")
    physical_device = int(os.environ["CUDA_VISIBLE_DEVICES"])
    if task.device_index != physical_device:
        raise ProtocolError("OE-PPUR GPU task reached the wrong physical worker.")
    _set_callback_context("cuda:0")
    callback = resolve_sealed_callback(descriptor, expected_role="gpu_prediction")
    _assert_callback_context(role="gpu", physical_device=physical_device)
    verified_inputs, input_bytes_revalidated = _validate_worker_input_bytes(
        task,
        evidence_mode=descriptor.result_evidence_mode,
    )
    result = callback(task)
    _assert_callback_context(role="gpu", physical_device=physical_device)
    return _build_worker_transport(
        role="gpu_prediction",
        ordinal=ordinal,
        task=task,
        descriptor=descriptor,
        result=result,
        verified_input_content_hashes=verified_inputs,
        input_bytes_revalidated=input_bytes_revalidated,
        physical_device_index=physical_device,
        callback_visible_logical_device="cuda:0",
    )


def _execute_cpu_outer_task(
    descriptor: SealedCallbackDescriptorDTO,
    ordinal: int,
    task: OuterFoldTaskDTO,
) -> WorkerExecutionDTO:
    assert_pickle_safe_label_free_dto(descriptor)
    assert_pickle_safe_label_free_dto(task)
    validate_worker_environment(os.environ, role="cpu")
    _set_callback_context("cpu")
    callback = resolve_sealed_callback(descriptor, expected_role="cpu_outer")
    _assert_callback_context(role="cpu", physical_device=None)
    verified_inputs, input_bytes_revalidated = _validate_worker_input_bytes(
        task,
        evidence_mode=descriptor.result_evidence_mode,
    )
    result = callback(task)
    _assert_callback_context(role="cpu", physical_device=None)
    return _build_worker_transport(
        role="cpu_outer",
        ordinal=ordinal,
        task=task,
        descriptor=descriptor,
        result=result,
        verified_input_content_hashes=verified_inputs,
        input_bytes_revalidated=input_bytes_revalidated,
        physical_device_index=None,
        callback_visible_logical_device="cpu",
    )


def _build_worker_transport(
    *,
    role: str,
    ordinal: int,
    task: PredictionTaskDTO | OuterFoldTaskDTO,
    descriptor: SealedCallbackDescriptorDTO,
    result: object,
    verified_input_content_hashes: tuple[str, ...],
    input_bytes_revalidated: bool,
    physical_device_index: int | None,
    callback_visible_logical_device: str,
) -> WorkerExecutionDTO:
    _validate_worker_result(task, result)
    evidence = capture_threadpool_evidence(
        role="gpu" if role == "gpu_prediction" else "cpu"
    )
    transport = WorkerExecutionDTO(
        role=role,
        task_ordinal=ordinal,
        task_hash=task.task_hash,
        callback_descriptor_hash=descriptor.descriptor_hash,
        result_paths=result.result_paths,
        result_hashes=result.result_hashes,
        row_count=result.row_count,
        row_index_sha256=result.row_index_sha256,
        source_surface_sha256=result.source_surface_sha256,
        verified_input_content_hashes=verified_input_content_hashes,
        input_bytes_revalidated=input_bytes_revalidated,
        worker_result_hash=result.result_hash,
        worker_pid=os.getpid(),
        physical_device_index=physical_device_index,
        callback_visible_logical_device=callback_visible_logical_device,
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        limiter_scope_entered=evidence.limiter_scope_entered,
        threadpool_observation_performed=evidence.observation_performed,
        loaded_pool_count=evidence.loaded_pool_count,
        loaded_pools=tuple(
            (row.user_api, row.internal_api, row.prefix, row.num_threads)
            for row in evidence.loaded_pools
        ),
        all_loaded_pools_one_thread=evidence.all_loaded_pools_one_thread,
        result_evidence_mode=descriptor.result_evidence_mode,
    )
    assert_pickle_safe_label_free_dto(transport)
    return transport


def _coerce_callback_descriptor(
    callback: object,
    *,
    role: str,
    result_evidence_mode: str,
) -> SealedCallbackDescriptorDTO:
    if isinstance(callback, SealedCallbackDescriptorDTO):
        descriptor = callback
        if (
            result_evidence_mode != "auto"
            and descriptor.result_evidence_mode != result_evidence_mode
        ):
            raise ProtocolError("OE-PPUR callback result-evidence mode drifted.")
    else:
        descriptor = seal_callback_descriptor(
            callback,
            callback_role=role,
            result_evidence_mode=result_evidence_mode,
        )
    assert_pickle_safe_label_free_dto(descriptor)
    return descriptor


def _assert_v1_test_only_launch(
    descriptor: SealedCallbackDescriptorDTO,
) -> None:
    if (
        descriptor.result_evidence_mode == "strict_test_fixture"
        and "PYTEST_CURRENT_TEST" in os.environ
        and "/tests/cvae/" in descriptor.source_path
        and descriptor.member_name.startswith("_synthetic_")
    ):
        return
    raise ProtocolError("OE-PPUR workstation execution is not authorized.")


def _validate_worker_result(
    task: PredictionTaskDTO | OuterFoldTaskDTO,
    result: object,
) -> None:
    if (
        not isinstance(result, WorkerResultDTO)
        or result.task_hash != task.task_hash
        or result.row_index_sha256 != task.row_index_sha256
        or result.source_surface_sha256 != _task_source_surface(task)
    ):
        raise ProtocolError("OE-PPUR worker result drifted from its task.")
    assert_pickle_safe_label_free_dto(result)


def _task_source_surface(
    task: PredictionTaskDTO | OuterFoldTaskDTO,
) -> str:
    if isinstance(task, PredictionTaskDTO):
        return task.source_surface_sha256
    if isinstance(task, OuterFoldTaskDTO):
        return task.candidate_probability_surface_sha256
    raise ProtocolError("OE-PPUR worker task source surface is untyped.")


def _task_input_paths(
    task: PredictionTaskDTO | OuterFoldTaskDTO,
) -> tuple[str, ...]:
    if isinstance(task, PredictionTaskDTO):
        return task.input_paths
    if isinstance(task, OuterFoldTaskDTO):
        return (task.probability_path,)
    raise ProtocolError("OE-PPUR worker input path inventory is untyped.")


def _task_input_hashes(
    task: PredictionTaskDTO | OuterFoldTaskDTO,
) -> tuple[str, ...]:
    if isinstance(task, PredictionTaskDTO):
        return task.input_hashes
    if isinstance(task, OuterFoldTaskDTO):
        return (task.probability_sha256,)
    raise ProtocolError("OE-PPUR worker input hash inventory is untyped.")


def _validate_worker_input_bytes(
    task: PredictionTaskDTO | OuterFoldTaskDTO,
    *,
    evidence_mode: str,
) -> tuple[tuple[str, ...], bool]:
    paths = _task_input_paths(task)
    expected = _task_input_hashes(task)
    if evidence_mode == "strict_test_fixture":
        return expected, False
    if evidence_mode != "regular_file":
        raise ProtocolError("OE-PPUR worker input evidence mode is invalid.")
    observed = tuple(_hash_regular_input_file(path) for path in paths)
    if observed != expected:
        raise ProtocolError("OE-PPUR worker input bytes drifted.")
    return observed, True


def _hash_regular_input_file(path_text: str) -> str:
    return hash_read_only_regular_file(path_text, role="worker input")


def _set_callback_context(logical_device: str) -> None:
    if logical_device not in {"cuda:0", "cpu"}:
        raise ProtocolError("OE-PPUR callback logical device is invalid.")
    os.environ[_CALLBACK_LOGICAL_DEVICE_ENV] = logical_device


def _assert_callback_context(*, role: str, physical_device: int | None) -> None:
    validate_worker_environment(os.environ, role=role)
    logical = os.environ.get(_CALLBACK_LOGICAL_DEVICE_ENV)
    if role == "gpu":
        if (
            physical_device not in (0, 1)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_device)
            or logical != "cuda:0"
        ):
            raise ProtocolError("OE-PPUR GPU callback context drifted.")
    elif (
        role != "cpu"
        or physical_device is not None
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or logical != "cpu"
    ):
        raise ProtocolError("OE-PPUR CPU callback context drifted.")


def _observe_worker_pid(role: str) -> int:
    validate_worker_environment(os.environ, role=role)
    time.sleep(0.05)
    return os.getpid()


def _initialize_cpu_outer_worker_with_barrier(barrier: object) -> None:
    global _CPU_STARTUP_BARRIER
    initialize_cpu_outer_worker()
    _CPU_STARTUP_BARRIER = barrier


def _observe_cpu_worker_pid_at_barrier() -> int:
    validate_worker_environment(os.environ, role="cpu")
    wait = getattr(_CPU_STARTUP_BARRIER, "wait", None)
    if not callable(wait):
        raise ProtocolError("OE-PPUR worker startup barrier is invalid.")
    try:
        wait(timeout=30.0)
    except Exception as exc:
        raise ProtocolError("OE-PPUR worker startup barrier failed.") from exc
    return os.getpid()


def _reopen_and_hash_result_files(value: WorkerExecutionDTO) -> None:
    for path_text, expected_hash in zip(
        value.result_paths,
        value.result_hashes,
        strict=True,
    ):
        observed_hash = hash_read_only_regular_file(
            path_text,
            role="claimed worker result",
        )
        if observed_hash != expected_hash:
            raise ProtocolError("OE-PPUR worker result bytes drifted.")


def _validate_strict_test_result_fixture(value: WorkerExecutionDTO) -> None:
    fixture_hash = hashlib.sha256(value.task_hash.encode("ascii")).hexdigest()
    if len(value.result_hashes) != 1 or value.result_hashes[0] != fixture_hash:
        raise ProtocolError("OE-PPUR strict synthetic result fixture drifted.")


__all__ = (
    "ExecutionBatchResult",
    "OuterFoldCallback",
    "PredictionCallback",
    "WorkerExecutionReceipt",
    "admit_worker_execution_transport",
    "run_cpu_outer_pool",
    "run_persistent_gpu_prediction_pool",
)
