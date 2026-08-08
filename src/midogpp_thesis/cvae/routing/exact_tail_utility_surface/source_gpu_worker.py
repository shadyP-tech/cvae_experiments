"""Spawned persistent-GPU workers for exact-tail source realization."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from queue import Empty
from typing import Mapping, Sequence

import numpy as np

from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from .contracts import GENERATION_SEEDS, SOURCE_PREFIX_ROWS_PER_CLASS
from .runtime import GENERATION_DEVICES
from .source_checkpoint_store import atomic_save_npy, atomic_save_npz, sha256_file
from .source_contracts import ExpertTask


GPU_RESULT_POLL_SECONDS = 1.0
GPU_WORKER_JOIN_SECONDS = 2.0


def spawn_expert_tasks(
    tasks: Sequence[ExpertTask], bank_root: Path, output_root: Path
) -> tuple[Mapping[str, object], ...]:
    """Run one long-lived spawned process per frozen GPU device."""

    if not tasks:
        return ()
    context = mp.get_context("spawn")
    queues = [context.Queue(), context.Queue()]
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=expert_worker_main,
            args=(queues[index], result_queue, str(bank_root), str(output_root)),
            name=f"exact-tail-gpu-{index}",
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for task in tasks:
        queues[GENERATION_DEVICES.index(task.device)].put(task)
    for queue in queues:
        queue.put(None)
    expected_task_keys = tuple(
        (task.source_center, task.training_seed) for task in tasks
    )
    results: dict[tuple[str, int], Mapping[str, object]] = {}
    try:
        while len(results) < len(tasks):
            try:
                payload = result_queue.get(timeout=GPU_RESULT_POLL_SECONDS)
            except Empty:
                failed = [
                    process
                    for process in processes
                    if process.exitcode not in (None, 0)
                ]
                if failed:
                    raise ProtocolError(
                        "Exact-tail GPU worker exited before returning its task result."
                    )
                if all(process.exitcode is not None for process in processes):
                    raise ProtocolError(
                        "Exact-tail GPU workers exited with incomplete result coverage."
                    )
                continue
            if not isinstance(payload, Mapping):
                raise ProtocolError(
                    "Exact-tail GPU worker returned an invalid payload."
                )
            if payload.get("error"):
                raise ProtocolError(
                    f"Exact-tail GPU worker failed: {payload['error']}."
                )
            raw_task_key = payload.get("task_key")
            if (
                not isinstance(raw_task_key, (list, tuple))
                or len(raw_task_key) != 2
            ):
                raise ProtocolError("Exact-tail GPU worker result lacks its task key.")
            try:
                task_key = (str(raw_task_key[0]), int(raw_task_key[1]))
            except (TypeError, ValueError) as exc:
                raise ProtocolError(
                    "Exact-tail GPU worker task key is malformed."
                ) from exc
            if task_key not in expected_task_keys or task_key in results:
                raise ProtocolError("Exact-tail GPU worker task result drifted.")
            results[task_key] = payload
    except BaseException:
        terminate_and_join_workers(processes)
        raise
    for process in processes:
        process.join(timeout=GPU_WORKER_JOIN_SECONDS)
    if any(process.is_alive() for process in processes):
        terminate_and_join_workers(processes)
        raise ProtocolError("Exact-tail GPU worker did not stop after task completion.")
    for process in processes:
        if process.exitcode != 0:
            raise ProtocolError("Exact-tail GPU worker exited unsuccessfully.")
    return tuple(results[key] for key in expected_task_keys)


def terminate_and_join_workers(processes: Sequence[object]) -> None:
    """Bound cleanup after a child failure without leaving GPU workers alive."""

    for process in processes:
        if process.is_alive():  # type: ignore[attr-defined]
            process.terminate()  # type: ignore[attr-defined]
    for process in processes:
        process.join(timeout=GPU_WORKER_JOIN_SECONDS)  # type: ignore[attr-defined]
    for process in processes:
        if process.is_alive():  # type: ignore[attr-defined]
            process.kill()  # type: ignore[attr-defined]
            process.join(timeout=GPU_WORKER_JOIN_SECONDS)  # type: ignore[attr-defined]


def expert_worker_main(
    task_queue: object, result_queue: object, bank_root: str, output_root: str
) -> None:
    # All tensor/CUDA imports and state changes stay inside the spawned child.
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)

    from ...expert_bank.uniform_b_v2_promotion import (
        load_routing_authorized_expert,
    )
    from ...generation.generation import generate_source_block
    from ..dense_residual_soft_router.compatibility import (
        score_variational_compatibility,
    )

    while True:
        task = task_queue.get()  # type: ignore[attr-defined]
        if task is None:
            return
        try:
            assert isinstance(task, ExpertTask)
            torch.cuda.set_device(task.device)
            expert = load_routing_authorized_expert(
                bank_root,
                source_center=task.source_center,
                training_seed=task.training_seed,
                device=task.device,
            )
            source_payloads: list[dict[str, object]] = []
            source_path_by_seed = {
                int(key): Path(value)
                for key, value in task.existing_source_path_by_generation_seed.items()
            }
            output = Path(output_root)
            for key in task.generation_keys:
                block = generate_source_block(
                    expert,
                    key,
                    per_class=SOURCE_PREFIX_ROWS_PER_CLASS,
                    device=task.device,
                )
                path = output / f"worker/source_{key.stream_id}.npy"
                atomic_save_npy(path, block.embeddings)
                source_path_by_seed[key.generation_seed] = path
                source_payloads.append(
                    {
                        "source_center": key.source_center,
                        "training_seed": key.training_seed,
                        "generation_seed": key.generation_seed,
                        "stream_id": key.stream_id,
                        "expert_lock_hash": key.expert_lock_hash,
                        "path": str(path),
                        "file_sha256": sha256_file(path),
                        "output_sha256": block.output_sha256,
                        "rows_per_class": SOURCE_PREFIX_ROWS_PER_CLASS,
                        "feature_dim": COMMON_OUTPUT_DIM,
                    }
                )
            if set(source_path_by_seed) != set(GENERATION_SEEDS):
                raise ProtocolError("Exact-tail GPU worker lacks source streams for MMD.")
            component_payloads: list[dict[str, object]] = []
            for query in task.query_centers:
                support = np.load(
                    task.support_array_path_by_center[query],
                    mmap_mode="r",
                    allow_pickle=False,
                )
                cases = task.support_case_ids_by_center[query]
                energy = score_variational_compatibility(expert, support, cases)
                path = output / (
                    f"worker/components_q{query}_e{task.source_center}_"
                    f"train{task.training_seed}.npz"
                )
                atomic_save_npz(
                    path,
                    reconstruction_0=energy.per_class_reconstruction_mse[0],
                    reconstruction_1=energy.per_class_reconstruction_mse[1],
                    kl_0=energy.per_class_normalized_ps_kl[0],
                    kl_1=energy.per_class_normalized_ps_kl[1],
                )
                support_mean = np.mean(support, axis=0, dtype=np.float64)
                linear_kernel_mmd2 = {}
                for generation_seed, source_path in source_path_by_seed.items():
                    generated = np.load(source_path, mmap_mode="r", allow_pickle=False)
                    generated_mean = np.mean(generated, axis=0, dtype=np.float64)
                    mean_difference = support_mean - generated_mean
                    linear_kernel_mmd2[generation_seed] = float(
                        np.dot(mean_difference, mean_difference)
                    )
                component_payloads.append(
                    {
                        "query_center": query,
                        "candidate_source": task.source_center,
                        "training_seed": task.training_seed,
                        "path": str(path),
                        "file_sha256": sha256_file(path),
                        "case_equal_energy": energy.case_equal_mean,
                        "linear_kernel_mmd2_by_generation_seed": linear_kernel_mmd2,
                        "support_partition_hash": task.support_partition_hash_by_center[
                            query
                        ],
                    }
                )
            result_queue.put(  # type: ignore[attr-defined]
                {
                    "task_key": [task.source_center, task.training_seed],
                    "sources": source_payloads,
                    "components": component_payloads,
                }
            )
            del expert
            torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - workstation only
            result_queue.put(  # type: ignore[attr-defined]
                {"error": f"{type(exc).__name__}: {exc}"}
            )
            return


__all__ = (
    "expert_worker_main",
    "spawn_expert_tasks",
    "terminate_and_join_workers",
)
