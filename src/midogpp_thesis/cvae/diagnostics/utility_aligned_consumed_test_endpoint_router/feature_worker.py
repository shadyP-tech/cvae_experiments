"""Spawn-only persistent GPU workers for label-free support compatibility."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from queue import Empty
from typing import Mapping, Sequence

import numpy as np

from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ...runtime.frozen_source_streams import source_block_sha256
from .artifact_io import sha256_file
from .feature_checkpoint_store import publish_feature_checkpoint
from .feature_runtime_contracts import FeatureTask


RESULT_POLL_SECONDS = 1.0
WORKER_JOIN_SECONDS = 3.0
FEATURE_DEVICES = ("cuda:0", "cuda:1")


def load_verified_source_means(task: FeatureTask) -> Mapping[int, np.ndarray]:
    """Hash-verify every task-bound source block before feature reduction."""

    source_array = np.load(task.source_array_path, mmap_mode="r", allow_pickle=False)
    source_means: dict[int, np.ndarray] = {}
    for generation_seed, ordinal in task.source_block_ordinal_by_generation_seed.items():
        try:
            block = np.asarray(source_array[ordinal])
        except IndexError as exc:
            raise ProtocolError(
                "Endpoint-router frozen source block ordinal drifted in feature worker."
            ) from exc
        if (
            block.shape != (540, COMMON_OUTPUT_DIM)
            or block.dtype != np.float32
            or not np.isfinite(block).all()
        ):
            raise ProtocolError(
                "Endpoint-router frozen source block drifted in feature worker."
            )
        expected_hash = task.source_block_output_sha256_by_generation_seed[
            generation_seed
        ]
        if source_block_sha256(block) != expected_hash:
            raise ProtocolError(
                "Endpoint-router frozen source block semantic hash drifted in "
                "feature worker."
            )
        source_means[generation_seed] = np.mean(block, axis=0, dtype=np.float64)
    return source_means


def execute_feature_tasks(
    tasks: Sequence[FeatureTask],
) -> tuple[tuple[str, tuple[Mapping[str, object], ...]], ...]:
    """Run tasks through exactly two persistent spawned GPU processes."""

    values = tuple(tasks)
    if not values:
        return ()
    if (
        {task.device for task in values} != set(FEATURE_DEVICES)
        or any(task.device != FEATURE_DEVICES[index % 2] for index, task in enumerate(values))
    ):
        raise ProtocolError("Endpoint-router feature worker topology drifted.")
    context = mp.get_context("spawn")
    queues = [context.Queue(), context.Queue()]
    results = context.Queue()
    processes = [
        context.Process(
            target=feature_worker_main,
            args=(queues[index], results),
            name=f"endpoint-feature-gpu-{index}",
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for task in values:
        queues[FEATURE_DEVICES.index(task.device)].put(task)
    for queue in queues:
        queue.put(None)
    expected = {task.task_hash for task in values}
    observed: dict[str, tuple[Mapping[str, object], ...]] = {}
    try:
        while len(observed) < len(values):
            try:
                raw = results.get(timeout=RESULT_POLL_SECONDS)
            except Empty:
                if any(process.exitcode not in (None, 0) for process in processes):
                    raise ProtocolError("Endpoint-router feature worker exited early.")
                if all(process.exitcode is not None for process in processes):
                    raise ProtocolError("Endpoint-router feature workers returned incomplete coverage.")
                continue
            if not isinstance(raw, Mapping):
                raise ProtocolError("Endpoint-router feature worker result is malformed.")
            if raw.get("error"):
                raise ProtocolError(f"Endpoint-router feature worker failed: {raw['error']}.")
            task_hash = str(raw.get("task_hash", ""))
            component_rows = raw.get("components")
            if (
                task_hash not in expected
                or task_hash in observed
                or not isinstance(component_rows, (list, tuple))
                or any(not isinstance(row, Mapping) for row in component_rows)
            ):
                raise ProtocolError("Endpoint-router feature worker result drifted.")
            observed[task_hash] = tuple(component_rows)
    except BaseException:
        _terminate(processes)
        raise
    for process in processes:
        process.join(timeout=WORKER_JOIN_SECONDS)
    if any(process.is_alive() or process.exitcode != 0 for process in processes):
        _terminate(processes)
        raise ProtocolError("Endpoint-router feature worker shutdown failed.")
    return tuple((task.task_hash, observed[task.task_hash]) for task in values)


def feature_worker_main(task_queue: object, result_queue: object) -> None:
    """Child entry point; all torch/CUDA imports and state remain here."""

    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    from ...expert_bank.uniform_b_v2_promotion import load_routing_authorized_expert
    from .feature_energy_adapter import score_label_free_support

    while True:
        task = task_queue.get()  # type: ignore[attr-defined]
        if task is None:
            return
        try:
            if not isinstance(task, FeatureTask):
                raise ProtocolError("Endpoint-router feature worker received another task type.")
            torch.cuda.set_device(task.device)
            source_means = load_verified_source_means(task)
            expert = load_routing_authorized_expert(
                task.expert_bank_root,
                source_center=task.source_center,
                training_seed=task.training_seed,
                device=task.device,
            )
            arrays: dict[str, np.ndarray] = {}
            components: list[dict[str, object]] = []
            for support in task.support_slices:
                path = Path(task.support_root) / support.relative_array_path
                if not path.is_file() or sha256_file(path) != support.array_sha256:
                    raise ProtocolError("Endpoint-router staged support array drifted.")
                embeddings = np.load(path, mmap_mode="r", allow_pickle=False)
                if (
                    embeddings.shape != (support.row_count, COMMON_OUTPUT_DIM)
                    or embeddings.dtype != np.float32
                    or not np.isfinite(embeddings).all()
                ):
                    raise ProtocolError("Endpoint-router staged support geometry drifted.")
                energy = score_label_free_support(expert, embeddings, support.case_ids)
                prefix = f"q{support.query_center}"
                for class_id in (0, 1):
                    arrays[f"{prefix}_reconstruction_{class_id}"] = np.asarray(
                        energy.per_class_reconstruction_mse[class_id], dtype=np.float64
                    )
                for class_id in (0, 1):
                    arrays[f"{prefix}_kl_{class_id}"] = np.asarray(
                        energy.per_class_normalized_ps_kl[class_id], dtype=np.float64
                    )
                support_mean = np.mean(embeddings, axis=0, dtype=np.float64)
                mmd = {}
                for generation_seed, source_mean in source_means.items():
                    difference = support_mean - source_mean
                    mmd[generation_seed] = float(np.dot(difference, difference))
                components.append(
                    {
                        "query_center": support.query_center,
                        "case_equal_energy": float(energy.case_equal_mean),
                        "linear_kernel_mmd2_by_generation_seed": mmd,
                    }
                )
            records = publish_feature_checkpoint(
                task, arrays=arrays, component_payloads=components
            )
            result_queue.put(  # type: ignore[attr-defined]
                {
                    "task_hash": task.task_hash,
                    "components": [record.to_payload() for record in records],
                }
            )
            del expert
            torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - workstation-only path
            result_queue.put(  # type: ignore[attr-defined]
                {"error": f"{type(exc).__name__}: {exc}"}
            )
            return


def _terminate(processes: Sequence[object]) -> None:
    for process in processes:
        if process.is_alive():  # type: ignore[attr-defined]
            process.terminate()  # type: ignore[attr-defined]
    for process in processes:
        process.join(timeout=WORKER_JOIN_SECONDS)  # type: ignore[attr-defined]
        if process.is_alive():  # type: ignore[attr-defined]
            process.kill()  # type: ignore[attr-defined]
            process.join(timeout=WORKER_JOIN_SECONDS)  # type: ignore[attr-defined]


__all__ = (
    "FEATURE_DEVICES",
    "execute_feature_tasks",
    "feature_worker_main",
    "load_verified_source_means",
)
