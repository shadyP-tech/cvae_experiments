"""GPU dispatch boundary for SCEPTRE v4 physical source generation.

Thread topology is configured only by the central worker initializer.  The
production task function deliberately contains no thread-setting calls so it
is safe after spawn-pool parallel work has begun.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, as_completed
import gc
from pathlib import Path

import numpy as np

from midogpp_thesis.cvae.generation.contracts import (
    TOTAL_PER_CLASS,
    SourceGenerationKey,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from ...sceptre_runtime.worker_lifecycle import single_worker_spawn_executor
from .source_checkpoints import publish_checkpoint
from .source_contracts import PRODUCTION_SOURCE_GEOMETRY, SourceGeometry
from .source_hashing import block_bundle_sha256, canonical_sha256
from .source_planning import task_identity
from .worker_runtime import (
    GPU_DEVICES,
    assert_gpu_worker_ready,
    initialize_gpu_worker,
)


def execute_gpu_tasks(
    tasks: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    if not tasks:
        return ()
    executors = tuple(
        single_worker_spawn_executor(
            initializer=initialize_gpu_worker,
            initargs=(device,),
        )
        for device in GPU_DEVICES
    )
    futures: dict[Future[Mapping[str, object]], Mapping[str, object]] = {}
    try:
        for task in tasks:
            device_index = GPU_DEVICES.index(str(task["device"]))
            futures[executors[device_index].submit(production_generation_worker, task)] = task
        return tuple(future.result() for future in as_completed(futures))
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def production_generation_worker(task: Mapping[str, object]) -> Mapping[str, object]:
    keys = tuple(task.get("generation_keys", ()))
    if (
        not keys
        or not all(isinstance(key, SourceGenerationKey) for key in keys)
        or task.get("target_cache_available") is not False
        or task.get("manifest_available") is not False
        or task.get("outcomes_available") is not False
        or task.get("amp_enabled") is not False
        or task.get("tf32_enabled") is not False
        or task.get("task_sha256") != canonical_sha256(task_identity(task))
    ):
        raise ProtocolError("SCEPTRE v4 source worker boundary drifted.")
    device = str(task["device"])
    if device not in GPU_DEVICES:
        raise ProtocolError("SCEPTRE v4 source worker device drifted.")
    assert_gpu_worker_ready(device)
    import torch
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
            if generated.output_sha256 != block_bundle_sha256(
                values, TOTAL_PER_CLASS
            ):
                raise ProtocolError("SCEPTRE v4 generated source semantic hash drifted.")
            blocks.append(values)
        return publish_checkpoint(
            task, blocks=blocks, geometry=PRODUCTION_SOURCE_GEOMETRY
        )
    finally:
        del expert
        gc.collect()
        torch.cuda.empty_cache()


def execute_injected_task(
    task: Mapping[str, object],
    *,
    geometry: SourceGeometry,
    generate_block: Callable[[object, int, str], np.ndarray],
) -> Mapping[str, object]:
    if task.get("task_sha256") != canonical_sha256(task_identity(task)):
        raise ProtocolError("SCEPTRE v4 injected source task identity drifted.")
    blocks = tuple(
        np.asarray(generate_block(key, geometry.rows_per_class, str(task["device"])))
        for key in task["generation_keys"]
    )
    return publish_checkpoint(task, blocks=blocks, geometry=geometry)


__all__ = (
    "execute_gpu_tasks",
    "execute_injected_task",
    "production_generation_worker",
)
