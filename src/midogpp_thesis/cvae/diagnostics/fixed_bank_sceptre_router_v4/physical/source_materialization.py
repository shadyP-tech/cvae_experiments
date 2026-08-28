"""Orchestrate label-free physical source generation and sealing."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import shutil

from midogpp_thesis.cvae.generation.contracts import GenerationLock
from midogpp_thesis.cvae.protocol import ProtocolError

from .gpu_dispatch import execute_gpu_tasks, execute_injected_task
from .source_checkpoints import (
    load_checkpoint_if_complete,
    validate_checkpoint_directory,
)
from .source_contracts import (
    CHECKPOINT_DIRECTORY,
    SourceRuntimeConfig,
    SourceRuntimeTestMode,
    SourceStreamStore,
)
from .source_planning import (
    assert_owned_root,
    assert_parent_cuda_free,
    assert_production_runtime,
    build_tasks,
    config_hash,
    final_paths,
    generation_keys,
    geometry_for,
    resolve_attempt_id,
    resolve_expert_bank_root,
    task_key,
    validate_generation_grid,
)
from .source_store import load_source_streams, publish_source_store


def materialize_source_streams(
    config: SourceRuntimeConfig,
    generation_lock: GenerationLock,
    *,
    root: Path,
    expert_bank_root: Path | None = None,
    attempt_id: str | None = None,
    test_mode: SourceRuntimeTestMode | None = None,
) -> SourceStreamStore:
    """Materialize every frozen source stream and return a read-only store.

    Production submits exactly 27 jobs to two persistent one-process spawn
    pools. Each job loads one source/training expert and emits all three
    generation seeds, so the expert is loaded only once per physical task.
    """

    geometry = geometry_for(test_mode)
    resolved_config_hash = config_hash(config)
    destination = Path(root)
    attempt = resolve_attempt_id(
        config,
        explicit=attempt_id,
        root=destination,
        synthetic=test_mode is not None,
    )
    bank_root = resolve_expert_bank_root(
        config,
        explicit=expert_bank_root,
        synthetic=test_mode is not None,
        owned_root=destination,
    )
    assert_owned_root(destination)
    assert_parent_cuda_free()
    if test_mode is None:
        assert_production_runtime(config.runtime)
    keys = generation_keys(generation_lock, test_mode=test_mode)
    validate_generation_grid(
        keys,
        generation_lock,
        geometry=geometry,
        test_mode=test_mode,
    )
    assert_parent_cuda_free()

    store_paths = final_paths(destination)
    present = tuple(path.is_file() for path in store_paths)
    if any(path.is_symlink() for path in store_paths):
        raise ProtocolError("SCEPTRE v4 source final store contains a symlink.")
    if all(present):
        return load_source_streams(
            destination,
            expected_config_hash=resolved_config_hash,
            expected_generation_lock_hash=generation_lock.generation_lock_hash,
            expected_attempt_id=attempt,
            test_mode=test_mode,
        )
    if any(present):
        raise ProtocolError("SCEPTRE v4 source final store is an unsafe partial state.")

    checkpoint_root = destination / CHECKPOINT_DIRECTORY
    validate_checkpoint_directory(checkpoint_root, geometry)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(
        config,
        generation_lock,
        expert_bank_root=bank_root,
        attempt_id=attempt,
        keys=keys,
        geometry=geometry,
        checkpoint_root=checkpoint_root,
    )
    completed: dict[tuple[str, int], Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        checkpoint = load_checkpoint_if_complete(task, geometry=geometry)
        if checkpoint is None:
            pending.append(task)
        else:
            completed[task_key(task)] = checkpoint

    if pending:
        if test_mode is None:
            results = execute_gpu_tasks(pending)
        else:
            results = tuple(
                execute_injected_task(
                    task,
                    geometry=geometry,
                    generate_block=test_mode.generate_block,
                )
                for task in pending
            )
        assert_parent_cuda_free()
        for result in results:
            key = (str(result["source_center"]), int(result["training_seed"]))
            task = next(task for task in pending if task_key(task) == key)
            loaded = load_checkpoint_if_complete(task, geometry=geometry)
            if loaded is None or loaded.get("checkpoint_sha256") != result.get(
                "checkpoint_sha256"
            ):
                raise ProtocolError("SCEPTRE v4 source worker checkpoint return drifted.")
            completed[key] = loaded

    if len(completed) != geometry.task_count:
        raise ProtocolError("SCEPTRE v4 source checkpoint coverage is incomplete.")
    store = publish_source_store(
        destination,
        attempt_id=attempt,
        config_hash=resolved_config_hash,
        generation_lock_hash=generation_lock.generation_lock_hash,
        tasks=tasks,
        completed=completed,
        geometry=geometry,
        test_mode=test_mode,
    )
    validate_checkpoint_directory(checkpoint_root, geometry)
    shutil.rmtree(checkpoint_root)
    return store


__all__ = ("materialize_source_streams",)
