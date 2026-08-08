"""Resumable orchestration for the independent case-OOF source cache."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Mapping, Protocol

from ...generation.contracts import GenerationLock
from ...protocol import ProtocolError
from .artifact_io import atomic_write_json
from .source_cache_contracts import (
    EXPECTED_SOURCE_TASK_COUNT,
    GENERATION_DEVICES,
    SOURCE_BLOCK_ARRAY_MEMBER,
    SOURCE_BLOCK_INDEX_MEMBER,
    SOURCE_CACHE_LOCK_MEMBER,
    COMPATIBILITY_CASE_MEMBER,
    SourceCache,
)
from .source_cache_planning import (
    build_source_tasks,
    execute_pending_tasks,
    source_task_key,
    write_support_scratch,
)
from .source_cache_store import load_source_cache, materialize_source_products
from .source_cache_validation import (
    build_source_cache_lock,
    validate_source_cache_lock,
)
from .source_cache_worker import load_generation_checkpoint


class SourceExecutionConfig(Protocol):
    contract_hash: str
    runtime: Mapping[str, object]


def materialize_source_cache(
    config: SourceExecutionConfig,
    generation_lock: GenerationLock,
    frame: object,
    crossfit: object,
    *,
    root: Path,
) -> SourceCache:
    """Materialize with two persistent spawned GPU workers or resume safely."""

    if tuple(config.runtime.get("generation_devices", ())) != GENERATION_DEVICES:
        raise ProtocolError("Case-OOF requires exactly the two frozen GPU devices.")
    products = (
        root / SOURCE_BLOCK_ARRAY_MEMBER,
        root / SOURCE_BLOCK_INDEX_MEMBER,
        root / COMPATIBILITY_CASE_MEMBER,
        root / SOURCE_CACHE_LOCK_MEMBER,
    )
    if all(path.is_file() for path in products):
        cache = load_source_cache(root)
        validate_source_cache_lock(
            root,
            config=config,
            generation_lock=generation_lock,
            frame=frame,
            crossfit=crossfit,
            source_cache=cache,
        )
        shutil.rmtree(root / "checkpoints/source_cache", ignore_errors=True)
        return cache

    checkpoint_root = root / "checkpoints/source_cache"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    support_path = checkpoint_root / "support_embeddings.npy"
    support_index_path = checkpoint_root / "support_index.json"
    support = write_support_scratch(
        support_path,
        support_index_path,
        frame=frame,
        crossfit=crossfit,
    )
    tasks, key_map = build_source_tasks(
        config,
        generation_lock,
        checkpoint_root=checkpoint_root,
        support_array_path=support_path,
        support_index_path=support_index_path,
        support_scratch_hash=str(support["support_scratch_hash"]),
    )
    completed: dict[tuple[str, int], Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        checkpoint_path = Path(str(task["checkpoint_path"]))
        if checkpoint_path.is_file():
            completed[source_task_key(task)] = load_generation_checkpoint(
                checkpoint_path, task=task
            )
        else:
            pending.append(task)
    task_by_key = {source_task_key(task): task for task in pending}
    for finished_count, payload in enumerate(
        execute_pending_tasks(pending), start=1
    ):
        key = (str(payload["source_center"]), int(payload["training_seed"]))
        task = task_by_key[key]
        verified = load_generation_checkpoint(
            Path(str(task["checkpoint_path"])), task=task
        )
        if payload.get("checkpoint_hash") != verified.get("checkpoint_hash"):
            raise ProtocolError("Case-OOF source checkpoint return drifted.")
        completed[key] = verified
        print(
            f"[case-oof] source jobs {len(completed)}/{EXPECTED_SOURCE_TASK_COUNT} "
            f"(new {finished_count}/{len(pending)})",
            flush=True,
        )
    if len(completed) != EXPECTED_SOURCE_TASK_COUNT:
        raise ProtocolError("Case-OOF source checkpoint coverage is incomplete.")

    cache = materialize_source_products(root, completed=completed, key_map=key_map)
    lock = build_source_cache_lock(
        root,
        config=config,
        generation_lock=generation_lock,
        frame=frame,
        crossfit=crossfit,
        source_cache=cache,
    )
    atomic_write_json(root / SOURCE_CACHE_LOCK_MEMBER, lock)
    validate_source_cache_lock(
        root,
        config=config,
        generation_lock=generation_lock,
        frame=frame,
        crossfit=crossfit,
        source_cache=cache,
    )
    shutil.rmtree(checkpoint_root, ignore_errors=True)
    return cache


__all__ = ("SourceExecutionConfig", "materialize_source_cache")
