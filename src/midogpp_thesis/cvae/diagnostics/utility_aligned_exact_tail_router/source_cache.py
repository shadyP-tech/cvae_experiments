"""Public facade and resumable orchestration for the Stage-90 source cache."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from typing import Mapping, Protocol

from ...generation.contracts import GenerationLock
from ...protocol import ProtocolError
from .source_cache_contracts import (
    COMPONENT_ARRAY_MEMBER,
    COMPONENT_INDEX_COLUMNS,
    COMPONENT_INDEX_MEMBER,
    EXPECTED_COMPONENT_RECORD_COUNT,
    EXPECTED_SOURCE_STREAM_COUNT,
    EXPECTED_SOURCE_TASK_COUNT,
    GENERATION_DEVICES,
    SOURCE_ARRAY_MEMBER,
    SOURCE_CACHE_LOCK_MEMBER,
    SOURCE_INDEX_COLUMNS,
    SOURCE_INDEX_MEMBER,
    SOURCE_ROWS_PER_CLASS,
    LabelFreeComponentRecord,
    SourceBlockRecord,
    SourceCache,
)
from .source_cache_planning import (
    build_source_tasks,
    execute_pending_tasks,
    source_task_key,
    write_support_scratch,
)
from .source_cache_store import (
    atomic_write_json,
    load_source_cache,
    materialize_source_products,
    read_json,
    sha256_file,
)
from .source_cache_validation import (
    build_source_cache_lock,
    validate_source_cache_inventory,
    validate_source_cache_lock,
)
from .source_cache_worker import load_generation_checkpoint


class SourceExecutionConfig(Protocol):
    contract_hash: str
    runtime: Mapping[str, object]


LOCAL_STAGE_DIRECTORY = (
    "midogpp_stage90_utility_aligned_exact_tail_router_v1/source_cache"
)


def materialize_source_cache(
    config: SourceExecutionConfig,
    generation_lock: GenerationLock,
    frame: object,
    partitions: object,
    *,
    root: Path,
) -> SourceCache:
    """Generate 81 independent streams with two persistent spawned workers."""

    if tuple(config.runtime.get("generation_devices", ())) != GENERATION_DEVICES:
        raise ProtocolError("Stage-90 source generation requires cuda:0 and cuda:1.")
    if (
        int(config.runtime.get("generation_workers_per_device", -1)) != 1
        or config.runtime.get("tf32_enabled") is not False
        or config.runtime.get("amp_enabled") is not False
    ):
        raise ProtocolError("Stage-90 source generation forbids mixed precision.")
    torch_module = sys.modules.get("torch")
    if (
        torch_module is not None
        and getattr(torch_module, "cuda", None) is not None
        and torch_module.cuda.is_initialized()
    ):
        raise ProtocolError("Stage-90 parent process must remain CUDA-free.")
    products = (
        root / SOURCE_ARRAY_MEMBER,
        root / COMPONENT_ARRAY_MEMBER,
        root / SOURCE_INDEX_MEMBER,
        root / COMPONENT_INDEX_MEMBER,
        root / SOURCE_CACHE_LOCK_MEMBER,
    )
    if all(path.is_file() for path in products):
        cache = load_source_cache(root)
        validate_source_cache_lock(
            root,
            config=config,
            generation_lock=generation_lock,
            frame=frame,
            partitions=partitions,
            source_cache=cache,
        )
        return cache

    checkpoint_root = root / "checkpoints/utility_aligned_source_cache"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    support_array_path = checkpoint_root / "fixed_support_embeddings.npy"
    support_index_path = checkpoint_root / "fixed_support_index.json"
    support = write_support_scratch(
        support_array_path,
        support_index_path,
        frame=frame,
        partitions=partitions,
        expected_support_case_count=_support_case_count(config),
    )
    tasks, key_map = build_source_tasks(
        config,
        generation_lock,
        checkpoint_root=checkpoint_root,
        support_array_path=support_array_path,
        support_index_path=support_index_path,
        support_scratch_hash=str(support["support_scratch_hash"]),
    )
    completed: dict[tuple[str, int], Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        path = Path(str(task["checkpoint_path"]))
        if path.is_file():
            completed[source_task_key(task)] = load_generation_checkpoint(
                path, task=task
            )
        else:
            pending.append(task)
    task_by_key = {source_task_key(task): task for task in pending}
    for new_count, payload in enumerate(execute_pending_tasks(pending), start=1):
        key = (str(payload["source_center"]), int(payload["training_seed"]))
        task = task_by_key[key]
        verified = load_generation_checkpoint(
            Path(str(task["checkpoint_path"])), task=task
        )
        if payload.get("checkpoint_hash") != verified.get("checkpoint_hash"):
            raise ProtocolError("Stage-90 source checkpoint return drifted.")
        completed[key] = verified
        print(
            f"[utility-exact-tail] source jobs {len(completed)}/{EXPECTED_SOURCE_TASK_COUNT} "
            f"(new {new_count}/{len(pending)})",
            flush=True,
        )
    if len(completed) != EXPECTED_SOURCE_TASK_COUNT:
        raise ProtocolError("Stage-90 source checkpoint coverage is incomplete.")

    cache = materialize_source_products(
        root,
        completed=completed,
        key_map=key_map,
        support_scratch_hash=str(support["support_scratch_hash"]),
        support_row_count=int(support["shape"][0]),
    )
    lock = build_source_cache_lock(
        root,
        config=config,
        generation_lock=generation_lock,
        frame=frame,
        partitions=partitions,
        source_cache=cache,
    )
    atomic_write_json(root / SOURCE_CACHE_LOCK_MEMBER, lock)
    validate_source_cache_lock(
        root,
        config=config,
        generation_lock=generation_lock,
        frame=frame,
        partitions=partitions,
        source_cache=cache,
    )
    # Canonical memmaps and their lock make worker duplicates redundant.
    shutil.rmtree(checkpoint_root, ignore_errors=True)
    return cache


def stage_source_cache_for_cpu(
    cache: SourceCache,
    *,
    scratch_root: Path,
    canonical_root: Path,
    local_stage_directory: str = LOCAL_STAGE_DIRECTORY,
) -> SourceCache:
    """Atomically stage the finalized cache on experiment-scoped local storage.

    The canonical artifact remains authoritative.  A copied lock is published
    last and every staged byte is checked against that canonical lock before a
    :class:`SourceCache` is returned.
    """

    canonical = Path(canonical_root).resolve()
    if cache.root.resolve() != canonical:
        raise ProtocolError("Stage-90 local staging received another canonical root.")
    canonical_lock = _validate_self_contained_cache(cache, canonical)
    destination = Path(scratch_root).resolve() / local_stage_directory
    if destination == canonical:
        print("[utility-exact-tail] source cache already on CPU scratch", flush=True)
        return cache
    destination.mkdir(parents=True, exist_ok=True)
    required = (
        SOURCE_ARRAY_MEMBER,
        COMPONENT_ARRAY_MEMBER,
        SOURCE_INDEX_MEMBER,
        COMPONENT_INDEX_MEMBER,
        SOURCE_CACHE_LOCK_MEMBER,
    )
    if all((destination / member).is_file() for member in required):
        try:
            staged = load_source_cache(destination)
            _validate_self_contained_cache(
                staged, destination, validate_inventory=False
            )
            if read_json(destination / SOURCE_CACHE_LOCK_MEMBER) == canonical_lock:
                print(
                    f"[utility-exact-tail] reused local CPU source cache: {destination}",
                    flush=True,
                )
                return staged
        except ProtocolError:
            # The local copy is disposable.  Replace only the five explicitly
            # named experiment members; never remove the shared scratch root.
            pass
    digest_fields = {
        SOURCE_ARRAY_MEMBER: "source_array_sha256",
        COMPONENT_ARRAY_MEMBER: "component_array_sha256",
        SOURCE_INDEX_MEMBER: "source_index_sha256",
        COMPONENT_INDEX_MEMBER: "component_index_sha256",
    }
    for member in required[:-1]:
        _atomic_copy(
            canonical / member,
            destination / member,
            expected_sha256=str(canonical_lock[digest_fields[member]]),
        )
    _atomic_copy(
        canonical / SOURCE_CACHE_LOCK_MEMBER,
        destination / SOURCE_CACHE_LOCK_MEMBER,
        expected_sha256=sha256_file(canonical / SOURCE_CACHE_LOCK_MEMBER),
    )
    staged = load_source_cache(destination)
    if read_json(destination / SOURCE_CACHE_LOCK_MEMBER) != canonical_lock:
        raise ProtocolError("Stage-90 staged source-cache lock differs from canonical.")
    _validate_self_contained_cache(staged, destination, validate_inventory=False)
    print(
        f"[utility-exact-tail] staged CPU source cache on local storage: {destination}",
        flush=True,
    )
    return staged


def _validate_self_contained_cache(
    cache: SourceCache, root: Path, *, validate_inventory: bool = True
) -> Mapping[str, object]:
    lock = read_json(root / SOURCE_CACHE_LOCK_MEMBER)
    expected_members = {
        SOURCE_ARRAY_MEMBER: "source_array_sha256",
        COMPONENT_ARRAY_MEMBER: "component_array_sha256",
        SOURCE_INDEX_MEMBER: "source_index_sha256",
        COMPONENT_INDEX_MEMBER: "component_index_sha256",
    }
    if (
        lock.get("schema_version")
        != "midogpp_stage90_utility_aligned_source_cache_lock_v1"
        or lock.get("status") not in {
            "COMPLETE_LABEL_FREE_FIXED_TWO_CASE_SUPPORT_CACHE",
            "COMPLETE_LABEL_FREE_FIXED_SUPPORT_CACHE",
        }
        or lock.get("source_cache_hash") != cache.source_cache_hash
        or lock.get("support_scratch_hash") != cache.support_scratch_hash
        or lock.get("source_stream_count") != EXPECTED_SOURCE_STREAM_COUNT
        or lock.get("component_record_count") != EXPECTED_COMPONENT_RECORD_COUNT
        or lock.get("labels_consumed") is not False
        or lock.get("prior_stage90_cache_consumed") is not False
    ):
        raise ProtocolError("Stage-90 staged source-cache semantic lock drifted.")
    for member, field in expected_members.items():
        path = root / member
        if not path.is_file() or sha256_file(path) != lock.get(field):
            raise ProtocolError("Stage-90 staged source-cache member hash drifted.")
    if validate_inventory:
        validate_source_cache_inventory(
            cache,
            expected_support_case_count=int(
                lock.get("fixed_support_case_count_per_center", -1)
            ),
        )
    return lock


def _atomic_copy(
    source: Path, destination: Path, *, expected_sha256: str
) -> None:
    if not source.is_file():
        raise ProtocolError(f"Stage-90 canonical cache member is absent: {source}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != expected_sha256:
        raise ProtocolError("Stage-90 local source-cache copy changed bytes.")
    os.replace(temporary, destination)


def _support_case_count(config: object) -> int:
    direct = getattr(config, "fixed_support_case_count_per_center", None)
    if direct is not None:
        return int(direct)
    protocol = getattr(config, "protocol", {})
    if isinstance(protocol, Mapping):
        return int(protocol.get("fixed_support_case_count_per_center", 2))
    return 2


__all__ = (
    "COMPONENT_ARRAY_MEMBER",
    "COMPONENT_INDEX_COLUMNS",
    "COMPONENT_INDEX_MEMBER",
    "EXPECTED_COMPONENT_RECORD_COUNT",
    "EXPECTED_SOURCE_STREAM_COUNT",
    "EXPECTED_SOURCE_TASK_COUNT",
    "GENERATION_DEVICES",
    "SOURCE_ARRAY_MEMBER",
    "SOURCE_CACHE_LOCK_MEMBER",
    "SOURCE_INDEX_COLUMNS",
    "SOURCE_INDEX_MEMBER",
    "SOURCE_ROWS_PER_CLASS",
    "LabelFreeComponentRecord",
    "SourceBlockRecord",
    "SourceCache",
    "build_source_cache_lock",
    "load_source_cache",
    "materialize_source_cache",
    "stage_source_cache_for_cpu",
    "validate_source_cache_inventory",
    "validate_source_cache_lock",
)
