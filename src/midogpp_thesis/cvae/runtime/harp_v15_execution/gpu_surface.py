"""Neutral frozen-bank source-stream materialization.

Only frozen expert and GenerationLock primitives are imported.  No diagnostic,
routing, support-utility, or label-access module is reachable from this file.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import gc
from itertools import product
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import sys
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...expert_bank.uniform_b_v2_promotion.serialization import (
    load_routing_authorized_expert,
)
from ...generation.contracts import COMMON_OUTPUT_DIM, GenerationLock, SourceGenerationKey
from ...generation.generation import generate_source_block, source_generation_plan
from ...protocol import ProtocolError
from ...routing.variational_compatibility import (
    ENERGY_SEMANTICS,
    score_variational_compatibility,
)
from ...routing.harp_protocol import canonical_hash
from ..artifact_io import atomic_json, read_json, sha256_array, sha256_file
from ..bounded_futures import execute_bounded
from .resident_stream_contracts import (
    CHECKPOINT_DIRECTORY,
    COMPATIBILITY_MEMBER,
    EXPECTED_STREAM_COUNT,
    EXPECTED_TASK_COUNT,
    ResidentExpertStreamCache,
    ResidentExpertStreamRecord,
    SOURCE_ARRAY_MEMBER,
    SOURCE_INDEX_MEMBER,
    SOURCE_LOCK_MEMBER,
    SOURCE_ROWS_PER_CLASS,
    source_block_sha256,
)
from .resident_stream_store import (
    load_resident_expert_streams,
    stage_resident_expert_streams,
)
from .json_payloads import plain_json_mapping


GENERATION_DEVICES = ("cuda:0", "cuda:1")


class ResidentExpertConfig(Protocol):
    contract_hash: str
    expert_bank_root: Path
    runtime: Mapping[str, object]


def materialize_resident_expert_streams(
    config: ResidentExpertConfig,
    generation_lock: GenerationLock,
    *,
    root: Path,
    support_binding: Mapping[str, object],
) -> ResidentExpertStreamCache:
    """Generate all 81 streams through two persistent one-process GPU pools."""

    _assert_runtime(config.runtime)
    support = _validate_support_binding(support_binding)
    _cleanup_final_atomic_temps(root)
    array_path = root / SOURCE_ARRAY_MEMBER
    index_path = root / SOURCE_INDEX_MEMBER
    lock_path = root / SOURCE_LOCK_MEMBER
    compatibility_path = root / COMPATIBILITY_MEMBER
    final_members = (array_path, index_path, compatibility_path, lock_path)
    present = tuple(path.is_file() for path in final_members)
    if any(path.is_symlink() for path in final_members):
        raise ProtocolError("HARP v15 resident expert final bundle contains a symlink.")
    if all(present):
        cache = load_resident_expert_streams(
            root,
            expected_config_hash=config.contract_hash,
            expected_generation_lock_hash=generation_lock.generation_lock_hash,
            expected_support_binding_hash=str(support["support_binding_hash"]),
        )
        _cleanup_completed_checkpoint_remnants(
            config, generation_lock, root=root, cache=cache
        )
        return cache
    if present not in {
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (True, True, True, False),
    }:
        raise ProtocolError("HARP v15 resident expert final bundle is an unsafe partial state.")

    checkpoint_root = root / CHECKPOINT_DIRECTORY
    _validate_checkpoint_tree(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    tasks = _build_tasks(config, generation_lock, checkpoint_root, support=support)
    completed: dict[tuple[str, int], Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        path = Path(str(task["checkpoint_path"]))
        if path.is_file():
            completed[_task_key(task)] = _load_checkpoint(path, task=task)
        else:
            pending.append(task)
    pending_by_key = {_task_key(task): task for task in pending}
    max_inflight = int(
        config.runtime.get("bounded_inflight_batches_per_gpu", max(1, len(pending)))
    )
    for payload in _execute_generation_tasks(
        pending, max_inflight_per_device=max_inflight
    ):
        key = (str(payload["source_center"]), int(payload["training_seed"]))
        task = pending_by_key[key]
        verified = _load_checkpoint(Path(str(task["checkpoint_path"])), task=task)
        if verified.get("checkpoint_hash") != payload.get("checkpoint_hash"):
            raise ProtocolError("HARP v15 resident expert checkpoint return drifted.")
        completed[key] = verified
        print(
            f"[fixed-bank:source-streams] source jobs {len(completed)}/{len(tasks)}",
            flush=True,
        )
    if len(completed) != EXPECTED_TASK_COUNT:
        raise ProtocolError("HARP v15 resident expert checkpoint coverage is incomplete.")

    records = _materialize_array(array_path, tasks=tasks, completed=completed)
    index_unhashed = {
        "schema_version": "midogpp_harp_v15_resident_expert_stream_index_v1",
        "config_contract_hash": config.contract_hash,
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "records": [record.to_payload() for record in records],
        "stream_count": len(records),
        "labels_consumed": False,
        "target_embeddings_consumed": False,
    }
    index = {**index_unhashed, "source_stream_index_hash": stable_hash(index_unhashed)}
    _persist_or_validate_json(index_path, index)
    compatibility_body = {
        "schema_version": "midogpp_harp_v15_role_qualified_compatibility_surface_v2",
        "support_binding": support,
        "support_binding_hash": support["support_binding_hash"],
        "training_seeds": list(TRAINING_SEEDS),
        "energy_semantics": ENERGY_SEMANTICS,
        "replicas": [
            {
                "source_center": str(task["source_center"]),
                "training_seed": int(task["training_seed"]),
                "expert_lock_hash": completed[_task_key(task)]["records"][0][
                    "expert_lock_hash"
                ],
                "checkpoint_sha256": completed[_task_key(task)][
                    "expert_checkpoint_sha256"
                ],
                "source_frame_hash": completed[_task_key(task)]["source_frame_hash"],
                "sampler_state_hash": completed[_task_key(task)]["sampler_state_hash"],
                "contexts": completed[_task_key(task)]["compatibility_contexts"],
                "compatibility_checkpoint_hash": completed[_task_key(task)][
                    "compatibility_hash"
                ],
            }
            for task in tasks
        ],
        "all_replicas_used_without_selection": True,
        "computed_while_expert_resident": True,
        "exact_nelbo": False,
        "labels_consumed": False,
        "source_train_embeddings_consumed": True,
        "target_test_embeddings_consumed": True,
        "evaluation_labels_consumed": False,
        "target_compatibility_is_case_local": True,
    }
    compatibility = {
        **compatibility_body,
        "compatibility_hash": canonical_hash(compatibility_body),
    }
    _persist_or_validate_json(compatibility_path, compatibility)
    lock_unhashed = {
        "schema_version": "midogpp_harp_v15_resident_expert_stream_lock_v1",
        "status": "COMPLETE_LABEL_FREE_RESIDENT_EXPERT_STREAMS",
        "config_contract_hash": config.contract_hash,
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "source_array_sha256": sha256_file(array_path),
        "source_stream_index_sha256": sha256_file(index_path),
        "source_stream_index_hash": index["source_stream_index_hash"],
        "support_compatibility_sha256": sha256_file(compatibility_path),
        "support_compatibility_hash": compatibility["compatibility_hash"],
        "support_binding_hash": support["support_binding_hash"],
        "stream_count": len(records),
        "rows_per_class": SOURCE_ROWS_PER_CLASS,
        "expert_bank_updated": False,
        "source_experts_updated": False,
        "labels_consumed": False,
        "support_embeddings_consumed": True,
        "target_test_embeddings_consumed_for_case_local_compatibility": True,
        "evaluation_labels_consumed_for_compatibility": False,
        "tf32_disabled": True,
        "amp_disabled": True,
        "float32_store": True,
    }
    lock = {**lock_unhashed, "source_stream_lock_hash": stable_hash(lock_unhashed)}
    _persist_or_validate_json(lock_path, lock)
    cache = load_resident_expert_streams(
        root,
        expected_config_hash=config.contract_hash,
        expected_generation_lock_hash=generation_lock.generation_lock_hash,
        expected_support_binding_hash=str(support["support_binding_hash"]),
    )
    _validate_checkpoint_tree(checkpoint_root)
    shutil.rmtree(checkpoint_root)
    return cache


def _assert_runtime(runtime: Mapping[str, object]) -> None:
    if (
        tuple(runtime.get("generation_devices", ())) != GENERATION_DEVICES
        or int(runtime.get("source_workers_per_device", -1)) != 1
        or int(runtime.get("generation_workers_per_device", -1)) != 1
        or runtime.get("persistent_source_workers") is not True
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("parent_cuda_context_forbidden") is not True
        or runtime.get("tf32_enabled") is not False
        or runtime.get("amp_enabled") is not False
        or runtime.get("generated_cache_format") != "float32_npy_memmap"
        or int(runtime.get("source_prefix_rows_per_class", -1))
        != SOURCE_ROWS_PER_CLASS
        or (
            "bounded_inflight_batches_per_gpu" in runtime
            and (
                type(runtime.get("bounded_inflight_batches_per_gpu")) is not int
                or int(runtime["bounded_inflight_batches_per_gpu"]) < 1
            )
        )
    ):
        raise ProtocolError("HARP v15 resident expert generation requires two exact float32 GPU streams.")
    torch_module = sys.modules.get("torch")
    if (
        torch_module is not None
        and getattr(torch_module, "cuda", None) is not None
        and torch_module.cuda.is_initialized()
    ):
        raise ProtocolError("HARP v15 resident expert parent process must remain CUDA-free.")


def _validate_checkpoint_tree(directory: Path) -> None:
    if not directory.exists():
        if directory.is_symlink():
            raise ProtocolError("HARP v15 resident expert checkpoint root is a dangling symlink.")
        return
    if directory.is_symlink() or not directory.is_dir():
        raise ProtocolError("HARP v15 resident expert checkpoint root is unsafe.")
    centers = r"(?:0|1|2|3|5|6|7|8|9)"
    seeds = r"(?:17|42|101)"
    member_pattern = rf"source_{centers}_train_{seeds}\.(?:json|npy)"
    observed: set[str] = set()
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("HARP v15 resident expert checkpoint tree contains an unsafe member.")
        match = re.fullmatch(r"(?P<base>.+)\.[1-9][0-9]*\.tmp", path.name)
        if match:
            if not re.fullmatch(member_pattern, match.group("base")):
                raise ProtocolError("HARP v15 resident expert checkpoint has an unknown atomic temp.")
            path.unlink()
            continue
        if not re.fullmatch(member_pattern, path.name):
            raise ProtocolError("HARP v15 resident expert checkpoint tree contains an unknown member.")
        observed.add(path.name)
    stems: dict[str, set[str]] = {}
    for member in observed:
        stem, suffix = member.rsplit(".", 1)
        stems.setdefault(stem, set()).add(suffix)
    if any(suffixes not in ({"npy"}, {"json", "npy"}) for suffixes in stems.values()):
        raise ProtocolError("HARP v15 resident expert checkpoint pair is unsafe.")


def _cleanup_final_atomic_temps(root: Path) -> None:
    """Remove only exact crash remnants for the ordered final source trio."""

    for member in (
        SOURCE_ARRAY_MEMBER,
        SOURCE_INDEX_MEMBER,
        COMPATIBILITY_MEMBER,
        SOURCE_LOCK_MEMBER,
    ):
        path = root / member
        parent = path.parent
        if not parent.exists():
            continue
        if parent.is_symlink() or not parent.is_dir():
            raise ProtocolError("HARP v15 resident expert final parent is unsafe.")
        pattern = re.compile(rf"{re.escape(path.name)}\.[1-9][0-9]*\.tmp")
        for candidate in parent.iterdir():
            if pattern.fullmatch(candidate.name):
                if candidate.is_symlink() or not candidate.is_file():
                    raise ProtocolError("HARP v15 resident expert final atomic temp is unsafe.")
                candidate.unlink()


def _cleanup_completed_checkpoint_remnants(
    config: ResidentExpertConfig,
    generation_lock: GenerationLock,
    *,
    root: Path,
    cache: ResidentExpertStreamCache,
) -> None:
    """Validate and remove only source checkpoints left after final publication.

    A crash between publishing the final trio and deleting task checkpoints may
    leave all 27 pairs, a validated subset from a partial ``rmtree``, or an
    array-only worker remnant.  Every remaining byte is rebound to the final
    cache before this exact package-owned directory is removed.
    """

    checkpoint_root = root / CHECKPOINT_DIRECTORY
    _validate_checkpoint_tree(checkpoint_root)
    if not checkpoint_root.exists():
        return
    raw_support = cache.compatibility_payload.get("support_binding")
    if not isinstance(raw_support, Mapping):
        raise ProtocolError("HARP v15 compatibility surface lacks its support binding.")
    tasks = _build_tasks(
        config,
        generation_lock,
        checkpoint_root,
        support=_validate_support_binding(raw_support),
    )
    expected_members = {
        Path(str(task["checkpoint_path"])).name: task for task in tasks
    } | {
        Path(str(task["array_path"])).name: task for task in tasks
    }
    observed = tuple(checkpoint_root.iterdir())
    if any(path.name not in expected_members for path in observed):
        raise ProtocolError("HARP v15 resident expert checkpoint remnant is not package-owned.")
    for task in tasks:
        json_path = Path(str(task["checkpoint_path"]))
        array_path = Path(str(task["array_path"]))
        if json_path.exists():
            if not array_path.is_file():
                raise ProtocolError("HARP v15 resident expert checkpoint JSON lacks its array.")
            payload = _load_checkpoint(json_path, task=task)
            _assert_checkpoint_matches_final_cache(payload, task=task, cache=cache)
        elif array_path.exists():
            _assert_checkpoint_array_matches_final_cache(
                array_path, task=task, cache=cache
            )
    # Revalidate immediately before deletion so a partial or unsafe tree cannot
    # be normalized into an apparently clean completed cache.
    _validate_checkpoint_tree(checkpoint_root)
    shutil.rmtree(checkpoint_root)


def _assert_checkpoint_matches_final_cache(
    payload: Mapping[str, object],
    *,
    task: Mapping[str, object],
    cache: ResidentExpertStreamCache,
) -> None:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ProtocolError("HARP v15 resident expert checkpoint records are absent.")
    replicas = cache.compatibility_payload.get("replicas")
    if not isinstance(replicas, list):
        raise ProtocolError("HARP v15 compatibility replica inventory is absent.")
    matched = [
        row
        for row in replicas
        if isinstance(row, Mapping)
        and row.get("source_center") == task["source_center"]
        and row.get("training_seed") == task["training_seed"]
    ]
    if (
        len(matched) != 1
        or matched[0].get("compatibility_checkpoint_hash")
        != payload.get("compatibility_hash")
    ):
        raise ProtocolError(
            "HARP v15 compatibility checkpoint differs from the final surface."
        )
    for raw, key in zip(records, task["generation_keys"], strict=True):
        final = cache.by_key[(
            str(task["source_center"]),
            int(task["training_seed"]),
            int(key.generation_seed),
        )]
        if (
            not isinstance(raw, Mapping)
            or raw.get("stream_id") != final.stream_id
            or raw.get("expert_lock_hash") != final.expert_lock_hash
            or raw.get("output_sha256") != final.output_sha256
        ):
            raise ProtocolError(
                "HARP v15 resident expert checkpoint remnant differs from the final cache."
            )


def _assert_checkpoint_array_matches_final_cache(
    path: Path,
    *,
    task: Mapping[str, object],
    cache: ResidentExpertStreamCache,
) -> None:
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("HARP v15 resident expert checkpoint array is unreadable.") from exc
    expected_shape = (
        len(GENERATION_SEEDS),
        2 * SOURCE_ROWS_PER_CLASS,
        COMMON_OUTPUT_DIM,
    )
    if values.shape != expected_shape or values.dtype != np.float32:
        raise ProtocolError("HARP v15 resident expert checkpoint array geometry drifted.")
    for ordinal, key in enumerate(task["generation_keys"]):
        final = cache.by_key[(
            str(task["source_center"]),
            int(task["training_seed"]),
            int(key.generation_seed),
        )]
        if source_block_sha256(values[ordinal]) != final.output_sha256:
            raise ProtocolError(
                "HARP v15 resident expert checkpoint array differs from the final cache."
            )


def _persist_or_validate_json(path: Path, payload: Mapping[str, object]) -> None:
    """Publish one ordered final JSON member without rewriting existing bytes."""

    normalized = plain_json_mapping(payload)
    if path.is_symlink():
        raise ProtocolError("HARP v15 resident expert final JSON member is a symlink.")
    if path.exists():
        if not path.is_file() or read_json(path) != normalized:
            raise ProtocolError(
                "Existing HARP v15 resident expert final JSON differs; refusing repair."
            )
        return
    atomic_json(path, normalized)


_SUPPORT_FRAME_CACHE: dict[str, tuple[str, np.ndarray]] = {}
_SUPPORT_BINDING_CACHE: dict[str, Mapping[str, object]] = {}


def _validate_support_binding(raw: Mapping[str, object]) -> Mapping[str, object]:
    raw_hash = raw.get("support_binding_hash")
    if type(raw_hash) is str and raw_hash in _SUPPORT_BINDING_CACHE:
        cached = _SUPPORT_BINDING_CACHE[raw_hash]
        if dict(cached) != dict(raw):
            raise ProtocolError("HARP v15 process-local support binding drifted.")
        return cached
    body = {key: value for key, value in raw.items() if key != "support_binding_hash"}
    path = Path(str(raw.get("frame_array_path", "")))
    contexts = raw.get("contexts")
    digest = raw.get("frame_array_sha256")
    if (
        raw.get("schema_version")
        != "midogpp_harp_v15_role_qualified_label_free_binding_v2"
        or raw.get("support_binding_hash") != canonical_hash(body)
        or not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or type(digest) is not str
        or len(digest) != 64
        or sha256_file(path) != digest
        or not isinstance(contexts, list)
        or len(contexts) != 2 * len(CENTERS)
        or raw.get("source_role") != "target_train_support"
        or raw.get("target_role") != "target_test_evaluation"
        or raw.get("source_train_target_test_case_disjoint") is not True
        or raw.get("labels_present") is not False
        or raw.get("source_train_embeddings_included") is not True
        or raw.get("target_test_embeddings_included") is not True
        or raw.get("target_test_embeddings_case_local_only") is not True
        or raw.get("evaluation_labels_included") is not False
    ):
        raise ProtocolError("HARP v15 label-free support binding drifted.")
    observed: list[tuple[str, str]] = []
    for context in contexts:
        if not isinstance(context, Mapping):
            raise ProtocolError("HARP v15 support context is malformed.")
        center = str(context.get("center", ""))
        role = str(context.get("role", ""))
        start = context.get("frame_start")
        stop = context.get("frame_stop")
        case_ids = context.get("case_ids")
        if (
            center not in CENTERS
            or role not in {raw.get("source_role"), raw.get("target_role")}
            or (role, center) in observed
            or type(start) is not int
            or type(stop) is not int
            or not 0 <= start < stop
            or not isinstance(case_ids, list)
            or len(case_ids) != stop - start
            or any(type(value) is not str or not value for value in case_ids)
        ):
            raise ProtocolError("HARP v15 support context geometry drifted.")
        observed.append((role, center))
    expected_contexts = tuple(
        (role, center)
        for role in (str(raw["source_role"]), str(raw["target_role"]))
        for center in CENTERS
    )
    if tuple(observed) != expected_contexts:
        raise ProtocolError("HARP v15 role-qualified context order drifted.")
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("HARP v15 support frame is unreadable.") from exc
    maximum = max(int(context["frame_stop"]) for context in contexts)
    if (
        values.dtype != np.float32
        or values.ndim != 2
        or values.shape[1] != COMMON_OUTPUT_DIM
        or maximum > len(values)
    ):
        raise ProtocolError("HARP v15 support frame geometry drifted.")
    validated = MappingProxyType(dict(raw))
    _SUPPORT_BINDING_CACHE[str(raw_hash)] = validated
    return validated


def _support_frame(task: Mapping[str, object]) -> tuple[np.ndarray, Mapping[str, object]]:
    raw = task.get("support_binding")
    if not isinstance(raw, Mapping):
        raise ProtocolError("HARP v15 resident task lacks support binding.")
    support = _validate_support_binding(raw)
    path = str(support["frame_array_path"])
    digest = str(support["frame_array_sha256"])
    cached = _SUPPORT_FRAME_CACHE.get(path)
    if cached is None:
        values = np.load(Path(path), mmap_mode="r", allow_pickle=False)
        _SUPPORT_FRAME_CACHE[path] = (digest, values)
    else:
        cached_digest, values = cached
        if cached_digest != digest:
            raise ProtocolError("HARP v15 process-local support cache identity drifted.")
    return values, support


def _score_support_contexts(
    expert: object, task: Mapping[str, object]
) -> list[dict[str, object]]:
    values, support = _support_frame(task)
    output: list[dict[str, object]] = []
    for raw in support["contexts"]:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v15 support context is malformed.")
        start, stop = int(raw["frame_start"]), int(raw["frame_stop"])
        case_ids = tuple(str(value) for value in raw["case_ids"])
        energy = score_variational_compatibility(
            expert,
            np.asarray(values[start:stop], dtype=np.float32),
            case_ids,
        )
        per_case = [
            float(np.float32(energy.per_case[case_id]))
            for case_id in energy.case_order
        ]
        output.append(
            {
                "role": str(raw["role"]),
                "query_center": str(raw["center"]),
                "case_order": list(energy.case_order),
                "per_case_energy_float32": per_case,
                "case_equal_mean_float64": float(
                    np.mean(np.asarray(per_case, dtype=np.float64), dtype=np.float64)
                ),
                "row_count": stop - start,
                "case_count": len(per_case),
                "energy_semantics": energy.energy_semantics,
                "exact_nelbo": False,
                "labels_consumed": False,
            }
        )
    return output


def _build_tasks(
    config: ResidentExpertConfig,
    generation_lock: GenerationLock,
    checkpoint_root: Path,
    *,
    support: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    keys = tuple(source_generation_plan(generation_lock))
    by_key = {(key.source_center, key.training_seed, key.generation_seed): key for key in keys}
    if set(by_key) != set(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS)):
        raise ProtocolError("GenerationLock source grid drifted.")
    tasks: list[Mapping[str, object]] = []
    for ordinal, (source, training_seed) in enumerate(product(CENTERS, TRAINING_SEEDS)):
        stem = f"source_{source}_train_{training_seed}"
        task = {
            "schema_version": "midogpp_harp_v15_resident_expert_stream_task_v1",
            "task_ordinal": ordinal,
            "source_center": source,
            "training_seed": training_seed,
            "generation_keys": tuple(by_key[(source, training_seed, seed)] for seed in GENERATION_SEEDS),
            "device": GENERATION_DEVICES[ordinal % len(GENERATION_DEVICES)],
            "expert_bank_root": str(config.expert_bank_root),
            "checkpoint_path": str(checkpoint_root / f"{stem}.json"),
            "array_path": str(checkpoint_root / f"{stem}.npy"),
            "config_contract_hash": config.contract_hash,
            "generation_lock_hash": generation_lock.generation_lock_hash,
            "support_binding": dict(support),
            "support_binding_hash": support["support_binding_hash"],
            "labels_available": False,
            "amp_enabled": False,
            "tf32_enabled": False,
        }
        tasks.append(task)
    return tuple(tasks)


def _execute_generation_tasks(
    tasks: Sequence[Mapping[str, object]], *, max_inflight_per_device: int
) -> tuple[Mapping[str, object], ...]:
    if not tasks:
        return ()
    context = mp.get_context("spawn")
    executors = [ProcessPoolExecutor(max_workers=1, mp_context=context) for _ in GENERATION_DEVICES]
    try:
        bounded = execute_bounded(
            executors,
            tasks,
            _generate_task,
            executor_index=lambda task: GENERATION_DEVICES.index(str(task["device"])),
            max_inflight_per_executor=max_inflight_per_device,
        )
        if any(
            observed > max_inflight_per_device
            for observed in bounded.stats.max_inflight_by_executor
        ):
            raise ProtocolError("HARP v15 resident expert GPU submission bound drifted.")
        return bounded.values
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def _generate_task(task: Mapping[str, object]) -> dict[str, object]:
    keys = tuple(task["generation_keys"])
    device = str(task["device"])
    if (
        not all(isinstance(key, SourceGenerationKey) for key in keys)
        or task.get("labels_available") is not False
        or task.get("amp_enabled") is not False
        or task.get("tf32_enabled") is not False
    ):
        raise ProtocolError("HARP v15 resident expert worker input drifted.")
    if device.startswith("cuda"):
        import torch

        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.set_num_threads(1)
    expert = load_routing_authorized_expert(
        Path(str(task["expert_bank_root"])),
        source_center=str(task["source_center"]),
        training_seed=int(task["training_seed"]),
        device=device,
    )
    try:
        blocks = [
            generate_source_block(expert, key, per_class=SOURCE_ROWS_PER_CLASS, device=device)
            for key in keys
        ]
        values = np.ascontiguousarray(np.stack([block.embeddings for block in blocks]), dtype=np.float32)
        array_path = Path(str(task["array_path"]))
        array_path.parent.mkdir(parents=True, exist_ok=True)
        _persist_or_validate_checkpoint_array(array_path, values)
        records = [
            {
                "generation_seed": block.key.generation_seed,
                "stream_id": block.key.stream_id,
                "expert_lock_hash": block.key.expert_lock_hash,
                "output_sha256": block.output_sha256,
                "array_sha256": sha256_array(block.embeddings),
            }
            for block in blocks
        ]
        compatibility_contexts = _score_support_contexts(expert, task)
        compatibility_body = {
            "schema_version": "midogpp_harp_v15_resident_compatibility_checkpoint_v1",
            "source_center": task["source_center"],
            "training_seed": task["training_seed"],
            "support_binding_hash": task["support_binding_hash"],
            "energy_semantics": ENERGY_SEMANTICS,
            "contexts": compatibility_contexts,
            "expert_checkpoint_sha256": str(expert.checkpoint_hash),
            "source_frame_hash": str(expert.source_frame.state_hash),
            "sampler_state_hash": str(expert.sampler.state_hash),
            "exact_nelbo": False,
            "labels_consumed": False,
            "source_train_embeddings_consumed": True,
            "target_test_embeddings_consumed": True,
            "evaluation_labels_consumed": False,
        }
        compatibility_hash = canonical_hash(compatibility_body)
        unhashed = {
            "schema_version": "midogpp_harp_v15_resident_expert_stream_checkpoint_v1",
            "status": "COMPLETE",
            "config_contract_hash": task["config_contract_hash"],
            "generation_lock_hash": task["generation_lock_hash"],
            "task_ordinal": task["task_ordinal"],
            "source_center": task["source_center"],
            "training_seed": task["training_seed"],
            "device": device,
            "array_path": str(array_path),
            "array_file_sha256": sha256_file(array_path),
            "records": records,
            "compatibility_contexts": compatibility_contexts,
            "compatibility_hash": compatibility_hash,
            "expert_checkpoint_sha256": str(expert.checkpoint_hash),
            "source_frame_hash": str(expert.source_frame.state_hash),
            "sampler_state_hash": str(expert.sampler.state_hash),
            "support_binding_hash": task["support_binding_hash"],
            "compatibility_computed_while_expert_resident": True,
            "labels_consumed": False,
            "source_expert_updated": False,
            "tf32_disabled": True,
            "amp_disabled": True,
            "float32_outputs": True,
        }
        payload = {**unhashed, "checkpoint_hash": stable_hash(unhashed)}
        _persist_or_validate_json(Path(str(task["checkpoint_path"])), payload)
        return payload
    finally:
        del expert
        gc.collect()
        try:
            import torch

            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


def _persist_or_validate_checkpoint_array(
    path: Path, expected: np.ndarray
) -> None:
    """Preserve an exact array-only task crash predecessor; reject drift.

    Source generation is deterministic under the frozen generation key.  A
    worker interrupted after publishing its NPY but before its JSON checkpoint
    therefore recomputes the expected values, validates the existing NPY
    semantically, and publishes only the missing successor.  Existing bytes are
    never rewritten.
    """

    if path.is_symlink():
        raise ProtocolError("HARP v15 resident expert checkpoint array is a symlink.")
    if path.exists():
        if not path.is_file():
            raise ProtocolError("HARP v15 resident expert checkpoint array is not a file.")
        try:
            observed = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ProtocolError(
                "Existing HARP v15 resident expert checkpoint array is unreadable; refusing repair."
            ) from exc
        if (
            observed.dtype != expected.dtype
            or observed.shape != expected.shape
            or sha256_array(observed) != sha256_array(expected)
        ):
            raise ProtocolError(
                "Existing HARP v15 resident expert checkpoint array differs; refusing repair."
            )
        return
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, expected, allow_pickle=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_checkpoint(path: Path, *, task: Mapping[str, object]) -> Mapping[str, object]:
    payload = read_json(path)
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    array_path = Path(str(payload.get("array_path", "")))
    records = payload.get("records")
    compatibility_contexts = payload.get("compatibility_contexts")
    if (
        payload.get("checkpoint_hash") != stable_hash(unhashed)
        or payload.get("schema_version") != "midogpp_harp_v15_resident_expert_stream_checkpoint_v1"
        or payload.get("status") != "COMPLETE"
        or payload.get("config_contract_hash") != task["config_contract_hash"]
        or payload.get("generation_lock_hash") != task["generation_lock_hash"]
        or payload.get("task_ordinal") != task["task_ordinal"]
        or payload.get("source_center") != task["source_center"]
        or payload.get("training_seed") != task["training_seed"]
        or payload.get("device") != task["device"]
        or payload.get("support_binding_hash") != task["support_binding_hash"]
        or array_path != Path(str(task["array_path"]))
        or not array_path.is_file()
        or payload.get("array_file_sha256") != sha256_file(array_path)
        or not isinstance(records, list)
        or len(records) != len(GENERATION_SEEDS)
        or not isinstance(compatibility_contexts, list)
        or len(compatibility_contexts) != 2 * len(CENTERS)
        or payload.get("compatibility_hash")
        != canonical_hash(
            {
                "schema_version": "midogpp_harp_v15_resident_compatibility_checkpoint_v1",
                "source_center": task["source_center"],
                "training_seed": task["training_seed"],
                "support_binding_hash": task["support_binding_hash"],
                "energy_semantics": ENERGY_SEMANTICS,
                "contexts": compatibility_contexts,
                "expert_checkpoint_sha256": payload.get("expert_checkpoint_sha256"),
                "source_frame_hash": payload.get("source_frame_hash"),
                "sampler_state_hash": payload.get("sampler_state_hash"),
                "exact_nelbo": False,
                "labels_consumed": False,
                "source_train_embeddings_consumed": True,
                "target_test_embeddings_consumed": True,
                "evaluation_labels_consumed": False,
            }
        )
        or payload.get("compatibility_computed_while_expert_resident") is not True
        or payload.get("labels_consumed") is not False
        or payload.get("source_expert_updated") is not False
        or payload.get("tf32_disabled") is not True
        or payload.get("amp_disabled") is not True
        or payload.get("float32_outputs") is not True
    ):
        raise ProtocolError("HARP v15 resident expert checkpoint failed validation.")
    values = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if values.shape != (len(GENERATION_SEEDS), 2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM) or values.dtype != np.float32:
        raise ProtocolError("HARP v15 resident expert checkpoint array geometry drifted.")
    for ordinal, (record, key) in enumerate(zip(records, task["generation_keys"], strict=True)):
        if (
            not isinstance(record, Mapping)
            or int(record.get("generation_seed", -1)) != key.generation_seed
            or record.get("stream_id") != key.stream_id
            or record.get("expert_lock_hash") != key.expert_lock_hash
            or record.get("array_sha256") != sha256_array(values[ordinal])
            or record.get("output_sha256")
            != source_block_sha256(values[ordinal])
        ):
            raise ProtocolError("HARP v15 resident expert checkpoint record drifted.")
    return payload


def _materialize_array(
    path: Path,
    *,
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[tuple[str, int], Mapping[str, object]],
) -> tuple[ResidentExpertStreamRecord, ...]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    records: list[ResidentExpertStreamRecord] = []
    cursor = 0
    try:
        values = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(
                EXPECTED_STREAM_COUNT,
                2 * SOURCE_ROWS_PER_CLASS,
                COMMON_OUTPUT_DIM,
            ),
        )
        for task in tasks:
            result = completed[_task_key(task)]
            task_values = np.load(
                Path(str(result["array_path"])),
                mmap_mode="r",
                allow_pickle=False,
            )
            for seed_ordinal, raw in enumerate(result["records"]):
                values[cursor] = task_values[seed_ordinal]
                records.append(
                    ResidentExpertStreamRecord(
                        block_ordinal=cursor,
                        source_center=str(task["source_center"]),
                        training_seed=int(task["training_seed"]),
                        generation_seed=int(raw["generation_seed"]),
                        stream_id=str(raw["stream_id"]),
                        expert_lock_hash=str(raw["expert_lock_hash"]),
                        rows_per_class=SOURCE_ROWS_PER_CLASS,
                        output_sha256=str(raw["output_sha256"]),
                    )
                )
                cursor += 1
        values.flush()
        del values
        if path.exists():
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != temporary.stat().st_size
                or sha256_file(path) != sha256_file(temporary)
            ):
                raise ProtocolError(
                    "Existing HARP v15 resident expert final array differs; refusing repair."
                )
            temporary.unlink()
        else:
            os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if cursor != EXPECTED_STREAM_COUNT:
        raise ProtocolError("HARP v15 resident expert stream materialization coverage drifted.")
    return tuple(records)


def _task_key(task: Mapping[str, object]) -> tuple[str, int]:
    return str(task["source_center"]), int(task["training_seed"])


__all__ = (
    "CHECKPOINT_DIRECTORY",
    "COMPATIBILITY_MEMBER",
    "EXPECTED_STREAM_COUNT",
    "ResidentExpertStreamCache",
    "ResidentExpertStreamRecord",
    "SOURCE_ARRAY_MEMBER",
    "SOURCE_INDEX_MEMBER",
    "SOURCE_LOCK_MEMBER",
    "SOURCE_ROWS_PER_CLASS",
    "load_resident_expert_streams",
    "materialize_resident_expert_streams",
    "source_block_sha256",
    "stage_resident_expert_streams",
)
