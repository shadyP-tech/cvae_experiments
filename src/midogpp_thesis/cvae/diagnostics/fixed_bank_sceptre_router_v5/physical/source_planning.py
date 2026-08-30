"""Fail-closed planning and runtime checks for v5 source generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
import sys

from midogpp_thesis.cvae.generation.contracts import (
    SOURCE_BUDGET_PER_CLASS,
    TOTAL_PER_CLASS,
    GenerationLock,
    SourceGenerationKey,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from .source_contracts import (
    PRODUCTION_SOURCE_GEOMETRY,
    SOURCE_ARRAY_MEMBER,
    SOURCE_INDEX_MEMBER,
    SOURCE_RECEIPT_MEMBER,
    SourceGeometry,
    SourceRuntimeConfig,
    SourceRuntimeTestMode,
)
from .source_hashing import canonical_sha256
from .worker_runtime import GPU_DEVICES


def geometry_for(test_mode: SourceRuntimeTestMode | None) -> SourceGeometry:
    return PRODUCTION_SOURCE_GEOMETRY if test_mode is None else test_mode.geometry


def config_hash(config: object) -> str:
    value = getattr(config, "config_hash", getattr(config, "contract_hash", ""))
    text = str(value)
    if not text:
        raise ProtocolError("SCEPTRE v5 source config hash is absent.")
    return text


def resolve_attempt_id(
    config: object,
    *,
    explicit: str | None,
    root: Path,
    synthetic: bool,
) -> str:
    raw = explicit if explicit is not None else getattr(config, "attempt_id", None)
    text = "" if raw is None else str(raw).strip()
    if not text and synthetic:
        text = canonical_sha256(
            {
                "schema_version": "sceptre_v5_synthetic_physical_attempt_v1",
                "root": str(root.resolve()),
                "config_hash": config_hash(config),
            }
        )
    if not text or len(text) > 256 or any(character.isspace() for character in text):
        raise ProtocolError("SCEPTRE v5 physical attempt identity is absent or malformed.")
    return text


def assert_owned_root(root: Path) -> None:
    if root.is_symlink():
        raise ProtocolError("SCEPTRE v5 source root is a symlink.")
    if root.exists() and not root.is_dir():
        raise ProtocolError("SCEPTRE v5 source root is not a directory.")
    root.mkdir(parents=True, exist_ok=True)


def final_paths(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / SOURCE_ARRAY_MEMBER,
        root / SOURCE_INDEX_MEMBER,
        root / SOURCE_RECEIPT_MEMBER,
    )


def assert_parent_cuda_free() -> None:
    torch_module = sys.modules.get("torch")
    cuda = getattr(torch_module, "cuda", None) if torch_module is not None else None
    if cuda is not None and bool(cuda.is_initialized()):
        raise ProtocolError("SCEPTRE v5 source parent process must remain CUDA-free.")


def assert_production_runtime(runtime: Mapping[str, object]) -> None:
    if (
        tuple(runtime.get("gpu_devices", ())) != GPU_DEVICES
        or int(runtime.get("persistent_gpu_generation_workers", -1)) != 2
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("top_level_spawn_pool_only") is not True
        or int(runtime.get("blas_threads_per_worker", -1)) != 1
        or int(runtime.get("native_threads_per_worker", -1)) != 1
    ):
        raise ProtocolError("SCEPTRE v5 source workstation topology drifted.")


def resolve_expert_bank_root(
    config: object,
    *,
    explicit: Path | None,
    synthetic: bool,
    owned_root: Path,
) -> Path:
    raw = explicit if explicit is not None else getattr(config, "expert_bank_root", None)
    if raw is None:
        if not synthetic:
            raise ProtocolError("SCEPTRE v5 expert-bank root is absent.")
        raw = owned_root / "__synthetic_expert_bank_unopened__"
    path = Path(raw).resolve()
    if path.is_symlink() or (not synthetic and (not path.is_dir())):
        raise ProtocolError("SCEPTRE v5 expert-bank root is absent or unsafe.")
    return path


def generation_keys(
    generation_lock: GenerationLock,
    *,
    test_mode: SourceRuntimeTestMode | None,
) -> tuple[object, ...]:
    if test_mode is not None:
        return tuple(test_mode.generation_keys)
    from midogpp_thesis.cvae.generation.generation import source_generation_plan

    return tuple(source_generation_plan(generation_lock))


def validate_generation_grid(
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
            raise ProtocolError("SCEPTRE v5 source generation key is malformed.") from exc
        if not stream_id or not expert_hash or identity in observed:
            raise ProtocolError("SCEPTRE v5 source generation key identity drifted.")
        if test_mode is None and (
            not isinstance(key, SourceGenerationKey)
            or int(key.max_samples_per_class) != TOTAL_PER_CLASS
            or int(key.equal_union_prefix_per_class) != SOURCE_BUDGET_PER_CLASS
        ):
            raise ProtocolError("SCEPTRE v5 GenerationLock source budget drifted.")
        observed[identity] = key
    expected = set(
        product(
            geometry.centers,
            geometry.training_seeds,
            geometry.generation_seeds,
        )
    )
    if set(observed) != expected or len(keys) != geometry.stream_count:
        raise ProtocolError("SCEPTRE v5 GenerationLock source grid drifted.")
    if not str(getattr(generation_lock, "generation_lock_hash", "")):
        raise ProtocolError("SCEPTRE v5 GenerationLock hash is absent.")


def build_tasks(
    config: SourceRuntimeConfig,
    generation_lock: GenerationLock,
    *,
    expert_bank_root: Path,
    attempt_id: str,
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
        task_generation_keys = tuple(
            by_key[(source, training_seed, seed)]
            for seed in geometry.generation_seeds
        )
        task_identity = {
            "schema_version": "midogpp_sceptre_v5_physical_source_task_v1",
            "attempt_id": attempt_id,
            "task_ordinal": ordinal,
            "source_center": source,
            "training_seed": training_seed,
            "generation_seeds": list(geometry.generation_seeds),
            "stream_ids": [
                str(getattr(key, "stream_id")) for key in task_generation_keys
            ],
            "expert_lock_hashes": [
                str(getattr(key, "expert_lock_hash")) for key in task_generation_keys
            ],
            "device": GPU_DEVICES[ordinal % 2],
            "config_hash": config_hash(config),
            "generation_lock_hash": generation_lock.generation_lock_hash,
            "geometry": geometry.to_payload(),
            "expert_bank_root": str(expert_bank_root),
            "target_cache_available": False,
            "manifest_available": False,
            "outcomes_available": False,
            "amp_enabled": False,
            "tf32_enabled": False,
        }
        tasks.append(
            {
                **task_identity,
                "task_sha256": canonical_sha256(task_identity),
                "generation_keys": task_generation_keys,
                "checkpoint_array_path": str(checkpoint_root / f"{stem}.npy"),
                "checkpoint_json_path": str(checkpoint_root / f"{stem}.json"),
            }
        )
    if len(tasks) != geometry.task_count:
        raise ProtocolError("SCEPTRE v5 source task coverage drifted.")
    return tuple(tasks)


def task_identity(task: Mapping[str, object]) -> dict[str, object]:
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


def task_key(task: Mapping[str, object]) -> tuple[str, int]:
    return str(task["source_center"]), int(task["training_seed"])


__all__ = (
    "assert_owned_root",
    "assert_parent_cuda_free",
    "assert_production_runtime",
    "build_tasks",
    "config_hash",
    "final_paths",
    "generation_keys",
    "geometry_for",
    "resolve_attempt_id",
    "resolve_expert_bank_root",
    "task_identity",
    "task_key",
    "validate_generation_grid",
)
