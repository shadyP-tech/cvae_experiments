"""Adapters for the neutral frozen-source runtime and phase transition."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.frozen_source_streams import (
    FrozenSourceStreamCache,
    SOURCE_ARRAY_MEMBER,
    SOURCE_INDEX_MEMBER,
    SOURCE_LOCK_MEMBER,
    load_frozen_source_streams,
    materialize_frozen_source_streams,
    stage_frozen_source_streams,
)


SCRATCH_DIRECTORY = "utility_aligned_consumed_test_endpoint_router_v1"
LOCAL_SOURCE_DIRECTORY = "source_cache"


@dataclass(frozen=True)
class _FrozenSourceConfigAdapter:
    contract_hash: str
    expert_bank_root: Path
    runtime: Mapping[str, object]


@dataclass(frozen=True)
class StagedSourceCache:
    cache: FrozenSourceStreamCache
    canonical_root: Path
    scratch_root: Path

    @property
    def used_local_scratch(self) -> bool:
        return self.cache.root.resolve() != self.canonical_root.resolve()

    def report_payload(self) -> dict[str, object]:
        return {
            "canonical_root": str(self.canonical_root.resolve()),
            "active_root": str(self.cache.root.resolve()),
            "scratch_root": str(self.scratch_root.resolve()),
            "used_local_scratch": self.used_local_scratch,
            "source_stream_lock_hash": self.cache.lock_hash,
            "source_stream_count": len(self.cache.records),
            "hash_validated": True,
        }


def source_generation_runtime(runtime: Mapping[str, object]) -> dict[str, object]:
    """Translate the experiment topology into the neutral source API."""

    required = {
        "generation_devices": ["cuda:0", "cuda:1"],
        "generation_workers_per_device": 1,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_forbidden": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "source_stream_count": 81,
        "array_storage_dtype": "float32",
        "scientific_reduction_dtype": "float64",
    }
    if any(runtime.get(key) != value for key, value in required.items()):
        raise ProtocolError("Endpoint-router source-generation topology drifted.")
    return {
        **dict(runtime),
        "source_workers_per_device": 1,
        "persistent_source_workers": True,
        "generated_cache_format": "float32_npy_memmap",
        "source_job_count": 27,
        "source_prefix_rows_per_class": 270,
        "scientific_reductions_dtype": "float64",
    }


def materialize_source_cache(
    config: object,
    generation_lock: object,
    *,
    root: Path,
) -> FrozenSourceStreamCache:
    """Recompute or hash-validate all 81 experiment-owned source streams."""

    final_members = tuple(
        root / member
        for member in (SOURCE_ARRAY_MEMBER, SOURCE_INDEX_MEMBER, SOURCE_LOCK_MEMBER)
    )
    present = tuple(path.is_file() for path in final_members)
    if any(present) and not all(present):
        raise ProtocolError("Endpoint-router source-cache final surface is incomplete.")
    adapter = _FrozenSourceConfigAdapter(
        contract_hash=str(getattr(config, "contract_hash")),
        expert_bank_root=Path(getattr(config, "expert_bank_root")),
        runtime=source_generation_runtime(getattr(config, "runtime")),
    )
    cache = materialize_frozen_source_streams(adapter, generation_lock, root=root)
    shutil.rmtree(root / "checkpoints/frozen_source_streams", ignore_errors=True)
    return cache


def load_source_cache(
    config: object,
    generation_lock: object,
    *,
    root: Path,
) -> FrozenSourceStreamCache:
    return load_frozen_source_streams(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=str(
            getattr(generation_lock, "generation_lock_hash")
        ),
    )


def stage_source_cache_for_cpu(
    cache: FrozenSourceStreamCache,
    *,
    artifact_root: Path,
    runtime: Mapping[str, object],
) -> StagedSourceCache:
    """Hash-copy source memmaps to local scratch, falling back to artifact disk."""

    scratch = _select_scratch_root(artifact_root, runtime=runtime)
    staged = stage_frozen_source_streams(
        cache,
        scratch_root=scratch,
        canonical_root=artifact_root,
        local_directory=LOCAL_SOURCE_DIRECTORY,
    )
    return StagedSourceCache(
        cache=staged,
        canonical_root=artifact_root,
        scratch_root=scratch,
    )


def enter_cuda_free_cpu_phase() -> None:
    """Seal the parent process against CUDA before spawning CPU classifiers."""

    torch_module = sys.modules.get("torch")
    if (
        torch_module is not None
        and getattr(torch_module, "cuda", None) is not None
        and torch_module.cuda.is_initialized()
    ):
        raise ProtocolError("Endpoint-router parent CUDA context was initialized.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"


def cleanup_staged_source_cache(staged: StagedSourceCache) -> None:
    """Remove only experiment-owned scratch after canonical artifacts are sealed."""

    if staged.used_local_scratch:
        _assert_owned_scratch(staged.cache.root, staged.scratch_root)
        shutil.rmtree(staged.cache.root, ignore_errors=True)
    if staged.scratch_root.name != SCRATCH_DIRECTORY or staged.scratch_root.is_symlink():
        raise ProtocolError("Endpoint-router scratch root ownership drifted.")
    try:
        staged.scratch_root.rmdir()
    except OSError:
        # A non-empty experiment scratch root is retained fail-closed so that
        # no unrecognized member is silently deleted.
        pass


def _select_scratch_root(
    artifact_root: Path, *, runtime: Mapping[str, object]
) -> Path:
    preferences = tuple(str(value) for value in runtime.get("scratch_preference", ()))
    if preferences != ("/data/local", "artifact_parent"):
        raise ProtocolError("Endpoint-router scratch preference drifted.")
    local_parent = Path(preferences[0])
    if local_parent.is_dir() and os.access(local_parent, os.W_OK):
        root = local_parent / SCRATCH_DIRECTORY
    else:
        root = artifact_root.parent / f".{SCRATCH_DIRECTORY}.scratch"
    if root.is_symlink():
        raise ProtocolError("Endpoint-router scratch root cannot be a symlink.")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.resolve() in {Path("/"), artifact_root.resolve()}:
        raise ProtocolError("Endpoint-router scratch root is unsafe.")
    return root


def _assert_owned_scratch(active_root: Path, scratch_root: Path) -> None:
    active = active_root
    scratch = scratch_root
    if active.is_symlink() or scratch.is_symlink():
        raise ProtocolError("Endpoint-router scratch cleanup rejected a symlink.")
    try:
        relative = active.resolve().relative_to(scratch.resolve())
    except ValueError as exc:
        raise ProtocolError("Endpoint-router scratch cleanup escaped its root.") from exc
    if relative != Path(LOCAL_SOURCE_DIRECTORY) or active.name != LOCAL_SOURCE_DIRECTORY:
        raise ProtocolError("Endpoint-router scratch cleanup target is not owned.")


__all__ = (
    "LOCAL_SOURCE_DIRECTORY",
    "SCRATCH_DIRECTORY",
    "StagedSourceCache",
    "cleanup_staged_source_cache",
    "enter_cuda_free_cpu_phase",
    "load_source_cache",
    "materialize_source_cache",
    "source_generation_runtime",
    "stage_source_cache_for_cpu",
)
