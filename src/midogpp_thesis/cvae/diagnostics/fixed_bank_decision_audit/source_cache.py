"""Experiment-owned facade over the low-level exact-tail source engine."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from ..utility_aligned_exact_tail_router.source_cache import (
    COMPONENT_ARRAY_MEMBER,
    COMPONENT_INDEX_MEMBER,
    SOURCE_ARRAY_MEMBER,
    SOURCE_CACHE_LOCK_MEMBER,
    SOURCE_INDEX_MEMBER,
    SourceCache,
    load_source_cache,
    materialize_source_cache as _materialize_source_cache,
    stage_source_cache_for_cpu as _stage_source_cache_for_cpu,
)
from ..utility_aligned_exact_tail_router.source_cache_validation import (
    validate_source_cache_lock,
)


# ``scratch_preference[0]`` is already experiment-scoped.
LOCAL_STAGE_DIRECTORY = "source_cache"


def materialize_source_cache(*args: object, **kwargs: object) -> SourceCache:
    root = Path(kwargs["root"])
    if "fixed_bank_decision_audit" not in root.as_posix():
        raise ProtocolError("Fixed-bank source cache requires its own artifact root.")
    return _materialize_source_cache(*args, **kwargs)


def stage_source_cache_for_cpu(
    cache: SourceCache, *, scratch_root: Path, canonical_root: Path
) -> SourceCache:
    scratch = Path(scratch_root)
    if scratch.as_posix() != "/data/local/fixed_bank_decision_audit_v1":
        raise ProtocolError("Fixed-bank CPU scratch preference drifted.")
    return _stage_source_cache_for_cpu(
        cache,
        scratch_root=scratch,
        canonical_root=canonical_root,
        local_stage_directory=LOCAL_STAGE_DIRECTORY,
    )


__all__ = (
    "COMPONENT_ARRAY_MEMBER",
    "COMPONENT_INDEX_MEMBER",
    "LOCAL_STAGE_DIRECTORY",
    "SOURCE_ARRAY_MEMBER",
    "SOURCE_CACHE_LOCK_MEMBER",
    "SOURCE_INDEX_MEMBER",
    "SourceCache",
    "load_source_cache",
    "materialize_source_cache",
    "stage_source_cache_for_cpu",
    "validate_source_cache_lock",
)
