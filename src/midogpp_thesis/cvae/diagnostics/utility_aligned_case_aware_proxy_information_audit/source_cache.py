"""V2-owned facade over the parameterized label-free source engine."""

from __future__ import annotations

from pathlib import Path

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


LOCAL_STAGE_DIRECTORY = "case_aware_proxy_information_audit_v1/source_cache"


def materialize_source_cache(*args: object, **kwargs: object) -> SourceCache:
    root = Path(kwargs["root"])
    if "case_aware_proxy_information_audit" not in root.as_posix():
        raise ValueError("Case-aware source cache requires its own artifact root.")
    return _materialize_source_cache(*args, **kwargs)


def stage_source_cache_for_cpu(
    cache: SourceCache, *, scratch_root: Path, canonical_root: Path
) -> SourceCache:
    return _stage_source_cache_for_cpu(
        cache,
        scratch_root=scratch_root,
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
