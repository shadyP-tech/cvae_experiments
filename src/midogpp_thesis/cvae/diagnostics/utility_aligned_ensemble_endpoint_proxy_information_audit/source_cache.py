"""Experiment-local adapter for the label-free 270-row source cache.

Only generation mechanics are reused.  The cache is always materialized under
this audit's own artifact root and staged under a distinct scratch namespace,
so no completed Stage-90 output is consumed.
"""

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


def materialize_source_cache(*args: object, **kwargs: object) -> SourceCache:
    root = Path(kwargs["root"])
    if "proxy_information_audit" not in root.as_posix():
        raise ValueError("Proxy-information source cache requires its own artifact root.")
    return _materialize_source_cache(*args, **kwargs)


def stage_source_cache_for_cpu(
    cache: SourceCache, *, scratch_root: Path, canonical_root: Path
) -> SourceCache:
    return _stage_source_cache_for_cpu(
        cache,
        scratch_root=Path(scratch_root) / "ensemble_endpoint_proxy_information_audit_v1",
        canonical_root=canonical_root,
    )


__all__ = (
    "COMPONENT_ARRAY_MEMBER",
    "COMPONENT_INDEX_MEMBER",
    "SOURCE_ARRAY_MEMBER",
    "SOURCE_CACHE_LOCK_MEMBER",
    "SOURCE_INDEX_MEMBER",
    "SourceCache",
    "load_source_cache",
    "materialize_source_cache",
    "stage_source_cache_for_cpu",
    "validate_source_cache_lock",
)
