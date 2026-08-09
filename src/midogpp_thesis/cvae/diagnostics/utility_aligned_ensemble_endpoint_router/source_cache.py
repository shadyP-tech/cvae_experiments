"""Narrow source-cache adapter for the new terminal experiment.

Only the label-free generation/cache leaf is shared.  All downstream
probability stores and seals are package-owned and cannot read the completed
exact-tail diagnostic artifact.
"""

from __future__ import annotations

from pathlib import Path

from ..utility_aligned_exact_tail_router.source_cache import (
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
    load_source_cache,
    materialize_source_cache as _materialize_source_cache,
    stage_source_cache_for_cpu as _stage_source_cache_for_cpu,
    validate_source_cache_inventory,
)
from ..utility_aligned_exact_tail_router.source_cache_validation import (
    build_source_cache_lock,
    validate_source_cache_lock,
)


def materialize_source_cache(*args: object, **kwargs: object) -> SourceCache:
    """Create a new cache in the caller's versioned artifact root."""

    root = Path(kwargs.get("root"))
    if "utility_aligned_exact_tail_router" in root.as_posix():
        raise ValueError("The ensemble endpoint experiment requires its own artifact root.")
    return _materialize_source_cache(*args, **kwargs)


def stage_source_cache_for_cpu(
    cache: SourceCache, *, scratch_root: Path, canonical_root: Path
) -> SourceCache:
    # Prefixing the shared scratch path prevents either Stage-90 diagnostic from
    # replacing the other's local cache while retaining the tested copy logic.
    return _stage_source_cache_for_cpu(
        cache,
        scratch_root=Path(scratch_root) / "ensemble_endpoint_router_v1",
        canonical_root=canonical_root,
    )


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
