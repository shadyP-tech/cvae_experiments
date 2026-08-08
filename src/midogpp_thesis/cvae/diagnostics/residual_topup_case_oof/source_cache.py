"""Public facade for the modular independent case-OOF source cache."""

from .source_cache_contracts import (
    COMPATIBILITY_CASE_COLUMNS,
    COMPATIBILITY_CASE_MEMBER,
    EXPECTED_SOURCE_BLOCK_COUNT,
    EXPECTED_SOURCE_TASK_COUNT,
    GENERATION_DEVICES,
    SOURCE_BLOCK_ARRAY_MEMBER,
    SOURCE_BLOCK_INDEX_COLUMNS,
    SOURCE_BLOCK_INDEX_MEMBER,
    SOURCE_CACHE_LOCK_MEMBER,
    SourceCache,
)
from .source_cache_execution import materialize_source_cache
from .source_cache_store import load_source_cache
from .source_cache_validation import (
    build_source_cache_lock,
    validate_source_cache_inventory,
    validate_source_cache_lock,
)


__all__ = (
    "COMPATIBILITY_CASE_COLUMNS",
    "COMPATIBILITY_CASE_MEMBER",
    "EXPECTED_SOURCE_BLOCK_COUNT",
    "EXPECTED_SOURCE_TASK_COUNT",
    "GENERATION_DEVICES",
    "SOURCE_BLOCK_ARRAY_MEMBER",
    "SOURCE_BLOCK_INDEX_COLUMNS",
    "SOURCE_BLOCK_INDEX_MEMBER",
    "SOURCE_CACHE_LOCK_MEMBER",
    "SourceCache",
    "build_source_cache_lock",
    "load_source_cache",
    "materialize_source_cache",
    "validate_source_cache_inventory",
    "validate_source_cache_lock",
)
