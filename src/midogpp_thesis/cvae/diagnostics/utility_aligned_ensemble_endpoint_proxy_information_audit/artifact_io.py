"""Package-local byte persistence facade.

The shared implementation has no scientific schemas or experiment imports.
This facade keeps the audit independent of another Stage-90 runner.
"""

from ..utility_aligned_exact_tail_router.artifact_io import (
    atomic_json,
    persist_or_validate_csv,
    persist_or_validate_json,
    read_json,
    relative_files,
    render_csv,
    sha256_file,
)

__all__ = (
    "atomic_json",
    "persist_or_validate_csv",
    "persist_or_validate_json",
    "read_json",
    "relative_files",
    "render_csv",
    "sha256_file",
)
