"""Package-local deterministic artifact persistence facade.

Only the low-level byte-stable helpers from the exact-tail execution package
are reused.  No scientific surface or prior Stage-90 artifact is imported.
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
