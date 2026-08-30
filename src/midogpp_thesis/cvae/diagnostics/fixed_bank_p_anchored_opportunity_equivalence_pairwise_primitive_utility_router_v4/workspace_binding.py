"""Canonical workspace binding for the single-use OE-PPUR v4 execution."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from .identity import CANONICAL_OUTPUT_RELATIVE_ROOT


def canonical_output_root() -> Path:
    """Resolve the sole sealed, catalog-pinned output root for this checkout."""

    source = Path(__file__).resolve()
    repository = next(
        (
            parent
            for parent in source.parents
            if (parent / "src/midogpp_thesis").is_dir()
            and (parent / "experiments/midogpp/artifact_catalog.yaml").is_file()
        ),
        None,
    )
    if repository is None or repository.is_symlink():
        raise ProtocolError("OE-PPUR v4 canonical output binding failed.")
    return repository / CANONICAL_OUTPUT_RELATIVE_ROOT


def assert_canonical_output_root(value: str | Path) -> Path:
    """Reject relocated resolved configs before they can mint a new lease."""

    observed = Path(value)
    expected = canonical_output_root()
    if not observed.is_absolute() or observed != expected:
        raise ProtocolError("OE-PPUR v4 resolved output is not catalog-canonical.")
    return expected


__all__ = ("assert_canonical_output_root", "canonical_output_root")
