"""Read-only memmap preflight and worker loading."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

import numpy as np

from ..artifacts.hashing import canonical_hash, sha256_file
from ..protocol import GovernanceError
from .dtos import MemmapReference


def validate_memmap_reference(reference: MemmapReference) -> str:
    if not isinstance(reference, MemmapReference):
        raise GovernanceError("SCALE-BP v2 memmap preflight received a foreign DTO.")
    path = Path(reference.path)
    stop = reference.offset_bytes + reference.byte_length
    if path.is_symlink() or not path.is_file() or path.stat().st_size < stop:
        raise GovernanceError("SCALE-BP v2 memmap byte extent is absent or unsafe.")
    observed = sha256_file(
        path, offset=reference.offset_bytes, length=reference.byte_length
    )
    if observed != reference.sha256:
        raise GovernanceError("SCALE-BP v2 memmap slice hash drifted.")
    return reference.reference_hash


def validate_memmap_references(
    references: Sequence[MemmapReference],
) -> tuple[str, ...]:
    """Hash each unique slice once in the coordinator before spawning."""

    rows = tuple(references)
    if not rows:
        raise GovernanceError("SCALE-BP v2 memmap inventory is empty.")
    observed: dict[str, str] = {}
    for row in rows:
        if row.reference_hash not in observed:
            observed[row.reference_hash] = validate_memmap_reference(row)
    return tuple(sorted(observed))


def open_readonly_memmap(
    reference: MemmapReference, *, verify_content: bool = True
) -> np.memmap:
    """Open an exact slice in mode ``r``; writable mappings never exist."""

    if verify_content:
        validate_memmap_reference(reference)
    else:
        path = Path(reference.path)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size < reference.offset_bytes + reference.byte_length
        ):
            raise GovernanceError("SCALE-BP v2 memmap changed after preflight.")
    try:
        values = np.memmap(
            reference.path,
            dtype=np.dtype(reference.dtype),
            mode="r",
            offset=reference.offset_bytes,
            shape=reference.shape,
            order=reference.order,
        )
    except (OSError, ValueError) as exc:
        raise GovernanceError("SCALE-BP v2 memmap could not be opened read-only.") from exc
    values.setflags(write=False)
    return values


@contextmanager
def open_memmap_bundle(
    references: Sequence[MemmapReference], *, verify_content: bool = True
) -> Iterator[Mapping[str, np.memmap]]:
    rows = tuple(references)
    if not rows or len({row.semantic_role for row in rows}) != len(rows):
        raise GovernanceError("SCALE-BP v2 memmap role inventory drifted.")
    opened: dict[str, np.memmap] = {}
    try:
        for row in rows:
            opened[row.semantic_role] = open_readonly_memmap(
                row, verify_content=verify_content
            )
        yield MappingProxyType(opened)
    finally:
        for values in opened.values():
            handle = getattr(values, "_mmap", None)
            if handle is not None:
                handle.close()
        opened.clear()


def row_index_hash(row_keys: object) -> str:
    try:
        rows = tuple(  # type: ignore[arg-type]
            tuple(str(value) for value in row) for row in row_keys
        )
    except TypeError as exc:
        raise GovernanceError("SCALE-BP v2 row index is not iterable.") from exc
    if not rows or rows != tuple(sorted(set(rows))) or any(
        not value for row in rows for value in row
    ):
        raise GovernanceError("SCALE-BP v2 row-index identity drifted.")
    return canonical_hash(
        {"schema_version": "scale_bp_v2_memmap_row_index_v1", "row_keys": rows}
    )


def validate_row_index(reference: MemmapReference, row_keys: object) -> str:
    digest = row_index_hash(row_keys)
    if digest != reference.row_index_hash:
        raise GovernanceError("SCALE-BP v2 memmap row-index hash drifted.")
    return digest


__all__ = (
    "open_memmap_bundle",
    "open_readonly_memmap",
    "row_index_hash",
    "validate_memmap_reference",
    "validate_memmap_references",
    "validate_row_index",
)
