"""Validated read-only loading for SCALE-BP workstation memmap references."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..hashing import canonical_hash
from ..physical.library import PhysicalCellIdentity
from ..protocol import ProtocolError
from .memmap_contracts import MemmapReference
from .physical_bank import PhysicalBankReceipt


def open_readonly_memmap(
    reference: MemmapReference,
    *,
    physical_bank: PhysicalBankReceipt | None = None,
    physical_identity: PhysicalCellIdentity | None = None,
) -> np.memmap:
    """Open only after validating size and the exact referenced byte slice."""

    if not isinstance(reference, MemmapReference):
        raise ProtocolError("SCALE-BP memmap loader reference drifted.")
    if reference.semantic_role == "physical_probabilities":
        if (
            not isinstance(physical_bank, PhysicalBankReceipt)
            or not isinstance(physical_identity, PhysicalCellIdentity)
        ):
            raise ProtocolError(
                "SCALE-BP physical memmap requires its bank receipt and cell identity."
            )
        physical_bank.assert_reference(physical_identity, reference)
    path = Path(reference.path)
    try:
        stat = path.stat()
    except OSError as exc:
        raise ProtocolError("SCALE-BP memmap file is absent.") from exc
    stop = reference.offset_bytes + reference.byte_length
    if (
        path.is_symlink()
        or path.resolve() != path
        or not path.is_file()
        or stat.st_size < stop
    ):
        raise ProtocolError("SCALE-BP memmap byte extent drifted.")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            handle.seek(reference.offset_bytes)
            remaining = reference.byte_length
            while remaining:
                chunk = handle.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise ProtocolError("SCALE-BP memmap byte slice is truncated.")
                digest.update(chunk)
                remaining -= len(chunk)
    except OSError as exc:
        raise ProtocolError("SCALE-BP memmap byte slice is unreadable.") from exc
    if digest.hexdigest() != reference.sha256:
        raise ProtocolError("SCALE-BP memmap content hash drifted.")
    try:
        values = np.memmap(
            path,
            dtype=np.dtype(reference.dtype),
            mode="r",
            offset=reference.offset_bytes,
            shape=reference.shape,
            order=reference.order,
        )
    except (OSError, ValueError) as exc:
        raise ProtocolError("SCALE-BP memmap could not open read-only.") from exc
    values.setflags(write=False)
    return values


def row_index_hash(row_keys: object) -> str:
    rows = tuple(tuple(str(value) for value in key) for key in row_keys)  # type: ignore[arg-type]
    if not rows or rows != tuple(sorted(set(rows))) or any(not value for row in rows for value in row):
        raise ProtocolError("SCALE-BP memmap row-index identity drifted.")
    return canonical_hash(
        {
            "schema_version": "scale_bp_memmap_row_index_v1",
            "row_keys": rows,
        }
    )


def validate_row_index(reference: MemmapReference, row_keys: object) -> str:
    digest = row_index_hash(row_keys)
    if digest != reference.row_index_hash:
        raise ProtocolError("SCALE-BP memmap row-index hash drifted.")
    return digest


__all__ = ("open_readonly_memmap", "row_index_hash", "validate_row_index")
