"""Read-only prediction-store admission for OE-PPUR worker processes.

The loader deliberately accepts only an already-sealed memmap DTO and a
separate immutable row-index receipt.  It never creates, repairs, truncates,
or reuses a store.  The file descriptor used for hashing is also the descriptor
used to construct the mapping, closing the usual path-swap window between
validation and mapping.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
import hashlib
import os
from os import open as _open_descriptor
from pathlib import Path
import stat
from typing import TYPE_CHECKING

from ..hashing import canonical_hash, require_sha256
from ..identity import CENTERS, EXPECTED_CASE_COUNT, EXPECTED_TEST_ROW_COUNT
from ..manifest_contract import (
    CANONICAL_TERMINAL_CASE_INVENTORY,
    CANONICAL_TERMINAL_ROW_COUNTS_BY_CENTER,
    CANONICAL_TERMINAL_SPLIT,
    CanonicalTerminalManifestReceipt,
    build_canonical_terminal_manifest_receipt,
)
from ..protocol import ProtocolError
from .dtos import MemmapSliceDTO

if TYPE_CHECKING:  # pragma: no cover - imported lazily in production workers
    import numpy as np


_HASH_CHUNK_BYTES = 1024 * 1024
EXPECTED_EXECUTABLE_TEST_CACHE_CONTENT_SHA256 = (
    "df0bdbf64881ee000fe7c56bc486724313accf373ef8e90896344f8d03d187db"
)
EXPECTED_EXECUTABLE_TEST_CACHE_ROW_ORDER_SHA256 = (
    "bd1a85b95496203500bfe2dc5232f8bfb383e73d222a8ba083e81b2c6b33c389"
)
_ROW_ALIGNMENT_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class ImmutableRowIndexReceipt:
    """Content-derived identity for one exact ordered, label-free row index."""

    row_ids: tuple[str, ...]
    source_manifest_sha256: str
    row_count: int = field(init=False)
    row_index_sha256: str = field(init=False)
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(str(value).strip() for value in self.row_ids)
        manifest = require_sha256(
            self.source_manifest_sha256,
            "row-index source-manifest hash",
        )
        if not rows or any(not value for value in rows) or len(set(rows)) != len(rows):
            raise ProtocolError("OE-PPUR row-index receipt is empty or non-unique.")
        row_hash = canonical_hash(
            {
                "schema_version": "oe_ppur_v1_immutable_row_index_v1",
                "ordered_row_ids": rows,
            }
        )
        body = {
            "schema_version": "oe_ppur_v1_immutable_row_index_receipt_v1",
            "source_manifest_sha256": manifest,
            "row_count": len(rows),
            "row_index_sha256": row_hash,
        }
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "source_manifest_sha256", manifest)
        object.__setattr__(self, "row_count", len(rows))
        object.__setattr__(self, "row_index_sha256", row_hash)
        object.__setattr__(self, "receipt_hash", canonical_hash(body))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_immutable_row_index_receipt_v1",
            "source_manifest_sha256": self.source_manifest_sha256,
            "row_count": self.row_count,
            "row_index_sha256": self.row_index_sha256,
            "receipt_hash": self.receipt_hash,
            "contains_labels": False,
        }


@dataclass(frozen=True, slots=True)
class CanonicalRowIdentity:
    """One label-free cache row joined to its canonical whole-case identity."""

    row_ordinal: int
    manifest_row_index: int
    evaluation_row_id: str
    center_id: str
    case_id: str
    split: str = CANONICAL_TERMINAL_SPLIT

    def __post_init__(self) -> None:
        evaluation_row_id = str(self.evaluation_row_id).strip()
        center_id = str(self.center_id).strip()
        case_id = str(self.case_id).strip()
        split = str(self.split).strip()
        if (
            int(self.row_ordinal) < 0
            or int(self.manifest_row_index) < 0
            or not evaluation_row_id
            or center_id not in CENTERS
            or not case_id
            or split != CANONICAL_TERMINAL_SPLIT
        ):
            raise ProtocolError("OE-PPUR canonical row identity drifted.")
        object.__setattr__(self, "row_ordinal", int(self.row_ordinal))
        object.__setattr__(self, "manifest_row_index", int(self.manifest_row_index))
        object.__setattr__(self, "evaluation_row_id", evaluation_row_id)
        object.__setattr__(self, "center_id", center_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "split", split)

    def to_payload(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "manifest_row_index": self.manifest_row_index,
            "evaluation_row_id": self.evaluation_row_id,
            "center_id": self.center_id,
            "case_id": self.case_id,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class CanonicalRowAlignmentReceipt:
    """Exact cache-row to manifest-case alignment required by a successor.

    The guarded factory validates the complete 9,928-row physical order, the
    manifest-order cache pin, every eligible case, and every center count.  It
    contains neutral identities only and never reads or stores labels.
    """

    manifest_receipt: CanonicalTerminalManifestReceipt
    manifest_rows: tuple[CanonicalRowIdentity, ...]
    rows: tuple[CanonicalRowIdentity, ...]
    cache_content_sha256: str
    cache_row_order_sha256: str
    _factory_token: InitVar[object] = None
    manifest_row_identity_sha256: str = field(init=False)
    physical_row_identity_sha256: str = field(init=False)
    row_index_sha256: str = field(init=False)
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _ROW_ALIGNMENT_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR canonical row alignment bypassed its guarded factory."
            )
        manifest = _validate_manifest_receipt(self.manifest_receipt)
        manifest_rows = tuple(self.manifest_rows)
        rows = tuple(self.rows)
        if (
            len(manifest_rows) != EXPECTED_TEST_ROW_COUNT
            or len(rows) != EXPECTED_TEST_ROW_COUNT
            or any(not isinstance(row, CanonicalRowIdentity) for row in manifest_rows)
            or any(not isinstance(row, CanonicalRowIdentity) for row in rows)
            or tuple(row.row_ordinal for row in manifest_rows)
            != tuple(range(EXPECTED_TEST_ROW_COUNT))
            or tuple(row.row_ordinal for row in rows)
            != tuple(range(EXPECTED_TEST_ROW_COUNT))
            or tuple(row.manifest_row_index for row in manifest_rows)
            != tuple(sorted(row.manifest_row_index for row in manifest_rows))
            or len({row.manifest_row_index for row in manifest_rows})
            != EXPECTED_TEST_ROW_COUNT
            or len({row.manifest_row_index for row in rows})
            != EXPECTED_TEST_ROW_COUNT
            or len({row.evaluation_row_id for row in manifest_rows})
            != EXPECTED_TEST_ROW_COUNT
            or len({row.evaluation_row_id for row in rows})
            != EXPECTED_TEST_ROW_COUNT
        ):
            raise ProtocolError("OE-PPUR canonical row inventory is not exact.")

        content = require_sha256(
            self.cache_content_sha256,
            "executable test-cache content hash",
        )
        cache_order = require_sha256(
            self.cache_row_order_sha256,
            "executable test-cache row-order hash",
        )
        derived_cache_order = canonical_hash(
            [row.evaluation_row_id for row in manifest_rows]
        )
        manifest_by_id = {row.evaluation_row_id: row for row in manifest_rows}
        if (
            set(manifest_by_id) != {row.evaluation_row_id for row in rows}
            or rows != manifest_rows
        ):
            raise ProtocolError(
                "OE-PPUR physical cache order drifted from canonical manifest rows."
            )
        row_counts = Counter(row.center_id for row in manifest_rows)
        observed_row_counts = tuple(
            (center, row_counts.get(center, 0)) for center in CENTERS
        )
        observed_cases = tuple(
            sorted({(row.center_id, row.case_id) for row in manifest_rows})
        )
        if (
            content != EXPECTED_EXECUTABLE_TEST_CACHE_CONTENT_SHA256
            or cache_order != EXPECTED_EXECUTABLE_TEST_CACHE_ROW_ORDER_SHA256
            or derived_cache_order != cache_order
            or observed_row_counts != CANONICAL_TERMINAL_ROW_COUNTS_BY_CENTER
            or observed_cases != manifest.case_inventory
            or observed_cases != CANONICAL_TERMINAL_CASE_INVENTORY
            or len(observed_cases) != EXPECTED_CASE_COUNT
        ):
            raise ProtocolError(
                "OE-PPUR cache rows drifted from canonical manifest alignment."
            )
        physical_hash = canonical_hash(
            {
                "schema_version": "oe_ppur_v1_physical_row_identity_v1",
                "ordered_rows": [row.to_payload() for row in rows],
            }
        )
        manifest_alignment_hash = canonical_hash(
            {
                "schema_version": "oe_ppur_v1_manifest_row_identity_v1",
                "manifest_order_rows": [
                    row.to_payload() for row in manifest_rows
                ],
            }
        )
        index = ImmutableRowIndexReceipt(
            tuple(row.evaluation_row_id for row in rows),
            manifest.manifest_content_sha256,
        )
        payload = {
            "schema_version": "oe_ppur_v1_canonical_row_alignment_receipt_v1",
            "manifest_receipt_hash": manifest.receipt_hash,
            "case_inventory_hash": manifest.case_inventory_hash,
            "cache_content_sha256": content,
            "cache_row_order_sha256": cache_order,
            "manifest_row_identity_sha256": manifest_alignment_hash,
            "physical_row_identity_sha256": physical_hash,
            "row_index_sha256": index.row_index_sha256,
            "row_count": len(rows),
            "case_count": len(observed_cases),
            "row_counts_by_center": observed_row_counts,
            "labels_present": False,
        }
        object.__setattr__(self, "manifest_receipt", manifest)
        object.__setattr__(self, "manifest_rows", manifest_rows)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cache_content_sha256", content)
        object.__setattr__(self, "cache_row_order_sha256", cache_order)
        object.__setattr__(
            self,
            "manifest_row_identity_sha256",
            manifest_alignment_hash,
        )
        object.__setattr__(self, "physical_row_identity_sha256", physical_hash)
        object.__setattr__(self, "row_index_sha256", index.row_index_sha256)
        object.__setattr__(self, "receipt_hash", canonical_hash(payload))

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def case_count(self) -> int:
        return len({(row.center_id, row.case_id) for row in self.rows})

    def to_row_index_receipt(self) -> ImmutableRowIndexReceipt:
        receipt = ImmutableRowIndexReceipt(
            tuple(row.evaluation_row_id for row in self.rows),
            self.manifest_receipt.manifest_content_sha256,
        )
        if receipt.row_index_sha256 != self.row_index_sha256:
            raise ProtocolError("OE-PPUR canonical row-index projection drifted.")
        return receipt

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_canonical_row_alignment_receipt_v1",
            "manifest_receipt_hash": self.manifest_receipt.receipt_hash,
            "case_inventory_hash": self.manifest_receipt.case_inventory_hash,
            "cache_content_sha256": self.cache_content_sha256,
            "cache_row_order_sha256": self.cache_row_order_sha256,
            "manifest_row_identity_sha256": self.manifest_row_identity_sha256,
            "physical_row_identity_sha256": self.physical_row_identity_sha256,
            "row_index_sha256": self.row_index_sha256,
            "row_count": self.row_count,
            "case_count": self.case_count,
            "row_counts_by_center": list(CANONICAL_TERMINAL_ROW_COUNTS_BY_CENTER),
            "receipt_hash": self.receipt_hash,
            "labels_present": False,
        }


def build_canonical_row_alignment_receipt(
    *,
    manifest_receipt: CanonicalTerminalManifestReceipt,
    manifest_rows: Sequence[CanonicalRowIdentity],
    rows: Sequence[CanonicalRowIdentity],
    cache_content_sha256: object,
    cache_row_order_sha256: object,
) -> CanonicalRowAlignmentReceipt:
    """Validate independently observed label-free cache alignment."""

    return CanonicalRowAlignmentReceipt(
        manifest_receipt=manifest_receipt,
        manifest_rows=tuple(manifest_rows),
        rows=tuple(rows),
        cache_content_sha256=str(cache_content_sha256),
        cache_row_order_sha256=str(cache_row_order_sha256),
        _factory_token=_ROW_ALIGNMENT_FACTORY_TOKEN,
    )


def validate_canonical_row_alignment_receipt(
    receipt: CanonicalRowAlignmentReceipt,
) -> CanonicalRowAlignmentReceipt:
    if not isinstance(receipt, CanonicalRowAlignmentReceipt):
        raise ProtocolError("OE-PPUR canonical row alignment is untyped.")
    rebuilt = build_canonical_row_alignment_receipt(
        manifest_receipt=receipt.manifest_receipt,
        manifest_rows=receipt.manifest_rows,
        rows=receipt.rows,
        cache_content_sha256=receipt.cache_content_sha256,
        cache_row_order_sha256=receipt.cache_row_order_sha256,
    )
    if rebuilt != receipt:
        raise ProtocolError("OE-PPUR canonical row alignment hash drifted.")
    return receipt


def _validate_manifest_receipt(
    receipt: CanonicalTerminalManifestReceipt,
) -> CanonicalTerminalManifestReceipt:
    if not isinstance(receipt, CanonicalTerminalManifestReceipt):
        raise ProtocolError("OE-PPUR row alignment manifest receipt is untyped.")
    rebuilt = build_canonical_terminal_manifest_receipt(
        annotation_artifact_id=receipt.annotation_artifact_id,
        manifest_member=receipt.manifest_member,
        manifest_content_sha256=receipt.manifest_content_sha256,
        split=receipt.split,
        eligible_center_ids=receipt.eligible_center_ids,
        row_count=receipt.row_count,
        case_count=receipt.case_count,
        row_counts_by_center=receipt.row_counts_by_center,
        case_counts_by_center=receipt.case_counts_by_center,
        case_inventory=receipt.case_inventory,
    )
    if rebuilt != receipt:
        raise ProtocolError("OE-PPUR row alignment manifest receipt hash drifted.")
    return receipt


@dataclass(frozen=True, slots=True)
class MemmapValidationReceipt:
    """Proof that one mapping was opened from the exact admitted bytes."""

    memmap_dto_hash: str
    row_index_receipt_hash: str
    content_sha256: str
    row_index_sha256: str
    file_byte_length: int
    mapped_byte_offset: int
    mapped_byte_length: int
    shape: tuple[int, ...]
    dtype: str
    mode: str
    writeable: bool
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        hashes = {
            "memmap_dto_hash": require_sha256(
                self.memmap_dto_hash, "memmap DTO hash"
            ),
            "row_index_receipt_hash": require_sha256(
                self.row_index_receipt_hash, "row-index receipt hash"
            ),
            "content_sha256": require_sha256(
                self.content_sha256, "memmap content hash"
            ),
            "row_index_sha256": require_sha256(
                self.row_index_sha256, "memmap row-index hash"
            ),
        }
        shape = tuple(int(value) for value in self.shape)
        if (
            not shape
            or any(value <= 0 for value in shape)
            or int(self.file_byte_length)
            != int(self.mapped_byte_offset) + int(self.mapped_byte_length)
            or self.dtype != "float32"
            or self.mode != "r"
            or bool(self.writeable)
        ):
            raise ProtocolError("OE-PPUR memmap validation receipt drifted.")
        for name, value in hashes.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_memmap_validation_receipt_v1",
            "memmap_dto_hash": self.memmap_dto_hash,
            "row_index_receipt_hash": self.row_index_receipt_hash,
            "content_sha256": self.content_sha256,
            "row_index_sha256": self.row_index_sha256,
            "file_byte_length": self.file_byte_length,
            "mapped_byte_offset": self.mapped_byte_offset,
            "mapped_byte_length": self.mapped_byte_length,
            "shape": self.shape,
            "dtype": self.dtype,
            "mode": self.mode,
            "writeable": self.writeable,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class LoadedReadOnlyFloat32Memmap:
    """Process-local mapping plus its immutable validation receipt.

    This wrapper is intentionally not an approved spawn DTO.  Callers pass the
    path/hash DTO through the process boundary and construct this object inside
    the destination worker.
    """

    array: "np.memmap"
    validation: MemmapValidationReceipt

    def __post_init__(self) -> None:
        if self.array.dtype.name != "float32" or bool(self.array.flags.writeable):
            raise ProtocolError("OE-PPUR loaded memmap is not read-only float32.")
        if tuple(int(value) for value in self.array.shape) != self.validation.shape:
            raise ProtocolError("OE-PPUR loaded memmap shape drifted.")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("OE-PPUR process-local memmaps cannot cross spawn boundaries.")


def validate_row_index_receipt(
    receipt: ImmutableRowIndexReceipt,
) -> ImmutableRowIndexReceipt:
    """Recompute every derived identity after a serialization boundary."""

    if not isinstance(receipt, ImmutableRowIndexReceipt):
        raise ProtocolError("OE-PPUR row-index receipt is untyped.")
    rebuilt = ImmutableRowIndexReceipt(
        row_ids=receipt.row_ids,
        source_manifest_sha256=receipt.source_manifest_sha256,
    )
    if rebuilt != receipt:
        raise ProtocolError("OE-PPUR row-index receipt hash drifted.")
    return receipt


def load_read_only_float32_memmap(
    dto: MemmapSliceDTO,
    *,
    row_index_receipt: ImmutableRowIndexReceipt,
) -> LoadedReadOnlyFloat32Memmap:
    """Validate and map an exact float32 store without a writable file handle."""

    if not isinstance(dto, MemmapSliceDTO):
        raise ProtocolError("OE-PPUR memmap loader requires a typed DTO.")
    rows = validate_row_index_receipt(row_index_receipt)
    if dto.mode != "r" or dto.dtype != "float32":
        raise ProtocolError("OE-PPUR memmap loader only accepts float32 mode r.")
    if dto.row_index_sha256 != rows.row_index_sha256:
        raise ProtocolError("OE-PPUR memmap row-index identity drifted.")
    if dto.shape[0] != rows.row_count:
        raise ProtocolError("OE-PPUR memmap row count drifted from its row index.")

    path = Path(dto.path)
    if not path.is_absolute():
        raise ProtocolError("OE-PPUR memmap path is not absolute.")
    try:
        path_stat = path.lstat()
    except OSError as exc:
        raise ProtocolError("OE-PPUR memmap file is absent.") from exc
    if path.is_symlink() or not stat.S_ISREG(path_stat.st_mode):
        raise ProtocolError("OE-PPUR memmap path is not a non-symlink regular file.")

    try:
        descriptor = _open_read_only_descriptor(path)
    except OSError as exc:
        raise ProtocolError("OE-PPUR memmap file could not be opened read-only.") from exc
    try:
        descriptor_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise ProtocolError("OE-PPUR memmap file identity changed during admission.")
        required_extent = int(dto.byte_offset) + int(dto.byte_length)
        if descriptor_stat.st_size != required_extent:
            raise ProtocolError("OE-PPUR memmap file byte extent drifted.")

        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            digest = hashlib.sha256()
            while True:
                chunk = handle.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
            observed_hash = digest.hexdigest()
            if observed_hash != dto.content_sha256:
                raise ProtocolError("OE-PPUR memmap full content hash drifted.")
            handle.seek(0)
            try:
                import numpy as np

                array = np.memmap(
                    handle,
                    dtype=np.float32,
                    mode="r",
                    offset=dto.byte_offset,
                    shape=dto.shape,
                    order="C",
                )
            except (OSError, TypeError, ValueError) as exc:
                raise ProtocolError("OE-PPUR memmap could not be mapped read-only.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    array.setflags(write=False)
    if array.dtype.name != "float32" or bool(array.flags.writeable):
        raise ProtocolError("OE-PPUR memmap mapping is unexpectedly writeable.")
    validation = MemmapValidationReceipt(
        memmap_dto_hash=dto.dto_hash,
        row_index_receipt_hash=rows.receipt_hash,
        content_sha256=observed_hash,
        row_index_sha256=rows.row_index_sha256,
        file_byte_length=path_stat.st_size,
        mapped_byte_offset=dto.byte_offset,
        mapped_byte_length=dto.byte_length,
        shape=dto.shape,
        dtype=dto.dtype,
        mode=dto.mode,
        writeable=bool(array.flags.writeable),
    )
    return LoadedReadOnlyFloat32Memmap(array=array, validation=validation)


def _open_read_only_descriptor(path: Path) -> int:
    """Open one descriptor with a fixed, non-callable-selected read-only mode.

    Keeping the flag construction inside this no-options helper makes the
    source-fence distinction structural: callers cannot supply a writable or
    dynamic mode while the generic fence continues to reject dynamic ``open``
    calls elsewhere.
    """

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return _open_descriptor(path, flags)


__all__ = (
    "CanonicalRowAlignmentReceipt",
    "CanonicalRowIdentity",
    "EXPECTED_EXECUTABLE_TEST_CACHE_CONTENT_SHA256",
    "EXPECTED_EXECUTABLE_TEST_CACHE_ROW_ORDER_SHA256",
    "ImmutableRowIndexReceipt",
    "LoadedReadOnlyFloat32Memmap",
    "MemmapValidationReceipt",
    "build_canonical_row_alignment_receipt",
    "load_read_only_float32_memmap",
    "validate_canonical_row_alignment_receipt",
    "validate_row_index_receipt",
)
