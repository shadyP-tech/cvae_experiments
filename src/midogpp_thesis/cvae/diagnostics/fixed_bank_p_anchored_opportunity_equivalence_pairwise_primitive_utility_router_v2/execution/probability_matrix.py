"""Secure byte parser for OE-PPUR v2 GPU probability shards.

The parser is independent of the planned v1 adapter.  It consumes primitive,
already-admitted lineage metadata, opens every raw shard with a read-only
no-follow descriptor, derives shape from bytes, validates the scientific
values, and delegates immutable receipt construction to the sibling receipt
module.  No labels or terminal capability exist on this surface.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import os
from os import open as _open_descriptor
from pathlib import Path
import stat

from ....protocol import ProtocolError
from ..row_binding import (
    CanonicalAdmittedRowBindingReceipt,
    validate_admitted_row_binding,
)
from .probability_matrix_receipts import (
    EXPECTED_PROBABILITY_COLUMN_COUNT,
    EXPECTED_PROBABILITY_COLUMNS,
    EXPECTED_PROBABILITY_ROW_COUNT,
    PROBABILITY_STORAGE_BYTE_ORDER,
    PROBABILITY_STORAGE_DTYPE,
    PROBABILITY_STORAGE_MEMORY_ORDER,
    ROW_BYTE_WIDTH,
    ParsedProbabilityMatrixScienceReceipt,
    ParsedProbabilityMatrixShardReceipt,
    ProbabilityMatrixShardSpec,
    _issue_parsed_probability_matrix_receipt,
    _issue_parsed_probability_shard_receipt,
    _ordered_unique,
    _rebuild_parsed_probability_matrix_receipt,
    _require_sha256,
)


def parse_probability_matrix_shards(
    shards: Sequence[ProbabilityMatrixShardSpec],
    *,
    scratch_root: str | Path,
    row_binding: CanonicalAdmittedRowBindingReceipt,
    gpu_prediction_batch_hash: object,
    gpu_result_surface_sha256: object,
    ordered_gpu_worker_result_hashes: Sequence[object],
    ordered_gpu_result_file_hashes: Sequence[object],
) -> ParsedProbabilityMatrixScienceReceipt:
    """Parse and bind exactly 9,928 rows and the seven canonical columns.

    No caller-supplied row count, column count, dtype alias, or storage-order
    override is accepted.  Those values are closed-world v2 science constants.
    """

    scratch = _require_admitted_scratch_root(scratch_root)
    binding = validate_admitted_row_binding(row_binding)
    expected = {
        "six_input_admission_hash": binding.six_input_admission_hash,
        "row_binding_hash": binding.receipt_hash,
        "row_index_sha256": binding.row_index_sha256,
        "row_alignment_receipt_hash": binding.row_alignment_receipt_hash,
        "gpu_prediction_batch_hash": _require_sha256(
            gpu_prediction_batch_hash, "GPU prediction-batch hash"
        ),
        "gpu_result_surface_sha256": _require_sha256(
            gpu_result_surface_sha256, "GPU result-surface hash"
        ),
    }
    workers = tuple(
        _require_sha256(value, "GPU worker-result hash")
        for value in ordered_gpu_worker_result_hashes
    )
    files = tuple(
        _require_sha256(value, "GPU result-file hash")
        for value in ordered_gpu_result_file_hashes
    )
    ordered = tuple(shards)
    if (
        not ordered
        or any(not isinstance(row, ProbabilityMatrixShardSpec) for row in ordered)
        or not workers
        or len(set(workers)) != len(workers)
        or len(files) != len(ordered)
        or tuple(row.content_sha256 for row in ordered) != files
        or len({row.path for row in ordered}) != len(ordered)
    ):
        raise ProtocolError("OE-PPUR v2 probability shard inventory drifted.")
    _validate_spec_lineage(ordered, expected)
    _validate_exact_spec_topology(ordered)
    if _ordered_unique(
        tuple(row.gpu_worker_result_sha256 for row in ordered)
    ) != workers:
        raise ProtocolError(
            "OE-PPUR v2 probability shard worker lineage drifted."
        )

    receipts: list[ParsedProbabilityMatrixShardReceipt] = []
    opened_identities: set[tuple[int, int]] = set()
    matrix_hasher = hashlib.sha256()
    column_hashers = [
        hashlib.sha256() for _ in range(EXPECTED_PROBABILITY_COLUMN_COUNT)
    ]
    scratch_descriptor, scratch_identity = _open_admitted_scratch_root(scratch)
    try:
        for ordinal, spec in enumerate(ordered):
            relative_path = _require_strict_scratch_descendant(
                Path(spec.path),
                scratch,
            )
            receipt, physical_identity, payload, column_payloads = (
                _parse_one_shard(
                    spec,
                    ordinal=ordinal,
                    scratch_descriptor=scratch_descriptor,
                    relative_path=relative_path,
                )
            )
            if physical_identity in opened_identities:
                raise ProtocolError(
                    "OE-PPUR v2 probability shards reused one physical file."
                )
            opened_identities.add(physical_identity)
            matrix_hasher.update(payload)
            for hasher, column_payload in zip(
                column_hashers,
                column_payloads,
                strict=True,
            ):
                hasher.update(column_payload)
            receipts.append(receipt)
        _revalidate_admitted_scratch_root(
            scratch,
            descriptor=scratch_descriptor,
            opened_identity=scratch_identity,
        )
    finally:
        os.close(scratch_descriptor)
    return _issue_parsed_probability_matrix_receipt(
        six_input_admission_hash=expected["six_input_admission_hash"],
        input_binding_hash=binding.input_binding_hash,
        row_binding_hash=binding.receipt_hash,
        cache_content_sha256=binding.cache_content_sha256,
        cache_row_order_sha256=binding.cache_row_order_sha256,
        manifest_sha256=binding.manifest_sha256,
        case_inventory_sha256=binding.case_inventory_sha256,
        row_index_sha256=expected["row_index_sha256"],
        row_alignment_receipt_hash=expected["row_alignment_receipt_hash"],
        gpu_prediction_batch_hash=expected["gpu_prediction_batch_hash"],
        gpu_result_surface_sha256=expected["gpu_result_surface_sha256"],
        gpu_worker_result_hashes=workers,
        gpu_result_file_hashes=files,
        shards=tuple(receipts),
        matrix_content_sha256=matrix_hasher.hexdigest(),
        column_content_sha256s=tuple(
            hasher.hexdigest() for hasher in column_hashers
        ),
        scientific_values_validated=True,
    )


def validate_parsed_probability_matrix_science_receipt(
    receipt: object,
    *,
    row_binding: CanonicalAdmittedRowBindingReceipt | None = None,
    shards: Sequence[ProbabilityMatrixShardSpec] | None = None,
    scratch_root: str | Path | None = None,
) -> ParsedProbabilityMatrixScienceReceipt:
    """Recompute a receipt structurally and optionally reparse its files."""

    if not isinstance(receipt, ParsedProbabilityMatrixScienceReceipt):
        raise ProtocolError("OE-PPUR v2 parsed probability receipt is untyped.")
    rebuilt = _rebuild_parsed_probability_matrix_receipt(receipt)
    if rebuilt != receipt:
        raise ProtocolError(
            "OE-PPUR v2 parsed probability science receipt hash drifted."
        )
    binding = (
        validate_admitted_row_binding(row_binding)
        if row_binding is not None
        else None
    )
    if binding is not None and not _receipt_matches_row_binding(receipt, binding):
        raise ProtocolError(
            "OE-PPUR v2 parsed matrix belongs to an unrelated row binding."
        )
    if shards is not None:
        if binding is None:
            raise ProtocolError(
                "OE-PPUR v2 file revalidation requires its typed row binding."
            )
        if scratch_root is None:
            raise ProtocolError(
                "OE-PPUR v2 file revalidation requires its admitted scratch root."
            )
        reparsed = parse_probability_matrix_shards(
            shards,
            scratch_root=scratch_root,
            row_binding=binding,
            gpu_prediction_batch_hash=receipt.gpu_prediction_batch_hash,
            gpu_result_surface_sha256=receipt.gpu_result_surface_sha256,
            ordered_gpu_worker_result_hashes=receipt.gpu_worker_result_hashes,
            ordered_gpu_result_file_hashes=receipt.gpu_result_file_hashes,
        )
        if reparsed != receipt:
            raise ProtocolError(
                "OE-PPUR v2 probability files drifted from their science receipt."
            )
    return receipt


def _receipt_matches_row_binding(
    receipt: ParsedProbabilityMatrixScienceReceipt,
    binding: CanonicalAdmittedRowBindingReceipt,
) -> bool:
    return all(
        getattr(receipt, receipt_field) == getattr(binding, binding_field)
        for receipt_field, binding_field in (
            ("six_input_admission_hash", "six_input_admission_hash"),
            ("input_binding_hash", "input_binding_hash"),
            ("row_binding_hash", "receipt_hash"),
            ("cache_content_sha256", "cache_content_sha256"),
            ("cache_row_order_sha256", "cache_row_order_sha256"),
            ("manifest_sha256", "manifest_sha256"),
            ("case_inventory_sha256", "case_inventory_sha256"),
            ("row_index_sha256", "row_index_sha256"),
            ("row_alignment_receipt_hash", "row_alignment_receipt_hash"),
            ("row_count", "row_count"),
            ("case_count", "case_count"),
        )
    )


def _validate_spec_lineage(
    shards: tuple[ProbabilityMatrixShardSpec, ...],
    expected: dict[str, str],
) -> None:
    for shard in shards:
        if any(getattr(shard, name) != value for name, value in expected.items()):
            raise ProtocolError(
                "OE-PPUR v2 probability shard admission lineage drifted."
            )


def _validate_exact_spec_topology(
    shards: tuple[ProbabilityMatrixShardSpec, ...],
) -> None:
    cursor = 0
    for shard in shards:
        if shard.row_start != cursor:
            raise ProtocolError(
                "OE-PPUR v2 probability shard intervals contain a gap, overlap, or reorder."
            )
        cursor = shard.row_stop
    if cursor != EXPECTED_PROBABILITY_ROW_COUNT:
        raise ProtocolError(
            "OE-PPUR v2 probability shards do not cover exactly 9,928 rows."
        )


def _parse_one_shard(
    spec: ProbabilityMatrixShardSpec,
    *,
    ordinal: int,
    scratch_descriptor: int,
    relative_path: Path,
) -> tuple[
    ParsedProbabilityMatrixShardReceipt,
    tuple[int, int],
    bytes,
    tuple[bytes, ...],
]:
    path = Path(spec.path)
    try:
        path_before = path.lstat()
    except OSError as exc:
        raise ProtocolError("OE-PPUR v2 probability shard is absent.") from exc
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise ProtocolError(
            "OE-PPUR v2 probability shard is not a non-symlink regular file."
        )

    descriptor = -1
    try:
        descriptor = _open_read_only_no_follow_beneath(
            scratch_descriptor,
            relative_path,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stable_identity(opened) != _stable_identity(path_before)
        ):
            raise ProtocolError(
                "OE-PPUR v2 probability shard changed before parsing."
            )
        if opened.st_size % ROW_BYTE_WIDTH != 0:
            raise ProtocolError(
                "OE-PPUR v2 probability shard byte extent is not row aligned."
            )
        derived_rows = opened.st_size // ROW_BYTE_WIDTH
        derived_shape = (derived_rows, EXPECTED_PROBABILITY_COLUMN_COUNT)
        expected_shape = (
            spec.row_stop - spec.row_start,
            EXPECTED_PROBABILITY_COLUMN_COUNT,
        )
        if derived_shape != expected_shape or derived_shape != spec.declared_shape:
            raise ProtocolError(
                "OE-PPUR v2 probability shard shape/extent drifted."
            )

        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            payload = handle.read(opened.st_size + 1)
            if len(payload) != opened.st_size:
                raise ProtocolError(
                    "OE-PPUR v2 probability shard changed while reading."
                )
            digest = hashlib.sha256(payload).hexdigest()
            if digest != spec.content_sha256:
                raise ProtocolError(
                    "OE-PPUR v2 probability shard content hash drifted."
                )
            minimum, maximum, column_payloads = _validate_probability_values(
                payload,
                expected_shape=expected_shape,
            )
            after = os.fstat(handle.fileno())
            try:
                path_after = path.lstat()
            except OSError as exc:
                raise ProtocolError(
                    "OE-PPUR v2 probability shard path changed after parsing."
                ) from exc
    except OSError as exc:
        raise ProtocolError(
            "OE-PPUR v2 probability shard could not be opened read-only."
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if (
        _stable_identity(opened) != _stable_identity(after)
        or _stable_identity(after) != _stable_identity(path_after)
        or stat.S_ISLNK(path_after.st_mode)
    ):
        raise ProtocolError(
            "OE-PPUR v2 probability shard changed during parsing."
        )
    column_hashes = tuple(
        hashlib.sha256(value).hexdigest() for value in column_payloads
    )
    receipt = _issue_parsed_probability_shard_receipt(
        shard_ordinal=ordinal,
        file_sha256=digest,
        gpu_worker_result_sha256=spec.gpu_worker_result_sha256,
        row_start=spec.row_start,
        row_stop=spec.row_stop,
        shape=derived_shape,
        column_ids=spec.column_ids,
        column_content_sha256s=column_hashes,
        dtype=spec.dtype,
        byte_order=spec.byte_order,
        memory_order=spec.memory_order,
        byte_length=opened.st_size,
        value_count=derived_rows * EXPECTED_PROBABILITY_COLUMN_COUNT,
        minimum_probability=minimum,
        maximum_probability=maximum,
        descriptor_read_only=True,
        no_follow_used=True,
        stable_identity_revalidated=True,
    )
    return receipt, _device_inode(after), payload, column_payloads


def _validate_probability_values(
    payload: bytes,
    *,
    expected_shape: tuple[int, int],
) -> tuple[float, float, tuple[bytes, ...]]:
    try:
        import numpy as np

        values = np.frombuffer(payload, dtype=np.dtype(PROBABILITY_STORAGE_DTYPE))
        matrix = values.reshape(expected_shape, order=PROBABILITY_STORAGE_MEMORY_ORDER)
    except (ImportError, TypeError, ValueError) as exc:
        raise ProtocolError(
            "OE-PPUR v2 probability shard could not be parsed as <f4 C-order."
        ) from exc
    if (
        matrix.dtype.str != PROBABILITY_STORAGE_DTYPE
        or not matrix.flags.c_contiguous
        or matrix.shape != expected_shape
    ):
        raise ProtocolError(
            "OE-PPUR v2 probability shard dtype/endian/order drifted."
        )
    if not bool(np.isfinite(matrix).all()):
        raise ProtocolError(
            "OE-PPUR v2 probability shard contains NaN or infinity."
        )
    if bool(np.any((matrix < 0.0) | (matrix > 1.0))):
        raise ProtocolError(
            "OE-PPUR v2 probability shard contains an out-of-range value."
        )
    column_payloads = tuple(
        np.ascontiguousarray(
            matrix[:, index],
            dtype=np.dtype(PROBABILITY_STORAGE_DTYPE),
        ).tobytes(order="C")
        for index in range(EXPECTED_PROBABILITY_COLUMN_COUNT)
    )
    return float(matrix.min()), float(matrix.max()), column_payloads


def _require_admitted_scratch_root(value: str | Path) -> Path:
    """Return one canonical, absolute, non-symlink scratch directory."""

    try:
        path = Path(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(
            "OE-PPUR v2 admitted scratch root is invalid."
        ) from exc
    if not path.is_absolute():
        raise ProtocolError(
            "OE-PPUR v2 admitted scratch root is not absolute."
        )
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(
            "OE-PPUR v2 admitted scratch root is absent or inaccessible."
        ) from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or resolved != path
    ):
        raise ProtocolError(
            "OE-PPUR v2 admitted scratch root is not a canonical non-symlink directory."
        )
    return path


def _open_admitted_scratch_root(
    path: Path,
) -> tuple[int, tuple[int, int, int, int, int]]:
    """Open and identity-pin the scratch root used for all relative opens."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise ProtocolError(
            "OE-PPUR v2 host cannot provide no-follow scratch admission."
        )
    flags = os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)
    descriptor = -1
    try:
        before = path.lstat()
        descriptor = _open_descriptor(path, flags)
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISDIR(before.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _stable_identity(before) != _stable_identity(opened)
        ):
            raise ProtocolError(
                "OE-PPUR v2 admitted scratch root changed before parsing."
            )
        return descriptor, _stable_identity(opened)
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise ProtocolError(
            "OE-PPUR v2 admitted scratch root could not be opened read-only."
        ) from exc
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _require_strict_scratch_descendant(path: Path, scratch_root: Path) -> Path:
    """Return the canonical relative shard path without following links."""

    if not path.is_absolute():
        raise ProtocolError("OE-PPUR v2 probability shard path is not absolute.")
    try:
        relative = path.relative_to(scratch_root)
    except ValueError as exc:
        raise ProtocolError(
            "OE-PPUR v2 probability shard is outside its admitted scratch root."
        ) from exc
    if (
        not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ProtocolError(
            "OE-PPUR v2 probability shard is not a strict scratch descendant."
        )
    return relative


def _revalidate_admitted_scratch_root(
    path: Path,
    *,
    descriptor: int,
    opened_identity: tuple[int, int, int, int, int],
) -> None:
    try:
        after_path = path.lstat()
        after_descriptor = os.fstat(descriptor)
    except OSError as exc:
        raise ProtocolError(
            "OE-PPUR v2 admitted scratch root changed during parsing."
        ) from exc
    if (
        stat.S_ISLNK(after_path.st_mode)
        or not stat.S_ISDIR(after_path.st_mode)
        or _stable_identity(after_path) != opened_identity
        or _stable_identity(after_descriptor) != opened_identity
    ):
        raise ProtocolError(
            "OE-PPUR v2 admitted scratch root changed during parsing."
        )


def _open_read_only_no_follow_beneath(
    scratch_descriptor: int,
    relative_path: Path,
) -> int:
    """Open a shard beneath scratch while rejecting every symlink component."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise ProtocolError(
            "OE-PPUR v2 host cannot provide no-follow probability admission."
        )
    components = relative_path.parts
    parent_descriptor = os.dup(scratch_descriptor)
    try:
        directory_flags = (
            os.O_RDONLY
            | no_follow
            | directory
            | getattr(os, "O_CLOEXEC", 0)
        )
        for component in components[:-1]:
            child_descriptor = _open_descriptor(
                component,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(child_descriptor)
            if not stat.S_ISDIR(opened.st_mode):
                os.close(child_descriptor)
                raise ProtocolError(
                    "OE-PPUR v2 probability shard parent is unsafe."
                )
            os.close(parent_descriptor)
            parent_descriptor = child_descriptor
        file_flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
        return _open_descriptor(
            components[-1],
            file_flags,
            dir_fd=parent_descriptor,
        )
    finally:
        os.close(parent_descriptor)


def _device_inode(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


def _stable_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


__all__ = (
    "EXPECTED_PROBABILITY_COLUMN_COUNT",
    "EXPECTED_PROBABILITY_COLUMNS",
    "EXPECTED_PROBABILITY_ROW_COUNT",
    "PROBABILITY_STORAGE_BYTE_ORDER",
    "PROBABILITY_STORAGE_DTYPE",
    "PROBABILITY_STORAGE_MEMORY_ORDER",
    "ParsedProbabilityMatrixScienceReceipt",
    "ParsedProbabilityMatrixShardReceipt",
    "ProbabilityMatrixShardSpec",
    "parse_probability_matrix_shards",
    "validate_parsed_probability_matrix_science_receipt",
)
