"""Builder-only opaque JPEG access for reserved Stage-70 rows.

The source location never becomes part of a reservation, shard, report, or
returned identity.  It exists only long enough to read the bytes for the
currently bound canonical-manifest row and is then discarded before yielding.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from midogpp_thesis.data.contract.paths import resolve_contract_path

from .contracts import (
    ManifestAccessEvent,
    TargetEvaluationContractError,
    TargetEvaluationReservation,
    TargetEvaluationRow,
)
from .projector import AccessLog, _record_access, file_sha256
from .validation import validate_target_evaluation_reservation


@dataclass(frozen=True)
class OpaqueImageBytes:
    """A reserved neutral row plus transient JPEG bytes; no source location."""

    row: TargetEvaluationRow
    jpeg_bytes: bytes


def iter_bound_image_bytes(
    manifest_path: str | Path,
    reservation: TargetEvaluationReservation,
    *,
    repo_root: str | Path,
    image_reader: Callable[[Path], bytes] | None = None,
    access_log: AccessLog = None,
    allow_test_fixture: bool = False,
) -> Iterator[OpaqueImageBytes]:
    """Yield bytes for exactly the reserved rows in canonical manifest order.

    This is the only pre-scoring API that reads ``image_path``.  It reads that
    cell only after a contract-row index is proven present in the validated
    reservation, never returns the location, and never touches source sample or
    outcome fields.
    """

    expected_counts = reservation.rows_by_center
    validate_target_evaluation_reservation(
        reservation,
        expected_manifest_sha256=reservation.manifest_sha256,
        expected_rows_by_center=expected_counts,
        allow_test_fixture=allow_test_fixture,
    )
    path = Path(manifest_path)
    if file_sha256(path) != reservation.manifest_sha256:
        raise TargetEvaluationContractError(
            "Opaque Stage-70 image access rejected manifest hash drift."
        )
    reader_fn = image_reader or _read_image_bytes
    resolved_repo_root = Path(repo_root).resolve()
    reserved_by_index = {row.contract_row_index: row for row in reservation.rows}
    yielded: list[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise TargetEvaluationContractError(
                "Opaque Stage-70 image access found an empty manifest."
            ) from exc
        if len(header) != len(set(header)) or "image_path" not in header:
            raise TargetEvaluationContractError(
                "Opaque Stage-70 image access requires one source-location column."
            )
        image_column = header.index("image_path")
        for contract_row_index, cells in enumerate(reader):
            row = reserved_by_index.get(contract_row_index)
            if row is None:
                continue
            if image_column >= len(cells):
                raise TargetEvaluationContractError(
                    f"Opaque Stage-70 image location is absent at row {contract_row_index}."
                )
            _record_access(
                access_log,
                ManifestAccessEvent(
                    phase="opaque_builder_image_read",
                    field="image_path",
                    contract_row_index=contract_row_index,
                ),
            )
            raw_image_path = cells[image_column]
            image_path = resolve_contract_path(resolved_repo_root, raw_image_path)
            try:
                image_path.relative_to(resolved_repo_root)
            except ValueError as exc:
                raise TargetEvaluationContractError(
                    "Opaque Stage-70 JPEG escaped the repository root at contract row "
                    f"{contract_row_index}."
                ) from exc
            if not image_path.is_file():
                raise TargetEvaluationContractError(
                    f"Opaque Stage-70 JPEG is missing at contract row {contract_row_index}."
                )
            try:
                jpeg_bytes = reader_fn(image_path)
            except OSError as exc:
                raise TargetEvaluationContractError(
                    f"Opaque Stage-70 JPEG is unreadable at contract row {contract_row_index}."
                ) from exc
            finally:
                # Make the lifetime of the sensitive source location explicit.
                del raw_image_path
                del image_path
            if not isinstance(jpeg_bytes, bytes) or not jpeg_bytes:
                raise TargetEvaluationContractError(
                    f"Opaque Stage-70 JPEG bytes are empty at row {contract_row_index}."
                )
            yielded.append(contract_row_index)
            yield OpaqueImageBytes(row=row, jpeg_bytes=jpeg_bytes)
    expected_indices = [row.contract_row_index for row in reservation.rows]
    if yielded != expected_indices:
        raise TargetEvaluationContractError(
            "Opaque Stage-70 image access did not cover reservation order exactly."
        )


def _read_image_bytes(path: Path) -> bytes:
    return path.read_bytes()


__all__ = ("OpaqueImageBytes", "iter_bound_image_bytes")
