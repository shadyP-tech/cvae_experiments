"""Capability-gated streaming access to consumed development labels.

The manifest-wide label column is never materialized.  Rows not named by the
durable global prediction seal are skipped before their label field is read.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    EXPECTED_MANIFEST_SHA256,
    OpenedLabelVector,
    ValidationRowIdentity,
    row_identity_hash,
)
from .seals import GlobalDevelopmentPredictionSeal


_REQUIRED_MANIFEST_FIELDS = frozenset(
    {"sample_id", "case_id", "center", "split", "label"}
)


def open_globally_sealed_development_labels(
    manifest_path: str | Path,
    evaluation_rows_by_query: Mapping[str, Sequence[ValidationRowIdentity]],
    *,
    seal: GlobalDevelopmentPredictionSeal,
    seal_path: str | Path,
    prediction_index_path: str | Path,
    prediction_arrays_path: str | Path,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> Mapping[str, OpenedLabelVector]:
    """Open query labels only after all 5,184 development cells are durable."""

    if not isinstance(seal, GlobalDevelopmentPredictionSeal):
        raise ProtocolError(
            "Local-utility labels require the complete global prediction seal."
        )
    seal.verify_complete()
    _verify_persisted_seal(seal_path, seal=seal)
    _verify_persisted_prediction_files(
        prediction_index_path=prediction_index_path,
        prediction_arrays_path=prediction_arrays_path,
        expected_index_sha256=seal.prediction_index_sha256,
        expected_arrays_sha256=seal.prediction_arrays_sha256,
    )
    if seal.validation_manifest_sha256 != expected_manifest_sha256:
        raise ProtocolError("Local-utility label capability binds another manifest.")

    normalized = {
        str(query): tuple(rows)
        for query, rows in evaluation_rows_by_query.items()
    }
    if tuple(normalized) != CENTERS:
        raise ProtocolError(
            "Local-utility label request lacks canonical query-center coverage."
        )
    all_rows: list[ValidationRowIdentity] = []
    for query in CENTERS:
        rows = normalized[query]
        _require_unique_requested_rows(rows)
        if any(
            row.partition_role != "evaluation" or row.center != query
            for row in rows
        ):
            raise ProtocolError(
                "Local-utility labels are limited to sealed q evaluation rows."
            )
        if tuple(row.sample_id for row in rows) != seal.evaluation_row_ids_by_query[query]:
            raise ProtocolError(
                "Local-utility label request differs from sealed row coverage."
            )
        if row_identity_hash(rows) != seal.evaluation_row_identity_hash_by_query[query]:
            raise ProtocolError(
                "Local-utility label request identities differ from the seal."
            )
        all_rows.extend(rows)
    _require_unique_requested_rows(tuple(all_rows))

    labels = _stream_requested_labels(
        manifest_path,
        tuple(all_rows),
        expected_manifest_sha256=expected_manifest_sha256,
    )
    labels_by_index = {
        row.manifest_row_index: label
        for row, label in zip(all_rows, labels, strict=True)
    }
    opened: dict[str, OpenedLabelVector] = {}
    for query in CENTERS:
        rows = normalized[query]
        query_labels = tuple(labels_by_index[row.manifest_row_index] for row in rows)
        if set(query_labels) != {0, 1}:
            raise ProtocolError(
                f"Local-utility query {query} lacks binary evaluation support."
            )
        vector_hash = stable_hash(
            {
                "query_center": query,
                "phase": "development_utility_surface",
                "row_identity_hash": row_identity_hash(rows),
                "labels": list(query_labels),
                "manifest_sha256": expected_manifest_sha256,
                "prediction_seal_hash": seal.seal_hash,
            }
        )
        opened[query] = OpenedLabelVector(
            query_center=query,
            rows=rows,
            labels=query_labels,
            manifest_sha256=expected_manifest_sha256,
            prediction_seal_hash=seal.seal_hash,
            label_vector_hash=vector_hash,
        )
    return opened


def _stream_requested_labels(
    manifest_path: str | Path,
    rows: tuple[ValidationRowIdentity, ...],
    *,
    expected_manifest_sha256: str,
) -> tuple[int, ...]:
    path = Path(manifest_path)
    _assert_sha256(path, expected_manifest_sha256)
    expected_by_index = {row.manifest_row_index: row for row in rows}
    labels_by_index: dict[int, int] = {}
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ProtocolError(
            f"Cannot open local-utility label manifest: {path}."
        ) from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not _REQUIRED_MANIFEST_FIELDS.issubset(
            reader.fieldnames
        ):
            raise ProtocolError(
                "Local-utility manifest lacks required scoring fields."
            )
        for manifest_row_index, raw in enumerate(reader):
            expected = expected_by_index.get(manifest_row_index)
            if expected is None:
                # This branch precedes the only label-field access.  Support,
                # train, test, and excluded rows therefore remain unopened.
                continue
            observed_identity = (
                str(raw.get("sample_id", "")),
                str(raw.get("case_id", "")),
                str(raw.get("center", "")),
                str(raw.get("split", "")),
            )
            expected_identity = (
                expected.sample_id,
                expected.case_id,
                expected.center,
                expected.split,
            )
            if observed_identity != expected_identity:
                raise ProtocolError(
                    "Local-utility scoring-manifest identity drifted."
                )
            labels_by_index[manifest_row_index] = _binary_label(raw["label"])
    if set(labels_by_index) != set(expected_by_index):
        raise ProtocolError(
            "Local-utility label coverage differs from sealed rows."
        )
    return tuple(labels_by_index[row.manifest_row_index] for row in rows)


def _verify_persisted_seal(
    path: str | Path,
    *,
    seal: GlobalDevelopmentPredictionSeal,
) -> None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            "Local-utility global prediction seal is not durably persisted."
        ) from exc
    if payload != seal.to_payload():
        raise ProtocolError(
            "Persisted local-utility seal differs from its capability."
        )


def _verify_persisted_prediction_files(
    *,
    prediction_index_path: str | Path,
    prediction_arrays_path: str | Path,
    expected_index_sha256: str,
    expected_arrays_sha256: str,
) -> None:
    index_path = Path(prediction_index_path)
    arrays_path = Path(prediction_arrays_path)
    if not index_path.is_file() or not arrays_path.is_file():
        raise ProtocolError(
            "Local-utility global prediction capability is not persisted."
        )
    if (
        _sha256_file(index_path) != expected_index_sha256
        or _sha256_file(arrays_path) != expected_arrays_sha256
    ):
        raise ProtocolError("Local-utility persisted prediction bytes drifted.")


def _require_unique_requested_rows(
    rows: tuple[ValidationRowIdentity, ...],
) -> None:
    if not rows:
        raise ProtocolError("Local-utility label request is empty.")
    sample_ids = [row.sample_id for row in rows]
    manifest_indices = [row.manifest_row_index for row in rows]
    if len(sample_ids) != len(set(sample_ids)) or len(manifest_indices) != len(
        set(manifest_indices)
    ):
        raise ProtocolError("Local-utility label request duplicates row identities.")


def _binary_label(value: object) -> int:
    try:
        numeric = float(str(value))
    except ValueError as exc:
        raise ProtocolError(
            "Local-utility requested scoring label is not binary."
        ) from exc
    if numeric not in (0.0, 1.0):
        raise ProtocolError(
            "Local-utility requested scoring label is outside {0,1}."
        )
    return int(numeric)


def _assert_sha256(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256_file(path) != expected:
        raise ProtocolError("Local-utility scoring-manifest SHA-256 drifted.")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = ("open_globally_sealed_development_labels",)
