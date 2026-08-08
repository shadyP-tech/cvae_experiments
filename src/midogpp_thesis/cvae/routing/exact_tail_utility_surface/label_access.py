"""Capability-gated access to fresh development evaluation labels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import CENTERS, DevelopmentPartition, EvaluationRowIdentity, row_identity_hash
from .seals import GlobalPredictionSeal


_REQUIRED_FIELDS = frozenset(
    {"manifest_row_index", "sample_id", "case_id", "center", "split", "label"}
)


@dataclass(frozen=True)
class OpenedDevelopmentLabels:
    labels_by_center: Mapping[str, tuple[int, ...]]
    row_hash_by_center: Mapping[str, str]
    manifest_sha256: str
    prediction_seal_hash: str
    capability_hash: str

    def __post_init__(self) -> None:
        labels = {
            str(center): tuple(int(value) for value in values)
            for center, values in self.labels_by_center.items()
        }
        rows = {str(center): str(value) for center, value in self.row_hash_by_center.items()}
        if tuple(labels) != CENTERS or tuple(rows) != CENTERS:
            raise ProtocolError("Exact-tail opened labels lack canonical center coverage.")
        if any(not values or set(values) != {0, 1} for values in labels.values()):
            raise ProtocolError("Exact-tail opened labels lack binary coverage.")
        payload = {
            "schema_version": "midogpp_exact_tail_opened_development_labels_v1",
            "row_hash_by_center": rows,
            "label_hash_by_center": {
                center: stable_hash(list(labels[center])) for center in CENTERS
            },
            "manifest_sha256": self.manifest_sha256,
            "prediction_seal_hash": self.prediction_seal_hash,
            "development_labels_used_for_scoring_only": True,
            "target_support_labels_used": False,
            "target_evaluation_labels_used": False,
        }
        if self.capability_hash != stable_hash(payload):
            raise ProtocolError("Exact-tail opened-label capability hash drifted.")
        object.__setattr__(self, "labels_by_center", MappingProxyType(labels))
        object.__setattr__(self, "row_hash_by_center", MappingProxyType(rows))


def open_globally_sealed_development_labels(
    manifest_path: str | Path,
    partitions: Mapping[str, DevelopmentPartition],
    *,
    seal: GlobalPredictionSeal,
    seal_path: str | Path,
    prediction_index_path: str | Path,
    prediction_arrays_path: str | Path,
) -> OpenedDevelopmentLabels:
    """Open only named development rows after the complete seal is durable."""

    if not isinstance(seal, GlobalPredictionSeal):
        raise ProtocolError("Exact-tail labels require a global prediction seal.")
    seal.verify_complete()
    _verify_persisted_seal(Path(seal_path), seal)
    _assert_sha256(Path(prediction_index_path), seal.prediction_index_sha256)
    _assert_sha256(Path(prediction_arrays_path), seal.prediction_arrays_sha256)
    manifest = Path(manifest_path)
    _assert_sha256(manifest, seal.development_manifest_sha256)

    normalized = {str(center): partition for center, partition in partitions.items()}
    if tuple(normalized) != CENTERS:
        raise ProtocolError("Exact-tail label request lacks all center partitions.")
    requested: list[EvaluationRowIdentity] = []
    for center in CENTERS:
        partition = normalized[center]
        if (
            not isinstance(partition, DevelopmentPartition)
            or partition.reservation_hash != seal.partition_hash_by_center[center]
            or row_identity_hash(partition.evaluation_rows)
            != seal.evaluation_row_hash_by_center[center]
        ):
            raise ProtocolError("Exact-tail label request differs from sealed partitions.")
        requested.extend(partition.evaluation_rows)
    sample_ids = [row.sample_id for row in requested]
    manifest_indices = [row.manifest_row_index for row in requested]
    if len(sample_ids) != len(set(sample_ids)) or len(manifest_indices) != len(
        set(manifest_indices)
    ):
        raise ProtocolError("Exact-tail label request duplicates row identities.")

    labels_by_index = _stream_requested_labels(manifest, tuple(requested))
    labels_by_center = {
        center: tuple(
            labels_by_index[row.manifest_row_index]
            for row in normalized[center].evaluation_rows
        )
        for center in CENTERS
    }
    payload = {
        "schema_version": "midogpp_exact_tail_opened_development_labels_v1",
        "row_hash_by_center": dict(seal.evaluation_row_hash_by_center),
        "label_hash_by_center": {
            center: stable_hash(list(labels_by_center[center])) for center in CENTERS
        },
        "manifest_sha256": seal.development_manifest_sha256,
        "prediction_seal_hash": seal.seal_hash,
        "development_labels_used_for_scoring_only": True,
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
    }
    return OpenedDevelopmentLabels(
        labels_by_center=labels_by_center,
        row_hash_by_center=seal.evaluation_row_hash_by_center,
        manifest_sha256=seal.development_manifest_sha256,
        prediction_seal_hash=seal.seal_hash,
        capability_hash=stable_hash(payload),
    )


def _stream_requested_labels(
    path: Path, rows: tuple[EvaluationRowIdentity, ...]
) -> dict[int, int]:
    expected = {row.manifest_row_index: row for row in rows}
    labels: dict[int, int] = {}
    observed_row_count = 0
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ProtocolError(f"Cannot open exact-tail development manifest: {path}.") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not _REQUIRED_FIELDS.issubset(reader.fieldnames):
            raise ProtocolError("Exact-tail manifest lacks required scoring fields.")
        for raw in reader:
            observed_row_count += 1
            try:
                index = int(str(raw.get("manifest_row_index", "")))
            except ValueError as exc:
                raise ProtocolError(
                    "Exact-tail scoring manifest row index is malformed."
                ) from exc
            wanted = expected.get(index)
            if wanted is None:
                raise ProtocolError(
                    "Exact-tail scoring manifest contains a row outside the "
                    "globally sealed development-evaluation set."
                )
            identity = (
                str(raw.get("sample_id", "")),
                str(raw.get("case_id", "")),
                str(raw.get("center", "")),
                str(raw.get("split", "")),
            )
            if identity != (
                wanted.sample_id,
                wanted.case_id,
                wanted.center,
                wanted.split,
            ):
                raise ProtocolError("Exact-tail scoring-manifest identity drifted.")
            labels[index] = _binary_label(raw["label"])
    if set(labels) != set(expected):
        raise ProtocolError("Exact-tail label coverage differs from sealed rows.")
    if observed_row_count != len(expected):
        raise ProtocolError(
            "Exact-tail scoring manifest must contain exactly the sealed rows."
        )
    return labels


def _verify_persisted_seal(path: Path, seal: GlobalPredictionSeal) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Exact-tail global seal is not durably persisted.") from exc
    if payload != seal.to_payload():
        raise ProtocolError("Persisted exact-tail seal differs from its capability.")


def _binary_label(value: object) -> int:
    try:
        numeric = float(str(value))
    except ValueError as exc:
        raise ProtocolError("Exact-tail requested label is not binary.") from exc
    if numeric not in (0.0, 1.0):
        raise ProtocolError("Exact-tail requested label is outside {0,1}.")
    return int(numeric)


def _assert_sha256(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256_file(path) != expected:
        raise ProtocolError("Exact-tail persisted bytes drifted from their seal.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ("OpenedDevelopmentLabels", "open_globally_sealed_development_labels")
