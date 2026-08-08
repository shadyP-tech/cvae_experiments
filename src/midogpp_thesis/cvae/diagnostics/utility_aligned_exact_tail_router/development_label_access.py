"""Capability-gated streaming access to sealed development evaluation labels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import CENTERS
from .development_seal import (
    DevelopmentPredictionCapability,
    GlobalDevelopmentPredictionSeal,
)
from .input_contracts import row_identity_hash
from .source_cache_store import read_json, sha256_file


_REQUIRED_MANIFEST_FIELDS = frozenset(
    {"sample_id", "case_id", "center", "split", "label"}
)


@dataclass(frozen=True)
class OpenedDevelopmentLabels:
    labels_by_center: Mapping[str, tuple[int, ...]]
    evaluation_row_hash_by_center: Mapping[str, str]
    label_hash_by_center: Mapping[str, str]
    manifest_sha256: str
    prediction_seal_hash: str
    capability_hash: str

    def __post_init__(self) -> None:
        labels = {
            str(center): tuple(int(value) for value in values)
            for center, values in self.labels_by_center.items()
        }
        rows = {
            str(center): str(value)
            for center, value in self.evaluation_row_hash_by_center.items()
        }
        hashes = {
            str(center): str(value) for center, value in self.label_hash_by_center.items()
        }
        if tuple(labels) != CENTERS or tuple(rows) != CENTERS or tuple(hashes) != CENTERS:
            raise ProtocolError("Stage-90 opened-label center coverage drifted.")
        if any(not values or set(values) != {0, 1} for values in labels.values()):
            raise ProtocolError("Stage-90 opened evaluation labels lack both classes.")
        expected_hashes = {
            center: stable_hash(list(labels[center])) for center in CENTERS
        }
        if hashes != expected_hashes:
            raise ProtocolError("Stage-90 opened-label vector hash drifted.")
        payload = {
            "schema_version": "midogpp_stage90_utility_aligned_opened_labels_v1",
            "evaluation_row_hash_by_center": rows,
            "label_hash_by_center": hashes,
            "manifest_sha256": self.manifest_sha256,
            "prediction_seal_hash": self.prediction_seal_hash,
            "development_labels_used_for_scoring_only": True,
            "support_labels_opened": False,
            "labels_persisted_in_prediction_store": False,
            "diagnostic_only": True,
        }
        if self.capability_hash != stable_hash(payload):
            raise ProtocolError("Stage-90 opened-label capability hash drifted.")
        object.__setattr__(self, "labels_by_center", MappingProxyType(labels))
        object.__setattr__(self, "evaluation_row_hash_by_center", MappingProxyType(rows))
        object.__setattr__(self, "label_hash_by_center", MappingProxyType(hashes))


def open_globally_sealed_development_labels(
    manifest_path: str | Path,
    partitions: object,
    *,
    capability: DevelopmentPredictionCapability,
) -> OpenedDevelopmentLabels:
    """Stream only E_q labels after the complete prediction seal is durable."""

    if not isinstance(capability, DevelopmentPredictionCapability) or not isinstance(
        capability.seal, GlobalDevelopmentPredictionSeal
    ):
        raise ProtocolError("Stage-90 labels require a typed global prediction capability.")
    seal = capability.seal
    if read_json(capability.seal_path) != seal.to_payload():
        raise ProtocolError("Stage-90 prediction seal is not durably persisted.")
    if (
        sha256_file(capability.prediction_index_path) != seal.prediction_index_sha256
        or sha256_file(capability.prediction_arrays_path)
        != seal.prediction_arrays_sha256
    ):
        raise ProtocolError("Stage-90 sealed prediction bytes drifted.")
    manifest = Path(manifest_path)
    if not manifest.is_file() or sha256_file(manifest) != seal.development_manifest_sha256:
        raise ProtocolError("Stage-90 development manifest drifted from the seal.")
    if str(getattr(partitions, "lock_hash", "")) != seal.partition_lock_hash:
        raise ProtocolError("Stage-90 label request uses another partition lock.")
    by_center = getattr(partitions, "evaluation_rows_by_center", None)
    if not isinstance(by_center, Mapping) or tuple(by_center) != CENTERS:
        raise ProtocolError("Stage-90 evaluation label request is malformed.")
    requested = tuple(row for center in CENTERS for row in by_center[center])
    if (
        len({row.sample_id for row in requested}) != len(requested)
        or len({row.manifest_row_index for row in requested}) != len(requested)
    ):
        raise ProtocolError("Stage-90 evaluation label request duplicates rows.")
    for center in CENTERS:
        if row_identity_hash(by_center[center]) != seal.evaluation_row_hash_by_center[center]:
            raise ProtocolError("Stage-90 evaluation row request differs from the seal.")
    labels_by_index = _stream_requested_labels(manifest, requested)
    labels = {
        center: tuple(
            labels_by_index[row.manifest_row_index] for row in by_center[center]
        )
        for center in CENTERS
    }
    label_hashes = {center: stable_hash(list(labels[center])) for center in CENTERS}
    payload = {
        "schema_version": "midogpp_stage90_utility_aligned_opened_labels_v1",
        "evaluation_row_hash_by_center": dict(seal.evaluation_row_hash_by_center),
        "label_hash_by_center": label_hashes,
        "manifest_sha256": seal.development_manifest_sha256,
        "prediction_seal_hash": seal.prediction_seal_hash,
        "development_labels_used_for_scoring_only": True,
        "support_labels_opened": False,
        "labels_persisted_in_prediction_store": False,
        "diagnostic_only": True,
    }
    return OpenedDevelopmentLabels(
        labels_by_center=labels,
        evaluation_row_hash_by_center=seal.evaluation_row_hash_by_center,
        label_hash_by_center=label_hashes,
        manifest_sha256=seal.development_manifest_sha256,
        prediction_seal_hash=seal.prediction_seal_hash,
        capability_hash=stable_hash(payload),
    )


def _stream_requested_labels(path: Path, rows: tuple[object, ...]) -> dict[int, int]:
    expected = {int(row.manifest_row_index): row for row in rows}
    labels: dict[int, int] = {}
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ProtocolError(f"Cannot open Stage-90 development manifest: {path}.") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not _REQUIRED_MANIFEST_FIELDS.issubset(
            reader.fieldnames
        ):
            raise ProtocolError("Stage-90 manifest lacks required scoring fields.")
        for manifest_row_index, raw in enumerate(reader):
            wanted = expected.get(manifest_row_index)
            if wanted is None:
                # The label field is intentionally not touched on skipped rows.
                continue
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
                raise ProtocolError("Stage-90 scoring-manifest identity drifted.")
            labels[manifest_row_index] = _binary_label(raw["label"])
    if set(labels) != set(expected):
        raise ProtocolError("Stage-90 label coverage differs from sealed E_q rows.")
    return labels


def _binary_label(value: object) -> int:
    try:
        numeric = float(str(value))
    except ValueError as exc:
        raise ProtocolError("Stage-90 requested label is not binary.") from exc
    if numeric not in (0.0, 1.0):
        raise ProtocolError("Stage-90 requested label is outside {0,1}.")
    return int(numeric)


__all__ = (
    "OpenedDevelopmentLabels",
    "open_globally_sealed_development_labels",
)
