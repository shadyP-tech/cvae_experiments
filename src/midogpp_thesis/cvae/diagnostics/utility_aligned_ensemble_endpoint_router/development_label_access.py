"""Capability-gated access to development evaluation labels only."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import sha256_file
from .contracts import CENTERS
from .development_seal import DevelopmentPredictionCapability, validate_global_development_seal
from .input_contracts import row_identity_hash


@dataclass(frozen=True)
class OpenedDevelopmentLabels:
    labels_by_center: Mapping[str, tuple[int, ...]]
    evaluation_row_hash_by_center: Mapping[str, str]
    label_hash_by_center: Mapping[str, str]
    manifest_sha256: str
    prediction_seal_hash: str
    capability_hash: str

    def __post_init__(self) -> None:
        labels = {str(key): tuple(int(value) for value in values) for key, values in self.labels_by_center.items()}
        rows = {str(key): str(value) for key, value in self.evaluation_row_hash_by_center.items()}
        hashes = {str(key): str(value) for key, value in self.label_hash_by_center.items()}
        if tuple(labels) != CENTERS or tuple(rows) != CENTERS or tuple(hashes) != CENTERS:
            raise ProtocolError("Opened development label center coverage drifted.")
        if any(not values or set(values) != {0, 1} for values in labels.values()):
            raise ProtocolError("Opened development labels lack both classes.")
        expected = _capability_payload(
            rows=rows, hashes=hashes, manifest_sha256=self.manifest_sha256,
            prediction_seal_hash=self.prediction_seal_hash,
        )
        if self.capability_hash != stable_hash(expected):
            raise ProtocolError("Opened development label capability drifted.")
        object.__setattr__(self, "labels_by_center", MappingProxyType(labels))
        object.__setattr__(self, "evaluation_row_hash_by_center", MappingProxyType(rows))
        object.__setattr__(self, "label_hash_by_center", MappingProxyType(hashes))


def open_globally_sealed_development_labels(
    manifest_path: str | Path,
    partitions: object,
    *,
    capability: DevelopmentPredictionCapability,
) -> OpenedDevelopmentLabels:
    seal = validate_global_development_seal(capability)
    manifest = Path(manifest_path)
    if sha256_file(manifest) != seal.get("development_manifest_sha256"):
        raise ProtocolError("Development manifest drifted from the durable seal.")
    if str(partitions.lock_hash) != seal.get("partition_lock_hash"):
        raise ProtocolError("Development label request uses another partition lock.")
    requested = tuple(
        row for center in CENTERS for row in partitions.evaluation_rows_by_center[center]
    )
    support_indices = {
        row.manifest_row_index for center in CENTERS for row in partitions.support_rows_by_center[center]
    }
    if support_indices & {row.manifest_row_index for row in requested}:
        raise ProtocolError("Development evaluation rows overlap fixed support.")
    expected_hashes = seal.get("evaluation_row_hash_by_center")
    if not isinstance(expected_hashes, Mapping) or any(
        row_identity_hash(partitions.evaluation_rows_by_center[center]) != expected_hashes.get(center)
        for center in CENTERS
    ):
        raise ProtocolError("Development evaluation row request differs from the seal.")
    by_index = _stream_labels(manifest, requested)
    labels = {
        center: tuple(by_index[row.manifest_row_index] for row in partitions.evaluation_rows_by_center[center])
        for center in CENTERS
    }
    label_hashes = {center: stable_hash(list(labels[center])) for center in CENTERS}
    rows = {center: str(expected_hashes[center]) for center in CENTERS}
    payload = _capability_payload(
        rows=rows, hashes=label_hashes, manifest_sha256=str(seal["development_manifest_sha256"]),
        prediction_seal_hash=str(seal["prediction_seal_hash"]),
    )
    return OpenedDevelopmentLabels(
        labels_by_center=labels, evaluation_row_hash_by_center=rows,
        label_hash_by_center=label_hashes,
        manifest_sha256=str(seal["development_manifest_sha256"]),
        prediction_seal_hash=str(seal["prediction_seal_hash"]),
        capability_hash=stable_hash(payload),
    )


def _stream_labels(path: Path, rows: tuple[object, ...]) -> dict[int, int]:
    expected = {int(row.manifest_row_index): row for row in rows}
    labels: dict[int, int] = {}
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ProtocolError("Cannot open development scoring manifest.") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "case_id", "center", "split", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ProtocolError("Development manifest fields drifted.")
        for index, raw in enumerate(reader):
            wanted = expected.get(index)
            if wanted is None:
                continue
            observed = (raw["case_id"], raw["center"], raw["split"])
            expected_identity = (wanted.case_id, wanted.center, wanted.split)
            opaque_test_identity = hasattr(wanted, "evaluation_row_id")
            if (
                observed != expected_identity
                or (
                    not opaque_test_identity
                    and raw["sample_id"] != wanted.sample_id
                )
            ):
                raise ProtocolError("Development manifest identity drifted.")
            labels[index] = _binary(raw["label"])
    if set(labels) != set(expected):
        raise ProtocolError("Development label coverage drifted.")
    return labels


def _binary(value: object) -> int:
    try:
        number = float(str(value))
    except ValueError as exc:
        raise ProtocolError("Development label is not numeric.") from exc
    if number not in (0.0, 1.0):
        raise ProtocolError("Development label is outside {0,1}.")
    return int(number)


def _capability_payload(
    *, rows: Mapping[str, str], hashes: Mapping[str, str], manifest_sha256: str,
    prediction_seal_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_stage90_ensemble_endpoint_opened_development_labels_v1",
        "evaluation_row_hash_by_center": dict(rows),
        "label_hash_by_center": dict(hashes),
        "manifest_sha256": manifest_sha256,
        "prediction_seal_hash": prediction_seal_hash,
        "evaluation_labels_used_for_scoring_only": True,
        "support_labels_opened": False,
        "diagnostic_only": True,
    }


__all__ = ("OpenedDevelopmentLabels", "open_globally_sealed_development_labels")
