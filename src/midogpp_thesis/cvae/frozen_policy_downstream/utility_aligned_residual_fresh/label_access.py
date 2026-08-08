"""The sole post-seal target-label opening boundary."""

from __future__ import annotations

import csv
import hashlib
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .contracts import CENTERS, EXPECTED_LOGICAL_PREDICTION_COUNT
from .prediction_seal import PredictionSealCapability, validate_prediction_seal
from .target_surface import FreshTargetSurface


SCORING_SCHEMA = "midogpp_utility_aligned_fresh_scoring_row_v1"
SCORING_COLUMNS = (
    "schema_version",
    "row_id",
    "center",
    "case_id",
    "label",
    "reservation_hash",
    "target_cache_content_hash",
)


def open_scoring_labels_after_prediction_seal(
    surface: FreshTargetSurface,
    capability: PredictionSealCapability,
) -> Mapping[str, int]:
    """Parse labels only after all 1,053 logical predictions are sealed."""

    summary = validate_prediction_seal(capability)
    if (
        summary.logical_prediction_count != EXPECTED_LOGICAL_PREDICTION_COUNT
        or not summary.logical_action_coverage_complete
        or not summary.row_coverage_complete
    ):
        raise ProtocolError("Utility-aligned scoring requires a complete global seal.")
    if _sha256_file(surface.scoring_manifest_path) != surface.scoring_manifest_sha256:
        raise ProtocolError("Utility-aligned scoring manifest changed after sealing.")
    try:
        with surface.scoring_manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SCORING_COLUMNS:
                raise ProtocolError("Utility-aligned scoring columns drifted.")
            rows = tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ProtocolError("Cannot open utility-aligned scoring labels.") from exc
    expected = {
        row_id
        for center in CENTERS
        for row_id in surface.frames_by_center[center].evaluation_row_ids
    }
    labels: dict[str, int] = {}
    for row in rows:
        center = str(row.get("center", ""))
        row_id = str(row.get("row_id", ""))
        if (
            row.get("schema_version") != SCORING_SCHEMA
            or center not in CENTERS
            or row.get("reservation_hash") != surface.reservation.reservation_hash
            or row.get("target_cache_content_hash") != surface.cache_content_hash
            or row_id in labels
        ):
            raise ProtocolError("Utility-aligned scoring identity drifted.")
        frame = surface.frames_by_center[center]
        case_by_row = dict(zip(frame.evaluation_row_ids, frame.case_ids, strict=True))
        if case_by_row.get(row_id) != str(row.get("case_id", "")):
            raise ProtocolError("Utility-aligned scoring row/case binding drifted.")
        try:
            label = int(row.get("label", ""))
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Utility-aligned scoring labels must be binary.") from exc
        if label not in {0, 1} or str(label) != str(row.get("label")):
            raise ProtocolError("Utility-aligned scoring labels must be binary.")
        labels[row_id] = label
    if set(labels) != expected:
        raise ProtocolError("Utility-aligned scoring rows do not match the seal.")
    return MappingProxyType(labels)


def _sha256_file(path: object) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:  # noqa: PTH123 - accepts Path and test doubles.
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "SCORING_COLUMNS",
    "SCORING_SCHEMA",
    "open_scoring_labels_after_prediction_seal",
)
