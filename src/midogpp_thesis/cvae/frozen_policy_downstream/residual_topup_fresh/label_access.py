"""The sole post-seal label-opening boundary for fresh Stage 70."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .contracts import CENTERS
from .prediction_seal import PredictionSealCapability, validate_prediction_seal
from .target_cache import FreshTargetSurface


SCORING_SCHEMA = "midogpp_residual_topup_fresh_scoring_row_v1"
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
    """Parse labels only after the complete 1,053-cell menu is sealed."""

    summary = validate_prediction_seal(capability)
    if summary.prediction_cell_count != 1053 or not summary.row_coverage_complete:
        raise ProtocolError("Fresh scoring labels require a complete prediction seal.")
    if _sha256_file(surface.scoring_manifest_path) != surface.scoring_manifest_sha256:
        raise ProtocolError("Fresh scoring manifest changed after prediction sealing.")
    try:
        with surface.scoring_manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SCORING_COLUMNS:
                raise ProtocolError("Fresh scoring manifest columns drifted.")
            rows = tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ProtocolError("Cannot open fresh scoring labels after sealing.") from exc
    expected_rows = {
        row_id
        for center in CENTERS
        for row_id in surface.frames_by_center[center].evaluation_row_ids
    }
    labels: dict[str, int] = {}
    for row in rows:
        if (
            row.get("schema_version") != SCORING_SCHEMA
            or row.get("center") not in CENTERS
            or row.get("reservation_hash") != surface.reservation.reservation_hash
            or row.get("target_cache_content_hash") != surface.cache_content_hash
        ):
            raise ProtocolError("Fresh scoring manifest identity drifted.")
        row_id = str(row.get("row_id", ""))
        center = str(row.get("center", ""))
        case_id = str(row.get("case_id", ""))
        frame = surface.frames_by_center[center]
        expected_case = dict(
            zip(frame.evaluation_row_ids, frame.case_ids, strict=True)
        )
        if row_id in labels or expected_case.get(row_id) != case_id:
            raise ProtocolError("Fresh scoring manifest row identity drifted.")
        try:
            label = int(row.get("label", ""))
        except (TypeError, ValueError) as exc:
            raise ProtocolError(
                "Fresh scoring labels must be binary integers."
            ) from exc
        if label not in {0, 1} or str(label) != str(row.get("label")):
            raise ProtocolError("Fresh scoring labels must be binary integers.")
        labels[row_id] = label
    if set(labels) != expected_rows:
        raise ProtocolError(
            "Fresh scoring manifest does not exactly cover sealed rows."
        )
    return MappingProxyType(labels)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "SCORING_COLUMNS",
    "SCORING_SCHEMA",
    "open_scoring_labels_after_prediction_seal",
)
