"""Stable public reader for an already-authorized source-inner surface.

Governance and byte authorization remain consumer-owned.  This module owns
the historical on-disk schema and converts a byte-validated seven-member
packet into the scientific DTOs used by SCEPTRE.  Consumers therefore do not
couple themselves to private CSV/NPZ parser helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from midogpp_thesis.cvae.protocol import ProtocolError

from .development_surface import SourceInnerDevelopmentSurface
from .hashing import canonical_hash, file_sha256
from .source_inner_authorization import (
    CASE_CONFUSIONS_MEMBER,
    CLASSIFIER_FITS_MEMBER,
    EVALUATION_ROWS_MEMBER,
    PREDICTION_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
    UTILITY_LOCK_MEMBER,
    UTILITY_TABLE_MEMBER,
    _read_utility_cells,
    _require_csv_row_count,
)
from .source_inner_evidence import (
    PredictionSurfaceReceipt,
    SourceInnerPredictionSurface,
    _array_sha256,
    _load_arrays,
    _load_evaluation_rows,
    _load_fit_rows,
    _load_index,
    _surface_receipt_body,
)


SOURCE_INNER_PUBLIC_READER_SCHEMA = "sceptre_source_inner_public_reader_v1"


def load_authorized_source_inner_surfaces(
    artifact_root: str | Path,
    *,
    amendment_sha256: str,
    expected_member_sha256: Mapping[str, str],
    expected_case_confusion_rows: int,
    expected_classifier_fit_rows: int,
    expected_evaluation_rows: int,
) -> tuple[SourceInnerDevelopmentSurface, SourceInnerPredictionSurface]:
    """Decode one byte-authorized source-inner packet without policy decisions."""

    root = Path(artifact_root)
    expected = {str(name): str(digest) for name, digest in expected_member_sha256.items()}
    required = {
        UTILITY_LOCK_MEMBER,
        UTILITY_TABLE_MEMBER,
        CASE_CONFUSIONS_MEMBER,
        PREDICTION_ARRAY_MEMBER,
        PREDICTION_INDEX_MEMBER,
        CLASSIFIER_FITS_MEMBER,
        EVALUATION_ROWS_MEMBER,
    }
    if (
        root.is_symlink()
        or not root.is_dir()
        or set(expected) != required
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (
            expected_case_confusion_rows,
            expected_classifier_fit_rows,
            expected_evaluation_rows,
        ))
    ):
        raise ProtocolError("SCEPTRE source-inner public reader contract drifted.")
    members = {relative: _safe_member(root, relative) for relative in expected}
    observed = {relative: file_sha256(path) for relative, path in members.items()}
    if observed != expected:
        raise ProtocolError("SCEPTRE source-inner public reader bytes drifted.")

    cells = _read_utility_cells(members[UTILITY_TABLE_MEMBER])
    _require_csv_row_count(
        members[CASE_CONFUSIONS_MEMBER], expected_case_confusion_rows
    )
    development = SourceInnerDevelopmentSurface(
        cells=cells,
        utility_lock_sha256=observed[UTILITY_LOCK_MEMBER],
        utility_table_sha256=observed[UTILITY_TABLE_MEMBER],
        case_confusions_sha256=observed[CASE_CONFUSIONS_MEMBER],
        amendment_sha256=amendment_sha256,
    )

    index = _load_index(members[PREDICTION_INDEX_MEMBER])
    fit_rows = _load_fit_rows(members[CLASSIFIER_FITS_MEMBER])
    evaluation_rows = _load_evaluation_rows(members[EVALUATION_ROWS_MEMBER])
    prob_pos, y_pred = _load_arrays(members[PREDICTION_ARRAY_MEMBER], index=index)
    if (
        len(fit_rows) != expected_classifier_fit_rows
        or len(evaluation_rows) != expected_evaluation_rows
    ):
        raise ProtocolError("SCEPTRE source-inner public reader geometry drifted.")
    receipt_body = _surface_receipt_body(
        prediction_array_file_sha256=observed[PREDICTION_ARRAY_MEMBER],
        prediction_index_sha256=observed[PREDICTION_INDEX_MEMBER],
        classifier_fits_sha256=observed[CLASSIFIER_FITS_MEMBER],
        evaluation_rows_sha256=observed[EVALUATION_ROWS_MEMBER],
        fit_count=len(fit_rows),
        evaluation_row_count=len(evaluation_rows),
        probability_array_sha256=_array_sha256(prob_pos),
        prediction_array_sha256=_array_sha256(y_pred),
    )
    prediction_receipt = PredictionSurfaceReceipt(
        prediction_array_file_sha256=observed[PREDICTION_ARRAY_MEMBER],
        prediction_index_sha256=observed[PREDICTION_INDEX_MEMBER],
        classifier_fits_sha256=observed[CLASSIFIER_FITS_MEMBER],
        evaluation_rows_sha256=observed[EVALUATION_ROWS_MEMBER],
        fit_count=len(fit_rows),
        evaluation_row_count=len(evaluation_rows),
        probability_array_sha256=_array_sha256(prob_pos),
        prediction_array_sha256=_array_sha256(y_pred),
        labels_stored=False,
        receipt_hash=canonical_hash(receipt_body),
    )
    prediction = SourceInnerPredictionSurface(
        prob_pos=prob_pos,
        y_pred=y_pred,
        fit_rows=fit_rows,
        evaluation_rows=evaluation_rows,
        receipt=prediction_receipt,
    )
    return development, prediction


def _safe_member(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*relative.split("/"))
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(
            f"SCEPTRE source-inner member is absent: {relative}."
        ) from exc
    if candidate.is_symlink() or resolved_root not in resolved.parents or not resolved.is_file():
        raise ProtocolError(f"SCEPTRE source-inner member is unsafe: {relative}.")
    return resolved


__all__ = (
    "SOURCE_INNER_PUBLIC_READER_SCHEMA",
    "load_authorized_source_inner_surfaces",
)
