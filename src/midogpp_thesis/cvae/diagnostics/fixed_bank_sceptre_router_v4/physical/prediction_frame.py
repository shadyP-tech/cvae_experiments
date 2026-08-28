"""Label-free evaluation-frame staging for SCEPTRE v4 prediction workers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_array, sha256_file

from .prediction_contracts import (
    EVALUATION_FRAME_MEMBER,
    EVALUATION_SCRATCH_MEMBER,
    PredictionGeometry,
)
from .prediction_io import canonical_sha256, persist_exact_json, persist_exact_npy


def stage_evaluation_frame(
    root: Path,
    frame: object,
    *,
    geometry: PredictionGeometry,
    attempt_id: str,
) -> Mapping[str, object]:
    assert_label_free_frame(frame)
    rows_by_center = getattr(frame, "rows_by_center", None)
    if not isinstance(rows_by_center, Mapping) or tuple(rows_by_center) != geometry.centers:
        raise ProtocolError("SCEPTRE v4 label-free row inventory is unavailable.")
    expected_counts = dict(geometry.rows_by_center)
    rows: list[object] = []
    row_ids: list[str] = []
    row_centers: list[str] = []
    offsets: dict[str, dict[str, int]] = {}
    cursor = 0
    for center in geometry.centers:
        center_rows = tuple(rows_by_center[center])
        if len(center_rows) != expected_counts[center]:
            raise ProtocolError("SCEPTRE v4 label-free rows-by-center drifted.")
        start = cursor
        for row in center_rows:
            assert_label_free_row(row, expected_center=center)
            rows.append(row)
            row_ids.append(opaque_row_id(row))
            row_centers.append(center)
            cursor += 1
        offsets[center] = {"start": start, "stop": cursor}
    if cursor != geometry.evaluation_rows or len(set(row_ids)) != len(row_ids):
        raise ProtocolError("SCEPTRE v4 label-free row coverage drifted.")
    embeddings_for = getattr(frame, "embeddings_for", None)
    if callable(embeddings_for):
        embeddings = np.asarray(embeddings_for(rows))
    else:
        embeddings = np.asarray(getattr(frame, "embeddings", None))
    values = np.ascontiguousarray(embeddings, dtype=np.float32)
    if (
        values.shape != (geometry.evaluation_rows, geometry.feature_dim)
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ProtocolError("SCEPTRE v4 label-free evaluation embeddings drifted.")
    cache_binding_hash = str(getattr(frame, "cache_binding_hash", ""))
    if not cache_binding_hash:
        raise ProtocolError("SCEPTRE v4 label-free cache binding hash is absent.")
    binding = getattr(frame, "cache_binding", None)
    if isinstance(binding, Mapping):
        if any(
            binding.get(key) is True
            for key in (
                "labels_persisted",
                "manifest_opened",
                "sample_paths_persisted",
                "raw_sample_paths_available",
            )
        ):
            raise ProtocolError("SCEPTRE v4 frame escaped the label-free path-free boundary.")
    array_path = root / EVALUATION_SCRATCH_MEMBER
    persist_exact_npy(array_path, values, role="evaluation scratch")
    frame_unhashed = {
        "schema_version": "midogpp_sceptre_v4_physical_label_free_evaluation_frame_v1",
        "attempt_id": attempt_id,
        "cache_binding_hash": cache_binding_hash,
        "evaluation_array_file_sha256": sha256_file(array_path),
        "evaluation_array_sha256": sha256_array(values),
        "shape": list(values.shape),
        "dtype": "float32",
        "row_ids": row_ids,
        "row_centers": row_centers,
        "offsets": offsets,
        "manifest_opened": False,
        "outcomes_available": False,
        "raw_sample_paths_available": False,
    }
    payload = {
        **frame_unhashed,
        "frame_sha256": canonical_sha256(frame_unhashed),
        "evaluation_array_path": str(array_path.resolve()),
    }
    persist_exact_json(root / EVALUATION_FRAME_MEMBER, payload)
    return payload


def assert_label_free_frame(frame: object) -> None:
    forbidden = (
        "labels",
        "targets",
        "truth",
        "outcomes",
        "manifest",
        "manifest_path",
        "sample_paths",
        "raw_sample_paths",
    )
    if any(hasattr(frame, name) for name in forbidden):
        raise ProtocolError("SCEPTRE v4 physical frame exposes labels or provenance paths.")


def assert_label_free_row(row: object, *, expected_center: str) -> None:
    if str(getattr(row, "center", "")) != expected_center:
        raise ProtocolError("SCEPTRE v4 label-free row center drifted.")
    forbidden = (
        "label",
        "target_label",
        "truth",
        "image_path",
        "sample_path",
        "raw_path",
        "file_path",
    )
    if any(hasattr(row, name) for name in forbidden):
        raise ProtocolError("SCEPTRE v4 prediction row exposes an outcome or raw path.")


def opaque_row_id(row: object) -> str:
    value = getattr(row, "evaluation_row_id", getattr(row, "sample_id", None))
    text = str(value) if value is not None else ""
    if not text:
        raise ProtocolError("SCEPTRE v4 prediction row lacks an opaque identity.")
    return text


__all__ = (
    "assert_label_free_frame",
    "assert_label_free_row",
    "opaque_row_id",
    "stage_evaluation_frame",
)
