"""Label-free frame scratch materialization and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import (
    atomic_json,
    atomic_npy,
    read_json,
    sha256_array,
    sha256_file,
)
from .constants import (
    CENTERS,
    FEATURE_DIM,
    SOURCE_CHECKPOINT_DIRECTORY,
    SOURCE_SCRATCH_ARRAY,
    TEST_CHECKPOINT_DIRECTORY,
    TEST_SCRATCH_ARRAY,
)
from .hashing import canonical_hash


def row_id(row: object) -> str:
    value = getattr(
        row,
        "source_row_id",
        getattr(row, "evaluation_row_id", getattr(row, "sample_id", None)),
    )
    if value is None or not str(value):
        raise ProtocolError("Prediction-only frame row lacks an opaque identity.")
    return str(value)


def write_frame_scratch(
    root: Path,
    *,
    frame_role: str,
    frame: object,
) -> Mapping[str, object]:
    if frame_role not in ("source", "test"):
        raise ProtocolError("Prediction-only scratch frame role is invalid.")
    checkpoint_dir = (
        SOURCE_CHECKPOINT_DIRECTORY
        if frame_role == "source"
        else TEST_CHECKPOINT_DIRECTORY
    )
    array_name = SOURCE_SCRATCH_ARRAY if frame_role == "source" else TEST_SCRATCH_ARRAY
    checkpoint_root = root / checkpoint_dir
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_root / f"{frame_role}_scratch.json"
    array_path = checkpoint_root / array_name
    binding_hash = str(getattr(frame, "cache_binding_hash"))
    if manifest_path.is_file() and array_path.is_file():
        payload = read_json(manifest_path)
        validate_frame_scratch(
            payload,
            frame_role=frame_role,
            expected_cache_binding_hash=binding_hash,
        )
        return payload
    rows: list[object] = []
    row_ids: dict[str, list[str]] = {}
    case_ids: dict[str, list[str]] = {}
    offsets: dict[str, dict[str, object]] = {}
    cursor = 0
    rows_by_center = getattr(frame, "rows_by_center")
    for center in CENTERS:
        center_rows = tuple(rows_by_center[center])
        identifiers = tuple(row_id(row) for row in center_rows)
        cases = tuple(str(getattr(row, "case_id")) for row in center_rows)
        offsets[center] = {
            "start": cursor,
            "stop": cursor + len(center_rows),
            "row_count": len(center_rows),
            "row_identity_hash": canonical_hash(list(identifiers)),
        }
        rows.extend(center_rows)
        row_ids[center] = list(identifiers)
        case_ids[center] = list(cases)
        cursor += len(center_rows)
    embeddings = np.ascontiguousarray(
        getattr(frame, "embeddings_for")(rows), dtype=np.float32
    )
    if embeddings.shape != (cursor, FEATURE_DIM) or not np.isfinite(embeddings).all():
        raise ProtocolError("Prediction-only scratch embedding geometry drifted.")
    for center in CENTERS:
        offset = offsets[center]
        offset["embedding_slice_sha256"] = sha256_array(
            embeddings[int(offset["start"]) : int(offset["stop"])]
        )
    atomic_npy(array_path, embeddings)
    unhashed = {
        "schema_version": "midogpp_prediction_only_frame_scratch_v1",
        "frame_role": frame_role,
        "array_path": str(array_path.resolve()),
        "array_file_sha256": sha256_file(array_path),
        "array_sha256": sha256_array(embeddings),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "cache_binding_hash": binding_hash,
        "offsets": offsets,
        "row_ids_by_center": row_ids,
        "case_ids_by_center": case_ids,
        "labels_stored": False,
        "manifest_opened": False,
    }
    payload = {**unhashed, "scratch_hash": canonical_hash(unhashed)}
    atomic_json(manifest_path, payload)
    return payload


def validate_frame_scratch(
    payload: Mapping[str, object],
    *,
    frame_role: str,
    expected_cache_binding_hash: str,
) -> None:
    path = Path(str(payload.get("array_path", "")))
    if not path.is_file():
        raise ProtocolError("Prediction-only scratch array is absent.")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    unhashed = {key: value for key, value in payload.items() if key != "scratch_hash"}
    offsets = payload.get("offsets")
    rows = payload.get("row_ids_by_center")
    cases = payload.get("case_ids_by_center")
    if (
        payload.get("scratch_hash") != canonical_hash(unhashed)
        or payload.get("frame_role") != frame_role
        or payload.get("cache_binding_hash") != expected_cache_binding_hash
        or payload.get("array_file_sha256") != sha256_file(path)
        or payload.get("array_sha256") != sha256_array(values)
        or payload.get("shape") != list(values.shape)
        or payload.get("dtype") != str(values.dtype)
        or values.ndim != 2
        or values.shape[1] != FEATURE_DIM
        or values.dtype != np.float32
        or not isinstance(offsets, Mapping)
        or not isinstance(rows, Mapping)
        or not isinstance(cases, Mapping)
        or tuple(offsets) != CENTERS
        or tuple(rows) != CENTERS
        or tuple(cases) != CENTERS
        or payload.get("labels_stored") is not False
        or payload.get("manifest_opened") is not False
    ):
        raise ProtocolError("Prediction-only frame scratch failed validation.")
    cursor = 0
    for center in CENTERS:
        raw = offsets[center]
        if not isinstance(raw, Mapping):
            raise ProtocolError("Prediction-only scratch offset is malformed.")
        start, stop = int(raw.get("start", -1)), int(raw.get("stop", -1))
        identities = tuple(str(value) for value in rows[center])
        center_cases = tuple(str(value) for value in cases[center])
        if (
            start != cursor
            or stop <= start
            or stop - start != len(identities)
            or len(center_cases) != len(identities)
            or raw.get("row_count") != len(identities)
            or raw.get("row_identity_hash") != canonical_hash(list(identities))
            or raw.get("embedding_slice_sha256") != sha256_array(values[start:stop])
        ):
            raise ProtocolError("Prediction-only scratch offset drifted.")
        cursor = stop
    if cursor != len(values):
        raise ProtocolError("Prediction-only scratch coverage drifted.")


__all__ = (
    "row_id",
    "validate_frame_scratch",
    "write_frame_scratch",
)
