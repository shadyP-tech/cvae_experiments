"""Label-free validation-cache loading and delayed manifest-label opening."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    EXPECTED_EVAL_CASES,
    EXPECTED_EVAL_ROWS,
    EXPECTED_MANIFEST_SHA256,
    FEATURE_DIM,
    EvaluationRow,
    evaluation_order_hash,
)


_CACHE_METADATA_KEYS = frozenset(
    {"sample_id", "case_id", "split", "center", "manifest_row_index"}
)
_FORBIDDEN_CACHE_METADATA_KEYS = frozenset(
    {
        "label",
        "labels",
        "y",
        "y_true",
        "target",
        "class",
        "class_label",
        "label_name",
    }
)


@dataclass(frozen=True)
class ManifestEvaluationRow:
    manifest_row_index: int
    sample_id: str
    case_id: str
    center: str
    split: str = "val"


@dataclass(frozen=True)
class UnlabeledValidationFrame:
    """Evaluation embeddings and identities, structurally incapable of labels."""

    embeddings: np.ndarray
    rows: tuple[EvaluationRow, ...]
    cache_binding: Mapping[str, object]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.embeddings)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.rows):
            raise ProtocolError("Unlabeled validation embeddings and rows do not align.")
        if matrix.shape[1] <= 0 or not np.isfinite(matrix).all():
            raise ProtocolError("Unlabeled validation embeddings failed shape/finiteness checks.")
        if tuple(row.row_ordinal for row in self.rows) != tuple(range(len(self.rows))):
            raise ProtocolError("Unlabeled validation row ordinals are not canonical.")
        sample_ids = tuple(row.sample_id for row in self.rows)
        if len(sample_ids) != len(set(sample_ids)):
            raise ProtocolError("Unlabeled validation sample IDs are duplicated.")

    @property
    def row_order_hash(self) -> str:
        return evaluation_order_hash(self.rows)

    @property
    def centers(self) -> np.ndarray:
        return np.asarray([row.center for row in self.rows], dtype=str)


@dataclass(frozen=True)
class ScoringLabels:
    """Labels opened only after the complete prediction pass is materialized."""

    labels: np.ndarray
    evaluation_order_hash: str
    manifest_sha256: str
    consumed_split: str = "val"

    def __post_init__(self) -> None:
        values = np.asarray(self.labels)
        if (
            values.ndim != 1
            or not np.issubdtype(values.dtype, np.integer)
            or not np.isin(values, [0, 1]).all()
            or sorted(set(int(value) for value in values.tolist())) != [0, 1]
        ):
            raise ProtocolError("Scoring labels must be a nonempty binary vector.")
        if self.consumed_split != "val":
            raise ProtocolError("Only validation labels may be consumed for source-inner scoring.")


def read_manifest_evaluation_index(
    path: str | Path,
    *,
    expected_sha256: str = EXPECTED_MANIFEST_SHA256,
    expected_rows: int | None = EXPECTED_EVAL_ROWS,
    expected_cases: int | None = EXPECTED_EVAL_CASES,
) -> tuple[ManifestEvaluationRow, ...]:
    """Read eligible validation identities without ever accessing ``label``."""

    manifest_path = Path(path)
    _assert_sha256(manifest_path, expected_sha256, "annotation manifest")
    rows: list[ManifestEvaluationRow] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "case_id", "center", "split", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ProtocolError("Annotation manifest lacks required source-inner fields.")
        for manifest_row_index, raw in enumerate(reader):
            split = str(raw.get("split", ""))
            center = str(raw.get("center", ""))
            if split != "val" or center not in CENTERS:
                # Deliberately skip before touching the label field.  Train, test,
                # and excluded-center labels are outside this consumption event.
                continue
            rows.append(
                ManifestEvaluationRow(
                    manifest_row_index=manifest_row_index,
                    sample_id=str(raw.get("sample_id", "")),
                    case_id=str(raw.get("case_id", "")),
                    center=center,
                )
            )
    if any(not row.sample_id or not row.case_id for row in rows):
        raise ProtocolError("Annotation manifest contains empty validation identities.")
    sample_ids = [row.sample_id for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ProtocolError("Annotation manifest duplicates an eligible validation sample.")
    if expected_rows is not None and len(rows) != int(expected_rows):
        raise ProtocolError(
            "Annotation manifest validation-row count drifted: "
            f"observed={len(rows)}, expected={expected_rows}."
        )
    n_cases = len({(row.center, row.case_id) for row in rows})
    if expected_cases is not None and n_cases != int(expected_cases):
        raise ProtocolError(
            "Annotation manifest validation-case count drifted: "
            f"observed={n_cases}, expected={expected_cases}."
        )
    return tuple(rows)


def load_unlabeled_validation_frame(
    cache_root: str | Path,
    manifest_rows: Sequence[ManifestEvaluationRow],
    *,
    expected_dim: int = FEATURE_DIM,
    expected_rows: int | None = EXPECTED_EVAL_ROWS,
) -> UnlabeledValidationFrame:
    """Validate and concatenate the nine label-free center shards."""

    root = Path(cache_root)
    cache_binding = _validate_cache_bundle(root)
    manifest_by_center = {
        center: tuple(row for row in manifest_rows if row.center == center)
        for center in CENTERS
    }
    arrays: list[np.ndarray] = []
    evaluation_rows: list[EvaluationRow] = []
    shard_hashes: dict[str, str] = {}
    row_ordinal = 0
    for center in CENTERS:
        relative = f"embeddings/by_center/center_{center}.pt"
        path = root / relative
        shard = _load_cache_shard(path, center=center, expected_dim=expected_dim)
        metadata = tuple(dict(row) for row in shard.metadata)
        expected_center_rows = manifest_by_center[center]
        if len(metadata) != len(expected_center_rows):
            raise ProtocolError(f"Validation cache row count drifted for center {center}.")
        for cache_row_index, (observed, expected) in enumerate(
            zip(metadata, expected_center_rows, strict=True)
        ):
            keys = {str(key) for key in observed}
            if keys != _CACHE_METADATA_KEYS:
                forbidden = sorted(keys.intersection(_FORBIDDEN_CACHE_METADATA_KEYS))
                detail = f"; forbidden={forbidden!r}" if forbidden else ""
                raise ProtocolError(
                    "Validation cache metadata keys drifted: "
                    f"observed={sorted(keys)!r}{detail}."
                )
            if any(key in observed for key in _FORBIDDEN_CACHE_METADATA_KEYS):
                raise ProtocolError("Validation cache illegally contains labels or classes.")
            identity = (
                str(observed.get("sample_id", "")),
                str(observed.get("case_id", "")),
                str(observed.get("center", "")),
                str(observed.get("split", "")),
                int(observed.get("manifest_row_index", -1)),
            )
            expected_identity = (
                expected.sample_id,
                expected.case_id,
                expected.center,
                expected.split,
                expected.manifest_row_index,
            )
            if identity != expected_identity:
                raise ProtocolError(
                    f"Validation cache/manifest alignment drifted for center {center}."
                )
            evaluation_rows.append(
                EvaluationRow(
                    row_ordinal=row_ordinal,
                    manifest_row_index=expected.manifest_row_index,
                    sample_id=expected.sample_id,
                    case_id=expected.case_id,
                    center=center,
                    split="val",
                    cache_shard_path=relative,
                    cache_row_index=cache_row_index,
                )
            )
            row_ordinal += 1
        matrix = np.asarray(shard.embeddings, dtype=np.float32)
        if matrix.shape != (len(metadata), int(expected_dim)) or not np.isfinite(matrix).all():
            raise ProtocolError(f"Validation cache embedding shard drifted for center {center}.")
        arrays.append(matrix)
        digest = str(getattr(shard, "cache_sha256", "")) or _sha256_file(path)
        if digest != _sha256_file(path):
            raise ProtocolError(f"Validation cache shard hash drifted for center {center}.")
        shard_hashes[center] = digest
    matrix = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    if expected_rows is not None and matrix.shape[0] != int(expected_rows):
        raise ProtocolError("Validation cache total row coverage drifted.")
    rows_tuple = tuple(evaluation_rows)
    binding = {
        **dict(cache_binding),
        "shard_sha256_by_center": shard_hashes,
        "row_count": int(matrix.shape[0]),
        "feature_dim": int(matrix.shape[1]),
        "evaluation_order_hash": evaluation_order_hash(rows_tuple),
        "labels_present": False,
    }
    return UnlabeledValidationFrame(
        embeddings=matrix,
        rows=rows_tuple,
        cache_binding=binding,
    )


def open_scoring_labels(
    manifest_path: str | Path,
    evaluation_rows: Sequence[EvaluationRow],
    *,
    expected_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> ScoringLabels:
    """Open only eligible val labels, after predictions have been persisted."""

    path = Path(manifest_path)
    _assert_sha256(path, expected_sha256, "annotation manifest")
    expected_by_index = {row.manifest_row_index: row for row in evaluation_rows}
    labels_by_index: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "label" not in reader.fieldnames:
            raise ProtocolError("Annotation manifest lacks the scoring label field.")
        for manifest_row_index, raw in enumerate(reader):
            expected = expected_by_index.get(manifest_row_index)
            if expected is None:
                # Train, test, center-4, and any other rows remain unconsumed.
                continue
            if (
                str(raw.get("split", "")) != "val"
                or str(raw.get("center", "")) != expected.center
                or str(raw.get("sample_id", "")) != expected.sample_id
                or str(raw.get("case_id", "")) != expected.case_id
            ):
                raise ProtocolError("Scoring-label manifest identity drifted.")
            try:
                label = int(float(str(raw.get("label", ""))))
            except ValueError as exc:
                raise ProtocolError("Scoring label is not binary.") from exc
            if label not in (0, 1):
                raise ProtocolError("Scoring label is outside {0,1}.")
            labels_by_index[manifest_row_index] = label
    if set(labels_by_index) != set(expected_by_index):
        raise ProtocolError("Scoring-label coverage differs from the prediction row index.")
    labels = np.asarray(
        [labels_by_index[row.manifest_row_index] for row in evaluation_rows],
        dtype=np.uint8,
    )
    centers = np.asarray([row.center for row in evaluation_rows], dtype=str)
    for center in CENTERS:
        if sorted(set(int(value) for value in labels[centers == center].tolist())) != [0, 1]:
            raise ProtocolError(f"Pseudo-target center {center} lacks binary label support.")
    return ScoringLabels(
        labels=labels,
        evaluation_order_hash=evaluation_order_hash(tuple(evaluation_rows)),
        manifest_sha256=expected_sha256,
    )


def _load_cache_shard(path: Path, *, center: str, expected_dim: int) -> object:
    try:
        from ....data.features.uniform_b_routing_validation import (
            load_unlabeled_validation_shard,
        )
    except ImportError as exc:  # pragma: no cover - coordinated package dependency
        raise RuntimeError(
            "Uniform-B routing validation-cache support is unavailable."
        ) from exc
    del expected_dim  # The cache loader enforces the frozen 3840-D contract.
    return load_unlabeled_validation_shard(path, expected_center=center)


def _validate_cache_bundle(root: Path) -> dict[str, object]:
    try:
        from ....data.features.uniform_b_routing_validation import (
            validate_uniform_b_routing_validation_cache,
        )
    except ImportError as exc:  # pragma: no cover - coordinated package dependency
        raise RuntimeError(
            "Uniform-B routing validation-cache support is unavailable."
        ) from exc
    checks = validate_uniform_b_routing_validation_cache(root)
    if not isinstance(checks, Mapping) or checks.get("status") != "PASS":
        raise ProtocolError("Routing validation cache has not passed validation.")
    protocol_path = root / "manifests/frozen_build_protocol.json"
    alignment_path = root / "manifests/row_alignment.json"
    content_path = root / "manifests/content_index.json"
    report_path = root / "reports/validation_report.json"
    for member in (protocol_path, alignment_path, content_path, report_path):
        if not member.is_file():
            raise ProtocolError(f"Routing validation cache member is missing: {member.name}.")
    protocol = _json(protocol_path)
    content = _json(content_path)
    validation = _json(report_path)
    if validation.get("status") != "PASS":
        raise ProtocolError("Routing validation cache validation report is not PASS.")
    return {
        "schema_version": "midogpp_source_inner_utility_cache_binding_v1",
        "frozen_build_protocol_sha256": _sha256_file(protocol_path),
        "row_alignment_sha256": _sha256_file(alignment_path),
        "content_index_sha256": _sha256_file(content_path),
        "validation_report_sha256": _sha256_file(report_path),
        "cache_name": protocol.get("cache_name"),
        "representation_id": protocol.get("representation_id"),
        "validation_split": protocol.get("validation_split"),
        "cache_protocol_hash": protocol.get("frozen_build_protocol_hash"),
        "cache_content_hash": content.get("content_hash"),
    }


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read source-inner cache JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Source-inner cache JSON must be an object: {path}.")
    return payload


def _assert_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _sha256_file(path) != expected:
        raise ProtocolError(f"Source-inner utility {label} SHA-256 drifted.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "ManifestEvaluationRow",
    "ScoringLabels",
    "UnlabeledValidationFrame",
    "load_unlabeled_validation_frame",
    "open_scoring_labels",
    "read_manifest_evaluation_index",
)
