"""MIDOG++ real-feature frame loading and provenance validation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .protocol import ProtocolError
from .schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS, MIDOGPP_EXCLUDED_CENTERS


@dataclass(frozen=True)
class RealFeatureRow:
    row_index: int
    sample_id: str
    case_id: str
    center: str
    label: int
    split: str
    image_path: str = ""


@dataclass(frozen=True)
class RealFeatureFrame:
    embeddings: Any
    rows: tuple[RealFeatureRow, ...]
    feature_extractor: Mapping[str, object]
    feature_cache_path: Path
    feature_cache_hash: str
    manifest_path: Path
    manifest_hash: str
    expected_feature_dim: int

    @property
    def eligible_centers(self) -> tuple[str, ...]:
        observed = {row.center for row in self.rows}
        return tuple(center for center in MIDOGPP_ELIGIBLE_CENTERS if center in observed)


def load_midogpp_real_feature_frame(
    *,
    manifest_path: Path,
    feature_cache_path: Path,
    expected_feature_dim: int = 2560,
    allow_excluded_center_omission: bool = False,
) -> RealFeatureFrame:
    """Load and align a MIDOG++ real-feature cache to the train manifest rows."""

    manifest_rows = _read_manifest_train_rows(Path(manifest_path))
    embeddings, metadata, feature_extractor = _load_feature_cache_payload(Path(feature_cache_path))
    _assert_feature_array(embeddings, expected_feature_dim=expected_feature_dim)
    if int(getattr(embeddings, "shape", (0,))[0]) != len(metadata):
        raise ProtocolError("Feature cache embedding row count does not match metadata row count.")
    _assert_real_feature_provenance(
        feature_extractor=feature_extractor,
        feature_cache_path=Path(feature_cache_path),
        manifest_path=Path(manifest_path),
    )
    manifest_by_id = {str(row["sample_id"]): row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ProtocolError("MIDOG++ manifest has duplicate train sample_id values.")
    rows: list[RealFeatureRow] = []
    seen_cache_ids: set[str] = set()
    for idx, meta in enumerate(metadata):
        sample_id = str(meta.get("sample_id", "")).strip()
        if not sample_id:
            raise ProtocolError(f"Feature cache metadata row {idx} lacks sample_id.")
        if sample_id in seen_cache_ids:
            raise ProtocolError(f"Feature cache has duplicate sample_id={sample_id!r}.")
        seen_cache_ids.add(sample_id)
        manifest_row = manifest_by_id.get(sample_id)
        if manifest_row is None:
            raise ProtocolError(f"Feature cache sample_id not found in train manifest rows: {sample_id!r}")
        cache_split = str(meta.get("split", manifest_row.get("split", ""))).strip().lower()
        if cache_split and cache_split != "train":
            raise ProtocolError(f"Feature cache row {sample_id!r} is not split=train.")
        label = _label_value(manifest_row.get("label"))
        cache_label = _label_value(meta.get("label", label))
        if cache_label != label:
            raise ProtocolError(f"Feature cache label mismatch for sample_id={sample_id!r}.")
        center = _center_value(manifest_row)
        cache_center = _center_value(meta) if _has_center(meta) else center
        if cache_center != center:
            raise ProtocolError(f"Feature cache center mismatch for sample_id={sample_id!r}.")
        if center not in set(MIDOGPP_ELIGIBLE_CENTERS).union(MIDOGPP_EXCLUDED_CENTERS):
            raise ProtocolError(f"Unknown MIDOG++ center in train manifest/cache: {center!r}")
        rows.append(
            RealFeatureRow(
                row_index=idx,
                sample_id=sample_id,
                case_id=str(manifest_row.get("case_id", meta.get("case_id", sample_id))),
                center=center,
                label=label,
                split="train",
                image_path=str(
                    manifest_row.get("image_path", meta.get("image_path", ""))
                ),
            )
        )
    missing_cache = sorted(set(manifest_by_id).difference(seen_cache_ids))
    if missing_cache:
        missing_centers = {
            _center_value(manifest_by_id[sample_id])
            for sample_id in missing_cache
        }
        allowed_omission = (
            allow_excluded_center_omission
            and missing_centers.issubset(set(MIDOGPP_EXCLUDED_CENTERS))
        )
        if not allowed_omission:
            raise ProtocolError(
                "Feature cache is missing train manifest sample_ids: "
                f"{missing_cache[:5]}"
            )
    return RealFeatureFrame(
        embeddings=embeddings,
        rows=tuple(rows),
        feature_extractor=feature_extractor,
        feature_cache_path=Path(feature_cache_path),
        feature_cache_hash=_file_sha256(Path(feature_cache_path)),
        manifest_path=Path(manifest_path),
        manifest_hash=_file_sha256(Path(manifest_path)),
        expected_feature_dim=int(expected_feature_dim),
    )


def _read_manifest_train_rows(path: Path) -> tuple[dict[str, object], ...]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProtocolError(f"Empty MIDOG++ manifest: {path}")
        required = {"sample_id", "label", "split"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ProtocolError(f"MIDOG++ manifest missing required fields: {missing}")
        if "center" not in reader.fieldnames and "magnification" not in reader.fieldnames:
            raise ProtocolError("MIDOG++ manifest requires center or magnification column.")
        rows = [dict(row) for row in reader if str(row.get("split", "")).strip().lower() == "train"]
    if not rows:
        raise ProtocolError(f"MIDOG++ manifest has no split=train rows: {path}")
    return tuple(rows)


def _load_feature_cache_payload(path: Path) -> tuple[Any, tuple[Mapping[str, object], ...], Mapping[str, object]]:
    path = Path(path)
    if path.suffix == ".npz":
        return _load_npz_feature_cache(path)
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise ProtocolError(f"Loading torch feature caches requires torch: {path}") from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Feature cache is not a mapping: {path}")
    return _cache_from_payload(payload, path)


def _load_npz_feature_cache(path: Path) -> tuple[Any, tuple[Mapping[str, object], ...], Mapping[str, object]]:
    import numpy as np  # type: ignore

    payload = np.load(path, allow_pickle=False)
    if "embeddings" not in payload or "metadata_json" not in payload:
        raise ProtocolError(f"NPZ feature cache must contain embeddings and metadata_json: {path}")
    metadata = json.loads(str(payload["metadata_json"].item()))
    feature_extractor = (
        json.loads(str(payload["feature_extractor_json"].item()))
        if "feature_extractor_json" in payload
        else {"loader": "npz_test_or_lightweight_cache"}
    )
    return payload["embeddings"], tuple(dict(row) for row in metadata), feature_extractor


def _cache_from_payload(payload: Mapping[str, Any], path: Path) -> tuple[Any, tuple[Mapping[str, object], ...], Mapping[str, object]]:
    if "embeddings" not in payload or "metadata" not in payload:
        raise ProtocolError(f"Feature cache must contain embeddings and metadata: {path}")
    feature_extractor = payload.get("feature_extractor", {})
    if not isinstance(feature_extractor, Mapping):
        feature_extractor = {}
    return payload["embeddings"], tuple(dict(row) for row in payload["metadata"]), dict(feature_extractor)


def _assert_feature_array(embeddings: Any, *, expected_feature_dim: int) -> None:
    shape = getattr(embeddings, "shape", ())
    if len(shape) != 2:
        raise ProtocolError(f"Feature cache embeddings must be 2D, got shape={shape!r}")
    if int(shape[1]) != int(expected_feature_dim):
        raise ProtocolError(
            f"Feature cache dimension mismatch: expected {expected_feature_dim}, got {int(shape[1])}"
        )


def _assert_real_feature_provenance(
    *,
    feature_extractor: Mapping[str, object],
    feature_cache_path: Path,
    manifest_path: Path,
) -> None:
    text = " ".join(
        [
            json.dumps(dict(feature_extractor), sort_keys=True, default=str),
            str(feature_cache_path),
            str(manifest_path),
        ]
    ).lower()
    if "virchow2" not in text:
        raise ProtocolError("Feature cache does not declare Virchow2 provenance.")
    if "midogpp" not in text:
        raise ProtocolError("Feature cache/manifest path does not declare MIDOG++ provenance.")


def _label_value(value: object) -> int:
    try:
        label = int(float(str(value)))
    except ValueError as exc:
        raise ProtocolError(f"Invalid MIDOG++ label value: {value!r}") from exc
    if label not in {0, 1}:
        raise ProtocolError(f"MIDOG++ labels must be binary 0/1, got {value!r}")
    return label


def _center_value(row: Mapping[str, object]) -> str:
    center = str(row.get("center", "") or row.get("magnification", "")).strip()
    if not center:
        raise ProtocolError(f"MIDOG++ row lacks center/magnification: {row}")
    return center


def _has_center(row: Mapping[str, object]) -> bool:
    return bool(str(row.get("center", "") or row.get("magnification", "")).strip())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
