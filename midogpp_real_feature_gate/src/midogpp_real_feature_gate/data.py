"""Manifest/cache loading adapter boundary.

Implementation must reuse or explicitly compare against the current SAIL
MIDOG++ manifest/cache alignment semantics before artifacts become thesis-facing.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import POSITIVE_LABEL
from .validation import ValidationError


@dataclass(frozen=True)
class ManifestRow:
    row_index: int
    sample_id: str
    case_id: str
    label: int
    split: str
    center: str
    tumor_domain: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class FeatureCache:
    embeddings: Any
    metadata: tuple[Mapping[str, object], ...]
    feature_extractor: Mapping[str, object]


def load_manifest(path: Path, *, positive_label: int = POSITIVE_LABEL) -> tuple[ManifestRow, ...]:
    """Load the MIDOG++ manifest.

    The canonical center field is resolved from `center`, then `scanner_model`,
    then `lab_or_origin`. The canonical tumor domain is resolved from
    `tumor_type`, then `tumor_domain`.
    """
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"empty MIDOG++ manifest: {path}")
        required = {"sample_id", "case_id", "label", "split"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValidationError(f"MIDOG++ manifest missing required columns: {missing}")
        rows: list[ManifestRow] = []
        for idx, row in enumerate(reader):
            raw_label = _clean_required(row.get("label"), "label", idx)
            try:
                label_value = int(float(raw_label))
            except ValueError as exc:
                raise ValidationError(f"invalid label in manifest row {idx}: {raw_label!r}") from exc
            rows.append(
                ManifestRow(
                    row_index=idx,
                    sample_id=_clean_required(row.get("sample_id"), "sample_id", idx),
                    case_id=_clean_required(row.get("case_id"), "case_id", idx),
                    label=1 if label_value == int(positive_label) else 0,
                    split=str(row.get("split", "")).strip().lower(),
                    center=_first_present(row, ("center", "scanner_model", "lab_or_origin"), idx),
                    tumor_domain=_first_present(row, ("tumor_type", "tumor_domain"), idx),
                    metadata={str(key): str(value).strip() for key, value in row.items()},
                )
            )
    return tuple(rows)


def load_feature_cache(path: Path) -> FeatureCache:
    """Load the real-feature cache.

    Supports the production torch mapping shape and the lightweight npz shape
    used by tests. This package does not import SAIL at runtime.
    """
    path = Path(path)
    if path.suffix == ".npz":
        return _load_npz_cache(path)
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise ValidationError(f"loading torch feature caches requires torch: {path}") from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValidationError(f"feature cache is not a mapping: {path}")
    return _cache_from_payload(payload, path)


def assert_cache_alignment(rows: Sequence[ManifestRow], cache: FeatureCache) -> None:
    metadata = tuple(cache.metadata)
    if len(rows) != len(metadata):
        raise ValidationError(f"cache_alignment_failed: manifest rows={len(rows)} cache rows={len(metadata)}")
    for idx, (row, meta) in enumerate(zip(rows, metadata)):
        cache_sample = str(meta.get("sample_id", "")).strip()
        if row.sample_id != cache_sample:
            raise ValidationError(
                f"cache_alignment_failed: row {idx} sample_id manifest={row.sample_id!r} cache={cache_sample!r}"
            )
        cache_label = meta.get("label")
        if cache_label is not None and int(float(str(cache_label))) != int(row.label):
            raise ValidationError(
                f"cache_alignment_failed: row {idx} label manifest={row.label!r} cache={cache_label!r}"
            )


def _load_npz_cache(path: Path) -> FeatureCache:
    import numpy as np  # type: ignore

    payload = np.load(path, allow_pickle=False)
    metadata = json.loads(str(payload["metadata_json"].item()))
    return FeatureCache(
        embeddings=payload["embeddings"],
        metadata=tuple(dict(row) for row in metadata),
        feature_extractor={"loader": "npz_test_or_lightweight_cache"},
    )


def _cache_from_payload(payload: Mapping[str, Any], path: Path) -> FeatureCache:
    if "embeddings" not in payload or "metadata" not in payload:
        raise ValidationError(f"feature cache must contain embeddings and metadata: {path}")
    metadata = payload["metadata"]
    if not isinstance(metadata, Sequence):
        raise ValidationError(f"feature cache metadata must be a sequence: {path}")
    extractor = payload.get("feature_extractor", {})
    return FeatureCache(
        embeddings=payload["embeddings"],
        metadata=tuple(row if isinstance(row, Mapping) else {} for row in metadata),
        feature_extractor=extractor if isinstance(extractor, Mapping) else {},
    )


def _clean_required(value: object, field: str, row_index: int) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValidationError(f"manifest row {row_index} missing required field {field}")
    return text


def _first_present(row: Mapping[str, object], fields: Sequence[str], row_index: int) -> str:
    for field in fields:
        value = str(row.get(field, "")).strip()
        if value:
            return value
    raise ValidationError(f"manifest row {row_index} missing one of {tuple(fields)}")
