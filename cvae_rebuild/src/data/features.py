from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class FeatureCache:
    embeddings: object
    metadata: tuple[Mapping[str, object], ...]
    path: Path


def default_cache_path(root: str | Path, *, seed: int, split: str) -> Path:
    return Path(root) / f"seed{int(seed)}" / "embeddings" / f"{split}.pt"


def load_feature_cache(path: str | Path) -> FeatureCache:
    cache_path = Path(path)
    if cache_path.suffix == ".npz":
        return _load_npz(cache_path)
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Loading .pt feature caches requires torch.") from exc
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Feature cache payload must be a mapping.")
    metadata = tuple(_normalize_meta(row) for row in payload.get("metadata", ()))
    return FeatureCache(embeddings=payload["embeddings"], metadata=metadata, path=cache_path)


def select_rows(
    embeddings: object,
    metadata: Sequence[Mapping[str, object]],
    indices: Sequence[int],
) -> tuple[object, tuple[Mapping[str, object], ...]]:
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
    except ModuleNotFoundError:
        np = None  # type: ignore
        torch = None  # type: ignore
    idx = [int(i) for i in indices]
    selected_meta = tuple(metadata[i] for i in idx)
    if torch is not None and hasattr(embeddings, "__class__") and embeddings.__class__.__module__.startswith("torch"):
        return embeddings[idx], selected_meta
    if np is not None:
        return np.asarray(embeddings)[idx], selected_meta
    return [embeddings[i] for i in idx], selected_meta


def _load_npz(path: Path) -> FeatureCache:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Loading .npz feature caches requires numpy.") from exc
    payload = np.load(path, allow_pickle=True)
    if "metadata_json" in payload:
        metadata_payload = json.loads(str(payload["metadata_json"].item()))
    else:
        metadata_payload = payload["metadata"].tolist()
    metadata = tuple(_normalize_meta(dict(row)) for row in metadata_payload)
    return FeatureCache(embeddings=payload["embeddings"], metadata=metadata, path=path)


def _normalize_meta(row: object) -> Mapping[str, object]:
    if not isinstance(row, Mapping):
        raise ValueError("Feature cache metadata rows must be mappings.")
    out = dict(row)
    if "center" not in out and "magnification" in out:
        out["center"] = out["magnification"]
    if "sample_id" not in out:
        out["sample_id"] = out.get("path", "")
    return out
