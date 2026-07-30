"""Validated I/O for immutable feature caches and center shards."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CacheRows:
    embeddings: Any
    metadata: tuple[Mapping[str, object], ...]
    cache_sha256: str

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(str(row.get("sample_id", "")) for row in self.metadata)


def load_cache_rows(
    path: str | Path,
    *,
    expected_dim: int | None = None,
) -> CacheRows:
    """Load one torch cache and reject shape, identity, or finiteness drift."""

    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("Feature-cache loading requires numpy and torch.") from exc

    cache_path = Path(path)
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Feature cache must contain a mapping: {cache_path}")
    embeddings = payload.get("embeddings")
    metadata = payload.get("metadata")
    if embeddings is None or not isinstance(metadata, Sequence):
        raise ValueError(f"Feature cache lacks embeddings/metadata: {cache_path}")
    array = np.asarray(embeddings)
    if array.ndim != 2 or array.shape[0] != len(metadata):
        raise ValueError(f"Feature cache shape/metadata mismatch: {cache_path}")
    if expected_dim is not None and int(array.shape[1]) != int(expected_dim):
        raise ValueError(
            f"Feature cache dimension mismatch: expected={expected_dim}, actual={array.shape[1]}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Feature cache contains non-finite values: {cache_path}")
    rows = tuple(dict(row) for row in metadata)
    sample_ids = tuple(str(row.get("sample_id", "")) for row in rows)
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Feature cache sample IDs are empty or duplicated: {cache_path}")
    return CacheRows(
        embeddings=embeddings,
        metadata=rows,
        cache_sha256=_sha256_file(cache_path),
    )


def write_center_shard(
    path: str | Path,
    *,
    embeddings: Any,
    metadata: Sequence[Mapping[str, object]],
    feature_extractor: Mapping[str, object],
    canonical_a_embeddings: Any | None = None,
) -> None:
    """Write one immutable center shard; differing existing bytes fail closed."""

    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("Feature-cache writing requires torch.") from exc

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable center shard: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "embeddings": embeddings,
        "metadata": [dict(row) for row in metadata],
        "feature_extractor": dict(feature_extractor),
    }
    if canonical_a_embeddings is not None:
        payload["canonical_a_embeddings"] = canonical_a_embeddings
    torch.save(payload, output)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
