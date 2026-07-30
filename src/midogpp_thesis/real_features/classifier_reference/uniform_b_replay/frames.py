"""B-only center-sharded loading for the retrospective replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..protocol import ProtocolError
from .config import CANONICAL_A, REPRESENTATION_DIMS, UNIFORM_B


@dataclass(frozen=True)
class ABFrame:
    sample_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    labels: Any
    centers: tuple[str, ...]
    embeddings: Mapping[str, Any]


class UniformBShardedStore:
    """Load canonical A and B from B shards without resolving C."""

    def __init__(self, b_cache_root: Path) -> None:
        self.b_cache_root = Path(b_cache_root)
        self.access_log: list[tuple[str, str]] = []

    def source_frame(
        self, *, heldout: str, eligible_centers: Sequence[str]
    ) -> ABFrame:
        centers = tuple(str(center) for center in eligible_centers if str(center) != heldout)
        return self._load(centers, role=f"source_outer_{heldout}")

    def target_frame(self, center: str) -> ABFrame:
        return self._load((str(center),), role=f"target_outer_{center}")

    def _load(self, centers: Sequence[str], *, role: str) -> ABFrame:
        try:
            import numpy as np  # type: ignore
            import torch  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("Uniform-B frames require numpy and torch.") from exc
        samples: list[str] = []
        cases: list[str] = []
        labels: list[int] = []
        observed_centers: list[str] = []
        row_indices: list[int] = []
        a_chunks: list[Any] = []
        b_chunks: list[Any] = []
        for center in centers:
            path = self.b_cache_root / "embeddings" / "by_center" / f"center_{center}.pt"
            payload = _torch_payload(torch, path)
            metadata = tuple(dict(row) for row in payload["metadata"])
            a = np.asarray(payload.get("canonical_a_embeddings"), dtype=float)
            b = np.asarray(payload["embeddings"], dtype=float)
            if (
                a.ndim != 2
                or b.shape != (a.shape[0], REPRESENTATION_DIMS[UNIFORM_B])
                or a.shape[1] != REPRESENTATION_DIMS[CANONICAL_A]
                or len(metadata) != a.shape[0]
                or not np.all(np.isfinite(a))
                or not np.all(np.isfinite(b))
            ):
                raise ProtocolError(f"Uniform-B shard dimension/content drift: {center}.")
            if not all(str(row.get("center")) == center for row in metadata):
                raise ProtocolError(f"Uniform-B shard center drift: {center}.")
            a_chunks.append(a)
            b_chunks.append(b)
            samples.extend(str(row["sample_id"]) for row in metadata)
            cases.extend(str(row["case_id"]) for row in metadata)
            labels.extend(int(row["label"]) for row in metadata)
            observed_centers.extend(str(row["center"]) for row in metadata)
            row_indices.extend(_contract_index(row, center) for row in metadata)
            self.access_log.append((role, center))
        if not samples or len(samples) != len(set(samples)):
            raise ProtocolError("Uniform-B frame is empty or has duplicate sample IDs.")
        if len(row_indices) != len(set(row_indices)):
            raise ProtocolError("Uniform-B frame has duplicate contract row indices.")
        order = np.argsort(np.asarray(row_indices, dtype=np.int64), kind="stable")
        return ABFrame(
            sample_ids=tuple(samples[index] for index in order),
            case_ids=tuple(cases[index] for index in order),
            labels=np.asarray(labels, dtype=int)[order],
            centers=tuple(observed_centers[index] for index in order),
            embeddings={
                CANONICAL_A: np.concatenate(a_chunks, axis=0)[order],
                UNIFORM_B: np.concatenate(b_chunks, axis=0)[order],
            },
        )


def _torch_payload(torch: Any, path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or not {
        "embeddings",
        "canonical_a_embeddings",
        "metadata",
    }.issubset(payload):
        raise ProtocolError(f"Uniform-B shard is invalid: {path}")
    return payload


def _contract_index(row: Mapping[str, object], center: str) -> int:
    raw = row.get("contract_row_index")
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Uniform-B row index is invalid: center={center}") from exc
    if isinstance(raw, bool) or value < 0 or str(raw).strip() != str(value):
        raise ProtocolError(f"Uniform-B row index is invalid: center={center}")
    return value
