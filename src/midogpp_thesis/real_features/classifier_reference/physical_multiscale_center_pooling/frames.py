"""Center-sharded A/B/C frame loading with role-scoped held-out exclusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..protocol import ProtocolError
from .profiles import CENTER_POOLING_PROFILE_V1, PhysicalMultiscaleProfile


@dataclass(frozen=True)
class MultiRepresentationFrame:
    sample_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    labels: Any
    centers: tuple[str, ...]
    embeddings: Mapping[str, Any]

    def indices_for(self, centers: Sequence[str]) -> Any:
        import numpy as np  # type: ignore

        allowed = {str(center) for center in centers}
        return np.asarray(
            [index for index, center in enumerate(self.centers) if center in allowed],
            dtype=int,
        )


class CenterShardedRepresentationStore:
    """Load only explicit center shards; no global all-center cache is retained."""

    def __init__(
        self,
        *,
        b_cache_root: Path,
        c_cache_root: Path,
        profile: PhysicalMultiscaleProfile = CENTER_POOLING_PROFILE_V1,
    ) -> None:
        self.b_cache_root = Path(b_cache_root)
        self.c_cache_root = Path(c_cache_root)
        self.profile = profile
        self.representation_order = profile.representation_order
        self.representation_dims = profile.representation_dims
        self.b_representation_id = self.representation_order[1]
        self.c_representation_id = self.representation_order[2]
        self.access_log: list[tuple[str, str]] = []

    def selector_frame(
        self,
        *,
        outer_target_center: str,
        eligible_centers: Sequence[str],
    ) -> MultiRepresentationFrame:
        allowed = tuple(
            center for center in eligible_centers if str(center) != str(outer_target_center)
        )
        if str(outer_target_center) in allowed:
            raise ProtocolError("Selector frame contains its outer target center.")
        return self._load(allowed, role=f"selector_outer_{outer_target_center}")

    def outer_frame(self, center: str) -> MultiRepresentationFrame:
        return self._load((str(center),), role=f"outer_eval_{center}")

    def _load(self, centers: Sequence[str], *, role: str) -> MultiRepresentationFrame:
        try:
            import numpy as np  # type: ignore
            import torch  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
            raise RuntimeError("Representation frames require numpy and torch.") from exc

        samples: list[str] = []
        cases: list[str] = []
        labels: list[int] = []
        observed_centers: list[str] = []
        contract_row_indices: list[int] = []
        by_rep: dict[str, list[Any]] = {
            rep: [] for rep in self.representation_order
        }
        for center in centers:
            b_path = self.b_cache_root / "embeddings" / "by_center" / f"center_{center}.pt"
            c_path = self.c_cache_root / "embeddings" / "by_center" / f"center_{center}.pt"
            b_payload = _torch_payload(torch, b_path)
            c_payload = _torch_payload(torch, c_path)
            b_meta = tuple(dict(row) for row in b_payload["metadata"])
            c_meta = tuple(dict(row) for row in c_payload["metadata"])
            if b_meta != c_meta:
                raise ProtocolError(f"B/C metadata mismatch for center {center}.")
            b = np.asarray(b_payload["embeddings"], dtype=float)
            c = np.asarray(c_payload["embeddings"], dtype=float)
            a = np.asarray(b_payload.get("canonical_a_embeddings"), dtype=float)
            if (
                a.ndim != 2
                or len(b_meta) != a.shape[0]
                or b.shape
                != (a.shape[0], self.representation_dims[self.b_representation_id])
                or c.shape
                != (a.shape[0], self.representation_dims[self.c_representation_id])
                or a.shape[1] != self.representation_dims["canonical_a"]
            ):
                raise ProtocolError(f"A/B/C shard dimension drift for center {center}.")
            if (
                not np.all(np.isfinite(a))
                or not np.all(np.isfinite(b))
                or not np.all(np.isfinite(c))
            ):
                raise ProtocolError(f"A/B/C shard contains non-finite values: {center}.")
            if not all(str(row.get("center")) == str(center) for row in b_meta):
                raise ProtocolError(f"Shard metadata center drift: {center}.")
            by_rep["canonical_a"].append(a)
            by_rep[self.b_representation_id].append(b)
            by_rep[self.c_representation_id].append(c)
            samples.extend(str(row["sample_id"]) for row in b_meta)
            cases.extend(str(row["case_id"]) for row in b_meta)
            labels.extend(int(row["label"]) for row in b_meta)
            observed_centers.extend(str(row["center"]) for row in b_meta)
            contract_row_indices.extend(
                _contract_row_index(row, center=center) for row in b_meta
            )
            self.access_log.append((role, str(center)))
        if not samples or len(samples) != len(set(samples)):
            raise ProtocolError("Role-scoped representation frame is empty or duplicated.")
        if len(contract_row_indices) != len(set(contract_row_indices)):
            raise ProtocolError(
                "Role-scoped representation frame has duplicate contract_row_index values."
            )
        order = np.argsort(
            np.asarray(contract_row_indices, dtype=np.int64),
            kind="stable",
        )
        arrays = {
            rep: np.concatenate(chunks, axis=0)[order]
            for rep, chunks in by_rep.items()
        }
        if any(not np.all(np.isfinite(array)) for array in arrays.values()):
            raise ProtocolError("Representation frame contains non-finite values.")
        return MultiRepresentationFrame(
            sample_ids=tuple(samples[index] for index in order),
            case_ids=tuple(cases[index] for index in order),
            labels=np.asarray(labels, dtype=int)[order],
            centers=tuple(observed_centers[index] for index in order),
            embeddings=arrays,
        )


def _torch_payload(torch: Any, path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Center shard is not a mapping: {path}")
    required = {"embeddings", "metadata"}
    if not required.issubset(payload):
        raise ProtocolError(f"Center shard lacks {sorted(required.difference(payload))}: {path}")
    return payload


def _contract_row_index(row: Mapping[str, object], *, center: str) -> int:
    raw = row.get("contract_row_index")
    if isinstance(raw, bool):
        raise ProtocolError(
            f"Shard metadata lacks a valid contract_row_index for center {center}."
        )
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProtocolError(
            f"Shard metadata lacks a valid contract_row_index for center {center}."
        ) from exc
    if value < 0 or str(raw).strip() != str(value):
        raise ProtocolError(
            f"Shard metadata lacks a valid contract_row_index for center {center}."
        )
    return value
