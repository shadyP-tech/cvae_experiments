"""Protocol-keyed reusable PCA frames for long prior-recovery runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.protocol import ProtocolError
from ..feature_frame import ExpertFeatureFrame, fit_expert_frame


FRAME_CACHE_SCHEMA = "midogpp_prior_recovery_feature_frame_cache_v1"
FRAME_INDEX_SCHEMA = "midogpp_prior_recovery_feature_frame_index_v1"


@dataclass
class FeatureFrameCache:
    root: Path
    records: dict[str, dict[str, object]] = field(default_factory=dict)

    def fit_or_load(
        self,
        *,
        expert_id: str,
        source_train_embeddings: object,
        fit_centers: Sequence[str],
        fit_row_hash: str,
        requested_dim: int,
        manifest_hash: str,
        feature_cache_hash: str,
        protocol_hash: str,
        code_version: str,
    ) -> tuple[ExpertFeatureFrame, bool]:
        import numpy as np
        import sklearn

        x = np.asarray(source_train_embeddings, dtype=float)
        if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] <= 0:
            raise ProtocolError("Feature-frame cache received malformed source embeddings.")
        effective_dim = min(int(requested_dim), int(x.shape[0]), int(x.shape[1]))
        key_payload = {
            "schema_version": FRAME_CACHE_SCHEMA,
            "expert_id": str(expert_id),
            "fit_centers": [str(value) for value in fit_centers],
            "fit_row_hash": str(fit_row_hash),
            "manifest_hash": str(manifest_hash),
            "feature_cache_hash": str(feature_cache_hash),
            "protocol_hash": str(protocol_hash),
            "code_version": str(code_version),
            "requested_dim": int(requested_dim),
            "effective_dim": int(effective_dim),
            "input_shape": [int(x.shape[0]), int(x.shape[1])],
            "backbone_output_frame_id": "virchow2:full_to_pca",
            "scaler": "sklearn_standard_scaler_v1",
            "pca": {"svd_solver": "auto", "random_state": 0},
            "numpy_version": str(np.__version__),
            "sklearn_version": str(sklearn.__version__),
        }
        key_hash = stable_hash(key_payload)
        sidecar_path = self.root / f"runtime_cache/feature_frames/by_key/{key_hash}.json"
        if sidecar_path.is_file():
            record = _read_json(sidecar_path)
            frame = _load_frame_record(self.root, record, expected_key=key_payload)
            self._record(record)
            return frame, True

        frame = fit_expert_frame(
            expert_id=str(expert_id),
            source_train_embeddings=x,
            requested_dim=int(requested_dim),
        )
        data_path = self.root / f"runtime_cache/feature_frames/frames/{frame.state_hash}.npz"
        if not data_path.is_file():
            _atomic_npz(
                data_path,
                scaler_mean=frame.scaler_mean,
                scaler_scale=frame.scaler_scale,
                pca_components=frame.pca_components,
                pca_mean=frame.pca_mean,
                pca_explained_variance=frame.pca_explained_variance,
            )
        record = {
            "frame_cache_key_hash": key_hash,
            "frame_cache_key": key_payload,
            "frame_state_hash": frame.state_hash,
            "relative_path": data_path.relative_to(self.root).as_posix(),
            "file_sha256": _file_sha256(data_path),
            "expert_id": frame.expert_id,
            "requested_dim": int(frame.requested_dim),
            "effective_dim": int(frame.effective_dim),
            "explained_variance_ratio_sum": float(frame.explained_variance_ratio_sum),
            "fit_scope": frame.fit_scope,
        }
        _atomic_json(sidecar_path, record)
        self._record(record)
        return frame, False

    def write_index(self) -> Path:
        path = self.root / "manifests/feature_frame_index.json"
        _atomic_json(
            path,
            {
                "schema_version": FRAME_INDEX_SCHEMA,
                "n_unique_frames": len(self.records),
                "records": [self.records[key] for key in sorted(self.records)],
            },
        )
        return path

    def _record(self, record: Mapping[str, object]) -> None:
        key_hash = str(record.get("frame_cache_key_hash", ""))
        normalized = dict(record)
        existing = self.records.get(key_hash)
        if not key_hash or (existing is not None and existing != normalized):
            raise ProtocolError("Feature-frame cache key collision or missing identity.")
        self.records[key_hash] = normalized


def validate_feature_frame_index(
    root: Path,
    *,
    expected_frame_hashes: set[str] | None = None,
) -> Mapping[str, object]:
    root = Path(root)
    payload = _read_json(root / "manifests/feature_frame_index.json")
    records = payload.get("records")
    if payload.get("schema_version") != FRAME_INDEX_SCHEMA or not isinstance(records, list):
        raise ProtocolError("Malformed feature-frame index.")
    if int(payload.get("n_unique_frames", -1)) != len(records):
        raise ProtocolError("Feature-frame index count mismatch.")
    observed_keys: set[str] = set()
    observed_frames: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Malformed feature-frame index record.")
        record = dict(raw)
        key_payload = record.get("frame_cache_key")
        if not isinstance(key_payload, Mapping):
            raise ProtocolError("Feature-frame index lacks its cache key payload.")
        key_hash = str(record.get("frame_cache_key_hash", ""))
        frame_hash = str(record.get("frame_state_hash", ""))
        if (
            stable_hash(key_payload) != key_hash
            or key_hash in observed_keys
            or not frame_hash
        ):
            raise ProtocolError("Feature-frame index contains duplicate or invalid identities.")
        sidecar = root / f"runtime_cache/feature_frames/by_key/{key_hash}.json"
        if _read_json(sidecar) != record:
            raise ProtocolError("Feature-frame index differs from its durable cache sidecar.")
        frame = _load_frame_record(root, record, expected_key=key_payload)
        if frame.state_hash != frame_hash:
            raise ProtocolError("Feature-frame state hash failed validation.")
        observed_keys.add(key_hash)
        observed_frames.add(frame_hash)
    if expected_frame_hashes is not None and observed_frames != set(expected_frame_hashes):
        raise ProtocolError("Feature-frame index coverage differs from metric frame identities.")
    return payload


def _load_frame_record(
    root: Path,
    record: Mapping[str, object],
    *,
    expected_key: Mapping[str, object],
) -> ExpertFeatureFrame:
    import numpy as np

    key_payload = record.get("frame_cache_key")
    key_hash = str(record.get("frame_cache_key_hash", ""))
    if key_payload != dict(expected_key) or stable_hash(expected_key) != key_hash:
        raise ProtocolError("Feature-frame cache key does not match the requested protocol identity.")
    frame_hash = str(record.get("frame_state_hash", ""))
    expected_relative_path = f"runtime_cache/feature_frames/frames/{frame_hash}.npz"
    if (
        len(frame_hash) != 16
        or any(character not in "0123456789abcdef" for character in frame_hash)
        or record.get("relative_path") != expected_relative_path
    ):
        raise ProtocolError("Feature-frame cache record has a noncanonical frame path.")
    if (
        record.get("expert_id") != expected_key.get("expert_id")
        or int(record.get("requested_dim", -1)) != int(expected_key.get("requested_dim", -2))
        or int(record.get("effective_dim", -1)) != int(expected_key.get("effective_dim", -2))
        or record.get("fit_scope") != "per_expert_source_train"
    ):
        raise ProtocolError("Feature-frame cache metadata differs from its exact cache key.")
    path = Path(root) / expected_relative_path
    if not path.is_file() or _file_sha256(path) != str(record.get("file_sha256", "")):
        raise ProtocolError("Feature-frame cache file is missing or corrupt.")
    try:
        with np.load(path, allow_pickle=False) as state:
            arrays = {name: np.asarray(state[name]) for name in (
                "scaler_mean",
                "scaler_scale",
                "pca_components",
                "pca_mean",
                "pca_explained_variance",
            )}
    except Exception as exc:
        raise ProtocolError("Malformed feature-frame cache payload.") from exc
    input_shape = expected_key.get("input_shape")
    if not isinstance(input_shape, list) or len(input_shape) != 2:
        raise ProtocolError("Feature-frame cache key lacks its input shape.")
    input_dim = int(input_shape[1])
    effective_dim = int(record["effective_dim"])
    expected_shapes = {
        "scaler_mean": (input_dim,),
        "scaler_scale": (input_dim,),
        "pca_components": (effective_dim, input_dim),
        "pca_mean": (input_dim,),
        "pca_explained_variance": (effective_dim,),
    }
    if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
        raise ProtocolError("Feature-frame cache arrays have incompatible dimensions.")
    if any(not np.isfinite(value).all() for value in arrays.values()) or not np.all(
        arrays["scaler_scale"] > 0.0
    ):
        raise ProtocolError("Feature-frame cache arrays contain invalid numerical state.")
    explained_ratio = float(record.get("explained_variance_ratio_sum", math.nan))
    if not math.isfinite(explained_ratio) or not 0.0 <= explained_ratio <= 1.0 + 1e-12:
        raise ProtocolError("Feature-frame cache explained-variance metadata is invalid.")
    frame = ExpertFeatureFrame(
        expert_id=str(record.get("expert_id", "")),
        requested_dim=int(record.get("requested_dim", -1)),
        effective_dim=int(record.get("effective_dim", -1)),
        explained_variance_ratio_sum=explained_ratio,
        fit_scope=str(record.get("fit_scope", "")),
        **arrays,
    )
    if frame.state_hash != frame_hash:
        raise ProtocolError("Feature-frame cache state hash mismatch.")
    return frame


def _atomic_npz(path: Path, **arrays: object) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Malformed feature-frame cache JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected feature-frame cache JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
