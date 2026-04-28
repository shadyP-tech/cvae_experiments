from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.torch_utils import safe_torch_load


MODEL_STATE_DICT_KEY = "model_state_dict"
CHECKPOINT_METADATA_KEY = "checkpoint_metadata"
CHECKPOINT_SCHEMA_VERSION = "checkpoint_provenance_v1"


@dataclass(frozen=True)
class LoadedCheckpoint:
    model_state_dict: Dict[str, Any]
    checkpoint_metadata: Dict[str, Any]
    legacy_format: bool


def build_checkpoint_metadata_from_cache(
    payload: Dict[str, Any],
    *,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    extractor = dict(payload.get("feature_extractor", {}) or {})
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "feature_extractor": extractor,
        "backbone_type": extractor.get("backbone_type", ""),
        "feature_extractor_name": extractor.get("feature_extractor_name", ""),
        "feature_extractor_checkpoint": extractor.get("feature_extractor_checkpoint", ""),
        "feature_extractor_layer": extractor.get("feature_extractor_layer", ""),
        "embedding_pooling": extractor.get("embedding_pooling", ""),
        "embedding_dim": int(extractor.get("embedding_dim", payload.get("embeddings").shape[1] if "embeddings" in payload else 0)),
        "image_size": int(extractor.get("image_size", 0) or 0),
    }
    if extra:
        metadata.update(dict(extra))
    return metadata


def wrap_model_state_dict(
    state_dict: Dict[str, Any],
    checkpoint_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        MODEL_STATE_DICT_KEY: state_dict,
        CHECKPOINT_METADATA_KEY: dict(checkpoint_metadata or {}),
    }


def unwrap_model_checkpoint_payload(payload: Any) -> LoadedCheckpoint:
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary, got {type(payload)}")
    if MODEL_STATE_DICT_KEY in payload:
        state = payload.get(MODEL_STATE_DICT_KEY)
        if not isinstance(state, dict):
            raise ValueError("Wrapped checkpoint payload has non-dictionary model_state_dict")
        metadata = payload.get(CHECKPOINT_METADATA_KEY, {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("Wrapped checkpoint payload has non-dictionary checkpoint_metadata")
        return LoadedCheckpoint(
            model_state_dict=state,
            checkpoint_metadata=dict(metadata),
            legacy_format=False,
        )
    if CHECKPOINT_METADATA_KEY in payload:
        raise ValueError("Malformed wrapped checkpoint payload: missing model_state_dict")
    return LoadedCheckpoint(
        model_state_dict=payload,
        checkpoint_metadata={},
        legacy_format=True,
    )


def load_model_checkpoint(path: Path, map_location: Any = "cpu") -> LoadedCheckpoint:
    return unwrap_model_checkpoint_payload(safe_torch_load(path, map_location=map_location))


def unwrap_hybrid_checkpoint_payload(payload: Any) -> LoadedCheckpoint:
    if not isinstance(payload, dict):
        raise ValueError(f"Hybrid checkpoint payload must be a dictionary, got {type(payload)}")
    if MODEL_STATE_DICT_KEY in payload:
        loaded = unwrap_model_checkpoint_payload(payload)
        hybrid_payload = loaded.model_state_dict
    else:
        hybrid_payload = payload
        loaded = LoadedCheckpoint(
            model_state_dict=hybrid_payload,
            checkpoint_metadata=dict(payload.get(CHECKPOINT_METADATA_KEY, {}) or {}),
            legacy_format=CHECKPOINT_METADATA_KEY not in payload,
        )
    required = {"variant", "domains", "input_dim", "projection_dim", "latent_dim"}
    missing = sorted(required.difference(hybrid_payload.keys()))
    if missing:
        raise ValueError(f"Malformed hybrid checkpoint payload missing keys: {missing}")
    metadata = hybrid_payload.get(CHECKPOINT_METADATA_KEY, loaded.checkpoint_metadata)
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("Hybrid checkpoint payload has non-dictionary checkpoint_metadata")
    return LoadedCheckpoint(
        model_state_dict=hybrid_payload,
        checkpoint_metadata=dict(metadata),
        legacy_format=loaded.legacy_format,
    )


def load_hybrid_checkpoint(path: Path, map_location: Any = "cpu") -> LoadedCheckpoint:
    return unwrap_hybrid_checkpoint_payload(safe_torch_load(path, map_location=map_location))
