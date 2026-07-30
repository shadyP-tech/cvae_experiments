"""Reusable, dataset-owned Virchow2 token extraction primitives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.data.features.virchow2_tokens import (
    VIRCHOW2_TOKEN_LAYOUT,
    Virchow2TokenLayout,
    assert_preprocessing_spatial_identity,
    central_virchow2_token_grid,
    describe_preprocessing_spatial_identity,
    normalized_coordinate_to_window_start,
    normalized_position_to_window_start,
    pool_virchow2_tokens_v2,
    validate_virchow2_token_layout,
    validate_window_starts,
)


CLS_DIM = 1280
PATCH_GRID_SIDE = 16
PATCH_TOKEN_COUNT = PATCH_GRID_SIDE * PATCH_GRID_SIDE
REGISTER_TOKEN_COUNT = 4
PATCH_TOKEN_START = 1 + REGISTER_TOKEN_COUNT
CENTER_SLICE = slice(6, 10)


def resolve_tensor(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        raise RuntimeError(f"Could not resolve tensor from output type {type(value)}")
    if getattr(value, "ndim", None) is not None:
        return value
    if isinstance(value, Mapping):
        for candidate in value.values():
            try:
                return resolve_tensor(candidate, depth=depth + 1)
            except RuntimeError:
                continue
    if hasattr(value, "to_tuple"):
        return resolve_tensor(value.to_tuple(), depth=depth + 1)
    if isinstance(value, (tuple, list)):
        for candidate in value:
            try:
                return resolve_tensor(candidate, depth=depth + 1)
            except RuntimeError:
                continue
    raise RuntimeError(f"Could not resolve tensor from output type {type(value)}")


def pool_virchow2_tokens(outputs: Any, *, include_center: bool) -> Any:
    """Pool CLS, all patch tokens, and optionally the frozen central 4x4 block."""

    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("Virchow2 token pooling requires torch.") from exc

    tokens = resolve_tensor(outputs)
    if getattr(tokens, "ndim", 0) != 3:
        raise RuntimeError(f"Virchow2 tokens must be rank 3, got {getattr(tokens, 'shape', None)}")
    expected = PATCH_TOKEN_START + PATCH_TOKEN_COUNT
    if int(tokens.shape[1]) != expected:
        raise RuntimeError(
            f"Virchow2 token count drift: expected={expected}, actual={int(tokens.shape[1])}"
        )
    patch = tokens[:, PATCH_TOKEN_START:]
    blocks = [tokens[:, 0], patch.mean(dim=1)]
    if include_center:
        grid = patch.reshape(
            int(tokens.shape[0]),
            PATCH_GRID_SIDE,
            PATCH_GRID_SIDE,
            int(tokens.shape[2]),
        )
        center = grid[:, CENTER_SLICE, CENTER_SLICE, :].reshape(
            int(tokens.shape[0]), -1, int(tokens.shape[2])
        )
        blocks.append(center.mean(dim=1))
    pooled = torch.cat(tuple(blocks), dim=-1)
    expected_dim = CLS_DIM * (3 if include_center else 2)
    if int(pooled.shape[1]) != expected_dim:
        raise RuntimeError(
            f"Virchow2 pooled dimension drift: expected={expected_dim}, actual={pooled.shape[1]}"
        )
    return pooled


class Virchow2TokenExtractor:
    """Pinned model/transform wrapper that exposes deterministic pooled blocks."""

    def __init__(
        self,
        *,
        model_ref: str,
        model_revision: str,
        device: str,
        expected_model_config_sha256: str,
        expected_checkpoint_file_sha256: str,
        expected_state_dict_sha256: str,
        expected_preprocessing_config_hash: str,
        hf_hub_cache_path: str | Path | None = None,
        hf_hub_local_files_only: bool = False,
    ) -> None:
        required = {
            "model config SHA256": expected_model_config_sha256,
            "checkpoint-file SHA256": expected_checkpoint_file_sha256,
            "state-dict SHA256": expected_state_dict_sha256,
            "preprocessing config hash": expected_preprocessing_config_hash,
        }
        unresolved = [
            label
            for label, value in required.items()
            if not value or str(value).startswith("TODO_")
        ]
        if unresolved:
            raise ValueError(
                f"Virchow2 identity must be resolved before cache building: {unresolved}"
            )
        loaded = _load_pinned_virchow2(
            model_ref=model_ref,
            model_revision=model_revision,
            device=device,
            hf_hub_cache_path=hf_hub_cache_path,
            hf_hub_local_files_only=hf_hub_local_files_only,
        )
        identity = loaded["identity"]
        assert isinstance(identity, Mapping)
        expected_actual = {
            "model_config_sha256": expected_model_config_sha256,
            "checkpoint_file_sha256": expected_checkpoint_file_sha256,
            "state_dict_sha256": expected_state_dict_sha256,
            "preprocessing_config_hash": expected_preprocessing_config_hash,
        }
        drift = {
            key: {"expected": expected, "actual": identity.get(key)}
            for key, expected in expected_actual.items()
            if identity.get(key) != expected
        }
        if drift:
            raise ValueError(
                f"Virchow2 pinned model identity mismatch: {drift}"
            )
        self.model = loaded["model"]
        self.transform = loaded["transform"]
        self.device = loaded["device"]
        self.identity = dict(identity)

    def extract_images(
        self,
        images: Sequence[Any],
        *,
        include_center: bool,
    ) -> Any:
        import torch  # type: ignore

        if not images:
            raise ValueError("Virchow2 extraction batch may not be empty.")
        with torch.no_grad():
            tensors = [self.transform(image.convert("RGB")) for image in images]
            outputs = self.model(torch.stack(tensors, dim=0).to(self.device))
            return pool_virchow2_tokens(
                outputs,
                include_center=include_center,
            ).detach().cpu().float()

    def extract_spatial_windows(
        self,
        images: Sequence[Any],
        *,
        window_starts: Sequence[Sequence[int]],
        layout: Virchow2TokenLayout = VIRCHOW2_TOKEN_LAYOUT,
    ) -> Any:
        """Extract strict-layout tokens and pool one spatial window per image."""

        import torch  # type: ignore

        if not images:
            raise ValueError("Virchow2 extraction batch may not be empty.")
        with torch.no_grad():
            tensors = [self.transform(image.convert("RGB")) for image in images]
            tokens = self.model.forward_features(
                torch.stack(tensors, dim=0).to(self.device)
            )
            validate_virchow2_token_layout(tokens, layout=layout)
            return pool_virchow2_tokens_v2(
                tokens,
                window_starts=window_starts,
                layout=layout,
            ).detach().cpu().float()

    def extract_images_v2(
        self,
        images: Sequence[Any],
        *,
        window_starts: Sequence[Sequence[int]],
        layout: Virchow2TokenLayout = VIRCHOW2_TOKEN_LAYOUT,
    ) -> Any:
        """Compatibility wrapper for the immutable v2 extraction surface."""

        return self.extract_spatial_windows(
            images,
            window_starts=window_starts,
            layout=layout,
        )

    def extract_central_token_grid(
        self,
        images: Sequence[Any],
        *,
        layout: Virchow2TokenLayout = VIRCHOW2_TOKEN_LAYOUT,
    ) -> Any:
        """Extract the unpooled ordered central ``4x4`` patch-token grid."""

        import torch  # type: ignore

        if not images:
            raise ValueError("Virchow2 extraction batch may not be empty.")
        with torch.inference_mode():
            tensors = [self.transform(image.convert("RGB")) for image in images]
            tokens = self.model.forward_features(
                torch.stack(tensors, dim=0).to(self.device)
            )
            return central_virchow2_token_grid(
                tokens,
                layout=layout,
            ).detach().cpu().float()


def checkpoint_sha256(model: Any) -> str:
    """Hash state tensors in stable key order without relying on cache filenames."""

    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def resolve_virchow2_identity(
    *,
    model_ref: str,
    model_revision: str,
    device: str,
) -> Mapping[str, object]:
    """Resolve a revision-pinned model and return its complete extraction identity."""

    loaded = _load_pinned_virchow2(
        model_ref=model_ref,
        model_revision=model_revision,
        device=device,
    )
    return dict(loaded["identity"])  # type: ignore[arg-type]


def _load_pinned_virchow2(
    *,
    model_ref: str,
    model_revision: str,
    device: str,
    hf_hub_cache_path: str | Path | None = None,
    hf_hub_local_files_only: bool = False,
) -> dict[str, object]:
    try:
        import timm  # type: ignore
        import torch  # type: ignore
        from huggingface_hub import hf_hub_download  # type: ignore
        from timm.data import resolve_data_config  # type: ignore
        from timm.data.transforms_factory import create_transform  # type: ignore
        from timm.layers import SwiGLUPacked  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError(
            "Pinned Virchow2 extraction requires torch, timm, huggingface_hub, and Pillow."
        ) from exc
    revision = str(model_revision)
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision.lower())
    ):
        raise ValueError("Virchow2 model_revision must be an exact 40-character commit.")
    prefix = "hf-hub:"
    if not str(model_ref).startswith(prefix):
        raise ValueError("Pinned Virchow2 loading requires an hf-hub model_ref.")
    repo_id = str(model_ref)[len(prefix) :]
    config_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
            cache_dir=None if hf_hub_cache_path is None else str(hf_hub_cache_path),
            local_files_only=hf_hub_local_files_only,
        )
    )
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            revision=revision,
            cache_dir=None if hf_hub_cache_path is None else str(hf_hub_cache_path),
            local_files_only=hf_hub_local_files_only,
        )
    )
    resolved_revisions = {
        _snapshot_revision(config_path),
        _snapshot_revision(checkpoint_path),
    }
    if resolved_revisions != {revision}:
        raise ValueError(
            f"Hugging Face revision resolution drifted: {sorted(resolved_revisions)}"
        )
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_payload, Mapping):
        raise ValueError("Virchow2 Hugging Face config must be a JSON object.")
    architecture = str(config_payload["architecture"])
    model_args = config_payload.get("model_args", {})
    pretrained_cfg = config_payload.get("pretrained_cfg", {})
    if not isinstance(model_args, Mapping) or not isinstance(pretrained_cfg, Mapping):
        raise ValueError("Virchow2 config model_args/pretrained_cfg must be mappings.")
    resolved_device = _resolve_device(torch, device)
    model = timm.create_model(
        architecture,
        pretrained=False,
        pretrained_cfg=dict(pretrained_cfg),
        checkpoint_path=str(checkpoint_path),
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        **dict(model_args),
    )
    model.eval()
    state_hash = checkpoint_sha256(model)
    preprocessing_config = dict(
        sorted(resolve_data_config(model.pretrained_cfg, model=model).items())
    )
    preprocessing_hash = stable_hash(preprocessing_config)
    identity = {
        "schema_version": "midogpp_virchow2_pinned_identity_v1",
        "model_ref": str(model_ref),
        "requested_revision": revision,
        "resolved_revision": revision,
        "model_config_sha256": _sha256_file(config_path),
        "checkpoint_file_sha256": _sha256_file(checkpoint_path),
        "state_dict_sha256": state_hash,
        "preprocessing_config": preprocessing_config,
        "preprocessing_config_hash": preprocessing_hash,
    }
    model.to(resolved_device)
    transform = create_transform(**preprocessing_config)
    return {
        "model": model,
        "transform": transform,
        "device": resolved_device,
        "identity": identity,
    }


def _snapshot_revision(path: Path) -> str:
    # Do not resolve the Hub symlink: its target lives under ``blobs/`` and
    # intentionally omits the snapshot commit encoded by the requested path.
    parts = path.parts
    try:
        index = parts.index("snapshots")
    except ValueError as exc:
        raise ValueError(f"Hugging Face file is not in a snapshot path: {path}") from exc
    if index + 1 >= len(parts):
        raise ValueError(f"Hugging Face snapshot revision is missing: {path}")
    return parts[index + 1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(torch: Any, raw: str) -> Any:
    requested = str(raw or "auto").lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(requested)


__all__ = [
    "CENTER_SLICE",
    "CLS_DIM",
    "PATCH_GRID_SIDE",
    "PATCH_TOKEN_COUNT",
    "PATCH_TOKEN_START",
    "VIRCHOW2_TOKEN_LAYOUT",
    "Virchow2TokenLayout",
    "Virchow2TokenExtractor",
    "assert_preprocessing_spatial_identity",
    "central_virchow2_token_grid",
    "checkpoint_sha256",
    "describe_preprocessing_spatial_identity",
    "normalized_coordinate_to_window_start",
    "normalized_position_to_window_start",
    "pool_virchow2_tokens",
    "pool_virchow2_tokens_v2",
    "resolve_tensor",
    "resolve_virchow2_identity",
    "validate_virchow2_token_layout",
    "validate_window_starts",
]
