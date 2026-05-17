from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F

from src.data.metadata_conditioning import build_domain_one_hot, resolve_domain_order
from src.torch_utils import safe_torch_load
from src.train.checkpoint_provenance import build_checkpoint_metadata_from_cache
from src.train.train_utils import run_training


def _indices_by_domain(payload: Dict[str, object], domain: int) -> list[int]:
    metadata = payload["metadata"]
    idxs = [i for i, m in enumerate(metadata) if int(m["magnification"]) == domain]
    return idxs


def _filter_by_domain(payload: Dict[str, object], domain: int):
    embeddings = payload["embeddings"]
    idxs = _indices_by_domain(payload, domain)
    if not idxs:
        return torch.empty((0, embeddings.shape[1]))
    return embeddings[idxs]


def build_label_one_hot(metadata: Sequence[dict], label_values: Sequence[int], label_field: str = "label") -> torch.Tensor:
    label_order = [int(v) for v in label_values]
    if not label_order:
        raise ValueError("label_values must be non-empty for label conditioning")
    label_to_index = {label: idx for idx, label in enumerate(label_order)}
    targets: list[int] = []
    for i, item in enumerate(metadata):
        if label_field not in item:
            raise ValueError(f"Missing label field '{label_field}' in metadata item at index {i}.")
        label = int(item[label_field])
        if label not in label_to_index:
            raise ValueError(
                f"Observed label '{label}' at index {i} is not present in configured label_values: {label_order}"
            )
        targets.append(label_to_index[label])
    target_tensor = torch.tensor(targets, dtype=torch.long)
    return F.one_hot(target_tensor, num_classes=len(label_order)).to(dtype=torch.float32)


def train_domain_experts(
    train_cache: Path,
    val_cache: Path,
    out_dir: Path,
    domains: list[int],
    hidden_dim: int,
    latent_dim: int,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    resume_from_dir: Path | None = None,
    conditioning_cfg: Dict[str, Any] | None = None,
    configured_domains: Sequence[int] | None = None,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
    label_conditioning_cfg: Dict[str, Any] | None = None,
    checkpoint_metadata: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_payload = safe_torch_load(train_cache, map_location="cpu")
    val_payload = safe_torch_load(val_cache, map_location="cpu")
    input_dim = int(train_payload["embeddings"].shape[1])
    cond_cfg = conditioning_cfg or {}
    conditioning_enabled = bool(cond_cfg.get("enabled", False))
    metadata_dim = 0
    train_meta_all = None
    val_meta_all = None
    if conditioning_enabled:
        domain_order = resolve_domain_order(configured_domains or domains)
        train_meta_all = build_domain_one_hot(train_payload["metadata"], domain_order)
        val_meta_all = build_domain_one_hot(val_payload["metadata"], domain_order)
        metadata_dim = int(len(domain_order))
    label_cfg = label_conditioning_cfg or {}
    label_conditioning_enabled = bool(label_cfg.get("enabled", False))
    label_values = [int(v) for v in label_cfg.get("label_values", [])]
    label_field = str(label_cfg.get("label_field", "label"))
    class_condition_dim = 0
    train_y_all = None
    val_y_all = None
    if label_conditioning_enabled:
        if not label_values:
            raise ValueError("model.label_conditioning.label_values must be non-empty when enabled")
        train_y_all = build_label_one_hot(train_payload["metadata"], label_values, label_field=label_field)
        val_y_all = build_label_one_hot(val_payload["metadata"], label_values, label_field=label_field)
        class_condition_dim = int(len(label_values))

    output: Dict[str, str] = {}
    for domain in domains:
        train_idxs = _indices_by_domain(train_payload, domain)
        val_idxs = _indices_by_domain(val_payload, domain)
        train_x = train_payload["embeddings"][train_idxs] if train_idxs else torch.empty((0, input_dim))
        val_x = val_payload["embeddings"][val_idxs] if val_idxs else torch.empty((0, input_dim))
        if train_x.numel() == 0 or val_x.numel() == 0:
            continue

        train_m = train_meta_all[train_idxs] if (conditioning_enabled and train_meta_all is not None) else None
        val_m = val_meta_all[val_idxs] if (conditioning_enabled and val_meta_all is not None) else None
        train_y = train_y_all[train_idxs] if (label_conditioning_enabled and train_y_all is not None) else None
        val_y = val_y_all[val_idxs] if (label_conditioning_enabled and val_y_all is not None) else None
        checkpoint_extra = {"model_name": f"expert_{domain}x", "expert_domain": int(domain)}
        if label_conditioning_enabled:
            checkpoint_extra.update(
                {
                    "expert_family": "family_c_label_conditioned_v1",
                    "condition_type": "class_label_one_hot",
                    "label_field": label_field,
                    "label_values": label_values,
                    "class_condition_dim": int(class_condition_dim),
                    "beta_kl_weight": 1.0,
                    "reconstruction_loss": "mse_sum",
                    "likelihood_variance_assumption": "unit",
                }
            )

        result = run_training(
            train_embeddings=train_x,
            val_embeddings=val_x,
            out_dir=out_dir,
            model_name=f"expert_{domain}x",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            lr=lr,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            resume_from=(resume_from_dir / f"expert_{domain}x.pt") if resume_from_dir is not None else None,
            train_metadata_vectors=train_m,
            val_metadata_vectors=val_m,
            train_class_condition_vectors=train_y,
            val_class_condition_vectors=val_y,
            metadata_dim=metadata_dim,
            class_condition_dim=class_condition_dim,
            metadata_constraint_cfg=metadata_constraint_cfg,
            checkpoint_metadata=checkpoint_metadata
            or build_checkpoint_metadata_from_cache(
                train_payload,
                extra=checkpoint_extra,
            ),
        )
        output[f"{domain}x"] = str(result.checkpoint_path)

    with (out_dir / "expert_checkpoints.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return output
