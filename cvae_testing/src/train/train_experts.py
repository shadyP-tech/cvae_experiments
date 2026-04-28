from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

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
            metadata_dim=metadata_dim,
            metadata_constraint_cfg=metadata_constraint_cfg,
            checkpoint_metadata=checkpoint_metadata
            or build_checkpoint_metadata_from_cache(
                train_payload,
                extra={"model_name": f"expert_{domain}x", "expert_domain": int(domain)},
            ),
        )
        output[f"{domain}x"] = str(result.checkpoint_path)

    with (out_dir / "expert_checkpoints.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return output
