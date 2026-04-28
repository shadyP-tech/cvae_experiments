from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from src.data.metadata_conditioning import build_domain_one_hot, resolve_domain_order
from src.torch_utils import safe_torch_load
from src.train.checkpoint_provenance import build_checkpoint_metadata_from_cache
from src.train.train_utils import run_training


def train_global_model(
    train_cache: Path,
    val_cache: Path,
    out_dir: Path,
    hidden_dim: int,
    latent_dim: int,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    resume_from: Path | None = None,
    conditioning_cfg: Dict[str, Any] | None = None,
    configured_domains: Sequence[int] | None = None,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
    checkpoint_metadata: Dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_payload = safe_torch_load(train_cache, map_location="cpu")
    val_payload = safe_torch_load(val_cache, map_location="cpu")

    input_dim = int(train_payload["embeddings"].shape[1])
    cond_cfg = conditioning_cfg or {}
    conditioning_enabled = bool(cond_cfg.get("enabled", False))
    metadata_dim = 0
    train_meta_vectors = None
    val_meta_vectors = None
    if conditioning_enabled:
        domains = resolve_domain_order(configured_domains or [])
        train_meta_vectors = build_domain_one_hot(train_payload["metadata"], domains)
        val_meta_vectors = build_domain_one_hot(val_payload["metadata"], domains)
        metadata_dim = int(len(domains))

    result = run_training(
        train_embeddings=train_payload["embeddings"],
        val_embeddings=val_payload["embeddings"],
        out_dir=out_dir,
        model_name="global_cvae",
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        lr=lr,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        resume_from=resume_from,
        train_metadata_vectors=train_meta_vectors,
        val_metadata_vectors=val_meta_vectors,
        metadata_dim=metadata_dim,
        metadata_constraint_cfg=metadata_constraint_cfg,
        checkpoint_metadata=checkpoint_metadata
        or build_checkpoint_metadata_from_cache(train_payload, extra={"model_name": "global_cvae"}),
    )
    return result.checkpoint_path
