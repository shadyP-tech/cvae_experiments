from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch

from src.torch_utils import safe_torch_load
from src.train.hybrid.trainer import HybridAblationTrainer
from src.train.hybrid.variants import VARIANT_POOLED


def train_hybrid_variant(
    train_cache: Path,
    val_cache: Path,
    out_dir: Path,
    domains: List[int],
    projection_dim: int,
    head_hidden_dim: int,
    cvae_hidden_dim: int,
    latent_dim: int,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    variant: str,
    model_name: str,
    resume_from: Path | None = None,
) -> Dict[str, object]:
    train_payload = safe_torch_load(train_cache, map_location="cpu")
    val_payload = safe_torch_load(val_cache, map_location="cpu")

    trainer = HybridAblationTrainer(
        train_payload=train_payload,
        val_payload=val_payload,
        domains=domains,
        projection_dim=projection_dim,
        head_hidden_dim=head_hidden_dim,
        cvae_hidden_dim=cvae_hidden_dim,
        latent_dim=latent_dim,
        lr=lr,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
        variant=variant,
    )

    ckpt_path, history = trainer.train(out_dir=out_dir, model_name=model_name, resume_from=resume_from)
    out = {
        "variant": variant,
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    with (out_dir / f"{model_name}_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def train_hybrid_pooled_baseline(
    train_cache: Path,
    val_cache: Path,
    out_dir: Path,
    domains: List[int],
    projection_dim: int,
    head_hidden_dim: int,
    cvae_hidden_dim: int,
    latent_dim: int,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    model_name: str = "hybrid_pooled_baseline",
    resume_from: Path | None = None,
) -> Dict[str, object]:
    return train_hybrid_variant(
        train_cache=train_cache,
        val_cache=val_cache,
        out_dir=out_dir,
        domains=domains,
        projection_dim=projection_dim,
        head_hidden_dim=head_hidden_dim,
        cvae_hidden_dim=cvae_hidden_dim,
        latent_dim=latent_dim,
        lr=lr,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
        variant=VARIANT_POOLED,
        model_name=model_name,
        resume_from=resume_from,
    )
