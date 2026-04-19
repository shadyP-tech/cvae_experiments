from __future__ import annotations

from pathlib import Path
from typing import List

import torch

from src.train.hybrid.variants import HybridModuleBundle


def build_hybrid_checkpoint_payload(
    *,
    variant: str,
    domains: List[int],
    input_dim: int,
    projection_dim: int,
    head_hidden_dim: int,
    cvae_hidden_dim: int,
    latent_dim: int,
    metadata_constraint_cfg: dict | None,
    aux_metadata_dim: int,
    bundle: HybridModuleBundle,
) -> dict:
    return {
        "variant": variant,
        "domains": domains,
        "input_dim": input_dim,
        "projection_dim": projection_dim,
        "head_hidden_dim": head_hidden_dim,
        "cvae_hidden_dim": cvae_hidden_dim,
        "latent_dim": latent_dim,
        "metadata_constraint_cfg": dict(metadata_constraint_cfg or {}),
        "aux_metadata_dim": int(aux_metadata_dim),
        "shared_head": bundle.shared_head.state_dict() if bundle.shared_head is not None else None,
        "heads": {str(d): m.state_dict() for d, m in bundle.heads.items()},
        "shared_cvae": bundle.shared_cvae.state_dict() if bundle.shared_cvae is not None else None,
        "cvaes": {str(d): m.state_dict() for d, m in bundle.cvaes.items()},
    }


def save_hybrid_checkpoint(
    path: Path,
    variant: str,
    domains: List[int],
    input_dim: int,
    projection_dim: int,
    head_hidden_dim: int,
    cvae_hidden_dim: int,
    latent_dim: int,
    metadata_constraint_cfg: dict | None,
    aux_metadata_dim: int,
    bundle: HybridModuleBundle,
) -> None:
    payload = build_hybrid_checkpoint_payload(
        variant=variant,
        domains=domains,
        input_dim=input_dim,
        projection_dim=projection_dim,
        head_hidden_dim=head_hidden_dim,
        cvae_hidden_dim=cvae_hidden_dim,
        latent_dim=latent_dim,
        metadata_constraint_cfg=metadata_constraint_cfg,
        aux_metadata_dim=aux_metadata_dim,
        bundle=bundle,
    )
    torch.save(payload, path)
