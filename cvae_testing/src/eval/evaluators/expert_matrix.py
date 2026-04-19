from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from src.data.metadata_conditioning import build_domain_one_hot, resolve_domain_order
from src.eval.metrics import mean_and_variance
from src.models.cvae_expert import CVAEExpert, elbo_components
from src.torch_utils import safe_torch_load


def _load_model(
    checkpoint: Path,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    device: torch.device,
    metadata_dim: int = 0,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
    aux_metadata_dim: int | None = None,
):
    model = CVAEExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        metadata_dim=metadata_dim,
        metadata_constraint_cfg=metadata_constraint_cfg,
        aux_metadata_dim=aux_metadata_dim,
    ).to(device)
    model.load_state_dict(safe_torch_load(checkpoint, map_location=device))
    model.eval()
    return model


def compute_expert_domain_matrix(
    test_cache: Path,
    expert_checkpoints: Dict[str, str],
    hidden_dim: int,
    latent_dim: int,
    conditioning_cfg: Dict[str, Any] | None = None,
    configured_domains: Sequence[int] | None = None,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
) -> Dict[str, object]:
    payload = safe_torch_load(test_cache, map_location="cpu")
    x = payload["embeddings"]
    meta = payload["metadata"]
    input_dim = int(x.shape[1])

    domains = sorted(set(int(m["magnification"]) for m in meta))
    by_domain_indices = {d: [i for i, m in enumerate(meta) if int(m["magnification"]) == d] for d in domains}

    cond_cfg = conditioning_cfg or {}
    conditioning_enabled = bool(cond_cfg.get("enabled", False))
    metadata_dim = 0
    metadata_vectors = None
    if conditioning_enabled:
        domain_order = resolve_domain_order(configured_domains or [])
        metadata_vectors = build_domain_one_hot(meta, domain_order)
        metadata_dim = int(len(domain_order))

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    x = x.to(device)
    constraint_cfg = metadata_constraint_cfg or {}

    matrix: Dict[str, Dict[str, float]] = {}
    confidence: Dict[str, Dict[str, dict]] = {}

    for expert_domain, ckpt in expert_checkpoints.items():
        model = _load_model(
            Path(ckpt),
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            device=device,
            metadata_dim=metadata_dim,
            metadata_constraint_cfg=constraint_cfg,
            aux_metadata_dim=metadata_dim,
        )
        matrix[expert_domain] = {}
        confidence[expert_domain] = {}

        with torch.no_grad():
            for d in domains:
                idxs = by_domain_indices[d]
                if not idxs:
                    continue
                xs = x[idxs]
                ms = metadata_vectors[idxs].to(device) if metadata_vectors is not None else None
                recon, mu, logvar = model(xs, m=ms)
                prior_mu, prior_logvar, kl_weight = model.metadata_constraint_prior(metadata_targets=ms)
                rec, kl = elbo_components(
                    recon,
                    xs,
                    mu,
                    logvar,
                    prior_mu=prior_mu,
                    prior_logvar=prior_logvar,
                    kl_weight=kl_weight,
                )
                nelbo = rec + kl
                matrix[expert_domain][f"{d}x"] = float(rec.mean().item())
                confidence[expert_domain][f"{d}x"] = mean_and_variance(nelbo.tolist())

    return {
        "reconstruction_matrix": matrix,
        "confidence": confidence,
    }
