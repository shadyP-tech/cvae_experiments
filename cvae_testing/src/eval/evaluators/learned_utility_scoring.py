from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.data.metadata_conditioning import build_domain_one_hot, resolve_domain_order
from src.models.cvae_expert import CVAEExpert, elbo_components
from src.torch_utils import safe_torch_load
from src.train.checkpoint_provenance import load_model_checkpoint


def _as_domain_from_meta(value: object) -> int:
    return int(str(value).replace("x", ""))


def _load_model(
    checkpoint: Path,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    device: torch.device,
    metadata_dim: int = 0,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
) -> CVAEExpert:
    model = CVAEExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        metadata_dim=int(metadata_dim),
        metadata_constraint_cfg=metadata_constraint_cfg,
        aux_metadata_dim=int(metadata_dim),
    ).to(device)
    model.load_state_dict(load_model_checkpoint(checkpoint, map_location=device).model_state_dict)
    model.eval()
    return model


def _score_model_nelbo(model: CVAEExpert, x: torch.Tensor, m: torch.Tensor | None = None) -> torch.Tensor:
    recon, mu, logvar = model(x, m=m)
    prior_mu, prior_logvar, kl_weight = model.metadata_constraint_prior(metadata_targets=m)
    rec, kl = elbo_components(
        recon,
        x,
        mu,
        logvar,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        kl_weight=kl_weight,
    )
    return rec + kl


def _parse_expert_domain(name: str) -> int:
    text = str(name)
    if text.startswith("expert_"):
        text = text[len("expert_") :]

    # Accept keys like: expert_40x, expert_100, expert_100.training
    match = re.match(r"^(\d+)", text.replace("x", ""))
    if match is not None:
        return int(match.group(1))

    raise ValueError(f"Cannot parse expert domain from checkpoint key: {name}")


def _score_experts_batched(
    *,
    test_cache: Path,
    expert_checkpoints: Dict[str, str],
    hidden_dim: int,
    latent_dim: int,
    pair_batch_size: int,
    conditioning_cfg: Dict[str, Any] | None = None,
    configured_domains: Sequence[int] | None = None,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[Dict[str, Any]]]:
    payload = safe_torch_load(test_cache, map_location="cpu")
    x_cpu = payload["embeddings"]
    metadata = payload["metadata"]
    sample_domains = np.asarray([_as_domain_from_meta(m["magnification"]) for m in metadata], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = int(x_cpu.shape[1])

    cond_cfg = conditioning_cfg or {}
    conditioning_enabled = bool(cond_cfg.get("enabled", False))
    metadata_vectors_cpu = None
    metadata_dim = 0
    if conditioning_enabled:
        observed_domains = sorted(set(int(v) for v in sample_domains.tolist()))
        domain_order = resolve_domain_order(configured_domains or observed_domains)
        metadata_vectors_cpu = build_domain_one_hot(metadata, domain_order)
        metadata_dim = int(len(domain_order))

    expert_names = sorted(expert_checkpoints.keys())
    expert_domains = [_parse_expert_domain(name) for name in expert_names]
    models = [
        _load_model(
            Path(expert_checkpoints[name]),
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            device=device,
            metadata_dim=metadata_dim,
            metadata_constraint_cfg=metadata_constraint_cfg,
        )
        for name in expert_names
    ]

    x_np = x_cpu.detach().cpu().numpy().astype(np.float64, copy=False)
    n_samples = int(x_np.shape[0])
    n_experts = len(models)
    nelbo = np.zeros((n_samples, n_experts), dtype=np.float64)

    expert_chunks: List[List[torch.Tensor]] = [[] for _ in range(n_experts)]
    with torch.no_grad():
        # Move each batch to device once, then score with all experts.
        for i in range(0, n_samples, int(pair_batch_size)):
            xb = x_cpu[i : i + int(pair_batch_size)].to(device)
            mb = metadata_vectors_cpu[i : i + int(pair_batch_size)].to(device) if metadata_vectors_cpu is not None else None
            for e_idx, model in enumerate(models):
                expert_chunks[e_idx].append(_score_model_nelbo(model, xb, m=mb).cpu())

    for e_idx in range(n_experts):
        nelbo[:, e_idx] = torch.cat(expert_chunks[e_idx], dim=0).numpy().astype(np.float64, copy=False)

    return x_np, sample_domains, nelbo, expert_domains, metadata


def _domain_to_expert_index(expert_domains: Sequence[int]) -> Dict[int, int]:
    return {int(d): idx for idx, d in enumerate(expert_domains)}
