from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from src.models.cvae_expert import CVAEExpert
from src.models.projection_head import ProjectionHead


VARIANT_A = "A"  # shared head + expert CVAE
VARIANT_B = "B"  # expert heads + shared pooled CVAE
VARIANT_C = "C"  # expert heads + expert CVAE
VARIANT_POOLED = "POOLED"  # shared head + shared pooled CVAE


@dataclass
class HybridModuleBundle:
    shared_head: ProjectionHead | None = None
    heads: Dict[int, ProjectionHead] = field(default_factory=dict)
    shared_cvae: CVAEExpert | None = None
    cvaes: Dict[int, CVAEExpert] = field(default_factory=dict)


def build_hybrid_modules(
    variant: str,
    device: torch.device,
    input_dim: int,
    projection_dim: int,
    head_hidden_dim: int,
    cvae_hidden_dim: int,
    latent_dim: int,
    domains: List[int],
    metadata_constraint_cfg: dict[str, object] | None = None,
    aux_metadata_dim: int | None = None,
) -> HybridModuleBundle:
    bundle = HybridModuleBundle()

    if variant == VARIANT_A:
        bundle.shared_head = ProjectionHead(input_dim, projection_dim, head_hidden_dim).to(device)
        for d in domains:
            bundle.cvaes[d] = CVAEExpert(
                projection_dim,
                cvae_hidden_dim,
                latent_dim,
                metadata_constraint_cfg=metadata_constraint_cfg,
                aux_metadata_dim=aux_metadata_dim,
            ).to(device)
        return bundle

    if variant == VARIANT_POOLED:
        bundle.shared_head = ProjectionHead(input_dim, projection_dim, head_hidden_dim).to(device)
        bundle.shared_cvae = CVAEExpert(
            projection_dim,
            cvae_hidden_dim,
            latent_dim,
            metadata_constraint_cfg=metadata_constraint_cfg,
            aux_metadata_dim=aux_metadata_dim,
        ).to(device)
        return bundle

    if variant == VARIANT_B:
        for d in domains:
            bundle.heads[d] = ProjectionHead(input_dim, projection_dim, head_hidden_dim).to(device)
        bundle.shared_cvae = CVAEExpert(
            projection_dim,
            cvae_hidden_dim,
            latent_dim,
            metadata_constraint_cfg=metadata_constraint_cfg,
            aux_metadata_dim=aux_metadata_dim,
        ).to(device)
        return bundle

    if variant == VARIANT_C:
        for d in domains:
            bundle.heads[d] = ProjectionHead(input_dim, projection_dim, head_hidden_dim).to(device)
            bundle.cvaes[d] = CVAEExpert(
                projection_dim,
                cvae_hidden_dim,
                latent_dim,
                metadata_constraint_cfg=metadata_constraint_cfg,
                aux_metadata_dim=aux_metadata_dim,
            ).to(device)
        return bundle

    raise ValueError(f"Unsupported hybrid variant: {variant}")
