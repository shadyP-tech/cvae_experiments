"""C4.1 heteroscedastic class-conditioned PCA generator utilities.

This module owns source-only generator mechanics. It does not make routing
decisions and does not consume target support or target evaluation labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import torch

from .protocol import ProtocolError


GENERATOR_FAMILY_PLAIN_CLASS_CONDITIONAL = "family_c_pca64_class_conditional_cvae_downstream_v1"
GENERATOR_FAMILY_HETEROSCEDASTIC = "family_c_pca64_class_conditional_heteroscedastic_cvae_downstream_v1"

GENERATION_MODE_POSTERIOR_DECODER_MEAN = "posterior_sample_decoder_mean"
GENERATION_MODE_POSTERIOR_DECODER_NOISE = "posterior_sample_decoder_noise"
C41_GENERATION_MODES = (
    GENERATION_MODE_POSTERIOR_DECODER_MEAN,
    GENERATION_MODE_POSTERIOR_DECODER_NOISE,
)

ROUTING_FAMILY_USED = "plain_pca64_class_conditional_cvae"
SELECTED_EXPERT_IDS_SOURCE = "locked_plain_class_conditional_support_nelbo"
ROUTING_SCORES_RECOMPUTED_FOR_HETEROSCEDASTIC = 0


class ClassConditionedDecoder(Protocol):
    decoder_likelihood: str
    decoder_logvar_min: float
    decoder_logvar_max: float
    decoder_min_variance: float

    def encode(self, x: torch.Tensor, y: torch.Tensor | None = None):
        ...

    def decode(self, z: torch.Tensor, y: torch.Tensor | None = None, return_distribution: bool = False):
        ...


@dataclass(frozen=True)
class SourceTrainPCAProjection:
    source_domain: str
    seed: int
    fit_split: str
    requested_components: int
    effective_components: int
    mean: torch.Tensor
    components: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_ratio: torch.Tensor

    def transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("PCA transform expects a 2D embedding tensor.")
        if embeddings.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"PCA transform width mismatch: expected {self.mean.shape[0]}, got {embeddings.shape[1]}."
            )
        return (embeddings.to(dtype=self.mean.dtype) - self.mean) @ self.components.T

    def inverse_transform(self, projected: torch.Tensor) -> torch.Tensor:
        if projected.ndim != 2:
            raise ValueError("PCA inverse_transform expects a 2D tensor.")
        if projected.shape[1] != self.effective_components:
            raise ValueError(
                "PCA inverse_transform width mismatch: "
                f"expected {self.effective_components}, got {projected.shape[1]}."
            )
        return projected @ self.components + self.mean

    def provenance(self) -> dict[str, object]:
        return {
            "projection_type": "pca",
            "projection_family": "source_train_pca64",
            "source_domain": self.source_domain,
            "seed": int(self.seed),
            "fit_split": self.fit_split,
            "requested_components": int(self.requested_components),
            "effective_components": int(self.effective_components),
            "explained_variance_sum": float(self.explained_variance.sum().item()),
            "explained_variance_ratio_sum": float(self.explained_variance_ratio.sum().item()),
        }


@dataclass(frozen=True)
class GeneratedBatch:
    embeddings: torch.Tensor
    labels: torch.Tensor
    generation_mode: str
    diagnostics: Mapping[str, float]


def c41_routing_provenance_fields() -> dict[str, object]:
    return {
        "routing_family_used": ROUTING_FAMILY_USED,
        "routing_scores_recomputed_for_heteroscedastic": ROUTING_SCORES_RECOMPUTED_FOR_HETEROSCEDASTIC,
        "selected_expert_ids_source": SELECTED_EXPERT_IDS_SOURCE,
    }


def fit_source_train_pca_projection(
    *,
    train_embeddings: torch.Tensor,
    train_metadata: Sequence[Mapping[str, object]],
    source_domain: str,
    seed: int,
    n_components: int = 64,
    domain_field: str = "magnification",
) -> SourceTrainPCAProjection:
    """Fit PCA using only source-train rows from one source expert domain."""

    if train_embeddings.ndim != 2:
        raise ValueError("train_embeddings must be a 2D tensor.")
    if train_embeddings.shape[0] != len(train_metadata):
        raise ValueError("train_embeddings and train_metadata must have the same number of rows.")
    if int(n_components) <= 0:
        raise ValueError("n_components must be positive.")

    indices = [
        idx
        for idx, row in enumerate(train_metadata)
        if str(_domain(row, domain_field=domain_field)) == str(source_domain)
    ]
    if not indices:
        raise ProtocolError(f"No source-train rows found for source_domain={source_domain!r}.")
    x = train_embeddings[indices].to(dtype=torch.float64)
    max_components = min(int(x.shape[0]), int(x.shape[1]), int(n_components))
    if max_components < int(n_components):
        raise ProtocolError(
            f"Cannot fit PCA{n_components} for source_domain={source_domain!r}: "
            f"only {x.shape[0]} source-train rows and {x.shape[1]} features are available."
        )
    mean = x.mean(dim=0)
    centered = x - mean
    _u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:max_components].contiguous()
    denom = float(max(int(x.shape[0]) - 1, 1))
    explained_variance_all = singular_values.pow(2) / denom
    if int(x.shape[0]) > 1:
        total_variance = centered.var(dim=0, unbiased=True).sum().clamp_min(1.0e-12)
    else:
        total_variance = centered.pow(2).sum().clamp_min(1.0e-12)
    explained_variance = explained_variance_all[:max_components].contiguous()
    explained_variance_ratio = (explained_variance / total_variance).contiguous()
    return SourceTrainPCAProjection(
        source_domain=str(source_domain),
        seed=int(seed),
        fit_split="source_train",
        requested_components=int(n_components),
        effective_components=int(max_components),
        mean=mean.to(dtype=train_embeddings.dtype),
        components=components.to(dtype=train_embeddings.dtype),
        explained_variance=explained_variance.to(dtype=train_embeddings.dtype),
        explained_variance_ratio=explained_variance_ratio.to(dtype=train_embeddings.dtype),
    )


def labels_from_metadata(metadata: Sequence[Mapping[str, object]], label_field: str = "label") -> torch.Tensor:
    labels: list[int] = []
    for row in metadata:
        if label_field not in row:
            raise ProtocolError(f"Metadata row is missing required label field {label_field!r}.")
        labels.append(int(row[label_field]))
    return torch.tensor(labels, dtype=torch.long)


def build_source_train_reference_pools(
    *,
    train_projected_embeddings: torch.Tensor,
    train_metadata: Sequence[Mapping[str, object]],
    source_domain: str,
    label_values: Sequence[int],
    domain_field: str = "magnification",
    label_field: str = "label",
) -> dict[int, torch.Tensor]:
    """Build class reference pools from source-train only."""

    if train_projected_embeddings.shape[0] != len(train_metadata):
        raise ValueError("train_projected_embeddings and train_metadata must have the same number of rows.")
    pools: dict[int, torch.Tensor] = {}
    for label in sorted(int(v) for v in label_values):
        indices = [
            idx
            for idx, row in enumerate(train_metadata)
            if str(_domain(row, domain_field=domain_field)) == str(source_domain)
            and int(row.get(label_field, -1)) == int(label)
        ]
        if not indices:
            raise ProtocolError(
                f"Empty source-train reference pool for source_domain={source_domain!r}, label={label}."
            )
        pools[int(label)] = train_projected_embeddings[indices]
    return pools


def generate_posterior_sampled_embeddings(
    *,
    model: ClassConditionedDecoder,
    reference_pool: torch.Tensor,
    class_label: int,
    n_samples: int,
    seed: int,
    generation_mode: str,
) -> GeneratedBatch:
    if generation_mode not in C41_GENERATION_MODES:
        raise ProtocolError(f"Unknown C4.1 generation_mode: {generation_mode}")
    if int(n_samples) <= 0:
        raise ValueError("n_samples must be positive.")
    if reference_pool.ndim != 2 or int(reference_pool.shape[0]) <= 0:
        raise ProtocolError("reference_pool must be a non-empty 2D tensor.")

    device = reference_pool.device
    index_gen = _generator_for_device(torch.device("cpu"), int(seed))
    indices = torch.randint(int(reference_pool.shape[0]), (int(n_samples),), generator=index_gen, device="cpu")
    xb = reference_pool[indices.to(device)]
    y = torch.full((int(n_samples),), int(class_label), dtype=torch.long, device=device)

    latent_gen = _generator_for_device(device, int(seed) + 104729)
    decoder_gen = _generator_for_device(device, int(seed) + 209759)
    with torch.no_grad():
        mu_z, logvar_z = model.encode(xb, y=y)
        z = mu_z + torch.exp(0.5 * logvar_z) * _randn_like(mu_z, generator=latent_gen)
        decoder_output = model.decode(z, y=y, return_distribution=True)
        if not isinstance(decoder_output, tuple) or len(decoder_output) != 2:
            raise ValueError("Class-conditioned decoder must return (mu_x, logvar_x) with return_distribution=True")
        mu_x, logvar_x = decoder_output
        if generation_mode == GENERATION_MODE_POSTERIOR_DECODER_MEAN:
            embeddings = mu_x
            noise = torch.zeros_like(mu_x)
        else:
            if logvar_x is None:
                raise ProtocolError("posterior_sample_decoder_noise requires a gaussian_diag decoder")
            noise = torch.exp(0.5 * logvar_x) * _randn_like(mu_x, generator=decoder_gen)
            embeddings = mu_x + noise

    labels = torch.full((int(n_samples),), int(class_label), dtype=torch.long)
    diagnostics = decoder_sample_diagnostics(model=model, mu_x=mu_x, logvar_x=logvar_x, noise=noise)
    return GeneratedBatch(
        embeddings=embeddings.detach().cpu(),
        labels=labels,
        generation_mode=generation_mode,
        diagnostics=diagnostics,
    )


def decoder_sample_diagnostics(
    *,
    model: ClassConditionedDecoder,
    mu_x: torch.Tensor,
    logvar_x: torch.Tensor | None,
    noise: torch.Tensor,
) -> dict[str, float]:
    mean_energy = mu_x.pow(2).sum(dim=1).mean().clamp_min(1.0e-12)
    noise_energy = noise.pow(2).sum(dim=1).mean()
    diagnostics = {
        "decoder_noise_energy_ratio": float((noise_energy / mean_energy).item()),
    }
    if logvar_x is None:
        return diagnostics

    sigma = torch.exp(0.5 * logvar_x)
    min_logvar = max(float(model.decoder_logvar_min), math.log(float(model.decoder_min_variance)))
    max_logvar = float(model.decoder_logvar_max)
    diagnostics.update(
        {
            "decoder_logvar_mean": float(logvar_x.mean().item()),
            "decoder_logvar_min": float(logvar_x.min().item()),
            "decoder_logvar_max": float(logvar_x.max().item()),
            "decoder_logvar_at_min_frac": float(
                torch.isclose(logvar_x, torch.tensor(min_logvar, device=logvar_x.device))
                .float()
                .mean()
                .item()
            ),
            "decoder_logvar_at_max_frac": float(
                torch.isclose(logvar_x, torch.tensor(max_logvar, device=logvar_x.device))
                .float()
                .mean()
                .item()
            ),
            "decoder_sigma_mean": float(sigma.mean().item()),
        }
    )
    return diagnostics


def decoder_logvar_diagnostics_by_class(
    *,
    model: ClassConditionedDecoder,
    reference_pools: Mapping[int, torch.Tensor],
) -> dict[str, float]:
    class_sigma_means: list[float] = []
    logvars: list[torch.Tensor] = []
    try:
        model_device = next(model.parameters()).device  # type: ignore[attr-defined]
    except (AttributeError, StopIteration):
        model_device = torch.device("cpu")
    for class_label, refs in sorted(reference_pools.items()):
        if refs.ndim != 2 or refs.shape[0] <= 0:
            continue
        refs_device = refs.to(model_device)
        y = torch.full((int(refs_device.shape[0]),), int(class_label), dtype=torch.long, device=model_device)
        with torch.no_grad():
            mu_z, _logvar_z = model.encode(refs_device, y=y)
            _mu_x, logvar_x = model.decode(mu_z, y=y, return_distribution=True)
        if logvar_x is None:
            continue
        logvars.append(logvar_x.detach().cpu())
        class_sigma_means.append(float(torch.exp(0.5 * logvar_x).mean().item()))
    if not logvars:
        return {
            "decoder_sigma_class_mean": math.nan,
            "decoder_sigma_class_ratio": math.nan,
        }
    sigma_min = min(class_sigma_means)
    sigma_max = max(class_sigma_means)
    all_logvar = torch.cat(logvars, dim=0)
    return {
        "decoder_logvar_mean": float(all_logvar.mean().item()),
        "decoder_logvar_min": float(all_logvar.min().item()),
        "decoder_logvar_max": float(all_logvar.max().item()),
        "decoder_sigma_mean": float(torch.exp(0.5 * all_logvar).mean().item()),
        "decoder_sigma_class_mean": float(sum(class_sigma_means) / float(len(class_sigma_means))),
        "decoder_sigma_class_ratio": float(sigma_max / max(sigma_min, 1.0e-12)),
    }


def _domain(row: Mapping[str, object], *, domain_field: str) -> object:
    if domain_field in row:
        return row[domain_field]
    for fallback in ("magnification", "center", "domain"):
        if fallback in row:
            return row[fallback]
    raise ProtocolError(f"Metadata row is missing domain field {domain_field!r}.")


def _generator_for_device(device: torch.device, seed: int) -> torch.Generator:
    if device.type == "cuda":
        return torch.Generator(device=device).manual_seed(int(seed))
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _randn_like(x: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
