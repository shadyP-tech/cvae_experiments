"""C4.2 source-class latent GMM prior utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .c41_heteroscedastic import GeneratedBatch
from .protocol import ProtocolError
from .schemas import (
    C42_LATENT_GMM_K1_GENERATION_MODE,
    C42_LATENT_GMM_K2_GENERATION_MODE,
    C42_LATENT_GMM_K4_GENERATION_MODE,
    C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE,
)


C42_LATENT_GMM_COMPONENTS_BY_MODE = {
    C42_LATENT_GMM_K1_GENERATION_MODE: 1,
    C42_LATENT_GMM_K2_GENERATION_MODE: 2,
    C42_LATENT_GMM_K4_GENERATION_MODE: 4,
}
C42_LATENT_GMM_GENERATION_MODES = tuple(C42_LATENT_GMM_COMPONENTS_BY_MODE)


@dataclass(frozen=True)
class SourceClassLatentDiagGMM:
    experiment_seed: int
    source_domain: str
    class_label: int
    requested_components: int
    effective_components: int
    covariance_floor: float
    weights: torch.Tensor
    means: torch.Tensor
    variances: torch.Tensor
    converged: int
    n_iter: int
    lower_bound: float
    class_count: int
    component_clipped: int
    diagnostics: Mapping[str, float]

    def sample(self, n_samples: int, *, seed: int, device: torch.device) -> torch.Tensor:
        if int(n_samples) <= 0:
            raise ValueError("n_samples must be positive.")
        weights = self.weights.to(device=device, dtype=torch.float32)
        means = self.means.to(device=device, dtype=torch.float32)
        variances = self.variances.to(device=device, dtype=torch.float32).clamp_min(float(self.covariance_floor))
        gen = torch.Generator(device=device if device.type == "cuda" else "cpu").manual_seed(int(seed))
        component_ids = torch.multinomial(weights, int(n_samples), replacement=True, generator=gen).to(device)
        eps = torch.randn((int(n_samples), means.shape[1]), generator=gen, device=device, dtype=means.dtype)
        return means[component_ids] + eps * torch.sqrt(variances[component_ids])

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "source_class_latent_diag_gmm_v1",
            "experiment_seed": int(self.experiment_seed),
            "source_domain": self.source_domain,
            "class_label": int(self.class_label),
            "requested_components": int(self.requested_components),
            "effective_components": int(self.effective_components),
            "covariance_floor": float(self.covariance_floor),
            "weights": self.weights.detach().cpu(),
            "means": self.means.detach().cpu(),
            "variances": self.variances.detach().cpu(),
            "converged": int(self.converged),
            "n_iter": int(self.n_iter),
            "lower_bound": float(self.lower_bound),
            "class_count": int(self.class_count),
            "component_clipped": int(self.component_clipped),
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SourceClassLatentDiagGMM":
        return cls(
            experiment_seed=int(payload["experiment_seed"]),
            source_domain=str(payload["source_domain"]),
            class_label=int(payload["class_label"]),
            requested_components=int(payload["requested_components"]),
            effective_components=int(payload["effective_components"]),
            covariance_floor=float(payload["covariance_floor"]),
            weights=torch.as_tensor(payload["weights"], dtype=torch.float32),
            means=torch.as_tensor(payload["means"], dtype=torch.float32),
            variances=torch.as_tensor(payload["variances"], dtype=torch.float32),
            converged=int(payload["converged"]),
            n_iter=int(payload["n_iter"]),
            lower_bound=float(payload["lower_bound"]),
            class_count=int(payload["class_count"]),
            component_clipped=int(payload["component_clipped"]),
            diagnostics=dict(payload.get("diagnostics", {}) or {}),
        )


def fit_source_class_latent_gmm(
    *,
    model,
    projected_embeddings: torch.Tensor,
    labels: torch.Tensor,
    experiment_seed: int,
    source_domain: str,
    class_label: int,
    requested_components: int,
    fit_seed: int,
    covariance_floor: float = 1.0e-4,
) -> SourceClassLatentDiagGMM:
    if int(requested_components) <= 0:
        raise ValueError("requested_components must be positive.")
    indices = (labels.long() == int(class_label)).nonzero(as_tuple=False).flatten()
    if int(indices.numel()) <= 0:
        raise ProtocolError(f"No source-train latent rows for source_domain={source_domain}, class={class_label}.")
    device = next(model.parameters()).device
    x = projected_embeddings[indices].to(device=device)
    y = torch.full((int(x.shape[0]),), int(class_label), dtype=torch.long, device=device)
    gen = _generator_for_device(device, int(fit_seed) + 7919)
    with torch.no_grad():
        mu, logvar = model.encode(x, y=y)
        posterior_samples = mu + torch.exp(0.5 * logvar) * torch.randn(
            mu.shape,
            generator=gen,
            device=device,
            dtype=mu.dtype,
        )
    mu_cpu = mu.detach().cpu().float()
    posterior_cpu = posterior_samples.detach().cpu().float()
    effective = min(int(requested_components), int(mu_cpu.shape[0]))
    try:
        from sklearn.mixture import GaussianMixture  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("C4.2 latent GMM fitting requires scikit-learn.") from exc
    gmm = GaussianMixture(
        n_components=effective,
        covariance_type="diag",
        reg_covar=float(covariance_floor),
        random_state=int(fit_seed),
        max_iter=200,
        n_init=1,
    )
    gmm.fit(mu_cpu.numpy())
    weights = torch.as_tensor(gmm.weights_, dtype=torch.float32)
    means = torch.as_tensor(gmm.means_, dtype=torch.float32)
    variances = torch.as_tensor(gmm.covariances_, dtype=torch.float32).clamp_min(float(covariance_floor))
    sampled = _sample_from_tensors(weights, means, variances, n_samples=int(mu_cpu.shape[0]), seed=fit_seed + 1543)
    diagnostics = latent_gmm_diagnostics(
        posterior_mu=mu_cpu,
        posterior_samples=posterior_cpu,
        gmm_samples=sampled,
        weights=weights,
        variances=variances,
        covariance_floor=float(covariance_floor),
    )
    diagnostics.update(
        {
            "requested_components": float(requested_components),
            "effective_components": float(effective),
            "component_clipped": float(int(effective < int(requested_components))),
            "class_count": float(mu_cpu.shape[0]),
            "samples_per_effective_component": float(mu_cpu.shape[0]) / max(float(effective), 1.0),
            "converged": float(int(bool(gmm.converged_))),
            "n_iter": float(gmm.n_iter_),
            "lower_bound": float(gmm.lower_bound_),
        }
    )
    return SourceClassLatentDiagGMM(
        experiment_seed=int(experiment_seed),
        source_domain=str(source_domain),
        class_label=int(class_label),
        requested_components=int(requested_components),
        effective_components=int(effective),
        covariance_floor=float(covariance_floor),
        weights=weights,
        means=means,
        variances=variances,
        converged=int(bool(gmm.converged_)),
        n_iter=int(gmm.n_iter_),
        lower_bound=float(gmm.lower_bound_),
        class_count=int(mu_cpu.shape[0]),
        component_clipped=int(effective < int(requested_components)),
        diagnostics=diagnostics,
    )


def generate_latent_gmm_decoder_mean(
    *,
    model,
    prior: SourceClassLatentDiagGMM,
    class_label: int,
    n_samples: int,
    seed: int,
    generation_mode: str,
) -> GeneratedBatch:
    if generation_mode not in C42_LATENT_GMM_COMPONENTS_BY_MODE:
        raise ProtocolError(f"Unknown C4.2 latent GMM generation mode: {generation_mode}")
    device = next(model.parameters()).device
    z = prior.sample(int(n_samples), seed=int(seed), device=device)
    y = torch.full((int(n_samples),), int(class_label), dtype=torch.long, device=device)
    with torch.no_grad():
        embeddings = model.decode(z, y=y).detach().cpu()
    labels = torch.full((int(n_samples),), int(class_label), dtype=torch.long)
    diagnostics = {
        "decoder_output_norm_mean": float(embeddings.norm(dim=1).mean().item()),
        "decoder_output_norm_std": float(embeddings.norm(dim=1).std(unbiased=False).item()),
        "nan_or_inf_generated": int(not torch.isfinite(embeddings).all().item()),
    }
    diagnostics.update(dict(prior.diagnostics))
    return GeneratedBatch(embeddings=embeddings, labels=labels, generation_mode=generation_mode, diagnostics=diagnostics)


def generate_standard_prior_decoder_mean(
    *,
    model,
    class_label: int,
    n_samples: int,
    seed: int,
) -> GeneratedBatch:
    device = next(model.parameters()).device
    latent_dim = int(getattr(model, "latent_dim"))
    gen = _generator_for_device(device, int(seed))
    z = torch.randn((int(n_samples), latent_dim), generator=gen, device=device)
    y = torch.full((int(n_samples),), int(class_label), dtype=torch.long, device=device)
    with torch.no_grad():
        embeddings = model.decode(z, y=y).detach().cpu()
    labels = torch.full((int(n_samples),), int(class_label), dtype=torch.long)
    diagnostics = {
        "standard_prior_norm_mean": float(z.detach().cpu().norm(dim=1).mean().item()),
        "decoder_output_norm_mean": float(embeddings.norm(dim=1).mean().item()),
        "decoder_output_norm_std": float(embeddings.norm(dim=1).std(unbiased=False).item()),
        "nan_or_inf_generated": int(not torch.isfinite(embeddings).all().item()),
    }
    return GeneratedBatch(
        embeddings=embeddings,
        labels=labels,
        generation_mode=C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE,
        diagnostics=diagnostics,
    )


def latent_gmm_diagnostics(
    *,
    posterior_mu: torch.Tensor,
    posterior_samples: torch.Tensor,
    gmm_samples: torch.Tensor,
    weights: torch.Tensor,
    variances: torch.Tensor,
    covariance_floor: float,
) -> dict[str, float]:
    standard = torch.randn_like(gmm_samples)
    entropy = -(weights.clamp_min(1.0e-12) * torch.log(weights.clamp_min(1.0e-12))).sum()
    return {
        "latent_mu_norm_mean": float(posterior_mu.norm(dim=1).mean().item()),
        "latent_mu_norm_std": float(posterior_mu.norm(dim=1).std(unbiased=False).item()),
        "latent_sample_norm_mean": float(posterior_samples.norm(dim=1).mean().item()),
        "gmm_sample_norm_mean": float(gmm_samples.norm(dim=1).mean().item()),
        "standard_prior_norm_mean": float(standard.norm(dim=1).mean().item()),
        "gmm_vs_posterior_mu_mmd": rbf_mmd(gmm_samples, posterior_mu),
        "gmm_vs_posterior_sample_mmd": rbf_mmd(gmm_samples, posterior_samples),
        "gmm_trace_cov": _trace_cov(gmm_samples),
        "posterior_mu_trace_cov": _trace_cov(posterior_mu),
        "posterior_sample_trace_cov": _trace_cov(posterior_samples),
        "component_weight_entropy": float(entropy.item()),
        "min_component_weight": float(weights.min().item()),
        "covariance_floor": float(covariance_floor),
        "covariance_ceiling_hit": 0.0,
    }


def generated_embedding_diagnostics(
    *,
    synthetic_embeddings: torch.Tensor,
    synthetic_labels: Sequence[int],
    source_train_embeddings: torch.Tensor,
    source_train_labels: Sequence[int],
) -> dict[str, float]:
    syn = synthetic_embeddings.detach().cpu().float()
    src = source_train_embeddings.detach().cpu().float()
    syn_labels = [int(v) for v in synthetic_labels]
    src_labels = [int(v) for v in source_train_labels]
    return {
        "synthetic_pca64_mean_l2_to_source_train": float((syn.mean(dim=0) - src.mean(dim=0)).norm().item()),
        "synthetic_pca64_cov_trace_ratio_to_source_train": _trace_cov(syn) / max(_trace_cov(src), 1.0e-12),
        "synthetic_pairwise_distance_ratio_to_source_train": _pairwise_distance_mean(syn) / max(_pairwise_distance_mean(src), 1.0e-12),
        "synthetic_classifier_train_class_balance": _class_balance(syn_labels),
        "synthetic_count_class_0": float(sum(1 for value in syn_labels if value == 0)),
        "synthetic_count_class_1": float(sum(1 for value in syn_labels if value == 1)),
        "real_source_train_count_class_0": float(sum(1 for value in src_labels if value == 0)),
        "real_source_train_count_class_1": float(sum(1 for value in src_labels if value == 1)),
        "nan_or_inf_generated": float(int(not torch.isfinite(syn).all().item())),
        "decoder_output_norm_mean": float(syn.norm(dim=1).mean().item()),
        "decoder_output_norm_std": float(syn.norm(dim=1).std(unbiased=False).item()),
    }


def rbf_mmd(left: torch.Tensor, right: torch.Tensor, *, max_points: int = 512) -> float:
    x = _cap_rows(left.detach().cpu().float(), max_points)
    y = _cap_rows(right.detach().cpu().float(), max_points)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return math.nan
    pooled = torch.cat([x, y], dim=0)
    distances = torch.pdist(pooled).pow(2)
    bandwidth = torch.median(distances[distances > 0]).clamp_min(1.0e-6) if torch.any(distances > 0) else torch.tensor(1.0)
    kxx = torch.exp(-torch.cdist(x, x).pow(2) / bandwidth).mean()
    kyy = torch.exp(-torch.cdist(y, y).pow(2) / bandwidth).mean()
    kxy = torch.exp(-torch.cdist(x, y).pow(2) / bandwidth).mean()
    return float((kxx + kyy - (2.0 * kxy)).item())


def _sample_from_tensors(
    weights: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    *,
    n_samples: int,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    component_ids = torch.multinomial(weights.float(), int(n_samples), replacement=True, generator=gen)
    eps = torch.randn((int(n_samples), means.shape[1]), generator=gen)
    return means.float()[component_ids] + eps * torch.sqrt(variances.float()[component_ids])


def _trace_cov(x: torch.Tensor) -> float:
    if int(x.shape[0]) < 2:
        return 0.0
    return float(x.var(dim=0, unbiased=True).sum().item())


def _pairwise_distance_mean(x: torch.Tensor, *, max_points: int = 512) -> float:
    capped = _cap_rows(x, max_points)
    if int(capped.shape[0]) < 2:
        return 0.0
    return float(torch.pdist(capped).mean().item())


def _cap_rows(x: torch.Tensor, max_points: int) -> torch.Tensor:
    if int(x.shape[0]) <= int(max_points):
        return x
    return x[: int(max_points)]


def _class_balance(labels: Sequence[int]) -> float:
    counts = [sum(1 for value in labels if int(value) == cls) for cls in (0, 1)]
    total = sum(counts)
    if total <= 0:
        return math.nan
    return float(min(counts) / max(counts)) if max(counts) else 0.0


def _generator_for_device(device: torch.device, seed: int) -> torch.Generator:
    if device.type == "cuda":
        return torch.Generator(device=device).manual_seed(int(seed))
    return torch.Generator(device="cpu").manual_seed(int(seed))
