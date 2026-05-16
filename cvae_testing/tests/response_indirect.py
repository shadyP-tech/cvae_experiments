from __future__ import annotations

import math
from typing import Dict, Sequence

import torch

from src.eval.evaluators.hybrid import HybridExpertBank


def compute_response_features(
    *,
    bank: HybridExpertBank,
    expert_domain: int,
    x_cpu: torch.Tensor,
    support_idxs: Sequence[int],
    device: torch.device,
    n_repeats: int,
    repeat_seed_base: int,
    include_residual_shape_features: bool = False,
    variance_epsilon: float = 1e-12,
) -> Dict[str, float]:
    if not support_idxs:
        return {
            "response_posterior_mu_norm": 0.0,
            "response_posterior_mu_mean": 0.0,
            "response_posterior_mu_std": 0.0,
            "response_posterior_mu_q75": 0.0,
            "response_posterior_mu_max": 0.0,
            "response_posterior_logvar_mean": 0.0,
            "response_posterior_logvar_std": 0.0,
            "response_posterior_logvar_q75": 0.0,
            "response_posterior_logvar_max": 0.0,
            "response_posterior_var_mean": 0.0,
            "response_posterior_var_std": 0.0,
            "response_posterior_var_q75": 0.0,
            "response_posterior_var_max": 0.0,
            "response_posterior_entropy_proxy": 0.0,
            "response_decode_repeat_var_mean": 0.0,
            "response_decode_repeat_var_std": 0.0,
            "response_decode_repeat_variance_mean": 0.0,
            "response_decode_repeat_variance_q75": 0.0,
            "response_decode_repeat_variance_max": 0.0,
            "response_recon_repeat_var_mean": 0.0,
            "response_recon_repeat_var_std": 0.0,
            "response_recon_repeat_variance_mean": 0.0,
            "response_recon_repeat_variance_q75": 0.0,
        }

    support_x = x_cpu[list(support_idxs)].to(device)
    cvae = bank.domain_cvae(int(expert_domain))
    with torch.no_grad():
        proj = bank.project(int(expert_domain), support_x)
        mu, logvar = cvae.encode(proj)

        mu_norm = torch.norm(mu, dim=1)
        mu_norm_mean = float(mu_norm.mean().item()) if mu_norm.numel() else 0.0
        mu_mean = float(mu.mean().item()) if mu.numel() else 0.0
        mu_std = float(mu.std(dim=0, unbiased=False).mean().item()) if mu.numel() else 0.0
        mu_abs = mu.abs().reshape(-1)
        mu_q75 = float(torch.quantile(mu_abs, 0.75).item()) if mu_abs.numel() else 0.0
        mu_max = float(mu_abs.max().item()) if mu_abs.numel() else 0.0
        logvar_mean = float(logvar.mean().item()) if logvar.numel() else 0.0
        logvar_std = float(logvar.std(dim=0, unbiased=False).mean().item()) if logvar.numel() else 0.0
        logvar_flat = logvar.reshape(-1)
        logvar_q75 = float(torch.quantile(logvar_flat, 0.75).item()) if logvar_flat.numel() else 0.0
        logvar_max = float(logvar_flat.max().item()) if logvar_flat.numel() else 0.0
        posterior_var = torch.exp(logvar)
        posterior_var_flat = posterior_var.reshape(-1)
        posterior_var_mean = float(posterior_var_flat.mean().item()) if posterior_var_flat.numel() else 0.0
        posterior_var_std = float(posterior_var_flat.std(unbiased=False).item()) if posterior_var_flat.numel() else 0.0
        posterior_var_q75 = float(torch.quantile(posterior_var_flat, 0.75).item()) if posterior_var_flat.numel() else 0.0
        posterior_var_max = float(posterior_var_flat.max().item()) if posterior_var_flat.numel() else 0.0
        entropy_const = math.log(2.0 * math.pi * math.e)
        entropy_proxy = 0.5 * (logvar + entropy_const).sum(dim=1)
        entropy_proxy_mean = float(entropy_proxy.mean().item()) if entropy_proxy.numel() else 0.0

        rep_count = max(1, int(n_repeats))
        recon_repeats = []
        recon_err_repeats = []
        kl_repeats = []
        residual_abs_repeats = []
        for rep in range(rep_count):
            rep_seed = int(repeat_seed_base) + int(rep) * 10007
            torch.manual_seed(rep_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rep_seed)
            z = cvae.reparameterize(mu, logvar)
            recon = cvae.decode(z)
            recon_repeats.append(recon)
            recon_err = (recon - proj).pow(2).sum(dim=1)
            recon_err_repeats.append(recon_err)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_repeats.append(kl)
            if bool(include_residual_shape_features):
                residual_abs_repeats.append((recon - proj).abs())

        if rep_count > 1:
            recon_stack = torch.stack(recon_repeats, dim=0)
            recon_var = recon_stack.var(dim=0, unbiased=False)
            recon_var_mean_per_sample = recon_var.mean(dim=1)
            decode_var_mean = float(recon_var_mean_per_sample.mean().item())
            decode_var_std = float(recon_var_mean_per_sample.std(unbiased=False).item())
            decode_var_q75 = float(torch.quantile(recon_var_mean_per_sample, 0.75).item())
            decode_var_max = float(recon_var_mean_per_sample.max().item())

            recon_err_stack = torch.stack(recon_err_repeats, dim=0)
            recon_err_var = recon_err_stack.var(dim=0, unbiased=False)
            recon_err_var_mean = float(recon_err_var.mean().item())
            recon_err_var_std = float(recon_err_var.std(unbiased=False).item())
            recon_err_var_q75 = float(torch.quantile(recon_err_var, 0.75).item())

            kl_stack = torch.stack(kl_repeats, dim=0)
            kl_var = kl_stack.var(dim=0, unbiased=False)
            include_kl_var = bool(float(kl_var.max().item()) > float(variance_epsilon)) if kl_var.numel() else False

            residual_shape: Dict[str, float] = {}
            if bool(include_residual_shape_features) and residual_abs_repeats:
                residual_abs = torch.stack(residual_abs_repeats, dim=0).reshape(-1)
                residual_shape = {
                    "response_residual_abs_mean": float(residual_abs.mean().item()) if residual_abs.numel() else 0.0,
                    "response_residual_abs_std": float(residual_abs.std(unbiased=False).item()) if residual_abs.numel() else 0.0,
                    "response_residual_abs_q75": float(torch.quantile(residual_abs, 0.75).item()) if residual_abs.numel() else 0.0,
                    "response_residual_abs_max": float(residual_abs.max().item()) if residual_abs.numel() else 0.0,
                }
        else:
            decode_var_mean = 0.0
            decode_var_std = 0.0
            decode_var_q75 = 0.0
            decode_var_max = 0.0
            recon_err_var_mean = 0.0
            recon_err_var_std = 0.0
            recon_err_var_q75 = 0.0
            kl_var = torch.empty((0,), device=device)
            include_kl_var = False
            residual_shape = {}

    features = {
        "response_posterior_mu_norm": mu_norm_mean,
        "response_posterior_mu_mean": mu_mean,
        "response_posterior_mu_std": mu_std,
        "response_posterior_mu_q75": mu_q75,
        "response_posterior_mu_max": mu_max,
        "response_posterior_logvar_mean": logvar_mean,
        "response_posterior_logvar_std": logvar_std,
        "response_posterior_logvar_q75": logvar_q75,
        "response_posterior_logvar_max": logvar_max,
        "response_posterior_var_mean": posterior_var_mean,
        "response_posterior_var_std": posterior_var_std,
        "response_posterior_var_q75": posterior_var_q75,
        "response_posterior_var_max": posterior_var_max,
        "response_posterior_entropy_proxy": entropy_proxy_mean,
        "response_decode_repeat_var_mean": decode_var_mean,
        "response_decode_repeat_var_std": decode_var_std,
        "response_decode_repeat_variance_mean": decode_var_mean,
        "response_decode_repeat_variance_q75": decode_var_q75,
        "response_decode_repeat_variance_max": decode_var_max,
        "response_recon_repeat_var_mean": recon_err_var_mean,
        "response_recon_repeat_var_std": recon_err_var_std,
        "response_recon_repeat_variance_mean": recon_err_var_mean,
        "response_recon_repeat_variance_q75": recon_err_var_q75,
    }
    if include_kl_var:
        features.update(
            {
                "response_kl_repeat_variance_mean": float(kl_var.mean().item()),
                "response_kl_repeat_variance_q75": float(torch.quantile(kl_var, 0.75).item()),
                "response_kl_repeat_variance_max": float(kl_var.max().item()),
            }
        )
    features.update(residual_shape)
    return features
