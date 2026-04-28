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
) -> Dict[str, float]:
    if not support_idxs:
        return {
            "response_posterior_mu_norm": 0.0,
            "response_posterior_mu_mean": 0.0,
            "response_posterior_mu_std": 0.0,
            "response_posterior_logvar_mean": 0.0,
            "response_posterior_logvar_std": 0.0,
            "response_posterior_entropy_proxy": 0.0,
            "response_decode_repeat_var_mean": 0.0,
            "response_decode_repeat_var_std": 0.0,
            "response_recon_repeat_var_mean": 0.0,
            "response_recon_repeat_var_std": 0.0,
            "response_kl_repeat_var_mean": 0.0,
            "response_kl_repeat_var_std": 0.0,
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
        logvar_mean = float(logvar.mean().item()) if logvar.numel() else 0.0
        logvar_std = float(logvar.std(dim=0, unbiased=False).mean().item()) if logvar.numel() else 0.0
        entropy_const = math.log(2.0 * math.pi * math.e)
        entropy_proxy = 0.5 * (logvar + entropy_const).sum(dim=1)
        entropy_proxy_mean = float(entropy_proxy.mean().item()) if entropy_proxy.numel() else 0.0

        rep_count = max(1, int(n_repeats))
        recon_repeats = []
        recon_err_repeats = []
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

        if rep_count > 1:
            recon_stack = torch.stack(recon_repeats, dim=0)
            recon_var = recon_stack.var(dim=0, unbiased=False)
            recon_var_mean_per_sample = recon_var.mean(dim=1)
            decode_var_mean = float(recon_var_mean_per_sample.mean().item())
            decode_var_std = float(recon_var_mean_per_sample.std(unbiased=False).item())

            recon_err_stack = torch.stack(recon_err_repeats, dim=0)
            recon_err_var = recon_err_stack.var(dim=0, unbiased=False)
            recon_err_var_mean = float(recon_err_var.mean().item())
            recon_err_var_std = float(recon_err_var.std(unbiased=False).item())

            kl_var_mean = 0.0
            kl_var_std = 0.0
        else:
            decode_var_mean = 0.0
            decode_var_std = 0.0
            recon_err_var_mean = 0.0
            recon_err_var_std = 0.0
            kl_var_mean = 0.0
            kl_var_std = 0.0

    return {
        "response_posterior_mu_norm": mu_norm_mean,
        "response_posterior_mu_mean": mu_mean,
        "response_posterior_mu_std": mu_std,
        "response_posterior_logvar_mean": logvar_mean,
        "response_posterior_logvar_std": logvar_std,
        "response_posterior_entropy_proxy": entropy_proxy_mean,
        "response_decode_repeat_var_mean": decode_var_mean,
        "response_decode_repeat_var_std": decode_var_std,
        "response_recon_repeat_var_mean": recon_err_var_mean,
        "response_recon_repeat_var_std": recon_err_var_std,
        "response_kl_repeat_var_mean": kl_var_mean,
        "response_kl_repeat_var_std": kl_var_std,
    }
