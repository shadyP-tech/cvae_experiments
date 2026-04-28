from __future__ import annotations

from pathlib import Path
import math
import random
from functools import cmp_to_key
from typing import Any, Dict, List, Optional, Sequence

import torch

from src.data.metadata_conditioning import build_domain_one_hot, resolve_domain_order
from src.eval.metrics import selection_accuracy
from src.models.cvae_expert import CVAEExpert, elbo_components
from src.torch_utils import safe_torch_load
from src.train.checkpoint_provenance import load_model_checkpoint
from src.routing.router import (
    confusion_update,
    equal_weight_scoring_weights,
    route_hard,
    route_soft,
)


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
    model.load_state_dict(load_model_checkpoint(checkpoint, map_location=device).model_state_dict)
    model.eval()
    return model


def _score_model(model: CVAEExpert, x: torch.Tensor, m: torch.Tensor | None = None) -> torch.Tensor:
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


def _reconstruction_only(model: CVAEExpert, x: torch.Tensor, m: torch.Tensor | None = None) -> torch.Tensor:
    recon, mu, logvar = model(x, m=m)
    rec, _ = elbo_components(recon, x, mu, logvar)
    return rec


def _summary_with_hist(values: List[float], bins: int = 10) -> Dict[str, object]:
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "hist": {"bin_edges": [], "counts": []},
        }
    t = torch.tensor(values, dtype=torch.float32)
    vmin = float(t.min().item())
    vmax = float(t.max().item())
    if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=1e-12):
        counts = [len(values)]
        edges = [vmin, vmax]
    else:
        hist = torch.histc(t, bins=bins, min=vmin, max=vmax)
        counts = [int(x) for x in hist.tolist()]
        step = (vmax - vmin) / float(bins)
        edges = [float(vmin + i * step) for i in range(bins + 1)]

    return {
        "n": int(t.numel()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "min": vmin,
        "max": vmax,
        "p10": float(torch.quantile(t, 0.10).item()),
        "p50": float(torch.quantile(t, 0.50).item()),
        "p90": float(torch.quantile(t, 0.90).item()),
        "hist": {
            "bin_edges": edges,
            "counts": counts,
        },
    }


def _rank_desc_with_tie_break(scores: List[float], tie_tol: float) -> List[int]:
    def _cmp(i: int, j: int) -> int:
        diff = float(scores[i] - scores[j])
        if abs(diff) <= tie_tol:
            if i < j:
                return -1
            if i > j:
                return 1
            return 0
        return -1 if diff > 0 else 1

    return sorted(range(len(scores)), key=cmp_to_key(_cmp))


def _spearman_from_scores_desc(x_scores: List[float], y_scores: List[float], tie_tol: float) -> float:
    if len(x_scores) != len(y_scores) or len(x_scores) < 2:
        return 0.0
    x_order = _rank_desc_with_tie_break(x_scores, tie_tol=tie_tol)
    y_order = _rank_desc_with_tie_break(y_scores, tie_tol=tie_tol)

    rx = [0.0] * len(x_scores)
    ry = [0.0] * len(y_scores)
    for pos, idx in enumerate(x_order, start=1):
        rx[idx] = float(pos)
    for pos, idx in enumerate(y_order, start=1):
        ry[idx] = float(pos)

    tx = torch.tensor(rx, dtype=torch.float32)
    ty = torch.tensor(ry, dtype=torch.float32)
    vx = tx - tx.mean()
    vy = ty - ty.mean()
    denom = torch.sqrt(torch.sum(vx * vx) * torch.sum(vy * vy)).item()
    if denom <= 1e-12:
        return 0.0
    return float(torch.sum(vx * vy).item() / denom)


def _has_utility_tie(nelbo_scores: List[float], tie_tol: float) -> bool:
    if not nelbo_scores:
        return False
    best = min(nelbo_scores)
    n_best = sum(1 for v in nelbo_scores if abs(v - best) <= tie_tol)
    return n_best >= 2


def evaluate_routing(
    test_cache: Path,
    expert_checkpoints: Dict[str, str],
    global_checkpoint: Path,
    hidden_dim: int,
    latent_dim: int,
    strategy: str,
    tau: float,
    temperature: float,
    seed: int,
    similarity_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    conditioning_cfg: Dict[str, Any] | None = None,
    configured_domains: Sequence[int] | None = None,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
) -> Dict[str, object]:
    rng = random.Random(seed)
    payload = safe_torch_load(test_cache, map_location="cpu")
    x_cpu = payload["embeddings"]
    meta = payload["metadata"]

    input_dim = int(x_cpu.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    x = x_cpu.to(device)

    cond_cfg = conditioning_cfg or {}
    conditioning_enabled = bool(cond_cfg.get("enabled", False))
    metadata_dim = 0
    metadata_vectors_cpu = None
    if conditioning_enabled:
        domain_order = resolve_domain_order(configured_domains or [])
        metadata_vectors_cpu = build_domain_one_hot(meta, domain_order)
        metadata_dim = int(len(domain_order))
    m = metadata_vectors_cpu.to(device) if metadata_vectors_cpu is not None else None
    constraint_cfg = metadata_constraint_cfg or {}

    expert_names = sorted(expert_checkpoints.keys())
    expert_mags = [
        int(name.replace("expert_", "").replace("x", "")) if "expert_" in name else int(name.replace("x", ""))
        for name in expert_names
    ]

    expert_models = [
        _load_model(
            Path(expert_checkpoints[name]),
            input_dim,
            hidden_dim,
            latent_dim,
            device,
            metadata_dim=metadata_dim,
            metadata_constraint_cfg=constraint_cfg,
            aux_metadata_dim=metadata_dim,
        )
        for name in expert_names
    ]
    global_model = _load_model(
        global_checkpoint,
        input_dim,
        hidden_dim,
        latent_dim,
        device,
        metadata_dim=metadata_dim,
        metadata_constraint_cfg=constraint_cfg,
        aux_metadata_dim=metadata_dim,
    )

    with torch.no_grad():
        all_scores = []
        all_recon = []
        for model in expert_models:
            all_scores.append(_score_model(model, x, m=m))
            all_recon.append(_reconstruction_only(model, x, m=m))
        # [num_experts, num_samples]
        expert_scores = torch.stack(all_scores, dim=0)
        expert_recon = torch.stack(all_recon, dim=0)
        global_scores = _score_model(global_model, x, m=m)
        global_recon = _reconstruction_only(global_model, x, m=m)

    hard_scores = []
    soft_scores = []
    random_scores = []
    uniform_sampling_scores = []
    equal_scores = []
    oracle_scores = []
    hard_recon = []
    soft_recon = []
    random_recon = []
    uniform_sampling_recon = []
    equal_recon = []
    oracle_recon = []
    global_baseline_scores = global_scores.tolist()
    global_baseline_recon = global_recon.tolist()

    tie_tolerance = 1e-8
    gap_normalization_eps = 1e-3
    top1_oracle_hit_true_utility: List[float] = []
    spearman_model_vs_true_utility: List[float] = []
    utility_tie_flags: List[float] = []
    expert_nelbo_std_per_query: List[float] = []
    best_expert_true_utility_nelbo: List[float] = []
    routed_to_global_gap_per_query: List[float] = []
    routed_to_true_oracle_gap_per_query: List[float] = []
    routed_to_global_gap_norm_per_query: List[float] = []
    normalized_gap_skipped_count = 0
    per_query_diagnostics: List[Dict[str, object]] = []

    true_domains: List[str] = []
    routed_domains: List[str] = []
    confusion: Dict[str, Dict[str, int]] = {}

    experts_meta = [{"magnification": m} for m in expert_mags]

    fixed_random_idx = rng.randrange(len(expert_models))

    for i, sample_meta in enumerate(meta):
        query_meta = {"magnification": int(sample_meta["magnification"])}
        true_domain = f"{query_meta['magnification']}x"
        true_domains.append(true_domain)

        hard_idx, sims = route_hard(
            query_meta,
            experts_meta,
            strategy=strategy,
            tau=tau,
            similarity_matrix=similarity_matrix,
        )
        model_ranked_experts = _rank_desc_with_tie_break([float(s) for s in sims], tie_tol=tie_tolerance)
        routed_domain = f"{expert_mags[hard_idx]}x"
        routed_domains.append(routed_domain)
        confusion_update(confusion, true_domain=true_domain, pred_domain=routed_domain)
        hard_scores.append(float(expert_scores[hard_idx, i].item()))
        hard_recon.append(float(expert_recon[hard_idx, i].item()))

        utility_scores_desc = [float(-expert_scores[j, i].item()) for j in range(len(expert_models))]
        true_utility_ranked_experts = _rank_desc_with_tie_break(utility_scores_desc, tie_tol=tie_tolerance)
        true_best_idx = int(true_utility_ranked_experts[0])
        true_best_nelbo = float(expert_scores[true_best_idx, i].item())
        best_expert_true_utility_nelbo.append(true_best_nelbo)

        top1_oracle_hit_true_utility.append(1.0 if model_ranked_experts[0] == true_best_idx else 0.0)
        spearman_model_vs_true_utility.append(
            _spearman_from_scores_desc(
                [float(s) for s in sims],
                utility_scores_desc,
                tie_tol=tie_tolerance,
            )
        )

        nelbo_row = [float(expert_scores[j, i].item()) for j in range(len(expert_models))]
        utility_tie_flags.append(1.0 if _has_utility_tie(nelbo_row, tie_tol=tie_tolerance) else 0.0)
        expert_nelbo_std_per_query.append(float(torch.tensor(nelbo_row, dtype=torch.float32).std(unbiased=False).item()))

        soft_w = route_soft(
            query_meta,
            experts_meta,
            strategy=strategy,
            tau=tau,
            temperature=temperature,
            similarity_matrix=similarity_matrix,
        )
        soft_scores.append(float(sum(soft_w[j] * expert_scores[j, i].item() for j in range(len(expert_models)))))
        soft_recon.append(float(sum(soft_w[j] * expert_recon[j, i].item() for j in range(len(expert_models)))))

        random_scores.append(float(expert_scores[fixed_random_idx, i].item()))
        random_recon.append(float(expert_recon[fixed_random_idx, i].item()))

        sampled_idx = rng.randrange(len(expert_models))
        uniform_sampling_scores.append(float(expert_scores[sampled_idx, i].item()))
        uniform_sampling_recon.append(float(expert_recon[sampled_idx, i].item()))

        eq_w = equal_weight_scoring_weights(len(expert_models))
        equal_scores.append(float(sum(eq_w[j] * expert_scores[j, i].item() for j in range(len(expert_models)))))
        equal_recon.append(float(sum(eq_w[j] * expert_recon[j, i].item() for j in range(len(expert_models)))))

        oracle_idx = expert_mags.index(query_meta["magnification"]) if query_meta["magnification"] in expert_mags else hard_idx
        oracle_scores.append(float(expert_scores[oracle_idx, i].item()))
        oracle_recon.append(float(expert_recon[oracle_idx, i].item()))

        routed_to_global_gap = float(hard_scores[-1] - global_baseline_scores[i])
        routed_to_true_oracle_gap = float(hard_scores[-1] - true_best_nelbo)
        routed_to_global_gap_per_query.append(routed_to_global_gap)
        routed_to_true_oracle_gap_per_query.append(routed_to_true_oracle_gap)
        if abs(global_baseline_scores[i]) >= gap_normalization_eps:
            routed_to_global_gap_norm_per_query.append(float(routed_to_global_gap / abs(global_baseline_scores[i])))
        else:
            normalized_gap_skipped_count += 1

        per_query_diagnostics.append(
            {
                "sample_index": int(i),
                "query_domain": true_domain,
                "routed_expert": f"{expert_mags[hard_idx]}x",
                "true_utility_best_expert": f"{expert_mags[true_best_idx]}x",
                "domain_oracle_expert": f"{expert_mags[oracle_idx]}x",
                "hard_routed_nelbo": float(hard_scores[-1]),
                "true_utility_best_nelbo": true_best_nelbo,
                "global_nelbo": float(global_baseline_scores[i]),
                "routed_to_global_gap": routed_to_global_gap,
                "routed_to_true_oracle_gap": routed_to_true_oracle_gap,
                "top1_hit_true_utility": int(model_ranked_experts[0] == true_best_idx),
                "utility_tie": int(utility_tie_flags[-1] > 0.0),
                "metadata_choice_rank_in_true_utility": int(true_utility_ranked_experts.index(hard_idx) + 1),
            }
        )

    routed_global_gap_tensor = torch.tensor(routed_to_global_gap_per_query, dtype=torch.float32)
    routed_true_oracle_gap_tensor = torch.tensor(routed_to_true_oracle_gap_per_query, dtype=torch.float32)
    routed_global_norm_gap_abs_median = (
        float(torch.tensor([abs(v) for v in routed_to_global_gap_norm_per_query], dtype=torch.float32).median().item())
        if routed_to_global_gap_norm_per_query
        else 0.0
    )

    results = {
        "metrics": {
            "hard_metadata_routing_nelbo": float(torch.tensor(hard_scores).mean().item()),
            "soft_metadata_routing_nelbo": float(torch.tensor(soft_scores).mean().item()),
            "random_expert_nelbo": float(torch.tensor(random_scores).mean().item()),
            "uniform_sampling_nelbo": float(torch.tensor(uniform_sampling_scores).mean().item()),
            "equal_weight_scoring_nelbo": float(torch.tensor(equal_scores).mean().item()),
            "global_cvae_nelbo": float(torch.tensor(global_baseline_scores).mean().item()),
            "oracle_expert_nelbo": float(torch.tensor(oracle_scores).mean().item()),
            "hard_metadata_routing_recon": float(torch.tensor(hard_recon).mean().item()),
            "soft_metadata_routing_recon": float(torch.tensor(soft_recon).mean().item()),
            "random_expert_recon": float(torch.tensor(random_recon).mean().item()),
            "uniform_sampling_recon": float(torch.tensor(uniform_sampling_recon).mean().item()),
            "equal_weight_scoring_recon": float(torch.tensor(equal_recon).mean().item()),
            "global_cvae_recon": float(torch.tensor(global_baseline_recon).mean().item()),
            "oracle_expert_recon": float(torch.tensor(oracle_recon).mean().item()),
            "routing_selection_accuracy": selection_accuracy(true_domains, routed_domains),
            "best_expert_true_utility_nelbo": float(torch.tensor(best_expert_true_utility_nelbo).mean().item()),
            "top1_oracle_hit_true_utility": float(torch.tensor(top1_oracle_hit_true_utility).mean().item()),
            "spearman_with_true_utility": float(torch.tensor(spearman_model_vs_true_utility).mean().item()),
            "utility_tie_rate": float(torch.tensor(utility_tie_flags).mean().item()),
            "n_experts": int(len(expert_models)),
            "routed_to_global_gap": float(routed_global_gap_tensor.mean().item()),
            "routed_to_true_oracle_gap": float(routed_true_oracle_gap_tensor.mean().item()),
            "routed_to_global_gap_abs_median": float(torch.median(torch.abs(routed_global_gap_tensor)).item()),
            "routed_to_global_gap_norm_abs_median": routed_global_norm_gap_abs_median,
            "routed_to_global_gap_norm_skipped_count": int(normalized_gap_skipped_count),
            "routed_to_global_gap_norm_eps": float(gap_normalization_eps),
        },
        "routing": {
            "confusion_matrix": confusion,
            "true_domains": true_domains,
            "routed_domains": routed_domains,
            "random_expert_choice": f"{expert_mags[fixed_random_idx]}x",
        },
        "diagnostics": {
            "tie_tolerance": float(tie_tolerance),
            "expert_nelbo_std_per_query": _summary_with_hist(expert_nelbo_std_per_query),
            "global_nelbo": _summary_with_hist(global_baseline_scores),
            "best_expert_true_utility_nelbo": _summary_with_hist(best_expert_true_utility_nelbo),
            "routed_expert_nelbo": _summary_with_hist(hard_scores),
            "per_query": per_query_diagnostics,
        },
    }
    return results
