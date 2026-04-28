from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

# Allow running as either:
# - python -m scripts.run_learned_compatibility_loqdo
# - python scripts/run_learned_compatibility_loqdo.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluators.hybrid import HybridExpertBank
from src.eval.evaluators.response_indirect import compute_response_features
from src.eval.metrics import spearman_corr
from src.routing.strategies import compute_similarity
from src.app.determinism import RESPONSE_SEED_SCHEME_VERSION, stable_response_seed
from src.torch_utils import safe_torch_load


@dataclass
class RunContext:
    run_dir: Path
    dataset_name: str
    seed: int
    backbone_type: str
    routing_strategy: str
    routing_tau: float
    variant: str
    test_cache: Path
    variant_checkpoint: Path


@dataclass
class ProbeConfig:
    enabled: bool
    hidden_dim: int
    dropout: float
    learning_rate: float
    epochs: int
    alpha: float
    beta: float
    support_fraction: float
    query_batch_size: int
    support_split_seed: int
    clip_target_percentile: float


@dataclass
class OracleProbeConfig:
    enabled: bool
    semi_oracle_risk_lambda: float


def _deterministic_query_split(
    idxs: Sequence[int],
    *,
    query_batch_size: int,
    support_fraction: float,
    split_seed: int,
) -> Tuple[List[int], List[int]]:
    if not idxs:
        return [], []
    rng = np.random.default_rng(int(split_seed))
    idxs_arr = np.asarray(list(idxs), dtype=np.int64)
    if int(query_batch_size) > 0 and int(idxs_arr.size) > int(query_batch_size):
        idxs_arr = rng.choice(idxs_arr, size=int(query_batch_size), replace=False)

    perm = rng.permutation(idxs_arr)
    n_total = int(perm.size)
    if n_total < 2:
        return perm.tolist(), []

    n_support = int(round(float(n_total) * float(support_fraction)))
    n_support = max(1, min(n_support, n_total - 1))
    support = perm[:n_support].tolist()
    evaluate = perm[n_support:].tolist()
    return support, evaluate


def _summary_stats(values: np.ndarray) -> Tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return float(values.mean()), float(values.std()), float(np.percentile(values, 90))


def _fit_standardizer(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


RESPONSE_FEATURE_KEYS = [
    "response_posterior_mu_norm",
    "response_posterior_mu_mean",
    "response_posterior_mu_std",
    "response_posterior_logvar_mean",
    "response_posterior_logvar_std",
    "response_posterior_entropy_proxy",
    "response_decode_repeat_var_mean",
    "response_decode_repeat_var_std",
    "response_recon_repeat_var_mean",
    "response_recon_repeat_var_std",
    "response_kl_repeat_var_mean",
    "response_kl_repeat_var_std",
]


def _fit_response_feature_standardizer(rows: Sequence[dict]) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for key in RESPONSE_FEATURE_KEYS:
        values = np.asarray([float(r.get(key, 0.0)) for r in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            stats[key] = (0.0, 1.0)
            continue
        mean = float(values.mean())
        std = float(values.std())
        if std < 1e-8:
            std = 1.0
        stats[key] = (mean, std)
    return stats


def _apply_response_feature_standardizer(rows: Sequence[dict], stats: Dict[str, Tuple[float, float]]) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        updated = dict(row)
        for key, (mean, std) in stats.items():
            value = float(updated.get(key, 0.0))
            if not math.isfinite(value):
                value = 0.0
            updated[key] = (value - float(mean)) / float(std)
        out.append(updated)
    return out


def _response_feature_norm_report_rows(
    *,
    ctx: RunContext,
    heldout_domain: int,
    split_id: str,
    stats: Dict[str, Tuple[float, float]],
    n_train_rows: int,
) -> List[dict]:
    rows: List[dict] = []
    for key, (mean, std) in stats.items():
        rows.append(
            {
                "dataset_name": ctx.dataset_name,
                "seed": int(ctx.seed),
                "backbone_type": ctx.backbone_type,
                "run_id": ctx.run_dir.name,
                "variant": ctx.variant,
                "heldout_query_domain": int(heldout_domain),
                "support_eval_split_id": str(split_id),
                "feature_key": str(key),
                "mean": float(mean),
                "std": float(std),
                "n_train_rows": int(n_train_rows),
                "normalization_policy": "train_fold_standardize",
            }
        )
    return rows


def _calibration_metrics(y_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> Tuple[float, float, float]:
    if y_pred.size == 0 or y_true.size == 0:
        return 0.0, 0.0, 0.0
    if y_pred.size == 1 or np.std(y_pred) < 1e-12:
        slope = 0.0
        intercept = float(np.mean(y_true))
    else:
        x_mean = float(np.mean(y_pred))
        y_mean = float(np.mean(y_true))
        cov = float(np.mean((y_pred - x_mean) * (y_true - y_mean)))
        var = float(np.var(y_pred))
        slope = cov / var if var > 1e-12 else 0.0
        intercept = y_mean - slope * x_mean

    order = np.argsort(y_pred)
    sorted_pred = y_pred[order]
    sorted_true = y_true[order]
    k = int(max(1, min(n_bins, y_pred.size)))
    # Equal-frequency bins for low-sample folds.
    bin_edges = np.linspace(0, y_pred.size, num=k + 1, dtype=int)
    gaps: List[float] = []
    for i in range(k):
        s = int(bin_edges[i])
        e = int(bin_edges[i + 1])
        if e <= s:
            continue
        gaps.append(abs(float(sorted_pred[s:e].mean()) - float(sorted_true[s:e].mean())))
    cal_err = float(np.mean(gaps)) if gaps else 0.0
    return float(slope), float(intercept), float(cal_err)


def _top1_margin(scores: np.ndarray) -> float:
    if scores.size < 2:
        return 0.0
    order = np.sort(scores)
    return float(order[-1] - order[-2])


def _normalized_utility_gap(selected_nelbo: float, oracle_nelbo: float, worst_nelbo: float, eps: float = 1e-12) -> float:
    denom = float(max(worst_nelbo - oracle_nelbo, float(eps)))
    return float(max(selected_nelbo - oracle_nelbo, 0.0) / denom)


def _oracle_pairwise_rank_scores(test_rows: Sequence[dict]) -> np.ndarray:
    utilities = np.asarray([float(r["oracle_utility"]) for r in test_rows], dtype=np.float64)
    experts = np.asarray([int(r["expert_domain"]) for r in test_rows], dtype=np.int64)
    n = int(utilities.size)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.ones((1,), dtype=np.float64)

    scores = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        wins = 0.0
        for j in range(n):
            if i == j:
                continue
            if utilities[i] > utilities[j] + 1e-12:
                wins += 1.0
            elif utilities[i] < utilities[j] - 1e-12:
                wins += 0.0
            else:
                wins += 0.5
        scores[i] = wins / float(max(n - 1, 1))

    # Deterministic tie-break: smaller expert_domain gets slightly higher score.
    max_domain = float(np.max(experts)) if experts.size else 0.0
    tie_break = (max_domain - experts.astype(np.float64)) * 1e-9
    return scores + tie_break


def _pairwise_inconsistency_rate(values: np.ndarray) -> float:
    n = int(values.size)
    if n < 3:
        return 0.0

    def _cmp(a: float, b: float) -> int:
        if a > b + 1e-12:
            return 1
        if a < b - 1e-12:
            return -1
        return 0

    cycles = 0
    total = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                ij = _cmp(float(values[i]), float(values[j]))
                jk = _cmp(float(values[j]), float(values[k]))
                ki = _cmp(float(values[k]), float(values[i]))
                if ij == 0 or jk == 0 or ki == 0:
                    continue
                total += 1
                if (ij == 1 and jk == 1 and ki == 1) or (ij == -1 and jk == -1 and ki == -1):
                    cycles += 1
    return float(cycles / total) if total > 0 else 0.0


def _pairwise_logistic_loss(scores: torch.Tensor, targets: torch.Tensor, query_ids: np.ndarray) -> torch.Tensor:
    q_arr = np.asarray(query_ids, dtype=np.int64)
    unique_q = sorted(set(int(v) for v in q_arr.tolist()))
    losses: List[torch.Tensor] = []
    for q in unique_q:
        idxs = np.where(q_arr == int(q))[0]
        if idxs.size < 2:
            continue
        for i_pos in range(int(idxs.size)):
            for j_pos in range(i_pos + 1, int(idxs.size)):
                i = int(idxs[i_pos])
                j = int(idxs[j_pos])
                y_i = float(targets[i].item())
                y_j = float(targets[j].item())
                if abs(y_i - y_j) < 1e-12:
                    continue
                sign = 1.0 if y_i > y_j else -1.0
                diff = scores[i] - scores[j]
                losses.append(torch.nn.functional.softplus(torch.tensor(-sign, device=diff.device) * diff))
    if not losses:
        return torch.tensor(0.0, dtype=scores.dtype, device=scores.device)
    return torch.stack(losses).mean()


def _train_utility_probe(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    query_ids: np.ndarray,
    x_test: np.ndarray,
    cfg: ProbeConfig,
) -> np.ndarray:
    if x_train.size == 0 or y_train.size == 0:
        return np.zeros((x_test.shape[0],), dtype=np.float64)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=dev)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=dev)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=dev)

    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], int(cfg.hidden_dim)),
        torch.nn.ReLU(),
        torch.nn.Dropout(float(cfg.dropout)),
        torch.nn.Linear(int(cfg.hidden_dim), 1),
    ).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_loss = math.inf
    patience = 25
    since_improve = 0
    for _ in range(int(cfg.epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(x_train_t).squeeze(-1)
        reg_loss = torch.nn.functional.mse_loss(pred, y_train_t)
        rank_loss = _pairwise_logistic_loss(pred, y_train_t, query_ids=query_ids)
        loss = float(cfg.alpha) * reg_loss + float(cfg.beta) * rank_loss
        loss.backward()
        opt.step()

        loss_val = float(loss.detach().cpu().item())
        if loss_val + 1e-8 < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_test = model(x_test_t).squeeze(-1).cpu().numpy().astype(np.float64, copy=False)
    return pred_test


def _as_int_domain(value: object) -> int:
    return int(str(value).replace("x", ""))


def _load_run_context(run_dir: Path, variant: str) -> RunContext:
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_name = str(cfg.get("experiment", {}).get("dataset_name", "unknown"))
    seed = int(cfg.get("seed", 0))
    backbone_type = str(cfg.get("features", {}).get("backbone_type", "unknown"))
    routing_strategy = str(cfg.get("routing", {}).get("strategy", "categorical_exact"))
    routing_tau = float(cfg.get("routing", {}).get("tau", 1.0))

    variant_name = str(variant).upper()
    test_cache = run_dir / "embeddings" / "test.pt"
    variant_checkpoint = run_dir / "checkpoints" / f"hybrid_variant_{variant_name}.pt"
    if not test_cache.exists():
        raise FileNotFoundError(f"Missing test cache: {test_cache}")
    if not variant_checkpoint.exists():
        raise FileNotFoundError(f"Missing variant checkpoint: {variant_checkpoint}")

    return RunContext(
        run_dir=run_dir,
        dataset_name=dataset_name,
        seed=seed,
        backbone_type=backbone_type,
        routing_strategy=routing_strategy,
        routing_tau=routing_tau,
        variant=variant_name,
        test_cache=test_cache,
        variant_checkpoint=variant_checkpoint,
    )


def _resolve_run_dir(path: Path) -> Path:
    if path.is_dir() and (path / "config_resolved.yaml").exists():
        return path

    latest = path / "latest.txt"
    if path.is_dir() and latest.exists():
        run_id = latest.read_text(encoding="utf-8").strip()
        resolved = path / run_id
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"latest.txt points to missing run: {resolved}")

    raise FileNotFoundError(
        f"Cannot resolve run directory from path: {path}. Expected run dir with config_resolved.yaml "
        "or experiment dir with latest.txt."
    )


def _score_domains_batched(
    bank: HybridExpertBank,
    expert_domains: Sequence[int],
    x_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int,
    n_repeats: int,
    repeat_seed_base: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rep_count = max(1, int(n_repeats))
    score_repeats: List[torch.Tensor] = []
    recon_repeats: List[torch.Tensor] = []

    with torch.no_grad():
        for rep in range(rep_count):
            rep_seed = int(repeat_seed_base) + int(rep) * 10007
            torch.manual_seed(rep_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rep_seed)

            scores: List[torch.Tensor] = []
            recons: List[torch.Tensor] = []
            for ed in expert_domains:
                nelbo_chunks: List[torch.Tensor] = []
                recon_chunks: List[torch.Tensor] = []
                for i in range(0, int(x_cpu.shape[0]), int(batch_size)):
                    xb = x_cpu[i : i + int(batch_size)].to(device)
                    nelbo_chunks.append(bank.score_domain_nelbo(int(ed), xb).cpu())
                    recon_chunks.append(bank.score_domain_recon(int(ed), xb).cpu())
                scores.append(torch.cat(nelbo_chunks, dim=0) if nelbo_chunks else torch.empty((0,), dtype=torch.float32))
                recons.append(torch.cat(recon_chunks, dim=0) if recon_chunks else torch.empty((0,), dtype=torch.float32))
            score_repeats.append(torch.stack(scores, dim=0))
            recon_repeats.append(torch.stack(recons, dim=0))

    score_stack = torch.stack(score_repeats, dim=0)
    recon_stack = torch.stack(recon_repeats, dim=0)
    return (
        score_stack.mean(dim=0),
        score_stack.std(dim=0, unbiased=False),
        recon_stack.mean(dim=0),
        recon_stack.std(dim=0, unbiased=False),
    )


def _domain_mean_embeddings(embeddings: np.ndarray, metadata: Sequence[dict], domains: Sequence[int]) -> Dict[int, np.ndarray]:
    by_domain: Dict[int, List[int]] = {int(d): [] for d in domains}
    for idx, item in enumerate(metadata):
        d = _as_int_domain(item["magnification"])
        if d in by_domain:
            by_domain[d].append(idx)

    means: Dict[int, np.ndarray] = {}
    for d, idxs in by_domain.items():
        if not idxs:
            continue
        means[d] = embeddings[idxs].mean(axis=0)
    return means


def _row_rank_desc(values: Sequence[float], selected_idx: int) -> int:
    order = sorted(range(len(values)), key=lambda i: float(values[i]), reverse=True)
    for r, idx in enumerate(order, start=1):
        if idx == selected_idx:
            return int(r)
    return len(values)


def _build_pair_rows(
    ctx: RunContext,
    *,
    batch_size: int,
    probe_cfg: ProbeConfig,
    uncertainty_repeats: int,
) -> Tuple[List[dict], List[int], Dict[int, Dict[int, float]], Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
    payload = safe_torch_load(ctx.test_cache, map_location="cpu")
    x_cpu: torch.Tensor = payload["embeddings"]
    metadata: List[dict] = payload["metadata"]
    if int(x_cpu.shape[0]) != len(metadata):
        raise RuntimeError("Embeddings/metadata length mismatch in test cache.")

    all_domains = sorted(set(_as_int_domain(m["magnification"]) for m in metadata))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    bank = HybridExpertBank(ctx.variant_checkpoint, device=device)
    expert_domains = [d for d in all_domains if d in bank.domains]
    if not expert_domains:
        raise RuntimeError("No overlapping expert domains found between data and checkpoint.")

    response_target_seed_base = stable_response_seed(
        dataset=ctx.dataset_name,
        seed=int(ctx.seed),
        query_id="all",
        expert_domain="all",
        repeat_id=0,
        stream_name="target_oracle",
    )
    response_feature_seed_base = stable_response_seed(
        dataset=ctx.dataset_name,
        seed=int(ctx.seed),
        query_id="all",
        expert_domain="all",
        repeat_id=0,
        stream_name="response_feature",
    )

    score_mean_tensor, score_std_tensor, recon_mean_tensor, recon_std_tensor = _score_domains_batched(
        bank=bank,
        expert_domains=expert_domains,
        x_cpu=x_cpu,
        device=device,
        batch_size=batch_size,
        n_repeats=int(uncertainty_repeats),
        repeat_seed_base=int(response_target_seed_base) + int(probe_cfg.support_split_seed) * 101,
    )

    feature_score_mean_tensor, feature_score_std_tensor, feature_recon_mean_tensor, feature_recon_std_tensor = _score_domains_batched(
        bank=bank,
        expert_domains=expert_domains,
        x_cpu=x_cpu,
        device=device,
        batch_size=batch_size,
        n_repeats=int(uncertainty_repeats),
        repeat_seed_base=int(response_feature_seed_base) + int(probe_cfg.support_split_seed) * 101,
    )
    _ = recon_std_tensor, feature_recon_std_tensor

    by_query: Dict[int, List[int]] = {d: [] for d in all_domains}
    for idx, item in enumerate(metadata):
        by_query[_as_int_domain(item["magnification"])].append(idx)

    np_x = x_cpu.detach().cpu().numpy().astype(np.float64, copy=False)
    mean_by_domain = _domain_mean_embeddings(np_x, metadata, domains=all_domains)

    min_d = float(min(all_domains))
    max_d = float(max(all_domains))
    domain_span = max(max_d - min_d, 1.0)

    rows: List[dict] = []
    utility_lookup: Dict[int, Dict[int, float]] = {}
    nelbo_lookup: Dict[int, Dict[int, float]] = {}
    nelbo_std_lookup: Dict[int, Dict[int, float]] = {}

    # Query-domain metadata route for query-side difficulty stats.
    meta_best_by_query: Dict[int, int] = {}
    for q in all_domains:
        sims = [
            compute_similarity(
                {"magnification": int(q)},
                {"magnification": int(e)},
                strategy=ctx.routing_strategy,
                tau=float(ctx.routing_tau),
                similarity_matrix=None,
            )
            for e in expert_domains
        ]
        meta_best_by_query[q] = int(expert_domains[int(np.argmax(np.asarray(sims, dtype=np.float64)))])

    for q in all_domains:
        idxs = by_query.get(q, [])
        if not idxs:
            continue

        split_seed = int(ctx.seed) * 1009 + int(q) * 131 + int(probe_cfg.support_split_seed)
        support_idxs, eval_idxs = _deterministic_query_split(
            idxs,
            query_batch_size=int(probe_cfg.query_batch_size),
            support_fraction=float(probe_cfg.support_fraction),
            split_seed=int(split_seed),
        )
        if not support_idxs or not eval_idxs:
            continue
        if set(support_idxs).intersection(set(eval_idxs)):
            raise RuntimeError(f"Support/eval overlap detected for query_domain={q}")

        utility_lookup[q] = {}
        nelbo_lookup[q] = {}
        nelbo_std_lookup[q] = {}
        query_mean = mean_by_domain.get(q)
        if query_mean is None:
            continue

        q_best_e = int(meta_best_by_query[q])
        q_best_idx = int(expert_domains.index(q_best_e))
        q_support_nelbo = (
            feature_score_mean_tensor[q_best_idx, support_idxs].detach().cpu().numpy().astype(np.float64, copy=False)
        )
        q_support_nelbo_unc = (
            feature_score_std_tensor[q_best_idx, support_idxs].detach().cpu().numpy().astype(np.float64, copy=False)
        )
        q_support_recon = (
            feature_recon_mean_tensor[q_best_idx, support_idxs].detach().cpu().numpy().astype(np.float64, copy=False)
        )
        q_support_kl = q_support_nelbo - q_support_recon
        q_nelbo_mean, q_nelbo_std, q_nelbo_p90 = _summary_stats(q_support_nelbo)
        q_recon_mean, q_recon_std, _ = _summary_stats(q_support_recon)
        q_kl_mean, q_kl_std, _ = _summary_stats(q_support_kl)
        q_unc_mean, q_unc_std, _ = _summary_stats(q_support_nelbo_unc)

        split_id = f"seed{ctx.seed}_q{q}_sup{len(support_idxs)}_eval{len(eval_idxs)}"

        for e_i, e in enumerate(expert_domains):
            support_nelbo = (
                feature_score_mean_tensor[e_i, support_idxs].detach().cpu().numpy().astype(np.float64, copy=False)
            )
            support_nelbo_unc = (
                feature_score_std_tensor[e_i, support_idxs].detach().cpu().numpy().astype(np.float64, copy=False)
            )
            eval_nelbo = score_mean_tensor[e_i, eval_idxs].detach().cpu().numpy().astype(np.float64, copy=False)
            eval_nelbo_unc = score_std_tensor[e_i, eval_idxs].detach().cpu().numpy().astype(np.float64, copy=False)
            if eval_nelbo.size == 0:
                continue
            mean_nelbo = float(eval_nelbo.mean())
            std_nelbo = float(eval_nelbo_unc.mean()) if eval_nelbo_unc.size else 0.0
            utility = -mean_nelbo
            utility_lookup[q][e] = utility
            nelbo_lookup[q][e] = mean_nelbo
            nelbo_std_lookup[q][e] = std_nelbo

            exp_sup_mean, exp_sup_std, exp_sup_p90 = _summary_stats(support_nelbo)
            exp_sup_unc_mean, exp_sup_unc_std, _ = _summary_stats(support_nelbo_unc)

            response_feature_seed = stable_response_seed(
                dataset=ctx.dataset_name,
                seed=int(ctx.seed),
                query_id=int(q),
                expert_domain=int(e),
                repeat_id=0,
                stream_name="response_feature",
            )
            response_feature_repeat_seed_base = int(response_feature_seed) + int(probe_cfg.support_split_seed) * 101
            response_features = compute_response_features(
                bank=bank,
                expert_domain=int(e),
                x_cpu=x_cpu,
                support_idxs=support_idxs,
                device=device,
                n_repeats=int(uncertainty_repeats),
                repeat_seed_base=int(response_feature_repeat_seed_base),
            )

            meta_similarity = compute_similarity(
                {"magnification": int(q)},
                {"magnification": int(e)},
                strategy=ctx.routing_strategy,
                tau=float(ctx.routing_tau),
                similarity_matrix=None,
            )
            meta_distance = 1.0 - float(meta_similarity)

            expert_mean = mean_by_domain.get(e)
            if expert_mean is None:
                continue
            embedding_distance = float(np.linalg.norm(query_mean - expert_mean, ord=2))

            rows.append(
                {
                    "dataset_name": ctx.dataset_name,
                    "seed": int(ctx.seed),
                    "backbone_type": ctx.backbone_type,
                    "run_dir": str(ctx.run_dir),
                    "variant": ctx.variant,
                    "query_domain": int(q),
                    "expert_domain": int(e),
                    "oracle_utility": utility,
                    "oracle_utility_std": std_nelbo,
                    "oracle_nelbo": mean_nelbo,
                    "oracle_nelbo_std": std_nelbo,
                    "metadata_similarity": float(meta_similarity),
                    "metadata_distance": meta_distance,
                    "embedding_distance": embedding_distance,
                    "query_domain_value": (float(q) - min_d) / domain_span,
                    "expert_domain_value": (float(e) - min_d) / domain_span,
                    "abs_domain_diff": abs(float(q) - float(e)) / domain_span,
                    "is_exact_domain_match": 1.0 if int(q) == int(e) else 0.0,
                    "query_nelbo_mean": q_nelbo_mean,
                    "query_nelbo_std": q_nelbo_std,
                    "query_nelbo_uncertainty_mean": q_unc_mean,
                    "query_nelbo_uncertainty_std": q_unc_std,
                    "query_nelbo_p90": q_nelbo_p90,
                    "query_recon_mean": q_recon_mean,
                    "query_recon_std": q_recon_std,
                    "query_kl_mean": q_kl_mean,
                    "query_kl_std": q_kl_std,
                    "expert_support_nelbo_mean": exp_sup_mean,
                    "expert_support_nelbo_std": exp_sup_std,
                    "expert_support_nelbo_uncertainty_mean": exp_sup_unc_mean,
                    "expert_support_nelbo_uncertainty_std": exp_sup_unc_std,
                    "expert_support_nelbo_p90": exp_sup_p90,
                    "expert_eval_nelbo_uncertainty_mean": std_nelbo,
                    "support_size": int(len(support_idxs)),
                    "eval_size": int(len(eval_idxs)),
                    "support_eval_split_id": split_id,
                    "response_seed_scheme_version": str(RESPONSE_SEED_SCHEME_VERSION),
                    "response_feature_stream_name": "response_feature",
                    "response_target_stream_name": "target_oracle",
                    "response_feature_seed_base": int(response_feature_seed_base),
                    "response_target_seed_base": int(response_target_seed_base),
                    "response_feature_seed": int(response_feature_repeat_seed_base),
                    "num_response_repeats": int(uncertainty_repeats),
                    **response_features,
                }
            )

    return rows, expert_domains, utility_lookup, nelbo_lookup, nelbo_std_lookup


def _normalize_targets_per_query(train_rows: Sequence[dict]) -> Dict[int, Tuple[float, float]]:
    by_q: Dict[int, List[float]] = {}
    for row in train_rows:
        by_q.setdefault(int(row["query_domain"]), []).append(float(row["oracle_utility"]))

    stats: Dict[int, Tuple[float, float]] = {}
    for q, vals in by_q.items():
        arr = np.asarray(vals, dtype=np.float64)
        mu = float(arr.mean())
        sigma = float(arr.std())
        if sigma < 1e-8:
            sigma = 1.0
        stats[q] = (mu, sigma)
    return stats


def _with_normalized_targets(rows: Sequence[dict], stats: Dict[int, Tuple[float, float]]) -> np.ndarray:
    ys: List[float] = []
    for row in rows:
        q = int(row["query_domain"])
        if q not in stats:
            raise RuntimeError(f"Missing normalization stats for query_domain={q}")
        mu, sigma = stats[q]
        y = (float(row["oracle_utility"]) - mu) / sigma
        ys.append(float(y))
    return np.asarray(ys, dtype=np.float64)


def _feature_matrix(
    rows: Sequence[dict],
    *,
    feature_set: str,
    expert_domains: Sequence[int],
    include_probe_features: bool,
    include_response_features: bool,
    include_expert_stats: bool,
    include_interaction_features: bool,
) -> np.ndarray:
    features: List[List[float]] = []
    for row in rows:
        base = [float(row["metadata_distance"])]
        if feature_set == "A":
            features.append(base)
            continue

        ext = [
            float(row["embedding_distance"]),
            float(row["query_domain_value"]),
            float(row["expert_domain_value"]),
            float(row["abs_domain_diff"]),
            float(row["is_exact_domain_match"]),
        ]
        # Safe under LOQDO: expert identity one-hot is allowed, query one-hot is intentionally excluded.
        e = int(row["expert_domain"])
        one_hot = [1.0 if e == int(d) else 0.0 for d in expert_domains]
        probe_feats: List[float] = []
        if include_probe_features:
            probe_feats.extend(
                [
                    float(row.get("query_nelbo_mean", 0.0)),
                    float(row.get("query_nelbo_std", 0.0)),
                    float(row.get("query_nelbo_p90", 0.0)),
                    float(row.get("query_recon_mean", 0.0)),
                    float(row.get("query_recon_std", 0.0)),
                    float(row.get("query_kl_mean", 0.0)),
                    float(row.get("query_kl_std", 0.0)),
                ]
            )
            if include_expert_stats:
                probe_feats.extend(
                    [
                        float(row.get("expert_support_nelbo_mean", 0.0)),
                        float(row.get("expert_support_nelbo_std", 0.0)),
                        float(row.get("expert_support_nelbo_p90", 0.0)),
                    ]
                )
        response_feats: List[float] = []
        if include_response_features:
            response_feats.extend(
                [
                    float(row.get("response_posterior_mu_norm", 0.0)),
                    float(row.get("response_posterior_mu_mean", 0.0)),
                    float(row.get("response_posterior_mu_std", 0.0)),
                    float(row.get("response_posterior_logvar_mean", 0.0)),
                    float(row.get("response_posterior_logvar_std", 0.0)),
                    float(row.get("response_posterior_entropy_proxy", 0.0)),
                    float(row.get("response_decode_repeat_var_mean", 0.0)),
                    float(row.get("response_decode_repeat_var_std", 0.0)),
                    float(row.get("response_recon_repeat_var_mean", 0.0)),
                    float(row.get("response_recon_repeat_var_std", 0.0)),
                    float(row.get("response_kl_repeat_var_mean", 0.0)),
                    float(row.get("response_kl_repeat_var_std", 0.0)),
                ]
            )
        interaction_feats: List[float] = []
        if include_interaction_features:
            md = float(row["metadata_distance"])
            ed = float(row["embedding_distance"])
            qv = float(row["query_domain_value"])
            ev = float(row["expert_domain_value"])
            dd = float(row["abs_domain_diff"])
            interaction_feats.extend(
                [
                    md * ed,
                    qv * ev,
                    md * dd,
                    ed * dd,
                    float(row["is_exact_domain_match"]) * ed,
                ]
            )
            if include_probe_features:
                qnm = float(row.get("query_nelbo_mean", 0.0))
                qns = float(row.get("query_nelbo_std", 0.0))
                esn = float(row.get("expert_support_nelbo_mean", 0.0))
                ess = float(row.get("expert_support_nelbo_std", 0.0))
                interaction_feats.extend(
                    [
                        md * qnm,
                        ed * qnm,
                        md * qns,
                        qnm * esn,
                        qns * ess,
                    ]
                )
        features.append(base + ext + one_hot + probe_feats + response_feats + interaction_feats)

    return np.asarray(features, dtype=np.float64)


def _method_scores(
    method: str,
    x_train: np.ndarray,
    y_train_norm: np.ndarray,
    x_test: np.ndarray,
    test_rows: Sequence[dict],
    train_rows: Sequence[dict],
    probe_cfg: ProbeConfig,
    oracle_cfg: OracleProbeConfig,
) -> np.ndarray:
    if method == "constant_mean":
        return np.full((x_test.shape[0],), float(y_train_norm.mean()) if y_train_norm.size else 0.0, dtype=np.float64)

    if method == "expert_prior":
        expert_means: Dict[int, float] = {}
        for row in train_rows:
            e = int(row["expert_domain"])
            expert_means.setdefault(e, 0.0)
        for e in list(expert_means.keys()):
            vals = [float(r["oracle_utility"]) for r in train_rows if int(r["expert_domain"]) == e]
            expert_means[e] = float(np.mean(vals)) if vals else 0.0
        return np.asarray([expert_means.get(int(r["expert_domain"]), 0.0) for r in test_rows], dtype=np.float64)

    if method == "linear_regression":
        try:
            import importlib

            linear_model = importlib.import_module("sklearn.linear_model")
            LinearRegression = getattr(linear_model, "LinearRegression")
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("linear_regression requires scikit-learn.") from exc
        model = LinearRegression()
        model.fit(x_train, y_train_norm)
        return model.predict(x_test)

    if method == "mlp_regression":
        try:
            import importlib

            neural_network = importlib.import_module("sklearn.neural_network")
            MLPRegressor = getattr(neural_network, "MLPRegressor")
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("mlp_regression requires scikit-learn.") from exc
        model = MLPRegressor(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=2000,
            random_state=0,
        )
        model.fit(x_train, y_train_norm)
        return model.predict(x_test)

    if method in {"utility_probe_v1_no_expert_stats", "utility_probe_v1_with_expert_stats"}:
        if not probe_cfg.enabled:
            raise RuntimeError(f"Method {method} requested but utility probe is disabled.")

        y_train = np.asarray([float(r["oracle_utility"]) for r in train_rows], dtype=np.float64)
        if y_train.size == 0:
            return np.zeros((x_test.shape[0],), dtype=np.float64)
        clip_pct = float(probe_cfg.clip_target_percentile)
        if 0.0 < clip_pct < 100.0:
            lo = np.percentile(y_train, 100.0 - clip_pct)
            hi = np.percentile(y_train, clip_pct)
            y_train = np.clip(y_train, lo, hi)

        x_mu, x_sigma = _fit_standardizer(x_train)
        x_train_std = _apply_standardizer(x_train, x_mu, x_sigma)
        x_test_std = _apply_standardizer(x_test, x_mu, x_sigma)
        train_q = np.asarray([int(r["query_domain"]) for r in train_rows], dtype=np.int64)

        return _train_utility_probe(
            x_train=x_train_std,
            y_train=y_train,
            query_ids=train_q,
            x_test=x_test_std,
            cfg=probe_cfg,
        )

    if method == "semi_oracle_support_mean":
        return np.asarray(
            [-float(r.get("expert_support_nelbo_mean", 0.0)) for r in test_rows],
            dtype=np.float64,
        )

    if method == "semi_oracle_support_riskaware":
        lam = float(oracle_cfg.semi_oracle_risk_lambda)
        return np.asarray(
            [
                -(float(r.get("expert_support_nelbo_mean", 0.0)) + lam * float(r.get("expert_support_nelbo_std", 0.0)))
                for r in test_rows
            ],
            dtype=np.float64,
        )

    if method == "oracle_eval_mean_cheat":
        return np.asarray([float(r["oracle_utility"]) for r in test_rows], dtype=np.float64)

    if method == "oracle_pairwise_rank_cheat":
        return _oracle_pairwise_rank_scores(test_rows)

    raise ValueError(f"Unknown method: {method}")


def _evaluate_holdout(
    *,
    ctx: RunContext,
    heldout_domain: int,
    feature_set: str,
    method: str,
    train_rows: Sequence[dict],
    test_rows: Sequence[dict],
    expert_domains: Sequence[int],
    utility_lookup: Dict[int, Dict[int, float]],
    nelbo_lookup: Dict[int, Dict[int, float]],
    nelbo_std_lookup: Dict[int, Dict[int, float]],
    probe_cfg: ProbeConfig,
    oracle_cfg: OracleProbeConfig,
    include_probe_features_override: Optional[bool] = None,
    include_response_features: bool = False,
    include_interaction_features: bool = False,
    disentanglement_arm: str = "default",
) -> dict:
    norm_stats = _normalize_targets_per_query(train_rows)
    include_probe_features = method.startswith("utility_probe_v1")
    if include_probe_features_override is not None:
        include_probe_features = bool(include_probe_features_override)
    include_expert_stats = method.endswith("with_expert_stats")
    x_train = _feature_matrix(
        train_rows,
        feature_set=feature_set,
        expert_domains=expert_domains,
        include_probe_features=include_probe_features,
        include_response_features=bool(include_response_features),
        include_expert_stats=include_expert_stats,
        include_interaction_features=bool(include_interaction_features),
    )
    x_test = _feature_matrix(
        test_rows,
        feature_set=feature_set,
        expert_domains=expert_domains,
        include_probe_features=include_probe_features,
        include_response_features=bool(include_response_features),
        include_expert_stats=include_expert_stats,
        include_interaction_features=bool(include_interaction_features),
    )
    y_train_norm = _with_normalized_targets(train_rows, norm_stats)

    y_true = np.asarray([float(r["oracle_utility"]) for r in test_rows], dtype=np.float64)
    scores = _method_scores(
        method=method,
        x_train=x_train,
        y_train_norm=y_train_norm,
        x_test=x_test,
        test_rows=test_rows,
        train_rows=train_rows,
        probe_cfg=probe_cfg,
        oracle_cfg=oracle_cfg,
    )

    slope, intercept, cal_err = _calibration_metrics(scores, y_true, n_bins=10)
    top1_margin = _top1_margin(scores)
    split_id = str(test_rows[0].get("support_eval_split_id", "na")) if test_rows else "na"

    pred_best_idx = int(np.argmax(scores))
    true_best_idx = int(np.argmax(y_true))
    selected_e = int(test_rows[pred_best_idx]["expert_domain"])
    oracle_e = int(test_rows[true_best_idx]["expert_domain"])

    util_vec = [utility_lookup[heldout_domain][int(e)] for e in expert_domains]
    selected_rank = _row_rank_desc(util_vec, selected_idx=expert_domains.index(selected_e))

    selected_nelbo = float(nelbo_lookup[heldout_domain][selected_e])
    oracle_nelbo = float(nelbo_lookup[heldout_domain][oracle_e])
    selected_nelbo_std = float(nelbo_std_lookup[heldout_domain][selected_e])
    oracle_nelbo_std = float(nelbo_std_lookup[heldout_domain][oracle_e])
    worst_nelbo = float(max(float(v) for v in nelbo_lookup[heldout_domain].values()))
    normalized_gap = _normalized_utility_gap(selected_nelbo, oracle_nelbo, worst_nelbo)

    return {
        "dataset_name": ctx.dataset_name,
        "seed": int(ctx.seed),
        "backbone_type": ctx.backbone_type,
        "run_id": ctx.run_dir.name,
        "variant": ctx.variant,
        "feature_set": feature_set,
        "method": method,
        "probe_feature_mode": "on" if include_probe_features else "off",
        "response_feature_mode": "on" if bool(include_response_features) else "off",
        "response_feature_normalization": "train_fold_standardize"
        if bool(include_response_features)
        else "none",
        "interaction_feature_mode": "on" if bool(include_interaction_features) else "off",
        "disentanglement_arm": str(disentanglement_arm),
        "heldout_query_domain": int(heldout_domain),
        "n_train_rows": int(len(train_rows)),
        "n_test_rows": int(len(test_rows)),
        "n_experts": int(len(expert_domains)),
        "selected_expert": int(selected_e),
        "oracle_expert": int(oracle_e),
        "top1_agreement_with_best_expert": 1.0 if selected_e == oracle_e else 0.0,
        "spearman_similarity_vs_neg_nelbo": float(spearman_corr(scores.tolist(), y_true.tolist())),
        "metadata_to_oracle_gap": float(selected_nelbo - oracle_nelbo),
        "normalized_metadata_to_oracle_gap": normalized_gap,
        "mean_rank_metadata_selected": float(selected_rank),
        "selected_routing_nelbo": selected_nelbo,
        "selected_routing_nelbo_std": selected_nelbo_std,
        "oracle_routing_nelbo": oracle_nelbo,
        "oracle_routing_nelbo_std": oracle_nelbo_std,
        "metadata_to_oracle_gap_std_approx": float(math.sqrt(max(selected_nelbo_std**2 + oracle_nelbo_std**2, 0.0))),
        "worst_routing_nelbo": worst_nelbo,
        "fold_id": f"{ctx.run_dir.name}:{int(heldout_domain)}",
        "support_eval_split_id": split_id,
        "split_policy": "loqdo_query_domain",
        "uncertainty_mode": "point_estimate",
        "model_capacity_profile": "default",
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "calibration_error_bin10": cal_err,
        "top1_margin": top1_margin,
        "target_variance": float(np.var(y_true)) if y_true.size else 0.0,
        "oracle_pairwise_inconsistency_rate": _pairwise_inconsistency_rate(y_true),
    }


def _evaluate_metadata_baseline(
    *,
    ctx: RunContext,
    heldout_domain: int,
    test_rows: Sequence[dict],
    expert_domains: Sequence[int],
    utility_lookup: Dict[int, Dict[int, float]],
    nelbo_lookup: Dict[int, Dict[int, float]],
    nelbo_std_lookup: Dict[int, Dict[int, float]],
) -> dict:
    sims = [
        compute_similarity(
            {"magnification": int(heldout_domain)},
            {"magnification": int(e)},
            strategy=ctx.routing_strategy,
            tau=float(ctx.routing_tau),
            similarity_matrix=None,
        )
        for e in expert_domains
    ]
    pred_best_idx = int(np.argmax(np.asarray(sims, dtype=np.float64)))
    util_vec = [utility_lookup[heldout_domain][int(e)] for e in expert_domains]
    true_best_idx = int(np.argmax(np.asarray(util_vec, dtype=np.float64)))
    selected_e = int(expert_domains[pred_best_idx])
    oracle_e = int(expert_domains[true_best_idx])
    selected_rank = _row_rank_desc(util_vec, selected_idx=pred_best_idx)
    selected_nelbo = float(nelbo_lookup[heldout_domain][selected_e])
    oracle_nelbo = float(nelbo_lookup[heldout_domain][oracle_e])
    selected_nelbo_std = float(nelbo_std_lookup[heldout_domain][selected_e])
    oracle_nelbo_std = float(nelbo_std_lookup[heldout_domain][oracle_e])
    worst_nelbo = float(max(float(v) for v in nelbo_lookup[heldout_domain].values()))
    normalized_gap = _normalized_utility_gap(selected_nelbo, oracle_nelbo, worst_nelbo)
    y_true = np.asarray(util_vec, dtype=np.float64)
    y_pred = np.asarray(sims, dtype=np.float64)
    slope, intercept, cal_err = _calibration_metrics(y_pred, y_true, n_bins=10)

    return {
        "dataset_name": ctx.dataset_name,
        "seed": int(ctx.seed),
        "backbone_type": ctx.backbone_type,
        "run_id": ctx.run_dir.name,
        "variant": ctx.variant,
        "feature_set": "baseline",
        "method": "metadata_routing",
        "probe_feature_mode": "off",
        "response_feature_mode": "off",
        "response_feature_normalization": "none",
        "interaction_feature_mode": "off",
        "disentanglement_arm": "baseline",
        "heldout_query_domain": int(heldout_domain),
        "n_train_rows": 0,
        "n_test_rows": int(len(test_rows)),
        "n_experts": int(len(expert_domains)),
        "selected_expert": int(selected_e),
        "oracle_expert": int(oracle_e),
        "top1_agreement_with_best_expert": 1.0 if selected_e == oracle_e else 0.0,
        "spearman_similarity_vs_neg_nelbo": float(spearman_corr(sims, util_vec)),
        "metadata_to_oracle_gap": float(selected_nelbo - oracle_nelbo),
        "normalized_metadata_to_oracle_gap": normalized_gap,
        "mean_rank_metadata_selected": float(selected_rank),
        "selected_routing_nelbo": selected_nelbo,
        "selected_routing_nelbo_std": selected_nelbo_std,
        "oracle_routing_nelbo": oracle_nelbo,
        "oracle_routing_nelbo_std": oracle_nelbo_std,
        "metadata_to_oracle_gap_std_approx": float(math.sqrt(max(selected_nelbo_std**2 + oracle_nelbo_std**2, 0.0))),
        "worst_routing_nelbo": worst_nelbo,
        "fold_id": f"{ctx.run_dir.name}:{int(heldout_domain)}",
        "support_eval_split_id": str(test_rows[0].get("support_eval_split_id", "na")) if test_rows else "na",
        "split_policy": "loqdo_query_domain",
        "uncertainty_mode": "point_estimate",
        "model_capacity_profile": "default",
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "calibration_error_bin10": cal_err,
        "top1_margin": _top1_margin(y_pred),
        "target_variance": float(np.var(y_true)) if y_true.size else 0.0,
        "oracle_pairwise_inconsistency_rate": _pairwise_inconsistency_rate(y_true),
    }


def _write_csv(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_parquet(rows: Sequence[dict], path: Path, required: bool = False) -> bool:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - environment dependent
        msg = "Parquet output requires pandas and pyarrow or fastparquet."
        if required:
            raise RuntimeError(msg) from exc
        print(f"[warn] {msg} Skipping parquet output for {path}.")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # pragma: no cover - environment dependent
        msg = "Failed to write parquet. Install pyarrow or fastparquet."
        if required:
            raise RuntimeError(msg) from exc
        print(f"[warn] {msg} Skipping parquet output for {path}.")
        return False

    return True


def _aggregate(rows: Sequence[dict]) -> List[dict]:
    metrics = [
        "metadata_to_oracle_gap",
        "metadata_to_oracle_gap_std_approx",
        "normalized_metadata_to_oracle_gap",
        "top1_agreement_with_best_expert",
        "spearman_similarity_vs_neg_nelbo",
        "mean_rank_metadata_selected",
        "calibration_slope",
        "calibration_intercept",
        "calibration_error_bin10",
        "top1_margin",
        "target_variance",
        "oracle_pairwise_inconsistency_rate",
    ]
    groups: Dict[Tuple[str, str, str, str, str, str, str, str], List[dict]] = {}
    for row in rows:
        key = (
            str(row["dataset_name"]),
            str(row["backbone_type"]),
            str(row["variant"]),
            str(row["feature_set"]),
            str(row["method"]),
            str(row.get("probe_feature_mode", "off")),
            str(row.get("response_feature_mode", "off")),
            str(row.get("interaction_feature_mode", "off")),
        )
        groups.setdefault(key, []).append(row)

    out: List[dict] = []
    for key, vals in groups.items():
        dataset_name, backbone_type, variant, feature_set, method, probe_mode, response_mode, interaction_mode = key
        row = {
            "dataset_name": dataset_name,
            "backbone_type": backbone_type,
            "variant": variant,
            "feature_set": feature_set,
            "method": method,
            "probe_feature_mode": probe_mode,
            "response_feature_mode": response_mode,
            "interaction_feature_mode": interaction_mode,
            "n_folds": int(len(vals)),
        }
        for m in metrics:
            arr = np.asarray([float(v[m]) for v in vals], dtype=np.float64)
            row[f"{m}_mean"] = float(arr.mean()) if arr.size else 0.0
            row[f"{m}_std"] = float(arr.std()) if arr.size else 0.0
        out.append(row)

    out.sort(key=lambda r: (r["dataset_name"], r["backbone_type"], r["feature_set"], r["method"]))
    return out


def _default_experiment_dirs(root: Path) -> List[Path]:
    return [
        root / "outputs" / "breakhis" / "hybrid_ablation_extractor_resnet18_v1",
        root / "outputs" / "breakhis" / "hybrid_ablation_extractor_resnet50_v1",
        root / "outputs" / "breakhis" / "hybrid_ablation_extractor_dinov2_vitb14_v1",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe LOQDO learned compatibility routing analysis.")
    parser.add_argument(
        "--experiment-dirs",
        nargs="+",
        default=None,
        help="Run directories or experiment directories (with latest.txt). Defaults to BreakHis backbone trio.",
    )
    parser.add_argument("--variant", default="B", help="Hybrid variant checkpoint to score (default: B).")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--skip-mlp", action="store_true", help="Skip the optional MLP regression method.")
    parser.add_argument("--skip-utility-probe", action="store_true", help="Skip utility_probe_v1 methods.")
    parser.add_argument("--skip-oracle-probes", action="store_true", help="Skip semi-oracle and oracle diagnostic methods.")
    parser.add_argument("--semi-oracle-risk-lambda", type=float, default=0.5)
    parser.add_argument("--utility-probe-hidden-dim", type=int, default=32)
    parser.add_argument("--utility-probe-dropout", type=float, default=0.2)
    parser.add_argument("--utility-probe-lr", type=float, default=1.0e-3)
    parser.add_argument("--utility-probe-epochs", type=int, default=300)
    parser.add_argument("--utility-probe-alpha", type=float, default=1.0)
    parser.add_argument("--utility-probe-beta", type=float, default=0.3)
    parser.add_argument("--support-fraction", type=float, default=0.5)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--support-split-seed", type=int, default=17)
    parser.add_argument(
        "--clip-target-percentile",
        type=float,
        default=99.0,
        help="Symmetric percentile clipping for utility-probe training targets (0-100).",
    )
    parser.add_argument(
        "--pair-table-dirname",
        default="learned_compatibility_loqdo",
        help="Directory name under each run's reports/ where pair tables are saved.",
    )
    parser.add_argument(
        "--raw-out",
        default="results/comparison_tables/learned_compatibility_loqdo_breakhis_raw.csv",
        help="Workspace-relative CSV path for fold-level metrics.",
    )
    parser.add_argument(
        "--stats-out",
        default="results/comparison_tables/learned_compatibility_loqdo_breakhis_stats.csv",
        help="Workspace-relative CSV path for aggregated metrics.",
    )
    parser.add_argument(
        "--summary-json-out",
        default="results/comparison_tables/learned_compatibility_loqdo_breakhis_summary.json",
        help="Workspace-relative JSON path for run metadata and outputs.",
    )
    parser.add_argument(
        "--require-parquet",
        action="store_true",
        help="Fail if parquet cannot be written (default: false, warn and continue with CSV).",
    )
    parser.add_argument(
        "--split-policy",
        type=str,
        default="loqdo_query_domain",
        help="Protocol identifier emitted in raw rows for provenance.",
    )
    parser.add_argument(
        "--uncertainty-mode",
        type=str,
        default="point_estimate",
        help="Uncertainty mode label emitted in raw rows for provenance.",
    )
    parser.add_argument(
        "--model-capacity-profile",
        type=str,
        default="default",
        help="Model-capacity profile label emitted in raw rows for provenance.",
    )
    parser.add_argument(
        "--uncertainty-repeats",
        type=int,
        default=1,
        help="Number of repeated expert scoring passes used to estimate per-pair NELBO uncertainty.",
    )
    parser.add_argument(
        "--enable-interaction-features",
        action="store_true",
        help="Enable interaction feature terms for non-baseline methods.",
    )
    parser.add_argument(
        "--probe-interaction-disentanglement",
        action="store_true",
        help="Evaluate four probe/interaction arms for utility probe methods: neither, interaction_only, probe_only, probe_plus_interaction.",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="all",
        help="Feature/method regime to run: all, metadata_only, static_A, static_B, response_indirect",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = PROJECT_ROOT
    effective_uncertainty_mode = str(args.uncertainty_mode)
    if int(args.uncertainty_repeats) > 1 and effective_uncertainty_mode == "point_estimate":
        effective_uncertainty_mode = "repeated_nelbo_meanstd"

    experiment_dirs = [Path(p) for p in args.experiment_dirs] if args.experiment_dirs else _default_experiment_dirs(workspace_root)
    resolved_runs = [_resolve_run_dir(p if p.is_absolute() else (workspace_root / p)) for p in experiment_dirs]

    probe_cfg = ProbeConfig(
        enabled=not bool(args.skip_utility_probe),
        hidden_dim=int(args.utility_probe_hidden_dim),
        dropout=float(args.utility_probe_dropout),
        learning_rate=float(args.utility_probe_lr),
        epochs=int(args.utility_probe_epochs),
        alpha=float(args.utility_probe_alpha),
        beta=float(args.utility_probe_beta),
        support_fraction=float(args.support_fraction),
        query_batch_size=int(args.query_batch_size),
        support_split_seed=int(args.support_split_seed),
        clip_target_percentile=float(args.clip_target_percentile),
    )
    oracle_cfg = OracleProbeConfig(
        enabled=not bool(args.skip_oracle_probes),
        semi_oracle_risk_lambda=float(args.semi_oracle_risk_lambda),
    )

    methods = ["constant_mean", "expert_prior", "linear_regression"]
    if not bool(args.skip_mlp):
        methods.append("mlp_regression")
    if probe_cfg.enabled:
        methods.extend(["utility_probe_v1_no_expert_stats", "utility_probe_v1_with_expert_stats"])
    if oracle_cfg.enabled:
        methods.extend(
            [
                "semi_oracle_support_mean",
                "semi_oracle_support_riskaware",
                "oracle_eval_mean_cheat",
                "oracle_pairwise_rank_cheat",
            ]
        )

    # Regime selection controls which feature sets / methods are executed.
    regime = str(args.regime).lower()
    if regime == "all":
        allowed_feature_sets = ["A", "B"]
        allowed_methods = methods
    elif regime == "metadata_only":
        allowed_feature_sets = ["A"]
        allowed_methods = [m for m in methods if m in {"constant_mean", "expert_prior", "linear_regression"}]
    elif regime == "static_a":
        allowed_feature_sets = ["A"]
        allowed_methods = methods
    elif regime == "static_b":
        allowed_feature_sets = ["B"]
        allowed_methods = methods
    elif regime == "response_indirect":
        allowed_feature_sets = ["B"]
        # Response-indirect regime focuses on utility-probe methods only.
        allowed_methods = [m for m in methods if m.startswith("utility_probe_v1")]
    else:
        print(f"[warn] Unknown regime '{args.regime}', falling back to 'all'.")
        allowed_feature_sets = ["A", "B"]
        allowed_methods = methods

    all_fold_rows: List[dict] = []
    run_summaries: List[dict] = []

    for run_dir in resolved_runs:
        ctx = _load_run_context(run_dir, variant=args.variant)
        pair_rows, expert_domains, utility_lookup, nelbo_lookup, nelbo_std_lookup = _build_pair_rows(
            ctx,
            batch_size=int(args.batch_size),
            probe_cfg=probe_cfg,
            uncertainty_repeats=int(args.uncertainty_repeats),
        )

        pair_rows = sorted(pair_rows, key=lambda r: (int(r["query_domain"]), int(r["expert_domain"])))
        pair_dir = ctx.run_dir / "reports" / str(args.pair_table_dirname)
        pair_dir.mkdir(parents=True, exist_ok=True)

        rows_a = [
            {
                **r,
                "feature_set": "A",
                "held_out_query_domain": None,
            }
            for r in pair_rows
        ]
        rows_b = [
            {
                **r,
                "feature_set": "B",
                "held_out_query_domain": None,
            }
            for r in pair_rows
        ]
        _write_csv(rows_a, pair_dir / "pair_table_A.csv")
        _write_csv(rows_b, pair_dir / "pair_table_B.csv")
        _write_parquet(rows_a, pair_dir / "pair_table_A.parquet", required=bool(args.require_parquet))
        _write_parquet(rows_b, pair_dir / "pair_table_B.parquet", required=bool(args.require_parquet))

        query_domains = sorted(set(int(r["query_domain"]) for r in pair_rows))
        run_rows: List[dict] = []
        response_norm_reports: List[dict] = []

        for heldout in query_domains:
            train_rows = [r for r in pair_rows if int(r["query_domain"]) != int(heldout)]
            test_rows = [r for r in pair_rows if int(r["query_domain"]) == int(heldout)]
            if not train_rows or not test_rows:
                continue

            norm_train_rows = train_rows
            norm_test_rows = test_rows
            if regime == "response_indirect":
                norm_stats = _fit_response_feature_standardizer(train_rows)
                norm_train_rows = _apply_response_feature_standardizer(train_rows, norm_stats)
                norm_test_rows = _apply_response_feature_standardizer(test_rows, norm_stats)
                split_id = str(test_rows[0].get("support_eval_split_id", "na")) if test_rows else "na"
                response_norm_reports.extend(
                    _response_feature_norm_report_rows(
                        ctx=ctx,
                        heldout_domain=heldout,
                        split_id=split_id,
                        stats=norm_stats,
                        n_train_rows=len(train_rows),
                    )
                )

            baseline_row = _evaluate_metadata_baseline(
                ctx=ctx,
                heldout_domain=heldout,
                test_rows=test_rows,
                expert_domains=expert_domains,
                utility_lookup=utility_lookup,
                nelbo_lookup=nelbo_lookup,
                nelbo_std_lookup=nelbo_std_lookup,
            )
            baseline_row["split_policy"] = str(args.split_policy)
            baseline_row["uncertainty_mode"] = effective_uncertainty_mode
            baseline_row["model_capacity_profile"] = str(args.model_capacity_profile)
            run_rows.append(baseline_row)

            for feature_set in allowed_feature_sets:
                for method in allowed_methods:
                    if method.startswith("utility_probe_v1") and feature_set != "B":
                        continue
                    if (method.startswith("semi_oracle_") or method.startswith("oracle_")) and feature_set != "B":
                        continue
                    arm_specs: List[Tuple[str, Optional[bool], bool]] = [("default", None, bool(args.enable_interaction_features))]
                    if bool(args.probe_interaction_disentanglement) and method.startswith("utility_probe_v1") and feature_set == "B":
                        arm_specs = [
                            ("neither", False, False),
                            ("interaction_only", False, True),
                            ("probe_only", True, False),
                            ("probe_plus_interaction", True, True),
                        ]

                    include_response_features = False
                    if regime == "response_indirect":
                        include_response_features = True
                        arm_specs = [("response_only", False, bool(args.enable_interaction_features))]

                    for arm_name, probe_override, interaction_on in arm_specs:
                        eval_train_rows = norm_train_rows if include_response_features else train_rows
                        eval_test_rows = norm_test_rows if include_response_features else test_rows
                        row = _evaluate_holdout(
                            ctx=ctx,
                            heldout_domain=heldout,
                            feature_set=feature_set,
                            method=method,
                            train_rows=eval_train_rows,
                            test_rows=eval_test_rows,
                            expert_domains=expert_domains,
                            utility_lookup=utility_lookup,
                            nelbo_lookup=nelbo_lookup,
                            nelbo_std_lookup=nelbo_std_lookup,
                            probe_cfg=probe_cfg,
                            oracle_cfg=oracle_cfg,
                            include_probe_features_override=probe_override,
                            include_response_features=include_response_features,
                            include_interaction_features=interaction_on,
                            disentanglement_arm=arm_name,
                        )
                        row["split_policy"] = str(args.split_policy)
                        row["uncertainty_mode"] = effective_uncertainty_mode
                        row["model_capacity_profile"] = str(args.model_capacity_profile)
                        run_rows.append(row)

        all_fold_rows.extend(run_rows)
        run_summaries.append(
            {
                "run_dir": str(ctx.run_dir),
                "dataset_name": ctx.dataset_name,
                "seed": int(ctx.seed),
                "backbone_type": ctx.backbone_type,
                "variant": ctx.variant,
                "n_pair_rows": int(len(pair_rows)),
                "n_fold_rows": int(len(run_rows)),
                "expert_domains": [int(d) for d in expert_domains],
                "split_policy": str(args.split_policy),
                "uncertainty_mode": effective_uncertainty_mode,
                "uncertainty_repeats": int(args.uncertainty_repeats),
                "model_capacity_profile": str(args.model_capacity_profile),
                "enable_interaction_features": bool(args.enable_interaction_features),
                "probe_interaction_disentanglement": bool(args.probe_interaction_disentanglement),
                "regime": str(args.regime),
            }
        )

        with (pair_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(run_summaries[-1], f, indent=2)

        if response_norm_reports:
            _write_csv(response_norm_reports, pair_dir / "response_feature_normalization_report.csv")

    raw_rows_sorted = sorted(
        all_fold_rows,
        key=lambda r: (
            str(r["dataset_name"]),
            str(r["backbone_type"]),
            str(r["run_id"]),
            str(r["feature_set"]),
            str(r["method"]),
            int(r["heldout_query_domain"]),
        ),
    )
    stats_rows = _aggregate(raw_rows_sorted)

    raw_out = workspace_root / str(args.raw_out)
    stats_out = workspace_root / str(args.stats_out)
    summary_out = workspace_root / str(args.summary_json_out)

    _write_csv(raw_rows_sorted, raw_out)
    _write_csv(stats_rows, stats_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "methods": methods,
                "probe_config": {
                    "enabled": probe_cfg.enabled,
                    "hidden_dim": probe_cfg.hidden_dim,
                    "dropout": probe_cfg.dropout,
                    "learning_rate": probe_cfg.learning_rate,
                    "epochs": probe_cfg.epochs,
                    "alpha": probe_cfg.alpha,
                    "beta": probe_cfg.beta,
                    "support_fraction": probe_cfg.support_fraction,
                    "query_batch_size": probe_cfg.query_batch_size,
                    "support_split_seed": probe_cfg.support_split_seed,
                    "clip_target_percentile": probe_cfg.clip_target_percentile,
                },
                "oracle_probe_config": {
                    "enabled": oracle_cfg.enabled,
                    "semi_oracle_risk_lambda": oracle_cfg.semi_oracle_risk_lambda,
                },
                "protocol": {
                    "split_policy": str(args.split_policy),
                    "uncertainty_mode": effective_uncertainty_mode,
                    "uncertainty_repeats": int(args.uncertainty_repeats),
                    "model_capacity_profile": str(args.model_capacity_profile),
                    "enable_interaction_features": bool(args.enable_interaction_features),
                    "probe_interaction_disentanglement": bool(args.probe_interaction_disentanglement),
                    "regime": str(args.regime),
                },
                "runs": run_summaries,
                "raw_csv": str(raw_out),
                "stats_csv": str(stats_out),
                "n_raw_rows": int(len(raw_rows_sorted)),
                "n_stats_rows": int(len(stats_rows)),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
