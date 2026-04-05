from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.eval.evaluators.latent_compatibility import (
    compute_distance_matrices,
    compute_domain_gaussian_stats,
    distance_to_similarity,
)
from src.eval.metrics import spearman_corr
from src.models.cvae_expert import CVAEExpert, elbo_components
from src.routing.strategies import compute_similarity
from src.torch_utils import safe_torch_load


_ALLOWED_NORM_POLICIES = {"per_query_zscore", "per_query_minmax"}
_DEFAULT_ALPHA_GRID = [i / 10.0 for i in range(11)]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_s = str(key)
            if key_s not in seen:
                seen.add(key_s)
                fieldnames.append(key_s)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_domain_from_meta(value: object) -> int:
    return int(str(value).replace("x", ""))


def _load_model(checkpoint: Path, input_dim: int, hidden_dim: int, latent_dim: int, device: torch.device) -> CVAEExpert:
    model = CVAEExpert(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(safe_torch_load(checkpoint, map_location=device))
    model.eval()
    return model


def _score_model_nelbo(model: CVAEExpert, x: torch.Tensor) -> torch.Tensor:
    recon, mu, logvar = model(x)
    rec, kl = elbo_components(recon, x, mu, logvar)
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[Dict[str, Any]]]:
    payload = safe_torch_load(test_cache, map_location="cpu")
    x_cpu = payload["embeddings"]
    metadata = payload["metadata"]
    sample_domains = np.asarray([_as_domain_from_meta(m["magnification"]) for m in metadata], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = int(x_cpu.shape[1])

    expert_names = sorted(expert_checkpoints.keys())
    expert_domains = [_parse_expert_domain(name) for name in expert_names]
    models = [
        _load_model(Path(expert_checkpoints[name]), input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, device=device)
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
            for e_idx, model in enumerate(models):
                expert_chunks[e_idx].append(_score_model_nelbo(model, xb).cpu())

    for e_idx in range(n_experts):
        nelbo[:, e_idx] = torch.cat(expert_chunks[e_idx], dim=0).numpy().astype(np.float64, copy=False)

    return x_np, sample_domains, nelbo, expert_domains, metadata


def _domain_to_expert_index(expert_domains: Sequence[int]) -> Dict[int, int]:
    return {int(d): idx for idx, d in enumerate(expert_domains)}


def _metadata_scores(sample_domains: np.ndarray, expert_domains: Sequence[int], strategy: str, tau: float) -> np.ndarray:
    n = int(sample_domains.shape[0])
    e = len(expert_domains)
    out = np.zeros((n, e), dtype=np.float64)
    cache: Dict[int, np.ndarray] = {}
    for i in range(n):
        q = int(sample_domains[i])
        if q not in cache:
            cache[q] = np.asarray(
                [
                    float(
                        compute_similarity(
                            {"magnification": q},
                            {"magnification": int(ed)},
                            strategy=strategy,
                            tau=float(tau),
                            similarity_matrix=None,
                        )
                    )
                    for ed in expert_domains
                ],
                dtype=np.float64,
            )
        out[i, :] = cache[q]
    return out


def _latent_wasserstein_scores(embeddings: np.ndarray, sample_domains: np.ndarray, expert_domains: Sequence[int]) -> np.ndarray:
    domain_order = sorted(set(int(d) for d in sample_domains.tolist()))
    domain_order, stats, _warnings = compute_domain_gaussian_stats(
        embeddings=embeddings,
        domains=sample_domains,
        covariance_regularization_lambda=1e-4,
        min_samples_per_domain=5,
    )
    distances = compute_distance_matrices(domain_order=domain_order, stats=stats, eigenvalue_floor=1e-10)
    similarity, _scale = distance_to_similarity(distances["wasserstein"], scale_floor=1e-8)

    d_to_row = {int(d): i for i, d in enumerate(domain_order)}
    q_rows = np.asarray([d_to_row[int(q)] for q in sample_domains.tolist()], dtype=np.int64)

    e_rows = np.asarray([d_to_row.get(int(ed), -1) for ed in expert_domains], dtype=np.int64)
    valid = e_rows >= 0

    out = np.full((int(sample_domains.shape[0]), len(expert_domains)), float("-inf"), dtype=np.float64)
    if np.any(valid):
        out[:, valid] = similarity[q_rows[:, None], e_rows[valid][None, :]]
    return out


def _normalize_scores_per_query(scores: np.ndarray, policy: str) -> np.ndarray:
    policy_norm = str(policy).strip().lower()
    if policy_norm not in _ALLOWED_NORM_POLICIES:
        raise ValueError(
            f"Unknown normalization policy: {policy}. Allowed: {sorted(_ALLOWED_NORM_POLICIES)}"
        )

    out = np.zeros_like(scores, dtype=np.float64)
    for i in range(scores.shape[0]):
        row = scores[i, :].astype(np.float64, copy=False)
        if not np.isfinite(row).all():
            raise ValueError("Normalization received non-finite proxy row values")

        if policy_norm == "per_query_zscore":
            mu = float(np.mean(row))
            sigma = float(np.std(row))
            # Explicit zero-variance policy: emit zeros for this row.
            if sigma < 1e-12:
                out[i, :] = 0.0
            else:
                out[i, :] = (row - mu) / sigma
        else:
            lo = float(np.min(row))
            hi = float(np.max(row))
            span = hi - lo
            # Explicit zero-variance policy: emit zeros for this row.
            if span < 1e-12:
                out[i, :] = 0.0
            else:
                out[i, :] = (row - lo) / span

    return out


def _proxy_diagnostic_rows(scores: np.ndarray, sample_domains: np.ndarray, method: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(scores.shape[0]):
        row = scores[i, :]
        rows.append(
            {
                "method": str(method),
                "sample_index": int(i),
                "query_domain": int(sample_domains[i]),
                "row_min": float(np.min(row)),
                "row_max": float(np.max(row)),
                "row_mean": float(np.mean(row)),
                "row_std": float(np.std(row)),
            }
        )
    return rows


def _stable_argmin_indices(matrix: np.ndarray) -> np.ndarray:
    n_rows, n_cols = matrix.shape
    tie_break = np.arange(n_cols, dtype=np.int64)
    out = np.zeros((n_rows,), dtype=np.int64)
    for i in range(n_rows):
        order = np.lexsort((tie_break, matrix[i, :]))
        out[i] = int(order[0])
    return out


def _selected_rank_in_true_matrix(selected_idx: np.ndarray, true_nelbo_matrix: np.ndarray) -> np.ndarray:
    n_rows, n_cols = true_nelbo_matrix.shape
    tie_break = np.arange(n_cols, dtype=np.int64)
    out = np.zeros((n_rows,), dtype=np.float64)
    for i in range(n_rows):
        order = np.lexsort((tie_break, true_nelbo_matrix[i, :]))
        ranks = np.empty((n_cols,), dtype=np.int64)
        ranks[order] = np.arange(1, n_cols + 1, dtype=np.int64)
        out[i] = float(ranks[int(selected_idx[i])])
    return out


def _build_pair_features(
    *,
    sample_embeddings: np.ndarray,
    sample_domains: np.ndarray,
    sample_indices: np.ndarray,
    expert_domains: Sequence[int],
    include_metadata_features: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    e = len(expert_domains)
    x_sel = sample_embeddings[sample_indices]
    q_sel = sample_domains[sample_indices]
    n = int(x_sel.shape[0])

    sample_rep = np.repeat(x_sel, repeats=e, axis=0)
    expert_oh = np.tile(np.eye(e, dtype=np.float64), (n, 1))
    features = [sample_rep, expert_oh]

    query_rep = np.repeat(q_sel.astype(np.float64), repeats=e)
    expert_rep = np.tile(np.asarray([int(d) for d in expert_domains], dtype=np.float64), reps=n)

    if include_metadata_features:
        span = max(float(np.max(sample_domains) - np.min(sample_domains)), 1.0)
        abs_diff = np.abs(query_rep - expert_rep) / span
        exact = (query_rep == expert_rep).astype(np.float64)
        features.append(np.stack([abs_diff, exact], axis=1))

    x = np.concatenate(features, axis=1)
    return x, query_rep.astype(np.int64), expert_rep.astype(np.int64), np.repeat(sample_indices.astype(np.int64), repeats=e)


def _zscore_features(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (x_train - mu) / sigma, (x_test - mu) / sigma


def _normalize_targets_per_query(y: np.ndarray, query_domains: np.ndarray) -> np.ndarray:
    y_norm = np.zeros_like(y, dtype=np.float64)
    for q in sorted(set(int(v) for v in query_domains.tolist())):
        idx = np.where(query_domains == int(q))[0]
        vals = y[idx]
        mu = float(vals.mean())
        sigma = float(vals.std())
        if sigma < 1e-8:
            sigma = 1.0
        y_norm[idx] = (vals - mu) / sigma
    return y_norm


@dataclass
class _LinearRegressor:
    l2: float = 1e-4
    w: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
        xtx = x_aug.T @ x_aug
        xtx += float(self.l2) * np.eye(xtx.shape[0], dtype=np.float64)
        self.w = np.linalg.solve(xtx, x_aug.T @ y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Linear regressor is not fitted")
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
        return x_aug @ self.w


@dataclass
class _MLPRegressor:
    seed: int
    hidden_dim: int = 128
    epochs: int = 40
    lr: float = 1e-3
    batch_size: int = 2048
    device: str = "auto"
    model: torch.nn.Module | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        torch.manual_seed(int(self.seed))
        if self.device == "auto":
            run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            run_device = torch.device(self.device)

        net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], int(self.hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(self.hidden_dim), 1),
        ).to(run_device)
        opt = torch.optim.Adam(net.parameters(), lr=float(self.lr))
        loss_fn = torch.nn.MSELoss()

        x_t = torch.from_numpy(x.astype(np.float32, copy=False)).to(run_device)
        y_t = torch.from_numpy(y.astype(np.float32, copy=False)).view(-1, 1).to(run_device)

        n = int(x_t.shape[0])
        for _ in range(int(self.epochs)):
            perm = torch.randperm(n)
            for i in range(0, n, int(self.batch_size)):
                idx = perm[i : i + int(self.batch_size)]
                pred = net(x_t[idx])
                loss = loss_fn(pred, y_t[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        self.model = net.eval()

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MLP regressor is not fitted")
        model_device = next(self.model.parameters()).device
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32, copy=False)).to(model_device)
            pred = self.model(x_t).view(-1)
        return pred.detach().cpu().numpy().astype(np.float64, copy=False)


def _pairwise_auc_single(score_row: np.ndarray, true_row: np.ndarray) -> float:
    n = int(score_row.shape[0])
    if n <= 1:
        return 0.5
    total = 0.0
    correct = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ti = float(true_row[i])
            tj = float(true_row[j])
            if abs(ti - tj) < 1e-12:
                continue

            si = float(score_row[i])
            sj = float(score_row[j])
            total += 1.0
            if abs(si - sj) < 1e-12:
                correct += 0.5
                continue

            true_better_i = ti < tj
            pred_better_i = si < sj
            if true_better_i == pred_better_i:
                correct += 1.0

    return float(correct / total) if total > 0.0 else 0.5


def _pairwise_auc_matrix(score_matrix: np.ndarray, true_nelbo_matrix: np.ndarray) -> float:
    vals = [
        _pairwise_auc_single(score_matrix[i, :], true_nelbo_matrix[i, :])
        for i in range(score_matrix.shape[0])
    ]
    return float(np.mean(vals)) if vals else 0.5


def _build_pairwise_training_pairs(
    *,
    y_train: np.ndarray,
    q_train: np.ndarray,
    s_train: np.ndarray,
    experts_per_sample: int,
    near_tie_delta: float,
    hard_pair_fraction: float,
    random_pair_fraction: float,
    max_pairs_per_sample: int,
    max_pairs_per_domain: int,
    seed: int,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    pairs: List[Tuple[int, int]] = []
    diagnostics: List[Dict[str, Any]] = []
    domain_pair_counts: Dict[int, int] = {}

    frac_sum = max(float(hard_pair_fraction) + float(random_pair_fraction), 1e-12)
    hard_ratio = float(hard_pair_fraction) / frac_sum

    for sample_index in sorted(set(int(v) for v in s_train.tolist())):
        idxs = np.where(s_train == int(sample_index))[0]
        if int(idxs.size) != int(experts_per_sample):
            continue

        query_domain = int(q_train[idxs[0]])
        candidates: List[Tuple[int, int, float]] = []
        for i in range(int(idxs.size)):
            for j in range(i + 1, int(idxs.size)):
                ii = int(idxs[i])
                jj = int(idxs[j])
                yi = float(y_train[ii])
                yj = float(y_train[jj])
                diff = abs(yi - yj)
                if diff <= float(near_tie_delta):
                    continue
                if yi < yj:
                    candidates.append((ii, jj, diff))
                else:
                    candidates.append((jj, ii, diff))

        if not candidates:
            diagnostics.append(
                {
                    "sample_index": int(sample_index),
                    "query_domain": int(query_domain),
                    "n_candidates": 0,
                    "n_selected": 0,
                    "n_hard_selected": 0,
                    "n_random_selected": 0,
                    "dropped_by_domain_cap": 0,
                }
            )
            continue

        candidates_sorted = sorted(candidates, key=lambda x: float(x[2]))
        target = min(int(max_pairs_per_sample), len(candidates_sorted))
        n_hard = min(int(round(target * hard_ratio)), target)
        hard_selected = candidates_sorted[:n_hard]

        remaining = candidates_sorted[n_hard:]
        n_random_target = min(target - n_hard, len(remaining))
        random_selected: List[Tuple[int, int, float]] = []
        if n_random_target > 0:
            rand_idxs = rng.choice(len(remaining), size=n_random_target, replace=False)
            random_selected = [remaining[int(k)] for k in np.asarray(rand_idxs).tolist()]

        selected = hard_selected + random_selected
        dropped_by_cap = 0
        added = 0
        for better_idx, worse_idx, _diff in selected:
            cur = int(domain_pair_counts.get(query_domain, 0))
            if cur >= int(max_pairs_per_domain):
                dropped_by_cap += 1
                continue
            pairs.append((int(better_idx), int(worse_idx)))
            domain_pair_counts[query_domain] = cur + 1
            added += 1

        diagnostics.append(
            {
                "sample_index": int(sample_index),
                "query_domain": int(query_domain),
                "n_candidates": int(len(candidates_sorted)),
                "n_selected": int(added),
                "n_hard_selected": int(min(len(hard_selected), added)),
                "n_random_selected": int(max(added - min(len(hard_selected), added), 0)),
                "dropped_by_domain_cap": int(dropped_by_cap),
            }
        )

    return pairs, diagnostics


@dataclass
class _PairwiseRanker:
    seed: int
    hidden_dim: int = 128
    epochs: int = 40
    lr: float = 1e-3
    batch_size: int = 2048
    margin: float = 1.0
    device: str = "auto"
    model: torch.nn.Module | None = None

    def fit(self, x: np.ndarray, pairs: List[Tuple[int, int]]) -> None:
        if not pairs:
            raise RuntimeError("Pairwise ranker received zero training pairs")

        torch.manual_seed(int(self.seed))
        if self.device == "auto":
            run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            run_device = torch.device(self.device)

        net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], int(self.hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(self.hidden_dim), 1),
        ).to(run_device)
        opt = torch.optim.Adam(net.parameters(), lr=float(self.lr))

        x_t = torch.from_numpy(x.astype(np.float32, copy=False)).to(run_device)
        pair_t = torch.tensor(pairs, dtype=torch.long, device=run_device)

        n_pairs = int(pair_t.shape[0])
        for _ in range(int(self.epochs)):
            perm = torch.randperm(n_pairs, device=run_device)
            for i in range(0, n_pairs, int(self.batch_size)):
                idx = perm[i : i + int(self.batch_size)]
                pair_batch = pair_t[idx]

                better_x = x_t[pair_batch[:, 0]]
                worse_x = x_t[pair_batch[:, 1]]
                pred_better = net(better_x).view(-1)
                pred_worse = net(worse_x).view(-1)

                # Lower score means better expert; enforce pred_worse - pred_better >= margin.
                loss = torch.relu(float(self.margin) - (pred_worse - pred_better)).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        self.model = net.eval()

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Pairwise ranker is not fitted")
        model_device = next(self.model.parameters()).device
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32, copy=False)).to(model_device)
            pred = self.model(x_t).view(-1)
        return pred.detach().cpu().numpy().astype(np.float64, copy=False)


def _selection_metrics(
    *,
    method: str,
    query_domains: np.ndarray,
    expert_domains: Sequence[int],
    score_matrix: np.ndarray,
    true_nelbo_matrix: np.ndarray,
    tie_policy: str = "stable_expert_index",
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    if str(tie_policy).strip().lower() != "stable_expert_index":
        raise ValueError("Only tie_policy='stable_expert_index' is currently supported")

    selected_idx = _stable_argmin_indices(score_matrix)
    oracle_idx = _stable_argmin_indices(true_nelbo_matrix)

    selected_nelbo = true_nelbo_matrix[np.arange(true_nelbo_matrix.shape[0]), selected_idx]
    oracle_nelbo = true_nelbo_matrix[np.arange(true_nelbo_matrix.shape[0]), oracle_idx]
    selected_rank = _selected_rank_in_true_matrix(selected_idx, true_nelbo_matrix)

    top1 = float(np.mean(selected_idx == oracle_idx)) if selected_idx.size else 0.0
    mean_rank = float(np.mean(selected_rank)) if selected_rank.size else 0.0
    gap = selected_nelbo - oracle_nelbo
    denom = np.maximum(np.abs(oracle_nelbo), 1e-12)
    gap_pct = (gap / denom) * 100.0

    rho_vals: List[float] = []
    pair_auc_vals: List[float] = []
    for i in range(score_matrix.shape[0]):
        pair_auc_vals.append(
            float(_pairwise_auc_single(score_matrix[i, :], true_nelbo_matrix[i, :]))
        )
        rho_vals.append(
            float(
                spearman_corr(
                    (-score_matrix[i, :]).tolist(),
                    (-true_nelbo_matrix[i, :]).tolist(),
                )
            )
        )

    rows: List[Dict[str, Any]] = []
    for i in range(score_matrix.shape[0]):
        rows.append(
            {
                "sample_index": int(i),
                "query_domain": int(query_domains[i]),
                "method": method,
                "selected_expert": int(expert_domains[int(selected_idx[i])]),
                "oracle_expert": int(expert_domains[int(oracle_idx[i])]),
                "selected_nelbo": float(selected_nelbo[i]),
                "oracle_nelbo": float(oracle_nelbo[i]),
                "oracle_gap": float(gap[i]),
                "oracle_gap_pct": float(gap_pct[i]),
                "top1_oracle_hit": int(selected_idx[i] == oracle_idx[i]),
                "selected_rank": float(selected_rank[i]),
                "pairwise_auc": float(pair_auc_vals[i]),
                "spearman": float(rho_vals[i]),
            }
        )

    metrics = {
        "top1_oracle_hit": float(top1),
        "mean_rank": float(mean_rank),
        "mean_oracle_gap": float(np.mean(gap)) if gap.size else 0.0,
        "mean_oracle_gap_pct": float(np.mean(gap_pct)) if gap_pct.size else 0.0,
        "pairwise_auc": float(np.mean(pair_auc_vals)) if pair_auc_vals else 0.5,
        "spearman": float(np.mean(rho_vals)) if rho_vals else 0.0,
        "selected_nelbo": float(np.mean(selected_nelbo)) if selected_nelbo.size else 0.0,
        "oracle_nelbo": float(np.mean(oracle_nelbo)) if oracle_nelbo.size else 0.0,
    }
    return metrics, rows


def evaluate_learned_utility_loqdo(
    *,
    test_cache: Path,
    expert_checkpoints: Dict[str, str],
    hidden_dim: int,
    latent_dim: int,
    strategy: str,
    tau: float,
    seed: int,
    learned_cfg: Dict[str, Any],
    reports_dir: Path,
) -> Dict[str, Any]:
    pair_batch_size = int(learned_cfg.get("scoring", {}).get("pair_batch_size", 4096))
    include_metadata_features = bool(learned_cfg.get("pair_features", {}).get("include_metadata_features", False))
    predictors = [str(v) for v in learned_cfg.get("predictors", ["linear_regressor", "mlp_regressor"])]
    predictor_params = learned_cfg.get("predictor_params", {})
    mlp_cfg = predictor_params.get("mlp", {}) if isinstance(predictor_params, dict) else {}
    pairwise_cfg = predictor_params.get("pairwise_ranker", {}) if isinstance(predictor_params, dict) else {}
    hybrid_cfg = learned_cfg.get("hybrid_scoring", {}) if isinstance(learned_cfg, dict) else {}
    hybrid_enabled = bool((hybrid_cfg or {}).get("enabled", False))
    hybrid_alphas = [float(v) for v in (hybrid_cfg or {}).get("alphas", _DEFAULT_ALPHA_GRID)]
    primary_norm_policy = str((hybrid_cfg or {}).get("normalization_primary", "per_query_zscore")).strip().lower()
    sensitivity_norm_policy = str((hybrid_cfg or {}).get("normalization_sensitivity", "per_query_minmax")).strip().lower()
    run_sensitivity = bool((hybrid_cfg or {}).get("run_sensitivity", True))
    tie_policy = str((hybrid_cfg or {}).get("tie_policy", "stable_expert_index")).strip().lower()
    hybrid_accept_cfg = (hybrid_cfg or {}).get("acceptance", {}) if isinstance(hybrid_cfg, dict) else {}
    min_rank_improvement_abs = float((hybrid_accept_cfg or {}).get("min_mean_rank_improvement_abs", 0.05))
    min_gap_pct_improvement_abs = float(
        (hybrid_accept_cfg or {}).get("min_mean_oracle_gap_pct_improvement_abs", 0.50)
    )
    max_top1_drop_abs = float((hybrid_accept_cfg or {}).get("max_top1_drop_abs", 0.0))

    print("[learned_utility] scoring expert NELBO matrix...")
    embeddings, sample_domains, true_nelbo, expert_domains, metadata = _score_experts_batched(
        test_cache=test_cache,
        expert_checkpoints=expert_checkpoints,
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        pair_batch_size=int(pair_batch_size),
    )
    _ = metadata
    print(
        f"[learned_utility] scored matrix shape={true_nelbo.shape}, n_samples={sample_domains.shape[0]}, n_experts={len(expert_domains)}"
    )

    domain_to_idx = _domain_to_expert_index(expert_domains)

    metadata_similarity = _metadata_scores(sample_domains, expert_domains, strategy=strategy, tau=float(tau))
    latent_similarity = _latent_wasserstein_scores(
        embeddings=embeddings,
        sample_domains=sample_domains,
        expert_domains=expert_domains,
    )
    if not np.isfinite(metadata_similarity).all() or not np.isfinite(latent_similarity).all():
        raise ValueError("Metadata/latent proxy similarity matrices must be finite")

    metadata_proxy = -metadata_similarity
    latent_proxy = -latent_similarity

    oracle_proxy = true_nelbo.copy()

    method_metrics: Dict[str, Dict[str, float]] = {}
    sample_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    pair_training_rows: List[Dict[str, Any]] = []
    proxy_diag_rows: List[Dict[str, Any]] = []
    proxy_diag_rows.extend(_proxy_diagnostic_rows(metadata_similarity, sample_domains, method="metadata_similarity_raw"))
    proxy_diag_rows.extend(_proxy_diagnostic_rows(latent_similarity, sample_domains, method="latent_similarity_raw"))

    proxy_methods: List[Tuple[str, np.ndarray]] = [
        ("metadata_routing", metadata_proxy),
        ("latent_wasserstein_routing", latent_proxy),
        ("oracle_routing", oracle_proxy),
    ]
    hybrid_method_meta: Dict[str, Dict[str, Any]] = {}

    if hybrid_enabled:
        norm_policies = [primary_norm_policy]
        if run_sensitivity and sensitivity_norm_policy != primary_norm_policy:
            norm_policies.append(sensitivity_norm_policy)

        hybrid_alphas = sorted(set(float(a) for a in hybrid_alphas))
        for alpha in hybrid_alphas:
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError(f"hybrid alpha must be in [0,1], got {alpha}")

        for norm_policy in norm_policies:
            metadata_norm = _normalize_scores_per_query(metadata_similarity, policy=norm_policy)
            latent_norm = _normalize_scores_per_query(latent_similarity, policy=norm_policy)
            proxy_diag_rows.extend(
                _proxy_diagnostic_rows(metadata_norm, sample_domains, method=f"metadata_similarity_{norm_policy}")
            )
            proxy_diag_rows.extend(
                _proxy_diagnostic_rows(latent_norm, sample_domains, method=f"latent_similarity_{norm_policy}")
            )

            for alpha in hybrid_alphas:
                mixed_similarity = (float(alpha) * metadata_norm) + ((1.0 - float(alpha)) * latent_norm)
                method_name = f"hybrid_alpha_{alpha:.1f}"
                if norm_policy != primary_norm_policy:
                    method_name = f"{method_name}_{norm_policy.replace('per_query_', '')}"
                proxy_methods.append((method_name, -mixed_similarity))
                hybrid_method_meta[method_name] = {
                    "alpha": float(alpha),
                    "normalization_policy": str(norm_policy),
                }

            alpha_one = (1.0 * metadata_norm) + (0.0 * latent_norm)
            alpha_zero = (0.0 * metadata_norm) + (1.0 * latent_norm)
            if not np.allclose(alpha_one, metadata_norm, atol=1e-12, rtol=1e-9):
                raise RuntimeError("Hybrid endpoint invariant failed for alpha=1.0")
            if not np.allclose(alpha_zero, latent_norm, atol=1e-12, rtol=1e-9):
                raise RuntimeError("Hybrid endpoint invariant failed for alpha=0.0")

    hybrid_summary_rows: List[Dict[str, Any]] = []
    for name, proxy in proxy_methods:
        metrics, rows = _selection_metrics(
            method=name,
            query_domains=sample_domains,
            expert_domains=expert_domains,
            score_matrix=proxy,
            true_nelbo_matrix=true_nelbo,
            tie_policy=tie_policy,
        )
        method_metrics[name] = metrics
        sample_rows.extend(rows)
        if name in hybrid_method_meta:
            hybrid_summary_rows.append(
                {
                    "method": str(name),
                    "alpha": float(hybrid_method_meta[name]["alpha"]),
                    "normalization_policy": str(hybrid_method_meta[name]["normalization_policy"]),
                    "top1_oracle_hit": float(metrics.get("top1_oracle_hit", 0.0)),
                    "mean_rank": float(metrics.get("mean_rank", 0.0)),
                    "mean_oracle_gap": float(metrics.get("mean_oracle_gap", 0.0)),
                    "mean_oracle_gap_pct": float(metrics.get("mean_oracle_gap_pct", 0.0)),
                    "pairwise_auc": float(metrics.get("pairwise_auc", 0.5)),
                    "spearman": float(metrics.get("spearman", 0.0)),
                }
            )

    unique_query_domains = sorted(set(int(v) for v in sample_domains.tolist()))
    embedding_feature_dim = int(embeddings.shape[1])
    learned_sample_rows: List[Dict[str, Any]] = []
    learned_method_metrics: Dict[str, List[Dict[str, float]]] = {p: [] for p in predictors}

    total_folds = len(unique_query_domains)
    for fold_idx, heldout_domain in enumerate(unique_query_domains, start=1):
        fold_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        fold_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if fold_start is not None:
            fold_start.record()
        print(f"[learned_utility] fold {fold_idx}/{total_folds} heldout_query_domain={heldout_domain}...")
        train_idx = np.where(sample_domains != int(heldout_domain))[0]
        test_idx = np.where(sample_domains == int(heldout_domain))[0]
        if train_idx.size == 0 or test_idx.size == 0:
            continue

        x_train, q_train, e_train, s_train = _build_pair_features(
            sample_embeddings=embeddings,
            sample_domains=sample_domains,
            sample_indices=train_idx,
            expert_domains=expert_domains,
            include_metadata_features=include_metadata_features,
        )
        x_test, q_test, e_test, s_test = _build_pair_features(
            sample_embeddings=embeddings,
            sample_domains=sample_domains,
            sample_indices=test_idx,
            expert_domains=expert_domains,
            include_metadata_features=include_metadata_features,
        )

        y_train = true_nelbo[s_train, [domain_to_idx[int(ed)] for ed in e_train]]
        y_test = true_nelbo[s_test, [domain_to_idx[int(ed)] for ed in e_test]]

        y_train_norm = _normalize_targets_per_query(y_train, q_train)
        x_train_z, x_test_z = _zscore_features(x_train, x_test)

        test_n = int(test_idx.size)
        e_n = len(expert_domains)

        models: Dict[str, Any] = {}
        if "linear_regressor" in predictors:
            linear = _LinearRegressor(l2=1e-4)
            linear.fit(x_train_z, y_train_norm)
            models["linear_regressor"] = linear
        if "mlp_regressor" in predictors:
            mlp = _MLPRegressor(
                seed=int(seed),
                hidden_dim=int(mlp_cfg.get("hidden_dim", 128)),
                epochs=int(mlp_cfg.get("epochs", 40)),
                lr=float(mlp_cfg.get("lr", 1e-3)),
                batch_size=int(mlp_cfg.get("batch_size", 2048)),
                device=str(mlp_cfg.get("device", "auto")),
            )
            mlp.fit(x_train_z, y_train_norm)
            models["mlp_regressor"] = mlp

        if "metadata_only_regressor" in predictors:
            # Minimal metadata-only variant built from exact-domain feature and normalized difference.
            x_train_meta = x_train_z[:, -2:] if include_metadata_features else np.zeros((x_train_z.shape[0], 2), dtype=np.float64)
            x_test_meta = x_test_z[:, -2:] if include_metadata_features else np.zeros((x_test_z.shape[0], 2), dtype=np.float64)
            meta_reg = _LinearRegressor(l2=1e-4)
            meta_reg.fit(x_train_meta, y_train_norm)
            models["metadata_only_regressor"] = (meta_reg, x_test_meta)

        if "pairwise_ranker" in predictors:
            near_tie_delta = float(pairwise_cfg.get("near_tie_delta", 0.0))
            hard_pair_fraction = float(pairwise_cfg.get("hard_pair_fraction", 0.5))
            random_pair_fraction = float(pairwise_cfg.get("random_pair_fraction", 0.5))
            max_pairs_per_sample = int(pairwise_cfg.get("max_pairs_per_sample", 12))
            max_pairs_per_domain = int(pairwise_cfg.get("max_pairs_per_domain", 5000))
            run_ablations = bool(pairwise_cfg.get("run_ablations", True))

            train_pairs, pair_diags = _build_pairwise_training_pairs(
                y_train=y_train,
                q_train=q_train,
                s_train=s_train,
                experts_per_sample=e_n,
                near_tie_delta=near_tie_delta,
                hard_pair_fraction=hard_pair_fraction,
                random_pair_fraction=random_pair_fraction,
                max_pairs_per_sample=max_pairs_per_sample,
                max_pairs_per_domain=max_pairs_per_domain,
                seed=int(seed) + int(heldout_domain),
            )
            for d in pair_diags:
                pair_training_rows.append(
                    {
                        "fold_query_domain": int(heldout_domain),
                        **d,
                    }
                )

            if train_pairs:
                span = max(float(np.max(sample_domains) - np.min(sample_domains)), 1.0)
                train_abs_diff = np.abs(q_train.astype(np.float64) - e_train.astype(np.float64)) / span
                test_abs_diff = np.abs(q_test.astype(np.float64) - e_test.astype(np.float64)) / span
                train_exact = (q_train == e_train).astype(np.float64)
                test_exact = (q_test == e_test).astype(np.float64)
                train_meta = np.stack([train_abs_diff, train_exact], axis=1)
                test_meta = np.stack([test_abs_diff, test_exact], axis=1)

                expert_oh_train = x_train[:, embedding_feature_dim : embedding_feature_dim + e_n]
                expert_oh_test = x_test[:, embedding_feature_dim : embedding_feature_dim + e_n]

                latent_train = np.concatenate([x_train[:, :embedding_feature_dim], expert_oh_train], axis=1)
                latent_test = np.concatenate([x_test[:, :embedding_feature_dim], expert_oh_test], axis=1)
                latent_train_z, latent_test_z = _zscore_features(latent_train, latent_test)

                metadata_train = np.concatenate([expert_oh_train, train_meta], axis=1)
                metadata_test = np.concatenate([expert_oh_test, test_meta], axis=1)
                metadata_train_z, metadata_test_z = _zscore_features(metadata_train, metadata_test)

                combined_train = np.concatenate([latent_train, train_meta], axis=1)
                combined_test = np.concatenate([latent_test, test_meta], axis=1)
                combined_train_z, combined_test_z = _zscore_features(combined_train, combined_test)

                pair_variants: List[Tuple[str, np.ndarray, np.ndarray]]
                if run_ablations:
                    pair_variants = [
                        ("pairwise_ranker_metadata_only", metadata_train_z, metadata_test_z),
                        ("pairwise_ranker_latent_only", latent_train_z, latent_test_z),
                        ("pairwise_ranker_combined", combined_train_z, combined_test_z),
                    ]
                else:
                    pair_variants = [
                        ("pairwise_ranker", combined_train_z, combined_test_z),
                    ]

                for method_name, x_tr_variant, x_te_variant in pair_variants:
                    ranker = _PairwiseRanker(
                        seed=int(seed),
                        hidden_dim=int(pairwise_cfg.get("hidden_dim", 128)),
                        epochs=int(pairwise_cfg.get("epochs", 40)),
                        lr=float(pairwise_cfg.get("lr", 1e-3)),
                        batch_size=int(pairwise_cfg.get("batch_size", 2048)),
                        margin=float(pairwise_cfg.get("margin", 1.0)),
                        device=str(pairwise_cfg.get("device", "auto")),
                    )
                    ranker.fit(x_tr_variant, train_pairs)
                    models[method_name] = (ranker, x_te_variant)

        for method, model in models.items():
            if isinstance(model, tuple):
                reg, x_m = model
                pred = reg.predict(x_m)
            else:
                pred = model.predict(x_test_z)

            pred_matrix = pred.reshape(test_n, e_n)
            true_matrix = y_test.reshape(test_n, e_n)

            metrics, rows = _selection_metrics(
                method=method,
                query_domains=sample_domains[test_idx],
                expert_domains=expert_domains,
                score_matrix=pred_matrix,
                true_nelbo_matrix=true_matrix,
                tie_policy=tie_policy,
            )
            learned_method_metrics.setdefault(method, []).append(metrics)

            for row in rows:
                row["sample_index"] = int(test_idx[int(row["sample_index"])])
                row["fold_query_domain"] = int(heldout_domain)
                learned_sample_rows.append(row)

            for k in range(pred.shape[0]):
                pair_rows.append(
                    {
                        "fold_query_domain": int(heldout_domain),
                        "method": method,
                        "sample_index": int(s_test[k]),
                        "query_domain": int(q_test[k]),
                        "expert_domain": int(e_test[k]),
                        "predicted_score": float(pred[k]),
                        "true_nelbo": float(y_test[k]),
                    }
                )

        if fold_end is not None:
            fold_end.record()
            torch.cuda.synchronize()
            elapsed_ms = float(fold_start.elapsed_time(fold_end))
            print(f"[learned_utility] fold {fold_idx}/{total_folds} done in {elapsed_ms / 1000.0:.2f}s")
        else:
            print(f"[learned_utility] fold {fold_idx}/{total_folds} done")

    for method, fold_metrics in learned_method_metrics.items():
        if not fold_metrics:
            continue
        method_metrics[method] = {
            "top1_oracle_hit": float(np.mean([m["top1_oracle_hit"] for m in fold_metrics])),
            "mean_rank": float(np.mean([m.get("mean_rank", 0.0) for m in fold_metrics])),
            "mean_oracle_gap": float(np.mean([m["mean_oracle_gap"] for m in fold_metrics])),
            "mean_oracle_gap_pct": float(np.mean([m["mean_oracle_gap_pct"] for m in fold_metrics])),
            "pairwise_auc": float(np.mean([m.get("pairwise_auc", 0.5) for m in fold_metrics])),
            "spearman": float(np.mean([m["spearman"] for m in fold_metrics])),
            "selected_nelbo": float(np.mean([m["selected_nelbo"] for m in fold_metrics])),
            "oracle_nelbo": float(np.mean([m["oracle_nelbo"] for m in fold_metrics])),
        }

    sample_rows.extend(learned_sample_rows)

    grouped: Dict[Tuple[str, int], Dict[str, float]] = {}
    for row in sample_rows:
        key = (str(row["method"]), int(row["query_domain"]))
        acc = grouped.setdefault(
            key,
            {
                "n_samples": 0.0,
                "top1_oracle_hit": 0.0,
                "oracle_gap": 0.0,
                "oracle_gap_pct": 0.0,
                "selected_rank": 0.0,
                "pairwise_auc": 0.0,
                "spearman": 0.0,
            },
        )
        acc["n_samples"] += 1.0
        acc["top1_oracle_hit"] += float(row["top1_oracle_hit"])
        acc["oracle_gap"] += float(row["oracle_gap"])
        acc["oracle_gap_pct"] += float(row["oracle_gap_pct"])
        acc["selected_rank"] += float(row.get("selected_rank", 0.0))
        acc["pairwise_auc"] += float(row.get("pairwise_auc", 0.5))
        acc["spearman"] += float(row["spearman"])

    domain_rows: List[Dict[str, Any]] = []
    for (method, q), acc in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n_samples = max(acc["n_samples"], 1.0)
        domain_rows.append(
            {
                "method": method,
                "query_domain": int(q),
                "n_samples": int(acc["n_samples"]),
                "top1_oracle_hit": float(acc["top1_oracle_hit"] / n_samples),
                "mean_rank": float(acc["selected_rank"] / n_samples),
                "mean_oracle_gap": float(acc["oracle_gap"] / n_samples),
                "mean_oracle_gap_pct": float(acc["oracle_gap_pct"] / n_samples),
                "pairwise_auc": float(acc["pairwise_auc"] / n_samples),
                "spearman": float(acc["spearman"] / n_samples),
            }
        )

    _write_csv(reports_dir / "learned_utility_pair_predictions.csv", pair_rows)
    _write_csv(reports_dir / "learned_utility_sample_selections.csv", sample_rows)
    _write_csv(reports_dir / "learned_utility_domain_breakdown.csv", domain_rows)
    _write_csv(reports_dir / "learned_utility_pair_training_diagnostics.csv", pair_training_rows)
    _write_csv(reports_dir / "learned_utility_proxy_diagnostics.csv", proxy_diag_rows)

    hybrid_best_by_policy: Dict[str, Dict[str, Any]] = {}
    hybrid_acceptance: Dict[str, Any] = {}
    if hybrid_summary_rows:
        _write_csv(reports_dir / "learned_utility_hybrid_alpha_summary.csv", hybrid_summary_rows)
        by_policy: Dict[str, List[Dict[str, Any]]] = {}
        for row in hybrid_summary_rows:
            by_policy.setdefault(str(row["normalization_policy"]), []).append(row)
        for policy, rows in by_policy.items():
            ordered = sorted(
                rows,
                key=lambda r: (
                    float(r["mean_oracle_gap_pct"]),
                    float(r["mean_rank"]),
                    -float(r["top1_oracle_hit"]),
                ),
            )
            hybrid_best_by_policy[policy] = {
                "best_method": str(ordered[0]["method"]),
                "best_alpha": float(ordered[0]["alpha"]),
                "ranking": ordered,
            }

        metadata_metrics = method_metrics.get("metadata_routing", {})
        baseline_top1 = float(metadata_metrics.get("top1_oracle_hit", 0.0))
        baseline_rank = float(metadata_metrics.get("mean_rank", 0.0))
        baseline_gap_pct = float(metadata_metrics.get("mean_oracle_gap_pct", 0.0))

        def _with_deltas(row: Dict[str, Any]) -> Dict[str, Any]:
            cand_top1 = float(row.get("top1_oracle_hit", 0.0))
            cand_rank = float(row.get("mean_rank", 0.0))
            cand_gap_pct = float(row.get("mean_oracle_gap_pct", 0.0))
            top1_delta = cand_top1 - baseline_top1
            rank_improvement = baseline_rank - cand_rank
            gap_pct_improvement = baseline_gap_pct - cand_gap_pct
            non_inferior_top1 = top1_delta >= -float(max_top1_drop_abs)
            efficacy_ok = (
                (rank_improvement >= float(min_rank_improvement_abs))
                or (gap_pct_improvement >= float(min_gap_pct_improvement_abs))
            )
            return {
                **row,
                "delta_vs_metadata_top1": float(top1_delta),
                "improvement_vs_metadata_mean_rank": float(rank_improvement),
                "improvement_vs_metadata_mean_oracle_gap_pct": float(gap_pct_improvement),
                "non_inferior_top1": bool(non_inferior_top1),
                "meets_effect_size_gate": bool(efficacy_ok),
                "passes_acceptance_gate": bool(non_inferior_top1 and efficacy_ok),
            }

        ranked_primary = hybrid_best_by_policy.get(primary_norm_policy, {}).get("ranking", [])
        ranked_sensitivity = hybrid_best_by_policy.get(sensitivity_norm_policy, {}).get("ranking", [])
        ranked_primary_delta = [_with_deltas(r) for r in ranked_primary]

        sensitivity_by_alpha: Dict[float, Dict[str, Any]] = {
            float(r.get("alpha", -1.0)): r for r in ranked_sensitivity
        }
        primary_best = ranked_primary_delta[0] if ranked_primary_delta else None
        sensitivity_match = None
        sensitivity_consistent = False
        if primary_best is not None and run_sensitivity:
            sensitivity_match = sensitivity_by_alpha.get(float(primary_best.get("alpha", -1.0)))
            if sensitivity_match is not None:
                sensitivity_consistent = bool(
                    _with_deltas(sensitivity_match).get("passes_acceptance_gate", False)
                )

        hybrid_acceptance = {
            "thresholds": {
                "min_mean_rank_improvement_abs": float(min_rank_improvement_abs),
                "min_mean_oracle_gap_pct_improvement_abs": float(min_gap_pct_improvement_abs),
                "max_top1_drop_abs": float(max_top1_drop_abs),
            },
            "baseline_metadata": {
                "top1_oracle_hit": baseline_top1,
                "mean_rank": baseline_rank,
                "mean_oracle_gap_pct": baseline_gap_pct,
            },
            "primary_normalization_policy": str(primary_norm_policy),
            "best_primary": primary_best,
            "best_primary_sensitivity_match": sensitivity_match,
            "best_primary_passes_sensitivity_gate": bool(sensitivity_consistent),
            "primary_policy_ranking_with_deltas": ranked_primary_delta,
        }

    return {
        "metrics_by_method": method_metrics,
        "artifacts": {
            "pair_predictions": "learned_utility_pair_predictions.csv",
            "sample_selections": "learned_utility_sample_selections.csv",
            "domain_breakdown": "learned_utility_domain_breakdown.csv",
            "pair_training_diagnostics": "learned_utility_pair_training_diagnostics.csv",
            "proxy_diagnostics": "learned_utility_proxy_diagnostics.csv",
            "hybrid_alpha_summary": "learned_utility_hybrid_alpha_summary.csv" if hybrid_summary_rows else "",
        },
        "hybrid_diagnostics": {
            "enabled": bool(hybrid_enabled),
            "tie_policy": str(tie_policy),
            "best_by_normalization_policy": hybrid_best_by_policy,
            "acceptance": hybrid_acceptance,
        },
        "n_samples": int(sample_domains.shape[0]),
        "n_experts": int(len(expert_domains)),
        "expert_domains": [int(v) for v in expert_domains],
    }
