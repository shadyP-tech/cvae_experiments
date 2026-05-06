from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.data.metadata_conditioning import build_domain_one_hot, resolve_domain_order
from src.eval.evaluators.learned_utility_protocol import (
    FoldCandidateSet,
    MethodProtocol,
    ProtocolError,
    _AGGREGATION_SOURCE,
    _CANDIDATE_EXPERT_ORDER,
    _CANDIDATE_POLICY,
    _LEARNED_PAIR_POLICY,
    _METRIC_AGGREGATION_POLICY,
    _MIN_CANDIDATES_FOR_RANK_METRICS,
    _ORACLE_POLICY,
    _PAIRWISE_AUC_NAN_POLICY,
    _PROTOCOL_VERSION,
    _SPEARMAN_NAN_POLICY,
    _aggregate_metrics_from_sample_rows,
    _assert_method_eligibility,
    _domain_breakdown_rows,
    _method_protocol,
    _protocol_row_fields,
)
from src.eval.evaluators.latent_compatibility import (
    compute_distance_matrices,
    compute_domain_gaussian_stats,
    distance_to_similarity,
)
from src.eval.evaluators.learned_utility_selection import _selection_metrics
from src.models.cvae_expert import CVAEExpert, elbo_components
from src.routing.strategies import compute_similarity
from src.torch_utils import safe_torch_load
from src.train.checkpoint_provenance import load_model_checkpoint


_ALLOWED_NORM_POLICIES = {"per_query_zscore", "per_query_minmax"}
_DEFAULT_ALPHA_GRID = [i / 10.0 for i in range(11)]


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    return float(np.mean(vals)) if vals else 0.0


def _std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    return float(np.std(vals)) if vals else 0.0


def _empirical_p_value(observed: float, null_values: Sequence[float], higher_is_better: bool) -> float:
    vals = [float(v) for v in null_values]
    if not vals:
        return 1.0
    if higher_is_better:
        count = sum(1 for v in vals if v >= float(observed))
    else:
        count = sum(1 for v in vals if v <= float(observed))
    return float((count + 1.0) / (len(vals) + 1.0))


def _build_random_rank_floor_proxy(sample_domains: np.ndarray, n_experts: int, seed: int) -> np.ndarray:
    out = np.zeros((int(sample_domains.shape[0]), int(n_experts)), dtype=np.float64)
    if int(n_experts) <= 0:
        return out

    rank_by_query: Dict[int, np.ndarray] = {}
    for i in range(out.shape[0]):
        q = int(sample_domains[i])
        if q not in rank_by_query:
            rng = np.random.default_rng(int(seed) + (1009 * int(q)))
            perm = np.asarray(rng.permutation(int(n_experts)), dtype=np.int64)
            rank = np.zeros((int(n_experts),), dtype=np.float64)
            for pos, idx in enumerate(perm.tolist()):
                rank[int(idx)] = float(pos)
            rank_by_query[q] = rank
        out[i, :] = rank_by_query[q]
    return out


def _build_random_score_floor_proxy(n_samples: int, n_experts: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.random((int(n_samples), int(n_experts)), dtype=np.float64)


def _permute_expert_labels_proxy(proxy: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    perm = np.asarray(rng.permutation(proxy.shape[1]), dtype=np.int64)
    return proxy[:, perm]


def _maybe_plot_hist_with_observed(
    *,
    out_path: Path,
    values: Sequence[float],
    observed: float,
    title: str,
    xlabel: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    vals = np.asarray([float(v) for v in values], dtype=np.float64)
    if vals.size == 0:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=30, alpha=0.7, color="#4C78A8", edgecolor="black")
    plt.axvline(float(observed), color="#D62728", linestyle="--", linewidth=2.0, label="observed")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def _maybe_plot_overlay(
    *,
    out_path: Path,
    values_a: Sequence[float],
    label_a: str,
    values_b: Sequence[float],
    label_b: str,
    title: str,
    xlabel: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    a = np.asarray([float(v) for v in values_a], dtype=np.float64)
    b = np.asarray([float(v) for v in values_b], dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(a, bins=40, alpha=0.55, density=True, label=str(label_a), color="#4C78A8")
    plt.hist(b, bins=40, alpha=0.55, density=True, label=str(label_b), color="#F58518")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


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


def _build_pair_features(
    *,
    sample_embeddings: np.ndarray,
    sample_domains: np.ndarray,
    sample_indices: np.ndarray,
    expert_domains: Sequence[int],
    expert_id_domains: Sequence[int] | None = None,
    include_metadata_features: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    e = len(expert_domains)
    expert_id_order = [int(d) for d in (expert_id_domains or expert_domains)]
    expert_id_to_col = {int(d): idx for idx, d in enumerate(expert_id_order)}
    x_sel = sample_embeddings[sample_indices]
    q_sel = sample_domains[sample_indices]
    n = int(x_sel.shape[0])

    sample_rep = np.repeat(x_sel, repeats=e, axis=0)
    expert_oh = np.zeros((n * e, len(expert_id_order)), dtype=np.float64)
    for idx, domain in enumerate(np.tile(np.asarray([int(d) for d in expert_domains], dtype=np.int64), reps=n).tolist()):
        if int(domain) not in expert_id_to_col:
            raise ProtocolError(f"Expert domain {domain} is missing from expert_id_domains")
        expert_oh[int(idx), expert_id_to_col[int(domain)]] = 1.0
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


def _build_fold_training_pair_features(
    *,
    sample_embeddings: np.ndarray,
    sample_domains: np.ndarray,
    train_indices: np.ndarray,
    expert_domains: Sequence[int],
    outer_heldout_domain: int,
    include_metadata_features: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_parts: List[np.ndarray] = []
    q_parts: List[np.ndarray] = []
    e_parts: List[np.ndarray] = []
    s_parts: List[np.ndarray] = []
    expert_domain_set = {int(d) for d in expert_domains}
    if int(outer_heldout_domain) not in expert_domain_set:
        raise ProtocolError(f"Outer heldout domain {outer_heldout_domain} has no matching expert checkpoint")

    for query_domain in sorted(set(int(sample_domains[int(i)]) for i in train_indices.tolist())):
        if int(query_domain) not in expert_domain_set:
            raise ProtocolError(f"Training query domain {query_domain} has no matching expert checkpoint")
        domain_indices = train_indices[sample_domains[train_indices] == int(query_domain)]
        if domain_indices.size == 0:
            continue
        fold = FoldCandidateSet.for_heldout_domain(
            heldout_domain=int(outer_heldout_domain),
            expert_domains=expert_domains,
            excluded_domains=[int(query_domain)],
        )
        if not fold.candidate_expert_domains:
            raise ProtocolError(
                "learned_pair_policy left zero training candidates for "
                f"outer_heldout_domain={outer_heldout_domain}, query_domain={query_domain}"
            )
        x, q, e, s = _build_pair_features(
            sample_embeddings=sample_embeddings,
            sample_domains=sample_domains,
            sample_indices=domain_indices,
            expert_domains=fold.candidate_expert_domains,
            expert_id_domains=expert_domains,
            include_metadata_features=include_metadata_features,
        )
        x_parts.append(x)
        q_parts.append(q)
        e_parts.append(e)
        s_parts.append(s)

    if not x_parts:
        raise ProtocolError(f"No learned training samples remain for heldout_domain={outer_heldout_domain}")

    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(q_parts, axis=0),
        np.concatenate(e_parts, axis=0),
        np.concatenate(s_parts, axis=0),
    )


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
    conditioning_cfg: Dict[str, Any] | None = None,
    configured_domains: Sequence[int] | None = None,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
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

    compatibility_cfg = learned_cfg.get("compatibility_research", {}) if isinstance(learned_cfg, dict) else {}
    if compatibility_cfg is None:
        compatibility_cfg = {}
    floors_cfg = compatibility_cfg.get("floors", {}) if isinstance(compatibility_cfg, dict) else {}
    permutation_cfg = compatibility_cfg.get("permutation_tests", {}) if isinstance(compatibility_cfg, dict) else {}
    diagnostics_cfg = compatibility_cfg.get("diagnostics", {}) if isinstance(compatibility_cfg, dict) else {}
    gate_cfg = compatibility_cfg.get("gate", {}) if isinstance(compatibility_cfg, dict) else {}
    strong_gate = gate_cfg.get("strong", {}) if isinstance(gate_cfg, dict) else {}
    weak_gate = gate_cfg.get("weak", {}) if isinstance(gate_cfg, dict) else {}
    instability_gate = gate_cfg.get("instability", {}) if isinstance(gate_cfg, dict) else {}

    enable_random_rank_floor = bool((floors_cfg or {}).get("random_rank_floor", True))
    enable_random_score_floor = bool((floors_cfg or {}).get("random_score_floor", True))
    run_expert_label_permutation = bool((permutation_cfg or {}).get("expert_label_permutation", True))
    run_metadata_permutation = bool((permutation_cfg or {}).get("metadata_permutation", True))
    permutation_repeats = int((permutation_cfg or {}).get("repeats", 200))
    save_distribution_plots = bool((diagnostics_cfg or {}).get("save_distribution_plots", True))
    uplift_reference_method = str((gate_cfg or {}).get("uplift_reference_method", "metadata_routing"))

    strong_spearman_uplift = float((strong_gate or {}).get("spearman_uplift_min", 0.05))
    strong_top1_uplift = float((strong_gate or {}).get("top1_uplift_min", 0.10))
    strong_gap_reduction = float((strong_gate or {}).get("oracle_gap_pct_reduction_min", 5.0))

    weak_spearman_uplift = float((weak_gate or {}).get("spearman_uplift_min", 0.025))
    weak_top1_uplift = float((weak_gate or {}).get("top1_uplift_min", 0.05))
    weak_gap_reduction = float((weak_gate or {}).get("oracle_gap_pct_reduction_min", 2.5))

    instability_std_threshold = float((instability_gate or {}).get("std_threshold", 0.05))
    instability_sign_inconsistency_min_count = int(
        (instability_gate or {}).get("sign_inconsistency_min_count", 2)
    )

    print("[learned_utility] scoring expert NELBO matrix...")
    embeddings, sample_domains, true_nelbo, expert_domains, metadata = _score_experts_batched(
        test_cache=test_cache,
        expert_checkpoints=expert_checkpoints,
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        pair_batch_size=int(pair_batch_size),
        conditioning_cfg=conditioning_cfg,
        configured_domains=configured_domains,
        metadata_constraint_cfg=metadata_constraint_cfg,
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

    sample_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    pair_training_rows: List[Dict[str, Any]] = []
    proxy_diag_rows: List[Dict[str, Any]] = []
    hybrid_method_meta: Dict[str, Dict[str, Any]] = {}
    permutation_sample_rows: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

    unique_query_domains = sorted(set(int(v) for v in sample_domains.tolist()))
    embedding_feature_dim = int(embeddings.shape[1])
    expert_feature_dim = int(len(expert_domains))

    norm_policies: List[str] = []
    if hybrid_enabled:
        norm_policies = [primary_norm_policy]
        if run_sensitivity and sensitivity_norm_policy != primary_norm_policy:
            norm_policies.append(sensitivity_norm_policy)
        hybrid_alphas = sorted(set(float(a) for a in hybrid_alphas))
        for alpha in hybrid_alphas:
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError(f"hybrid alpha must be in [0,1], got {alpha}")

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

        fold = FoldCandidateSet.for_heldout_domain(
            heldout_domain=int(heldout_domain),
            expert_domains=expert_domains,
        )
        true_eval = fold.slice_nelbo(true_nelbo, test_idx)
        global_eval = true_nelbo[np.asarray(test_idx, dtype=np.int64)]
        metadata_similarity_eval = metadata_similarity[np.asarray(test_idx, dtype=np.int64)][:, list(fold.candidate_col_indices)]
        latent_similarity_eval = latent_similarity[np.asarray(test_idx, dtype=np.int64)][:, list(fold.candidate_col_indices)]

        for diag_method, diag_scores in [
            ("metadata_similarity_raw", metadata_similarity_eval),
            ("latent_similarity_raw", latent_similarity_eval),
        ]:
            diag_protocol = MethodProtocol(
                method_role="diagnostic",
                adoption_eligible=0,
                diagnostic_only=1,
                routing_uses_query_features=1 if diag_method.startswith("metadata") else 0,
                routing_uses_eval_domain_statistics=0 if diag_method.startswith("metadata") else 1,
            )
            for row in _proxy_diagnostic_rows(diag_scores, sample_domains[test_idx], method=diag_method):
                row.update(_protocol_row_fields(fold=fold, method_protocol=diag_protocol, method=diag_method))
                proxy_diag_rows.append(row)

        proxy_methods: List[Tuple[str, np.ndarray]] = [
            ("metadata_routing", -metadata_similarity_eval),
            ("latent_wasserstein_routing", -latent_similarity_eval),
            ("candidate_oracle_routing", true_eval),
        ]
        if enable_random_rank_floor:
            proxy_methods.append(
                (
                    "random_rank_floor",
                    _build_random_rank_floor_proxy(
                        sample_domains=sample_domains[test_idx],
                        n_experts=len(fold.candidate_expert_domains),
                        seed=int(seed) + 131 + int(heldout_domain),
                    ),
                )
            )
        if enable_random_score_floor:
            proxy_methods.append(
                (
                    "random_score_floor",
                    _build_random_score_floor_proxy(
                        n_samples=int(test_idx.shape[0]),
                        n_experts=len(fold.candidate_expert_domains),
                        seed=int(seed) + 241 + int(heldout_domain),
                    ),
                )
            )

        if hybrid_enabled:
            for norm_policy in norm_policies:
                metadata_norm = _normalize_scores_per_query(metadata_similarity_eval, policy=norm_policy)
                latent_norm = _normalize_scores_per_query(latent_similarity_eval, policy=norm_policy)
                for diag_method, diag_scores in [
                    (f"metadata_similarity_{norm_policy}", metadata_norm),
                    (f"latent_similarity_{norm_policy}", latent_norm),
                ]:
                    diag_protocol = MethodProtocol(
                        method_role="diagnostic",
                        adoption_eligible=0,
                        diagnostic_only=1,
                        routing_uses_query_features=1 if diag_method.startswith("metadata") else 0,
                        routing_uses_eval_domain_statistics=0 if diag_method.startswith("metadata") else 1,
                    )
                    for row in _proxy_diagnostic_rows(diag_scores, sample_domains[test_idx], method=diag_method):
                        row.update(_protocol_row_fields(fold=fold, method_protocol=diag_protocol, method=diag_method))
                        proxy_diag_rows.append(row)

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

                if not np.allclose(metadata_norm, (1.0 * metadata_norm) + (0.0 * latent_norm), atol=1e-12, rtol=1e-9):
                    raise RuntimeError("Hybrid endpoint invariant failed for alpha=1.0")
                if not np.allclose(latent_norm, (0.0 * metadata_norm) + (1.0 * latent_norm), atol=1e-12, rtol=1e-9):
                    raise RuntimeError("Hybrid endpoint invariant failed for alpha=0.0")

        for name, proxy in proxy_methods:
            _metrics_unused, rows = _selection_metrics(
                method=name,
                query_domains=sample_domains[test_idx],
                expert_domains=fold.candidate_expert_domains,
                score_matrix=proxy,
                true_nelbo_matrix=true_eval,
                fold=fold,
                global_true_nelbo_matrix=global_eval,
                global_expert_domains=expert_domains,
                tie_policy=tie_policy,
            )
            for row in rows:
                row["sample_index"] = int(test_idx[int(row["sample_index"])])
                sample_rows.append(row)

        if int(permutation_repeats) > 0 and (run_expert_label_permutation or run_metadata_permutation):
            metadata_proxy_eval = -metadata_similarity_eval
            for rep in range(int(permutation_repeats)):
                if run_expert_label_permutation:
                    perm_proxy = _permute_expert_labels_proxy(
                        metadata_proxy_eval,
                        seed=int(seed) + 10000 + int(rep) + int(heldout_domain),
                    )
                    _metrics_unused, rows = _selection_metrics(
                        method="expert_label_permutation",
                        query_domains=sample_domains[test_idx],
                        expert_domains=fold.candidate_expert_domains,
                        score_matrix=perm_proxy,
                        true_nelbo_matrix=true_eval,
                        fold=fold,
                        global_true_nelbo_matrix=global_eval,
                        global_expert_domains=expert_domains,
                        tie_policy=tie_policy,
                    )
                    for row in rows:
                        row["sample_index"] = int(test_idx[int(row["sample_index"])])
                    permutation_sample_rows.setdefault(("expert_label_permutation", int(rep)), []).extend(rows)

                if run_metadata_permutation:
                    rng = np.random.default_rng(int(seed) + 20000 + int(rep) + int(heldout_domain))
                    shuffled_domains = np.asarray(rng.permutation(sample_domains[test_idx]), dtype=np.int64)
                    shuffled_similarity = _metadata_scores(
                        shuffled_domains,
                        fold.candidate_expert_domains,
                        strategy=strategy,
                        tau=float(tau),
                    )
                    shuffled_proxy = -shuffled_similarity
                    _metrics_unused, rows = _selection_metrics(
                        method="metadata_permutation",
                        query_domains=sample_domains[test_idx],
                        expert_domains=fold.candidate_expert_domains,
                        score_matrix=shuffled_proxy,
                        true_nelbo_matrix=true_eval,
                        fold=fold,
                        global_true_nelbo_matrix=global_eval,
                        global_expert_domains=expert_domains,
                        tie_policy=tie_policy,
                    )
                    for row in rows:
                        row["sample_index"] = int(test_idx[int(row["sample_index"])])
                    permutation_sample_rows.setdefault(("metadata_permutation", int(rep)), []).extend(rows)

        x_train, q_train, e_train, s_train = _build_fold_training_pair_features(
            sample_embeddings=embeddings,
            sample_domains=sample_domains,
            train_indices=train_idx,
            expert_domains=expert_domains,
            outer_heldout_domain=int(heldout_domain),
            include_metadata_features=include_metadata_features,
        )
        x_test, q_test, e_test, s_test = _build_pair_features(
            sample_embeddings=embeddings,
            sample_domains=sample_domains,
            sample_indices=test_idx,
            expert_domains=fold.candidate_expert_domains,
            expert_id_domains=expert_domains,
            include_metadata_features=include_metadata_features,
        )

        y_train = true_nelbo[s_train, [domain_to_idx[int(ed)] for ed in e_train]]
        y_test = true_nelbo[s_test, [domain_to_idx[int(ed)] for ed in e_test]]

        y_train_norm = _normalize_targets_per_query(y_train, q_train)
        x_train_z, x_test_z = _zscore_features(x_train, x_test)

        test_n = int(test_idx.size)
        e_n = len(fold.candidate_expert_domains)
        train_candidates_per_sample = max(int(len(expert_domains)) - 2, 0)
        if train_candidates_per_sample < 1:
            raise ProtocolError("learned_pair_policy requires at least one non-self source candidate per source query")

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
                experts_per_sample=train_candidates_per_sample,
                near_tie_delta=near_tie_delta,
                hard_pair_fraction=hard_pair_fraction,
                random_pair_fraction=random_pair_fraction,
                max_pairs_per_sample=max_pairs_per_sample,
                max_pairs_per_domain=max_pairs_per_domain,
                seed=int(seed) + int(heldout_domain),
            )
            for d in pair_diags:
                training_diag_fold = FoldCandidateSet.for_heldout_domain(
                    heldout_domain=int(heldout_domain),
                    expert_domains=expert_domains,
                    excluded_domains=[int(d["query_domain"])],
                )
                training_diag_protocol = _protocol_row_fields(
                    fold=training_diag_fold,
                    method_protocol=_method_protocol("pairwise_ranker"),
                    method="pairwise_ranker_training_pairs",
                )
                training_diag_protocol["fold_query_domain"] = int(heldout_domain)
                training_diag_protocol["excluded_experts"] = "|".join(
                    str(int(v)) for v in sorted({int(heldout_domain), int(d["query_domain"])})
                )
                pair_training_rows.append(
                    {
                        **training_diag_protocol,
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

                expert_oh_train = x_train[:, embedding_feature_dim : embedding_feature_dim + expert_feature_dim]
                expert_oh_test = x_test[:, embedding_feature_dim : embedding_feature_dim + expert_feature_dim]

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

            expected_pred_rows = int(test_n) * int(e_n)
            if int(pred.shape[0]) != expected_pred_rows:
                raise ProtocolError(
                    f"Evaluation pair predictions for {method} have {pred.shape[0]} rows; "
                    f"expected {expected_pred_rows}"
                )
            pred_matrix = pred.reshape(test_n, e_n)
            true_matrix = y_test.reshape(test_n, e_n)

            _metrics_unused, rows = _selection_metrics(
                method=method,
                query_domains=sample_domains[test_idx],
                expert_domains=fold.candidate_expert_domains,
                score_matrix=pred_matrix,
                true_nelbo_matrix=true_matrix,
                fold=fold,
                global_true_nelbo_matrix=global_eval,
                global_expert_domains=expert_domains,
                tie_policy=tie_policy,
            )

            for row in rows:
                row["sample_index"] = int(test_idx[int(row["sample_index"])])
                sample_rows.append(row)

            row_protocol = _method_protocol(method)
            for k in range(pred.shape[0]):
                pair_rows.append(
                    {
                        **_protocol_row_fields(fold=fold, method_protocol=row_protocol, method=method),
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

    method_metrics = _aggregate_metrics_from_sample_rows(sample_rows)
    domain_rows = _domain_breakdown_rows(sample_rows)

    permutation_rows: List[Dict[str, Any]] = []
    for (null_type, rep), rows in sorted(permutation_sample_rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        perm_metrics = _aggregate_metrics_from_sample_rows(rows).get(str(null_type), {})
        permutation_rows.append(
            {
                "protocol_version": _PROTOCOL_VERSION,
                "null_type": str(null_type),
                "repeat": int(rep),
                "method_role": "control",
                "adoption_eligible": 0,
                "diagnostic_only": 0,
                "top1_oracle_hit": float(perm_metrics.get("top1_oracle_hit", float("nan"))),
                "spearman": float(perm_metrics.get("spearman", float("nan"))),
                "mean_oracle_gap_pct": float(perm_metrics.get("mean_oracle_gap_pct", float("nan"))),
                "n_samples_micro": float(perm_metrics.get("n_samples_micro", 0.0)),
                "n_query_domains_macro": float(perm_metrics.get("n_query_domains_macro", 0.0)),
            }
        )

    permutation_summary: Dict[str, Dict[str, Any]] = {}
    baseline_for_nulls = method_metrics.get("metadata_routing", {})
    baseline_top1 = float(baseline_for_nulls.get("top1_oracle_hit", 0.0))
    baseline_spearman = float(baseline_for_nulls.get("spearman", 0.0))
    baseline_gap_pct = float(baseline_for_nulls.get("mean_oracle_gap_pct", 0.0))
    random_rank_gap = float(method_metrics.get("random_rank_floor", {}).get("mean_oracle_gap_pct", 0.0))
    random_score_gap = float(method_metrics.get("random_score_floor", {}).get("mean_oracle_gap_pct", 0.0))
    for null_type in sorted(set(str(r["null_type"]) for r in permutation_rows)):
        rows = [r for r in permutation_rows if str(r["null_type"]) == null_type]
        top1_vals = [float(r["top1_oracle_hit"]) for r in rows if np.isfinite(float(r["top1_oracle_hit"]))]
        spearman_vals = [float(r["spearman"]) for r in rows if np.isfinite(float(r["spearman"]))]
        gap_vals = [float(r["mean_oracle_gap_pct"]) for r in rows if np.isfinite(float(r["mean_oracle_gap_pct"]))]
        permutation_summary[null_type] = {
            "n_repeats": int(len(rows)),
            "top1_mean": _mean(top1_vals),
            "top1_std": _std(top1_vals),
            "spearman_mean": _mean(spearman_vals),
            "spearman_std": _std(spearman_vals),
            "mean_oracle_gap_pct_mean": _mean(gap_vals),
            "mean_oracle_gap_pct_std": _std(gap_vals),
            "p_value_vs_metadata_top1": _empirical_p_value(
                observed=baseline_top1,
                null_values=top1_vals,
                higher_is_better=True,
            ),
            "p_value_vs_metadata_spearman": _empirical_p_value(
                observed=baseline_spearman,
                null_values=spearman_vals,
                higher_is_better=True,
            ),
            "p_value_vs_metadata_gap_pct": _empirical_p_value(
                observed=baseline_gap_pct,
                null_values=gap_vals,
                higher_is_better=False,
            ),
            "delta_vs_metadata_top1": float(baseline_top1 - _mean(top1_vals)),
            "delta_vs_metadata_spearman": float(baseline_spearman - _mean(spearman_vals)),
            "gap_reduction_vs_null_pct": float(_mean(gap_vals) - baseline_gap_pct),
            "delta_vs_random_rank_floor_gap_pct": float(random_rank_gap - _mean(gap_vals)),
            "delta_vs_random_score_floor_gap_pct": float(random_score_gap - _mean(gap_vals)),
        }

    hybrid_summary_rows: List[Dict[str, Any]] = []
    for method_name, meta in sorted(hybrid_method_meta.items()):
        metrics = method_metrics.get(method_name, {})
        hybrid_summary_rows.append(
            {
                "protocol_version": _PROTOCOL_VERSION,
                "method": str(method_name),
                "alpha": float(meta["alpha"]),
                "normalization_policy": str(meta["normalization_policy"]),
                "method_role": "diagnostic",
                "adoption_eligible": 0,
                "diagnostic_only": 1,
                "routing_uses_eval_domain_statistics": 1,
                "top1_oracle_hit": float(metrics.get("top1_oracle_hit", float("nan"))),
                "mean_rank": float(metrics.get("mean_rank", float("nan"))),
                "mean_oracle_gap": float(metrics.get("mean_oracle_gap", float("nan"))),
                "mean_oracle_gap_pct": float(metrics.get("mean_oracle_gap_pct", float("nan"))),
                "pairwise_auc": float(metrics.get("pairwise_auc", float("nan"))),
                "spearman": float(metrics.get("spearman", float("nan"))),
            }
        )

    _write_csv(reports_dir / "learned_utility_pair_predictions.csv", pair_rows)
    _write_csv(reports_dir / "learned_utility_sample_selections.csv", sample_rows)
    _write_csv(reports_dir / "learned_utility_domain_breakdown.csv", domain_rows)
    _write_csv(reports_dir / "learned_utility_pair_training_diagnostics.csv", pair_training_rows)
    _write_csv(reports_dir / "learned_utility_proxy_diagnostics.csv", proxy_diag_rows)
    if permutation_rows:
        _write_csv(reports_dir / "learned_utility_permutation_nulls.csv", permutation_rows)

    diagnostic_plot_artifacts: List[str] = []
    if save_distribution_plots and permutation_rows:
        for null_type in sorted(set(str(r["null_type"]) for r in permutation_rows)):
            rows = [r for r in permutation_rows if str(r["null_type"]) == null_type]
            for metric_name, observed_value, xlabel in [
                ("top1_oracle_hit", baseline_top1, "top1 oracle hit"),
                ("spearman", baseline_spearman, "spearman"),
                ("mean_oracle_gap_pct", baseline_gap_pct, "mean oracle gap percent"),
            ]:
                out_name = f"learned_utility_dist_{null_type}_{metric_name}.png"
                ok = _maybe_plot_hist_with_observed(
                    out_path=reports_dir / out_name,
                    values=[float(r[metric_name]) for r in rows],
                    observed=float(observed_value),
                    title=f"{null_type} distribution: {metric_name}",
                    xlabel=xlabel,
                )
                if ok:
                    diagnostic_plot_artifacts.append(out_name)

    candidate_methods = sorted(
        method
        for method, metrics in method_metrics.items()
        if int(float(metrics.get("adoption_eligible", 0.0))) == 1
        and str(_method_protocol(method).method_role) == "learned"
    )

    baseline_metrics = method_metrics.get(uplift_reference_method, method_metrics.get("metadata_routing", {}))
    baseline_top1_gate = float(baseline_metrics.get("top1_oracle_hit", 0.0))
    baseline_spearman_gate = float(baseline_metrics.get("spearman", 0.0))
    baseline_gap_pct_gate = float(baseline_metrics.get("mean_oracle_gap_pct", 0.0))

    seed_gate_by_method: Dict[str, Dict[str, Any]] = {}
    for method in sorted(candidate_methods):
        mm = method_metrics.get(method, {})
        top1_uplift = float(mm.get("top1_oracle_hit", 0.0)) - baseline_top1_gate
        spearman_uplift = float(mm.get("spearman", 0.0)) - baseline_spearman_gate
        gap_pct_reduction = baseline_gap_pct_gate - float(mm.get("mean_oracle_gap_pct", 0.0))

        strong_pass = bool(
            spearman_uplift >= strong_spearman_uplift
            and top1_uplift >= strong_top1_uplift
            and gap_pct_reduction >= strong_gap_reduction
        )
        weak_pass = bool(
            spearman_uplift >= weak_spearman_uplift
            and top1_uplift >= weak_top1_uplift
            and gap_pct_reduction >= weak_gap_reduction
        )

        if strong_pass:
            tier = "strong_pass_seed"
        elif weak_pass:
            tier = "weak_pass_seed"
        else:
            tier = "fail_seed"

        seed_gate_by_method[method] = {
            "tier": str(tier),
            "uplift_reference_method": str(uplift_reference_method),
            "spearman_uplift": float(spearman_uplift),
            "top1_uplift": float(top1_uplift),
            "oracle_gap_pct_reduction": float(gap_pct_reduction),
            "strong_thresholds": {
                "spearman_uplift_min": float(strong_spearman_uplift),
                "top1_uplift_min": float(strong_top1_uplift),
                "oracle_gap_pct_reduction_min": float(strong_gap_reduction),
            },
            "weak_thresholds": {
                "spearman_uplift_min": float(weak_spearman_uplift),
                "top1_uplift_min": float(weak_top1_uplift),
                "oracle_gap_pct_reduction_min": float(weak_gap_reduction),
            },
        }

    best_candidate_method = ""
    adoption_methods = sorted(
        method
        for method, metrics in method_metrics.items()
        if int(float(metrics.get("adoption_eligible", 0.0))) == 1
    )
    if adoption_methods:
        best_candidate_method = str(
            min(
                adoption_methods,
                key=lambda m: float(method_metrics.get(m, {}).get("mean_oracle_gap_pct", 1e12)),
            )
        )
    best_diagnostic_method = ""
    diagnostic_methods = sorted(
        method
        for method, metrics in method_metrics.items()
        if int(float(metrics.get("diagnostic_only", 0.0))) == 1
    )
    if diagnostic_methods:
        best_diagnostic_method = str(
            min(
                diagnostic_methods,
                key=lambda m: float(method_metrics.get(m, {}).get("mean_oracle_gap_pct", 1e12)),
            )
        )

    if save_distribution_plots and best_candidate_method:
        baseline_vals = [
            float(r["oracle_gap_pct"])
            for r in sample_rows
            if str(r.get("method", "")) == str(uplift_reference_method)
        ]
        best_vals = [
            float(r["oracle_gap_pct"])
            for r in sample_rows
            if str(r.get("method", "")) == str(best_candidate_method)
        ]
        overlay_name = "learned_utility_overlay_gap_pct_baseline_vs_best.png"
        ok_overlay = _maybe_plot_overlay(
            out_path=reports_dir / overlay_name,
            values_a=baseline_vals,
            label_a=str(uplift_reference_method),
            values_b=best_vals,
            label_b=str(best_candidate_method),
            title="Oracle gap percent: baseline vs best learned",
            xlabel="oracle gap percent",
        )
        if ok_overlay:
            diagnostic_plot_artifacts.append(overlay_name)

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
            "adoption_eligible": False,
            "not_adoption_eligible_reason": "hybrid methods use target evaluation-domain latent statistics in v2",
        }

    return {
        "metrics_by_method": method_metrics,
        "artifacts": {
            "pair_predictions": "learned_utility_pair_predictions.csv",
            "sample_selections": "learned_utility_sample_selections.csv",
            "domain_breakdown": "learned_utility_domain_breakdown.csv",
            "pair_training_diagnostics": "learned_utility_pair_training_diagnostics.csv",
            "proxy_diagnostics": "learned_utility_proxy_diagnostics.csv",
            "permutation_nulls": "learned_utility_permutation_nulls.csv" if permutation_rows else "",
            "diagnostic_plots": diagnostic_plot_artifacts,
            "hybrid_alpha_summary": "learned_utility_hybrid_alpha_summary.csv" if hybrid_summary_rows else "",
        },
        "protocol_version": _PROTOCOL_VERSION,
        "protocol_contract": {
            "protocol_version": _PROTOCOL_VERSION,
            "candidate_policy": _CANDIDATE_POLICY,
            "candidate_expert_order": _CANDIDATE_EXPERT_ORDER,
            "oracle_policy": _ORACLE_POLICY,
            "learned_pair_policy": _LEARNED_PAIR_POLICY,
            "metric_aggregation_policy": _METRIC_AGGREGATION_POLICY,
            "aggregation_source": _AGGREGATION_SOURCE,
            "global_oracle_used_for_metrics": False,
            "metrics_comparable_to_previous_protocol": False,
            "previous_protocol_invalidated_by_target_candidate_leakage": True,
            "spearman_nan_policy": _SPEARMAN_NAN_POLICY,
            "pairwise_auc_nan_policy": _PAIRWISE_AUC_NAN_POLICY,
            "min_candidates_for_rank_metrics": _MIN_CANDIDATES_FOR_RANK_METRICS,
        },
        "compatibility_protocol": {
            "uplift_reference_method": str(uplift_reference_method),
            "floors": {
                "random_rank_floor_enabled": bool(enable_random_rank_floor),
                "random_score_floor_enabled": bool(enable_random_score_floor),
            },
            "permutation_tests": {
                "expert_label_permutation": bool(run_expert_label_permutation),
                "metadata_permutation": bool(run_metadata_permutation),
                "repeats": int(permutation_repeats),
                "summary": permutation_summary,
            },
            "gate": {
                "seed_level": seed_gate_by_method,
                "strong": {
                    "spearman_uplift_min": float(strong_spearman_uplift),
                    "top1_uplift_min": float(strong_top1_uplift),
                    "oracle_gap_pct_reduction_min": float(strong_gap_reduction),
                },
                "weak": {
                    "spearman_uplift_min": float(weak_spearman_uplift),
                    "top1_uplift_min": float(weak_top1_uplift),
                    "oracle_gap_pct_reduction_min": float(weak_gap_reduction),
                },
                "instability": {
                    "std_threshold": float(instability_std_threshold),
                    "sign_inconsistency_min_count": int(instability_sign_inconsistency_min_count),
                    "note": "Instability is evaluated across seeds in aggregated decision-table stage.",
                },
            },
            "best_candidate_method_by_gap_pct": str(best_candidate_method),
            "best_diagnostic_method_by_gap_pct": str(best_diagnostic_method),
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
