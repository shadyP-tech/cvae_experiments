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


def _selection_metrics(
    *,
    method: str,
    query_domains: np.ndarray,
    expert_domains: Sequence[int],
    score_matrix: np.ndarray,
    true_nelbo_matrix: np.ndarray,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    selected_idx = np.argmin(score_matrix, axis=1)
    oracle_idx = np.argmin(true_nelbo_matrix, axis=1)

    selected_nelbo = true_nelbo_matrix[np.arange(true_nelbo_matrix.shape[0]), selected_idx]
    oracle_nelbo = true_nelbo_matrix[np.arange(true_nelbo_matrix.shape[0]), oracle_idx]

    top1 = float(np.mean(selected_idx == oracle_idx)) if selected_idx.size else 0.0
    gap = selected_nelbo - oracle_nelbo
    denom = np.maximum(np.abs(oracle_nelbo), 1e-12)
    gap_pct = (gap / denom) * 100.0

    rho_vals: List[float] = []
    for i in range(score_matrix.shape[0]):
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
                "spearman": float(rho_vals[i]),
            }
        )

    metrics = {
        "top1_oracle_hit": float(top1),
        "mean_oracle_gap": float(np.mean(gap)) if gap.size else 0.0,
        "mean_oracle_gap_pct": float(np.mean(gap_pct)) if gap_pct.size else 0.0,
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

    metadata_proxy = -_metadata_scores(sample_domains, expert_domains, strategy=strategy, tau=float(tau))
    latent_proxy = -_latent_wasserstein_scores(embeddings=embeddings, sample_domains=sample_domains, expert_domains=expert_domains)

    oracle_proxy = true_nelbo.copy()

    method_metrics: Dict[str, Dict[str, float]] = {}
    sample_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    for name, proxy in [
        ("metadata_routing", metadata_proxy),
        ("latent_wasserstein_routing", latent_proxy),
        ("oracle_routing", oracle_proxy),
    ]:
        metrics, rows = _selection_metrics(
            method=name,
            query_domains=sample_domains,
            expert_domains=expert_domains,
            score_matrix=proxy,
            true_nelbo_matrix=true_nelbo,
        )
        method_metrics[name] = metrics
        sample_rows.extend(rows)

    unique_query_domains = sorted(set(int(v) for v in sample_domains.tolist()))
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

        for method, model in models.items():
            if method == "metadata_only_regressor":
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
            )
            learned_method_metrics[method].append(metrics)

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
            "mean_oracle_gap": float(np.mean([m["mean_oracle_gap"] for m in fold_metrics])),
            "mean_oracle_gap_pct": float(np.mean([m["mean_oracle_gap_pct"] for m in fold_metrics])),
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
                "spearman": 0.0,
            },
        )
        acc["n_samples"] += 1.0
        acc["top1_oracle_hit"] += float(row["top1_oracle_hit"])
        acc["oracle_gap"] += float(row["oracle_gap"])
        acc["oracle_gap_pct"] += float(row["oracle_gap_pct"])
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
                "mean_oracle_gap": float(acc["oracle_gap"] / n_samples),
                "mean_oracle_gap_pct": float(acc["oracle_gap_pct"] / n_samples),
                "spearman": float(acc["spearman"] / n_samples),
            }
        )

    _write_csv(reports_dir / "learned_utility_pair_predictions.csv", pair_rows)
    _write_csv(reports_dir / "learned_utility_sample_selections.csv", sample_rows)
    _write_csv(reports_dir / "learned_utility_domain_breakdown.csv", domain_rows)

    return {
        "metrics_by_method": method_metrics,
        "artifacts": {
            "pair_predictions": "learned_utility_pair_predictions.csv",
            "sample_selections": "learned_utility_sample_selections.csv",
            "domain_breakdown": "learned_utility_domain_breakdown.csv",
        },
        "n_samples": int(sample_domains.shape[0]),
        "n_experts": int(len(expert_domains)),
        "expert_domains": [int(v) for v in expert_domains],
    }
