from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml

from src.eval.evaluators.latent_compatibility import (
    compute_distance_matrices,
    compute_domain_gaussian_stats,
    distance_to_similarity,
)
from src.routing.strategies import compute_similarity
from src.torch_utils import safe_torch_load


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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


def _stable_top2_indices_desc(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = matrix.shape
    tie_break = np.arange(n_cols, dtype=np.int64)
    top1 = np.zeros((n_rows,), dtype=np.int64)
    top2 = np.full((n_rows,), -1, dtype=np.int64)
    for i in range(n_rows):
        order = np.lexsort((tie_break, -matrix[i, :]))
        top1[i] = int(order[0])
        if n_cols > 1:
            top2[i] = int(order[1])
    return top1, top2


def _metadata_similarity(
    *,
    sample_domains: np.ndarray,
    expert_domains: Sequence[int],
    strategy: str,
    tau: float,
) -> np.ndarray:
    n_samples = int(sample_domains.shape[0])
    n_experts = len(expert_domains)
    out = np.zeros((n_samples, n_experts), dtype=np.float64)
    cache: Dict[int, np.ndarray] = {}
    for i in range(n_samples):
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


def _latent_similarity(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    expert_domains: Sequence[int],
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    domain_order, stats, _warnings = compute_domain_gaussian_stats(
        embeddings=embeddings,
        domains=sample_domains,
        covariance_regularization_lambda=1e-4,
        min_samples_per_domain=5,
    )
    distances = compute_distance_matrices(
        domain_order=domain_order,
        stats=stats,
        eigenvalue_floor=1e-10,
    )
    similarity, _scale = distance_to_similarity(distances["wasserstein"], scale_floor=1e-8)

    d_to_row = {int(d): i for i, d in enumerate(domain_order)}
    q_rows = np.asarray([d_to_row[int(q)] for q in sample_domains.tolist()], dtype=np.int64)
    e_rows = np.asarray([d_to_row.get(int(ed), -1) for ed in expert_domains], dtype=np.int64)
    valid = e_rows >= 0

    out = np.full((int(sample_domains.shape[0]), len(expert_domains)), float("-inf"), dtype=np.float64)
    if np.any(valid):
        out[:, valid] = similarity[q_rows[:, None], e_rows[valid][None, :]]

    return out, domain_order, similarity


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "min": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _max_within_query_std(scores: np.ndarray, sample_domains: np.ndarray) -> float:
    max_std = 0.0
    for q in sorted(set(int(v) for v in sample_domains.tolist())):
        idx = np.where(sample_domains == int(q))[0]
        if idx.size <= 1:
            continue
        std_by_col = np.std(scores[idx, :], axis=0)
        max_std = max(max_std, float(np.max(std_by_col)))
    return float(max_std)


def _load_cfg(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose why metadata and latent proxies are selection-equivalent."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory (e.g., outputs/.../<run_id>)",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="proxy_selection_equivalence",
        help="Prefix for report files written to run_dir/reports/",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    reports_dir = run_dir / "reports"
    config_path = run_dir / "config_resolved.yaml"
    test_cache = run_dir / "embeddings" / "test.pt"
    results_path = reports_dir / "learned_utility_results.json"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_resolved.yaml: {config_path}")
    if not test_cache.exists():
        raise FileNotFoundError(f"Missing test embedding cache: {test_cache}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing learned_utility_results.json: {results_path}")

    cfg = _load_cfg(config_path)
    results = _read_json(results_path)

    strategy = str(cfg.get("routing", {}).get("strategy", "categorical_exact"))
    tau = float(cfg.get("routing", {}).get("tau", 1.0))

    expert_domains = [int(v) for v in results.get("expert_domains", [])]
    if not expert_domains:
        expert_domains = [int(v) for v in cfg.get("data", {}).get("magnifications", [])]
    if not expert_domains:
        raise ValueError("Unable to determine expert domain order")

    payload = safe_torch_load(test_cache, map_location="cpu")
    embeddings = payload["embeddings"].detach().cpu().numpy().astype(np.float64, copy=False)
    metadata = payload["metadata"]
    sample_domains = np.asarray([int(m["magnification"]) for m in metadata], dtype=np.int64)

    meta_sim = _metadata_similarity(
        sample_domains=sample_domains,
        expert_domains=expert_domains,
        strategy=strategy,
        tau=tau,
    )
    latent_sim, latent_domain_order, latent_domain_matrix = _latent_similarity(
        embeddings=embeddings,
        sample_domains=sample_domains,
        expert_domains=expert_domains,
    )

    meta_top1, meta_top2 = _stable_top2_indices_desc(meta_sim)
    latent_top1, latent_top2 = _stable_top2_indices_desc(latent_sim)

    per_sample_rows: List[Dict[str, Any]] = []
    meta_margins: List[float] = []
    latent_margins: List[float] = []
    same_selection_flags: List[int] = []
    metadata_hits_query: List[int] = []
    latent_hits_query: List[int] = []

    for i in range(sample_domains.shape[0]):
        q = int(sample_domains[i])
        m1 = int(meta_top1[i])
        m2 = int(meta_top2[i])
        l1 = int(latent_top1[i])
        l2 = int(latent_top2[i])

        meta_best_sim = float(meta_sim[i, m1])
        meta_second_sim = float(meta_sim[i, m2]) if m2 >= 0 else float("nan")
        latent_best_sim = float(latent_sim[i, l1])
        latent_second_sim = float(latent_sim[i, l2]) if l2 >= 0 else float("nan")

        meta_margin = float(meta_best_sim - meta_second_sim) if m2 >= 0 else float("nan")
        latent_margin = float(latent_best_sim - latent_second_sim) if l2 >= 0 else float("nan")

        same = int(m1 == l1)
        meta_hit_q = int(int(expert_domains[m1]) == q)
        latent_hit_q = int(int(expert_domains[l1]) == q)

        if np.isfinite(meta_margin):
            meta_margins.append(meta_margin)
        if np.isfinite(latent_margin):
            latent_margins.append(latent_margin)
        same_selection_flags.append(same)
        metadata_hits_query.append(meta_hit_q)
        latent_hits_query.append(latent_hit_q)

        per_sample_rows.append(
            {
                "sample_index": int(i),
                "query_domain": q,
                "metadata_selected_expert": int(expert_domains[m1]),
                "latent_selected_expert": int(expert_domains[l1]),
                "same_selected_expert": int(same),
                "metadata_hits_query_domain": int(meta_hit_q),
                "latent_hits_query_domain": int(latent_hit_q),
                "metadata_best_similarity": meta_best_sim,
                "metadata_second_similarity": meta_second_sim,
                "metadata_margin_best_minus_second": meta_margin,
                "latent_best_similarity": latent_best_sim,
                "latent_second_similarity": latent_second_sim,
                "latent_margin_best_minus_second": latent_margin,
            }
        )

    domain_rows: List[Dict[str, Any]] = []
    for q in sorted(set(int(v) for v in sample_domains.tolist())):
        idx = np.where(sample_domains == q)[0]
        same_rate = float(np.mean([same_selection_flags[int(i)] for i in idx.tolist()])) if idx.size else 0.0
        meta_margin_q = [float(per_sample_rows[int(i)]["metadata_margin_best_minus_second"]) for i in idx.tolist()]
        latent_margin_q = [float(per_sample_rows[int(i)]["latent_margin_best_minus_second"]) for i in idx.tolist()]

        domain_rows.append(
            {
                "query_domain": int(q),
                "n_samples": int(idx.size),
                "selection_equivalence_rate": same_rate,
                "metadata_margin_mean": float(np.mean(meta_margin_q)) if meta_margin_q else 0.0,
                "latent_margin_mean": float(np.mean(latent_margin_q)) if latent_margin_q else 0.0,
                "metadata_margin_min": float(np.min(meta_margin_q)) if meta_margin_q else 0.0,
                "latent_margin_min": float(np.min(latent_margin_q)) if latent_margin_q else 0.0,
            }
        )

    latent_diag_gap: Dict[str, float] = {}
    d_to_row = {int(d): i for i, d in enumerate(latent_domain_order)}
    for d in expert_domains:
        if int(d) not in d_to_row:
            continue
        i = int(d_to_row[int(d)])
        row = latent_domain_matrix[i, :]
        diag = float(row[i])
        offdiag = [float(row[j]) for j in range(row.shape[0]) if j != i]
        best_offdiag = max(offdiag) if offdiag else float("nan")
        latent_diag_gap[str(int(d))] = float(diag - best_offdiag) if np.isfinite(best_offdiag) else float("nan")

    summary = {
        "run_dir": str(run_dir),
        "n_samples": int(sample_domains.shape[0]),
        "n_experts": int(len(expert_domains)),
        "expert_domains": [int(v) for v in expert_domains],
        "routing_strategy": str(strategy),
        "tau": float(tau),
        "selection_equivalence_rate": float(np.mean(same_selection_flags)) if same_selection_flags else 0.0,
        "metadata_selects_query_domain_rate": float(np.mean(metadata_hits_query)) if metadata_hits_query else 0.0,
        "latent_selects_query_domain_rate": float(np.mean(latent_hits_query)) if latent_hits_query else 0.0,
        "metadata_margin_quantiles": _quantiles(meta_margins),
        "latent_margin_quantiles": _quantiles(latent_margins),
        "metadata_is_query_domain_only_proxy": bool(_max_within_query_std(meta_sim, sample_domains) < 1e-12),
        "latent_is_query_domain_only_proxy": bool(_max_within_query_std(latent_sim, sample_domains) < 1e-12),
        "latent_domain_diagonal_minus_best_offdiagonal": latent_diag_gap,
        "interpretation": {
            "headline": (
                "Selections are equivalent when both proxies deterministically map each query domain "
                "to the same best expert domain with positive top1 margin."
            ),
            "metadata_note": (
                "With categorical-exact metadata routing and one expert per domain, metadata proxy "
                "always selects the expert matching query magnification."
            ),
            "latent_note": (
                "Latent Wasserstein similarity is domain-level and diagonal-max by construction "
                "(self-distance is zero), so it also selects query-domain expert when diagonal dominates."
            ),
        },
    }

    prefix = str(args.out_prefix).strip() or "proxy_selection_equivalence"
    _write_csv(reports_dir / f"{prefix}_per_sample.csv", per_sample_rows)
    _write_csv(reports_dir / f"{prefix}_by_query_domain.csv", domain_rows)
    _write_json(reports_dir / f"{prefix}_summary.json", summary)

    print("[proxy-equivalence] wrote:")
    print(f"- {reports_dir / f'{prefix}_per_sample.csv'}")
    print(f"- {reports_dir / f'{prefix}_by_query_domain.csv'}")
    print(f"- {reports_dir / f'{prefix}_summary.json'}")
    print(
        "[proxy-equivalence] selection_equivalence_rate="
        f"{summary['selection_equivalence_rate']:.6f}, "
        "metadata_selects_query_domain_rate="
        f"{summary['metadata_selects_query_domain_rate']:.6f}, "
        "latent_selects_query_domain_rate="
        f"{summary['latent_selects_query_domain_rate']:.6f}"
    )


if __name__ == "__main__":
    main()
