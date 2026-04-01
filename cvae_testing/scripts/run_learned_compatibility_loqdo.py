from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

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
from src.eval.metrics import spearman_corr
from src.routing.strategies import compute_similarity
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
) -> torch.Tensor:
    scores: List[torch.Tensor] = []
    with torch.no_grad():
        for ed in expert_domains:
            chunks: List[torch.Tensor] = []
            for i in range(0, int(x_cpu.shape[0]), int(batch_size)):
                xb = x_cpu[i : i + int(batch_size)].to(device)
                chunks.append(bank.score_domain_nelbo(int(ed), xb).cpu())
            scores.append(torch.cat(chunks, dim=0) if chunks else torch.empty((0,), dtype=torch.float32))
    return torch.stack(scores, dim=0)


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


def _build_pair_rows(ctx: RunContext, batch_size: int) -> Tuple[List[dict], List[int], Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
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

    score_tensor = _score_domains_batched(
        bank=bank,
        expert_domains=expert_domains,
        x_cpu=x_cpu,
        device=device,
        batch_size=batch_size,
    )

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
    for q in all_domains:
        idxs = by_query.get(q, [])
        if not idxs:
            continue

        utility_lookup[q] = {}
        nelbo_lookup[q] = {}
        query_mean = mean_by_domain.get(q)
        if query_mean is None:
            continue

        for e_i, e in enumerate(expert_domains):
            mean_nelbo = float(score_tensor[e_i, idxs].mean().item())
            utility = -mean_nelbo
            utility_lookup[q][e] = utility
            nelbo_lookup[q][e] = mean_nelbo

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
                    "oracle_nelbo": mean_nelbo,
                    "metadata_similarity": float(meta_similarity),
                    "metadata_distance": meta_distance,
                    "embedding_distance": embedding_distance,
                    "query_domain_value": (float(q) - min_d) / domain_span,
                    "expert_domain_value": (float(e) - min_d) / domain_span,
                    "abs_domain_diff": abs(float(q) - float(e)) / domain_span,
                    "is_exact_domain_match": 1.0 if int(q) == int(e) else 0.0,
                }
            )

    return rows, expert_domains, utility_lookup, nelbo_lookup


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


def _feature_matrix(rows: Sequence[dict], feature_set: str, expert_domains: Sequence[int]) -> np.ndarray:
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
        features.append(base + ext + one_hot)

    return np.asarray(features, dtype=np.float64)


def _method_scores(
    method: str,
    x_train: np.ndarray,
    y_train_norm: np.ndarray,
    x_test: np.ndarray,
    test_rows: Sequence[dict],
    train_rows: Sequence[dict],
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
) -> dict:
    norm_stats = _normalize_targets_per_query(train_rows)
    x_train = _feature_matrix(train_rows, feature_set=feature_set, expert_domains=expert_domains)
    x_test = _feature_matrix(test_rows, feature_set=feature_set, expert_domains=expert_domains)
    y_train_norm = _with_normalized_targets(train_rows, norm_stats)

    y_true = np.asarray([float(r["oracle_utility"]) for r in test_rows], dtype=np.float64)
    scores = _method_scores(
        method=method,
        x_train=x_train,
        y_train_norm=y_train_norm,
        x_test=x_test,
        test_rows=test_rows,
        train_rows=train_rows,
    )

    pred_best_idx = int(np.argmax(scores))
    true_best_idx = int(np.argmax(y_true))
    selected_e = int(test_rows[pred_best_idx]["expert_domain"])
    oracle_e = int(test_rows[true_best_idx]["expert_domain"])

    util_vec = [utility_lookup[heldout_domain][int(e)] for e in expert_domains]
    selected_rank = _row_rank_desc(util_vec, selected_idx=expert_domains.index(selected_e))

    selected_nelbo = float(nelbo_lookup[heldout_domain][selected_e])
    oracle_nelbo = float(nelbo_lookup[heldout_domain][oracle_e])

    return {
        "dataset_name": ctx.dataset_name,
        "seed": int(ctx.seed),
        "backbone_type": ctx.backbone_type,
        "run_id": ctx.run_dir.name,
        "variant": ctx.variant,
        "feature_set": feature_set,
        "method": method,
        "heldout_query_domain": int(heldout_domain),
        "n_train_rows": int(len(train_rows)),
        "n_test_rows": int(len(test_rows)),
        "n_experts": int(len(expert_domains)),
        "selected_expert": int(selected_e),
        "oracle_expert": int(oracle_e),
        "top1_agreement_with_best_expert": 1.0 if selected_e == oracle_e else 0.0,
        "spearman_similarity_vs_neg_nelbo": float(spearman_corr(scores.tolist(), y_true.tolist())),
        "metadata_to_oracle_gap": float(selected_nelbo - oracle_nelbo),
        "mean_rank_metadata_selected": float(selected_rank),
        "selected_routing_nelbo": selected_nelbo,
        "oracle_routing_nelbo": oracle_nelbo,
    }


def _evaluate_metadata_baseline(
    *,
    ctx: RunContext,
    heldout_domain: int,
    test_rows: Sequence[dict],
    expert_domains: Sequence[int],
    utility_lookup: Dict[int, Dict[int, float]],
    nelbo_lookup: Dict[int, Dict[int, float]],
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

    return {
        "dataset_name": ctx.dataset_name,
        "seed": int(ctx.seed),
        "backbone_type": ctx.backbone_type,
        "run_id": ctx.run_dir.name,
        "variant": ctx.variant,
        "feature_set": "baseline",
        "method": "metadata_routing",
        "heldout_query_domain": int(heldout_domain),
        "n_train_rows": 0,
        "n_test_rows": int(len(test_rows)),
        "n_experts": int(len(expert_domains)),
        "selected_expert": int(selected_e),
        "oracle_expert": int(oracle_e),
        "top1_agreement_with_best_expert": 1.0 if selected_e == oracle_e else 0.0,
        "spearman_similarity_vs_neg_nelbo": float(spearman_corr(sims, util_vec)),
        "metadata_to_oracle_gap": float(selected_nelbo - oracle_nelbo),
        "mean_rank_metadata_selected": float(selected_rank),
        "selected_routing_nelbo": selected_nelbo,
        "oracle_routing_nelbo": oracle_nelbo,
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
        "top1_agreement_with_best_expert",
        "spearman_similarity_vs_neg_nelbo",
        "mean_rank_metadata_selected",
    ]
    groups: Dict[Tuple[str, str, str, str, str], List[dict]] = {}
    for row in rows:
        key = (
            str(row["dataset_name"]),
            str(row["backbone_type"]),
            str(row["variant"]),
            str(row["feature_set"]),
            str(row["method"]),
        )
        groups.setdefault(key, []).append(row)

    out: List[dict] = []
    for key, vals in groups.items():
        dataset_name, backbone_type, variant, feature_set, method = key
        row = {
            "dataset_name": dataset_name,
            "backbone_type": backbone_type,
            "variant": variant,
            "feature_set": feature_set,
            "method": method,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = PROJECT_ROOT

    experiment_dirs = [Path(p) for p in args.experiment_dirs] if args.experiment_dirs else _default_experiment_dirs(workspace_root)
    resolved_runs = [_resolve_run_dir(p if p.is_absolute() else (workspace_root / p)) for p in experiment_dirs]

    methods = ["constant_mean", "expert_prior", "linear_regression"]
    if not bool(args.skip_mlp):
        methods.append("mlp_regression")

    all_fold_rows: List[dict] = []
    run_summaries: List[dict] = []

    for run_dir in resolved_runs:
        ctx = _load_run_context(run_dir, variant=args.variant)
        pair_rows, expert_domains, utility_lookup, nelbo_lookup = _build_pair_rows(ctx, batch_size=int(args.batch_size))

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

        for heldout in query_domains:
            train_rows = [r for r in pair_rows if int(r["query_domain"]) != int(heldout)]
            test_rows = [r for r in pair_rows if int(r["query_domain"]) == int(heldout)]
            if not train_rows or not test_rows:
                continue

            baseline_row = _evaluate_metadata_baseline(
                ctx=ctx,
                heldout_domain=heldout,
                test_rows=test_rows,
                expert_domains=expert_domains,
                utility_lookup=utility_lookup,
                nelbo_lookup=nelbo_lookup,
            )
            run_rows.append(baseline_row)

            for feature_set in ["A", "B"]:
                for method in methods:
                    row = _evaluate_holdout(
                        ctx=ctx,
                        heldout_domain=heldout,
                        feature_set=feature_set,
                        method=method,
                        train_rows=train_rows,
                        test_rows=test_rows,
                        expert_domains=expert_domains,
                        utility_lookup=utility_lookup,
                        nelbo_lookup=nelbo_lookup,
                    )
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
            }
        )

        with (pair_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(run_summaries[-1], f, indent=2)

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
