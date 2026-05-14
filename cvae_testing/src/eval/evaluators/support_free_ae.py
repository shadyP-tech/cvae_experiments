from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

from src.eval.evaluators.learned_utility_config import AutoencoderProxyConfig
from src.eval.evaluators.learned_utility_protocol import (
    FoldCandidateSet,
    MethodProtocol,
    ProtocolError,
    _protocol_row_fields,
)
from src.eval.evaluators.learned_utility_selection import (
    _selection_metrics,
    _stable_argmin_indices,
)
from src.models.feature_autoencoder import FeatureAutoencoder, reconstruction_mse_per_sample
from src.train.checkpoint_provenance import load_model_checkpoint


@dataclass(frozen=True)
class AutoencoderScoreMatrices:
    raw_mse_matrix: np.ndarray
    zscore_matrix: np.ndarray
    quality_rows: List[Dict[str, Any]]
    provenance_rows: List[Dict[str, Any]]
    overlap_rows: List[Dict[str, Any]]
    provenance: Dict[str, Any]


@dataclass(frozen=True)
class AutoencoderProxyFoldOutputs:
    sample_rows: List[Dict[str, Any]]
    proxy_diag_rows: List[Dict[str, Any]]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    import csv

    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if str(key) not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_domain_entry(
    autoencoder_artifacts: Mapping[str, Any],
    domain: int,
) -> Mapping[str, Any]:
    provenance = autoencoder_artifacts.get("provenance", {})
    domains = provenance.get("domains", {}) if isinstance(provenance, Mapping) else {}
    entry = domains.get(str(int(domain))) if isinstance(domains, Mapping) else None
    if isinstance(entry, Mapping):
        return entry
    checkpoints = autoencoder_artifacts.get("checkpoints", {})
    if isinstance(checkpoints, Mapping):
        ckpt = checkpoints.get(str(int(domain))) or checkpoints.get(f"{int(domain)}x")
        if ckpt:
            return {"checkpoint": str(ckpt), "source_domain": int(domain)}
    raise ProtocolError(f"Missing source-trained AE artifact for domain {domain}")


def _score_autoencoder(
    *,
    embeddings: np.ndarray,
    entry: Mapping[str, Any],
    cfg: AutoencoderProxyConfig,
) -> np.ndarray:
    ckpt = Path(str(entry["checkpoint"]))
    ae_cfg = dict(entry.get("autoencoder_config", {}) or {})
    input_dim = int(entry.get("input_dim", embeddings.shape[1]))
    hidden_dim = int(ae_cfg.get("hidden_dim", cfg.hidden_dim))
    latent_dim = int(ae_cfg.get("latent_dim", cfg.latent_dim))
    device = _device()
    model = FeatureAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(load_model_checkpoint(ckpt, map_location=device).model_state_dict)
    model.eval()

    x = torch.as_tensor(embeddings, dtype=torch.float32)
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(cfg.batch_size)):
            xb = x[start : start + int(cfg.batch_size)].to(device)
            chunks.append(reconstruction_mse_per_sample(model, xb).cpu())
    if not chunks:
        return np.asarray([], dtype=np.float64)
    return torch.cat(chunks, dim=0).numpy().astype(np.float64, copy=False)


def build_autoencoder_score_matrices(
    *,
    embeddings: np.ndarray,
    expert_domains: Sequence[int],
    autoencoder_artifacts: Mapping[str, Any],
    cfg: AutoencoderProxyConfig,
) -> AutoencoderScoreMatrices:
    if not bool(cfg.enabled):
        raise ProtocolError("Autoencoder scoring was requested while autoencoder_proxy.enabled=false")
    n_samples = int(embeddings.shape[0])
    n_experts = int(len(expert_domains))
    raw = np.full((n_samples, n_experts), np.nan, dtype=np.float64)
    zscore = np.full((n_samples, n_experts), np.nan, dtype=np.float64)
    quality_rows: List[Dict[str, Any]] = []
    provenance_rows: List[Dict[str, Any]] = []

    provenance = dict(autoencoder_artifacts.get("provenance", {}) or {})
    for col, domain_raw in enumerate(expert_domains):
        domain = int(domain_raw)
        entry = dict(_resolve_domain_entry(autoencoder_artifacts, domain))
        scores = _score_autoencoder(embeddings=embeddings, entry=entry, cfg=cfg)
        mean = float(
            entry.get("source_val_reconstruction_mse", entry.get("source_val_mean_recon_mse", float("nan")))
        )
        std = float(
            entry.get("source_val_reconstruction_std", entry.get("source_val_std_recon_mse", float("nan")))
        )
        eps = float(cfg.score_normalization_eps)
        denom = std if np.isfinite(std) and std > eps else eps
        raw[:, col] = scores
        zscore[:, col] = (scores - mean) / denom
        quality_rows.append(
            {
                "source_domain": int(domain),
                "source_val_reconstruction_mse_by_domain": float(mean),
                "source_val_reconstruction_std_by_domain": float(std),
                "ae_training_converged": int(entry.get("ae_training_converged", 0)),
                "ae_best_epoch": int(entry.get("ae_best_epoch", -1)),
                "ae_val_loss": float(entry.get("ae_val_loss", float("nan"))),
                "train_size": int(entry.get("train_size", 0)),
                "val_size": int(entry.get("val_size", 0)),
                "score_normalization": str(cfg.score_normalization),
                "score_normalization_eps": float(eps),
            }
        )
        provenance_rows.append(
            {
                "source_domain": int(domain),
                "checkpoint": str(entry.get("checkpoint", "")),
                "source_val_stats_source": "source_validation_cache",
                "score_normalization": str(cfg.score_normalization),
                "uses_target_support": 0,
                "uses_target_labels": 0,
                "uses_target_domain_normalization_statistics": 0,
            }
        )

    if not np.isfinite(raw).all() or not np.isfinite(zscore).all():
        raise ProtocolError("AE raw MSE/z-score matrices must be finite for all scored expert domains")

    overlap = dict(provenance.get("overlap_audit", {}) or {})
    overlap_rows = [overlap] if overlap else []
    return AutoencoderScoreMatrices(
        raw_mse_matrix=raw,
        zscore_matrix=zscore,
        quality_rows=quality_rows,
        provenance_rows=provenance_rows,
        overlap_rows=overlap_rows,
        provenance=provenance,
    )


def _metadata_selected_local_indices(metadata_similarity_eval: np.ndarray) -> np.ndarray:
    n_rows, n_cols = metadata_similarity_eval.shape
    tie_break = np.arange(n_cols, dtype=np.int64)
    out = np.zeros((n_rows,), dtype=np.int64)
    for i in range(n_rows):
        order = np.lexsort((tie_break, -metadata_similarity_eval[i, :]))
        out[i] = int(order[0])
    return out


def _margin_gated_selection(
    *,
    ae_zscore_eval: np.ndarray,
    metadata_similarity_eval: np.ndarray,
    margin_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    best = _stable_argmin_indices(ae_zscore_eval)
    metadata_local = _metadata_selected_local_indices(metadata_similarity_eval)
    selected = metadata_local.copy()
    margins = np.zeros((ae_zscore_eval.shape[0],), dtype=np.float64)
    for i in range(ae_zscore_eval.shape[0]):
        order = np.lexsort((np.arange(ae_zscore_eval.shape[1], dtype=np.int64), ae_zscore_eval[i, :]))
        best_idx = int(order[0])
        second_score = float(ae_zscore_eval[i, int(order[1])]) if ae_zscore_eval.shape[1] > 1 else float("inf")
        margin = float(second_score - float(ae_zscore_eval[i, best_idx]))
        margins[i] = margin
        if margin >= float(margin_threshold):
            selected[i] = int(best_idx)
    return selected, margins


def _ae_proxy_diag_rows(
    *,
    fold: FoldCandidateSet,
    test_idx: np.ndarray,
    sample_domains: np.ndarray,
    ae_zscore_eval: np.ndarray,
    ae_raw_eval: np.ndarray,
    true_eval: np.ndarray,
) -> List[Dict[str, Any]]:
    protocol = MethodProtocol(
        method_role="diagnostic",
        adoption_eligible=0,
        diagnostic_only=1,
        routing_uses_query_features=1,
    )
    fields = _protocol_row_fields(fold=fold, method_protocol=protocol, method="ae_reconstruction_zscore_raw")
    rows: List[Dict[str, Any]] = []
    for local_sample, sample_index in enumerate(np.asarray(test_idx, dtype=np.int64).tolist()):
        scores = ae_zscore_eval[int(local_sample)]
        order = np.lexsort((np.arange(scores.shape[0], dtype=np.int64), scores))
        ranks = np.empty((scores.shape[0],), dtype=np.int64)
        ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=np.int64)
        oracle_order = np.lexsort(
            (np.arange(true_eval.shape[1], dtype=np.int64), true_eval[int(local_sample), :])
        )
        oracle_ranks = np.empty((true_eval.shape[1],), dtype=np.int64)
        oracle_ranks[oracle_order] = np.arange(1, true_eval.shape[1] + 1, dtype=np.int64)
        for local_expert, expert_domain in enumerate(fold.candidate_expert_domains):
            rows.append(
                {
                    **fields,
                    "method": "ae_reconstruction_zscore_raw",
                    "sample_index": int(sample_index),
                    "query_domain": int(sample_domains[int(sample_index)]),
                    "expert_domain": int(expert_domain),
                    "ae_raw_mse": float(ae_raw_eval[int(local_sample), int(local_expert)]),
                    "ae_zscore": float(ae_zscore_eval[int(local_sample), int(local_expert)]),
                    "ae_rank": int(ranks[int(local_expert)]),
                    "heldout_nelbo": float(true_eval[int(local_sample), int(local_expert)]),
                    "oracle_rank": int(oracle_ranks[int(local_expert)]),
                    "score_direction": "lower_ae_zscore_is_more_source_manifold_fit",
                    "proxy_claim_boundary": "AE residual is proxy only; final utility is held-out NELBO.",
                }
            )
    return rows


def run_autoencoder_proxy_methods_for_fold(
    *,
    sample_domains: np.ndarray,
    expert_domains: Sequence[int],
    test_idx: np.ndarray,
    fold: FoldCandidateSet,
    true_eval: np.ndarray,
    global_eval: np.ndarray,
    metadata_similarity_eval: np.ndarray,
    ae_zscore_matrix: np.ndarray,
    ae_raw_mse_matrix: np.ndarray,
    margin_threshold: float,
    tie_policy: str,
) -> AutoencoderProxyFoldOutputs:
    candidate_cols = list(fold.candidate_col_indices)
    ae_zscore_eval = ae_zscore_matrix[np.asarray(test_idx, dtype=np.int64)][:, candidate_cols]
    ae_raw_eval = ae_raw_mse_matrix[np.asarray(test_idx, dtype=np.int64)][:, candidate_cols]
    if not np.isfinite(ae_zscore_eval).all() or not np.isfinite(ae_raw_eval).all():
        raise ProtocolError("AE proxy fold scores must be finite")

    proxy_diag_rows = _ae_proxy_diag_rows(
        fold=fold,
        test_idx=test_idx,
        sample_domains=sample_domains,
        ae_zscore_eval=ae_zscore_eval,
        ae_raw_eval=ae_raw_eval,
        true_eval=true_eval,
    )

    sample_rows: List[Dict[str, Any]] = []
    for method, selected_override, extra in [
        ("ae_argmin_zscore", None, {}),
        (
            "ae_argmin_margin_gated",
            _margin_gated_selection(
                ae_zscore_eval=ae_zscore_eval,
                metadata_similarity_eval=metadata_similarity_eval,
                margin_threshold=float(margin_threshold),
            ),
            {"ae_margin_threshold": float(margin_threshold)},
        ),
    ]:
        selected_idx_override = None
        margins = None
        if selected_override is not None:
            selected_idx_override, margins = selected_override
        _metrics_unused, rows = _selection_metrics(
            method=method,
            query_domains=sample_domains[test_idx],
            expert_domains=fold.candidate_expert_domains,
            score_matrix=ae_zscore_eval,
            true_nelbo_matrix=true_eval,
            fold=fold,
            global_true_nelbo_matrix=global_eval,
            global_expert_domains=expert_domains,
            tie_policy=tie_policy,
            selected_idx_override=selected_idx_override,
            ranking_score_matrix=ae_zscore_eval,
        )
        for row in rows:
            local = int(row["sample_index"])
            row["sample_index"] = int(test_idx[local])
            row["support_free_ae_proxy"] = 1
            row["score_direction"] = "lower_ae_zscore_is_more_source_manifold_fit"
            row["proxy_claim_boundary"] = "AE residual is proxy only; final utility is held-out NELBO."
            if margins is not None:
                row["ae_margin_to_second_best"] = float(margins[local])
            row.update(extra)
            sample_rows.append(row)

    return AutoencoderProxyFoldOutputs(sample_rows=sample_rows, proxy_diag_rows=proxy_diag_rows)


def write_support_free_ae_artifacts(
    *,
    reports_dir: Path,
    ae_scores: AutoencoderScoreMatrices | None,
    proxy_diag_rows: Sequence[Dict[str, Any]],
    residual_override_rows: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    if ae_scores is None:
        return {}
    reports_dir.mkdir(parents=True, exist_ok=True)
    provenance = dict(ae_scores.provenance)
    provenance.setdefault("target_support_free", 1)
    provenance.setdefault("target_support_used", 0)
    provenance.setdefault("target_labels_used", 0)
    provenance.setdefault("target_domain_normalization_statistics_used", 0)
    provenance.setdefault("target_ae_excluded", 1)
    provenance.setdefault("source_inner_self_ae_excluded", 1)
    provenance.setdefault("claim_boundary", "source-domain reconstruction fit proxy, not compatibility")
    provenance_path = reports_dir / "support_free_ae_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    _write_csv(reports_dir / "support_free_ae_proxy_diagnostics.csv", proxy_diag_rows)
    _write_csv(reports_dir / "support_free_ae_quality_diagnostics.csv", ae_scores.quality_rows)
    _write_csv(reports_dir / "support_free_ae_overlap_audit.csv", ae_scores.overlap_rows)
    plot_artifacts = _write_support_free_ae_plots(
        reports_dir=reports_dir,
        proxy_diag_rows=proxy_diag_rows,
        quality_rows=ae_scores.quality_rows,
        residual_override_rows=residual_override_rows,
    )
    return {
        "support_free_ae_provenance": "support_free_ae_provenance.json",
        "support_free_ae_proxy_diagnostics": "support_free_ae_proxy_diagnostics.csv",
        "support_free_ae_overlap_audit": "support_free_ae_overlap_audit.csv",
        "support_free_ae_quality_diagnostics": "support_free_ae_quality_diagnostics.csv",
        "support_free_ae_plots": plot_artifacts,
    }


def _write_support_free_ae_plots(
    *,
    reports_dir: Path,
    proxy_diag_rows: Sequence[Dict[str, Any]],
    quality_rows: Sequence[Dict[str, Any]],
    residual_override_rows: Sequence[Dict[str, Any]],
) -> List[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    artifacts: List[str] = []

    def _save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(reports_dir / name)
        plt.close()
        artifacts.append(name)

    ae_rows = [r for r in proxy_diag_rows if "ae_zscore" in r and "heldout_nelbo" in r]
    if ae_rows:
        plt.figure(figsize=(6, 4))
        plt.scatter(
            [float(r["ae_zscore"]) for r in ae_rows],
            [float(r["heldout_nelbo"]) for r in ae_rows],
            s=12,
            alpha=0.65,
        )
        plt.xlabel("AE z-score reconstruction MSE")
        plt.ylabel("Held-out NELBO")
        plt.title("AE proxy vs held-out CVAE utility")
        _save("support_free_ae_zscore_vs_heldout_nelbo.png")

        plt.figure(figsize=(6, 4))
        plt.scatter(
            [float(r["ae_rank"]) for r in ae_rows],
            [float(r["oracle_rank"]) for r in ae_rows],
            s=18,
            alpha=0.65,
        )
        plt.xlabel("AE rank")
        plt.ylabel("Oracle NELBO rank")
        plt.title("AE rank vs oracle rank")
        _save("support_free_ae_rank_vs_oracle_rank.png")

    if residual_override_rows:
        rows = [
            r
            for r in residual_override_rows
            if str(r.get("method", "")) in {"metadata_ae_residual_safe_override_v1", "metadata_residual_inner_selected"}
        ]
        if rows:
            plt.figure(figsize=(7, 4))
            labels = [str(r.get("fold_query_domain", "")) for r in rows]
            gains = [float(r.get("net_override_gain", 0.0)) for r in rows]
            plt.bar(labels, gains)
            plt.xlabel("Held-out domain")
            plt.ylabel("Net override gain")
            plt.title("Override gain/loss by held-out domain")
            _save("support_free_ae_override_gain_loss_distribution.png")

            x = np.arange(len(rows))
            width = 0.38
            plt.figure(figsize=(7, 4))
            plt.bar(x - width / 2, [float(r.get("harmful_override_rate", 0.0)) for r in rows], width, label="harmful")
            plt.bar(
                x + width / 2,
                [float(r.get("utility_improving_override_rate", 0.0)) for r in rows],
                width,
                label="improving",
            )
            plt.xticks(x, labels)
            plt.xlabel("Held-out domain")
            plt.ylabel("Rate among overrides")
            plt.title("Harmful vs improving overrides")
            plt.legend(loc="best")
            _save("support_free_ae_harmful_vs_improving_override_rates.png")

            plt.figure(figsize=(7, 4))
            plt.plot(labels, [float(r.get("override_rate", 0.0)) for r in rows], marker="o", label="override")
            plt.plot(labels, [float(r.get("harmful_override_rate", 0.0)) for r in rows], marker="o", label="harmful")
            plt.xlabel("Held-out domain")
            plt.ylabel("Rate")
            plt.title("Per-domain override diagnostics")
            plt.legend(loc="best")
            _save("support_free_ae_per_domain_override_diagnostics.png")

    if quality_rows:
        rows = sorted(quality_rows, key=lambda r: int(r.get("source_domain", 0)))
        labels = [str(int(r.get("source_domain", 0))) for r in rows]
        means = [float(r.get("source_val_reconstruction_mse_by_domain", 0.0)) for r in rows]
        stds = [float(r.get("source_val_reconstruction_std_by_domain", 0.0)) for r in rows]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, means, yerr=stds, capsize=3)
        plt.xlabel("Source domain")
        plt.ylabel("Validation reconstruction MSE")
        plt.title("AE source-validation quality by domain")
        _save("support_free_ae_source_validation_quality_by_domain.png")

    return artifacts
