from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from src.eval.evaluators.learned_utility_config import AEFirstRoutingConfig, AutoencoderProxyConfig
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
from src.eval.metrics import spearman_corr
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


@dataclass(frozen=True)
class AEFirstFoldOutputs:
    sample_rows: List[Dict[str, Any]]
    raw_rows: List[Dict[str, Any]]
    policy_audit_rows: List[Dict[str, Any]]
    source_inner_validation_rows: List[Dict[str, Any]]
    selection_diag_rows: List[Dict[str, Any]]
    margin_bin_rows: List[Dict[str, Any]]
    calibration_rows: List[Dict[str, Any]]


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
    domain_entries: Dict[int, Dict[str, Any]] = {}
    source_val_stds: List[float] = []
    for col, domain_raw in enumerate(expert_domains):
        domain = int(domain_raw)
        entry = dict(_resolve_domain_entry(autoencoder_artifacts, domain))
        domain_entries[domain] = entry
        std = float(
            entry.get("source_val_reconstruction_std", entry.get("source_val_std_recon_mse", float("nan")))
        )
        if np.isfinite(std) and std > 0.0:
            source_val_stds.append(float(std))

    sigma_floor = float(cfg.score_normalization_eps)
    if bool(cfg.ae_first.enabled) or bool(cfg.utility_calibrator.enabled):
        if str(cfg.ae_first.ae_z_sigma_floor_mode) != "global_source_val_std_quantile":
            raise ProtocolError(
                "AE-first routing currently supports ae_z_sigma_floor_mode='global_source_val_std_quantile'"
            )
        if source_val_stds:
            sigma_floor = max(
                float(np.quantile(np.asarray(source_val_stds, dtype=np.float64), cfg.ae_first.ae_z_sigma_floor_quantile)),
                float(cfg.score_normalization_eps),
            )

    for col, domain_raw in enumerate(expert_domains):
        domain = int(domain_raw)
        entry = domain_entries[domain]
        scores = _score_autoencoder(embeddings=embeddings, entry=entry, cfg=cfg)
        mean = float(
            entry.get("source_val_reconstruction_mse", entry.get("source_val_mean_recon_mse", float("nan")))
        )
        std = float(
            entry.get("source_val_reconstruction_std", entry.get("source_val_std_recon_mse", float("nan")))
        )
        eps = float(cfg.score_normalization_eps)
        floor = float(sigma_floor) if (bool(cfg.ae_first.enabled) or bool(cfg.utility_calibrator.enabled)) else eps
        denom = std if np.isfinite(std) and std > floor else floor
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
                "ae_source_val_count": int(entry.get("val_size", entry.get("source_val_count", 0))),
                "ae_z_sigma_floor": float(floor),
                "ae_z_sigma_floor_applied": int(float(denom) > float(eps) and np.isfinite(std) and float(std) < float(floor)),
                "ae_z_std_used": float(denom),
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


def _threshold_label(value: float) -> str:
    return "__inf__" if not np.isfinite(float(value)) else f"{float(value):.6g}"


def _finite_mean(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float(default)


def _ae_best_indices_and_margins(ae_zscore_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = ae_zscore_eval.shape
    best = np.zeros((n_rows,), dtype=np.int64)
    margins = np.full((n_rows,), float("inf"), dtype=np.float64)
    tie_break = np.arange(n_cols, dtype=np.int64)
    for i in range(n_rows):
        order = np.lexsort((tie_break, ae_zscore_eval[i, :]))
        best_idx = int(order[0])
        best[i] = best_idx
        if n_cols > 1:
            margins[i] = float(ae_zscore_eval[i, int(order[1])] - ae_zscore_eval[i, best_idx])
    return best, margins


def _selected_local_indices_from_scores(scores: np.ndarray, n_rows: int) -> np.ndarray:
    idx = int(_stable_argmin_indices(np.asarray(scores, dtype=np.float64).reshape(1, -1))[0])
    return np.full((int(n_rows),), idx, dtype=np.int64)


def _source_prior_scores(
    *,
    true_nelbo: np.ndarray,
    sample_domains: np.ndarray,
    expert_domains: Sequence[int],
    source_indices: np.ndarray,
    fold: FoldCandidateSet,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean source utility per candidate, excluding target and candidate-self query utility when available."""
    source_indices = np.asarray(source_indices, dtype=np.int64)
    domain_to_col = {int(domain): int(idx) for idx, domain in enumerate(expert_domains)}
    scores: List[float] = []
    counts: List[int] = []
    no_nonself_evidence: List[int] = []
    for candidate_domain in fold.candidate_expert_domains:
        col = int(domain_to_col[int(candidate_domain)])
        nonself = source_indices[np.asarray(sample_domains[source_indices] != int(candidate_domain), dtype=bool)]
        vals = true_nelbo[nonself, col] if nonself.size else np.asarray([], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            scores.append(float(np.mean(vals)))
            counts.append(int(vals.size))
            no_nonself_evidence.append(0)
            continue
        fallback_vals = true_nelbo[source_indices, col] if source_indices.size else np.asarray([], dtype=np.float64)
        fallback_vals = fallback_vals[np.isfinite(fallback_vals)]
        scores.append(float(np.mean(fallback_vals)) if fallback_vals.size else 0.0)
        counts.append(int(fallback_vals.size))
        no_nonself_evidence.append(1)
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
        np.asarray(no_nonself_evidence, dtype=np.int64),
    )


def _summary_for_selection(
    *,
    selected_idx: np.ndarray,
    true_eval: np.ndarray,
    ranking_score_matrix: np.ndarray,
    metadata_idx: np.ndarray,
    source_prior_idx: np.ndarray,
) -> Dict[str, float]:
    selected_idx = np.asarray(selected_idx, dtype=np.int64)
    metadata_idx = np.asarray(metadata_idx, dtype=np.int64)
    source_prior_idx = np.asarray(source_prior_idx, dtype=np.int64)
    oracle_idx = _stable_argmin_indices(true_eval)
    selected_nelbo = true_eval[np.arange(true_eval.shape[0]), selected_idx]
    oracle_nelbo = true_eval[np.arange(true_eval.shape[0]), oracle_idx]
    metadata_nelbo = true_eval[np.arange(true_eval.shape[0]), metadata_idx]
    source_prior_nelbo = true_eval[np.arange(true_eval.shape[0]), source_prior_idx]
    gap = selected_nelbo - oracle_nelbo
    gap_pct = (gap / np.maximum(np.abs(oracle_nelbo), 1e-12)) * 100.0
    spearman_vals: List[float] = []
    if true_eval.shape[1] >= 2:
        for i in range(true_eval.shape[0]):
            spearman_vals.append(
                float(spearman_corr((-ranking_score_matrix[i, :]).tolist(), (-true_eval[i, :]).tolist()))
            )
    metadata_gain = metadata_nelbo - selected_nelbo
    source_prior_gain = source_prior_nelbo - selected_nelbo
    return {
        "top1_oracle_hit": float(np.mean(selected_idx == oracle_idx)) if selected_idx.size else 0.0,
        "mean_oracle_gap_pct": float(np.mean(gap_pct)) if gap_pct.size else 0.0,
        "mean_oracle_gap": float(np.mean(gap)) if gap.size else 0.0,
        "spearman": _finite_mean(spearman_vals, default=0.0),
        "selected_nelbo": float(np.mean(selected_nelbo)) if selected_nelbo.size else 0.0,
        "metadata_relative_gain": float(np.mean(metadata_gain)) if metadata_gain.size else 0.0,
        "source_prior_relative_gain": float(np.mean(source_prior_gain)) if source_prior_gain.size else 0.0,
        "harmful_vs_metadata_rate": float(np.mean(selected_nelbo > metadata_nelbo)) if selected_nelbo.size else 0.0,
        "improving_vs_metadata_rate": float(np.mean(selected_nelbo < metadata_nelbo)) if selected_nelbo.size else 0.0,
        "harmful_vs_source_prior_rate": float(np.mean(selected_nelbo > source_prior_nelbo)) if selected_nelbo.size else 0.0,
        "improving_vs_source_prior_rate": float(np.mean(selected_nelbo < source_prior_nelbo)) if selected_nelbo.size else 0.0,
    }


def _oracle_ranks_for_matrix(true_eval: np.ndarray) -> np.ndarray:
    ranks = np.zeros_like(true_eval, dtype=np.int64)
    tie_break = np.arange(true_eval.shape[1], dtype=np.int64)
    for i in range(true_eval.shape[0]):
        order = np.lexsort((tie_break, true_eval[i, :]))
        ranks[i, order] = np.arange(1, true_eval.shape[1] + 1, dtype=np.int64)
    return ranks


def _proxy_ranks_for_matrix(score_matrix: np.ndarray, *, lower_is_better: bool = True) -> np.ndarray:
    ranks = np.zeros_like(score_matrix, dtype=np.int64)
    tie_break = np.arange(score_matrix.shape[1], dtype=np.int64)
    for i in range(score_matrix.shape[0]):
        primary = score_matrix[i, :] if lower_is_better else -score_matrix[i, :]
        order = np.lexsort((tie_break, primary))
        ranks[i, order] = np.arange(1, score_matrix.shape[1] + 1, dtype=np.int64)
    return ranks


def _threshold_passes_risk_gates(
    *,
    summary: Mapping[str, float],
    metadata_summary: Mapping[str, float],
    cfg: AEFirstRoutingConfig,
) -> bool:
    return bool(
        float(summary["metadata_relative_gain"]) >= 0.0
        and float(summary["source_prior_relative_gain"]) > 0.0
        and float(summary["harmful_vs_metadata_rate"]) <= float(summary["improving_vs_metadata_rate"])
        and float(summary["harmful_vs_source_prior_rate"]) <= float(summary["improving_vs_source_prior_rate"])
        and (float(summary["top1_oracle_hit"]) - float(metadata_summary["top1_oracle_hit"]))
        >= -float(cfg.max_top1_drop_abs)
        and (float(summary["spearman"]) - float(metadata_summary["spearman"]))
        >= -float(cfg.max_raw_spearman_drop_abs)
        and (float(summary["mean_oracle_gap_pct"]) - float(metadata_summary["mean_oracle_gap_pct"]))
        <= float(cfg.max_gap_pct_degradation)
    )


def _macro_summary(values: Sequence[Mapping[str, float]], keys: Sequence[str]) -> Dict[str, float]:
    return {key: _finite_mean([float(row.get(key, float("nan"))) for row in values]) for key in keys}


_AE_FIRST_SUMMARY_KEYS = (
    "top1_oracle_hit",
    "mean_oracle_gap_pct",
    "mean_oracle_gap",
    "spearman",
    "selected_nelbo",
    "metadata_relative_gain",
    "source_prior_relative_gain",
    "harmful_vs_metadata_rate",
    "improving_vs_metadata_rate",
    "harmful_vs_source_prior_rate",
    "improving_vs_source_prior_rate",
)


def _select_ae_first_threshold(
    *,
    sample_domains: np.ndarray,
    expert_domains: Sequence[int],
    train_idx: np.ndarray,
    outer_fold: FoldCandidateSet,
    true_nelbo: np.ndarray,
    metadata_similarity: np.ndarray,
    ae_zscore_matrix: np.ndarray,
    cfg: AEFirstRoutingConfig,
) -> Tuple[float, List[Dict[str, Any]]]:
    thresholds = tuple(dict.fromkeys(float(v) for v in cfg.margin_thresholds))
    if not any(not np.isfinite(v) for v in thresholds):
        thresholds = tuple(list(thresholds) + [float("inf")])

    source_domains = sorted(set(int(sample_domains[int(i)]) for i in np.asarray(train_idx, dtype=np.int64)))
    validation_rows: List[Dict[str, Any]] = []
    threshold_summaries: List[Dict[str, Any]] = []

    for tau in thresholds:
        tau_domain_summaries: List[Dict[str, float]] = []
        tau_metadata_summaries: List[Dict[str, float]] = []
        for pseudo_domain in source_domains:
            val_idx = np.asarray([i for i in train_idx.tolist() if int(sample_domains[int(i)]) == int(pseudo_domain)], dtype=np.int64)
            inner_source_idx = np.asarray(
                [i for i in train_idx.tolist() if int(sample_domains[int(i)]) != int(pseudo_domain)],
                dtype=np.int64,
            )
            if val_idx.size == 0 or inner_source_idx.size == 0:
                continue
            inner_fold = FoldCandidateSet.for_heldout_domain(
                heldout_domain=int(outer_fold.heldout_domain),
                expert_domains=expert_domains,
                excluded_domains=[int(pseudo_domain)],
            )
            if len(inner_fold.candidate_expert_domains) == 0:
                continue
            cols = list(inner_fold.candidate_col_indices)
            true_val = inner_fold.slice_nelbo(true_nelbo, val_idx)
            ae_val = ae_zscore_matrix[val_idx][:, cols]
            metadata_val = metadata_similarity[val_idx][:, cols]
            metadata_idx = _metadata_selected_local_indices(metadata_val)
            source_prior_scores, _counts, no_nonself = _source_prior_scores(
                true_nelbo=true_nelbo,
                sample_domains=sample_domains,
                expert_domains=expert_domains,
                source_indices=inner_source_idx,
                fold=inner_fold,
            )
            source_prior_idx = _selected_local_indices_from_scores(source_prior_scores, int(val_idx.shape[0]))
            ae_best_idx, margins = _ae_best_indices_and_margins(ae_val)
            selected_idx = np.where(margins >= float(tau), ae_best_idx, source_prior_idx)

            metadata_summary = _summary_for_selection(
                selected_idx=metadata_idx,
                true_eval=true_val,
                ranking_score_matrix=-metadata_val,
                metadata_idx=metadata_idx,
                source_prior_idx=source_prior_idx,
            )
            summary = _summary_for_selection(
                selected_idx=selected_idx,
                true_eval=true_val,
                ranking_score_matrix=ae_val,
                metadata_idx=metadata_idx,
                source_prior_idx=source_prior_idx,
            )
            summary["ae_coverage_rate"] = float(np.mean(margins >= float(tau))) if margins.size else 0.0
            summary["fallback_rate"] = 1.0 - float(summary["ae_coverage_rate"])
            tau_domain_summaries.append(summary)
            tau_metadata_summaries.append(metadata_summary)
            validation_rows.append(
                {
                    "method": "ae_first_margin_gated_v1",
                    "fold_query_domain": int(outer_fold.heldout_domain),
                    "pseudo_query_domain": int(pseudo_domain),
                    "tau_margin": _threshold_label(float(tau)),
                    "threshold_selection_policy": "source_inner_risk_gated_metadata_gain",
                    "n_validation_samples": int(val_idx.shape[0]),
                    "n_candidate_experts": int(len(inner_fold.candidate_expert_domains)),
                    "candidate_experts": inner_fold.label(),
                    "pseudo_query_domain_excluded": int(
                        int(pseudo_domain) not in set(int(d) for d in inner_fold.candidate_expert_domains)
                    ),
                    "heldout_target_domain_excluded": int(
                        int(outer_fold.heldout_domain)
                        not in set(int(d) for d in inner_fold.candidate_expert_domains)
                    ),
                    "source_inner_self_ae_excluded": 1,
                    "source_inner_self_expert_excluded": 1,
                    "source_prior_no_nonself_evidence": int(np.any(no_nonself)),
                    **{f"macro_{key}": float(summary[key]) for key in _AE_FIRST_SUMMARY_KEYS},
                    "ae_coverage_rate": float(summary["ae_coverage_rate"]),
                    "fallback_rate": float(summary["fallback_rate"]),
                    "metadata_top1_oracle_hit": float(metadata_summary["top1_oracle_hit"]),
                    "metadata_mean_oracle_gap_pct": float(metadata_summary["mean_oracle_gap_pct"]),
                    "metadata_spearman": float(metadata_summary["spearman"]),
                }
            )

        if not tau_domain_summaries:
            continue
        macro = _macro_summary(tau_domain_summaries, list(_AE_FIRST_SUMMARY_KEYS) + ["ae_coverage_rate", "fallback_rate"])
        metadata_macro = _macro_summary(tau_metadata_summaries, ["top1_oracle_hit", "mean_oracle_gap_pct", "spearman"])
        passes = _threshold_passes_risk_gates(summary=macro, metadata_summary=metadata_macro, cfg=cfg)
        threshold_summaries.append(
            {
                "tau_margin": float(tau),
                "passes_source_inner_risk_gates": bool(passes),
                **macro,
                "metadata_top1_oracle_hit": float(metadata_macro["top1_oracle_hit"]),
                "metadata_mean_oracle_gap_pct": float(metadata_macro["mean_oracle_gap_pct"]),
                "metadata_spearman": float(metadata_macro["spearman"]),
            }
        )

    passing = [row for row in threshold_summaries if bool(row["passes_source_inner_risk_gates"])]
    if not passing:
        selected_tau = float("inf")
    else:
        selected = sorted(
            passing,
            key=lambda row: (
                float(row["metadata_relative_gain"]),
                float(row["top1_oracle_hit"]),
                -float(row["harmful_vs_metadata_rate"]),
                float(row["tau_margin"]),
            ),
            reverse=True,
        )[0]
        selected_tau = float(selected["tau_margin"])

    for row in validation_rows:
        row["selected_tau_margin"] = _threshold_label(selected_tau)
        row["selected_by_source_inner_validation"] = int(row["tau_margin"] == _threshold_label(selected_tau))
    return selected_tau, validation_rows


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


def _add_global_sample_indices(rows: Sequence[Dict[str, Any]], test_idx: np.ndarray) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row["sample_index"] = int(test_idx[int(new_row["sample_index"])])
        out.append(new_row)
    return out


def _selection_decomposition(
    *,
    selected_idx: np.ndarray,
    ae_selected_mask: np.ndarray,
    true_eval: np.ndarray,
    source_prior_idx: np.ndarray,
    metadata_idx: np.ndarray,
) -> Dict[str, float]:
    oracle_idx = _stable_argmin_indices(true_eval)
    rows = np.arange(true_eval.shape[0])
    selected_nelbo = true_eval[rows, selected_idx]
    source_prior_nelbo = true_eval[rows, source_prior_idx]
    metadata_nelbo = true_eval[rows, metadata_idx]
    oracle_nelbo = true_eval[rows, oracle_idx]
    mask = np.asarray(ae_selected_mask, dtype=bool)
    return {
        "overall_method_nelbo": float(np.mean(selected_nelbo)) if selected_nelbo.size else 0.0,
        "fallback_only_nelbo": float(np.mean(source_prior_nelbo[~mask])) if np.any(~mask) else float("nan"),
        "ae_selected_subset_nelbo": float(np.mean(selected_nelbo[mask])) if np.any(mask) else float("nan"),
        "source_prior_on_ae_selected_subset_nelbo": float(np.mean(source_prior_nelbo[mask])) if np.any(mask) else float("nan"),
        "metadata_on_ae_selected_subset_nelbo": float(np.mean(metadata_nelbo[mask])) if np.any(mask) else float("nan"),
        "oracle_on_ae_selected_subset_nelbo": float(np.mean(oracle_nelbo[mask])) if np.any(mask) else float("nan"),
    }


def _margin_bin_rows(
    *,
    fold: FoldCandidateSet,
    margins: np.ndarray,
    selected_idx: np.ndarray,
    true_eval: np.ndarray,
    metadata_idx: np.ndarray,
    source_prior_idx: np.ndarray,
) -> List[Dict[str, Any]]:
    bins = [
        (0.0, 0.05),
        (0.05, 0.10),
        (0.10, 0.25),
        (0.25, 0.50),
        (0.50, 1.0),
        (1.0, float("inf")),
    ]
    rows = np.arange(true_eval.shape[0])
    oracle_idx = _stable_argmin_indices(true_eval)
    selected_nelbo = true_eval[rows, selected_idx]
    metadata_nelbo = true_eval[rows, metadata_idx]
    source_prior_nelbo = true_eval[rows, source_prior_idx]
    oracle_nelbo = true_eval[rows, oracle_idx]
    gap_pct = ((selected_nelbo - oracle_nelbo) / np.maximum(np.abs(oracle_nelbo), 1e-12)) * 100.0
    out: List[Dict[str, Any]] = []
    for lo, hi in bins:
        mask = (margins >= lo) & (margins < hi if np.isfinite(hi) else margins >= lo)
        out.append(
            {
                "method": "ae_first_margin_gated_v1",
                "fold_query_domain": int(fold.heldout_domain),
                "margin_bin": f"[{lo:.2f},{'inf' if not np.isfinite(hi) else f'{hi:.2f}'})",
                "n_samples": int(np.sum(mask)),
                "harmful_vs_metadata_rate": float(np.mean(selected_nelbo[mask] > metadata_nelbo[mask])) if np.any(mask) else 0.0,
                "harmful_vs_source_prior_rate": float(np.mean(selected_nelbo[mask] > source_prior_nelbo[mask])) if np.any(mask) else 0.0,
                "mean_oracle_gap_pct": float(np.mean(gap_pct[mask])) if np.any(mask) else float("nan"),
                "top1_oracle_hit": float(np.mean(selected_idx[mask] == oracle_idx[mask])) if np.any(mask) else float("nan"),
            }
        )
    return out


def run_ae_first_methods_for_fold(
    *,
    sample_domains: np.ndarray,
    expert_domains: Sequence[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: FoldCandidateSet,
    true_nelbo: np.ndarray,
    true_eval: np.ndarray,
    global_eval: np.ndarray,
    metadata_similarity: np.ndarray,
    metadata_similarity_eval: np.ndarray,
    ae_scores: AutoencoderScoreMatrices,
    cfg: AEFirstRoutingConfig,
    tie_policy: str,
) -> AEFirstFoldOutputs:
    if not bool(cfg.enabled):
        return AEFirstFoldOutputs([], [], [], [], [], [], [])
    if str(cfg.primary_method) != "ae_first_margin_gated_v1":
        raise ProtocolError("AE-first routing currently supports primary_method='ae_first_margin_gated_v1'")
    if str(cfg.fallback_baseline) != "source_prior_fallback":
        raise ProtocolError("AE-first routing currently supports fallback_baseline='source_prior_fallback'")

    candidate_cols = list(fold.candidate_col_indices)
    ae_zscore_eval = ae_scores.zscore_matrix[np.asarray(test_idx, dtype=np.int64)][:, candidate_cols]
    ae_raw_eval = ae_scores.raw_mse_matrix[np.asarray(test_idx, dtype=np.int64)][:, candidate_cols]
    if not np.isfinite(ae_zscore_eval).all() or not np.isfinite(ae_raw_eval).all():
        raise ProtocolError("AE-first fold scores must be finite")

    selected_tau, validation_rows = _select_ae_first_threshold(
        sample_domains=sample_domains,
        expert_domains=expert_domains,
        train_idx=train_idx,
        outer_fold=fold,
        true_nelbo=true_nelbo,
        metadata_similarity=metadata_similarity,
        ae_zscore_matrix=ae_scores.zscore_matrix,
        cfg=cfg,
    )

    source_prior_scores, source_prior_counts, no_nonself = _source_prior_scores(
        true_nelbo=true_nelbo,
        sample_domains=sample_domains,
        expert_domains=expert_domains,
        source_indices=train_idx,
        fold=fold,
    )
    source_prior_score_matrix = np.tile(source_prior_scores.reshape(1, -1), (int(test_idx.shape[0]), 1))
    source_prior_idx = _selected_local_indices_from_scores(source_prior_scores, int(test_idx.shape[0]))
    source_prior_expert = int(fold.candidate_expert_domains[int(source_prior_idx[0])])

    ae_best_idx, margins = _ae_best_indices_and_margins(ae_zscore_eval)
    ae_selected_mask = margins >= float(selected_tau)
    selected_idx = np.where(ae_selected_mask, ae_best_idx, source_prior_idx)
    metadata_idx = _metadata_selected_local_indices(metadata_similarity_eval)
    oracle_ranks = _oracle_ranks_for_matrix(true_eval)
    ae_ranks = _proxy_ranks_for_matrix(ae_zscore_eval, lower_is_better=True)
    metadata_ranks = _proxy_ranks_for_matrix(-metadata_similarity_eval, lower_is_better=True)
    rows_idx = np.arange(true_eval.shape[0])
    selected_nelbo = true_eval[rows_idx, selected_idx]
    metadata_nelbo = true_eval[rows_idx, metadata_idx]
    source_prior_nelbo = true_eval[rows_idx, source_prior_idx]
    sample_rows: List[Dict[str, Any]] = []
    for method, scores, override, ranking in [
        ("source_prior_fallback", source_prior_score_matrix, source_prior_idx, source_prior_score_matrix),
        ("ae_first_margin_gated_v1", ae_zscore_eval, selected_idx, ae_zscore_eval),
    ]:
        _metrics_unused, rows = _selection_metrics(
            method=method,
            query_domains=sample_domains[test_idx],
            expert_domains=fold.candidate_expert_domains,
            score_matrix=scores,
            true_nelbo_matrix=true_eval,
            fold=fold,
            global_true_nelbo_matrix=global_eval,
            global_expert_domains=expert_domains,
            tie_policy=tie_policy,
            selected_idx_override=override,
            ranking_score_matrix=ranking,
        )
        rows = _add_global_sample_indices(rows, test_idx)
        for row in rows:
            local = int(np.where(np.asarray(test_idx, dtype=np.int64) == int(row["sample_index"]))[0][0])
            ae_best_expert = int(fold.candidate_expert_domains[int(ae_best_idx[local])])
            metadata_expert = int(fold.candidate_expert_domains[int(metadata_idx[local])])
            row_selected_idx = int(override[local])
            row_selected_nelbo = float(true_eval[local, row_selected_idx])
            row_metadata_nelbo = float(metadata_nelbo[local])
            row_source_prior_nelbo = float(source_prior_nelbo[local])
            row.update(
                {
                    "policy_name": str(method),
                    "threshold_selection_policy": "source_inner_risk_gated_metadata_gain",
                    "selection_source": "source_inner_only",
                    "support_free_ae_proxy": 1,
                    "target_support_free": 1,
                    "target_support_used": 0,
                    "target_ae_excluded": 1,
                    "source_inner_self_ae_excluded": 1,
                    "source_inner_self_expert_excluded": 1,
                    "metadata_role": "auxiliary_only",
                    "fallback_baseline": "source_prior_fallback",
                    "source_prior_fallback_expert": int(source_prior_expert),
                    "source_prior_score_by_expert_json": json.dumps(
                        {
                            str(int(domain)): float(source_prior_scores[i])
                            for i, domain in enumerate(fold.candidate_expert_domains)
                        },
                        sort_keys=True,
                    ),
                    "source_prior_evidence_count_by_expert_json": json.dumps(
                        {
                            str(int(domain)): int(source_prior_counts[i])
                            for i, domain in enumerate(fold.candidate_expert_domains)
                        },
                        sort_keys=True,
                    ),
                    "source_prior_no_nonself_evidence": int(np.any(no_nonself)),
                    "selected_tau_margin": _threshold_label(float(selected_tau)),
                    "tau_margin": _threshold_label(float(selected_tau)),
                    "ae_margin": float(margins[local]),
                    "ae_coverage_rate": float(np.mean(ae_selected_mask)) if ae_selected_mask.size else 0.0,
                    "fallback_rate": 1.0 - (float(np.mean(ae_selected_mask)) if ae_selected_mask.size else 0.0),
                    "ae_selected_by_gate": int(bool(ae_selected_mask[local])),
                    "fallback_used": int(not bool(ae_selected_mask[local])),
                    "ae_best_expert": int(ae_best_expert),
                    "metadata_selected_expert": int(metadata_expert),
                    "metadata_rank_of_ae_best": int(metadata_ranks[local, int(ae_best_idx[local])]),
                    "metadata_agrees_with_ae_best": int(int(metadata_expert) == int(ae_best_expert)),
                    "metadata_margin": float(
                        np.sort(metadata_similarity_eval[local, :])[-1]
                        - (np.sort(metadata_similarity_eval[local, :])[-2] if metadata_similarity_eval.shape[1] > 1 else 0.0)
                    ),
                    "metadata_relative_gain": float(row_metadata_nelbo - row_selected_nelbo),
                    "source_prior_relative_gain": float(row_source_prior_nelbo - row_selected_nelbo),
                    "harmful_vs_metadata": int(row_selected_nelbo > row_metadata_nelbo),
                    "improving_vs_metadata": int(row_selected_nelbo < row_metadata_nelbo),
                    "harmful_vs_source_prior": int(row_selected_nelbo > row_source_prior_nelbo),
                    "improving_vs_source_prior": int(row_selected_nelbo < row_source_prior_nelbo),
                    "oracle_rank_of_ae_best": int(oracle_ranks[local, int(ae_best_idx[local])]),
                    "ae_rank_of_selected": int(ae_ranks[local, int(selected_idx[local])]),
                    "score_direction": "lower_ae_zscore_is_more_source_manifold_fit",
                    "proxy_claim_boundary": "AE reconstruction fit is a proxy for CVAE utility, not compatibility.",
                }
            )
            if method == "source_prior_fallback":
                row["selected_by_inner_validation"] = 0
                row["ae_selected_by_gate"] = 0
                row["fallback_used"] = 1
            else:
                row["selected_by_inner_validation"] = 1
            sample_rows.append(row)

    raw_rows: List[Dict[str, Any]] = []
    for local, sample_index in enumerate(np.asarray(test_idx, dtype=np.int64).tolist()):
        for local_expert, expert_domain in enumerate(fold.candidate_expert_domains):
            raw_rows.append(
                {
                    "method": "ae_first_margin_gated_v1",
                    "fold_query_domain": int(fold.heldout_domain),
                    "sample_index": int(sample_index),
                    "query_domain": int(sample_domains[int(sample_index)]),
                    "expert_domain": int(expert_domain),
                    "selected_tau_margin": _threshold_label(float(selected_tau)),
                    "ae_raw_mse": float(ae_raw_eval[local, local_expert]),
                    "ae_zscore": float(ae_zscore_eval[local, local_expert]),
                    "ae_rank": int(ae_ranks[local, local_expert]),
                    "oracle_rank": int(oracle_ranks[local, local_expert]),
                    "oracle_rank_of_ae_best": int(oracle_ranks[local, int(ae_best_idx[local])]),
                    "heldout_nelbo": float(true_eval[local, local_expert]),
                    "ae_margin": float(margins[local]),
                    "ae_best_expert": int(fold.candidate_expert_domains[int(ae_best_idx[local])]),
                    "source_prior_fallback_expert": int(source_prior_expert),
                    "metadata_selected_expert": int(fold.candidate_expert_domains[int(metadata_idx[local])]),
                    "selected_expert": int(fold.candidate_expert_domains[int(selected_idx[local])]),
                    "ae_selected_by_gate": int(bool(ae_selected_mask[local])),
                    "target_ae_excluded": 1,
                    "source_inner_self_ae_excluded": 1,
                    "proxy_claim_boundary": "AE reconstruction fit is a proxy for CVAE utility, not compatibility.",
                }
            )

    summary = _summary_for_selection(
        selected_idx=selected_idx,
        true_eval=true_eval,
        ranking_score_matrix=ae_zscore_eval,
        metadata_idx=metadata_idx,
        source_prior_idx=source_prior_idx,
    )
    metadata_summary = _summary_for_selection(
        selected_idx=metadata_idx,
        true_eval=true_eval,
        ranking_score_matrix=-metadata_similarity_eval,
        metadata_idx=metadata_idx,
        source_prior_idx=source_prior_idx,
    )
    source_prior_summary = _summary_for_selection(
        selected_idx=source_prior_idx,
        true_eval=true_eval,
        ranking_score_matrix=source_prior_score_matrix,
        metadata_idx=metadata_idx,
        source_prior_idx=source_prior_idx,
    )
    oracle_rank_of_ae_best = oracle_ranks[rows_idx, ae_best_idx]
    decomposition = _selection_decomposition(
        selected_idx=selected_idx,
        ae_selected_mask=ae_selected_mask,
        true_eval=true_eval,
        source_prior_idx=source_prior_idx,
        metadata_idx=metadata_idx,
    )
    ae_coverage = float(np.mean(ae_selected_mask)) if ae_selected_mask.size else 0.0
    margin_bin_rows = _margin_bin_rows(
        fold=fold,
        margins=margins,
        selected_idx=selected_idx,
        true_eval=true_eval,
        metadata_idx=metadata_idx,
        source_prior_idx=source_prior_idx,
    )
    policy_row = {
        "method": "ae_first_margin_gated_v1",
        "fold_query_domain": int(fold.heldout_domain),
        "query_domain": int(fold.heldout_domain),
        "aggregation_unit": "seed_x_heldout_domain_x_query_domain",
        "primary_aggregation": "macro_by_domain",
        "selected_tau_margin": _threshold_label(float(selected_tau)),
        "threshold_selection_policy": "source_inner_risk_gated_metadata_gain",
        "selection_source": "source_inner_only",
        "ae_coverage_rate": float(ae_coverage),
        "fallback_rate": float(1.0 - ae_coverage),
        "source_prior_fallback_expert": int(source_prior_expert),
        "metadata_top1_oracle_hit": float(metadata_summary["top1_oracle_hit"]),
        "metadata_mean_oracle_gap_pct": float(metadata_summary["mean_oracle_gap_pct"]),
        "metadata_spearman": float(metadata_summary["spearman"]),
        "source_prior_top1_oracle_hit": float(source_prior_summary["top1_oracle_hit"]),
        "source_prior_mean_oracle_gap_pct": float(source_prior_summary["mean_oracle_gap_pct"]),
        "source_prior_spearman": float(source_prior_summary["spearman"]),
        **{key: float(summary[key]) for key in _AE_FIRST_SUMMARY_KEYS},
        **decomposition,
        "p_oracle_rank_of_ae_best_eq_1": float(np.mean(oracle_rank_of_ae_best == 1)) if oracle_rank_of_ae_best.size else 0.0,
        "p_oracle_rank_of_ae_best_leq_2": float(np.mean(oracle_rank_of_ae_best <= 2)) if oracle_rank_of_ae_best.size else 0.0,
        "mean_oracle_rank_of_ae_best": float(np.mean(oracle_rank_of_ae_best)) if oracle_rank_of_ae_best.size else 0.0,
        "target_ae_excluded": 1,
        "source_inner_self_ae_excluded": 1,
        "metadata_role": "auxiliary_only",
    }
    selection_diag_rows = [dict(policy_row)]

    quality_by_domain = {int(r["source_domain"]): r for r in ae_scores.quality_rows}
    calibration_rows: List[Dict[str, Any]] = []
    for local_expert, expert_domain in enumerate(fold.candidate_expert_domains):
        q = quality_by_domain.get(int(expert_domain), {})
        target_z = ae_zscore_eval[:, local_expert]
        calibration_rows.append(
            {
                "method": "ae_first_margin_gated_v1",
                "fold_query_domain": int(fold.heldout_domain),
                "source_domain": int(expert_domain),
                "ae_source_val_mean": float(q.get("source_val_reconstruction_mse_by_domain", float("nan"))),
                "ae_source_val_std": float(q.get("source_val_reconstruction_std_by_domain", float("nan"))),
                "ae_source_val_count": int(q.get("ae_source_val_count", q.get("val_size", 0))),
                "ae_z_sigma_floor": float(q.get("ae_z_sigma_floor", float("nan"))),
                "ae_z_sigma_floor_applied": int(q.get("ae_z_sigma_floor_applied", 0)),
                "ae_target_z_min": float(np.min(target_z)) if target_z.size else float("nan"),
                "ae_target_z_max": float(np.max(target_z)) if target_z.size else float("nan"),
                "ae_target_z_outlier_rate": float(np.mean(np.abs(target_z) > 3.0)) if target_z.size else 0.0,
                "target_ae_excluded": 1,
            }
        )

    return AEFirstFoldOutputs(
        sample_rows=sample_rows,
        raw_rows=raw_rows,
        policy_audit_rows=[policy_row],
        source_inner_validation_rows=validation_rows,
        selection_diag_rows=selection_diag_rows,
        margin_bin_rows=margin_bin_rows,
        calibration_rows=calibration_rows,
    )


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


def write_ae_first_artifacts(
    *,
    reports_dir: Path,
    raw_rows: Sequence[Dict[str, Any]],
    policy_audit_rows: Sequence[Dict[str, Any]],
    source_inner_validation_rows: Sequence[Dict[str, Any]],
    selection_diag_rows: Sequence[Dict[str, Any]],
    margin_bin_rows: Sequence[Dict[str, Any]],
    calibration_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not (
        raw_rows
        or policy_audit_rows
        or source_inner_validation_rows
        or selection_diag_rows
        or margin_bin_rows
        or calibration_rows
    ):
        return {}
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(reports_dir / "ae_first_raw.csv", raw_rows)
    _write_csv(reports_dir / "ae_first_policy_audit.csv", policy_audit_rows)
    _write_csv(reports_dir / "ae_first_source_inner_validation.csv", source_inner_validation_rows)
    _write_csv(reports_dir / "ae_first_selection_diagnostics.csv", selection_diag_rows)
    _write_csv(reports_dir / "ae_first_domain_breakdown.csv", policy_audit_rows)
    _write_csv(reports_dir / "ae_first_margin_bins.csv", margin_bin_rows)
    _write_csv(reports_dir / "ae_first_calibration_diagnostics.csv", calibration_rows)
    plot_artifacts = _write_ae_first_plots(
        reports_dir=reports_dir,
        raw_rows=raw_rows,
        policy_audit_rows=policy_audit_rows,
        margin_bin_rows=margin_bin_rows,
        calibration_rows=calibration_rows,
    )
    return {
        "ae_first_raw": "ae_first_raw.csv",
        "ae_first_policy_audit": "ae_first_policy_audit.csv",
        "ae_first_source_inner_validation": "ae_first_source_inner_validation.csv",
        "ae_first_selection_diagnostics": "ae_first_selection_diagnostics.csv",
        "ae_first_domain_breakdown": "ae_first_domain_breakdown.csv",
        "ae_first_margin_bins": "ae_first_margin_bins.csv",
        "ae_first_calibration_diagnostics": "ae_first_calibration_diagnostics.csv",
        "ae_first_plots": plot_artifacts,
    }


def _write_ae_first_plots(
    *,
    reports_dir: Path,
    raw_rows: Sequence[Dict[str, Any]],
    policy_audit_rows: Sequence[Dict[str, Any]],
    margin_bin_rows: Sequence[Dict[str, Any]],
    calibration_rows: Sequence[Dict[str, Any]],
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

    if raw_rows:
        selected_rows = [r for r in raw_rows if int(float(r.get("ae_selected_by_gate", 0) or 0)) == 1]
        if selected_rows:
            plt.figure(figsize=(6, 4))
            plt.scatter(
                [float(r.get("ae_margin", 0.0)) for r in selected_rows],
                [float(r.get("heldout_nelbo", 0.0)) for r in selected_rows],
                s=12,
                alpha=0.65,
            )
            plt.xlabel("AE margin")
            plt.ylabel("Held-out NELBO")
            plt.title("AE margin vs held-out NELBO")
            _save("ae_first_margin_vs_heldout_nelbo.png")

        ae_best_rows = [
            r for r in raw_rows if int(r.get("expert_domain", -1)) == int(r.get("ae_best_expert", -2))
        ]
        plt.figure(figsize=(6, 4))
        plt.hist([float(r.get("oracle_rank_of_ae_best", 0.0)) for r in ae_best_rows], bins=10)
        plt.xlabel("Oracle rank of AE-best expert")
        plt.ylabel("Count")
        plt.title("AE-best oracle rank distribution")
        _save("ae_first_oracle_rank_of_ae_best.png")

    if policy_audit_rows:
        rows = sorted(policy_audit_rows, key=lambda r: int(r.get("fold_query_domain", 0)))
        labels = [str(int(r.get("fold_query_domain", 0))) for r in rows]
        x = np.arange(len(rows))
        width = 0.38
        plt.figure(figsize=(7, 4))
        plt.bar(x - width / 2, [float(r.get("ae_coverage_rate", 0.0)) for r in rows], width, label="AE coverage")
        plt.bar(x + width / 2, [float(r.get("fallback_rate", 0.0)) for r in rows], width, label="fallback")
        plt.xticks(x, labels)
        plt.xlabel("Held-out domain")
        plt.ylabel("Rate")
        plt.title("AE-first coverage and fallback by domain")
        plt.legend(loc="best")
        _save("ae_first_coverage_by_domain.png")

        plt.figure(figsize=(7, 4))
        plt.bar(
            x - width / 2,
            [float(r.get("harmful_vs_metadata_rate", 0.0)) for r in rows],
            width,
            label="harmful vs metadata",
        )
        plt.bar(
            x + width / 2,
            [float(r.get("improving_vs_metadata_rate", 0.0)) for r in rows],
            width,
            label="improving vs metadata",
        )
        plt.xticks(x, labels)
        plt.xlabel("Held-out domain")
        plt.ylabel("Rate")
        plt.title("AE-first harmful vs improving selections")
        plt.legend(loc="best")
        _save("ae_first_harmful_vs_improving_by_domain.png")

    if margin_bin_rows:
        rows = [r for r in margin_bin_rows if int(r.get("n_samples", 0) or 0) > 0]
        if rows:
            labels = [str(r.get("margin_bin", "")) for r in rows]
            plt.figure(figsize=(8, 4))
            plt.plot(labels, [float(r.get("harmful_vs_metadata_rate", 0.0)) for r in rows], marker="o")
            plt.xticks(rotation=30, ha="right")
            plt.xlabel("AE margin bin")
            plt.ylabel("Harmful rate vs metadata")
            plt.title("Harmful rate by AE margin bin")
            _save("ae_first_harmful_rate_by_margin_bin.png")

    if calibration_rows:
        rows = sorted(calibration_rows, key=lambda r: (int(r.get("fold_query_domain", 0)), int(r.get("source_domain", 0))))
        labels = [f"{int(r.get('fold_query_domain', 0))}:{int(r.get('source_domain', 0))}" for r in rows]
        plt.figure(figsize=(max(7, len(rows) * 0.4), 4))
        plt.bar(labels, [float(r.get("ae_target_z_outlier_rate", 0.0)) for r in rows])
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Held-out:source domain")
        plt.ylabel("Target z outlier rate")
        plt.title("AE z-score calibration diagnostics")
        _save("ae_first_zscore_outlier_rate.png")

    return artifacts


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
