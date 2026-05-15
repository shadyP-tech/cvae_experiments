from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.eval.evaluators.learned_utility_config import SourceReliabilityConfig
from src.eval.evaluators.learned_utility_models import _PairwiseRanker
from src.eval.evaluators.learned_utility_pairs import (
    _build_pair_features,
    _build_pairwise_training_pairs,
    _zscore_features,
)
from src.eval.evaluators.learned_utility_protocol import (
    FoldCandidateSet,
    ProtocolError,
    _method_protocol,
    _protocol_row_fields,
)
from src.eval.evaluators.learned_utility_selection import _selection_metrics
from src.eval.metrics import spearman_corr


PRIMARY_METHOD = "source_subdomain_reliability_selected_router_v1"
FALLBACK_METHOD = "ae_argmin_zscore"

SELECTION_SELECTED = "selected_candidate"
SELECTION_NO_PASS = "fallback_no_candidate_passed"
SELECTION_INSUFFICIENT = "fallback_insufficient_reliability_evidence"
SELECTION_PROVENANCE_FAILED = "fallback_candidate_provenance_failed"
SELECTION_POOL_TOO_SMALL = "fallback_candidate_pool_too_small"


@dataclass(frozen=True)
class SourceReliabilityFoldOutputs:
    sample_rows: List[Dict[str, Any]]
    pseudo_domain_rows: List[Dict[str, Any]]
    source_inner_unit_rows: List[Dict[str, Any]]
    candidate_metric_rows: List[Dict[str, Any]]
    parent_guard_rows: List[Dict[str, Any]]
    selection_policy_rows: List[Dict[str, Any]]
    policy_audit_rows: List[Dict[str, Any]]
    predicted_vs_realized_rows: List[Dict[str, Any]]
    selected_method_rows: List[Dict[str, Any]]


@dataclass(frozen=True)
class _PseudoDomains:
    pseudo_rows: List[Dict[str, Any]]
    unit_indices: Dict[Tuple[int, int], np.ndarray]
    group_key: str
    status: str


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def _finite_mean(values: Sequence[float], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float(default)


def _metrics_from_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    return {
        "top1_oracle_hit": _finite_mean([float(r.get("top1_oracle_hit", float("nan"))) for r in rows], 0.0),
        "spearman": _finite_mean([float(r.get("spearman", float("nan"))) for r in rows], float("nan")),
        "mean_oracle_gap_pct": _finite_mean([float(r.get("oracle_gap_pct", float("nan"))) for r in rows], 0.0),
        "mean_oracle_gap": _finite_mean([float(r.get("oracle_gap", float("nan"))) for r in rows], 0.0),
    }


def _resolve_group_key(
    metadata: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    candidates: Sequence[str],
) -> str:
    for key in candidates:
        key_s = str(key).strip()
        if not key_s:
            continue
        values = [str(metadata[int(i)].get(key_s, "") or "").strip() for i in indices]
        if values and all(values):
            return key_s
    return ""


def _pca_project(x: np.ndarray, dim: int) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.zeros((x.shape[0], 1), dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    out_dim = max(1, min(int(dim), int(vt.shape[0])))
    return centered @ vt[:out_dim].T


def _kmeans_labels(x: np.ndarray, k: int, iterations: int) -> np.ndarray:
    if int(k) <= 1:
        return np.zeros((x.shape[0],), dtype=np.int64)
    order = np.lexsort(tuple(x[:, j] for j in range(x.shape[1] - 1, -1, -1)))
    init_pos = np.linspace(0, max(len(order) - 1, 0), int(k)).round().astype(np.int64)
    centers = x[order[init_pos]].copy()
    labels = np.zeros((x.shape[0],), dtype=np.int64)
    for _ in range(int(iterations)):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dist, axis=1).astype(np.int64, copy=False)
        for cluster in range(int(k)):
            mask = labels == int(cluster)
            if np.any(mask):
                centers[int(cluster)] = x[mask].mean(axis=0)
                continue
            nearest = np.min(dist, axis=1)
            centers[int(cluster)] = x[int(np.argmax(nearest))]
    return labels


def build_source_pseudo_domains(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    source_indices: np.ndarray,
    heldout_domain: int,
    cfg: SourceReliabilityConfig,
) -> _PseudoDomains:
    source_indices = np.asarray(source_indices, dtype=np.int64)
    if source_indices.size == 0:
        return _PseudoDomains([], {}, "", SELECTION_INSUFFICIENT)
    if np.any(sample_domains[source_indices] == int(heldout_domain)):
        raise ProtocolError("Source reliability pseudo-domain construction received held-out target rows")
    group_key = _resolve_group_key(metadata, source_indices.tolist(), cfg.group_key_candidates)
    if not group_key:
        return _PseudoDomains([], {}, "", SELECTION_INSUFFICIENT)

    pseudo_rows: List[Dict[str, Any]] = []
    unit_indices: Dict[Tuple[int, int], np.ndarray] = {}
    for parent_domain in sorted(set(int(sample_domains[int(i)]) for i in source_indices.tolist())):
        parent_idx = np.asarray(
            [int(i) for i in source_indices.tolist() if int(sample_domains[int(i)]) == int(parent_domain)],
            dtype=np.int64,
        )
        groups: Dict[str, List[int]] = {}
        for idx in parent_idx.tolist():
            gid = str(metadata[int(idx)].get(group_key, "") or "").strip()
            if gid:
                groups.setdefault(gid, []).append(int(idx))
        if len(groups) < int(cfg.min_pseudo_domains_per_source) * int(cfg.min_groups_per_pseudo_domain):
            continue

        group_ids = sorted(groups)
        centroids = np.asarray(
            [embeddings[np.asarray(groups[gid], dtype=np.int64)].mean(axis=0) for gid in group_ids],
            dtype=np.float64,
        )
        projected = _pca_project(centroids, int(cfg.pca_dim))
        max_k = min(int(cfg.n_pseudo_domains_per_source), len(group_ids) // int(cfg.min_groups_per_pseudo_domain))
        chosen_labels: np.ndarray | None = None
        chosen_k = 0
        for k in range(int(max_k), int(cfg.min_pseudo_domains_per_source) - 1, -1):
            labels = _kmeans_labels(projected, int(k), int(cfg.kmeans_iterations))
            ok = True
            for cluster in range(int(k)):
                cluster_groups = [gid for gid, label in zip(group_ids, labels.tolist()) if int(label) == int(cluster)]
                sample_count = sum(len(groups[gid]) for gid in cluster_groups)
                if len(cluster_groups) < int(cfg.min_groups_per_pseudo_domain):
                    ok = False
                if sample_count < int(cfg.min_samples_per_pseudo_domain):
                    ok = False
            if ok:
                chosen_labels = labels
                chosen_k = int(k)
                break
        if chosen_labels is None:
            continue

        for cluster in range(int(chosen_k)):
            cluster_groups = [gid for gid, label in zip(group_ids, chosen_labels.tolist()) if int(label) == int(cluster)]
            idxs = np.asarray(
                [idx for gid in cluster_groups for idx in groups[gid]],
                dtype=np.int64,
            )
            unit_indices[(int(parent_domain), int(cluster))] = idxs
            for gid in cluster_groups:
                for idx in groups[gid]:
                    pseudo_rows.append(
                        {
                            "outer_heldout_domain": int(heldout_domain),
                            "parent_domain": int(parent_domain),
                            "pseudo_domain": int(cluster),
                            "pseudo_domain_id": f"{int(parent_domain)}_{int(cluster)}",
                            "sample_index": int(idx),
                            "group_key": str(group_key),
                            "group_id": str(gid),
                            "source_only": 1,
                        }
                    )

    status = "available" if unit_indices else SELECTION_INSUFFICIENT
    return _PseudoDomains(pseudo_rows, unit_indices, group_key, status)


def _features_for_pair_rows(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    sample_indices: np.ndarray,
    expert_domains: Sequence[int],
    expert_id_domains: Sequence[int],
    ae_zscore_matrix: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_x, q, e, s = _build_pair_features(
        sample_embeddings=embeddings,
        sample_domains=sample_domains,
        sample_indices=sample_indices,
        expert_domains=expert_domains,
        expert_id_domains=expert_id_domains,
        include_metadata_features=True,
    )
    domain_to_col = {int(domain): int(i) for i, domain in enumerate(expert_id_domains)}
    ae_x = np.asarray(
        [[float(ae_zscore_matrix[int(sample_idx), domain_to_col[int(expert)]])] for sample_idx, expert in zip(s, e)],
        dtype=np.float64,
    )
    if str(method) == "pairwise_ranker_ae_only":
        x = ae_x
    elif str(method) == "pairwise_ranker_ae_combined":
        x = np.concatenate([ae_x, base_x], axis=1)
    else:
        raise ProtocolError(f"Unsupported source reliability candidate method: {method}")
    return x, q, e, s


def _build_source_training_features(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    train_idx: np.ndarray,
    expert_domains: Sequence[int],
    outer_heldout_domain: int,
    parent_domain: int,
    ae_zscore_matrix: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    x_parts: List[np.ndarray] = []
    q_parts: List[np.ndarray] = []
    e_parts: List[np.ndarray] = []
    s_parts: List[np.ndarray] = []
    candidate_counts: set[int] = set()
    excluded_global = {int(outer_heldout_domain), int(parent_domain)}
    for query_domain in sorted(set(int(sample_domains[int(i)]) for i in np.asarray(train_idx, dtype=np.int64).tolist())):
        if int(query_domain) in excluded_global:
            continue
        domain_indices = np.asarray(
            [int(i) for i in np.asarray(train_idx, dtype=np.int64).tolist() if int(sample_domains[int(i)]) == int(query_domain)],
            dtype=np.int64,
        )
        candidates = [int(d) for d in expert_domains if int(d) not in excluded_global | {int(query_domain)}]
        if len(candidates) < 2 or domain_indices.size == 0:
            continue
        x, q, e, s = _features_for_pair_rows(
            embeddings=embeddings,
            sample_domains=sample_domains,
            sample_indices=domain_indices,
            expert_domains=candidates,
            expert_id_domains=expert_domains,
            ae_zscore_matrix=ae_zscore_matrix,
            method=method,
        )
        x_parts.append(x)
        q_parts.append(q)
        e_parts.append(e)
        s_parts.append(s)
        candidate_counts.add(int(len(candidates)))
    if not x_parts or len(candidate_counts) != 1:
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            0,
        )
    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(q_parts, axis=0),
        np.concatenate(e_parts, axis=0),
        np.concatenate(s_parts, axis=0),
        int(next(iter(candidate_counts))),
    )


def _fit_predict_pairwise_unit(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold: FoldCandidateSet,
    ae_zscore_matrix: np.ndarray,
    method: str,
    pairwise_cfg: Mapping[str, Any],
    seed: int,
    parent_domain: int,
) -> np.ndarray | None:
    x_train, q_train, e_train, s_train, candidates_per_sample = _build_source_training_features(
        embeddings=embeddings,
        sample_domains=sample_domains,
        train_idx=train_idx,
        expert_domains=expert_domains,
        outer_heldout_domain=int(fold.heldout_domain),
        parent_domain=int(parent_domain),
        ae_zscore_matrix=ae_zscore_matrix,
        method=str(method),
    )
    if x_train.size == 0 or int(candidates_per_sample) < 2:
        return None
    domain_to_col = {int(domain): int(i) for i, domain in enumerate(expert_domains)}
    y_train = true_nelbo[s_train, [domain_to_col[int(ed)] for ed in e_train]]
    train_pairs, _diag = _build_pairwise_training_pairs(
        y_train=y_train,
        q_train=q_train,
        s_train=s_train,
        experts_per_sample=int(candidates_per_sample),
        near_tie_delta=float(pairwise_cfg.get("near_tie_delta", 0.0)),
        hard_pair_fraction=float(pairwise_cfg.get("hard_pair_fraction", 0.5)),
        random_pair_fraction=float(pairwise_cfg.get("random_pair_fraction", 0.5)),
        max_pairs_per_sample=int(pairwise_cfg.get("max_pairs_per_sample", 12)),
        max_pairs_per_domain=int(pairwise_cfg.get("max_pairs_per_domain", 5000)),
        seed=int(seed) + int(parent_domain) + int(fold.heldout_domain),
    )
    if not train_pairs:
        return None
    x_val, _q_val, _e_val, _s_val = _features_for_pair_rows(
        embeddings=embeddings,
        sample_domains=sample_domains,
        sample_indices=val_idx,
        expert_domains=fold.candidate_expert_domains,
        expert_id_domains=expert_domains,
        ae_zscore_matrix=ae_zscore_matrix,
        method=str(method),
    )
    x_train_z, x_val_z = _zscore_features(x_train, x_val)
    ranker = _PairwiseRanker(
        seed=int(seed) + int(parent_domain) + int(fold.heldout_domain),
        hidden_dim=int(pairwise_cfg.get("hidden_dim", 128)),
        epochs=int(pairwise_cfg.get("epochs", 40)),
        lr=float(pairwise_cfg.get("lr", 1e-3)),
        batch_size=int(pairwise_cfg.get("batch_size", 2048)),
        margin=float(pairwise_cfg.get("margin", 1.0)),
        device=str(pairwise_cfg.get("device", "auto")),
    )
    ranker.fit(x_train_z, train_pairs)
    pred = ranker.predict(x_val_z)
    return pred.reshape(int(val_idx.shape[0]), int(len(fold.candidate_expert_domains)))


def _candidate_rows_clean(candidate_rows: Sequence[Mapping[str, Any]], *, fold: FoldCandidateSet) -> Tuple[bool, str]:
    if not candidate_rows:
        return False, "candidate_rows_missing"
    for row in candidate_rows:
        if int(float(row.get("routing_uses_eval_nelbo", 0) or 0)) != 0:
            return False, "candidate_method_routing_uses_eval_nelbo"
        if int(float(row.get("routing_uses_eval_domain_statistics", 0) or 0)) != 0:
            return False, "candidate_method_routing_uses_eval_domain_statistics"
        if int(float(row.get("target_expert_excluded", 0) or 0)) != 1:
            return False, "candidate_method_target_expert_not_excluded"
        if int(float(row.get("adoption_eligible", 0) or 0)) != 1:
            return False, "candidate_method_not_adoption_eligible"
        if int(float(row.get("diagnostic_only", 0) or 0)) != 0:
            return False, "candidate_method_diagnostic_only"
        candidates = {int(v) for v in str(row.get("candidate_experts", "")).split("|") if str(v).strip()}
        if int(fold.heldout_domain) in candidates:
            return False, "candidate_method_includes_target_expert"
    return True, ""


def _material_degradation(candidate: Mapping[str, float], fallback: Mapping[str, float], cfg: SourceReliabilityConfig) -> bool:
    return bool(
        float(fallback["top1_oracle_hit"]) - float(candidate["top1_oracle_hit"]) > float(cfg.max_top1_drop_abs)
        or float(fallback["spearman"]) - float(candidate["spearman"]) > float(cfg.max_spearman_drop_abs)
        or float(candidate["mean_oracle_gap_pct"]) - float(fallback["mean_oracle_gap_pct"]) > float(cfg.max_gap_pct_degradation)
    )


def _source_inner_metrics_for_candidate(
    *,
    unit_rows: Sequence[Mapping[str, Any]],
    method: str,
    cfg: SourceReliabilityConfig,
) -> Dict[str, Any]:
    rows = [dict(r) for r in unit_rows if str(r.get("candidate_method", "")) == str(method)]
    if not rows:
        return {
            "candidate_method": str(method),
            "n_source_inner_units": 0,
            "n_parent_domains": 0,
            "passes_reliability_gates": 0,
        }
    gains = [float(r["gap_pct_reduction_vs_fallback"]) for r in rows]
    positive_gains = [max(float(v), 0.0) for v in gains]
    total_positive = float(sum(positive_gains))
    parents = sorted(set(int(r["parent_domain"]) for r in rows))
    parent_means = {
        p: _finite_mean([float(r["gap_pct_reduction_vs_fallback"]) for r in rows if int(r["parent_domain"]) == int(p)], 0.0)
        for p in parents
    }
    parent_unit_counts = {p: sum(1 for r in rows if int(r["parent_domain"]) == int(p)) for p in parents}
    valid_gain_share_parents = [
        p for p, count in parent_unit_counts.items() if int(count) >= int(cfg.min_units_per_parent_for_gain_share)
    ]
    parent_positive_gains = {
        p: sum(
            max(float(r["gap_pct_reduction_vs_fallback"]), 0.0)
            for r in rows
            if int(r["parent_domain"]) == int(p)
        )
        for p in valid_gain_share_parents
    }
    max_gain_share = (
        max(parent_positive_gains.values()) / total_positive
        if total_positive > 0.0 and parent_positive_gains
        else 1.0
    )
    positive_unit_rate = float(np.mean(np.asarray(gains, dtype=np.float64) > 0.0)) if gains else 0.0
    positive_parent_rate = (
        float(np.mean(np.asarray([float(v) for v in parent_means.values()], dtype=np.float64) > 0.0))
        if parent_means
        else 0.0
    )
    material_count = int(sum(int(r.get("material_degradation_vs_fallback", 0)) for r in rows))
    worst_degradation = max(float(r["gap_pct_degradation_vs_fallback"]) for r in rows)
    predicted_gain = _finite_mean(gains, 0.0)
    passes = bool(
        len(rows) >= int(cfg.min_source_inner_units)
        and len(parents) >= int(cfg.min_parent_domains)
        and material_count == 0
        and worst_degradation <= float(cfg.max_worst_unit_gap_degradation)
        and predicted_gain >= float(cfg.min_gap_reduction_vs_fallback)
        and positive_unit_rate >= float(cfg.min_positive_unit_rate)
        and positive_parent_rate >= float(cfg.min_positive_parent_rate)
        and max_gain_share <= float(cfg.max_positive_gain_share)
    )
    return {
        "candidate_method": str(method),
        "n_source_inner_units": int(len(rows)),
        "n_parent_domains": int(len(parents)),
        "source_inner_predicted_gain": float(predicted_gain),
        "source_inner_top1_delta": _finite_mean([float(r["top1_delta_vs_fallback"]) for r in rows], 0.0),
        "source_inner_spearman_delta": _finite_mean([float(r["spearman_delta_vs_fallback"]) for r in rows], 0.0),
        "positive_unit_rate": float(positive_unit_rate),
        "positive_parent_rate": float(positive_parent_rate),
        "max_positive_gain_share": float(max_gain_share),
        "worst_unit_gap_degradation": float(worst_degradation),
        "material_degradation_count": int(material_count),
        "parent_holdout_guard_passed": int(passes),
        "passes_reliability_gates": int(passes),
    }


def _fallback_rows(
    *,
    sample_domains: np.ndarray,
    expert_domains: Sequence[int],
    test_idx: np.ndarray,
    fold: FoldCandidateSet,
    true_eval: np.ndarray,
    global_eval: np.ndarray,
    ae_zscore_matrix: np.ndarray,
    tie_policy: str,
    selection_status: str,
    selected_source_method: str,
) -> List[Dict[str, Any]]:
    score_matrix = ae_zscore_matrix[np.asarray(test_idx, dtype=np.int64)][:, list(fold.candidate_col_indices)]
    _metrics, rows = _selection_metrics(
        method=PRIMARY_METHOD,
        query_domains=sample_domains[test_idx],
        expert_domains=fold.candidate_expert_domains,
        score_matrix=score_matrix,
        true_nelbo_matrix=true_eval,
        fold=fold,
        global_true_nelbo_matrix=global_eval,
        global_expert_domains=expert_domains,
        tie_policy=tie_policy,
    )
    for row in rows:
        row["sample_index"] = int(test_idx[int(row["sample_index"])])
        row["source_reliability_selected_method"] = str(selected_source_method)
        row["selection_status"] = str(selection_status)
        row["selection_source"] = "source_only_reliability"
    return rows


def _copy_candidate_rows(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    fold: FoldCandidateSet,
    selection_status: str,
    selected_source_method: str,
) -> List[Dict[str, Any]]:
    method_protocol = _method_protocol(PRIMARY_METHOD)
    protocol_fields = _protocol_row_fields(fold=fold, method_protocol=method_protocol, method=PRIMARY_METHOD)
    out: List[Dict[str, Any]] = []
    for row in candidate_rows:
        copied = dict(row)
        copied.update(protocol_fields)
        copied["method"] = PRIMARY_METHOD
        copied["source_reliability_selected_method"] = str(selected_source_method)
        copied["selection_status"] = str(selection_status)
        copied["selection_source"] = "source_only_reliability"
        out.append(copied)
    return out


def _realized_rows(
    *,
    fallback_final_rows: Sequence[Mapping[str, Any]],
    learned_sample_rows: Sequence[Mapping[str, Any]],
    candidate_metrics: Sequence[Mapping[str, Any]],
    candidate_methods: Sequence[str],
    selected_method: str,
    selection_status: str,
) -> List[Dict[str, Any]]:
    fallback_metrics = _metrics_from_rows(fallback_final_rows)
    predicted_by_method = {
        str(row.get("candidate_method")): float(row.get("source_inner_predicted_gain", float("nan")))
        for row in candidate_metrics
    }
    realized_by_method: Dict[str, float] = {}
    method_gap: Dict[str, float] = {str(FALLBACK_METHOD): float(fallback_metrics["mean_oracle_gap_pct"])}
    for method in candidate_methods:
        rows = [r for r in learned_sample_rows if str(r.get("method", "")) == str(method)]
        if not rows:
            continue
        metrics = _metrics_from_rows(rows)
        method_gap[str(method)] = float(metrics["mean_oracle_gap_pct"])
        realized_by_method[str(method)] = float(fallback_metrics["mean_oracle_gap_pct"] - metrics["mean_oracle_gap_pct"])
    if realized_by_method:
        best_method = max(realized_by_method, key=lambda m: realized_by_method[m])
        if realized_by_method[best_method] <= 0.0:
            best_method = FALLBACK_METHOD
    else:
        best_method = FALLBACK_METHOD
    pred_values = [float(predicted_by_method.get(str(m), float("nan"))) for m in candidate_methods]
    real_values = [float(realized_by_method.get(str(m), float("nan"))) for m in candidate_methods]
    finite_pairs = [(p, r) for p, r in zip(pred_values, real_values) if np.isfinite(p) and np.isfinite(r)]
    rho = (
        float(spearman_corr([p for p, _r in finite_pairs], [r for _p, r in finite_pairs]))
        if len(finite_pairs) >= 2
        else float("nan")
    )
    selected_candidate = str(selection_status) == SELECTION_SELECTED
    selected_realized_gain = float(realized_by_method.get(str(selected_method), 0.0))
    wrong_activation = int(bool(selected_candidate) and selected_realized_gain < 0.0)
    wrong_abstention = int((not selected_candidate) and any(float(v) > 0.0 for v in realized_by_method.values()))
    rows: List[Dict[str, Any]] = []
    for method in candidate_methods:
        rows.append(
            {
                "candidate_method": str(method),
                "source_inner_predicted_gain": float(predicted_by_method.get(str(method), float("nan"))),
                "heldout_realized_gain": float(realized_by_method.get(str(method), float("nan"))),
                "source_inner_selected_method": str(selected_method),
                "heldout_best_non_oracle_method": str(best_method),
                "selection_correct": int(str(selected_method) == str(best_method)),
                "wrong_activation_rate": int(wrong_activation),
                "wrong_abstention_rate": int(wrong_abstention),
                "spearman_predicted_vs_realized_gain": float(rho),
                "method_selection_accuracy": int(str(selected_method) == str(best_method)),
                "fallback_gap_pct": float(fallback_metrics["mean_oracle_gap_pct"]),
                "candidate_gap_pct": float(method_gap.get(str(method), float("nan"))),
            }
        )
    return rows


def run_source_reliability_for_fold(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: FoldCandidateSet,
    true_eval: np.ndarray,
    global_eval: np.ndarray,
    ae_zscore_matrix: np.ndarray,
    learned_sample_rows: Sequence[Mapping[str, Any]],
    pairwise_cfg: Mapping[str, Any],
    cfg: SourceReliabilityConfig,
    seed: int,
    tie_policy: str,
) -> SourceReliabilityFoldOutputs:
    if str(cfg.primary_method) != PRIMARY_METHOD:
        raise ProtocolError(f"source_reliability currently supports primary_method={PRIMARY_METHOD!r}")
    if str(cfg.fallback_method) != FALLBACK_METHOD:
        raise ProtocolError("source_reliability fallback_method must be ae_argmin_zscore")

    fallback_final_rows = _fallback_rows(
        sample_domains=sample_domains,
        expert_domains=expert_domains,
        test_idx=test_idx,
        fold=fold,
        true_eval=true_eval,
        global_eval=global_eval,
        ae_zscore_matrix=ae_zscore_matrix,
        tie_policy=tie_policy,
        selection_status=SELECTION_INSUFFICIENT,
        selected_source_method=FALLBACK_METHOD,
    )
    source_idx = np.asarray(train_idx, dtype=np.int64)
    pseudo = build_source_pseudo_domains(
        embeddings=embeddings,
        sample_domains=sample_domains,
        metadata=metadata,
        source_indices=source_idx,
        heldout_domain=int(fold.heldout_domain),
        cfg=cfg,
    )

    candidate_methods = tuple(str(m) for m in cfg.candidate_methods)
    candidate_rows_by_method = {
        method: [dict(r) for r in learned_sample_rows if str(r.get("method", "")) == str(method)]
        for method in candidate_methods
    }
    provenance_rows: List[Dict[str, Any]] = []
    clean_candidates: List[str] = []
    for method in candidate_methods:
        clean, reason = _candidate_rows_clean(candidate_rows_by_method.get(method, []), fold=fold)
        provenance_rows.append(
            {
                "method": PRIMARY_METHOD,
                "outer_heldout_domain": int(fold.heldout_domain),
                "candidate_method": str(method),
                "candidate_method_routing_uses_eval_nelbo": 0 if clean else int(reason == "candidate_method_routing_uses_eval_nelbo"),
                "candidate_method_routing_uses_eval_domain_statistics": 0
                if clean
                else int(reason == "candidate_method_routing_uses_eval_domain_statistics"),
                "candidate_method_excludes_target_expert": int(
                    clean
                    or reason
                    not in {
                        "candidate_rows_missing",
                        "candidate_method_target_expert_not_excluded",
                        "candidate_method_includes_target_expert",
                    }
                ),
                "candidate_method_adoption_eligible": int(
                    clean or reason not in {"candidate_rows_missing", "candidate_method_not_adoption_eligible"}
                ),
                "candidate_method_diagnostic_only": int((not clean) and reason == "candidate_method_diagnostic_only"),
                "candidate_provenance_passed": int(clean),
                "candidate_provenance_failure_reason": str(reason),
            }
        )
        if clean:
            clean_candidates.append(str(method))

    selection_status = SELECTION_INSUFFICIENT
    selected_method = FALLBACK_METHOD
    selected_metric: Dict[str, Any] | None = None
    source_inner_unit_rows: List[Dict[str, Any]] = []
    candidate_metric_rows: List[Dict[str, Any]] = []
    parent_guard_rows: List[Dict[str, Any]] = []

    source_domains = sorted(set(int(sample_domains[int(i)]) for i in source_idx.tolist()))
    pool_too_small = any(
        len(FoldCandidateSet.for_heldout_domain(
            heldout_domain=int(fold.heldout_domain),
            expert_domains=expert_domains,
            excluded_domains=[int(parent_domain)],
        ).candidate_expert_domains) < int(cfg.min_candidate_pool_size)
        for parent_domain in source_domains
    )

    if pool_too_small:
        selection_status = SELECTION_POOL_TOO_SMALL
    elif not clean_candidates:
        selection_status = SELECTION_PROVENANCE_FAILED
    elif pseudo.status != "available":
        selection_status = SELECTION_INSUFFICIENT
    else:
        for (parent_domain, pseudo_domain), val_idx in sorted(pseudo.unit_indices.items()):
            inner_fold = FoldCandidateSet.for_heldout_domain(
                heldout_domain=int(fold.heldout_domain),
                expert_domains=expert_domains,
                excluded_domains=[int(parent_domain)],
            )
            if len(inner_fold.candidate_expert_domains) < int(cfg.min_candidate_pool_size):
                continue
            train_unit_idx = np.asarray(
                [
                    int(i)
                    for i in source_idx.tolist()
                    if int(sample_domains[int(i)]) not in {int(parent_domain), int(fold.heldout_domain)}
                ],
                dtype=np.int64,
            )
            true_unit = inner_fold.slice_nelbo(true_nelbo, val_idx)
            global_unit = true_nelbo[np.asarray(val_idx, dtype=np.int64)]
            fallback_score = ae_zscore_matrix[np.asarray(val_idx, dtype=np.int64)][:, list(inner_fold.candidate_col_indices)]
            fallback_metrics, _fallback_unit_rows = _selection_metrics(
                method=FALLBACK_METHOD,
                query_domains=sample_domains[val_idx],
                expert_domains=inner_fold.candidate_expert_domains,
                score_matrix=fallback_score,
                true_nelbo_matrix=true_unit,
                fold=inner_fold,
                global_true_nelbo_matrix=global_unit,
                global_expert_domains=expert_domains,
                tie_policy=tie_policy,
            )
            for method in clean_candidates:
                pred_score = _fit_predict_pairwise_unit(
                    embeddings=embeddings,
                    sample_domains=sample_domains,
                    true_nelbo=true_nelbo,
                    expert_domains=expert_domains,
                    train_idx=train_unit_idx,
                    val_idx=val_idx,
                    fold=inner_fold,
                    ae_zscore_matrix=ae_zscore_matrix,
                    method=str(method),
                    pairwise_cfg=pairwise_cfg,
                    seed=int(seed),
                    parent_domain=int(parent_domain),
                )
                if pred_score is None:
                    continue
                candidate_metrics, _candidate_unit_rows = _selection_metrics(
                    method=str(method),
                    query_domains=sample_domains[val_idx],
                    expert_domains=inner_fold.candidate_expert_domains,
                    score_matrix=pred_score,
                    true_nelbo_matrix=true_unit,
                    fold=inner_fold,
                    global_true_nelbo_matrix=global_unit,
                    global_expert_domains=expert_domains,
                    tie_policy=tie_policy,
                )
                top1_delta = float(candidate_metrics["top1_oracle_hit"]) - float(fallback_metrics["top1_oracle_hit"])
                spearman_delta = float(candidate_metrics["spearman"]) - float(fallback_metrics["spearman"])
                gap_degradation = float(candidate_metrics["mean_oracle_gap_pct"]) - float(
                    fallback_metrics["mean_oracle_gap_pct"]
                )
                gap_reduction = -gap_degradation
                material = _material_degradation(candidate_metrics, fallback_metrics, cfg)
                unit_row = {
                    "method": PRIMARY_METHOD,
                    "candidate_method": str(method),
                    "outer_heldout_domain": int(fold.heldout_domain),
                    "parent_domain": int(parent_domain),
                    "pseudo_domain": int(pseudo_domain),
                    "pseudo_domain_id": f"{int(parent_domain)}_{int(pseudo_domain)}",
                    "n_validation_samples": int(val_idx.shape[0]),
                    "n_training_samples": int(train_unit_idx.shape[0]),
                    "candidate_experts": inner_fold.label(),
                    "excluded_target_expert": int(fold.heldout_domain),
                    "excluded_parent_expert": int(parent_domain),
                    "heldout_target_nelbo_used_for_selection": 0,
                    "parent_holdout_guard_applied": int(cfg.require_parent_holdout_guard),
                    "fallback_top1_oracle_hit": float(fallback_metrics["top1_oracle_hit"]),
                    "fallback_spearman": float(fallback_metrics["spearman"]),
                    "fallback_mean_oracle_gap_pct": float(fallback_metrics["mean_oracle_gap_pct"]),
                    "candidate_top1_oracle_hit": float(candidate_metrics["top1_oracle_hit"]),
                    "candidate_spearman": float(candidate_metrics["spearman"]),
                    "candidate_mean_oracle_gap_pct": float(candidate_metrics["mean_oracle_gap_pct"]),
                    "top1_delta_vs_fallback": float(top1_delta),
                    "spearman_delta_vs_fallback": float(spearman_delta),
                    "gap_pct_reduction_vs_fallback": float(gap_reduction),
                    "gap_pct_degradation_vs_fallback": float(gap_degradation),
                    "material_degradation_vs_fallback": int(material),
                }
                source_inner_unit_rows.append(unit_row)

        for method in clean_candidates:
            metric = _source_inner_metrics_for_candidate(
                unit_rows=source_inner_unit_rows,
                method=str(method),
                cfg=cfg,
            )
            metric.update(
                {
                    "method": PRIMARY_METHOD,
                    "outer_heldout_domain": int(fold.heldout_domain),
                    "heldout_target_nelbo_used_for_selection": 0,
                }
            )
            candidate_metric_rows.append(metric)
            parent_guard_rows.append(
                {
                    "method": PRIMARY_METHOD,
                    "outer_heldout_domain": int(fold.heldout_domain),
                    "candidate_method": str(method),
                    "parent_holdout_guard_required": int(cfg.require_parent_holdout_guard),
                    "parent_holdout_guard_passed": int(metric.get("parent_holdout_guard_passed", 0)),
                    "n_source_inner_units": int(metric.get("n_source_inner_units", 0)),
                    "n_parent_domains": int(metric.get("n_parent_domains", 0)),
                }
            )

        passing = [row for row in candidate_metric_rows if int(row.get("passes_reliability_gates", 0)) == 1]
        if not source_inner_unit_rows:
            selection_status = SELECTION_INSUFFICIENT
        elif not passing:
            selection_status = SELECTION_NO_PASS
        else:
            order = {method: idx for idx, method in enumerate(candidate_methods)}
            selected_metric = sorted(
                passing,
                key=lambda row: (
                    float(row.get("source_inner_predicted_gain", float("-inf"))),
                    float(row.get("source_inner_top1_delta", float("-inf"))),
                    float(row.get("source_inner_spearman_delta", float("-inf"))),
                    -float(row.get("worst_unit_gap_degradation", float("inf"))),
                    -float(row.get("max_positive_gain_share", float("inf"))),
                    -float(order.get(str(row.get("candidate_method")), 10**6)),
                ),
                reverse=True,
            )[0]
            selected_method = str(selected_metric["candidate_method"])
            selection_status = SELECTION_SELECTED

    if selection_status == SELECTION_SELECTED:
        selected_rows = _copy_candidate_rows(
            candidate_rows=candidate_rows_by_method.get(selected_method, []),
            fold=fold,
            selection_status=selection_status,
            selected_source_method=selected_method,
        )
    else:
        selected_rows = _fallback_rows(
            sample_domains=sample_domains,
            expert_domains=expert_domains,
            test_idx=test_idx,
            fold=fold,
            true_eval=true_eval,
            global_eval=global_eval,
            ae_zscore_matrix=ae_zscore_matrix,
            tie_policy=tie_policy,
            selection_status=selection_status,
            selected_source_method=FALLBACK_METHOD,
        )

    predicted_rows = _realized_rows(
        fallback_final_rows=fallback_final_rows,
        learned_sample_rows=learned_sample_rows,
        candidate_metrics=candidate_metric_rows,
        candidate_methods=candidate_methods,
        selected_method=selected_method,
        selection_status=selection_status,
    )
    selected_method_row = {
        "method": PRIMARY_METHOD,
        "outer_heldout_domain": int(fold.heldout_domain),
        "selected_method_by_outer_domain": str(selected_method),
        "selection_status": str(selection_status),
        "source_inner_predicted_gain": float((selected_metric or {}).get("source_inner_predicted_gain", 0.0)),
        "heldout_target_nelbo_used_for_selection": 0,
    }
    selected_metrics = _metrics_from_rows(selected_rows)
    fallback_metrics = _metrics_from_rows(fallback_final_rows)
    policy_audit_row = {
        "method": PRIMARY_METHOD,
        "outer_heldout_domain": int(fold.heldout_domain),
        "source_train_domains": "|".join(str(int(v)) for v in source_domains),
        "group_key": str(pseudo.group_key),
        "selection_status": str(selection_status),
        "selected_source_method": str(selected_method),
        "fallback_method": FALLBACK_METHOD,
        "heldout_target_nelbo_used_for_selection": 0,
        "pseudo_domain_strategy": str(cfg.pseudo_domain_strategy),
        "n_source_inner_units": int(len(source_inner_unit_rows)),
        "n_pseudo_domain_rows": int(len(pseudo.pseudo_rows)),
        "selected_top1_oracle_hit": float(selected_metrics["top1_oracle_hit"]),
        "selected_spearman": float(selected_metrics["spearman"]),
        "selected_mean_oracle_gap_pct": float(selected_metrics["mean_oracle_gap_pct"]),
        "fallback_top1_oracle_hit": float(fallback_metrics["top1_oracle_hit"]),
        "fallback_spearman": float(fallback_metrics["spearman"]),
        "fallback_mean_oracle_gap_pct": float(fallback_metrics["mean_oracle_gap_pct"]),
    }

    return SourceReliabilityFoldOutputs(
        sample_rows=selected_rows,
        pseudo_domain_rows=pseudo.pseudo_rows,
        source_inner_unit_rows=source_inner_unit_rows,
        candidate_metric_rows=candidate_metric_rows,
        parent_guard_rows=parent_guard_rows,
        selection_policy_rows=[selected_method_row],
        policy_audit_rows=[policy_audit_row, *provenance_rows],
        predicted_vs_realized_rows=predicted_rows,
        selected_method_rows=[selected_method_row],
    )


def write_source_reliability_artifacts(
    *,
    reports_dir: Path,
    pseudo_domain_rows: Sequence[Mapping[str, Any]],
    source_inner_unit_rows: Sequence[Mapping[str, Any]],
    candidate_metric_rows: Sequence[Mapping[str, Any]],
    parent_guard_rows: Sequence[Mapping[str, Any]],
    selection_policy_rows: Sequence[Mapping[str, Any]],
    policy_audit_rows: Sequence[Mapping[str, Any]],
    predicted_vs_realized_rows: Sequence[Mapping[str, Any]],
    selected_method_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not (
        pseudo_domain_rows
        or source_inner_unit_rows
        or candidate_metric_rows
        or parent_guard_rows
        or selection_policy_rows
        or policy_audit_rows
        or predicted_vs_realized_rows
        or selected_method_rows
    ):
        return {}

    _write_csv(reports_dir / "source_reliability_pseudo_domains.csv", pseudo_domain_rows)
    _write_csv(reports_dir / "source_reliability_source_inner_units.csv", source_inner_unit_rows)
    _write_csv(reports_dir / "source_reliability_candidate_metrics.csv", candidate_metric_rows)
    _write_csv(reports_dir / "source_reliability_parent_guard.csv", parent_guard_rows)
    _write_csv(reports_dir / "source_reliability_selection_policy.csv", selection_policy_rows)
    _write_csv(reports_dir / "source_reliability_policy_audit.csv", policy_audit_rows)
    _write_csv(reports_dir / "source_reliability_predicted_vs_realized.csv", predicted_vs_realized_rows)
    _write_csv(reports_dir / "source_reliability_selected_method_by_outer_domain.csv", selected_method_rows)

    statuses = [str(r.get("selection_status", "")) for r in selected_method_rows]
    n = max(len(statuses), 1)
    fallback_count = sum(1 for status in statuses if status != SELECTION_SELECTED)
    selected_count = sum(1 for status in statuses if status == SELECTION_SELECTED)
    pred_pairs = [
        (
            float(r.get("source_inner_predicted_gain", float("nan"))),
            float(r.get("heldout_realized_gain", float("nan"))),
        )
        for r in predicted_vs_realized_rows
    ]
    finite_pairs = [(p, r) for p, r in pred_pairs if np.isfinite(p) and np.isfinite(r)]
    rho = (
        float(spearman_corr([p for p, _r in finite_pairs], [r for _p, r in finite_pairs]))
        if len(finite_pairs) >= 2
        else float("nan")
    )
    accuracy = _finite_mean([float(r.get("selection_correct", float("nan"))) for r in predicted_vs_realized_rows])
    reliability_verdict = "DIAGNOSTIC ONLY"
    if np.isfinite(rho) and rho >= 0.30:
        reliability_verdict = "PASS"
    elif np.isfinite(rho) and rho > 0.0:
        reliability_verdict = "WEAK PASS"
    summary = {
        "method": PRIMARY_METHOD,
        "protocol_status": "PASS",
        "reliability_estimation_verdict": reliability_verdict,
        "routing_adoption_verdict": "DIAGNOSTIC ONLY",
        "fallback_rate_by_dataset": float(fallback_count / n),
        "candidate_selection_rate_by_dataset": float(selected_count / n),
        "spearman_predicted_vs_realized_gain": float(rho),
        "method_selection_accuracy": float(accuracy),
        "selection_status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
    }
    _write_csv(reports_dir / "source_reliability_dataset_selection_summary.csv", [summary])
    (reports_dir / "source_reliability_provenance.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    return {
        "source_reliability_pseudo_domains": "source_reliability_pseudo_domains.csv",
        "source_reliability_source_inner_units": "source_reliability_source_inner_units.csv",
        "source_reliability_candidate_metrics": "source_reliability_candidate_metrics.csv",
        "source_reliability_parent_guard": "source_reliability_parent_guard.csv",
        "source_reliability_selection_policy": "source_reliability_selection_policy.csv",
        "source_reliability_policy_audit": "source_reliability_policy_audit.csv",
        "source_reliability_predicted_vs_realized": "source_reliability_predicted_vs_realized.csv",
        "source_reliability_selected_method_by_outer_domain": "source_reliability_selected_method_by_outer_domain.csv",
        "source_reliability_dataset_selection_summary": "source_reliability_dataset_selection_summary.csv",
        "source_reliability_provenance": "source_reliability_provenance.json",
    }
