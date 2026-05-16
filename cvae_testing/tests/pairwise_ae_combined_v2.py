from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

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
from src.eval.evaluators.learned_utility_selection import _pairwise_auc_single, _selection_metrics


BASELINE_METHOD = "pairwise_ranker_ae_combined"
PRIMARY_METHOD = "pairwise_ranker_ae_combined_inner_selected_v2"
STRICT_PRIMARY_METHOD = "pairwise_ranker_ae_combined_strict_inner_selected_v2"
TARGET_BATCH_AGREEMENT_PRIMARY_METHOD = "pairwise_ranker_ae_combined_target_batch_agreement_gated_v3"
TARGET_BATCH_AGREEMENT_V31_PRIMARY_METHOD = "pairwise_ranker_ae_combined_target_batch_agreement_gated_v31"
RANK_MARGIN_UNWEIGHTED = "pairwise_ranker_ae_combined_rank_margin_unweighted_v2"
RAW_AE_WEIGHTED = "pairwise_ranker_ae_combined_raw_ae_weighted_v2"
RANK_MARGIN_WEIGHTED = "pairwise_ranker_ae_combined_rank_margin_weighted_v2"
V2_CANDIDATE_METHODS = (
    BASELINE_METHOD,
    RANK_MARGIN_UNWEIGHTED,
    RAW_AE_WEIGHTED,
    RANK_MARGIN_WEIGHTED,
)
_SIMPLER_METHOD_ORDER = {method: idx for idx, method in enumerate(V2_CANDIDATE_METHODS)}
_STRICT_METHOD_ORDER = {
    RANK_MARGIN_UNWEIGHTED: 0,
    RAW_AE_WEIGHTED: 1,
    RANK_MARGIN_WEIGHTED: 2,
}


def _v2_cfg(pairwise_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return dict((pairwise_cfg.get("utility_weighted_v2", {}) or {}))


def _selection_mode(pairwise_cfg: Mapping[str, Any]) -> str:
    return str(_v2_cfg(pairwise_cfg).get("selection_mode", "standard")).strip().lower()


def _is_strict_mode(pairwise_cfg: Mapping[str, Any]) -> bool:
    return _selection_mode(pairwise_cfg) == "strict_adoption"


def _is_target_batch_agreement_mode(pairwise_cfg: Mapping[str, Any]) -> bool:
    return _selection_mode(pairwise_cfg) == "target_batch_agreement_gated"


def _uses_strict_source_inner_gates(pairwise_cfg: Mapping[str, Any]) -> bool:
    return _is_strict_mode(pairwise_cfg) or _is_target_batch_agreement_mode(pairwise_cfg)


def _primary_method(pairwise_cfg: Mapping[str, Any]) -> str:
    cfg = _v2_cfg(pairwise_cfg)
    if _is_target_batch_agreement_mode(pairwise_cfg):
        target_cfg = dict((cfg.get("target_batch_agreement", {}) or {}))
        gate_scope = str(target_cfg.get("gate_scope", "rank_margin_only")).strip().lower()
        default = TARGET_BATCH_AGREEMENT_V31_PRIMARY_METHOD if gate_scope == "all_nonbaseline" else TARGET_BATCH_AGREEMENT_PRIMARY_METHOD
    elif _is_strict_mode(pairwise_cfg):
        default = STRICT_PRIMARY_METHOD
    else:
        default = PRIMARY_METHOD
    primary = str(cfg.get("primary_method", default)).strip() or default
    expected = default
    if primary != expected:
        raise ProtocolError(f"utility_weighted_v2.primary_method must be {expected!r} for selection_mode={_selection_mode(pairwise_cfg)!r}")
    return primary


def _strict_cfg(pairwise_cfg: Mapping[str, Any]) -> Dict[str, float]:
    cfg = dict((_v2_cfg(pairwise_cfg).get("strict_adoption", {}) or {}))
    return {
        "min_macro_gap_reduction_pp": float(cfg.get("min_macro_gap_reduction_pp", 0.5)),
        "max_top1_drop_abs": float(cfg.get("max_top1_drop_abs", 0.02)),
        "max_spearman_drop_abs": float(cfg.get("max_spearman_drop_abs", 0.03)),
        "max_worst_inner_center_gap_degradation_pp": float(cfg.get("max_worst_inner_center_gap_degradation_pp", 0.25)),
        "min_positive_inner_center_rate": float(cfg.get("min_positive_inner_center_rate", 0.75)),
        "min_non_degrading_inner_center_rate": float(cfg.get("min_non_degrading_inner_center_rate", 1.0)),
        "min_passing_inner_centers": float(cfg.get("min_passing_inner_centers", 2)),
    }


def _target_batch_agreement_cfg(pairwise_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = dict((_v2_cfg(pairwise_cfg).get("target_batch_agreement", {}) or {}))
    primary = str(_v2_cfg(pairwise_cfg).get("primary_method", "")).strip()
    default_gate_scope = "all_nonbaseline" if primary == TARGET_BATCH_AGREEMENT_V31_PRIMARY_METHOD else "rank_margin_only"
    return {
        "agreement_threshold": float(cfg.get("agreement_threshold", 0.60)),
        "agreement_threshold_source": str(cfg.get("agreement_threshold_source", "predeclared_development_seed_diagnostic")),
        "reference_method": str(cfg.get("reference_method", RAW_AE_WEIGHTED)),
        "gate_scope": str(cfg.get("gate_scope", default_gate_scope)).strip().lower() or default_gate_scope,
        "min_query_count": int(cfg.get("min_query_count", 100)),
        "min_group_count": int(cfg.get("min_group_count", 2)),
        "group_key_candidates": list(cfg.get("group_key_candidates", ["patient_id", "slide_id", "case_id"]) or []),
    }


@dataclass(frozen=True)
class PairwiseAECombinedV2FoldOutputs:
    sample_rows: List[Dict[str, Any]]
    pair_rows: List[Dict[str, Any]]
    training_pair_rows: List[Dict[str, Any]]
    feature_diagnostic_rows: List[Dict[str, Any]]
    inner_selection_rows: List[Dict[str, Any]]
    decision_rows: List[Dict[str, Any]]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _finite_mean(values: Sequence[float], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float(default)


def _stable_argmin(row: np.ndarray) -> int:
    order = np.lexsort((np.arange(row.shape[0], dtype=np.int64), row))
    return int(order[0])


def _selected_experts_from_scores(score_matrix: np.ndarray, candidate_domains: Sequence[int]) -> np.ndarray:
    candidates = np.asarray([int(v) for v in candidate_domains], dtype=np.int64)
    selected = [int(candidates[_stable_argmin(np.asarray(row, dtype=np.float64))]) for row in np.asarray(score_matrix)]
    return np.asarray(selected, dtype=np.int64)


def _resolve_group_ids(
    *,
    sample_metadata: Sequence[Mapping[str, Any]] | None,
    sample_indices: np.ndarray,
    group_key_candidates: Sequence[str],
) -> Tuple[List[str], str, int]:
    if sample_metadata is None:
        return [], "", 0
    indices = [int(v) for v in np.asarray(sample_indices, dtype=np.int64).tolist()]
    for key in group_key_candidates:
        key_s = str(key).strip()
        if not key_s:
            continue
        values: List[str] = []
        valid = True
        for idx in indices:
            if idx < 0 or idx >= len(sample_metadata):
                valid = False
                break
            value = sample_metadata[idx].get(key_s, "")
            text = str(value).strip()
            if not text or text.lower() == "nan":
                valid = False
                break
            values.append(text)
        if valid and values:
            return values, key_s, len(set(values))
    return [], "", 0


def _group_macro_agreement(selected: np.ndarray, reference: np.ndarray, group_ids: Sequence[str]) -> Tuple[float, int]:
    if not group_ids:
        return float("nan"), 0
    by_group: Dict[str, List[float]] = {}
    for idx, group in enumerate(group_ids):
        if int(idx) >= int(selected.shape[0]) or int(idx) >= int(reference.shape[0]):
            continue
        by_group.setdefault(str(group), []).append(float(int(selected[int(idx)] == reference[int(idx)])))
    if not by_group:
        return float("nan"), 0
    return _finite_mean([_finite_mean(vals, float("nan")) for vals in by_group.values()], float("nan")), len(by_group)


def _agreement_score(query_agreement: float, group_agreement: float) -> float:
    if not np.isfinite(float(query_agreement)) or not np.isfinite(float(group_agreement)):
        return float("nan")
    return float(min(float(query_agreement), float(group_agreement)))


def _target_batch_agreement_policy(
    *,
    source_inner_selected_method: str,
    method_predictions: Mapping[str, np.ndarray],
    candidate_domains: Sequence[int],
    test_idx: np.ndarray,
    sample_metadata: Sequence[Mapping[str, Any]] | None,
    pairwise_cfg: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    cfg = _target_batch_agreement_cfg(pairwise_cfg)
    threshold = float(cfg["agreement_threshold"])
    reference_method = str(cfg["reference_method"])
    gate_scope = str(cfg.get("gate_scope", "rank_margin_only")).strip().lower()
    min_query_count = int(cfg["min_query_count"])
    min_group_count = int(cfg["min_group_count"])
    gate_base = {
        "gate_scope": str(gate_scope),
        "agreement_threshold": float(threshold),
        "agreement_threshold_source": str(cfg["agreement_threshold_source"]),
        "agreement_gate_applied": 0,
        "agreement_gate_passed": 0,
        "agreement_gate_skipped_due_to_small_batch": 0,
        "gate_num_queries": int(np.asarray(test_idx).shape[0]),
        "gate_num_groups": 0,
        "gate_group_key": "",
        "selected_vs_raw_agreement_rate_query_weighted": float("nan"),
        "selected_vs_raw_agreement_rate_group_macro": float("nan"),
        "agreement_reference_methods": "",
        "agreement_reference_best_method": "",
        "selected_vs_reference_agreement_rate_query_weighted": float("nan"),
        "selected_vs_reference_agreement_rate_group_macro": float("nan"),
        "selected_vs_reference_best_agreement": float("nan"),
        "selected_vs_reference_mean_agreement": float("nan"),
        "selected_vs_reference_min_agreement": float("nan"),
        "raw_peer_agreement_with_rank_margin_unweighted": float("nan"),
        "raw_peer_agreement_with_rank_margin_weighted": float("nan"),
        "used_target_embeddings_for_gate": 1,
        "used_target_group_ids_for_gate": 0,
        "used_target_labels_for_gate": 0,
        "used_target_nelbo_for_gate": 0,
        "used_target_support_for_gate": 0,
        "used_target_fitting_for_gate": 0,
        "used_target_normalization_for_gate": 0,
        "heldout_target_nelbo_used_for_selection": 0,
        "agreement_gate_reason": "",
    }
    selected = str(source_inner_selected_method)
    if selected == BASELINE_METHOD:
        gate_base["agreement_gate_reason"] = "source_inner_selected_baseline"
        return BASELINE_METHOD, gate_base
    if gate_scope != "all_nonbaseline" and selected not in {RANK_MARGIN_UNWEIGHTED, RANK_MARGIN_WEIGHTED}:
        gate_base["agreement_gate_reason"] = "selected_variant_not_rank_margin_gate_not_required"
        gate_base["agreement_gate_passed"] = 1
        return selected, gate_base

    if selected == RAW_AE_WEIGHTED and gate_scope == "all_nonbaseline":
        reference_methods = [RANK_MARGIN_UNWEIGHTED, RANK_MARGIN_WEIGHTED]
    else:
        reference_methods = [reference_method]
    reference_methods = [str(method) for method in reference_methods if str(method) != selected]
    gate_base["agreement_reference_methods"] = "|".join(reference_methods)

    if selected not in method_predictions:
        gate_base["agreement_gate_reason"] = "missing_selected_or_reference_predictions"
        return BASELINE_METHOD, gate_base
    available_reference_methods = [method for method in reference_methods if method in method_predictions]
    if not available_reference_methods:
        gate_base["agreement_gate_reason"] = "missing_selected_or_reference_predictions"
        return BASELINE_METHOD, gate_base

    selected_experts = _selected_experts_from_scores(method_predictions[selected], candidate_domains)
    group_ids, group_key, group_count = _resolve_group_ids(
        sample_metadata=sample_metadata,
        sample_indices=np.asarray(test_idx, dtype=np.int64),
        group_key_candidates=cfg["group_key_candidates"],
    )
    agreement_rows: List[Tuple[str, float, float, float]] = []
    for method in available_reference_methods:
        reference_experts = _selected_experts_from_scores(method_predictions[method], candidate_domains)
        query_agreement = float(np.mean(selected_experts == reference_experts)) if selected_experts.size else float("nan")
        group_macro, resolved_group_count = _group_macro_agreement(selected_experts, reference_experts, group_ids)
        group_count = int(resolved_group_count or group_count)
        score = _agreement_score(query_agreement, group_macro)
        agreement_rows.append((str(method), float(query_agreement), float(group_macro), float(score)))

    if not agreement_rows:
        gate_base["agreement_gate_reason"] = "missing_selected_or_reference_predictions"
        return BASELINE_METHOD, gate_base
    # Higher min(query agreement, group agreement) is the best-peer agreement.
    best_method, best_query_agreement, best_group_macro, best_score = sorted(
        agreement_rows,
        key=lambda item: (-float(item[3]) if np.isfinite(float(item[3])) else float("inf"), _SIMPLER_METHOD_ORDER.get(item[0], 10**6), item[0]),
    )[0]
    agreement_scores = [float(row[3]) for row in agreement_rows]
    raw_peer = {method: score for method, _q, _g, score in agreement_rows if selected == RAW_AE_WEIGHTED}
    gate_base.update(
        {
            "agreement_gate_applied": 1,
            "gate_num_groups": int(group_count),
            "gate_group_key": str(group_key),
            "selected_vs_raw_agreement_rate_query_weighted": float(best_query_agreement) if best_method == reference_method else float("nan"),
            "selected_vs_raw_agreement_rate_group_macro": float(best_group_macro) if best_method == reference_method else float("nan"),
            "agreement_reference_best_method": str(best_method),
            "selected_vs_reference_agreement_rate_query_weighted": float(best_query_agreement),
            "selected_vs_reference_agreement_rate_group_macro": float(best_group_macro),
            "selected_vs_reference_best_agreement": float(best_score),
            "selected_vs_reference_mean_agreement": _finite_mean(agreement_scores, float("nan")),
            "selected_vs_reference_min_agreement": float(min(agreement_scores)) if agreement_scores else float("nan"),
            "raw_peer_agreement_with_rank_margin_unweighted": float(raw_peer.get(RANK_MARGIN_UNWEIGHTED, float("nan"))),
            "raw_peer_agreement_with_rank_margin_weighted": float(raw_peer.get(RANK_MARGIN_WEIGHTED, float("nan"))),
            "used_target_group_ids_for_gate": int(group_count > 0),
        }
    )
    if int(selected_experts.size) < min_query_count or group_count < min_group_count:
        gate_base["agreement_gate_skipped_due_to_small_batch"] = 1
        gate_base["agreement_gate_reason"] = "agreement_gate_skipped_due_to_small_batch"
        return BASELINE_METHOD, gate_base
    passed = bool(float(best_query_agreement) >= threshold and float(best_group_macro) >= threshold)
    gate_base["agreement_gate_passed"] = int(passed)
    gate_base["agreement_gate_reason"] = "agreement_gate_passed" if passed else "agreement_gate_blocked_low_agreement"
    return selected if passed else BASELINE_METHOD, gate_base


def _metadata_selected_expert(query_domain: int, candidate_domains: Sequence[int]) -> int:
    candidates = [int(v) for v in candidate_domains]
    exact = [d for d in candidates if int(d) == int(query_domain)]
    return int(exact[0]) if exact else int(sorted(candidates)[0])


def _metadata_features(q: np.ndarray, e: np.ndarray, sample_domains: np.ndarray) -> np.ndarray:
    span = max(float(np.max(sample_domains) - np.min(sample_domains)), 1.0)
    abs_diff = np.abs(q.astype(np.float64) - e.astype(np.float64)) / span
    exact = (q == e).astype(np.float64)
    return np.stack([abs_diff, exact], axis=1)


def _ae_rank_margin_features(
    *,
    ae_zscore_matrix: np.ndarray,
    sample_indices: np.ndarray,
    candidate_domains: Sequence[int],
    expert_domains: Sequence[int],
) -> Tuple[np.ndarray, List[str]]:
    domain_to_col = {int(domain): int(i) for i, domain in enumerate(expert_domains)}
    candidate_cols = [domain_to_col[int(domain)] for domain in candidate_domains]
    ae = np.asarray(ae_zscore_matrix[np.asarray(sample_indices, dtype=np.int64)][:, candidate_cols], dtype=np.float64)
    n_rows, n_candidates = ae.shape
    ranks = np.zeros_like(ae, dtype=np.float64)
    best_scores = np.zeros((n_rows,), dtype=np.float64)
    second_scores = np.zeros((n_rows,), dtype=np.float64)
    med_scores = np.median(ae, axis=1)
    best_mask = np.zeros_like(ae, dtype=np.float64)
    second_mask = np.zeros_like(ae, dtype=np.float64)
    margins = np.zeros((n_rows,), dtype=np.float64)
    tie = np.arange(n_candidates, dtype=np.int64)
    for i in range(n_rows):
        order = np.lexsort((tie, ae[i, :]))
        rank_row = np.empty((n_candidates,), dtype=np.float64)
        rank_row[order] = np.arange(1, n_candidates + 1, dtype=np.float64)
        ranks[i, :] = rank_row
        best_idx = int(order[0])
        second_idx = int(order[1]) if n_candidates > 1 else best_idx
        best_scores[i] = float(ae[i, best_idx])
        second_scores[i] = float(ae[i, second_idx])
        margins[i] = float(second_scores[i] - best_scores[i]) if n_candidates > 1 else float("inf")
        best_mask[i, best_idx] = 1.0
        if n_candidates > 1:
            second_mask[i, second_idx] = 1.0

    norm_den = max(float(n_candidates - 1), 1.0)
    candidate_z = ae.reshape(-1)
    rank_flat = ranks.reshape(-1)
    best_rep = np.repeat(best_scores, n_candidates)
    second_rep = np.repeat(second_scores, n_candidates)
    median_rep = np.repeat(med_scores, n_candidates)
    margin_rep = np.repeat(margins, n_candidates)
    best_flat = best_mask.reshape(-1)
    second_flat = second_mask.reshape(-1)
    not_best_flat = 1.0 - best_flat
    features = np.stack(
        [
            candidate_z,
            rank_flat,
            (rank_flat - 1.0) / norm_den,
            best_flat,
            second_flat,
            candidate_z - best_rep,
            candidate_z - second_rep,
            candidate_z - median_rep,
            best_flat * margin_rep,
            second_flat * margin_rep,
            not_best_flat * margin_rep,
        ],
        axis=1,
    ).astype(np.float64, copy=False)
    names = [
        "candidate_ae_zscore",
        "candidate_ae_rank",
        "candidate_normalized_ae_rank",
        "is_ae_best_candidate",
        "is_ae_second_candidate",
        "candidate_minus_best_ae_zscore",
        "candidate_minus_second_ae_zscore",
        "candidate_minus_median_ae_zscore",
        "ae_margin_if_best",
        "ae_margin_if_second",
        "ae_margin_if_not_best",
    ]
    return features, names


def _build_pairwise_features_for_candidates(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    sample_indices: np.ndarray,
    candidate_domains: Sequence[int],
    expert_domains: Sequence[int],
    embedding_feature_dim: int,
    expert_feature_dim: int,
    ae_zscore_matrix: np.ndarray,
    feature_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    x, q, e, s = _build_pair_features(
        sample_embeddings=embeddings,
        sample_domains=sample_domains,
        sample_indices=sample_indices,
        expert_domains=candidate_domains,
        expert_id_domains=expert_domains,
        include_metadata_features=True,
    )
    expert_oh = x[:, embedding_feature_dim : embedding_feature_dim + expert_feature_dim]
    meta = _metadata_features(q, e, sample_domains)
    latent = np.concatenate([x[:, :embedding_feature_dim], expert_oh], axis=1)
    combined = np.concatenate([latent, meta], axis=1)
    ae_rank, ae_names = _ae_rank_margin_features(
        ae_zscore_matrix=ae_zscore_matrix,
        sample_indices=sample_indices,
        candidate_domains=candidate_domains,
        expert_domains=expert_domains,
    )
    if str(feature_mode) == "raw_ae_combined":
        features = np.concatenate([ae_rank[:, :1], combined], axis=1)
        names = ["candidate_ae_zscore"] + [f"combined_{i}" for i in range(combined.shape[1])]
    elif str(feature_mode) == "rank_margin_combined":
        features = np.concatenate([ae_rank, combined], axis=1)
        names = ae_names + [f"combined_{i}" for i in range(combined.shape[1])]
    else:
        raise ProtocolError(f"Unknown pairwise AE-combined v2 feature_mode={feature_mode}")
    return features, q, e, s, names


def _build_training_features(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    train_indices: np.ndarray,
    expert_domains: Sequence[int],
    outer_heldout_domain: int,
    globally_excluded_domains: Sequence[int],
    embedding_feature_dim: int,
    expert_feature_dim: int,
    ae_zscore_matrix: np.ndarray,
    feature_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, List[str]]:
    x_parts: List[np.ndarray] = []
    q_parts: List[np.ndarray] = []
    e_parts: List[np.ndarray] = []
    s_parts: List[np.ndarray] = []
    feature_names: List[str] = []
    candidate_counts: set[int] = set()
    excluded_global = {int(outer_heldout_domain), *[int(v) for v in globally_excluded_domains]}
    for query_domain in sorted(set(int(sample_domains[int(i)]) for i in train_indices.tolist())):
        if int(query_domain) in excluded_global:
            continue
        domain_indices = train_indices[sample_domains[train_indices] == int(query_domain)]
        excluded = sorted(excluded_global | {int(query_domain)})
        fold = FoldCandidateSet.for_heldout_domain(
            heldout_domain=int(outer_heldout_domain),
            expert_domains=expert_domains,
            excluded_domains=excluded,
        )
        if len(fold.candidate_expert_domains) < 2:
            continue
        x, q, e, s, names = _build_pairwise_features_for_candidates(
            embeddings=embeddings,
            sample_domains=sample_domains,
            sample_indices=domain_indices,
            candidate_domains=fold.candidate_expert_domains,
            expert_domains=expert_domains,
            embedding_feature_dim=embedding_feature_dim,
            expert_feature_dim=expert_feature_dim,
            ae_zscore_matrix=ae_zscore_matrix,
            feature_mode=feature_mode,
        )
        x_parts.append(x)
        q_parts.append(q)
        e_parts.append(e)
        s_parts.append(s)
        feature_names = names
        candidate_counts.add(int(len(fold.candidate_expert_domains)))
    if not x_parts or len(candidate_counts) != 1:
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            0,
            feature_names,
        )
    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(q_parts, axis=0),
        np.concatenate(e_parts, axis=0),
        np.concatenate(s_parts, axis=0),
        int(next(iter(candidate_counts))),
        feature_names,
    )


def _pair_weight(
    *,
    better_nelbo: float,
    worse_nelbo: float,
    source_inner_median_abs_nelbo: float,
    cfg: Mapping[str, Any],
) -> float:
    scale = max(float(source_inner_median_abs_nelbo), 1.0e-12)
    delta_scaled = abs(float(worse_nelbo) - float(better_nelbo)) / scale
    alpha = float(cfg.get("pair_weight_alpha", 4.0))
    delta_clip = float(cfg.get("pair_weight_delta_clip", 0.50))
    w_min = float(cfg.get("pair_weight_min", 1.0))
    w_max = float(cfg.get("pair_weight_max", 3.0))
    return float(np.clip(1.0 + alpha * min(delta_scaled, delta_clip), w_min, w_max))


def _build_utility_weighted_training_pairs(
    *,
    y_train: np.ndarray,
    q_train: np.ndarray,
    s_train: np.ndarray,
    experts_per_sample: int,
    near_tie_delta: float,
    hard_pair_fraction: float,
    utility_pair_fraction: float,
    random_pair_fraction: float,
    max_pairs_per_sample: int,
    max_pairs_per_domain: int,
    seed: int,
    cfg: Mapping[str, Any],
) -> Tuple[List[Tuple[int, int]], np.ndarray, List[Dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    pairs: List[Tuple[int, int]] = []
    weights: List[float] = []
    diagnostics: List[Dict[str, Any]] = []
    domain_pair_counts: Dict[int, int] = {}
    frac_sum = max(float(hard_pair_fraction) + float(utility_pair_fraction) + float(random_pair_fraction), 1.0e-12)
    hard_ratio = float(hard_pair_fraction) / frac_sum
    utility_ratio = float(utility_pair_fraction) / frac_sum
    source_scale = float(np.median(np.abs(np.asarray(y_train, dtype=np.float64)))) if y_train.size else 1.0
    if not np.isfinite(source_scale) or source_scale <= 1.0e-12:
        source_scale = 1.0

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
                candidates.append((ii, jj, diff) if yi < yj else (jj, ii, diff))
        if not candidates:
            diagnostics.append(
                {
                    "sample_index": int(sample_index),
                    "query_domain": int(query_domain),
                    "n_candidates": 0,
                    "n_selected": 0,
                    "n_hard_selected": 0,
                    "n_utility_selected": 0,
                    "n_random_selected": 0,
                    "mean_pair_weight": float("nan"),
                    "dropped_by_domain_cap": 0,
                    "source_inner_median_abs_nelbo": float(source_scale),
                }
            )
            continue
        target = min(int(max_pairs_per_sample), len(candidates))
        n_hard = min(int(round(target * hard_ratio)), target)
        n_utility = min(int(round(target * utility_ratio)), target - n_hard)
        hard_sorted = sorted(candidates, key=lambda x: float(x[2]))
        utility_sorted = sorted(candidates, key=lambda x: float(x[2]), reverse=True)
        selected: List[Tuple[int, int, float, str]] = []
        used: set[Tuple[int, int]] = set()
        for better, worse, diff in hard_sorted:
            if len([v for v in selected if v[3] == "hard"]) >= n_hard:
                break
            key = (int(better), int(worse))
            if key not in used:
                selected.append((int(better), int(worse), float(diff), "hard"))
                used.add(key)
        for better, worse, diff in utility_sorted:
            if len([v for v in selected if v[3] == "utility"]) >= n_utility:
                break
            key = (int(better), int(worse))
            if key not in used:
                selected.append((int(better), int(worse), float(diff), "utility"))
                used.add(key)
        remaining = [(b, w, d) for b, w, d in candidates if (int(b), int(w)) not in used]
        n_random = min(target - len(selected), len(remaining))
        if n_random > 0:
            for k in rng.choice(len(remaining), size=n_random, replace=False).tolist():
                b, w, d = remaining[int(k)]
                selected.append((int(b), int(w), float(d), "random"))

        dropped = 0
        added_weights: List[float] = []
        added_by_source = {"hard": 0, "utility": 0, "random": 0}
        for better, worse, _diff, source in selected:
            cur = int(domain_pair_counts.get(query_domain, 0))
            if cur >= int(max_pairs_per_domain):
                dropped += 1
                continue
            weight = _pair_weight(
                better_nelbo=float(y_train[int(better)]),
                worse_nelbo=float(y_train[int(worse)]),
                source_inner_median_abs_nelbo=source_scale,
                cfg=cfg,
            )
            pairs.append((int(better), int(worse)))
            weights.append(float(weight))
            added_weights.append(float(weight))
            added_by_source[str(source)] += 1
            domain_pair_counts[query_domain] = cur + 1

        diagnostics.append(
            {
                "sample_index": int(sample_index),
                "query_domain": int(query_domain),
                "n_candidates": int(len(candidates)),
                "n_selected": int(sum(added_by_source.values())),
                "n_hard_selected": int(added_by_source["hard"]),
                "n_utility_selected": int(added_by_source["utility"]),
                "n_random_selected": int(added_by_source["random"]),
                "mean_pair_weight": _finite_mean(added_weights),
                "dropped_by_domain_cap": int(dropped),
                "source_inner_median_abs_nelbo": float(source_scale),
            }
        )
    return pairs, np.asarray(weights, dtype=np.float64), diagnostics


def _train_predict_variant(
    *,
    method: str,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    domain_to_idx: Mapping[int, int],
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    outer_heldout_domain: int,
    globally_excluded_domains: Sequence[int],
    eval_candidate_domains: Sequence[int],
    embedding_feature_dim: int,
    expert_feature_dim: int,
    ae_zscore_matrix: np.ndarray,
    pairwise_cfg: Mapping[str, Any],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    v2_cfg = dict((pairwise_cfg.get("utility_weighted_v2", {}) or {}))
    feature_mode = "rank_margin_combined" if method in {RANK_MARGIN_UNWEIGHTED, RANK_MARGIN_WEIGHTED} else "raw_ae_combined"
    weighted = method in {RAW_AE_WEIGHTED, RANK_MARGIN_WEIGHTED}
    x_train, q_train, e_train, s_train, experts_per_sample, feature_names = _build_training_features(
        embeddings=embeddings,
        sample_domains=sample_domains,
        train_indices=np.asarray(train_idx, dtype=np.int64),
        expert_domains=expert_domains,
        outer_heldout_domain=int(outer_heldout_domain),
        globally_excluded_domains=globally_excluded_domains,
        embedding_feature_dim=int(embedding_feature_dim),
        expert_feature_dim=int(expert_feature_dim),
        ae_zscore_matrix=ae_zscore_matrix,
        feature_mode=feature_mode,
    )
    if x_train.size == 0 or int(experts_per_sample) < 2:
        raise ProtocolError(f"No v2 pairwise training rows for method={method}")
    x_eval, _q_eval, _e_eval, _s_eval, _eval_names = _build_pairwise_features_for_candidates(
        embeddings=embeddings,
        sample_domains=sample_domains,
        sample_indices=np.asarray(eval_idx, dtype=np.int64),
        candidate_domains=eval_candidate_domains,
        expert_domains=expert_domains,
        embedding_feature_dim=int(embedding_feature_dim),
        expert_feature_dim=int(expert_feature_dim),
        ae_zscore_matrix=ae_zscore_matrix,
        feature_mode=feature_mode,
    )
    x_train_z, x_eval_z = _zscore_features(x_train, x_eval)
    y_train = true_nelbo[s_train, [int(domain_to_idx[int(ed)]) for ed in e_train]]
    if str(method) == BASELINE_METHOD:
        pairs, pair_diags = _build_pairwise_training_pairs(
            y_train=y_train,
            q_train=q_train,
            s_train=s_train,
            experts_per_sample=int(experts_per_sample),
            near_tie_delta=float(pairwise_cfg.get("near_tie_delta", 0.0)),
            hard_pair_fraction=float(pairwise_cfg.get("hard_pair_fraction", 0.5)),
            random_pair_fraction=float(pairwise_cfg.get("random_pair_fraction", 0.5)),
            max_pairs_per_sample=int(pairwise_cfg.get("max_pairs_per_sample", 12)),
            max_pairs_per_domain=int(pairwise_cfg.get("max_pairs_per_domain", 5000)),
            seed=int(seed) + int(outer_heldout_domain),
        )
        weights = np.ones((len(pairs),), dtype=np.float64)
        for row in pair_diags:
            row["n_utility_selected"] = 0
            row["mean_pair_weight"] = 1.0 if int(row.get("n_selected", 0)) > 0 else float("nan")
            row["source_inner_median_abs_nelbo"] = float(
                np.median(np.abs(np.asarray(y_train, dtype=np.float64)))
            )
    else:
        pairs, weights, pair_diags = _build_utility_weighted_training_pairs(
            y_train=y_train,
            q_train=q_train,
            s_train=s_train,
            experts_per_sample=int(experts_per_sample),
            near_tie_delta=float(pairwise_cfg.get("near_tie_delta", 0.0)),
            hard_pair_fraction=float(v2_cfg.get("hard_pair_fraction", 0.40)),
            utility_pair_fraction=float(v2_cfg.get("utility_pair_fraction", 0.40)),
            random_pair_fraction=float(v2_cfg.get("random_pair_fraction", 0.20)),
            max_pairs_per_sample=int(pairwise_cfg.get("max_pairs_per_sample", 12)),
            max_pairs_per_domain=int(pairwise_cfg.get("max_pairs_per_domain", 5000)),
            seed=int(seed) + int(outer_heldout_domain),
            cfg=v2_cfg,
        )
    if not pairs:
        raise ProtocolError(f"No v2 pairwise training pairs for method={method}")
    feature_diag_rows = _feature_diagnostics(method=method, x_train=x_train_z, pairs=pairs, feature_names=feature_names)
    ranker = _PairwiseRanker(
        seed=int(seed),
        hidden_dim=int(pairwise_cfg.get("hidden_dim", 128)),
        epochs=int(pairwise_cfg.get("epochs", 40)),
        lr=float(pairwise_cfg.get("lr", 1e-3)),
        batch_size=int(pairwise_cfg.get("batch_size", 2048)),
        margin=float(pairwise_cfg.get("margin", 1.0)),
        device=str(pairwise_cfg.get("device", "auto")),
    )
    ranker.fit(x_train_z, pairs, pair_weights=weights if weighted else None)
    pred = ranker.predict(x_eval_z).reshape(int(eval_idx.shape[0]), int(len(eval_candidate_domains)))
    diag_rows: List[Dict[str, Any]] = []
    mean_by_query: Dict[int, List[float]] = {}
    for row in pair_diags:
        query = int(row["query_domain"])
        if np.isfinite(float(row.get("mean_pair_weight", float("nan")))):
            mean_by_query.setdefault(query, []).append(float(row["mean_pair_weight"]))
        diag = {
            "method": str(method),
            "outer_heldout_domain": int(outer_heldout_domain),
            "weighted_loss": int(weighted),
            "feature_mode": str(feature_mode),
            **row,
        }
        diag_rows.append(diag)
    for query, vals in mean_by_query.items():
        diag_rows.append(
            {
                "method": str(method),
                "outer_heldout_domain": int(outer_heldout_domain),
                "query_domain": int(query),
                "diagnostic": "mean_pair_weight_by_query_domain",
                "mean_pair_weight_by_query_domain": _finite_mean(vals),
            }
        )
    expert_weight_map: Dict[int, List[float]] = {}
    for pair_idx, (better, worse) in enumerate(pairs):
        weight = float(weights[int(pair_idx)]) if int(pair_idx) < int(weights.shape[0]) else 1.0
        expert_weight_map.setdefault(int(e_train[int(better)]), []).append(weight)
        expert_weight_map.setdefault(int(e_train[int(worse)]), []).append(weight)
    for expert, vals in sorted(expert_weight_map.items()):
        diag_rows.append(
            {
                "method": str(method),
                "outer_heldout_domain": int(outer_heldout_domain),
                "candidate_expert": int(expert),
                "diagnostic": "mean_pair_weight_by_candidate_expert",
                "mean_pair_weight_by_candidate_expert": _finite_mean(vals),
            }
        )
    return pred, weights, diag_rows, feature_diag_rows, feature_names


def _feature_diagnostics(
    *,
    method: str,
    x_train: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    feature_names: Sequence[str],
) -> List[Dict[str, Any]]:
    if not pairs or x_train.size == 0:
        return []
    pair_arr = np.asarray(pairs, dtype=np.int64)
    diffs = x_train[pair_arr[:, 0], :] - x_train[pair_arr[:, 1], :]
    rows: List[Dict[str, Any]] = []
    for idx, name in enumerate(feature_names):
        nonzero = np.abs(diffs[:, idx]) > 1.0e-12
        rows.append(
            {
                "method": str(method),
                "feature_name": str(name),
                "feature_nonzero_rate_after_pairwise_difference": float(np.mean(nonzero)),
            }
        )
    rows.append(
        {
            "method": str(method),
            "feature_name": "__all__",
            "feature_nonzero_rate_after_pairwise_difference": float(np.mean(np.abs(diffs) > 1.0e-12)),
        }
    )
    return rows


def _metrics_from_sample_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    return {
        "top1_oracle_hit": _finite_mean([float(r.get("top1_oracle_hit", float("nan"))) for r in rows], 0.0),
        "spearman": _finite_mean([float(r.get("spearman", float("nan"))) for r in rows], float("nan")),
        "mean_oracle_gap_pct": _finite_mean([float(r.get("oracle_gap_pct", float("nan"))) for r in rows], 0.0),
    }


def _evaluate_variant_on_indices(
    *,
    method: str,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    domain_to_idx: Mapping[int, int],
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    outer_heldout_domain: int,
    globally_excluded_domains: Sequence[int],
    eval_fold: FoldCandidateSet,
    global_eval: np.ndarray,
    embedding_feature_dim: int,
    expert_feature_dim: int,
    ae_zscore_matrix: np.ndarray,
    pairwise_cfg: Mapping[str, Any],
    seed: int,
    tie_policy: str,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    pred, _weights, _diag_rows, _feature_rows, _names = _train_predict_variant(
        method=method,
        embeddings=embeddings,
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        domain_to_idx=domain_to_idx,
        train_idx=train_idx,
        eval_idx=eval_idx,
        outer_heldout_domain=int(outer_heldout_domain),
        globally_excluded_domains=globally_excluded_domains,
        eval_candidate_domains=eval_fold.candidate_expert_domains,
        embedding_feature_dim=int(embedding_feature_dim),
        expert_feature_dim=int(expert_feature_dim),
        ae_zscore_matrix=ae_zscore_matrix,
        pairwise_cfg=pairwise_cfg,
        seed=int(seed),
    )
    true_eval = eval_fold.slice_nelbo(true_nelbo, eval_idx)
    return _selection_metrics(
        method=method,
        query_domains=sample_domains[eval_idx],
        expert_domains=eval_fold.candidate_expert_domains,
        score_matrix=pred,
        true_nelbo_matrix=true_eval,
        fold=eval_fold,
        global_true_nelbo_matrix=global_eval,
        global_expert_domains=expert_domains,
        tie_policy=tie_policy,
    )


def _source_inner_selection(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    domain_to_idx: Mapping[int, int],
    train_idx: np.ndarray,
    outer_fold: FoldCandidateSet,
    embedding_feature_dim: int,
    expert_feature_dim: int,
    ae_zscore_matrix: np.ndarray,
    pairwise_cfg: Mapping[str, Any],
    seed: int,
    tie_policy: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    strict_mode = _uses_strict_source_inner_gates(pairwise_cfg)
    strict = _strict_cfg(pairwise_cfg)
    source_domains = sorted(set(int(sample_domains[int(i)]) for i in np.asarray(train_idx, dtype=np.int64).tolist()))
    inner_rows: List[Dict[str, Any]] = []
    per_method_by_inner: Dict[str, List[Tuple[int, Dict[str, float]]]] = {method: [] for method in V2_CANDIDATE_METHODS}
    for inner_domain in source_domains:
        inner_eval_idx = np.asarray(
            [int(i) for i in np.asarray(train_idx, dtype=np.int64).tolist() if int(sample_domains[int(i)]) == int(inner_domain)],
            dtype=np.int64,
        )
        inner_train_idx = np.asarray(
            [int(i) for i in np.asarray(train_idx, dtype=np.int64).tolist() if int(sample_domains[int(i)]) != int(inner_domain)],
            dtype=np.int64,
        )
        if inner_eval_idx.size == 0 or inner_train_idx.size == 0:
            continue
        inner_fold = FoldCandidateSet.for_heldout_domain(
            heldout_domain=int(outer_fold.heldout_domain),
            expert_domains=expert_domains,
            excluded_domains=[int(inner_domain)],
        )
        if int(outer_fold.heldout_domain) in set(inner_fold.candidate_expert_domains):
            raise ProtocolError("Inner v2 candidate pool includes outer target expert")
        if int(inner_domain) in set(inner_fold.candidate_expert_domains):
            raise ProtocolError("Inner v2 candidate pool includes query-self expert")
        if len(inner_fold.candidate_expert_domains) < 2:
            continue
        global_eval = true_nelbo[inner_eval_idx]
        method_rows: Dict[str, Dict[str, float]] = {}
        for method in V2_CANDIDATE_METHODS:
            metrics, rows = _evaluate_variant_on_indices(
                method=method,
                embeddings=embeddings,
                sample_domains=sample_domains,
                true_nelbo=true_nelbo,
                expert_domains=expert_domains,
                domain_to_idx=domain_to_idx,
                train_idx=inner_train_idx,
                eval_idx=inner_eval_idx,
                outer_heldout_domain=int(outer_fold.heldout_domain),
                globally_excluded_domains=[int(inner_domain)],
                eval_fold=inner_fold,
                global_eval=global_eval,
                embedding_feature_dim=int(embedding_feature_dim),
                expert_feature_dim=int(expert_feature_dim),
                ae_zscore_matrix=ae_zscore_matrix,
                pairwise_cfg=pairwise_cfg,
                seed=int(seed) + int(inner_domain) * 17,
                tie_policy=tie_policy,
            )
            _ = rows
            method_rows[method] = metrics
            per_method_by_inner[method].append((int(inner_domain), metrics))
        baseline = method_rows.get(BASELINE_METHOD)
        if not baseline:
            continue
        for method in V2_CANDIDATE_METHODS:
            candidate = method_rows.get(method)
            if not candidate:
                continue
            inner_rows.append(
                {
                    "seed": int(seed),
                    "outer_heldout_center": int(outer_fold.heldout_domain),
                    "inner_heldout_source_center": int(inner_domain),
                    "baseline_method": BASELINE_METHOD,
                    "candidate_method": str(method),
                    "source_inner_macro_gap_baseline": float(baseline["mean_oracle_gap_pct"]),
                    "source_inner_macro_gap_candidate": float(candidate["mean_oracle_gap_pct"]),
                    "source_inner_macro_top1_baseline": float(baseline["top1_oracle_hit"]),
                    "source_inner_macro_top1_candidate": float(candidate["top1_oracle_hit"]),
                    "source_inner_macro_spearman_baseline": float(baseline["spearman"]),
                    "source_inner_macro_spearman_candidate": float(candidate["spearman"]),
                    "inner_gap_delta": float(baseline["mean_oracle_gap_pct"] - candidate["mean_oracle_gap_pct"]),
                    "inner_top1_delta": float(candidate["top1_oracle_hit"] - baseline["top1_oracle_hit"]),
                    "inner_spearman_delta": float(candidate["spearman"] - baseline["spearman"]),
                    "inner_center_gap_degradation_pp": float(candidate["mean_oracle_gap_pct"] - baseline["mean_oracle_gap_pct"]),
                    "heldout_target_nelbo_used_for_selection": 0,
                }
            )

    baseline_units = per_method_by_inner.get(BASELINE_METHOD, [])
    if not baseline_units:
        selected = BASELINE_METHOD
        scored: List[Dict[str, Any]] = []
    else:
        scored = []
        baseline_by_inner = {inner: metrics for inner, metrics in baseline_units}
        for method in V2_CANDIDATE_METHODS:
            units = per_method_by_inner.get(method, [])
            if not units:
                continue
            gap_deltas = []
            top1_deltas = []
            spearman_deltas = []
            degradations = []
            baseline_gaps = []
            candidate_gaps = []
            baseline_top1s = []
            candidate_top1s = []
            baseline_spearmans = []
            candidate_spearmans = []
            for inner, metrics in units:
                base = baseline_by_inner.get(inner)
                if not base:
                    continue
                gap_delta = float(base["mean_oracle_gap_pct"] - metrics["mean_oracle_gap_pct"])
                gap_deltas.append(gap_delta)
                top1_deltas.append(float(metrics["top1_oracle_hit"] - base["top1_oracle_hit"]))
                spearman_deltas.append(float(metrics["spearman"] - base["spearman"]))
                degradations.append(float(metrics["mean_oracle_gap_pct"] - base["mean_oracle_gap_pct"]))
                baseline_gaps.append(float(base["mean_oracle_gap_pct"]))
                candidate_gaps.append(float(metrics["mean_oracle_gap_pct"]))
                baseline_top1s.append(float(base["top1_oracle_hit"]))
                candidate_top1s.append(float(metrics["top1_oracle_hit"]))
                baseline_spearmans.append(float(base["spearman"]))
                candidate_spearmans.append(float(metrics["spearman"]))
            macro_gap = _finite_mean(gap_deltas, 0.0)
            macro_top1 = _finite_mean(top1_deltas, 0.0)
            macro_spearman = _finite_mean(spearman_deltas, 0.0)
            worst_degradation = max(degradations) if degradations else float("inf")
            positive_count = sum(1 for v in gap_deltas if float(v) > 0.0)
            non_degrading_count = sum(1 for v in degradations if float(v) <= 0.0)
            unit_count = max(len(gap_deltas), 1)
            positive_rate = float(positive_count / unit_count)
            non_degrading_rate = float(non_degrading_count / unit_count)
            if strict_mode:
                passed_macro = macro_gap >= float(strict["min_macro_gap_reduction_pp"])
                passed_top1 = macro_top1 >= -float(strict["max_top1_drop_abs"])
                passed_spearman = macro_spearman >= -float(strict["max_spearman_drop_abs"])
                passed_worst = worst_degradation <= float(strict["max_worst_inner_center_gap_degradation_pp"])
                passed_positive = positive_rate >= float(strict["min_positive_inner_center_rate"])
                passed_non_degrading = non_degrading_rate >= float(strict["min_non_degrading_inner_center_rate"])
                passed_min_centers = positive_count >= int(strict["min_passing_inner_centers"])
                passed = bool(
                    method != BASELINE_METHOD
                    and passed_macro
                    and passed_top1
                    and passed_spearman
                    and passed_worst
                    and passed_positive
                    and passed_non_degrading
                    and passed_min_centers
                )
            else:
                passed_macro = macro_gap >= 0.0
                passed_top1 = macro_top1 >= -0.02
                passed_spearman = macro_spearman >= -0.03
                passed_worst = worst_degradation <= 1.0
                passed_positive = True
                passed_non_degrading = True
                passed_min_centers = True
                passed = bool(passed_macro and passed_top1 and passed_spearman and passed_worst)
            scored.append(
                {
                    "method": str(method),
                    "macro_gap": float(macro_gap),
                    "macro_top1": float(macro_top1),
                    "macro_spearman": float(macro_spearman),
                    "worst_degradation": float(worst_degradation),
                    "baseline_inner_macro_oracle_gap_pct": _finite_mean(baseline_gaps, float("nan")),
                    "candidate_inner_macro_oracle_gap_pct": _finite_mean(candidate_gaps, float("nan")),
                    "baseline_inner_macro_top1": _finite_mean(baseline_top1s, float("nan")),
                    "candidate_inner_macro_top1": _finite_mean(candidate_top1s, float("nan")),
                    "baseline_inner_macro_spearman": _finite_mean(baseline_spearmans, float("nan")),
                    "candidate_inner_macro_spearman": _finite_mean(candidate_spearmans, float("nan")),
                    "positive_inner_center_rate": float(positive_rate),
                    "non_degrading_inner_center_rate": float(non_degrading_rate),
                    "positive_inner_center_count": int(positive_count),
                    "inner_center_count": int(len(gap_deltas)),
                    "passed_macro_gap_gate": int(passed_macro),
                    "passed_top1_gate": int(passed_top1),
                    "passed_spearman_gate": int(passed_spearman),
                    "passed_worst_center_gate": int(passed_worst),
                    "passed_positive_center_rate_gate": int(passed_positive),
                    "passed_non_degrading_center_rate_gate": int(passed_non_degrading),
                    "passed_min_passing_inner_centers_gate": int(passed_min_centers),
                    "passed": int(passed),
                    "candidate_passed_strict_gate": int(passed) if strict_mode else 0,
                }
            )
        passing = [row for row in scored if int(row["passed"]) == 1]
        if not passing:
            selected = BASELINE_METHOD
        else:
            order = _STRICT_METHOD_ORDER if strict_mode else _SIMPLER_METHOD_ORDER
            selected = str(
                sorted(
                    passing,
                    key=lambda row: (
                        float(row["macro_gap"]),
                        float(row["macro_top1"]),
                        float(row["macro_spearman"]),
                        -float(order.get(str(row["method"]), 10**6)),
                    ),
                    reverse=True,
                )[0]["method"]
            )

    selected_score_by_method = {row["candidate_method"]: [] for row in inner_rows}
    scored_by_method = {str(row["method"]): row for row in scored}
    for row in inner_rows:
        selected_score_by_method.setdefault(str(row["candidate_method"]), []).append(row)
    for method, rows in selected_score_by_method.items():
        base_rows = [r for r in rows if str(r["candidate_method"]) == str(method)]
        if not base_rows:
            continue
        score = scored_by_method.get(str(method), {})
        worst = float(score.get("worst_degradation", float("nan")))
        passed = bool(int(score.get("passed", 0)))
        for row in base_rows:
            row["selected_method"] = str(selected)
            row["selected_variant"] = str(selected)
            row["selection_mode"] = _selection_mode(pairwise_cfg)
            row["selection_reason"] = (
                "selected_by_strict_source_inner_policy"
                if strict_mode and str(method) == str(selected) and str(selected) != BASELINE_METHOD
                else "fallback_to_pairwise_ae_combined"
                if strict_mode and str(selected) == BASELINE_METHOD
                else "selected_by_source_inner_policy"
                if str(method) == str(selected)
                else "not_selected"
            )
            row["inner_worst_center_gap_degradation"] = float(worst)
            row["gap_reduction_pp"] = float(score.get("macro_gap", float("nan")))
            row["baseline_inner_macro_oracle_gap_pct"] = float(score.get("baseline_inner_macro_oracle_gap_pct", float("nan")))
            row["candidate_inner_macro_oracle_gap_pct"] = float(score.get("candidate_inner_macro_oracle_gap_pct", float("nan")))
            row["baseline_inner_macro_top1"] = float(score.get("baseline_inner_macro_top1", float("nan")))
            row["candidate_inner_macro_top1"] = float(score.get("candidate_inner_macro_top1", float("nan")))
            row["baseline_inner_macro_spearman"] = float(score.get("baseline_inner_macro_spearman", float("nan")))
            row["candidate_inner_macro_spearman"] = float(score.get("candidate_inner_macro_spearman", float("nan")))
            row["positive_inner_center_rate"] = float(score.get("positive_inner_center_rate", float("nan")))
            row["non_degrading_inner_center_rate"] = float(score.get("non_degrading_inner_center_rate", float("nan")))
            row["passed_macro_gap_gate"] = int(score.get("passed_macro_gap_gate", 0))
            row["passed_top1_gate"] = int(score.get("passed_top1_gate", 0))
            row["passed_spearman_gate"] = int(score.get("passed_spearman_gate", 0))
            row["passed_worst_center_gate"] = int(score.get("passed_worst_center_gate", 0))
            row["passed_positive_center_rate_gate"] = int(score.get("passed_positive_center_rate_gate", 0))
            row["passed_non_degrading_center_rate_gate"] = int(score.get("passed_non_degrading_center_rate_gate", 0))
            row["passed_min_passing_inner_centers_gate"] = int(score.get("passed_min_passing_inner_centers_gate", 0))
            row["candidate_passed_no_harm_gate"] = int(passed)
            row["candidate_passed_strict_gate"] = int(score.get("candidate_passed_strict_gate", 0))
            row["fallback_to_baseline"] = int(str(selected) == BASELINE_METHOD)
            row["fallback_used"] = int(str(selected) == BASELINE_METHOD)
            row["selected_by_inner_policy"] = int(str(method) == str(selected))
    return selected, inner_rows


def run_pairwise_ae_combined_v2_for_fold(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    domain_to_idx: Mapping[int, int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: FoldCandidateSet,
    global_eval: np.ndarray,
    pairwise_cfg: Mapping[str, Any],
    seed: int,
    embedding_feature_dim: int,
    expert_feature_dim: int,
    tie_policy: str,
    ae_zscore_matrix: np.ndarray | None,
    sample_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> PairwiseAECombinedV2FoldOutputs:
    v2_cfg = _v2_cfg(pairwise_cfg)
    if not bool(pairwise_cfg.get("run_utility_weighted_v2", False)) or not bool(v2_cfg.get("enabled", False)):
        return PairwiseAECombinedV2FoldOutputs([], [], [], [], [], [])
    if ae_zscore_matrix is None:
        raise ProtocolError("pairwise AE-combined v2 requires autoencoder_proxy AE z-score matrix")
    primary_method = _primary_method(pairwise_cfg)
    selection_mode = _selection_mode(pairwise_cfg)

    source_inner_selected_method, inner_rows = _source_inner_selection(
        embeddings=embeddings,
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        domain_to_idx=domain_to_idx,
        train_idx=train_idx,
        outer_fold=fold,
        embedding_feature_dim=int(embedding_feature_dim),
        expert_feature_dim=int(expert_feature_dim),
        ae_zscore_matrix=ae_zscore_matrix,
        pairwise_cfg=pairwise_cfg,
        seed=int(seed),
        tie_policy=tie_policy,
    )

    sample_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    training_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    method_predictions: Dict[str, np.ndarray] = {}
    method_sample_rows: Dict[str, List[Dict[str, Any]]] = {}
    true_eval = fold.slice_nelbo(true_nelbo, test_idx)
    for method in V2_CANDIDATE_METHODS:
        pred, weights, diag_rows, feat_rows, _names = _train_predict_variant(
            method=method,
            embeddings=embeddings,
            sample_domains=sample_domains,
            true_nelbo=true_nelbo,
            expert_domains=expert_domains,
            domain_to_idx=domain_to_idx,
            train_idx=train_idx,
            eval_idx=test_idx,
            outer_heldout_domain=int(fold.heldout_domain),
            globally_excluded_domains=[],
            eval_candidate_domains=fold.candidate_expert_domains,
            embedding_feature_dim=int(embedding_feature_dim),
            expert_feature_dim=int(expert_feature_dim),
            ae_zscore_matrix=ae_zscore_matrix,
            pairwise_cfg=pairwise_cfg,
            seed=int(seed),
        )
        method_predictions[method] = pred
        metrics, rows = _selection_metrics(
            method=method,
            query_domains=sample_domains[test_idx],
            expert_domains=fold.candidate_expert_domains,
            score_matrix=pred,
            true_nelbo_matrix=true_eval,
            fold=fold,
            global_true_nelbo_matrix=global_eval,
            global_expert_domains=expert_domains,
            tie_policy=tie_policy,
        )
        _ = metrics
        for row in rows:
            row["sample_index"] = int(test_idx[int(row["sample_index"])])
            row["source_inner_selected_method"] = str(source_inner_selected_method)
            row["posthoc_diagnostic_only"] = int(method != BASELINE_METHOD)
            row["used_for_selection"] = 0
        method_sample_rows[method] = rows
        if method != BASELINE_METHOD:
            sample_rows.extend(rows)
        feature_rows.extend([{**row, "outer_heldout_center": int(fold.heldout_domain)} for row in feat_rows])
        for row in diag_rows:
            training_rows.append({**row, "seed": int(seed)})
        pred_flat = pred.reshape(-1)
        s_rep = np.repeat(test_idx.astype(np.int64), repeats=len(fold.candidate_expert_domains))
        q_rep = np.repeat(sample_domains[test_idx].astype(np.int64), repeats=len(fold.candidate_expert_domains))
        e_rep = np.tile(np.asarray(fold.candidate_expert_domains, dtype=np.int64), reps=int(test_idx.shape[0]))
        true_flat = true_eval.reshape(-1)
        row_protocol = _method_protocol(method)
        for k in range(pred_flat.shape[0]):
            pair_rows.append(
                {
                    **_protocol_row_fields(fold=fold, method_protocol=row_protocol, method=method),
                    "method": method,
                    "sample_index": int(s_rep[k]),
                    "query_domain": int(q_rep[k]),
                    "expert_domain": int(e_rep[k]),
                    "predicted_score": float(pred_flat[k]),
                    "true_nelbo": float(true_flat[k]),
                    "source_inner_selected_method": str(source_inner_selected_method),
                    "posthoc_diagnostic_only": int(method != BASELINE_METHOD),
                    "used_for_selection": 0,
                }
            )

    gate_fields: Dict[str, Any] = {}
    deployed_method = str(source_inner_selected_method)
    if _is_target_batch_agreement_mode(pairwise_cfg):
        deployed_method, gate_fields = _target_batch_agreement_policy(
            source_inner_selected_method=str(source_inner_selected_method),
            method_predictions=method_predictions,
            candidate_domains=fold.candidate_expert_domains,
            test_idx=test_idx,
            sample_metadata=sample_metadata,
            pairwise_cfg=pairwise_cfg,
        )
    selected_pred = method_predictions.get(str(deployed_method), method_predictions[BASELINE_METHOD])
    selected_metrics, selected_rows = _selection_metrics(
        method=primary_method,
        query_domains=sample_domains[test_idx],
        expert_domains=fold.candidate_expert_domains,
        score_matrix=selected_pred,
        true_nelbo_matrix=true_eval,
        fold=fold,
        global_true_nelbo_matrix=global_eval,
        global_expert_domains=expert_domains,
        tie_policy=tie_policy,
    )
    _ = selected_metrics
    baseline_rows = method_sample_rows[BASELINE_METHOD]
    baseline_by_sample = {int(r["sample_index"]): r for r in baseline_rows}
    source_inner_rows_by_sample = {
        int(r["sample_index"]): r
        for r in method_sample_rows.get(str(source_inner_selected_method), method_sample_rows[BASELINE_METHOD])
    }
    ae_scores = ae_zscore_matrix[np.asarray(test_idx, dtype=np.int64)][:, list(fold.candidate_col_indices)]
    decision_rows: List[Dict[str, Any]] = []
    for local_i, row in enumerate(selected_rows):
        sample_index = int(test_idx[int(row["sample_index"])])
        row["sample_index"] = sample_index
        row["source_inner_selected_method"] = str(source_inner_selected_method)
        row["fallback_to_baseline"] = int(str(deployed_method) == BASELINE_METHOD)
        row["fallback_used"] = int(str(deployed_method) == BASELINE_METHOD)
        row["deployed_method"] = str(deployed_method)
        row["selected_variant"] = str(source_inner_selected_method)
        row["selection_mode"] = str(selection_mode)
        row["used_for_selection"] = 1
        row["posthoc_diagnostic_only"] = 0
        sample_rows.append(row)
        base = baseline_by_sample.get(sample_index, {})
        source_inner_row = source_inner_rows_by_sample.get(sample_index, base)
        source_inner_gap = float(source_inner_row.get("oracle_gap_pct", float("nan")))
        baseline_gap = float(base.get("oracle_gap_pct", float("nan")))
        v2_delta_gap_vs_baseline = source_inner_gap - baseline_gap
        gate_applied = int(gate_fields.get("agreement_gate_applied", 0) or 0)
        gate_passed = int(gate_fields.get("agreement_gate_passed", 0) or 0)
        gate_blocked = int(
            _is_target_batch_agreement_mode(pairwise_cfg)
            and str(source_inner_selected_method) != BASELINE_METHOD
            and gate_applied == 1
            and gate_passed == 0
            and str(deployed_method) == BASELINE_METHOD
        )
        gate_allowed = int(
            _is_target_batch_agreement_mode(pairwise_cfg)
            and str(source_inner_selected_method) != BASELINE_METHOD
            and gate_applied == 1
            and gate_passed == 1
            and str(deployed_method) != BASELINE_METHOD
        )
        nonbaseline_bypass = int(
            _is_target_batch_agreement_mode(pairwise_cfg)
            and str(source_inner_selected_method) != BASELINE_METHOD
            and str(deployed_method) != BASELINE_METHOD
            and gate_applied != 1
        )
        ae_order = np.lexsort((np.arange(ae_scores.shape[1], dtype=np.int64), ae_scores[int(local_i), :]))
        ae_best = int(fold.candidate_expert_domains[int(ae_order[0])])
        ae_margin = (
            float(ae_scores[int(local_i), int(ae_order[1])] - ae_scores[int(local_i), int(ae_order[0])])
            if int(ae_scores.shape[1]) > 1
            else float("inf")
        )
        metadata_selected = _metadata_selected_expert(
            query_domain=int(sample_domains[sample_index]),
            candidate_domains=fold.candidate_expert_domains,
        )
        v31_counterfactual_changed = int(
            str(primary_method) == TARGET_BATCH_AGREEMENT_V31_PRIMARY_METHOD
            and str(source_inner_selected_method) == RAW_AE_WEIGHTED
            and str(deployed_method) == BASELINE_METHOD
        )
        v3_counterfactual_gap = float(source_inner_gap) if v31_counterfactual_changed else float(row["oracle_gap_pct"])
        v3_counterfactual_selected_expert = (
            int(source_inner_row.get("selected_expert", -1))
            if v31_counterfactual_changed
            else int(row["selected_expert"])
        )
        oracle_expert = int(row["oracle_expert"])
        decision_rows.append(
            {
                "seed": int(seed),
                "outer_heldout_center": int(fold.heldout_domain),
                "sample_index": int(sample_index),
                "primary_method": str(primary_method),
                "selection_mode": str(selection_mode),
                "source_inner_selected_method": str(source_inner_selected_method),
                "selected_method": str(deployed_method),
                "selected_variant": str(source_inner_selected_method),
                "deployed_method": str(deployed_method),
                "fallback_used": int(str(deployed_method) == BASELINE_METHOD),
                "selected_expert": int(row["selected_expert"]),
                "oracle_expert": int(oracle_expert),
                "selected_nelbo": float(row["selected_nelbo"]),
                "oracle_nelbo": float(row["oracle_nelbo"]),
                "oracle_gap_pct": float(row["oracle_gap_pct"]),
                "selected_oracle_gap_pct": float(row["oracle_gap_pct"]),
                "source_inner_selected_expert": int(source_inner_row.get("selected_expert", -1)),
                "source_inner_selected_oracle_gap_pct": float(source_inner_gap),
                "baseline_selected_expert": int(base.get("selected_expert", -1)),
                "baseline_oracle_gap_pct": float(base.get("oracle_gap_pct", float("nan"))),
                "baseline_top1_oracle_hit": int(int(base.get("selected_expert", -1)) == int(base.get("oracle_expert", -2))),
                "delta_gap_vs_baseline": float(base.get("oracle_gap_pct", float("nan"))) - float(row["oracle_gap_pct"]),
                "delta_gap_vs_baseline_pp": float(base.get("oracle_gap_pct", float("nan"))) - float(row["oracle_gap_pct"]),
                "v2_delta_gap_vs_baseline": float(v2_delta_gap_vs_baseline),
                "posthoc_target_nelbo_diagnostic": int(_is_target_batch_agreement_mode(pairwise_cfg)),
                "posthoc_diagnostics_used_for_selection": 0,
                "false_veto_gate_applied": int(gate_blocked and v2_delta_gap_vs_baseline < 0.0),
                "false_allow_gate_applied": int(gate_allowed and v2_delta_gap_vs_baseline > 0.0),
                "false_veto": int(gate_blocked and v2_delta_gap_vs_baseline < 0.0),
                "false_allow": int(gate_allowed and v2_delta_gap_vs_baseline > 0.0),
                "blocked_harmful_deployment": int(gate_blocked and v2_delta_gap_vs_baseline > 0.0),
                "allowed_beneficial_deployment": int(gate_allowed and v2_delta_gap_vs_baseline < 0.0),
                "harmful_nonbaseline_bypass": int(nonbaseline_bypass and v2_delta_gap_vs_baseline > 0.0),
                "blocked_v2_delta_gap_vs_baseline": float(v2_delta_gap_vs_baseline) if gate_blocked else float("nan"),
                "allowed_v2_delta_gap_vs_baseline": float(v2_delta_gap_vs_baseline) if gate_allowed else float("nan"),
                "nonbaseline_bypass_delta_gap_vs_baseline": float(v2_delta_gap_vs_baseline) if nonbaseline_bypass else float("nan"),
                "v3_counterfactual_deployed_method": str(source_inner_selected_method) if v31_counterfactual_changed else str(deployed_method),
                "v3_counterfactual_oracle_gap_pct": float(v3_counterfactual_gap),
                "delta_gap_v31_vs_v3": float(v3_counterfactual_gap) - float(row["oracle_gap_pct"]),
                "delta_top1_v31_vs_v3": float(int(int(row["selected_expert"]) == oracle_expert) - int(v3_counterfactual_selected_expert == oracle_expert)),
                "delta_spearman_v31_vs_v3": float("nan"),
                "v31_additional_blocks_over_v3": int(v31_counterfactual_changed),
                "v31_additional_false_vetoes_over_v3": int(v31_counterfactual_changed and v2_delta_gap_vs_baseline < 0.0),
                "v31_additional_harm_prevented_over_v3": int(v31_counterfactual_changed and v2_delta_gap_vs_baseline > 0.0),
                "ae_best_expert": int(ae_best),
                "ae_best_vs_second_margin": float(ae_margin),
                "metadata_selected_expert": int(metadata_selected),
                "top1_oracle_hit": int(row["selected_expert"] == row["oracle_expert"]),
                **gate_fields,
            }
        )
    return PairwiseAECombinedV2FoldOutputs(
        sample_rows=sample_rows,
        pair_rows=pair_rows,
        training_pair_rows=training_rows,
        feature_diagnostic_rows=feature_rows,
        inner_selection_rows=inner_rows,
        decision_rows=decision_rows,
    )


def _pairwise_auc_from_prediction_rows(rows: Sequence[Mapping[str, Any]], *, mode: str) -> float:
    grouped: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("method", "")), int(row.get("sample_index", -1)))
        grouped.setdefault(key, []).append(row)

    pair_records: List[Tuple[float, float]] = []
    sample_aucs: List[float] = []
    for (_method, _sample), sample_rows in grouped.items():
        if len(sample_rows) < 2:
            continue
        scores = np.asarray([float(r.get("predicted_score", float("nan"))) for r in sample_rows], dtype=np.float64)
        true = np.asarray([float(r.get("true_nelbo", float("nan"))) for r in sample_rows], dtype=np.float64)
        if mode == "all":
            sample_aucs.append(float(_pairwise_auc_single(scores, true)))
            continue
        for i in range(len(sample_rows)):
            for j in range(i + 1, len(sample_rows)):
                diff = abs(float(true[i]) - float(true[j]))
                if diff < 1.0e-12:
                    continue
                pred_diff = abs(float(scores[i]) - float(scores[j]))
                if pred_diff < 1.0e-12:
                    correct = 0.5
                else:
                    correct = float((true[i] < true[j]) == (scores[i] < scores[j]))
                pair_records.append((float(diff), float(correct)))
    if mode == "all":
        return _finite_mean(sample_aucs, float("nan"))
    if not pair_records:
        return float("nan")
    diffs = np.asarray([v[0] for v in pair_records], dtype=np.float64)
    if mode == "near_boundary":
        threshold = float(np.quantile(diffs, 0.25))
        selected = [float(score) for diff, score in pair_records if float(diff) <= threshold]
    elif mode == "high_utility_gap":
        threshold = float(np.quantile(diffs, 0.75))
        selected = [float(score) for diff, score in pair_records if float(diff) >= threshold]
    else:
        selected = []
    return _finite_mean(selected, float("nan"))


def _nested_counts(rows: Sequence[Mapping[str, Any]], *, key_field: str, value_field: str = "selected_method") -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for row in rows:
        key = str(row.get(key_field, ""))
        value = str(row.get(value_field, ""))
        if not key or not value:
            continue
        out.setdefault(key, {})
        out[key][value] = int(out[key].get(value, 0)) + 1
    return out


def _top1_when_low_ae_margin(rows: Sequence[Mapping[str, Any]]) -> float:
    vals = [
        (float(r.get("ae_best_vs_second_margin", float("nan"))), float(r.get("top1_oracle_hit", float("nan"))))
        for r in rows
        if np.isfinite(float(r.get("ae_best_vs_second_margin", float("nan"))))
    ]
    if not vals:
        return float("nan")
    threshold = float(np.quantile(np.asarray([v[0] for v in vals], dtype=np.float64), 0.25))
    return _finite_mean([hit for margin, hit in vals if float(margin) <= threshold], float("nan"))


def _gap_delta(row: Mapping[str, Any]) -> float:
    if "delta_gap_vs_baseline_pp" in row:
        return float(row.get("delta_gap_vs_baseline_pp", float("nan")))
    return float(row.get("delta_gap_vs_baseline", float("nan")))


def _seed_gap_improved_count(rows: Sequence[Mapping[str, Any]]) -> int:
    by_seed: Dict[str, List[float]] = {}
    for row in rows:
        by_seed.setdefault(str(row.get("seed", "")), []).append(_gap_delta(row))
    return int(sum(1 for vals in by_seed.values() if _finite_mean(vals, 0.0) > 0.0))


def _seed_top1_nondegrading_count(rows: Sequence[Mapping[str, Any]]) -> int:
    by_seed: Dict[str, List[float]] = {}
    for row in rows:
        selected_top1 = float(row.get("top1_oracle_hit", float("nan")))
        baseline_top1 = float(row.get("baseline_top1_oracle_hit", float("nan")))
        by_seed.setdefault(str(row.get("seed", "")), []).append(selected_top1 - baseline_top1)
    return int(sum(1 for vals in by_seed.values() if _finite_mean(vals, 0.0) >= -0.02))


def _worst_center_gap_degradation(rows: Sequence[Mapping[str, Any]]) -> float:
    by_unit: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        key = (str(row.get("seed", "")), str(row.get("outer_heldout_center", "")))
        by_unit.setdefault(key, []).append(_gap_delta(row))
    degradations = [-_finite_mean(vals, 0.0) for vals in by_unit.values()]
    return float(max(degradations)) if degradations else 0.0


def _policy_unit_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_unit: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        by_unit.setdefault((str(row.get("seed", "")), str(row.get("outer_heldout_center", ""))), []).append(row)
    out: List[Dict[str, Any]] = []
    policy_fields = [
        "seed",
        "outer_heldout_center",
        "primary_method",
        "selection_mode",
        "source_inner_selected_method",
        "selected_method",
        "selected_variant",
        "deployed_method",
        "fallback_used",
        "gate_scope",
        "gate_num_queries",
        "gate_num_groups",
        "gate_group_key",
        "selected_vs_raw_agreement_rate_query_weighted",
        "selected_vs_raw_agreement_rate_group_macro",
        "agreement_reference_methods",
        "agreement_reference_best_method",
        "selected_vs_reference_agreement_rate_query_weighted",
        "selected_vs_reference_agreement_rate_group_macro",
        "selected_vs_reference_best_agreement",
        "selected_vs_reference_mean_agreement",
        "selected_vs_reference_min_agreement",
        "raw_peer_agreement_with_rank_margin_unweighted",
        "raw_peer_agreement_with_rank_margin_weighted",
        "agreement_threshold",
        "agreement_threshold_source",
        "agreement_gate_applied",
        "agreement_gate_passed",
        "agreement_gate_skipped_due_to_small_batch",
        "agreement_gate_reason",
        "used_target_embeddings_for_gate",
        "used_target_group_ids_for_gate",
        "used_target_labels_for_gate",
        "used_target_nelbo_for_gate",
        "used_target_support_for_gate",
        "used_target_fitting_for_gate",
        "used_target_normalization_for_gate",
        "heldout_target_nelbo_used_for_selection",
        "v3_counterfactual_deployed_method",
        "v31_additional_blocks_over_v3",
        "v31_additional_false_vetoes_over_v3",
        "v31_additional_harm_prevented_over_v3",
    ]
    for _key, unit_rows in sorted(by_unit.items()):
        first = dict(unit_rows[0])
        row = {field: first.get(field, "") for field in policy_fields}
        row["n_queries"] = int(len(unit_rows))
        row["posthoc_target_nelbo_diagnostic"] = 1
        row["posthoc_diagnostics_used_for_selection"] = 0
        row["mean_blocked_v2_delta_gap_vs_baseline"] = _finite_mean(
            [float(r.get("blocked_v2_delta_gap_vs_baseline", float("nan"))) for r in unit_rows],
            float("nan"),
        )
        row["mean_allowed_v2_delta_gap_vs_baseline"] = _finite_mean(
            [float(r.get("allowed_v2_delta_gap_vs_baseline", float("nan"))) for r in unit_rows],
            float("nan"),
        )
        row["mean_nonbaseline_bypass_delta_gap_vs_baseline"] = _finite_mean(
            [float(r.get("nonbaseline_bypass_delta_gap_vs_baseline", float("nan"))) for r in unit_rows],
            float("nan"),
        )
        row["false_veto"] = int(
            int(first.get("agreement_gate_applied", 0)) == 1
            and int(first.get("agreement_gate_passed", 0)) == 0
            and np.isfinite(float(row["mean_blocked_v2_delta_gap_vs_baseline"]))
            and float(row["mean_blocked_v2_delta_gap_vs_baseline"]) < 0.0
        )
        row["false_allow"] = int(
            int(first.get("agreement_gate_applied", 0)) == 1
            and int(first.get("agreement_gate_passed", 0)) == 1
            and np.isfinite(float(row["mean_allowed_v2_delta_gap_vs_baseline"]))
            and float(row["mean_allowed_v2_delta_gap_vs_baseline"]) > 0.0
        )
        row["blocked_harmful_deployment"] = int(
            int(first.get("agreement_gate_applied", 0)) == 1
            and int(first.get("agreement_gate_passed", 0)) == 0
            and np.isfinite(float(row["mean_blocked_v2_delta_gap_vs_baseline"]))
            and float(row["mean_blocked_v2_delta_gap_vs_baseline"]) > 0.0
        )
        row["allowed_beneficial_deployment"] = int(
            int(first.get("agreement_gate_applied", 0)) == 1
            and int(first.get("agreement_gate_passed", 0)) == 1
            and np.isfinite(float(row["mean_allowed_v2_delta_gap_vs_baseline"]))
            and float(row["mean_allowed_v2_delta_gap_vs_baseline"]) < 0.0
        )
        row["false_veto_gate_applied"] = int(row["false_veto"])
        row["false_allow_gate_applied"] = int(row["false_allow"])
        row["harmful_nonbaseline_bypass"] = int(
            int(first.get("agreement_gate_applied", 0)) != 1
            and str(first.get("deployed_method", "")) != BASELINE_METHOD
            and np.isfinite(float(row["mean_nonbaseline_bypass_delta_gap_vs_baseline"]))
            and float(row["mean_nonbaseline_bypass_delta_gap_vs_baseline"]) > 0.0
        )
        out.append(row)
    return out


def _threshold_sensitivity_rows(policy_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
    out: List[Dict[str, Any]] = []
    for row in policy_rows:
        if int(row.get("agreement_gate_applied", 0) or 0) != 1:
            continue
        q = float(row.get("selected_vs_reference_agreement_rate_query_weighted", row.get("selected_vs_raw_agreement_rate_query_weighted", float("nan"))))
        g = float(row.get("selected_vs_reference_agreement_rate_group_macro", row.get("selected_vs_raw_agreement_rate_group_macro", float("nan"))))
        for threshold in thresholds:
            out.append(
                {
                    "seed": row.get("seed", ""),
                    "outer_heldout_center": row.get("outer_heldout_center", ""),
                    "candidate_threshold": float(threshold),
                    "would_pass_threshold": int(np.isfinite(q) and np.isfinite(g) and q >= threshold and g >= threshold),
                    "posthoc_diagnostic_only": 1,
                    "used_for_selection": 0,
                }
            )
    return out


def write_pairwise_ae_combined_v2_artifacts(
    *,
    reports_dir: Path,
    training_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    inner_selection_rows: Sequence[Mapping[str, Any]],
    pair_prediction_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not (training_rows or feature_rows or inner_selection_rows or pair_prediction_rows or decision_rows):
        return {}
    has_strict = any(str(row.get("primary_method", "")) == STRICT_PRIMARY_METHOD for row in decision_rows)
    has_v3 = any(str(row.get("primary_method", "")) == TARGET_BATCH_AGREEMENT_PRIMARY_METHOD for row in decision_rows)
    has_v31 = any(str(row.get("primary_method", "")) == TARGET_BATCH_AGREEMENT_V31_PRIMARY_METHOD for row in decision_rows)
    has_target_batch = bool(has_v3 or has_v31)
    _write_csv(reports_dir / "pairwise_ae_combined_v2_training_pairs.csv", training_rows)
    _write_csv(reports_dir / "pairwise_ae_combined_v2_feature_diagnostics.csv", feature_rows)
    _write_csv(reports_dir / "pairwise_ae_combined_v2_inner_selection_table.csv", inner_selection_rows)
    _write_csv(reports_dir / "pairwise_ae_combined_v2_pair_predictions.csv", pair_prediction_rows)
    _write_csv(reports_dir / "pairwise_ae_combined_v2_decision_table.csv", decision_rows)
    if has_strict:
        _write_csv(reports_dir / "pairwise_ae_combined_v2_strict_inner_selection_table.csv", inner_selection_rows)
        _write_csv(reports_dir / "pairwise_ae_combined_v2_strict_decision_table.csv", decision_rows)
    if has_target_batch:
        v3_policy_rows = _policy_unit_rows(decision_rows)
        target_prefix = "pairwise_ae_combined_v31" if has_v31 else "pairwise_ae_combined_v3"
        _write_csv(reports_dir / f"{target_prefix}_inner_selection_table.csv", inner_selection_rows)
        _write_csv(reports_dir / f"{target_prefix}_agreement_policy.csv", v3_policy_rows)
        _write_csv(reports_dir / f"{target_prefix}_threshold_sensitivity.csv", _threshold_sensitivity_rows(v3_policy_rows))
        _write_csv(reports_dir / f"{target_prefix}_decision_table.csv", decision_rows)
    else:
        v3_policy_rows = []

    selected_methods = [str(r.get("selected_method", "")) for r in decision_rows]
    n = max(len(selected_methods), 1)
    fallback_count = sum(1 for method in selected_methods if method == BASELINE_METHOD)
    selected_counts = {method: selected_methods.count(method) for method in sorted(set(selected_methods)) if method}
    gap_deltas = [_gap_delta(r) for r in decision_rows]
    summary = {
        "method": TARGET_BATCH_AGREEMENT_V31_PRIMARY_METHOD if has_v31 else TARGET_BATCH_AGREEMENT_PRIMARY_METHOD if has_v3 else STRICT_PRIMARY_METHOD if has_strict else PRIMARY_METHOD,
        "selected_method_count_total": selected_counts,
        "selected_method_count_by_seed": _nested_counts(decision_rows, key_field="seed"),
        "selected_method_count_by_outer_center": _nested_counts(decision_rows, key_field="outer_heldout_center"),
        "fallback_to_baseline_rate": float(fallback_count / n),
        "v2_adoption_rate": float((n - fallback_count) / n),
        "mean_delta_gap_vs_baseline": _finite_mean(gap_deltas, 0.0),
        "pairwise_auc_all": _pairwise_auc_from_prediction_rows(pair_prediction_rows, mode="all"),
        "pairwise_auc_near_boundary": _pairwise_auc_from_prediction_rows(pair_prediction_rows, mode="near_boundary"),
        "pairwise_auc_high_utility_gap": _pairwise_auc_from_prediction_rows(pair_prediction_rows, mode="high_utility_gap"),
        "top1_when_margin_small": _top1_when_low_ae_margin(decision_rows),
        "top1_when_ae_confidence_low": _top1_when_low_ae_margin(decision_rows),
        "feature_nonzero_rate_after_pairwise_difference": _finite_mean(
            [
                float(r.get("feature_nonzero_rate_after_pairwise_difference", float("nan")))
                for r in feature_rows
                if str(r.get("feature_name", "")) == "__all__"
            ],
            float("nan"),
        ),
        "heldout_target_nelbo_used_for_selection": 0,
    }
    if has_strict:
        summary.update(
            {
                "strict_v2_adoption_rate": float((n - fallback_count) / n),
                "fallback_to_pairwise_ae_combined_rate": float(fallback_count / n),
                "mean_gap_reduction_vs_pairwise_ae_combined": _finite_mean(gap_deltas, 0.0),
                "worst_center_gap_degradation": _worst_center_gap_degradation(decision_rows),
                "seed_gap_improved_count": _seed_gap_improved_count(decision_rows),
                "seed_top1_nondegrading_count": _seed_top1_nondegrading_count(decision_rows),
                "always_baseline_fallback": bool(fallback_count == len(selected_methods)),
            }
        )
    if has_target_batch:
        gate_activation = sum(1 for row in v3_policy_rows if int(row.get("agreement_gate_applied", 0) or 0) == 1)
        gate_pass = sum(1 for row in v3_policy_rows if int(row.get("agreement_gate_applied", 0) or 0) == 1 and int(row.get("agreement_gate_passed", 0) or 0) == 1)
        gate_block = sum(1 for row in v3_policy_rows if int(row.get("agreement_gate_applied", 0) or 0) == 1 and int(row.get("agreement_gate_passed", 0) or 0) == 0)
        deployed_nonbaseline_rows = [row for row in decision_rows if str(row.get("deployed_method", row.get("selected_method", ""))) != BASELINE_METHOD]
        gated_deployed_nonbaseline_rows = [
            row for row in deployed_nonbaseline_rows if int(row.get("agreement_gate_applied", 0) or 0) == 1
        ]
        ungated_deployed_nonbaseline_rows = [
            row for row in deployed_nonbaseline_rows if int(row.get("agreement_gate_applied", 0) or 0) != 1
        ]
        gate_pass_by_method: Dict[str, int] = {}
        gate_block_by_method: Dict[str, int] = {}
        for row in v3_policy_rows:
            method = str(row.get("source_inner_selected_method", ""))
            if int(row.get("agreement_gate_applied", 0) or 0) != 1 or not method:
                continue
            if int(row.get("agreement_gate_passed", 0) or 0) == 1:
                gate_pass_by_method[method] = int(gate_pass_by_method.get(method, 0) + 1)
            else:
                gate_block_by_method[method] = int(gate_block_by_method.get(method, 0) + 1)
        summary.update(
            {
                "gate_activation_count": int(gate_activation),
                "gate_pass_count": int(gate_pass),
                "gate_block_count": int(gate_block),
                "false_veto_count": int(sum(int(row.get("false_veto", 0) or 0) for row in v3_policy_rows)),
                "false_allow_count": int(sum(int(row.get("false_allow", 0) or 0) for row in v3_policy_rows)),
                "false_veto_gate_applied_count": int(sum(int(row.get("false_veto_gate_applied", 0) or 0) for row in v3_policy_rows)),
                "false_allow_gate_applied_count": int(sum(int(row.get("false_allow_gate_applied", 0) or 0) for row in v3_policy_rows)),
                "blocked_harmful_count": int(sum(int(row.get("blocked_harmful_deployment", 0) or 0) for row in v3_policy_rows)),
                "allowed_beneficial_count": int(sum(int(row.get("allowed_beneficial_deployment", 0) or 0) for row in v3_policy_rows)),
                "nonbaseline_deployment_count": int(len(deployed_nonbaseline_rows)),
                "gated_nonbaseline_count": int(len(gated_deployed_nonbaseline_rows)),
                "ungated_nonbaseline_count": int(len(ungated_deployed_nonbaseline_rows)),
                "gate_pass_count_by_selected_method": gate_pass_by_method,
                "gate_block_count_by_selected_method": gate_block_by_method,
                "harmful_nonbaseline_bypass": int(sum(int(row.get("harmful_nonbaseline_bypass", 0) or 0) for row in v3_policy_rows)),
                "mean_blocked_v2_delta_gap_vs_baseline": _finite_mean(
                    [float(row.get("blocked_v2_delta_gap_vs_baseline", float("nan"))) for row in decision_rows],
                    float("nan"),
                ),
                "mean_allowed_v2_delta_gap_vs_baseline": _finite_mean(
                    [float(row.get("allowed_v2_delta_gap_vs_baseline", float("nan"))) for row in decision_rows],
                    float("nan"),
                ),
                "target_batch_agreement_threshold": float(
                    next(
                        (
                            float(row.get("agreement_threshold"))
                            for row in decision_rows
                            if np.isfinite(float(row.get("agreement_threshold", float("nan"))))
                        ),
                        0.60,
                    )
                ),
                "agreement_threshold_source": str(
                    next(
                        (
                            str(row.get("agreement_threshold_source"))
                            for row in decision_rows
                            if str(row.get("agreement_threshold_source", "")).strip()
                        ),
                        "predeclared_development_seed_diagnostic",
                    )
                ),
                "used_target_embeddings_for_gate": 1,
                "used_target_group_ids_for_gate": int(any(int(row.get("used_target_group_ids_for_gate", 0) or 0) == 1 for row in v3_policy_rows)),
                "used_target_labels_for_gate": 0,
                "used_target_nelbo_for_gate": 0,
                "used_target_support_for_gate": 0,
                "used_target_fitting_for_gate": 0,
                "used_target_normalization_for_gate": 0,
                "heldout_target_nelbo_used_for_selection": 0,
                "delta_gap_v31_vs_v3": _finite_mean(
                    [float(row.get("delta_gap_v31_vs_v3", float("nan"))) for row in decision_rows],
                    float("nan"),
                ),
                "delta_top1_v31_vs_v3": _finite_mean(
                    [float(row.get("delta_top1_v31_vs_v3", float("nan"))) for row in decision_rows],
                    float("nan"),
                ),
                "delta_spearman_v31_vs_v3": _finite_mean(
                    [float(row.get("delta_spearman_v31_vs_v3", float("nan"))) for row in decision_rows],
                    float("nan"),
                ),
                "v31_additional_blocks_over_v3": int(sum(int(row.get("v31_additional_blocks_over_v3", 0) or 0) for row in v3_policy_rows)),
                "v31_additional_false_vetoes_over_v3": int(sum(int(row.get("v31_additional_false_vetoes_over_v3", 0) or 0) for row in v3_policy_rows)),
                "v31_additional_harm_prevented_over_v3": int(sum(int(row.get("v31_additional_harm_prevented_over_v3", 0) or 0) for row in v3_policy_rows)),
            }
        )
    _write_csv(reports_dir / "pairwise_ae_combined_v2_decision_summary.csv", [summary])
    (reports_dir / "pairwise_ae_combined_v2_decision_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    if has_strict:
        _write_csv(reports_dir / "pairwise_ae_combined_v2_strict_decision_summary.csv", [summary])
        (reports_dir / "pairwise_ae_combined_v2_strict_decision_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
            encoding="utf-8",
        )
    if has_target_batch:
        target_prefix = "pairwise_ae_combined_v31" if has_v31 else "pairwise_ae_combined_v3"
        _write_csv(reports_dir / f"{target_prefix}_decision_summary.csv", [summary])
        (reports_dir / f"{target_prefix}_decision_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
            encoding="utf-8",
        )
    artifacts = {
        "pairwise_ae_combined_v2_training_pairs": "pairwise_ae_combined_v2_training_pairs.csv",
        "pairwise_ae_combined_v2_feature_diagnostics": "pairwise_ae_combined_v2_feature_diagnostics.csv",
        "pairwise_ae_combined_v2_inner_selection_table": "pairwise_ae_combined_v2_inner_selection_table.csv",
        "pairwise_ae_combined_v2_pair_predictions": "pairwise_ae_combined_v2_pair_predictions.csv",
        "pairwise_ae_combined_v2_decision_table": "pairwise_ae_combined_v2_decision_table.csv",
        "pairwise_ae_combined_v2_decision_summary": "pairwise_ae_combined_v2_decision_summary.json",
    }
    if has_strict:
        artifacts.update(
            {
                "pairwise_ae_combined_v2_strict_inner_selection_table": "pairwise_ae_combined_v2_strict_inner_selection_table.csv",
                "pairwise_ae_combined_v2_strict_decision_table": "pairwise_ae_combined_v2_strict_decision_table.csv",
                "pairwise_ae_combined_v2_strict_decision_summary": "pairwise_ae_combined_v2_strict_decision_summary.json",
            }
        )
    if has_target_batch:
        target_prefix = "pairwise_ae_combined_v31" if has_v31 else "pairwise_ae_combined_v3"
        artifacts.update(
            {
                f"{target_prefix}_inner_selection_table": f"{target_prefix}_inner_selection_table.csv",
                f"{target_prefix}_agreement_policy": f"{target_prefix}_agreement_policy.csv",
                f"{target_prefix}_threshold_sensitivity": f"{target_prefix}_threshold_sensitivity.csv",
                f"{target_prefix}_decision_table": f"{target_prefix}_decision_table.csv",
                f"{target_prefix}_decision_summary": f"{target_prefix}_decision_summary.json",
            }
        )
    return artifacts
