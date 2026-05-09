from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from src.app.determinism import stable_response_seed
from src.eval.evaluators.learned_utility_protocol import (
    FoldCandidateSet,
    ProtocolError,
    _method_protocol,
    _protocol_row_fields,
    _aggregate_metrics_from_sample_rows,
    _domain_breakdown_rows,
)
from src.eval.evaluators.learned_utility_scoring import _load_model, _parse_expert_domain
from src.eval.evaluators.response_indirect import compute_response_features
from src.eval.evaluators.support_set_calibration import make_support_eval_split
from src.eval.metrics import spearman_corr
from src.routing.strategies import compute_similarity


SUPPORT_RESPONSE_PROTOCOL_VERSION = "support_response_candidate_specific_v1"
SUPPORT_RESPONSE_AGGREGATION_SOURCE = "support_response_sample_selections.csv"
PRIVACY_PROVENANCE_FIELDS: Dict[str, object] = {
    "target_support_data_location": "target_local",
    "raw_target_images_exported": False,
    "target_embeddings_exported": False,
    "exported_support_signal": "aggregate_response_features_only",
    "source_expert_artifacts_shared": True,
}
STATIC_SUPPORT_FEATURES = [
    "metadata_distance",
    "abs_domain_diff",
    "is_exact_domain_match",
    "embedding_distance",
]
DIRECT_UTILITY_BLOCK_TERMS = [
    "nelbo",
    "elbo",
    "oracle",
    "rank",
    "target",
    "eval",
    "candidate",
    "query_id",
    "expert_id",
    "domain_id",
    "response_recon_mean",
    "response_recon_loss",
    "response_reconstruction",
    "response_kl_mean",
    "label",
    "selected",
]


@dataclass(frozen=True)
class SupportResponseConfig:
    enabled: bool
    support_sizes: Tuple[int, ...]
    support_seeds: Tuple[int, ...]
    sampling_policies: Tuple[str, ...]
    feature_regimes: Tuple[str, ...]
    primary_feature_regime: str
    ranker: str
    ridge_l2: float
    num_response_repeats: int
    tie_policy: str
    domain_level_aggregation: bool
    source_leave_pseudo_domain_out_diagnostic: bool
    include_residual_shape_features: bool = False


@dataclass(frozen=True)
class FeatureAuditResult:
    matrix: np.ndarray
    feature_names: List[str]
    blocked_features: List[str]
    blocked_feature_terms: List[str]
    dropped_zero_variance: List[str]
    missing_features: List[str]
    no_data_reason: str


@dataclass(frozen=True)
class FeatureScaler:
    feature_names: Tuple[str, ...]
    mu: np.ndarray
    sigma: np.ndarray

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[1] != len(self.feature_names):
            raise ProtocolError(
                f"Feature matrix width {matrix.shape[1]} does not match scaler width {len(self.feature_names)}"
            )
        return (matrix - self.mu) / self.sigma


class LinearPairwiseRidge:
    """Low-capacity pairwise linear ranker.

    The emitted score is always NELBO-direction: lower score means more compatible.
    """

    def __init__(self, *, ridge_l2: float) -> None:
        self.ridge_l2 = float(ridge_l2)
        self.w: np.ndarray | None = None

    def fit(self, x: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> None:
        if x.ndim != 2:
            raise ValueError("x must be 2D")
        if not pairs:
            self.w = np.zeros((int(x.shape[1]),), dtype=np.float64)
            return
        diffs: List[np.ndarray] = []
        targets: List[float] = []
        for better_idx, worse_idx in pairs:
            diff = x[int(better_idx)] - x[int(worse_idx)]
            # Better candidate must receive a lower predicted NELBO score.
            diffs.append(diff)
            targets.append(-1.0)
            diffs.append(-diff)
            targets.append(1.0)
        design = np.asarray(diffs, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        xtx = design.T @ design
        xtx += float(self.ridge_l2) * np.eye(xtx.shape[0], dtype=np.float64)
        self.w = np.linalg.solve(xtx, design.T @ y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("LinearPairwiseRidge is not fitted")
        if x.shape[1] != self.w.shape[0]:
            raise ProtocolError(f"Feature width mismatch: {x.shape[1]} vs {self.w.shape[0]}")
        return x @ self.w


class _ResponseExpertBank:
    def __init__(
        self,
        *,
        expert_checkpoints: Mapping[str, str],
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        device: torch.device,
        metadata_constraint_cfg: Mapping[str, object] | None = None,
    ) -> None:
        if bool((metadata_constraint_cfg or {}).get("enabled", False)):
            raise ProtocolError("support_response_routing does not support metadata-constraint checkpoints in v1")
        self.device = device
        self.cvaes: Dict[int, Any] = {}
        for name in sorted(expert_checkpoints.keys()):
            domain = _parse_expert_domain(str(name))
            self.cvaes[int(domain)] = _load_model(
                Path(str(expert_checkpoints[name])),
                input_dim=int(input_dim),
                hidden_dim=int(hidden_dim),
                latent_dim=int(latent_dim),
                device=device,
                metadata_dim=0,
                metadata_constraint_cfg=dict(metadata_constraint_cfg or {}),
            )

    def domain_cvae(self, domain: int):
        return self.cvaes[int(domain)]

    def project(self, domain: int, x: torch.Tensor) -> torch.Tensor:
        _ = domain
        return x


def parse_support_response_config(learned_cfg: Mapping[str, Any]) -> SupportResponseConfig:
    raw = learned_cfg.get("support_response_routing", {}) if isinstance(learned_cfg, Mapping) else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("learned_utility.support_response_routing must be a dictionary")
    enabled = bool(raw.get("enabled", False))
    return SupportResponseConfig(
        enabled=enabled,
        support_sizes=tuple(int(v) for v in raw.get("support_sizes", [8, 16, 32])),
        support_seeds=tuple(int(v) for v in raw.get("support_seeds", [17, 23])),
        sampling_policies=tuple(str(v).strip().lower() for v in raw.get("sampling_policies", ["random"])),
        feature_regimes=tuple(
            str(v).strip().lower()
            for v in raw.get("feature_regimes", ["static_response_indirect", "response_indirect_shuffled"])
        ),
        primary_feature_regime=str(raw.get("primary_feature_regime", "static_response_indirect")).strip().lower(),
        ranker=str(raw.get("ranker", "linear_pairwise_ridge")).strip().lower(),
        ridge_l2=float(raw.get("ridge_l2", 1.0e-3)),
        num_response_repeats=int(raw.get("num_response_repeats", 8)),
        tie_policy=str(raw.get("tie_policy", "stable_expert_index")).strip().lower(),
        domain_level_aggregation=bool(raw.get("domain_level_aggregation", True)),
        source_leave_pseudo_domain_out_diagnostic=bool(
            raw.get("source_leave_pseudo_domain_out_diagnostic", True)
        ),
        include_residual_shape_features=bool(raw.get("include_residual_shape_features", False)),
    )


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


def _as_domain(value: object) -> int:
    return int(str(value).replace("x", ""))


def _as_label(meta: Mapping[str, object]) -> int:
    return int(meta.get("label", 0))


def _json_mapping(keys: Sequence[int], values: Sequence[float]) -> str:
    return json.dumps(
        {str(int(k)): float(v) for k, v in zip(keys, values)},
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_argmin(values: Sequence[float], experts: Sequence[int]) -> int:
    order = sorted(range(len(values)), key=lambda i: (float(values[i]), int(experts[i])))
    return int(order[0])


def _selected_rank(selected_idx: int, true_nelbo: Sequence[float], experts: Sequence[int]) -> float:
    order = sorted(range(len(true_nelbo)), key=lambda i: (float(true_nelbo[i]), int(experts[i])))
    ranks = {int(idx): rank for rank, idx in enumerate(order, start=1)}
    return float(ranks[int(selected_idx)])


def _pairwise_auc(score_row: Sequence[float], true_nelbo_row: Sequence[float]) -> float:
    total = 0.0
    correct = 0.0
    for i in range(len(score_row)):
        for j in range(i + 1, len(score_row)):
            ti = float(true_nelbo_row[i])
            tj = float(true_nelbo_row[j])
            if abs(ti - tj) < 1e-12:
                continue
            si = float(score_row[i])
            sj = float(score_row[j])
            total += 1.0
            if abs(si - sj) < 1e-12:
                correct += 0.5
            elif (si < sj) == (ti < tj):
                correct += 1.0
    return float(correct / total) if total > 0.0 else float("nan")


def _domain_centroids(embeddings: np.ndarray, sample_domains: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for domain in sorted(set(int(v) for v in sample_domains.tolist())):
        idxs = np.where(sample_domains == int(domain))[0]
        if idxs.size:
            out[int(domain)] = embeddings[idxs].mean(axis=0)
    return out


def _privacy_fields(data_cfg: Mapping[str, Any] | None) -> Dict[str, Any]:
    data_cfg = data_cfg or {}
    return {
        "dataset_domain_semantics": str(data_cfg.get("dataset_domain_semantics", "unknown")),
        "storage_field": str(data_cfg.get("legacy_domain_field_alias", "magnification")),
        **PRIVACY_PROVENANCE_FIELDS,
    }


def _block_terms(feature_name: str, *, allow_candidate_identity: bool) -> List[str]:
    name = str(feature_name)
    terms: List[str] = []
    for term in DIRECT_UTILITY_BLOCK_TERMS:
        if allow_candidate_identity and term in {"expert_id", "candidate"}:
            continue
        if term in name:
            terms.append(term)
    return terms


def _candidate_feature_names(rows: Sequence[Mapping[str, Any]], regime: str) -> List[str]:
    names: List[str] = []
    regime = str(regime).strip().lower()
    if regime == "expert_id_only":
        expert_domains = sorted({int(r["candidate_expert"]) for r in rows if "candidate_expert" in r})
        return [f"expert_onehot_{int(e)}" for e in expert_domains]
    if regime == "static_response_indirect":
        names.extend(STATIC_SUPPORT_FEATURES)
    elif regime not in {"response_indirect", "response_indirect_shuffled"}:
        raise ValueError(f"Unknown support-response feature regime: {regime}")
    response_names = sorted({str(k) for row in rows for k in row.keys() if str(k).startswith("response_")})
    names.extend(response_names)
    return names


def audit_support_response_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    regime: str,
    feature_names: Sequence[str] | None = None,
    allow_candidate_identity: bool = False,
    drop_zero_variance: bool = True,
    zero_variance_eps: float = 1e-12,
) -> FeatureAuditResult:
    rows = [dict(r) for r in rows]
    candidates = list(feature_names) if feature_names is not None else _candidate_feature_names(rows, regime)
    blocked: List[str] = []
    terms: List[str] = []
    allowed: List[str] = []
    for feature in candidates:
        feature_terms = _block_terms(str(feature), allow_candidate_identity=allow_candidate_identity)
        if feature_terms:
            blocked.append(str(feature))
            terms.extend(feature_terms)
            continue
        allowed.append(str(feature))

    cols: List[List[float]] = []
    names: List[str] = []
    missing: List[str] = []
    for feature in allowed:
        vals: List[float] = []
        present = False
        if feature.startswith("expert_onehot_"):
            expert = int(feature.replace("expert_onehot_", ""))
            for row in rows:
                vals.append(1.0 if int(row.get("candidate_expert", -1)) == expert else 0.0)
                present = True
        else:
            for row in rows:
                if feature in row:
                    present = True
                try:
                    val = float(row.get(feature, 0.0))
                except Exception:
                    val = 0.0
                vals.append(val if np.isfinite(val) else 0.0)
        if not present:
            missing.append(feature)
            continue
        cols.append(vals)
        names.append(feature)

    matrix = np.asarray(cols, dtype=np.float64).T if cols else np.empty((len(rows), 0), dtype=np.float64)
    dropped: List[str] = []
    if drop_zero_variance and matrix.shape[1] > 0:
        keep: List[int] = []
        for idx, feature in enumerate(names):
            if float(np.var(matrix[:, idx])) <= float(zero_variance_eps):
                dropped.append(feature)
            else:
                keep.append(idx)
        matrix = matrix[:, keep] if keep else np.empty((len(rows), 0), dtype=np.float64)
        names = [names[i] for i in keep]

    return FeatureAuditResult(
        matrix=matrix,
        feature_names=names,
        blocked_features=blocked,
        blocked_feature_terms=sorted(set(terms)),
        dropped_zero_variance=dropped,
        missing_features=missing,
        no_data_reason="" if matrix.shape[1] > 0 else "no_features_after_audit",
    )


def fit_support_response_scaler(audit: FeatureAuditResult) -> FeatureScaler:
    if audit.matrix.shape[1] != len(audit.feature_names):
        raise ProtocolError("Feature audit matrix/name width mismatch")
    mu = audit.matrix.mean(axis=0, keepdims=True) if audit.matrix.size else np.zeros((1, 0), dtype=np.float64)
    sigma = audit.matrix.std(axis=0, keepdims=True) if audit.matrix.size else np.ones((1, 0), dtype=np.float64)
    sigma[sigma < 1e-8] = 1.0
    return FeatureScaler(feature_names=tuple(audit.feature_names), mu=mu, sigma=sigma)


def build_candidate_specific_pairs(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    by_query: Dict[int, List[int]] = {}
    for idx, row in enumerate(candidate_rows):
        by_query.setdefault(int(row["pseudo_query_domain"]), []).append(int(idx))
    pairs: List[Tuple[int, int]] = []
    pair_rows: List[Dict[str, Any]] = []
    for pseudo_query, idxs in sorted(by_query.items()):
        for pos_i in range(len(idxs)):
            for pos_j in range(pos_i + 1, len(idxs)):
                i = int(idxs[pos_i])
                j = int(idxs[pos_j])
                yi = float(candidate_rows[i]["label_nelbo"])
                yj = float(candidate_rows[j]["label_nelbo"])
                if abs(yi - yj) < 1e-12:
                    continue
                better, worse = (i, j) if yi < yj else (j, i)
                pairs.append((better, worse))
                pair_rows.append(
                    {
                        "pseudo_query_domain": int(pseudo_query),
                        "better_candidate_expert": int(candidate_rows[better]["candidate_expert"]),
                        "worse_candidate_expert": int(candidate_rows[worse]["candidate_expert"]),
                        "better_label_nelbo": float(candidate_rows[better]["label_nelbo"]),
                        "worse_label_nelbo": float(candidate_rows[worse]["label_nelbo"]),
                        "comparison_scope": "within_pseudo_query_domain",
                    }
                )
    return pairs, pair_rows


def _shuffle_response_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    seed: int,
    fold_id: str,
    split_id: str,
) -> List[Dict[str, Any]]:
    out = [dict(r) for r in rows]
    response_names = sorted({str(k) for row in rows for k in row.keys() if str(k).startswith("response_")})
    if len(out) <= 1 or not response_names:
        return out
    shuffle_seed = stable_response_seed(
        dataset=str(dataset),
        seed=int(seed),
        query_id=str(fold_id),
        expert_domain="response_indirect_shuffled",
        repeat_id=0,
        stream_name=f"support_response_shuffle:{split_id}",
    )
    rng = np.random.default_rng(int(shuffle_seed))
    perm = rng.permutation(len(out))
    for dest_idx, src_idx in enumerate(perm.tolist()):
        src = rows[int(src_idx)]
        for name in response_names:
            out[dest_idx][name] = src.get(name, 0.0)
    return out


def _static_features(
    *,
    query_domain: int,
    candidate_expert: int,
    support_indices: Sequence[int],
    embeddings: np.ndarray,
    centroids: Mapping[int, np.ndarray],
    expert_domains: Sequence[int],
    strategy: str,
    tau: float,
) -> Dict[str, float]:
    similarity = compute_similarity(
        {"magnification": int(query_domain)},
        {"magnification": int(candidate_expert)},
        strategy=strategy,
        tau=float(tau),
        similarity_matrix=None,
    )
    span = max(float(max(expert_domains) - min(expert_domains)), 1.0)
    support_centroid = embeddings[np.asarray(support_indices, dtype=np.int64)].mean(axis=0)
    candidate_centroid = centroids.get(int(candidate_expert))
    embedding_distance = (
        float(np.linalg.norm(support_centroid - candidate_centroid, ord=2))
        if candidate_centroid is not None
        else float("inf")
    )
    return {
        "metadata_distance": float(1.0 - float(similarity)),
        "abs_domain_diff": float(abs(int(query_domain) - int(candidate_expert)) / span),
        "is_exact_domain_match": 1.0 if int(query_domain) == int(candidate_expert) else 0.0,
        "embedding_distance": float(embedding_distance),
    }


def _score_method_row(
    *,
    method: str,
    fold: FoldCandidateSet,
    target_domain: int,
    support_seed: int,
    support_size: int,
    sampling_policy: str,
    support_eval_split_id: str,
    candidate_experts: Sequence[int],
    predicted_scores: Sequence[float],
    eval_mean_nelbo: Sequence[float],
    support_mean_nelbo: Sequence[float],
    sample_index: int,
    run_seed: int,
    privacy_fields: Mapping[str, Any],
) -> Dict[str, Any]:
    if str(method) == "support_candidate_oracle":
        method_protocol = _method_protocol("support_candidate_oracle")
    else:
        method_protocol = _method_protocol(method)
    selected_idx = _stable_argmin(predicted_scores, candidate_experts)
    oracle_idx = _stable_argmin(eval_mean_nelbo, candidate_experts)
    selected_expert = int(candidate_experts[selected_idx])
    oracle_expert = int(candidate_experts[oracle_idx])
    selected_nelbo = float(eval_mean_nelbo[selected_idx])
    oracle_nelbo = float(eval_mean_nelbo[oracle_idx])
    gap = float(selected_nelbo - oracle_nelbo)
    gap_pct = float((gap / max(abs(oracle_nelbo), 1e-12)) * 100.0)
    rank_score = np.asarray(predicted_scores, dtype=np.float64)
    true_nelbo = np.asarray(eval_mean_nelbo, dtype=np.float64)
    base = _protocol_row_fields(fold=fold, method_protocol=method_protocol, method=method)
    base["protocol_version"] = SUPPORT_RESPONSE_PROTOCOL_VERSION
    base["aggregation_source"] = SUPPORT_RESPONSE_AGGREGATION_SOURCE
    return {
        **base,
        **dict(privacy_fields),
        "sample_index": int(sample_index),
        "seed": int(run_seed),
        "query_domain": int(target_domain),
        "target_domain": int(target_domain),
        "support_seed": int(support_seed),
        "support_size_requested": int(support_size),
        "sampling_policy": str(sampling_policy),
        "support_eval_split_id": str(support_eval_split_id),
        "score_direction": "lower_predicted_score_is_higher_compatibility",
        "selected_expert": selected_expert,
        "candidate_oracle_expert": oracle_expert,
        "oracle_expert": oracle_expert,
        "selected_nelbo": selected_nelbo,
        "oracle_nelbo": oracle_nelbo,
        "candidate_oracle_nelbo": oracle_nelbo,
        "oracle_gap": gap,
        "oracle_gap_pct": gap_pct,
        "mean_oracle_gap_pct": gap_pct,
        "top1_oracle_hit": int(selected_expert == oracle_expert),
        "selected_rank": _selected_rank(selected_idx, true_nelbo, candidate_experts),
        "spearman": float(spearman_corr((-rank_score).tolist(), (-true_nelbo).tolist())),
        "pairwise_auc": _pairwise_auc(rank_score, true_nelbo),
        "predicted_score_by_expert_json": _json_mapping(candidate_experts, predicted_scores),
        "eval_nelbo_by_expert_json": _json_mapping(candidate_experts, eval_mean_nelbo),
        "support_nelbo_by_expert_json": _json_mapping(candidate_experts, support_mean_nelbo),
    }


def _support_split_manifest_row(
    *,
    run_seed: int,
    outer_target_domain: int,
    query_domain: int,
    split: Any,
    split_role: str,
    privacy_fields: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        **dict(privacy_fields),
        "seed": int(run_seed),
        "outer_target_domain": int(outer_target_domain),
        "query_domain": int(query_domain),
        "split_role": str(split_role),
        "support_seed": int(split.support_eval_split_id.split("_seed", 1)[1].split("_", 1)[0]),
        "support_size_requested": int(split.support_size_requested),
        "support_size_actual": int(split.support_size_actual),
        "eval_size": int(split.eval_size),
        "sampling_policy": str(split.sampling_policy_requested),
        "sampling_policy_effective": str(split.sampling_policy_effective),
        "split_status": str(split.split_status),
        "support_eval_disjoint": int(set(split.support_indices).isdisjoint(set(split.eval_indices))),
        "support_labels_used": int(split.support_labels_used),
        "support_eval_split_id": str(split.support_eval_split_id),
    }


def _candidate_rows_for_query(
    *,
    outer_target_domain: int,
    query_domain: int,
    candidate_experts: Sequence[int],
    split: Any,
    embeddings: np.ndarray,
    centroids: Mapping[int, np.ndarray],
    nelbo_matrix: np.ndarray,
    expert_domains: Sequence[int],
    expert_to_col: Mapping[int, int],
    strategy: str,
    tau: float,
    response_feature_fn: Callable[[Sequence[int], int, str], Mapping[str, float]],
    split_role: str,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    support_idxs = np.asarray(split.support_indices, dtype=np.int64)
    eval_idxs = np.asarray(split.eval_indices, dtype=np.int64)
    support_mean: List[float] = []
    eval_mean: List[float] = []
    rows: List[Dict[str, Any]] = []
    for expert in candidate_experts:
        col = int(expert_to_col[int(expert)])
        support_nelbo = float(np.mean(nelbo_matrix[support_idxs, col])) if support_idxs.size else float("nan")
        label_nelbo = float(np.mean(nelbo_matrix[eval_idxs, col])) if eval_idxs.size else float("nan")
        support_mean.append(support_nelbo)
        eval_mean.append(label_nelbo)
        row = {
            "outer_target_domain": int(outer_target_domain),
            "pseudo_query_domain": int(query_domain),
            "query_domain": int(query_domain),
            "candidate_expert": int(expert),
            "expert_domain": int(expert),
            "split_role": str(split_role),
            "support_eval_split_id": str(split.support_eval_split_id),
            "support_size_requested": int(split.support_size_requested),
            "support_seed": int(split.support_eval_split_id.split("_seed", 1)[1].split("_", 1)[0]),
            "label_nelbo": label_nelbo,
            "support_mean_nelbo": support_nelbo,
            **_static_features(
                query_domain=int(query_domain),
                candidate_expert=int(expert),
                support_indices=split.support_indices,
                embeddings=embeddings,
                centroids=centroids,
                expert_domains=expert_domains,
                strategy=strategy,
                tau=float(tau),
            ),
            **dict(response_feature_fn(split.support_indices, int(expert), str(split.support_eval_split_id))),
        }
        rows.append(row)
    return rows, np.asarray(support_mean, dtype=np.float64), np.asarray(eval_mean, dtype=np.float64)


def evaluate_support_response_routing_from_arrays(
    *,
    embeddings: np.ndarray,
    metadata: Sequence[Mapping[str, object]],
    nelbo_matrix: np.ndarray,
    expert_domains: Sequence[int],
    seed: int,
    dataset_name: str,
    strategy: str,
    tau: float,
    support_cfg: SupportResponseConfig,
    reports_dir: Path,
    response_feature_fn: Callable[[Sequence[int], int, str], Mapping[str, float]],
    data_cfg: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if not support_cfg.enabled:
        return {"enabled": False}
    if support_cfg.ranker != "linear_pairwise_ridge":
        raise ValueError("support_response_routing.ranker must be 'linear_pairwise_ridge'")
    if support_cfg.tie_policy != "stable_expert_index":
        raise ValueError("support_response_routing.tie_policy must be 'stable_expert_index'")
    if len(metadata) != int(embeddings.shape[0]):
        raise ValueError("Embedding and metadata lengths do not match")
    if nelbo_matrix.shape != (int(embeddings.shape[0]), len(expert_domains)):
        raise ValueError("NELBO matrix must be n_samples x n_experts")

    expert_domains_int = [int(d) for d in expert_domains]
    expert_to_col = {int(e): idx for idx, e in enumerate(expert_domains_int)}
    sample_domains = np.asarray([_as_domain(m["magnification"]) for m in metadata], dtype=np.int64)
    labels_by_index = {idx: _as_label(m) for idx, m in enumerate(metadata)}
    centroids = _domain_centroids(np.asarray(embeddings, dtype=np.float64), sample_domains)
    privacy = _privacy_fields(data_cfg)

    sample_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    feature_audit_rows: List[Dict[str, Any]] = []
    sample_index_counter = 0

    for outer_target in sorted(set(int(v) for v in sample_domains.tolist())):
        target_indices = [int(i) for i, d in enumerate(sample_domains.tolist()) if int(d) == int(outer_target)]
        target_fold = FoldCandidateSet.for_heldout_domain(
            heldout_domain=int(outer_target),
            expert_domains=expert_domains_int,
        )
        target_candidates = list(target_fold.candidate_expert_domains)

        for support_seed in support_cfg.support_seeds:
            for sampling_policy in support_cfg.sampling_policies:
                for support_size in support_cfg.support_sizes:
                    target_split = make_support_eval_split(
                        target_domain=int(outer_target),
                        target_indices=target_indices,
                        labels_by_index=labels_by_index,
                        support_size=int(support_size),
                        sampling_policy=str(sampling_policy),
                        support_seed=int(support_seed),
                    )
                    split_rows.append(
                        _support_split_manifest_row(
                            run_seed=int(seed),
                            outer_target_domain=int(outer_target),
                            query_domain=int(outer_target),
                            split=target_split,
                            split_role="target",
                            privacy_fields=privacy,
                        )
                    )
                    if target_split.split_status != "ok":
                        continue

                    source_rows: List[Dict[str, Any]] = []
                    source_split_by_domain: Dict[int, Any] = {}
                    source_eval_by_query_expert: Dict[Tuple[int, int], float] = {}
                    for pseudo_query in sorted(set(int(v) for v in sample_domains.tolist()) - {int(outer_target)}):
                        pseudo_indices = [
                            int(i)
                            for i, d in enumerate(sample_domains.tolist())
                            if int(d) == int(pseudo_query)
                        ]
                        pseudo_split = make_support_eval_split(
                            target_domain=int(pseudo_query),
                            target_indices=pseudo_indices,
                            labels_by_index=labels_by_index,
                            support_size=int(support_size),
                            sampling_policy=str(sampling_policy),
                            support_seed=int(support_seed),
                        )
                        source_split_by_domain[int(pseudo_query)] = pseudo_split
                        split_rows.append(
                            _support_split_manifest_row(
                                run_seed=int(seed),
                                outer_target_domain=int(outer_target),
                                query_domain=int(pseudo_query),
                                split=pseudo_split,
                                split_role="source_train",
                                privacy_fields=privacy,
                            )
                        )
                        if pseudo_split.split_status != "ok":
                            continue
                        source_fold = FoldCandidateSet.for_heldout_domain(
                            heldout_domain=int(outer_target),
                            expert_domains=expert_domains_int,
                            excluded_domains=[int(pseudo_query)],
                        )
                        rows, _support_mean, eval_mean = _candidate_rows_for_query(
                            outer_target_domain=int(outer_target),
                            query_domain=int(pseudo_query),
                            candidate_experts=source_fold.candidate_expert_domains,
                            split=pseudo_split,
                            embeddings=embeddings,
                            centroids=centroids,
                            nelbo_matrix=nelbo_matrix,
                            expert_domains=expert_domains_int,
                            expert_to_col=expert_to_col,
                            strategy=strategy,
                            tau=float(tau),
                            response_feature_fn=response_feature_fn,
                            split_role="source_train",
                        )
                        source_rows.extend(rows)
                        for expert, value in zip(source_fold.candidate_expert_domains, eval_mean.tolist()):
                            source_eval_by_query_expert[(int(pseudo_query), int(expert))] = float(value)

                    if not source_rows:
                        continue

                    target_rows, target_support_mean, target_eval_mean = _candidate_rows_for_query(
                        outer_target_domain=int(outer_target),
                        query_domain=int(outer_target),
                        candidate_experts=target_candidates,
                        split=target_split,
                        embeddings=embeddings,
                        centroids=centroids,
                        nelbo_matrix=nelbo_matrix,
                        expert_domains=expert_domains_int,
                        expert_to_col=expert_to_col,
                        strategy=strategy,
                        tau=float(tau),
                        response_feature_fn=response_feature_fn,
                        split_role="target",
                    )

                    # Matched non-learned baselines.
                    metadata_scores = [float(row["metadata_distance"]) for row in target_rows]
                    sample_rows.append(
                        _score_method_row(
                            method="support_metadata_routing",
                            fold=target_fold,
                            target_domain=int(outer_target),
                            support_seed=int(support_seed),
                            support_size=int(support_size),
                            sampling_policy=str(sampling_policy),
                            support_eval_split_id=target_split.support_eval_split_id,
                            candidate_experts=target_candidates,
                            predicted_scores=metadata_scores,
                            eval_mean_nelbo=target_eval_mean,
                            support_mean_nelbo=target_support_mean,
                            sample_index=sample_index_counter,
                            run_seed=int(seed),
                            privacy_fields=privacy,
                        )
                    )
                    sample_index_counter += 1

                    embedding_scores = [float(row["embedding_distance"]) for row in target_rows]
                    sample_rows.append(
                        _score_method_row(
                            method="support_static_embedding_routing",
                            fold=target_fold,
                            target_domain=int(outer_target),
                            support_seed=int(support_seed),
                            support_size=int(support_size),
                            sampling_policy=str(sampling_policy),
                            support_eval_split_id=target_split.support_eval_split_id,
                            candidate_experts=target_candidates,
                            predicted_scores=embedding_scores,
                            eval_mean_nelbo=target_eval_mean,
                            support_mean_nelbo=target_support_mean,
                            sample_index=sample_index_counter,
                            run_seed=int(seed),
                            privacy_fields=privacy,
                        )
                    )
                    sample_index_counter += 1

                    sample_rows.append(
                        _score_method_row(
                            method="support_set_nelbo_top1",
                            fold=target_fold,
                            target_domain=int(outer_target),
                            support_seed=int(support_seed),
                            support_size=int(support_size),
                            sampling_policy=str(sampling_policy),
                            support_eval_split_id=target_split.support_eval_split_id,
                            candidate_experts=target_candidates,
                            predicted_scores=target_support_mean,
                            eval_mean_nelbo=target_eval_mean,
                            support_mean_nelbo=target_support_mean,
                            sample_index=sample_index_counter,
                            run_seed=int(seed),
                            privacy_fields=privacy,
                        )
                    )
                    sample_index_counter += 1

                    prior_scores: List[float] = []
                    for expert in target_candidates:
                        vals = [
                            v
                            for (pseudo_query, candidate), v in source_eval_by_query_expert.items()
                            if int(candidate) == int(expert)
                            and int(pseudo_query) != int(outer_target)
                            and int(pseudo_query) != int(expert)
                        ]
                        prior_scores.append(float(np.mean(vals)) if vals else float("inf"))
                    sample_rows.append(
                        _score_method_row(
                            method="source_global_prior_routing",
                            fold=target_fold,
                            target_domain=int(outer_target),
                            support_seed=int(support_seed),
                            support_size=int(support_size),
                            sampling_policy=str(sampling_policy),
                            support_eval_split_id=target_split.support_eval_split_id,
                            candidate_experts=target_candidates,
                            predicted_scores=prior_scores,
                            eval_mean_nelbo=target_eval_mean,
                            support_mean_nelbo=target_support_mean,
                            sample_index=sample_index_counter,
                            run_seed=int(seed),
                            privacy_fields=privacy,
                        )
                    )
                    sample_index_counter += 1

                    sample_rows.append(
                        _score_method_row(
                            method="support_candidate_oracle",
                            fold=target_fold,
                            target_domain=int(outer_target),
                            support_seed=int(support_seed),
                            support_size=int(support_size),
                            sampling_policy=str(sampling_policy),
                            support_eval_split_id=target_split.support_eval_split_id,
                            candidate_experts=target_candidates,
                            predicted_scores=target_eval_mean,
                            eval_mean_nelbo=target_eval_mean,
                            support_mean_nelbo=target_support_mean,
                            sample_index=sample_index_counter,
                            run_seed=int(seed),
                            privacy_fields=privacy,
                        )
                    )
                    sample_index_counter += 1

                    learned_regimes = list(dict.fromkeys(list(support_cfg.feature_regimes) + ["expert_id_only"]))
                    for regime in learned_regimes:
                        method = (
                            "expert_id_only_pairwise"
                            if regime == "expert_id_only"
                            else f"support_response_pairwise_{regime}"
                        )
                        train_rows = source_rows
                        target_feature_rows = target_rows
                        if regime == "response_indirect_shuffled":
                            train_rows = _shuffle_response_features(
                                source_rows,
                                dataset=dataset_name,
                                seed=int(seed),
                                fold_id=f"outer{outer_target}",
                                split_id=f"{support_seed}_{support_size}_{sampling_policy}_train",
                            )
                            target_feature_rows = _shuffle_response_features(
                                target_rows,
                                dataset=dataset_name,
                                seed=int(seed),
                                fold_id=f"outer{outer_target}",
                                split_id=f"{support_seed}_{support_size}_{sampling_policy}_target",
                            )

                        allow_identity = regime == "expert_id_only"
                        train_audit = audit_support_response_features(
                            train_rows,
                            regime=regime,
                            allow_candidate_identity=allow_identity,
                        )
                        for feature in sorted(set(train_audit.feature_names + train_audit.blocked_features + train_audit.dropped_zero_variance + train_audit.missing_features)):
                            feature_audit_rows.append(
                                {
                                    **dict(privacy),
                                    "seed": int(seed),
                                    "outer_target_domain": int(outer_target),
                                    "support_seed": int(support_seed),
                                    "support_size_requested": int(support_size),
                                    "sampling_policy": str(sampling_policy),
                                    "feature_regime": str(regime),
                                    "feature_name": str(feature),
                                    "included": int(feature in train_audit.feature_names),
                                    "blocked": int(feature in train_audit.blocked_features),
                                    "dropped_zero_variance": int(feature in train_audit.dropped_zero_variance),
                                    "missing": int(feature in train_audit.missing_features),
                                    "blocked_feature_terms": "|".join(train_audit.blocked_feature_terms),
                                    "no_data_reason": train_audit.no_data_reason,
                                    "scaler_fit_scope": "source_training_pairs_only",
                                }
                            )
                        scaler = fit_support_response_scaler(train_audit)
                        x_train = scaler.transform(train_audit.matrix)
                        target_audit = audit_support_response_features(
                            target_feature_rows,
                            regime=regime,
                            feature_names=scaler.feature_names,
                            allow_candidate_identity=allow_identity,
                            drop_zero_variance=False,
                        )
                        x_target = scaler.transform(target_audit.matrix)
                        pairs, train_pair_rows = build_candidate_specific_pairs(train_rows)
                        for pair_row in train_pair_rows:
                            pair_rows.append(
                                {
                                    **dict(privacy),
                                    "seed": int(seed),
                                    "outer_target_domain": int(outer_target),
                                    "support_seed": int(support_seed),
                                    "support_size_requested": int(support_size),
                                    "sampling_policy": str(sampling_policy),
                                    "feature_regime": str(regime),
                                    "method": str(method),
                                    **pair_row,
                                }
                            )
                        ranker = LinearPairwiseRidge(ridge_l2=float(support_cfg.ridge_l2))
                        ranker.fit(x_train, pairs)
                        predicted = ranker.predict(x_target)
                        sample_rows.append(
                            _score_method_row(
                                method=method,
                                fold=target_fold,
                                target_domain=int(outer_target),
                                support_seed=int(support_seed),
                                support_size=int(support_size),
                                sampling_policy=str(sampling_policy),
                                support_eval_split_id=target_split.support_eval_split_id,
                                candidate_experts=target_candidates,
                                predicted_scores=predicted,
                                eval_mean_nelbo=target_eval_mean,
                                support_mean_nelbo=target_support_mean,
                                sample_index=sample_index_counter,
                                run_seed=int(seed),
                                privacy_fields=privacy,
                            )
                        )
                        sample_index_counter += 1

                        if support_cfg.source_leave_pseudo_domain_out_diagnostic and regime == support_cfg.primary_feature_regime:
                            for validation_domain in sorted(source_split_by_domain):
                                validation_rows = [
                                    r for r in source_rows if int(r["pseudo_query_domain"]) == int(validation_domain)
                                ]
                                inner_rows = [
                                    r for r in source_rows if int(r["pseudo_query_domain"]) != int(validation_domain)
                                ]
                                if not validation_rows or not inner_rows:
                                    continue
                                inner_audit = audit_support_response_features(
                                    inner_rows,
                                    regime=regime,
                                    allow_candidate_identity=False,
                                )
                                inner_scaler = fit_support_response_scaler(inner_audit)
                                inner_x = inner_scaler.transform(inner_audit.matrix)
                                validation_audit = audit_support_response_features(
                                    validation_rows,
                                    regime=regime,
                                    feature_names=inner_scaler.feature_names,
                                    allow_candidate_identity=False,
                                    drop_zero_variance=False,
                                )
                                validation_x = inner_scaler.transform(validation_audit.matrix)
                                inner_pairs, _inner_pair_rows = build_candidate_specific_pairs(inner_rows)
                                inner_ranker = LinearPairwiseRidge(ridge_l2=float(support_cfg.ridge_l2))
                                inner_ranker.fit(inner_x, inner_pairs)
                                validation_scores = inner_ranker.predict(validation_x)
                                validation_candidates = [int(r["candidate_expert"]) for r in validation_rows]
                                validation_eval = np.asarray(
                                    [float(r["label_nelbo"]) for r in validation_rows],
                                    dtype=np.float64,
                                )
                                validation_support = np.asarray(
                                    [float(r["support_mean_nelbo"]) for r in validation_rows],
                                    dtype=np.float64,
                                )
                                validation_fold = FoldCandidateSet.for_heldout_domain(
                                    heldout_domain=int(outer_target),
                                    expert_domains=expert_domains_int,
                                    excluded_domains=[int(validation_domain)],
                                )
                                diag = _score_method_row(
                                    method="source_leave_pseudo_domain_out_ranker_diagnostic",
                                    fold=validation_fold,
                                    target_domain=int(validation_domain),
                                    support_seed=int(support_seed),
                                    support_size=int(support_size),
                                    sampling_policy=str(sampling_policy),
                                    support_eval_split_id=str(source_split_by_domain[int(validation_domain)].support_eval_split_id),
                                    candidate_experts=validation_candidates,
                                    predicted_scores=validation_scores,
                                    eval_mean_nelbo=validation_eval,
                                    support_mean_nelbo=validation_support,
                                    sample_index=sample_index_counter,
                                    run_seed=int(seed),
                                    privacy_fields=privacy,
                                )
                                diag["outer_target_domain"] = int(outer_target)
                                diag["validation_pseudo_query_domain"] = int(validation_domain)
                                sample_rows.append(diag)
                                sample_index_counter += 1

    method_metrics = _aggregate_metrics_from_sample_rows(sample_rows) if sample_rows else {}
    domain_rows = [
        {**dict(privacy), **row}
        for row in (_domain_breakdown_rows(sample_rows) if sample_rows else [])
    ]
    method_summary = [
        {**dict(privacy), "method": method, **metrics}
        for method, metrics in sorted(method_metrics.items(), key=lambda item: item[0])
    ]
    artifacts = {
        "protocol_lock": "support_response_protocol_lock.json",
        "split_manifest": "support_response_split_manifest.csv",
        "feature_audit": "support_response_feature_audit.csv",
        "pair_predictions": "support_response_pair_predictions.csv",
        "sample_selections": "support_response_sample_selections.csv",
        "domain_breakdown": "support_response_domain_breakdown.csv",
        "method_summary": "support_response_method_summary.csv",
        "results": "support_response_results.json",
    }
    protocol_lock = {
        "protocol_version": SUPPORT_RESPONSE_PROTOCOL_VERSION,
        "score_direction": "predicted_score_is_predicted_mean_nelbo_lower_is_better",
        "candidate_label_definition": "mean_NELBO(pseudo_query_eval, candidate_expert)",
        "pairwise_comparison_scope": "within_same_pseudo_query_domain",
        "ranker": support_cfg.ranker,
        "ridge_l2": float(support_cfg.ridge_l2),
        "scaler_fit_scope": "source_training_pairs_only",
        "domain_level_aggregation": bool(support_cfg.domain_level_aggregation),
        **privacy,
    }
    results = {
        **privacy,
        "enabled": True,
        "protocol_version": SUPPORT_RESPONSE_PROTOCOL_VERSION,
        "protocol_lock": protocol_lock,
        "metrics_by_method": method_metrics,
        "artifacts": artifacts,
        "n_domain_level_rows": int(len(sample_rows)),
        "n_pairwise_training_comparisons": int(len(pair_rows)),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / artifacts["protocol_lock"]).write_text(json.dumps(protocol_lock, indent=2) + "\n", encoding="utf-8")
    _write_csv(reports_dir / artifacts["split_manifest"], split_rows)
    _write_csv(reports_dir / artifacts["feature_audit"], feature_audit_rows)
    _write_csv(reports_dir / artifacts["pair_predictions"], pair_rows)
    _write_csv(reports_dir / artifacts["sample_selections"], sample_rows)
    _write_csv(reports_dir / artifacts["domain_breakdown"], domain_rows)
    _write_csv(reports_dir / artifacts["method_summary"], method_summary)
    (reports_dir / artifacts["results"]).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def evaluate_support_response_routing_for_checkpoints(
    *,
    embeddings: np.ndarray,
    metadata: Sequence[Mapping[str, object]],
    nelbo_matrix: np.ndarray,
    expert_domains: Sequence[int],
    expert_checkpoints: Mapping[str, str],
    hidden_dim: int,
    latent_dim: int,
    seed: int,
    dataset_name: str,
    strategy: str,
    tau: float,
    support_cfg: SupportResponseConfig,
    reports_dir: Path,
    data_cfg: Mapping[str, Any] | None = None,
    metadata_constraint_cfg: Mapping[str, object] | None = None,
) -> Dict[str, Any]:
    x_cpu = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank = _ResponseExpertBank(
        expert_checkpoints=expert_checkpoints,
        input_dim=int(x_cpu.shape[1]),
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        device=device,
        metadata_constraint_cfg=metadata_constraint_cfg,
    )
    cache: Dict[Tuple[str, int], Dict[str, float]] = {}

    def response_feature_fn(support_indices: Sequence[int], expert_domain: int, split_id: str) -> Mapping[str, float]:
        key = (str(split_id), int(expert_domain))
        if key not in cache:
            repeat_seed = stable_response_seed(
                dataset=str(dataset_name),
                seed=int(seed),
                query_id=str(split_id),
                expert_domain=int(expert_domain),
                repeat_id=0,
                stream_name="support_response_feature",
            )
            cache[key] = compute_response_features(
                bank=bank,  # type: ignore[arg-type]
                expert_domain=int(expert_domain),
                x_cpu=x_cpu,
                support_idxs=list(int(i) for i in support_indices),
                device=device,
                n_repeats=int(support_cfg.num_response_repeats),
                repeat_seed_base=int(repeat_seed),
                include_residual_shape_features=bool(support_cfg.include_residual_shape_features),
            )
        return cache[key]

    return evaluate_support_response_routing_from_arrays(
        embeddings=embeddings,
        metadata=metadata,
        nelbo_matrix=nelbo_matrix,
        expert_domains=expert_domains,
        seed=int(seed),
        dataset_name=str(dataset_name),
        strategy=strategy,
        tau=float(tau),
        support_cfg=support_cfg,
        reports_dir=reports_dir,
        response_feature_fn=response_feature_fn,
        data_cfg=data_cfg,
    )
