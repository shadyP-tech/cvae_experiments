from __future__ import annotations

import csv
from dataclasses import dataclass, field
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
RISK_CONSTRAINED_METHOD = "risk_constrained_response_routing"
RISK_CONSTRAINED_POLICY_NAME = "metadata_anchored_response_routing_with_support_regret_gate"
SUPPORT_CONSERVATIVE_METHOD = "support_set_nelbo_conservative"
SUPPORT_ALPHA_SELECTION_POLICY = "source_inner_gap_min_with_non_regression"
SUPPORT_ALPHA_GRID_DEFAULT = (0.0, 0.5, 1.0, 1.5, 2.0)
SUPPORT_ALPHA_SPEARMAN_TOLERANCE = 0.05
HIGH_REGRET_GAP_PCT_THRESHOLD = 2.0
BOTTOM_HALF_RANK_THRESHOLD = 3
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
class RiskConstrainedResponseConfig:
    enabled: bool
    margin_thresholds: Tuple[float, ...]
    support_regret_thresholds: Tuple[float, ...]
    top1_tolerance: float
    spearman_tolerance: float
    focus_query_domain: int
    focus_expert: int


@dataclass(frozen=True)
class SupportUtilityConfig:
    enabled: bool = False
    alpha_grid: Tuple[float, ...] = SUPPORT_ALPHA_GRID_DEFAULT
    alpha_selection_policy: str = SUPPORT_ALPHA_SELECTION_POLICY
    require_unlabeled_support: bool = True


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
    support_utility: SupportUtilityConfig = field(default_factory=SupportUtilityConfig)
    risk_constrained: RiskConstrainedResponseConfig = field(
        default_factory=lambda: RiskConstrainedResponseConfig(
            enabled=False,
            margin_thresholds=(0.0, 0.25, 0.5, 1.0, 1.5),
            support_regret_thresholds=(0.0, 2.5, 5.0, 10.0),
            top1_tolerance=0.02,
            spearman_tolerance=0.05,
            focus_query_domain=3,
            focus_expert=4,
        )
    )


@dataclass(frozen=True)
class _RiskThresholdSelection:
    tau_margin: float
    tau_regret: float
    num_source_inner_units: int
    source_inner_top1: float
    source_inner_spearman: float
    source_inner_gap_pct: float
    source_inner_harmful_override_rate: float
    fallback_used: bool


@dataclass(frozen=True)
class _SupportAlphaSelection:
    selected_alpha: float
    num_source_inner_units: int
    source_inner_gap_pct_alpha0: float
    source_inner_gap_pct_selected: float
    source_inner_top1_alpha0: float
    source_inner_top1_selected: float
    source_inner_spearman_alpha0: float
    source_inner_spearman_selected: float
    source_inner_gap_variance_alpha0: float
    source_inner_gap_variance_selected: float
    fallback_to_alpha0: bool
    n_aggregation_units: int
    top1_tolerance_abs: float


@dataclass(frozen=True)
class _RiskPolicyDecision:
    selected_expert: int
    metadata_anchor_expert: int
    response_proposal_expert: int
    confidence_margin: float
    support_regret_pct_vs_anchor: float
    override_candidate: int
    accepted_override: int
    true_harmful_override: int
    true_improving_override: int


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
    risk_raw = raw.get("risk_constrained_response_routing", {})
    if risk_raw is None:
        risk_raw = {}
    if not isinstance(risk_raw, Mapping):
        raise ValueError(
            "learned_utility.support_response_routing.risk_constrained_response_routing must be a dictionary"
        )
    risk_cfg = RiskConstrainedResponseConfig(
        enabled=bool(risk_raw.get("enabled", False)),
        margin_thresholds=tuple(float(v) for v in risk_raw.get("margin_thresholds", [0.0, 0.25, 0.5, 1.0, 1.5])),
        support_regret_thresholds=tuple(
            float(v) for v in risk_raw.get("support_regret_thresholds", [0.0, 2.5, 5.0, 10.0])
        ),
        top1_tolerance=float(risk_raw.get("top1_tolerance", 0.02)),
        spearman_tolerance=float(risk_raw.get("spearman_tolerance", 0.05)),
        focus_query_domain=int(risk_raw.get("focus_query_domain", 3)),
        focus_expert=int(risk_raw.get("focus_expert", 4)),
    )
    utility_raw = raw.get("support_utility", {})
    if utility_raw is None:
        utility_raw = {}
    if not isinstance(utility_raw, Mapping):
        raise ValueError("learned_utility.support_response_routing.support_utility must be a dictionary")
    support_utility_cfg = SupportUtilityConfig(
        enabled=bool(utility_raw.get("enabled", False)),
        alpha_grid=tuple(float(v) for v in utility_raw.get("alpha_grid", SUPPORT_ALPHA_GRID_DEFAULT)),
        alpha_selection_policy=str(
            utility_raw.get("alpha_selection_policy", SUPPORT_ALPHA_SELECTION_POLICY)
        ).strip(),
        require_unlabeled_support=bool(utility_raw.get("require_unlabeled_support", True)),
    )
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
        support_utility=support_utility_cfg,
        risk_constrained=risk_cfg,
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


def _support_stderr(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(float(n)))


def _support_eval_stats_for_candidates(
    *,
    split: Any,
    candidate_experts: Sequence[int],
    nelbo_matrix: np.ndarray,
    expert_to_col: Mapping[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    support_idxs = np.asarray(split.support_indices, dtype=np.int64)
    eval_idxs = np.asarray(split.eval_indices, dtype=np.int64)
    support_mean: List[float] = []
    support_stderr: List[float] = []
    eval_mean: List[float] = []
    for expert in candidate_experts:
        col = int(expert_to_col[int(expert)])
        if support_idxs.size:
            support_values = np.asarray(nelbo_matrix[support_idxs, col], dtype=np.float64)
            support_mean.append(float(np.mean(support_values)))
            support_stderr.append(_support_stderr(support_values))
        else:
            support_mean.append(float("nan"))
            support_stderr.append(float("nan"))
        eval_mean.append(float(np.mean(nelbo_matrix[eval_idxs, col])) if eval_idxs.size else float("nan"))
    return (
        np.asarray(support_mean, dtype=np.float64),
        np.asarray(support_stderr, dtype=np.float64),
        np.asarray(eval_mean, dtype=np.float64),
    )


def _conservative_support_scores(
    support_mean_nelbo: Sequence[float],
    support_stderr_nelbo: Sequence[float],
    alpha: float,
) -> np.ndarray:
    return np.asarray(support_mean_nelbo, dtype=np.float64) + (
        float(alpha) * np.asarray(support_stderr_nelbo, dtype=np.float64)
    )


def _gap_variance(rows: Sequence[Mapping[str, Any]]) -> float:
    vals = [float(row.get("oracle_gap_pct", float("nan"))) for row in rows]
    clean = [v for v in vals if np.isfinite(v)]
    return float(np.var(clean)) if clean else 0.0


def _alpha_grid_label(values: Sequence[float]) -> str:
    return json.dumps([float(v) for v in values], separators=(",", ":"))


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
    support_stderr_nelbo: Sequence[float] | None,
    sample_index: int,
    run_seed: int,
    privacy_fields: Mapping[str, Any],
    selected_expert_override: int | None = None,
    ranking_scores: Sequence[float] | None = None,
    alpha: float = 0.0,
    support_n: int = 0,
    support_labels_used_for_routing: int = 0,
    conservative_scores: Sequence[float] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if str(method) == "support_candidate_oracle":
        method_protocol = _method_protocol("support_candidate_oracle")
    else:
        method_protocol = _method_protocol(method)
    if selected_expert_override is None:
        selected_idx = _stable_argmin(predicted_scores, candidate_experts)
    else:
        selected_lookup = {int(expert): idx for idx, expert in enumerate(candidate_experts)}
        if int(selected_expert_override) not in selected_lookup:
            raise ProtocolError(
                f"selected_expert_override={selected_expert_override} is outside candidate experts"
            )
        selected_idx = int(selected_lookup[int(selected_expert_override)])
    oracle_idx = _stable_argmin(eval_mean_nelbo, candidate_experts)
    selected_expert = int(candidate_experts[selected_idx])
    oracle_expert = int(candidate_experts[oracle_idx])
    selected_nelbo = float(eval_mean_nelbo[selected_idx])
    oracle_nelbo = float(eval_mean_nelbo[oracle_idx])
    gap = float(selected_nelbo - oracle_nelbo)
    gap_pct = float((gap / max(abs(oracle_nelbo), 1e-12)) * 100.0)
    rank_score = np.asarray(ranking_scores if ranking_scores is not None else predicted_scores, dtype=np.float64)
    true_nelbo = np.asarray(eval_mean_nelbo, dtype=np.float64)
    selected_rank = _selected_rank(selected_idx, true_nelbo, candidate_experts)
    support_mean_arr = np.asarray(support_mean_nelbo, dtype=np.float64)
    support_stderr_arr = (
        np.asarray(support_stderr_nelbo, dtype=np.float64)
        if support_stderr_nelbo is not None
        else np.zeros_like(support_mean_arr, dtype=np.float64)
    )
    conservative_arr = (
        np.asarray(conservative_scores, dtype=np.float64)
        if conservative_scores is not None
        else _conservative_support_scores(support_mean_arr, support_stderr_arr, alpha=float(alpha))
    )
    base = _protocol_row_fields(fold=fold, method_protocol=method_protocol, method=method)
    base["protocol_version"] = SUPPORT_RESPONSE_PROTOCOL_VERSION
    base["aggregation_source"] = SUPPORT_RESPONSE_AGGREGATION_SOURCE
    row = {
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
        "selected_rank": selected_rank,
        "spearman": float(spearman_corr((-rank_score).tolist(), (-true_nelbo).tolist())),
        "pairwise_auc": _pairwise_auc(rank_score, true_nelbo),
        "mean_support_nelbo": float(support_mean_arr[int(selected_idx)]),
        "stderr_support_nelbo": float(support_stderr_arr[int(selected_idx)]),
        "conservative_support_score": float(conservative_arr[int(selected_idx)]),
        "alpha": float(alpha),
        "support_n": int(support_n),
        "support_labels_used_for_routing": int(support_labels_used_for_routing),
        "bottom_half_selection": int(float(selected_rank) >= float(BOTTOM_HALF_RANK_THRESHOLD)),
        "high_regret_selection": int(float(gap_pct) >= float(HIGH_REGRET_GAP_PCT_THRESHOLD)),
        "catastrophic_mistake": int(
            float(selected_rank) >= float(BOTTOM_HALF_RANK_THRESHOLD)
            or float(gap_pct) >= float(HIGH_REGRET_GAP_PCT_THRESHOLD)
        ),
        "predicted_score_by_expert_json": _json_mapping(candidate_experts, predicted_scores),
        "eval_nelbo_by_expert_json": _json_mapping(candidate_experts, eval_mean_nelbo),
        "support_nelbo_by_expert_json": _json_mapping(candidate_experts, support_mean_nelbo),
        "support_stderr_nelbo_by_expert_json": _json_mapping(candidate_experts, support_stderr_arr),
        "conservative_support_score_by_expert_json": _json_mapping(candidate_experts, conservative_arr),
    }
    if extra_fields:
        row.update(dict(extra_fields))
    return row


def _risk_policy_decision(
    *,
    candidate_experts: Sequence[int],
    learned_scores: Sequence[float],
    metadata_scores: Sequence[float],
    support_mean_nelbo: Sequence[float],
    eval_mean_nelbo: Sequence[float],
    tau_margin: float,
    tau_regret: float,
) -> _RiskPolicyDecision:
    learned_arr = np.asarray(learned_scores, dtype=np.float64)
    metadata_arr = np.asarray(metadata_scores, dtype=np.float64)
    support_arr = np.asarray(support_mean_nelbo, dtype=np.float64)
    eval_arr = np.asarray(eval_mean_nelbo, dtype=np.float64)
    experts = [int(e) for e in candidate_experts]

    metadata_idx = _stable_argmin(metadata_arr, experts)
    proposal_idx = _stable_argmin(learned_arr, experts)
    order = sorted(range(len(experts)), key=lambda i: (float(learned_arr[i]), int(experts[i])))
    confidence_margin = (
        float(learned_arr[int(order[1])] - learned_arr[int(order[0])])
        if len(order) >= 2
        else 0.0
    )
    anchor_support = float(support_arr[int(metadata_idx)])
    proposal_support = float(support_arr[int(proposal_idx)])
    support_regret = float(
        ((proposal_support - anchor_support) / max(abs(anchor_support), 1e-12)) * 100.0
    )
    override_candidate = int(proposal_idx != metadata_idx)
    accepted = int(
        bool(override_candidate)
        and confidence_margin >= float(tau_margin)
        and support_regret <= float(tau_regret)
    )
    selected_idx = int(proposal_idx if accepted else metadata_idx)
    selected_eval = float(eval_arr[int(selected_idx)])
    metadata_eval = float(eval_arr[int(metadata_idx)])
    return _RiskPolicyDecision(
        selected_expert=int(experts[int(selected_idx)]),
        metadata_anchor_expert=int(experts[int(metadata_idx)]),
        response_proposal_expert=int(experts[int(proposal_idx)]),
        confidence_margin=float(confidence_margin),
        support_regret_pct_vs_anchor=float(support_regret),
        override_candidate=int(override_candidate),
        accepted_override=int(accepted),
        true_harmful_override=int(bool(accepted) and selected_eval > metadata_eval + 1e-12),
        true_improving_override=int(bool(accepted) and selected_eval < metadata_eval - 1e-12),
    )


def _risk_extra_fields(
    *,
    decision: _RiskPolicyDecision,
    tau_margin: float,
    tau_regret: float,
    selection: _RiskThresholdSelection,
    focus_query_domain: int,
    focus_expert: int,
    query_domain: int,
) -> Dict[str, Any]:
    focus_candidate = int(
        decision.response_proposal_expert == int(focus_expert)
        and bool(decision.override_candidate)
    )
    return {
        "policy_name": RISK_CONSTRAINED_POLICY_NAME,
        "threshold_selection_policy": "source_inner_only",
        "selection_source": "source_inner_only",
        "tau_margin": float(tau_margin),
        "tau_regret": float(tau_regret),
        "selected_tau": f"margin={float(tau_margin):.6g};regret={float(tau_regret):.6g}",
        "fallback_used": int(bool(selection.fallback_used)),
        "created_before_target_eval_scoring": 1,
        "score_source": "learned_response_scores",
        "risk_gate_source": "support_nelbo_regret_vs_metadata_anchor",
        "metadata_anchor_expert": int(decision.metadata_anchor_expert),
        "response_proposal_expert": int(decision.response_proposal_expert),
        "confidence_margin": float(decision.confidence_margin),
        "support_regret_pct_vs_anchor": float(decision.support_regret_pct_vs_anchor),
        "override_candidate": int(decision.override_candidate),
        "accepted_override": int(decision.accepted_override),
        "true_harmful_override": int(decision.true_harmful_override),
        "true_improving_override": int(decision.true_improving_override),
        "focus_query_domain": int(focus_query_domain),
        "focus_expert": int(focus_expert),
        "focus_query_row": int(int(query_domain) == int(focus_query_domain)),
        "focus_expert_override_candidate": int(focus_candidate),
        "focus_expert_override_accepted": int(
            bool(focus_candidate) and bool(decision.accepted_override)
        ),
        "focus_expert_override_blocked": int(
            bool(focus_candidate) and not bool(decision.accepted_override)
        ),
    }


def _score_risk_constrained_row(
    *,
    fold: FoldCandidateSet,
    target_domain: int,
    support_seed: int,
    support_size: int,
    sampling_policy: str,
    support_eval_split_id: str,
    candidate_experts: Sequence[int],
    learned_scores: Sequence[float],
    metadata_scores: Sequence[float],
    eval_mean_nelbo: Sequence[float],
    support_mean_nelbo: Sequence[float],
    support_stderr_nelbo: Sequence[float] | None,
    support_n: int,
    sample_index: int,
    run_seed: int,
    privacy_fields: Mapping[str, Any],
    selection: _RiskThresholdSelection,
    focus_query_domain: int,
    focus_expert: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    decision = _risk_policy_decision(
        candidate_experts=candidate_experts,
        learned_scores=learned_scores,
        metadata_scores=metadata_scores,
        support_mean_nelbo=support_mean_nelbo,
        eval_mean_nelbo=eval_mean_nelbo,
        tau_margin=float(selection.tau_margin),
        tau_regret=float(selection.tau_regret),
    )
    extra = _risk_extra_fields(
        decision=decision,
        tau_margin=float(selection.tau_margin),
        tau_regret=float(selection.tau_regret),
        selection=selection,
        focus_query_domain=int(focus_query_domain),
        focus_expert=int(focus_expert),
        query_domain=int(target_domain),
    )
    row = _score_method_row(
        method=RISK_CONSTRAINED_METHOD,
        fold=fold,
        target_domain=int(target_domain),
        support_seed=int(support_seed),
        support_size=int(support_size),
        sampling_policy=str(sampling_policy),
        support_eval_split_id=str(support_eval_split_id),
        candidate_experts=candidate_experts,
        predicted_scores=learned_scores,
        eval_mean_nelbo=eval_mean_nelbo,
        support_mean_nelbo=support_mean_nelbo,
        support_stderr_nelbo=support_stderr_nelbo,
        sample_index=int(sample_index),
        run_seed=int(run_seed),
        privacy_fields=privacy_fields,
        selected_expert_override=int(decision.selected_expert),
        ranking_scores=learned_scores,
        support_n=int(support_n),
        support_labels_used_for_routing=0,
        extra_fields=extra,
    )
    audit = {
        **dict(privacy_fields),
        "method": RISK_CONSTRAINED_METHOD,
        "seed": int(run_seed),
        "outer_center": int(fold.heldout_domain),
        "query_domain": int(target_domain),
        "support_seed": int(support_seed),
        "support_size_requested": int(support_size),
        "sampling_policy": str(sampling_policy),
        "support_eval_split_id": str(support_eval_split_id),
        "selected_expert": int(decision.selected_expert),
        **extra,
    }
    expert4 = {
        **audit,
        "expert4_audit_scope": "focus_expert_from_config",
    }
    return row, audit, expert4


def _harmful_override_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    accepted = [r for r in rows if int(float(r.get("accepted_override", 0) or 0)) == 1]
    if not accepted:
        return 0.0
    harmful = sum(int(float(r.get("true_harmful_override", 0) or 0)) for r in accepted)
    return float(harmful / max(len(accepted), 1))


def _fit_predict_support_response(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    regime: str,
    ridge_l2: float,
    allow_candidate_identity: bool = False,
) -> Tuple[np.ndarray, FeatureAuditResult, List[Dict[str, Any]]]:
    train_audit = audit_support_response_features(
        train_rows,
        regime=str(regime),
        allow_candidate_identity=bool(allow_candidate_identity),
    )
    scaler = fit_support_response_scaler(train_audit)
    x_train = scaler.transform(train_audit.matrix)
    target_audit = audit_support_response_features(
        target_rows,
        regime=str(regime),
        feature_names=scaler.feature_names,
        allow_candidate_identity=bool(allow_candidate_identity),
        drop_zero_variance=False,
    )
    x_target = scaler.transform(target_audit.matrix)
    pairs, pair_rows = build_candidate_specific_pairs(train_rows)
    ranker = LinearPairwiseRidge(ridge_l2=float(ridge_l2))
    ranker.fit(x_train, pairs)
    return ranker.predict(x_target), train_audit, pair_rows


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
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    support_mean, support_stderr, eval_mean = _support_eval_stats_for_candidates(
        split=split,
        candidate_experts=candidate_experts,
        nelbo_matrix=nelbo_matrix,
        expert_to_col=expert_to_col,
    )
    rows: List[Dict[str, Any]] = []
    for idx, expert in enumerate(candidate_experts):
        support_nelbo = float(support_mean[int(idx)])
        support_se = float(support_stderr[int(idx)])
        label_nelbo = float(eval_mean[int(idx)])
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
            "support_stderr_nelbo": support_se,
            "support_n": int(split.support_size_actual),
            "support_labels_used_for_routing": 0,
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
    return rows, support_mean, support_stderr, eval_mean


def _source_inner_units_for_outer(
    *,
    outer_target: int,
    support_cfg: SupportResponseConfig,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    labels_by_index: Mapping[int, int],
    centroids: Mapping[int, np.ndarray],
    nelbo_matrix: np.ndarray,
    expert_domains_int: Sequence[int],
    expert_to_col: Mapping[int, int],
    strategy: str,
    tau: float,
    response_feature_fn: Callable[[Sequence[int], int, str], Mapping[str, float]],
) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    source_domains = sorted(set(int(v) for v in sample_domains.tolist()) - {int(outer_target)})
    for support_seed in support_cfg.support_seeds:
        for sampling_policy in support_cfg.sampling_policies:
            for support_size in support_cfg.support_sizes:
                source_rows: List[Dict[str, Any]] = []
                source_split_by_domain: Dict[int, Any] = {}
                for pseudo_query in source_domains:
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
                    if pseudo_split.split_status != "ok":
                        continue
                    source_fold = FoldCandidateSet.for_heldout_domain(
                        heldout_domain=int(outer_target),
                        expert_domains=expert_domains_int,
                        excluded_domains=[int(pseudo_query)],
                    )
                    rows, _support_mean, _support_stderr, _eval_mean = _candidate_rows_for_query(
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
                        split_role="source_inner_threshold",
                    )
                    source_rows.extend(rows)
                if not source_rows:
                    continue

                for validation_domain in sorted(source_split_by_domain):
                    validation_rows = [
                        r for r in source_rows if int(r["pseudo_query_domain"]) == int(validation_domain)
                    ]
                    inner_rows = [
                        r for r in source_rows if int(r["pseudo_query_domain"]) != int(validation_domain)
                    ]
                    if not validation_rows or not inner_rows:
                        continue
                    predicted, _audit, _pairs = _fit_predict_support_response(
                        train_rows=inner_rows,
                        target_rows=validation_rows,
                        regime=support_cfg.primary_feature_regime,
                        ridge_l2=float(support_cfg.ridge_l2),
                        allow_candidate_identity=False,
                    )
                    validation_split = source_split_by_domain[int(validation_domain)]
                    units.append(
                        {
                            "outer_target": int(outer_target),
                            "validation_domain": int(validation_domain),
                            "support_seed": int(support_seed),
                            "support_size": int(support_size),
                            "sampling_policy": str(sampling_policy),
                            "support_eval_split_id": str(validation_split.support_eval_split_id),
                            "candidate_experts": [int(r["candidate_expert"]) for r in validation_rows],
                            "metadata_scores": [float(r["metadata_distance"]) for r in validation_rows],
                            "learned_scores": predicted.astype(np.float64),
                            "eval_mean_nelbo": [float(r["label_nelbo"]) for r in validation_rows],
                            "support_mean_nelbo": [float(r["support_mean_nelbo"]) for r in validation_rows],
                            "support_stderr_nelbo": [
                                float(r.get("support_stderr_nelbo", 0.0)) for r in validation_rows
                            ],
                            "support_n": int(validation_split.support_size_actual),
                            "fold": FoldCandidateSet.for_heldout_domain(
                                heldout_domain=int(outer_target),
                                expert_domains=expert_domains_int,
                                excluded_domains=[int(validation_domain)],
                            ),
                        }
                    )
    return units


def _score_source_inner_units(
    *,
    units: Sequence[Mapping[str, Any]],
    seed: int,
    privacy: Mapping[str, Any],
    selection: _RiskThresholdSelection,
    risk_cfg: RiskConstrainedResponseConfig,
    primary_feature_regime: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    metadata_rows: List[Dict[str, Any]] = []
    unrestricted_rows: List[Dict[str, Any]] = []
    risk_rows: List[Dict[str, Any]] = []
    for idx, unit in enumerate(units):
        fold = unit["fold"]
        candidate_experts = list(unit["candidate_experts"])
        metadata_scores = list(unit["metadata_scores"])
        learned_scores = list(unit["learned_scores"])
        eval_mean = list(unit["eval_mean_nelbo"])
        support_mean = list(unit["support_mean_nelbo"])
        support_stderr = list(unit.get("support_stderr_nelbo", [0.0 for _ in support_mean]))
        support_n = int(unit.get("support_n", 0))
        metadata_rows.append(
            _score_method_row(
                method="support_metadata_routing",
                fold=fold,
                target_domain=int(unit["validation_domain"]),
                support_seed=int(unit["support_seed"]),
                support_size=int(unit["support_size"]),
                sampling_policy=str(unit["sampling_policy"]),
                support_eval_split_id=str(unit["support_eval_split_id"]),
                candidate_experts=candidate_experts,
                predicted_scores=metadata_scores,
                eval_mean_nelbo=eval_mean,
                support_mean_nelbo=support_mean,
                support_stderr_nelbo=support_stderr,
                sample_index=idx,
                run_seed=int(seed),
                privacy_fields=privacy,
                support_n=support_n,
                support_labels_used_for_routing=0,
            )
        )
        unrestricted_rows.append(
            _score_method_row(
                method=f"support_response_pairwise_{primary_feature_regime}",
                fold=fold,
                target_domain=int(unit["validation_domain"]),
                support_seed=int(unit["support_seed"]),
                support_size=int(unit["support_size"]),
                sampling_policy=str(unit["sampling_policy"]),
                support_eval_split_id=str(unit["support_eval_split_id"]),
                candidate_experts=candidate_experts,
                predicted_scores=learned_scores,
                eval_mean_nelbo=eval_mean,
                support_mean_nelbo=support_mean,
                support_stderr_nelbo=support_stderr,
                sample_index=idx,
                run_seed=int(seed),
                privacy_fields=privacy,
                support_n=support_n,
                support_labels_used_for_routing=0,
            )
        )
        row, _audit, _expert4 = _score_risk_constrained_row(
            fold=fold,
            target_domain=int(unit["validation_domain"]),
            support_seed=int(unit["support_seed"]),
            support_size=int(unit["support_size"]),
            sampling_policy=str(unit["sampling_policy"]),
            support_eval_split_id=str(unit["support_eval_split_id"]),
            candidate_experts=candidate_experts,
            learned_scores=learned_scores,
            metadata_scores=metadata_scores,
            eval_mean_nelbo=eval_mean,
            support_mean_nelbo=support_mean,
            support_stderr_nelbo=support_stderr,
            support_n=support_n,
            sample_index=idx,
            run_seed=int(seed),
            privacy_fields=privacy,
            selection=selection,
            focus_query_domain=int(risk_cfg.focus_query_domain),
            focus_expert=int(risk_cfg.focus_expert),
        )
        risk_rows.append(row)
    return metadata_rows, unrestricted_rows, risk_rows


def _select_risk_threshold_for_outer(
    *,
    outer_target: int,
    support_cfg: SupportResponseConfig,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    labels_by_index: Mapping[int, int],
    centroids: Mapping[int, np.ndarray],
    nelbo_matrix: np.ndarray,
    expert_domains_int: Sequence[int],
    expert_to_col: Mapping[int, int],
    seed: int,
    strategy: str,
    tau: float,
    response_feature_fn: Callable[[Sequence[int], int, str], Mapping[str, float]],
    privacy: Mapping[str, Any],
) -> Tuple[_RiskThresholdSelection, Dict[str, Any]]:
    risk_cfg = support_cfg.risk_constrained
    units = _source_inner_units_for_outer(
        outer_target=int(outer_target),
        support_cfg=support_cfg,
        embeddings=embeddings,
        sample_domains=sample_domains,
        labels_by_index=labels_by_index,
        centroids=centroids,
        nelbo_matrix=nelbo_matrix,
        expert_domains_int=expert_domains_int,
        expert_to_col=expert_to_col,
        strategy=strategy,
        tau=float(tau),
        response_feature_fn=response_feature_fn,
    )
    if not units:
        selection = _RiskThresholdSelection(
            tau_margin=float("inf"),
            tau_regret=0.0,
            num_source_inner_units=0,
            source_inner_top1=0.0,
            source_inner_spearman=0.0,
            source_inner_gap_pct=0.0,
            source_inner_harmful_override_rate=0.0,
            fallback_used=True,
        )
        return selection, _selected_threshold_artifact_row(
            seed=seed,
            outer_center=int(outer_target),
            selection=selection,
            privacy=privacy,
        )

    fallback = _RiskThresholdSelection(
        tau_margin=float("inf"),
        tau_regret=0.0,
        num_source_inner_units=len(units),
        source_inner_top1=0.0,
        source_inner_spearman=0.0,
        source_inner_gap_pct=0.0,
        source_inner_harmful_override_rate=0.0,
        fallback_used=True,
    )
    metadata_rows, unrestricted_rows, fallback_rows = _score_source_inner_units(
        units=units,
        seed=int(seed),
        privacy=privacy,
        selection=fallback,
        risk_cfg=risk_cfg,
        primary_feature_regime=str(support_cfg.primary_feature_regime),
    )
    metadata_metrics = _aggregate_metrics_from_sample_rows(metadata_rows)["support_metadata_routing"]
    unrestricted_metrics = _aggregate_metrics_from_sample_rows(unrestricted_rows)[
        f"support_response_pairwise_{support_cfg.primary_feature_regime}"
    ]
    unrestricted_harmful = _harmful_override_rate(
        [
            _risk_extra_fields(
                decision=_risk_policy_decision(
                    candidate_experts=unit["candidate_experts"],
                    learned_scores=unit["learned_scores"],
                    metadata_scores=unit["metadata_scores"],
                    support_mean_nelbo=unit["support_mean_nelbo"],
                    eval_mean_nelbo=unit["eval_mean_nelbo"],
                    tau_margin=float("-inf"),
                    tau_regret=float("inf"),
                ),
                tau_margin=float("-inf"),
                tau_regret=float("inf"),
                selection=fallback,
                focus_query_domain=int(risk_cfg.focus_query_domain),
                focus_expert=int(risk_cfg.focus_expert),
                query_domain=int(unit["validation_domain"]),
            )
            for unit in units
        ]
    )

    candidates: List[Tuple[Tuple[float, float, float, float, float, float, float], _RiskThresholdSelection]] = []
    for tau_margin in risk_cfg.margin_thresholds:
        for tau_regret in risk_cfg.support_regret_thresholds:
            current = _RiskThresholdSelection(
                tau_margin=float(tau_margin),
                tau_regret=float(tau_regret),
                num_source_inner_units=len(units),
                source_inner_top1=0.0,
                source_inner_spearman=0.0,
                source_inner_gap_pct=0.0,
                source_inner_harmful_override_rate=0.0,
                fallback_used=False,
            )
            _meta, _unrestricted, risk_rows = _score_source_inner_units(
                units=units,
                seed=int(seed),
                privacy=privacy,
                selection=current,
                risk_cfg=risk_cfg,
                primary_feature_regime=str(support_cfg.primary_feature_regime),
            )
            risk_metrics = _aggregate_metrics_from_sample_rows(risk_rows)[RISK_CONSTRAINED_METHOD]
            harmful_rate = _harmful_override_rate(risk_rows)
            override_rate = float(
                np.mean([float(r.get("accepted_override", 0.0)) for r in risk_rows])
            ) if risk_rows else 0.0
            eligible = (
                float(risk_metrics["mean_oracle_gap_pct"]) <= float(metadata_metrics["mean_oracle_gap_pct"])
                and harmful_rate <= float(unrestricted_harmful)
                and float(risk_metrics["top1_oracle_hit"])
                >= float(metadata_metrics["top1_oracle_hit"]) - float(risk_cfg.top1_tolerance)
                and float(risk_metrics["spearman"])
                >= float(metadata_metrics["spearman"]) - float(risk_cfg.spearman_tolerance)
            )
            if not eligible:
                continue
            selected = _RiskThresholdSelection(
                tau_margin=float(tau_margin),
                tau_regret=float(tau_regret),
                num_source_inner_units=len(units),
                source_inner_top1=float(risk_metrics["top1_oracle_hit"]),
                source_inner_spearman=float(risk_metrics["spearman"]),
                source_inner_gap_pct=float(risk_metrics["mean_oracle_gap_pct"]),
                source_inner_harmful_override_rate=float(harmful_rate),
                fallback_used=False,
            )
            score = (
                float(metadata_metrics["mean_oracle_gap_pct"]) - float(risk_metrics["mean_oracle_gap_pct"]),
                float(risk_metrics["top1_oracle_hit"]) - float(metadata_metrics["top1_oracle_hit"]),
                float(risk_metrics["spearman"]) - float(metadata_metrics["spearman"]),
                -float(harmful_rate),
                -float(override_rate),
                float(tau_margin),
                -float(tau_regret),
            )
            candidates.append((score, selected))

    if candidates:
        selection = sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]
    else:
        fallback_metrics = _aggregate_metrics_from_sample_rows(fallback_rows)[RISK_CONSTRAINED_METHOD]
        selection = _RiskThresholdSelection(
            tau_margin=float("inf"),
            tau_regret=0.0,
            num_source_inner_units=len(units),
            source_inner_top1=float(fallback_metrics["top1_oracle_hit"]),
            source_inner_spearman=float(fallback_metrics["spearman"]),
            source_inner_gap_pct=float(fallback_metrics["mean_oracle_gap_pct"]),
            source_inner_harmful_override_rate=0.0,
            fallback_used=True,
        )

    _ = unrestricted_metrics
    return selection, _selected_threshold_artifact_row(
        seed=seed,
        outer_center=int(outer_target),
        selection=selection,
        privacy=privacy,
    )


def _selected_threshold_artifact_row(
    *,
    seed: int,
    outer_center: int,
    selection: _RiskThresholdSelection,
    privacy: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        **dict(privacy),
        "method": RISK_CONSTRAINED_METHOD,
        "policy_name": RISK_CONSTRAINED_POLICY_NAME,
        "seed": int(seed),
        "outer_center": int(outer_center),
        "tau_margin": "inf" if np.isinf(float(selection.tau_margin)) else float(selection.tau_margin),
        "tau_regret": "inf" if np.isinf(float(selection.tau_regret)) else float(selection.tau_regret),
        "selection_source": "source_inner_only",
        "num_source_inner_units": int(selection.num_source_inner_units),
        "source_inner_top1": float(selection.source_inner_top1),
        "source_inner_spearman": float(selection.source_inner_spearman),
        "source_inner_gap_pct": float(selection.source_inner_gap_pct),
        "source_inner_harmful_override_rate": float(selection.source_inner_harmful_override_rate),
        "fallback_used": int(bool(selection.fallback_used)),
        "created_before_target_eval_scoring": 1,
    }


def _source_inner_support_units_for_outer(
    *,
    outer_target: int,
    support_cfg: SupportResponseConfig,
    sample_domains: np.ndarray,
    labels_by_index: Mapping[int, int],
    nelbo_matrix: np.ndarray,
    expert_domains_int: Sequence[int],
    expert_to_col: Mapping[int, int],
) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    source_domains = sorted(set(int(v) for v in sample_domains.tolist()) - {int(outer_target)})
    for support_seed in support_cfg.support_seeds:
        for sampling_policy in support_cfg.sampling_policies:
            for support_size in support_cfg.support_sizes:
                for pseudo_query in source_domains:
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
                    if pseudo_split.split_status != "ok":
                        continue
                    if (
                        bool(support_cfg.support_utility.require_unlabeled_support)
                        and int(pseudo_split.support_labels_used) != 0
                    ):
                        raise ProtocolError(
                            "support_utility requires unlabeled support routing, but support split "
                            f"{pseudo_split.support_eval_split_id} used target support labels"
                        )
                    source_fold = FoldCandidateSet.for_heldout_domain(
                        heldout_domain=int(outer_target),
                        expert_domains=expert_domains_int,
                        excluded_domains=[int(pseudo_query)],
                    )
                    support_mean, support_stderr, eval_mean = _support_eval_stats_for_candidates(
                        split=pseudo_split,
                        candidate_experts=source_fold.candidate_expert_domains,
                        nelbo_matrix=nelbo_matrix,
                        expert_to_col=expert_to_col,
                    )
                    units.append(
                        {
                            "outer_target": int(outer_target),
                            "validation_domain": int(pseudo_query),
                            "support_seed": int(support_seed),
                            "support_size": int(support_size),
                            "sampling_policy": str(sampling_policy),
                            "support_eval_split_id": str(pseudo_split.support_eval_split_id),
                            "candidate_experts": list(source_fold.candidate_expert_domains),
                            "eval_mean_nelbo": eval_mean.tolist(),
                            "support_mean_nelbo": support_mean.tolist(),
                            "support_stderr_nelbo": support_stderr.tolist(),
                            "support_n": int(pseudo_split.support_size_actual),
                            "fold": source_fold,
                        }
                    )
    return units


def _score_support_alpha_units(
    *,
    units: Sequence[Mapping[str, Any]],
    alpha: float,
    seed: int,
    privacy: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, unit in enumerate(units):
        support_mean = list(unit["support_mean_nelbo"])
        support_stderr = list(unit["support_stderr_nelbo"])
        scores = _conservative_support_scores(support_mean, support_stderr, alpha=float(alpha))
        rows.append(
            _score_method_row(
                method=SUPPORT_CONSERVATIVE_METHOD,
                fold=unit["fold"],
                target_domain=int(unit["validation_domain"]),
                support_seed=int(unit["support_seed"]),
                support_size=int(unit["support_size"]),
                sampling_policy=str(unit["sampling_policy"]),
                support_eval_split_id=str(unit["support_eval_split_id"]),
                candidate_experts=list(unit["candidate_experts"]),
                predicted_scores=scores,
                eval_mean_nelbo=list(unit["eval_mean_nelbo"]),
                support_mean_nelbo=support_mean,
                support_stderr_nelbo=support_stderr,
                sample_index=int(idx),
                run_seed=int(seed),
                privacy_fields=privacy,
                alpha=float(alpha),
                support_n=int(unit.get("support_n", 0)),
                support_labels_used_for_routing=0,
                conservative_scores=scores,
                extra_fields={
                    "selection_source": "source_inner_only",
                    "alpha_selection_policy": SUPPORT_ALPHA_SELECTION_POLICY,
                },
            )
        )
    return rows


def _select_support_alpha(
    *,
    outer_target: int,
    support_size: int,
    sampling_policy: str,
    units: Sequence[Mapping[str, Any]],
    support_cfg: SupportResponseConfig,
    seed: int,
    privacy: Mapping[str, Any],
) -> Tuple[_SupportAlphaSelection, Dict[str, Any]]:
    alpha_grid = tuple(float(v) for v in support_cfg.support_utility.alpha_grid)
    filtered_units = [
        unit
        for unit in units
        if int(unit.get("support_size", -1)) == int(support_size)
        and str(unit.get("sampling_policy", "")) == str(sampling_policy)
    ]
    if not filtered_units:
        selection = _SupportAlphaSelection(
            selected_alpha=0.0,
            num_source_inner_units=0,
            source_inner_gap_pct_alpha0=0.0,
            source_inner_gap_pct_selected=0.0,
            source_inner_top1_alpha0=0.0,
            source_inner_top1_selected=0.0,
            source_inner_spearman_alpha0=0.0,
            source_inner_spearman_selected=0.0,
            source_inner_gap_variance_alpha0=0.0,
            source_inner_gap_variance_selected=0.0,
            fallback_to_alpha0=True,
            n_aggregation_units=0,
            top1_tolerance_abs=0.0,
        )
        return selection, _selected_alpha_artifact_row(
            seed=seed,
            outer_center=int(outer_target),
            support_size=int(support_size),
            alpha_grid=alpha_grid,
            selection=selection,
            privacy=privacy,
        )

    alpha0_rows = _score_support_alpha_units(
        units=filtered_units,
        alpha=0.0,
        seed=int(seed),
        privacy=privacy,
    )
    alpha0_metrics = _aggregate_metrics_from_sample_rows(alpha0_rows)[SUPPORT_CONSERVATIVE_METHOD]
    alpha0_gap_var = _gap_variance(alpha0_rows)
    n_units = int(len(alpha0_rows))
    top1_tolerance_abs = float(1.0 / max(n_units, 1))

    candidates: List[Tuple[Tuple[float, float, float], _SupportAlphaSelection]] = []
    for alpha in alpha_grid:
        rows = _score_support_alpha_units(
            units=filtered_units,
            alpha=float(alpha),
            seed=int(seed),
            privacy=privacy,
        )
        metrics = _aggregate_metrics_from_sample_rows(rows)[SUPPORT_CONSERVATIVE_METHOD]
        gap_var = _gap_variance(rows)
        top1_ok = float(metrics["top1_oracle_hit"]) >= (
            float(alpha0_metrics["top1_oracle_hit"]) - top1_tolerance_abs
        )
        spearman_ok = float(metrics["spearman"]) >= (
            float(alpha0_metrics["spearman"]) - SUPPORT_ALPHA_SPEARMAN_TOLERANCE
        )
        if not (top1_ok and spearman_ok):
            continue
        selection = _SupportAlphaSelection(
            selected_alpha=float(alpha),
            num_source_inner_units=n_units,
            source_inner_gap_pct_alpha0=float(alpha0_metrics["mean_oracle_gap_pct"]),
            source_inner_gap_pct_selected=float(metrics["mean_oracle_gap_pct"]),
            source_inner_top1_alpha0=float(alpha0_metrics["top1_oracle_hit"]),
            source_inner_top1_selected=float(metrics["top1_oracle_hit"]),
            source_inner_spearman_alpha0=float(alpha0_metrics["spearman"]),
            source_inner_spearman_selected=float(metrics["spearman"]),
            source_inner_gap_variance_alpha0=float(alpha0_gap_var),
            source_inner_gap_variance_selected=float(gap_var),
            fallback_to_alpha0=False,
            n_aggregation_units=n_units,
            top1_tolerance_abs=top1_tolerance_abs,
        )
        score = (
            -float(metrics["mean_oracle_gap_pct"]),
            -float(gap_var),
            -float(alpha),
        )
        candidates.append((score, selection))

    if candidates:
        selection = sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]
    else:
        selection = _SupportAlphaSelection(
            selected_alpha=0.0,
            num_source_inner_units=n_units,
            source_inner_gap_pct_alpha0=float(alpha0_metrics["mean_oracle_gap_pct"]),
            source_inner_gap_pct_selected=float(alpha0_metrics["mean_oracle_gap_pct"]),
            source_inner_top1_alpha0=float(alpha0_metrics["top1_oracle_hit"]),
            source_inner_top1_selected=float(alpha0_metrics["top1_oracle_hit"]),
            source_inner_spearman_alpha0=float(alpha0_metrics["spearman"]),
            source_inner_spearman_selected=float(alpha0_metrics["spearman"]),
            source_inner_gap_variance_alpha0=float(alpha0_gap_var),
            source_inner_gap_variance_selected=float(alpha0_gap_var),
            fallback_to_alpha0=True,
            n_aggregation_units=n_units,
            top1_tolerance_abs=top1_tolerance_abs,
        )

    return selection, _selected_alpha_artifact_row(
        seed=seed,
        outer_center=int(outer_target),
        support_size=int(support_size),
        alpha_grid=alpha_grid,
        selection=selection,
        privacy=privacy,
    )


def _selected_alpha_artifact_row(
    *,
    seed: int,
    outer_center: int,
    support_size: int,
    alpha_grid: Sequence[float],
    selection: _SupportAlphaSelection,
    privacy: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        **dict(privacy),
        "method": SUPPORT_CONSERVATIVE_METHOD,
        "seed": int(seed),
        "outer_center": int(outer_center),
        "support_size": int(support_size),
        "alpha_grid": _alpha_grid_label(alpha_grid),
        "alpha_selection_policy": SUPPORT_ALPHA_SELECTION_POLICY,
        "selected_alpha": float(selection.selected_alpha),
        "selection_source": "source_inner_only",
        "selected_before_target_eval_scoring": 1,
        "source_inner_gap_pct_alpha0": float(selection.source_inner_gap_pct_alpha0),
        "source_inner_gap_pct_selected": float(selection.source_inner_gap_pct_selected),
        "source_inner_top1_alpha0": float(selection.source_inner_top1_alpha0),
        "source_inner_top1_selected": float(selection.source_inner_top1_selected),
        "source_inner_spearman_alpha0": float(selection.source_inner_spearman_alpha0),
        "source_inner_spearman_selected": float(selection.source_inner_spearman_selected),
        "source_inner_gap_variance_alpha0": float(selection.source_inner_gap_variance_alpha0),
        "source_inner_gap_variance_selected": float(selection.source_inner_gap_variance_selected),
        "fallback_to_alpha0": int(bool(selection.fallback_to_alpha0)),
        "n_aggregation_units": int(selection.n_aggregation_units),
        "top1_tolerance_abs": float(selection.top1_tolerance_abs),
    }


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
    risk_threshold_rows: List[Dict[str, Any]] = []
    risk_override_rows: List[Dict[str, Any]] = []
    risk_expert4_rows: List[Dict[str, Any]] = []
    support_utility_hyper_rows: List[Dict[str, Any]] = []
    sample_index_counter = 0

    if support_cfg.risk_constrained.enabled:
        reports_dir.mkdir(parents=True, exist_ok=True)

    for outer_target in sorted(set(int(v) for v in sample_domains.tolist())):
        target_indices = [int(i) for i, d in enumerate(sample_domains.tolist()) if int(d) == int(outer_target)]
        target_fold = FoldCandidateSet.for_heldout_domain(
            heldout_domain=int(outer_target),
            expert_domains=expert_domains_int,
        )
        target_candidates = list(target_fold.candidate_expert_domains)
        risk_selection: _RiskThresholdSelection | None = None
        support_alpha_by_key: Dict[Tuple[int, str], _SupportAlphaSelection] = {}
        if support_cfg.support_utility.enabled:
            source_support_units = _source_inner_support_units_for_outer(
                outer_target=int(outer_target),
                support_cfg=support_cfg,
                sample_domains=sample_domains,
                labels_by_index=labels_by_index,
                nelbo_matrix=nelbo_matrix,
                expert_domains_int=expert_domains_int,
                expert_to_col=expert_to_col,
            )
            for sampling_policy in support_cfg.sampling_policies:
                for support_size in support_cfg.support_sizes:
                    selection, alpha_row = _select_support_alpha(
                        outer_target=int(outer_target),
                        support_size=int(support_size),
                        sampling_policy=str(sampling_policy),
                        units=source_support_units,
                        support_cfg=support_cfg,
                        seed=int(seed),
                        privacy=privacy,
                    )
                    support_alpha_by_key[(int(support_size), str(sampling_policy))] = selection
                    support_utility_hyper_rows.append(alpha_row)
        if support_cfg.risk_constrained.enabled:
            risk_selection, threshold_row = _select_risk_threshold_for_outer(
                outer_target=int(outer_target),
                support_cfg=support_cfg,
                embeddings=np.asarray(embeddings, dtype=np.float64),
                sample_domains=sample_domains,
                labels_by_index=labels_by_index,
                centroids=centroids,
                nelbo_matrix=nelbo_matrix,
                expert_domains_int=expert_domains_int,
                expert_to_col=expert_to_col,
                seed=int(seed),
                strategy=strategy,
                tau=float(tau),
                response_feature_fn=response_feature_fn,
                privacy=privacy,
            )
            risk_threshold_rows.append(threshold_row)
            _write_csv(reports_dir / "risk_constrained_selected_thresholds.csv", risk_threshold_rows)

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
                    if (
                        bool(support_cfg.support_utility.enabled)
                        and bool(support_cfg.support_utility.require_unlabeled_support)
                        and int(target_split.support_labels_used) != 0
                    ):
                        raise ProtocolError(
                            "support_utility requires unlabeled support routing, but target support split "
                            f"{target_split.support_eval_split_id} used labels"
                        )

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
                        rows, _support_mean, _support_stderr, eval_mean = _candidate_rows_for_query(
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

                    target_rows, target_support_mean, target_support_stderr, target_eval_mean = _candidate_rows_for_query(
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
                            support_stderr_nelbo=target_support_stderr,
                            sample_index=sample_index_counter,
                            run_seed=int(seed),
                            privacy_fields=privacy,
                            support_n=int(target_split.support_size_actual),
                            support_labels_used_for_routing=0,
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
                            support_stderr_nelbo=target_support_stderr,
                            sample_index=sample_index_counter,
                            run_seed=int(seed),
                            privacy_fields=privacy,
                            support_n=int(target_split.support_size_actual),
                            support_labels_used_for_routing=0,
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
                            support_stderr_nelbo=target_support_stderr,
                            sample_index=sample_index_counter,
                            run_seed=int(seed),
                            privacy_fields=privacy,
                            alpha=0.0,
                            support_n=int(target_split.support_size_actual),
                            support_labels_used_for_routing=0,
                            conservative_scores=target_support_mean,
                        )
                    )
                    sample_index_counter += 1

                    if support_cfg.support_utility.enabled:
                        alpha_selection = support_alpha_by_key[(int(support_size), str(sampling_policy))]
                        conservative_scores = _conservative_support_scores(
                            target_support_mean,
                            target_support_stderr,
                            alpha=float(alpha_selection.selected_alpha),
                        )
                        sample_rows.append(
                            _score_method_row(
                                method=SUPPORT_CONSERVATIVE_METHOD,
                                fold=target_fold,
                                target_domain=int(outer_target),
                                support_seed=int(support_seed),
                                support_size=int(support_size),
                                sampling_policy=str(sampling_policy),
                                support_eval_split_id=target_split.support_eval_split_id,
                                candidate_experts=target_candidates,
                                predicted_scores=conservative_scores,
                                eval_mean_nelbo=target_eval_mean,
                                support_mean_nelbo=target_support_mean,
                                support_stderr_nelbo=target_support_stderr,
                                sample_index=sample_index_counter,
                                run_seed=int(seed),
                                privacy_fields=privacy,
                                alpha=float(alpha_selection.selected_alpha),
                                support_n=int(target_split.support_size_actual),
                                support_labels_used_for_routing=0,
                                conservative_scores=conservative_scores,
                                extra_fields={
                                    "alpha_grid": _alpha_grid_label(support_cfg.support_utility.alpha_grid),
                                    "alpha_selection_policy": SUPPORT_ALPHA_SELECTION_POLICY,
                                    "selection_source": "source_inner_only",
                                    "selected_before_target_eval_scoring": 1,
                                    "source_inner_gap_pct_alpha0": float(
                                        alpha_selection.source_inner_gap_pct_alpha0
                                    ),
                                    "source_inner_gap_pct_selected": float(
                                        alpha_selection.source_inner_gap_pct_selected
                                    ),
                                    "source_inner_top1_alpha0": float(
                                        alpha_selection.source_inner_top1_alpha0
                                    ),
                                    "source_inner_top1_selected": float(
                                        alpha_selection.source_inner_top1_selected
                                    ),
                                    "source_inner_spearman_alpha0": float(
                                        alpha_selection.source_inner_spearman_alpha0
                                    ),
                                    "source_inner_spearman_selected": float(
                                        alpha_selection.source_inner_spearman_selected
                                    ),
                                    "source_inner_gap_variance_alpha0": float(
                                        alpha_selection.source_inner_gap_variance_alpha0
                                    ),
                                    "source_inner_gap_variance_selected": float(
                                        alpha_selection.source_inner_gap_variance_selected
                                    ),
                                    "fallback_to_alpha0": int(bool(alpha_selection.fallback_to_alpha0)),
                                    "n_aggregation_units": int(alpha_selection.n_aggregation_units),
                                    "top1_tolerance_abs": float(alpha_selection.top1_tolerance_abs),
                                },
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
                            support_stderr_nelbo=target_support_stderr,
                            sample_index=sample_index_counter,
                            run_seed=int(seed),
                            privacy_fields=privacy,
                            support_n=int(target_split.support_size_actual),
                            support_labels_used_for_routing=0,
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
                            support_stderr_nelbo=target_support_stderr,
                            sample_index=sample_index_counter,
                            run_seed=int(seed),
                            privacy_fields=privacy,
                            support_n=int(target_split.support_size_actual),
                            support_labels_used_for_routing=0,
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
                                support_stderr_nelbo=target_support_stderr,
                                sample_index=sample_index_counter,
                                run_seed=int(seed),
                                privacy_fields=privacy,
                                support_n=int(target_split.support_size_actual),
                                support_labels_used_for_routing=0,
                            )
                        )
                        sample_index_counter += 1

                        if (
                            support_cfg.risk_constrained.enabled
                            and regime == support_cfg.primary_feature_regime
                            and risk_selection is not None
                        ):
                            risk_row, risk_audit, expert4_audit = _score_risk_constrained_row(
                                fold=target_fold,
                                target_domain=int(outer_target),
                                support_seed=int(support_seed),
                                support_size=int(support_size),
                                sampling_policy=str(sampling_policy),
                                support_eval_split_id=target_split.support_eval_split_id,
                                candidate_experts=target_candidates,
                                learned_scores=predicted,
                                metadata_scores=metadata_scores,
                                eval_mean_nelbo=target_eval_mean,
                                support_mean_nelbo=target_support_mean,
                                support_stderr_nelbo=target_support_stderr,
                                support_n=int(target_split.support_size_actual),
                                sample_index=sample_index_counter,
                                run_seed=int(seed),
                                privacy_fields=privacy,
                                selection=risk_selection,
                                focus_query_domain=int(support_cfg.risk_constrained.focus_query_domain),
                                focus_expert=int(support_cfg.risk_constrained.focus_expert),
                            )
                            sample_rows.append(risk_row)
                            risk_override_rows.append(risk_audit)
                            risk_expert4_rows.append(expert4_audit)
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
                                validation_support_stderr = np.asarray(
                                    [float(r.get("support_stderr_nelbo", 0.0)) for r in validation_rows],
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
                                    support_stderr_nelbo=validation_support_stderr,
                                    sample_index=sample_index_counter,
                                    run_seed=int(seed),
                                    privacy_fields=privacy,
                                    support_n=int(
                                        source_split_by_domain[int(validation_domain)].support_size_actual
                                    ),
                                    support_labels_used_for_routing=0,
                                )
                                diag["outer_target_domain"] = int(outer_target)
                                diag["validation_pseudo_query_domain"] = int(validation_domain)
                                sample_rows.append(diag)
                                sample_index_counter += 1

    method_metrics = _aggregate_metrics_from_sample_rows(sample_rows) if sample_rows else {}
    if support_cfg.risk_constrained.enabled:
        metadata_by_unit = {
            (
                int(row.get("seed", 0)),
                int(row.get("query_domain", 0)),
                int(row.get("support_seed", 0)),
                int(row.get("support_size_requested", 0)),
                str(row.get("sampling_policy", "")),
            ): row
            for row in sample_rows
            if str(row.get("method", "")) == "support_metadata_routing"
        }
        primary_method = f"support_response_pairwise_{support_cfg.primary_feature_regime}"
        primary_rows = [row for row in sample_rows if str(row.get("method", "")) == primary_method]
        primary_override_count = 0
        primary_harmful_count = 0
        primary_improving_count = 0
        for row in primary_rows:
            key = (
                int(row.get("seed", 0)),
                int(row.get("query_domain", 0)),
                int(row.get("support_seed", 0)),
                int(row.get("support_size_requested", 0)),
                str(row.get("sampling_policy", "")),
            )
            metadata_row = metadata_by_unit.get(key)
            if metadata_row is None:
                continue
            if int(row.get("selected_expert", -1)) == int(metadata_row.get("selected_expert", -2)):
                continue
            primary_override_count += 1
            delta = float(row.get("selected_nelbo", 0.0)) - float(metadata_row.get("selected_nelbo", 0.0))
            if delta > 1e-12:
                primary_harmful_count += 1
            if delta < -1e-12:
                primary_improving_count += 1
        if primary_method in method_metrics:
            denom = max(len(primary_rows), 1)
            method_metrics[primary_method].update(
                {
                    "override_rate": float(primary_override_count / denom),
                    "harmful_override_rate": float(primary_harmful_count / max(primary_override_count, 1)),
                    "utility_improving_override_rate": float(
                        primary_improving_count / max(primary_override_count, 1)
                    ),
                    "accepted_override_count": float(primary_override_count),
                    "harmful_override_count": float(primary_harmful_count),
                    "utility_improving_override_count": float(primary_improving_count),
                }
            )
    if support_cfg.risk_constrained.enabled and RISK_CONSTRAINED_METHOD in method_metrics:
        risk_rows_for_metrics = [
            row for row in sample_rows if str(row.get("method", "")) == RISK_CONSTRAINED_METHOD
        ]
        accepted_count = sum(int(float(row.get("accepted_override", 0) or 0)) for row in risk_rows_for_metrics)
        override_candidate_count = sum(
            int(float(row.get("override_candidate", 0) or 0)) for row in risk_rows_for_metrics
        )
        harmful_count = sum(
            int(float(row.get("true_harmful_override", 0) or 0)) for row in risk_rows_for_metrics
        )
        improving_count = sum(
            int(float(row.get("true_improving_override", 0) or 0)) for row in risk_rows_for_metrics
        )
        expert4_candidate_count = sum(
            int(float(row.get("focus_expert_override_candidate", 0) or 0)) for row in risk_rows_for_metrics
        )
        expert4_accepted_count = sum(
            int(float(row.get("focus_expert_override_accepted", 0) or 0)) for row in risk_rows_for_metrics
        )
        expert4_blocked_count = sum(
            int(float(row.get("focus_expert_override_blocked", 0) or 0)) for row in risk_rows_for_metrics
        )
        denom = max(len(risk_rows_for_metrics), 1)
        method_metrics[RISK_CONSTRAINED_METHOD].update(
            {
                "override_rate": float(accepted_count / denom),
                "override_candidate_rate": float(override_candidate_count / denom),
                "harmful_override_rate": _harmful_override_rate(risk_rows_for_metrics),
                "utility_improving_override_rate": float(improving_count / max(accepted_count, 1)),
                "expert4_override_candidate_rate": float(expert4_candidate_count / denom),
                "expert4_override_accepted_rate": float(expert4_accepted_count / denom),
                "expert4_override_blocked_rate": float(expert4_blocked_count / max(expert4_candidate_count, 1)),
                "accepted_override_count": float(accepted_count),
                "harmful_override_count": float(harmful_count),
                "utility_improving_override_count": float(improving_count),
            }
        )
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
    if support_cfg.support_utility.enabled:
        artifacts["support_utility_selected_hyperparams"] = "support_utility_selected_hyperparams.csv"
    if support_cfg.risk_constrained.enabled:
        artifacts.update(
            {
                "risk_constrained_selected_thresholds": "risk_constrained_selected_thresholds.csv",
                "risk_constrained_sample_selections": "risk_constrained_sample_selections.csv",
                "risk_constrained_domain_breakdown": "risk_constrained_domain_breakdown.csv",
                "risk_constrained_override_audit": "risk_constrained_override_audit.csv",
                "risk_constrained_expert4_audit": "risk_constrained_expert4_audit.csv",
            }
        )
    protocol_lock = {
        "protocol_version": SUPPORT_RESPONSE_PROTOCOL_VERSION,
        "score_direction": "predicted_score_is_predicted_mean_nelbo_lower_is_better",
        "candidate_label_definition": "mean_NELBO(pseudo_query_eval, candidate_expert)",
        "pairwise_comparison_scope": "within_same_pseudo_query_domain",
        "ranker": support_cfg.ranker,
        "ridge_l2": float(support_cfg.ridge_l2),
        "scaler_fit_scope": "source_training_pairs_only",
        "domain_level_aggregation": bool(support_cfg.domain_level_aggregation),
        "support_estimated_utility": {
            "enabled": bool(support_cfg.support_utility.enabled),
            "method": SUPPORT_CONSERVATIVE_METHOD,
            "direct_support_baseline_method": "support_set_nelbo_top1",
            "score_definition": "mean_support_nelbo_plus_alpha_times_stderr_support_nelbo",
            "alpha_grid": list(float(v) for v in support_cfg.support_utility.alpha_grid),
            "alpha_selection_policy": str(support_cfg.support_utility.alpha_selection_policy),
            "alpha_selection_scope": "source_inner_only_per_outer_center_x_support_size",
            "selected_before_target_eval_scoring": 1,
            "support_labels_used_for_routing": 0,
            "require_unlabeled_support": bool(support_cfg.support_utility.require_unlabeled_support),
            "high_regret_gap_pct_threshold": float(HIGH_REGRET_GAP_PCT_THRESHOLD),
            "bottom_half_rank_threshold": int(BOTTOM_HALF_RANK_THRESHOLD),
        },
        "risk_constrained_response_routing": {
            "enabled": bool(support_cfg.risk_constrained.enabled),
            "method": RISK_CONSTRAINED_METHOD,
            "policy_name": RISK_CONSTRAINED_POLICY_NAME,
            "threshold_selection_policy": "source_inner_only",
            "margin_thresholds": list(float(v) for v in support_cfg.risk_constrained.margin_thresholds),
            "support_regret_thresholds": list(
                float(v) for v in support_cfg.risk_constrained.support_regret_thresholds
            ),
            "top1_tolerance": float(support_cfg.risk_constrained.top1_tolerance),
            "spearman_tolerance": float(support_cfg.risk_constrained.spearman_tolerance),
            "target_evaluation_labels_scope": (
                "final_heldout_utility_scorer_only_if_required_for_conditional_nelbo"
            ),
        },
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
        "n_support_utility_hyperparameter_rows": int(len(support_utility_hyper_rows)),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / artifacts["protocol_lock"]).write_text(json.dumps(protocol_lock, indent=2) + "\n", encoding="utf-8")
    _write_csv(reports_dir / artifacts["split_manifest"], split_rows)
    _write_csv(reports_dir / artifacts["feature_audit"], feature_audit_rows)
    _write_csv(reports_dir / artifacts["pair_predictions"], pair_rows)
    _write_csv(reports_dir / artifacts["sample_selections"], sample_rows)
    _write_csv(reports_dir / artifacts["domain_breakdown"], domain_rows)
    _write_csv(reports_dir / artifacts["method_summary"], method_summary)
    if support_cfg.support_utility.enabled:
        _write_csv(reports_dir / artifacts["support_utility_selected_hyperparams"], support_utility_hyper_rows)
    if support_cfg.risk_constrained.enabled:
        risk_sample_rows = [row for row in sample_rows if str(row.get("method", "")) == RISK_CONSTRAINED_METHOD]
        risk_domain_rows = [
            {**dict(privacy), **row}
            for row in (_domain_breakdown_rows(risk_sample_rows) if risk_sample_rows else [])
        ]
        _write_csv(reports_dir / artifacts["risk_constrained_selected_thresholds"], risk_threshold_rows)
        _write_csv(reports_dir / artifacts["risk_constrained_sample_selections"], risk_sample_rows)
        _write_csv(reports_dir / artifacts["risk_constrained_domain_breakdown"], risk_domain_rows)
        _write_csv(reports_dir / artifacts["risk_constrained_override_audit"], risk_override_rows)
        _write_csv(reports_dir / artifacts["risk_constrained_expert4_audit"], risk_expert4_rows)
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
