from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.app.determinism import stable_response_seed


ROUTING_TIME_AVAILABLE_DEFINITION = (
    "Computable before expert selection without using target expert NELBO, oracle rank, "
    "selected NELBO, target-domain aggregate utility, or post-selection label-derived "
    "evaluation quantities."
)


BLOCKED_EXACT = {
    "query_id",
    "expert_id",
    "domain_id",
    "source_domain",
    "target_domain",
    "oracle_utility",
    "site",
    "scanner",
    "hospital",
    "magnification",
}
BLOCKED_PREFIX = {
    "oracle_",
    "target_",
    "embedding_",
    "metadata_",
}
BLOCKED_SUBSTRINGS = {
    "nelbo",
    "recon_mean",
    "kl_mean",
}

ALLOWED_RESPONSE_FEATURE_PREFIXES = (
    "response_posterior_",
    "response_decode_",
    "response_recon_",
    "response_kl_repeat_variance_",
    "response_residual_",
)


@dataclass(frozen=True)
class FeatureRegime:
    name: str
    adoption_eligible: bool
    diagnostic_only: bool
    control_only: bool
    include_static_metadata: bool = False
    include_static_embedding: bool = False
    include_response_indirect: bool = False
    include_target_adjacent: bool = False
    include_oracle: bool = False


@dataclass(frozen=True)
class FeatureMatrixResult:
    matrix: np.ndarray
    feature_names: List[str]
    included_features: List[str]
    dropped_zero_variance: List[str]
    blocked_features: List[str]
    missing_features: List[str]
    blocked_feature_terms: List[str]
    feature_schema_hash: str
    regime: str
    no_data_reason: str | None


FEATURE_REGISTRY: Dict[str, FeatureRegime] = {
    "static_metadata": FeatureRegime(
        name="static_metadata",
        adoption_eligible=True,
        diagnostic_only=False,
        control_only=False,
        include_static_metadata=True,
    ),
    "static_embedding": FeatureRegime(
        name="static_embedding",
        adoption_eligible=True,
        diagnostic_only=False,
        control_only=False,
        include_static_embedding=True,
    ),
    "static_combined": FeatureRegime(
        name="static_combined",
        adoption_eligible=True,
        diagnostic_only=False,
        control_only=False,
        include_static_metadata=True,
        include_static_embedding=True,
    ),
    "response_indirect": FeatureRegime(
        name="response_indirect",
        adoption_eligible=True,
        diagnostic_only=False,
        control_only=False,
        include_response_indirect=True,
    ),
    "static_response_indirect": FeatureRegime(
        name="static_response_indirect",
        adoption_eligible=True,
        diagnostic_only=False,
        control_only=False,
        include_static_metadata=True,
        include_static_embedding=True,
        include_response_indirect=True,
    ),
    "response_indirect_shuffled": FeatureRegime(
        name="response_indirect_shuffled",
        adoption_eligible=False,
        diagnostic_only=False,
        control_only=True,
        include_response_indirect=True,
    ),
    "response_target_adjacent_diagnostic": FeatureRegime(
        name="response_target_adjacent_diagnostic",
        adoption_eligible=False,
        diagnostic_only=True,
        control_only=False,
        include_target_adjacent=True,
    ),
    "response_oracle_diagnostic": FeatureRegime(
        name="response_oracle_diagnostic",
        adoption_eligible=False,
        diagnostic_only=True,
        control_only=False,
        include_oracle=True,
    ),
}


LEGACY_REGIME_ALIASES = {
    "metadata_only": "static_metadata",
    "static_a": "static_metadata",
    "static_b": "static_combined",
}


STATIC_METADATA_FEATURES = [
    "metadata_distance",
    "query_domain_value",
    "expert_domain_value",
    "abs_domain_diff",
    "is_exact_domain_match",
]
STATIC_EMBEDDING_FEATURES = [
    "embedding_distance",
]
TARGET_ADJACENT_FEATURES = [
    "query_nelbo_mean",
    "query_nelbo_std",
    "query_nelbo_p90",
    "query_recon_mean",
    "query_recon_std",
    "query_kl_mean",
    "query_kl_std",
    "expert_support_nelbo_mean",
    "expert_support_nelbo_std",
    "expert_support_nelbo_p90",
    "expert_support_nelbo_uncertainty_mean",
    "expert_support_nelbo_uncertainty_std",
]
ORACLE_DIAGNOSTIC_FEATURES = [
    "oracle_utility",
    "oracle_utility_std",
    "oracle_nelbo",
    "oracle_nelbo_std",
    "expert_eval_nelbo_uncertainty_mean",
]


def normalize_regime_name(name: str) -> str:
    raw = str(name).strip().lower()
    return LEGACY_REGIME_ALIASES.get(raw, raw)


def get_feature_regime(name: str) -> FeatureRegime:
    normalized = normalize_regime_name(name)
    if normalized not in FEATURE_REGISTRY:
        raise ValueError(
            f"Unknown feature regime '{name}'. Expected one of {sorted(FEATURE_REGISTRY)}."
        )
    return FEATURE_REGISTRY[normalized]


def feature_schema_hash(regime_name: str, feature_names: Sequence[str]) -> str:
    payload = {
        "regime": str(regime_name),
        "feature_names": [str(f) for f in feature_names],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def blocked_terms_for_feature(feature_name: str) -> List[str]:
    name = str(feature_name)
    terms: List[str] = []
    if name in BLOCKED_EXACT:
        terms.append(name)
    for prefix in sorted(BLOCKED_PREFIX):
        if name.startswith(prefix):
            terms.append(prefix)
    for term in sorted(BLOCKED_SUBSTRINGS):
        if term in name:
            terms.append(term)
    return terms


def is_blocked_feature(feature_name: str) -> bool:
    return bool(blocked_terms_for_feature(feature_name))


def response_feature_names(rows: Sequence[dict]) -> List[str]:
    names: set[str] = set()
    for row in rows:
        for key in row.keys():
            name = str(key)
            if name.startswith(ALLOWED_RESPONSE_FEATURE_PREFIXES) or blocked_terms_for_feature(name):
                names.add(name)
    return sorted(names)


def _candidate_features(
    rows: Sequence[dict],
    regime: FeatureRegime,
    expert_domains: Sequence[int],
) -> List[str]:
    features: List[str] = []
    if regime.include_static_metadata:
        features.extend(STATIC_METADATA_FEATURES)
        features.extend([f"expert_onehot_{int(d)}" for d in expert_domains])
    if regime.include_static_embedding:
        features.extend(STATIC_EMBEDDING_FEATURES)
    if regime.include_response_indirect:
        features.extend(response_feature_names(rows))
    if regime.include_target_adjacent:
        features.extend(TARGET_ADJACENT_FEATURES)
    if regime.include_oracle:
        features.extend(ORACLE_DIAGNOSTIC_FEATURES)
    return features


def _feature_value(row: dict, feature_name: str) -> Tuple[float, bool]:
    if feature_name.startswith("expert_onehot_"):
        expert = int(str(feature_name).replace("expert_onehot_", ""))
        return (1.0 if int(row.get("expert_domain", -1)) == expert else 0.0), True
    if feature_name not in row:
        return 0.0, False
    value = row.get(feature_name, 0.0)
    try:
        value_f = float(value)
    except Exception:
        value_f = 0.0
    if not np.isfinite(value_f):
        value_f = 0.0
    return value_f, True


def build_feature_matrix(
    rows: Sequence[dict],
    *,
    regime: FeatureRegime,
    expert_domains: Sequence[int],
    feature_names: Sequence[str] | None = None,
    drop_zero_variance: bool = True,
    zero_variance_eps: float = 1e-12,
) -> FeatureMatrixResult:
    if not rows:
        names = list(feature_names or [])
        return FeatureMatrixResult(
            matrix=np.empty((0, len(names)), dtype=np.float64),
            feature_names=names,
            included_features=[],
            dropped_zero_variance=[],
            blocked_features=[],
            missing_features=[],
            blocked_feature_terms=[],
            feature_schema_hash=feature_schema_hash(regime.name, names),
            regime=regime.name,
            no_data_reason="no_rows",
        )

    candidates = list(feature_names) if feature_names is not None else _candidate_features(rows, regime, expert_domains)
    blocked_features: List[str] = []
    blocked_terms: List[str] = []
    allowed_candidates: List[str] = []

    for feature in candidates:
        terms = blocked_terms_for_feature(feature)
        should_enforce_response_isolation = regime.name in {
            "response_indirect",
            "response_indirect_shuffled",
        }
        should_enforce_response_term_blocks = should_enforce_response_isolation or (
            regime.name == "static_response_indirect" and feature.startswith("response_")
        )
        if should_enforce_response_isolation and not feature.startswith("response_"):
            terms.append("non_response_feature")
        if terms and should_enforce_response_term_blocks and regime.adoption_eligible:
            blocked_features.append(feature)
            blocked_terms.extend(terms)
            continue
        if terms and should_enforce_response_term_blocks and regime.control_only:
            blocked_features.append(feature)
            blocked_terms.extend(terms)
            continue
        allowed_candidates.append(feature)

    matrix_cols: List[List[float]] = []
    missing: List[str] = []
    for feature in allowed_candidates:
        values: List[float] = []
        present_any = False
        for row in rows:
            value, present = _feature_value(row, feature)
            present_any = present_any or present
            values.append(value)
        if not present_any:
            missing.append(feature)
            continue
        matrix_cols.append(values)

    if matrix_cols:
        matrix = np.asarray(matrix_cols, dtype=np.float64).T
        names = [f for f in allowed_candidates if f not in set(missing)]
    else:
        matrix = np.empty((len(rows), 0), dtype=np.float64)
        names = []

    dropped: List[str] = []
    if drop_zero_variance and matrix.shape[1] > 0:
        keep_idxs: List[int] = []
        for idx, feature in enumerate(names):
            var = float(np.var(matrix[:, idx]))
            if var <= float(zero_variance_eps):
                dropped.append(feature)
            else:
                keep_idxs.append(idx)
        matrix = matrix[:, keep_idxs] if keep_idxs else np.empty((len(rows), 0), dtype=np.float64)
        names = [names[i] for i in keep_idxs]

    no_data_reason = None if matrix.shape[1] > 0 else "no_features_after_audit"
    return FeatureMatrixResult(
        matrix=matrix,
        feature_names=names,
        included_features=list(names),
        dropped_zero_variance=dropped,
        blocked_features=blocked_features,
        missing_features=missing,
        blocked_feature_terms=sorted(set(blocked_terms)),
        feature_schema_hash=feature_schema_hash(regime.name, names),
        regime=regime.name,
        no_data_reason=no_data_reason,
    )


def shuffle_response_feature_rows(
    rows: Sequence[dict],
    *,
    dataset: str,
    seed: int,
    fold_id: str,
    split_id: str,
    regime_name: str,
    stream_name: str = "response_feature_shuffle",
) -> List[dict]:
    out = [dict(r) for r in rows]
    if len(out) <= 1:
        return out
    response_names = response_feature_names(out)
    if not response_names:
        return out

    shuffle_seed = stable_response_seed(
        dataset=str(dataset),
        seed=int(seed),
        query_id=str(fold_id),
        expert_domain=str(regime_name),
        repeat_id=0,
        stream_name=f"{stream_name}:{split_id}",
    )
    rng = np.random.default_rng(int(shuffle_seed))
    perm = rng.permutation(len(out))
    for dest_idx, src_idx in enumerate(perm.tolist()):
        src = rows[int(src_idx)]
        for name in response_names:
            out[dest_idx][name] = src.get(name, 0.0)
    return out


def serialize_feature_list(values: Iterable[str]) -> str:
    return "|".join(str(v) for v in values)
