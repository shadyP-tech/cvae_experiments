from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.eval.evaluators.learned_utility_protocol import (
    FoldCandidateSet,
    MethodProtocol,
    _protocol_row_fields,
)
from src.eval.evaluators.learned_utility_proxies import (
    _build_random_rank_floor_proxy,
    _build_random_score_floor_proxy,
    _metadata_scores,
    _normalize_scores_per_query,
    _permute_expert_labels_proxy,
    _proxy_diagnostic_rows,
)
from src.eval.evaluators.learned_utility_selection import _selection_metrics


@dataclass(frozen=True)
class ProxyFoldOutputs:
    sample_rows: List[Dict[str, Any]]
    proxy_diag_rows: List[Dict[str, Any]]
    hybrid_method_meta: Dict[str, Dict[str, Any]]
    permutation_sample_rows: Dict[Tuple[str, int], List[Dict[str, Any]]]


def _run_proxy_methods_for_fold(
    *,
    sample_domains: np.ndarray,
    expert_domains: Sequence[int],
    test_idx: np.ndarray,
    fold: FoldCandidateSet,
    true_eval: np.ndarray,
    global_eval: np.ndarray,
    metadata_similarity_eval: np.ndarray,
    latent_similarity_eval: np.ndarray,
    strategy: str,
    tau: float,
    seed: int,
    tie_policy: str,
    enable_random_rank_floor: bool,
    enable_random_score_floor: bool,
    hybrid_enabled: bool,
    norm_policies: Sequence[str],
    hybrid_alphas: Sequence[float],
    primary_norm_policy: str,
    permutation_repeats: int,
    run_expert_label_permutation: bool,
    run_metadata_permutation: bool,
) -> ProxyFoldOutputs:
    sample_rows: List[Dict[str, Any]] = []
    proxy_diag_rows: List[Dict[str, Any]] = []
    hybrid_method_meta: Dict[str, Dict[str, Any]] = {}
    permutation_sample_rows: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    heldout_domain = int(fold.heldout_domain)

    for diag_method, diag_scores in [
        ("metadata_similarity_raw", metadata_similarity_eval),
        ("latent_similarity_raw", latent_similarity_eval),
    ]:
        diag_protocol = MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1 if diag_method.startswith("metadata") else 0,
            routing_uses_eval_domain_statistics=0 if diag_method.startswith("metadata") else 1,
        )
        for row in _proxy_diagnostic_rows(diag_scores, sample_domains[test_idx], method=diag_method):
            row.update(_protocol_row_fields(fold=fold, method_protocol=diag_protocol, method=diag_method))
            proxy_diag_rows.append(row)

    proxy_methods: List[Tuple[str, np.ndarray]] = [
        ("metadata_routing", -metadata_similarity_eval),
        ("latent_wasserstein_routing", -latent_similarity_eval),
        ("candidate_oracle_routing", true_eval),
    ]
    if enable_random_rank_floor:
        proxy_methods.append(
            (
                "random_rank_floor",
                _build_random_rank_floor_proxy(
                    sample_domains=sample_domains[test_idx],
                    n_experts=len(fold.candidate_expert_domains),
                    seed=int(seed) + 131 + heldout_domain,
                ),
            )
        )
    if enable_random_score_floor:
        proxy_methods.append(
            (
                "random_score_floor",
                _build_random_score_floor_proxy(
                    n_samples=int(test_idx.shape[0]),
                    n_experts=len(fold.candidate_expert_domains),
                    seed=int(seed) + 241 + heldout_domain,
                ),
            )
        )

    if hybrid_enabled:
        for norm_policy in norm_policies:
            metadata_norm = _normalize_scores_per_query(metadata_similarity_eval, policy=norm_policy)
            latent_norm = _normalize_scores_per_query(latent_similarity_eval, policy=norm_policy)
            for diag_method, diag_scores in [
                (f"metadata_similarity_{norm_policy}", metadata_norm),
                (f"latent_similarity_{norm_policy}", latent_norm),
            ]:
                diag_protocol = MethodProtocol(
                    method_role="diagnostic",
                    adoption_eligible=0,
                    diagnostic_only=1,
                    routing_uses_query_features=1 if diag_method.startswith("metadata") else 0,
                    routing_uses_eval_domain_statistics=0 if diag_method.startswith("metadata") else 1,
                )
                for row in _proxy_diagnostic_rows(diag_scores, sample_domains[test_idx], method=diag_method):
                    row.update(_protocol_row_fields(fold=fold, method_protocol=diag_protocol, method=diag_method))
                    proxy_diag_rows.append(row)

            for alpha in hybrid_alphas:
                mixed_similarity = (float(alpha) * metadata_norm) + ((1.0 - float(alpha)) * latent_norm)
                method_name = f"hybrid_alpha_{alpha:.1f}"
                if norm_policy != primary_norm_policy:
                    method_name = f"{method_name}_{norm_policy.replace('per_query_', '')}"
                proxy_methods.append((method_name, -mixed_similarity))
                hybrid_method_meta[method_name] = {
                    "alpha": float(alpha),
                    "normalization_policy": str(norm_policy),
                }

            if not np.allclose(metadata_norm, (1.0 * metadata_norm) + (0.0 * latent_norm), atol=1e-12, rtol=1e-9):
                raise RuntimeError("Hybrid endpoint invariant failed for alpha=1.0")
            if not np.allclose(latent_norm, (0.0 * metadata_norm) + (1.0 * latent_norm), atol=1e-12, rtol=1e-9):
                raise RuntimeError("Hybrid endpoint invariant failed for alpha=0.0")

    for name, proxy in proxy_methods:
        _metrics_unused, rows = _selection_metrics(
            method=name,
            query_domains=sample_domains[test_idx],
            expert_domains=fold.candidate_expert_domains,
            score_matrix=proxy,
            true_nelbo_matrix=true_eval,
            fold=fold,
            global_true_nelbo_matrix=global_eval,
            global_expert_domains=expert_domains,
            tie_policy=tie_policy,
        )
        for row in rows:
            row["sample_index"] = int(test_idx[int(row["sample_index"])])
            sample_rows.append(row)

    if int(permutation_repeats) > 0 and (run_expert_label_permutation or run_metadata_permutation):
        metadata_proxy_eval = -metadata_similarity_eval
        for rep in range(int(permutation_repeats)):
            if run_expert_label_permutation:
                perm_proxy = _permute_expert_labels_proxy(
                    metadata_proxy_eval,
                    seed=int(seed) + 10000 + int(rep) + heldout_domain,
                )
                _metrics_unused, rows = _selection_metrics(
                    method="expert_label_permutation",
                    query_domains=sample_domains[test_idx],
                    expert_domains=fold.candidate_expert_domains,
                    score_matrix=perm_proxy,
                    true_nelbo_matrix=true_eval,
                    fold=fold,
                    global_true_nelbo_matrix=global_eval,
                    global_expert_domains=expert_domains,
                    tie_policy=tie_policy,
                )
                for row in rows:
                    row["sample_index"] = int(test_idx[int(row["sample_index"])])
                permutation_sample_rows.setdefault(("expert_label_permutation", int(rep)), []).extend(rows)

            if run_metadata_permutation:
                rng = np.random.default_rng(int(seed) + 20000 + int(rep) + heldout_domain)
                shuffled_domains = np.asarray(rng.permutation(sample_domains[test_idx]), dtype=np.int64)
                shuffled_similarity = _metadata_scores(
                    shuffled_domains,
                    fold.candidate_expert_domains,
                    strategy=strategy,
                    tau=float(tau),
                )
                shuffled_proxy = -shuffled_similarity
                _metrics_unused, rows = _selection_metrics(
                    method="metadata_permutation",
                    query_domains=sample_domains[test_idx],
                    expert_domains=fold.candidate_expert_domains,
                    score_matrix=shuffled_proxy,
                    true_nelbo_matrix=true_eval,
                    fold=fold,
                    global_true_nelbo_matrix=global_eval,
                    global_expert_domains=expert_domains,
                    tie_policy=tie_policy,
                )
                for row in rows:
                    row["sample_index"] = int(test_idx[int(row["sample_index"])])
                permutation_sample_rows.setdefault(("metadata_permutation", int(rep)), []).extend(rows)

    return ProxyFoldOutputs(
        sample_rows=sample_rows,
        proxy_diag_rows=proxy_diag_rows,
        hybrid_method_meta=hybrid_method_meta,
        permutation_sample_rows=permutation_sample_rows,
    )
