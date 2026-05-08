from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.eval.evaluators.learned_utility_protocol import FoldCandidateSet
from src.eval.evaluators.learned_utility_config import _parse_learned_utility_config
from src.eval.evaluators.learned_utility_learned_methods import _run_learned_methods_for_fold
from src.eval.evaluators.learned_utility_proxy_methods import _run_proxy_methods_for_fold
from src.eval.evaluators.learned_utility_proxies import (
    _latent_wasserstein_scores,
    _metadata_scores,
)
from src.eval.evaluators.learned_utility_scoring import (
    _domain_to_expert_index,
    _score_experts_batched,
)
from src.eval.evaluators.learned_utility_reporting import (
    _finalize_learned_utility_outputs,
)


def evaluate_learned_utility_loqdo(
    *,
    test_cache: Path,
    expert_checkpoints: Dict[str, str],
    hidden_dim: int,
    latent_dim: int,
    strategy: str,
    tau: float,
    seed: int,
    learned_cfg: Dict[str, Any],
    reports_dir: Path,
    conditioning_cfg: Dict[str, Any] | None = None,
    configured_domains: Sequence[int] | None = None,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    eval_cfg = _parse_learned_utility_config(learned_cfg)
    hybrid_cfg = eval_cfg.hybrid
    compatibility_cfg = eval_cfg.compatibility

    print("[learned_utility] scoring expert NELBO matrix...")
    embeddings, sample_domains, true_nelbo, expert_domains, metadata = _score_experts_batched(
        test_cache=test_cache,
        expert_checkpoints=expert_checkpoints,
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        pair_batch_size=int(eval_cfg.pair_batch_size),
        conditioning_cfg=conditioning_cfg,
        configured_domains=configured_domains,
        metadata_constraint_cfg=metadata_constraint_cfg,
    )
    _ = metadata
    print(
        f"[learned_utility] scored matrix shape={true_nelbo.shape}, n_samples={sample_domains.shape[0]}, n_experts={len(expert_domains)}"
    )

    domain_to_idx = _domain_to_expert_index(expert_domains)

    metadata_similarity = _metadata_scores(sample_domains, expert_domains, strategy=strategy, tau=float(tau))
    latent_similarity = _latent_wasserstein_scores(
        embeddings=embeddings,
        sample_domains=sample_domains,
        expert_domains=expert_domains,
    )
    if not np.isfinite(metadata_similarity).all() or not np.isfinite(latent_similarity).all():
        raise ValueError("Metadata/latent proxy similarity matrices must be finite")

    sample_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    pair_training_rows: List[Dict[str, Any]] = []
    proxy_diag_rows: List[Dict[str, Any]] = []
    hybrid_method_meta: Dict[str, Dict[str, Any]] = {}
    permutation_sample_rows: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

    unique_query_domains = sorted(set(int(v) for v in sample_domains.tolist()))
    embedding_feature_dim = int(embeddings.shape[1])
    expert_feature_dim = int(len(expert_domains))

    norm_policies: List[str] = []
    hybrid_alphas = list(hybrid_cfg.alphas)
    if hybrid_cfg.enabled:
        norm_policies = [hybrid_cfg.primary_norm_policy]
        if hybrid_cfg.run_sensitivity and hybrid_cfg.sensitivity_norm_policy != hybrid_cfg.primary_norm_policy:
            norm_policies.append(hybrid_cfg.sensitivity_norm_policy)
        hybrid_alphas = sorted(set(float(a) for a in hybrid_alphas))
        for alpha in hybrid_alphas:
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError(f"hybrid alpha must be in [0,1], got {alpha}")

    total_folds = len(unique_query_domains)
    for fold_idx, heldout_domain in enumerate(unique_query_domains, start=1):
        fold_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        fold_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if fold_start is not None:
            fold_start.record()
        print(f"[learned_utility] fold {fold_idx}/{total_folds} heldout_query_domain={heldout_domain}...")
        train_idx = np.where(sample_domains != int(heldout_domain))[0]
        test_idx = np.where(sample_domains == int(heldout_domain))[0]
        if train_idx.size == 0 or test_idx.size == 0:
            continue

        fold = FoldCandidateSet.for_heldout_domain(
            heldout_domain=int(heldout_domain),
            expert_domains=expert_domains,
        )
        true_eval = fold.slice_nelbo(true_nelbo, test_idx)
        global_eval = true_nelbo[np.asarray(test_idx, dtype=np.int64)]
        metadata_similarity_eval = metadata_similarity[np.asarray(test_idx, dtype=np.int64)][:, list(fold.candidate_col_indices)]
        latent_similarity_eval = latent_similarity[np.asarray(test_idx, dtype=np.int64)][:, list(fold.candidate_col_indices)]

        proxy_outputs = _run_proxy_methods_for_fold(
            sample_domains=sample_domains,
            expert_domains=expert_domains,
            test_idx=test_idx,
            fold=fold,
            true_eval=true_eval,
            global_eval=global_eval,
            metadata_similarity_eval=metadata_similarity_eval,
            latent_similarity_eval=latent_similarity_eval,
            strategy=strategy,
            tau=float(tau),
            seed=int(seed),
            tie_policy=hybrid_cfg.tie_policy,
            enable_random_rank_floor=compatibility_cfg.enable_random_rank_floor,
            enable_random_score_floor=compatibility_cfg.enable_random_score_floor,
            hybrid_enabled=hybrid_cfg.enabled,
            norm_policies=norm_policies,
            hybrid_alphas=hybrid_alphas,
            primary_norm_policy=hybrid_cfg.primary_norm_policy,
            permutation_repeats=compatibility_cfg.permutation_repeats,
            run_expert_label_permutation=compatibility_cfg.run_expert_label_permutation,
            run_metadata_permutation=compatibility_cfg.run_metadata_permutation,
        )
        sample_rows.extend(proxy_outputs.sample_rows)
        proxy_diag_rows.extend(proxy_outputs.proxy_diag_rows)
        hybrid_method_meta.update(proxy_outputs.hybrid_method_meta)
        for key, rows in proxy_outputs.permutation_sample_rows.items():
            permutation_sample_rows.setdefault(key, []).extend(rows)

        learned_outputs = _run_learned_methods_for_fold(
            embeddings=embeddings,
            sample_domains=sample_domains,
            true_nelbo=true_nelbo,
            expert_domains=expert_domains,
            domain_to_idx=domain_to_idx,
            train_idx=train_idx,
            test_idx=test_idx,
            fold=fold,
            global_eval=global_eval,
            predictors=eval_cfg.predictors,
            mlp_cfg=eval_cfg.mlp_cfg,
            pairwise_cfg=eval_cfg.pairwise_cfg,
            include_metadata_features=eval_cfg.include_metadata_features,
            seed=int(seed),
            embedding_feature_dim=embedding_feature_dim,
            expert_feature_dim=expert_feature_dim,
            tie_policy=hybrid_cfg.tie_policy,
        )
        sample_rows.extend(learned_outputs.sample_rows)
        pair_rows.extend(learned_outputs.pair_rows)
        pair_training_rows.extend(learned_outputs.pair_training_rows)

        if fold_end is not None:
            fold_end.record()
            torch.cuda.synchronize()
            elapsed_ms = float(fold_start.elapsed_time(fold_end))
            print(f"[learned_utility] fold {fold_idx}/{total_folds} done in {elapsed_ms / 1000.0:.2f}s")
        else:
            print(f"[learned_utility] fold {fold_idx}/{total_folds} done")

    return _finalize_learned_utility_outputs(
        reports_dir=reports_dir,
        sample_rows=sample_rows,
        pair_rows=pair_rows,
        pair_training_rows=pair_training_rows,
        proxy_diag_rows=proxy_diag_rows,
        permutation_sample_rows=permutation_sample_rows,
        hybrid_method_meta=hybrid_method_meta,
        sample_domains=sample_domains,
        expert_domains=expert_domains,
        save_distribution_plots=compatibility_cfg.save_distribution_plots,
        uplift_reference_method=str(compatibility_cfg.uplift_reference_method),
        strong_spearman_uplift=compatibility_cfg.strong_spearman_uplift,
        strong_top1_uplift=compatibility_cfg.strong_top1_uplift,
        strong_gap_reduction=compatibility_cfg.strong_gap_reduction,
        weak_spearman_uplift=compatibility_cfg.weak_spearman_uplift,
        weak_top1_uplift=compatibility_cfg.weak_top1_uplift,
        weak_gap_reduction=compatibility_cfg.weak_gap_reduction,
        decision_policy_version=compatibility_cfg.decision_policy_version,
        instability_std_threshold=compatibility_cfg.instability_std_threshold,
        top1_uplift_std_threshold=compatibility_cfg.top1_uplift_std_threshold,
        spearman_uplift_std_threshold=compatibility_cfg.spearman_uplift_std_threshold,
        gap_pct_reduction_std_threshold=compatibility_cfg.gap_pct_reduction_std_threshold,
        instability_sign_inconsistency_min_count=compatibility_cfg.instability_sign_inconsistency_min_count,
        min_positive_fraction=compatibility_cfg.min_positive_fraction,
        ci_level=compatibility_cfg.ci_level,
        ci_bootstrap_reps=compatibility_cfg.ci_bootstrap_reps,
        ci_bootstrap_seed=compatibility_cfg.ci_bootstrap_seed,
        allow_missing_domain_breakdown_as_diagnostic=compatibility_cfg.allow_missing_domain_breakdown_as_diagnostic,
        hybrid_enabled=hybrid_cfg.enabled,
        tie_policy=hybrid_cfg.tie_policy,
        primary_norm_policy=hybrid_cfg.primary_norm_policy,
        sensitivity_norm_policy=hybrid_cfg.sensitivity_norm_policy,
        run_sensitivity=hybrid_cfg.run_sensitivity,
        min_rank_improvement_abs=hybrid_cfg.min_rank_improvement_abs,
        min_gap_pct_improvement_abs=hybrid_cfg.min_gap_pct_improvement_abs,
        max_top1_drop_abs=hybrid_cfg.max_top1_drop_abs,
        enable_random_rank_floor=compatibility_cfg.enable_random_rank_floor,
        enable_random_score_floor=compatibility_cfg.enable_random_score_floor,
        run_expert_label_permutation=compatibility_cfg.run_expert_label_permutation,
        run_metadata_permutation=compatibility_cfg.run_metadata_permutation,
        permutation_repeats=compatibility_cfg.permutation_repeats,
    )
