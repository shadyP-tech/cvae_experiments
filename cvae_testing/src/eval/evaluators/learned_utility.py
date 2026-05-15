from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.eval.evaluators.learned_utility_protocol import FoldCandidateSet
from src.eval.evaluators.ae_utility_calibrator import (
    run_ae_utility_calibrator_methods_for_fold,
    write_ae_utility_calibrator_artifacts,
)
from src.eval.evaluators.learned_utility_config import _parse_learned_utility_config
from src.eval.evaluators.learned_utility_learned_methods import _run_learned_methods_for_fold
from src.eval.evaluators.learned_utility_proxy_methods import _run_proxy_methods_for_fold
from src.eval.evaluators.learned_utility_proxies import (
    _latent_wasserstein_scores,
    _metadata_scores,
)
from src.eval.evaluators.learned_utility_residual import run_residual_methods_for_fold
from src.eval.evaluators.learned_utility_scoring import (
    _domain_to_expert_index,
    _score_experts_batched,
)
from src.eval.evaluators.learned_utility_reporting import (
    _finalize_learned_utility_outputs,
)
from src.eval.evaluators.pairwise_ae_combined_v2 import write_pairwise_ae_combined_v2_artifacts
from src.eval.evaluators.source_reliability import (
    run_source_reliability_for_fold,
    write_source_reliability_artifacts,
)
from src.eval.evaluators.support_free_ae import (
    AutoencoderScoreMatrices,
    build_autoencoder_score_matrices,
    run_ae_first_methods_for_fold,
    run_autoencoder_proxy_methods_for_fold,
    write_ae_first_artifacts,
    write_support_free_ae_artifacts,
)
from src.eval.evaluators.support_response_routing import (
    evaluate_support_response_routing_for_checkpoints,
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
    data_cfg: Dict[str, Any] | None = None,
    autoencoder_artifacts: Dict[str, Any] | None = None,
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

    ae_scores: AutoencoderScoreMatrices | None = None
    if eval_cfg.autoencoder.enabled:
        if not autoencoder_artifacts:
            raise ValueError("learned_utility.autoencoder_proxy.enabled requires autoencoder_artifacts")
        ae_scores = build_autoencoder_score_matrices(
            embeddings=embeddings,
            expert_domains=expert_domains,
            autoencoder_artifacts=autoencoder_artifacts,
            cfg=eval_cfg.autoencoder,
        )

    sample_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    pair_training_rows: List[Dict[str, Any]] = []
    proxy_diag_rows: List[Dict[str, Any]] = []
    residual_sample_rows: List[Dict[str, Any]] = []
    residual_raw_rows: List[Dict[str, Any]] = []
    residual_override_rows: List[Dict[str, Any]] = []
    residual_audit_rows: List[Dict[str, Any]] = []
    residual_confusion_rows: List[Dict[str, Any]] = []
    ae_first_raw_rows: List[Dict[str, Any]] = []
    ae_first_policy_rows: List[Dict[str, Any]] = []
    ae_first_validation_rows: List[Dict[str, Any]] = []
    ae_first_selection_diag_rows: List[Dict[str, Any]] = []
    ae_first_margin_bin_rows: List[Dict[str, Any]] = []
    ae_first_calibration_rows: List[Dict[str, Any]] = []
    ae_utility_raw_rows: List[Dict[str, Any]] = []
    ae_utility_validation_rows: List[Dict[str, Any]] = []
    ae_utility_policy_rows: List[Dict[str, Any]] = []
    ae_utility_override_rows: List[Dict[str, Any]] = []
    ae_utility_headroom_rows: List[Dict[str, Any]] = []
    ae_utility_selected_feature_rows: List[Dict[str, Any]] = []
    ae_utility_precision_rows: List[Dict[str, Any]] = []
    ae_utility_anchor_rank_rows: List[Dict[str, Any]] = []
    source_reliability_pseudo_rows: List[Dict[str, Any]] = []
    source_reliability_unit_rows: List[Dict[str, Any]] = []
    source_reliability_candidate_rows: List[Dict[str, Any]] = []
    source_reliability_parent_guard_rows: List[Dict[str, Any]] = []
    source_reliability_selection_rows: List[Dict[str, Any]] = []
    source_reliability_policy_rows: List[Dict[str, Any]] = []
    source_reliability_predicted_rows: List[Dict[str, Any]] = []
    source_reliability_selected_method_rows: List[Dict[str, Any]] = []
    pairwise_v2_training_rows: List[Dict[str, Any]] = []
    pairwise_v2_feature_rows: List[Dict[str, Any]] = []
    pairwise_v2_inner_selection_rows: List[Dict[str, Any]] = []
    pairwise_v2_pair_prediction_rows: List[Dict[str, Any]] = []
    pairwise_v2_decision_rows: List[Dict[str, Any]] = []
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

        if ae_scores is not None and eval_cfg.autoencoder.run_diagnostics:
            ae_outputs = run_autoencoder_proxy_methods_for_fold(
                sample_domains=sample_domains,
                expert_domains=expert_domains,
                test_idx=test_idx,
                fold=fold,
                true_eval=true_eval,
                global_eval=global_eval,
                metadata_similarity_eval=metadata_similarity_eval,
                ae_zscore_matrix=ae_scores.zscore_matrix,
                ae_raw_mse_matrix=ae_scores.raw_mse_matrix,
                margin_threshold=eval_cfg.autoencoder.margin_threshold,
                tie_policy=hybrid_cfg.tie_policy,
            )
            sample_rows.extend(ae_outputs.sample_rows)
            proxy_diag_rows.extend(ae_outputs.proxy_diag_rows)

        if ae_scores is not None and eval_cfg.autoencoder.ae_first.enabled:
            ae_first_outputs = run_ae_first_methods_for_fold(
                sample_domains=sample_domains,
                expert_domains=expert_domains,
                train_idx=train_idx,
                test_idx=test_idx,
                fold=fold,
                true_nelbo=true_nelbo,
                true_eval=true_eval,
                global_eval=global_eval,
                metadata_similarity=metadata_similarity,
                metadata_similarity_eval=metadata_similarity_eval,
                ae_scores=ae_scores,
                cfg=eval_cfg.autoencoder.ae_first,
                tie_policy=hybrid_cfg.tie_policy,
            )
            sample_rows.extend(ae_first_outputs.sample_rows)
            ae_first_raw_rows.extend(ae_first_outputs.raw_rows)
            ae_first_policy_rows.extend(ae_first_outputs.policy_audit_rows)
            ae_first_validation_rows.extend(ae_first_outputs.source_inner_validation_rows)
            ae_first_selection_diag_rows.extend(ae_first_outputs.selection_diag_rows)
            ae_first_margin_bin_rows.extend(ae_first_outputs.margin_bin_rows)
            ae_first_calibration_rows.extend(ae_first_outputs.calibration_rows)

        if ae_scores is not None and eval_cfg.autoencoder.utility_calibrator.enabled:
            ae_utility_outputs = run_ae_utility_calibrator_methods_for_fold(
                embeddings=embeddings,
                sample_domains=sample_domains,
                expert_domains=expert_domains,
                train_idx=train_idx,
                test_idx=test_idx,
                fold=fold,
                true_nelbo=true_nelbo,
                true_eval=true_eval,
                global_eval=global_eval,
                metadata_similarity=metadata_similarity,
                metadata_similarity_eval=metadata_similarity_eval,
                ae_scores=ae_scores,
                cfg=eval_cfg.autoencoder.utility_calibrator,
                seed=int(seed),
                tie_policy=hybrid_cfg.tie_policy,
            )
            sample_rows.extend(ae_utility_outputs.sample_rows)
            ae_utility_raw_rows.extend(ae_utility_outputs.raw_rows)
            ae_utility_validation_rows.extend(ae_utility_outputs.source_inner_validation_rows)
            ae_utility_policy_rows.extend(ae_utility_outputs.policy_audit_rows)
            ae_utility_override_rows.extend(ae_utility_outputs.override_diagnostic_rows)
            ae_utility_headroom_rows.extend(ae_utility_outputs.oracle_headroom_rows)
            ae_utility_selected_feature_rows.extend(ae_utility_outputs.selected_feature_rows)
            ae_utility_precision_rows.extend(ae_utility_outputs.override_precision_rows)
            ae_utility_anchor_rank_rows.extend(ae_utility_outputs.anchor_rank_rows)

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
            ae_zscore_matrix=None if ae_scores is None else ae_scores.zscore_matrix,
        )
        sample_rows.extend(learned_outputs.sample_rows)
        pair_rows.extend(learned_outputs.pair_rows)
        pair_training_rows.extend(learned_outputs.pair_training_rows)
        pairwise_v2_training_rows.extend(learned_outputs.pairwise_v2_training_rows)
        pairwise_v2_feature_rows.extend(learned_outputs.pairwise_v2_feature_rows)
        pairwise_v2_inner_selection_rows.extend(learned_outputs.pairwise_v2_inner_selection_rows)
        pairwise_v2_pair_prediction_rows.extend(learned_outputs.pairwise_v2_pair_prediction_rows)
        pairwise_v2_decision_rows.extend(learned_outputs.pairwise_v2_decision_rows)

        if eval_cfg.source_reliability.enabled:
            if ae_scores is None:
                raise ValueError("learned_utility.source_reliability.enabled requires autoencoder_proxy.enabled")
            source_reliability_outputs = run_source_reliability_for_fold(
                embeddings=embeddings,
                sample_domains=sample_domains,
                metadata=metadata,
                true_nelbo=true_nelbo,
                expert_domains=expert_domains,
                train_idx=train_idx,
                test_idx=test_idx,
                fold=fold,
                true_eval=true_eval,
                global_eval=global_eval,
                ae_zscore_matrix=ae_scores.zscore_matrix,
                learned_sample_rows=learned_outputs.sample_rows,
                pairwise_cfg=eval_cfg.pairwise_cfg,
                cfg=eval_cfg.source_reliability,
                seed=int(seed),
                tie_policy=hybrid_cfg.tie_policy,
            )
            sample_rows.extend(source_reliability_outputs.sample_rows)
            source_reliability_pseudo_rows.extend(source_reliability_outputs.pseudo_domain_rows)
            source_reliability_unit_rows.extend(source_reliability_outputs.source_inner_unit_rows)
            source_reliability_candidate_rows.extend(source_reliability_outputs.candidate_metric_rows)
            source_reliability_parent_guard_rows.extend(source_reliability_outputs.parent_guard_rows)
            source_reliability_selection_rows.extend(source_reliability_outputs.selection_policy_rows)
            source_reliability_policy_rows.extend(source_reliability_outputs.policy_audit_rows)
            source_reliability_predicted_rows.extend(source_reliability_outputs.predicted_vs_realized_rows)
            source_reliability_selected_method_rows.extend(source_reliability_outputs.selected_method_rows)

        residual_outputs = run_residual_methods_for_fold(
            embeddings=embeddings,
            sample_domains=sample_domains,
            true_nelbo=true_nelbo,
            expert_domains=expert_domains,
            metadata_similarity=metadata_similarity,
            train_idx=train_idx,
            test_idx=test_idx,
            fold=fold,
            global_eval=global_eval,
            residual_cfg=eval_cfg.residual,
            learned_sample_rows=learned_outputs.sample_rows,
            tie_policy=hybrid_cfg.tie_policy,
            ae_zscore_matrix=None if ae_scores is None else ae_scores.zscore_matrix,
        )
        sample_rows.extend(residual_outputs.sample_rows)
        residual_sample_rows.extend(residual_outputs.sample_rows)
        residual_raw_rows.extend(residual_outputs.raw_rows)
        residual_override_rows.extend(residual_outputs.override_rows)
        residual_audit_rows.extend(residual_outputs.audit_rows)
        residual_confusion_rows.extend(residual_outputs.confusion_rows)

        if fold_end is not None:
            fold_end.record()
            torch.cuda.synchronize()
            elapsed_ms = float(fold_start.elapsed_time(fold_end))
            print(f"[learned_utility] fold {fold_idx}/{total_folds} done in {elapsed_ms / 1000.0:.2f}s")
        else:
            print(f"[learned_utility] fold {fold_idx}/{total_folds} done")

    results = _finalize_learned_utility_outputs(
        reports_dir=reports_dir,
        sample_rows=sample_rows,
        pair_rows=pair_rows,
        pair_training_rows=pair_training_rows,
        proxy_diag_rows=proxy_diag_rows,
        residual_sample_rows=residual_sample_rows,
        residual_raw_rows=residual_raw_rows,
        residual_override_rows=residual_override_rows,
        residual_audit_rows=residual_audit_rows,
        residual_confusion_rows=residual_confusion_rows,
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
    ae_artifacts = write_support_free_ae_artifacts(
        reports_dir=reports_dir,
        ae_scores=ae_scores,
        proxy_diag_rows=[
            row for row in proxy_diag_rows if str(row.get("method", "")) == "ae_reconstruction_zscore_raw"
        ],
        residual_override_rows=residual_override_rows,
    )
    if ae_artifacts:
        results.setdefault("artifacts", {}).update(ae_artifacts)
    ae_first_artifacts = write_ae_first_artifacts(
        reports_dir=reports_dir,
        raw_rows=ae_first_raw_rows,
        policy_audit_rows=ae_first_policy_rows,
        source_inner_validation_rows=ae_first_validation_rows,
        selection_diag_rows=ae_first_selection_diag_rows,
        margin_bin_rows=ae_first_margin_bin_rows,
        calibration_rows=ae_first_calibration_rows,
    )
    if ae_first_artifacts:
        results.setdefault("artifacts", {}).update(ae_first_artifacts)
    ae_utility_artifacts = write_ae_utility_calibrator_artifacts(
        reports_dir=reports_dir,
        raw_rows=ae_utility_raw_rows,
        source_inner_validation_rows=ae_utility_validation_rows,
        policy_audit_rows=ae_utility_policy_rows,
        override_diagnostic_rows=ae_utility_override_rows,
        oracle_headroom_rows=ae_utility_headroom_rows,
        selected_feature_rows=ae_utility_selected_feature_rows,
        override_precision_rows=ae_utility_precision_rows,
        anchor_rank_rows=ae_utility_anchor_rank_rows,
    )
    if ae_utility_artifacts:
        results.setdefault("artifacts", {}).update(ae_utility_artifacts)
    source_reliability_artifacts = write_source_reliability_artifacts(
        reports_dir=reports_dir,
        pseudo_domain_rows=source_reliability_pseudo_rows,
        source_inner_unit_rows=source_reliability_unit_rows,
        candidate_metric_rows=source_reliability_candidate_rows,
        parent_guard_rows=source_reliability_parent_guard_rows,
        selection_policy_rows=source_reliability_selection_rows,
        policy_audit_rows=source_reliability_policy_rows,
        predicted_vs_realized_rows=source_reliability_predicted_rows,
        selected_method_rows=source_reliability_selected_method_rows,
    )
    if source_reliability_artifacts:
        results.setdefault("artifacts", {}).update(source_reliability_artifacts)
    pairwise_v2_artifacts = write_pairwise_ae_combined_v2_artifacts(
        reports_dir=reports_dir,
        training_rows=pairwise_v2_training_rows,
        feature_rows=pairwise_v2_feature_rows,
        inner_selection_rows=pairwise_v2_inner_selection_rows,
        pair_prediction_rows=pairwise_v2_pair_prediction_rows,
        decision_rows=pairwise_v2_decision_rows,
    )
    if pairwise_v2_artifacts:
        results.setdefault("artifacts", {}).update(pairwise_v2_artifacts)
    if eval_cfg.support_response.enabled:
        print("[learned_utility] running candidate-specific support-response routing...")
        support_response_results = evaluate_support_response_routing_for_checkpoints(
            embeddings=embeddings,
            metadata=metadata,
            nelbo_matrix=true_nelbo,
            expert_domains=expert_domains,
            expert_checkpoints=expert_checkpoints,
            hidden_dim=int(hidden_dim),
            latent_dim=int(latent_dim),
            seed=int(seed),
            dataset_name=str((data_cfg or {}).get("dataset_type", "unknown")),
            strategy=str(strategy),
            tau=float(tau),
            support_cfg=eval_cfg.support_response,
            reports_dir=reports_dir,
            data_cfg=data_cfg or {},
            metadata_constraint_cfg=metadata_constraint_cfg,
        )
        results["support_response_results"] = support_response_results
        results.setdefault("artifacts", {})["support_response_results"] = "support_response_results.json"
    return results
