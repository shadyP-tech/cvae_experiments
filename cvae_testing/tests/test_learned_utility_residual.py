from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluators import learned_utility as lu
from src.eval.evaluators import ae_utility_calibrator as auc
from src.eval.evaluators import support_free_ae as sfa
from src.eval.evaluators.learned_utility_config import ResidualRoutingConfig
from src.eval.evaluators.learned_utility_protocol import FoldCandidateSet
from src.eval.evaluators.learned_utility_residual import (
    _build_residual_training_rows,
    _feature_context,
    _safe_v2_adoption_feature_sets,
    _safe_v2_validation_report,
    _selected_from_residual,
)
from src.eval.evaluators.support_free_ae import AutoencoderScoreMatrices
from src.train.train_autoencoders import build_support_free_ae_overlap_audit


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fake_scored_payload():
    expert_domains = [40, 100, 200]
    sample_domains = np.asarray([40, 40, 100, 100, 200, 200], dtype=np.int64)
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
            [2.0, 0.0],
            [2.1, 0.0],
        ],
        dtype=np.float64,
    )
    nelbo = np.asarray(
        [
            [0.1, 0.5, 0.8],
            [0.1, 0.4, 0.9],
            [0.6, 0.1, 0.7],
            [0.8, 0.1, 0.5],
            [0.9, 0.4, 0.1],
            [0.8, 0.3, 0.1],
        ],
        dtype=np.float64,
    )
    metadata = [
        {"magnification": int(domain), "sample_id": f"s{i}"}
        for i, domain in enumerate(sample_domains.tolist())
    ]
    return embeddings, sample_domains, nelbo, expert_domains, metadata


def _residual_cfg() -> dict:
    return {
        "predictors": ["linear_regressor"],
        "pair_features": {"include_metadata_features": True},
        "scoring": {"pair_batch_size": 2},
        "hybrid_scoring": {
            "enabled": False,
            "tie_policy": "stable_expert_index",
        },
        "residual_routing": {
            "enabled": True,
            "models": ["ridge"],
            "thresholds": [0, 0.01, "inf"],
            "feature_sets": ["minimal", "latent"],
            "selection_metric": "validation_safe_gap_then_top1",
            "unconstrained_reference_method": "linear_regressor",
            "ridge_l2": 1.0e-4,
        },
        "compatibility_research": {
            "floors": {"random_rank_floor": False, "random_score_floor": False},
            "permutation_tests": {
                "expert_label_permutation": False,
                "metadata_permutation": False,
                "repeats": 1,
            },
            "diagnostics": {"save_distribution_plots": False},
            "gate": {"decision_policy_version": "sign_ci_v2", "uplift_reference_method": "metadata_routing"},
        },
    }


def _safe_v2_cfg() -> dict:
    cfg = _residual_cfg()
    cfg["residual_routing"] = {
        **cfg["residual_routing"],
        "residual_policy_version": "metadata_residual_safe_override_v2",
        "feature_sets": ["minimal", "latent", "calibrated"],
        "adoption_feature_sets": ["minimal", "latent"],
        "diagnostic_feature_sets": ["calibrated"],
        "allow_calibrated_adoption": False,
        "harmful_override_max": 0.05,
        "gap_regression_max": 2.0,
        "catastrophic_top1_floor": -0.05,
    }
    return cfg


def _support_free_ae_cfg(*, thresholds=None) -> dict:
    cfg = _residual_cfg()
    cfg["autoencoder_proxy"] = {
        "enabled": True,
        "hidden_dim": 8,
        "latent_dim": 2,
        "learning_rate": 1.0e-3,
        "epochs": 2,
        "patience": 1,
        "batch_size": 8,
        "score_normalization": "source_val_zscore",
        "score_normalization_eps": 1.0e-6,
        "margin_threshold": 0.0,
        "run_diagnostics": True,
    }
    cfg["residual_routing"] = {
        **cfg["residual_routing"],
        "residual_policy_version": "metadata_ae_residual_safe_override_v1",
        "thresholds": list(thresholds if thresholds is not None else ["inf"]),
        "feature_sets": ["ae"],
        "adoption_feature_sets": ["ae"],
        "diagnostic_feature_sets": [],
        "allow_calibrated_adoption": False,
        "harmful_override_max": 0.50,
        "gap_regression_max": 1.0,
        "catastrophic_top1_floor": -0.02,
        "unconstrained_reference_method": "linear_regressor",
    }
    return cfg


def _fake_ae_scores() -> AutoencoderScoreMatrices:
    z = np.asarray(
        [
            [0.0, 0.8, 1.4],
            [0.0, 0.7, 1.5],
            [1.2, 0.0, 0.6],
            [1.3, 0.0, 0.4],
            [1.5, 0.5, 0.0],
            [1.4, 0.4, 0.0],
        ],
        dtype=np.float64,
    )
    return AutoencoderScoreMatrices(
        raw_mse_matrix=z.copy(),
        zscore_matrix=z,
        quality_rows=[
            {
                "source_domain": d,
                "source_val_reconstruction_mse_by_domain": 0.0,
                "source_val_reconstruction_std_by_domain": 1.0,
                "ae_training_converged": 1,
                "ae_best_epoch": 0,
                "ae_val_loss": 0.0,
            }
            for d in [40, 100, 200]
        ],
        provenance_rows=[],
        overlap_rows=[
            {
                "ae_train_query_overlap_count": 0,
                "ae_val_query_overlap_count": 0,
                "ae_train_cache_hash": "train",
                "ae_val_cache_hash": "val",
                "routing_query_cache_hash": "test",
                "routing_eval_cache_hash": "test",
            }
        ],
        provenance={
            "overlap_audit": {
                "ae_train_query_overlap_count": 0,
                "ae_val_query_overlap_count": 0,
                "ae_train_cache_hash": "train",
                "ae_val_cache_hash": "val",
                "routing_query_cache_hash": "test",
                "routing_eval_cache_hash": "test",
            }
        },
    )


def _fake_payload_ae_first():
    expert_domains = [10, 20, 30, 40]
    sample_domains = np.asarray([10, 10, 20, 20, 30, 30, 40, 40], dtype=np.int64)
    true_nelbo = np.asarray(
        [
            [0.1, 3.0, 1.0, 4.0],
            [0.1, 3.2, 1.1, 4.2],
            [9.0, 0.1, 5.0, 6.0],
            [9.0, 0.1, 5.2, 6.2],
            [9.0, 1.0, 0.1, 4.0],
            [9.0, 1.2, 0.1, 4.2],
            [9.0, 1.0, 5.0, 0.1],
            [9.0, 1.2, 5.2, 0.1],
        ],
        dtype=np.float64,
    )
    metadata_similarity = np.asarray(
        [
            [0.0, 1.0, 0.2, 0.1],
            [0.0, 1.0, 0.2, 0.1],
            [0.0, 0.0, 0.2, 1.0],
            [0.0, 0.0, 0.2, 1.0],
            [0.0, 0.3, 0.0, 1.0],
            [0.0, 0.3, 0.0, 1.0],
            [0.0, 1.0, 0.4, 0.0],
            [0.0, 1.0, 0.4, 0.0],
        ],
        dtype=np.float64,
    )
    z = np.asarray(
        [
            [9.0, 1.0, 0.0, 2.0],
            [9.0, 1.1, 0.0, 2.1],
            [9.0, 9.0, 0.0, 2.0],
            [9.0, 9.0, 0.1, 2.1],
            [9.0, 0.0, 9.0, 2.0],
            [9.0, 0.1, 9.0, 2.1],
            [9.0, 0.0, 2.0, 9.0],
            [9.0, 0.1, 2.1, 9.0],
        ],
        dtype=np.float64,
    )
    ae_scores = AutoencoderScoreMatrices(
        raw_mse_matrix=z.copy(),
        zscore_matrix=z,
        quality_rows=[
            {
                "source_domain": d,
                "source_val_reconstruction_mse_by_domain": 0.0,
                "source_val_reconstruction_std_by_domain": 1.0,
                "ae_source_val_count": 2,
                "ae_z_sigma_floor": 0.1,
                "ae_z_sigma_floor_applied": 0,
            }
            for d in expert_domains
        ],
        provenance_rows=[],
        overlap_rows=[],
        provenance={},
    )
    return sample_domains, expert_domains, true_nelbo, metadata_similarity, ae_scores


def _ae_first_cfg(*, thresholds=None):
    cfg = _support_free_ae_cfg()
    cfg["autoencoder_proxy"]["ae_first_routing"] = {
        "enabled": True,
        "primary_method": "ae_first_margin_gated_v1",
        "fallback_baseline": "source_prior_fallback",
        "margin_thresholds": list(thresholds if thresholds is not None else [0.0, "__inf__"]),
        "metadata_auxiliary_features": True,
        "ae_z_sigma_floor_mode": "global_source_val_std_quantile",
        "ae_z_sigma_floor_quantile": 0.05,
        "min_ae_coverage_rate_for_weak_pass": 0.10,
        "min_ae_coverage_rate_for_pass": 0.20,
        "risk_gates": {
            "max_top1_drop_abs": 0.02,
            "max_raw_spearman_drop_abs": 0.03,
            "max_gap_pct_degradation": 1.0,
        },
    }
    return lu._parse_learned_utility_config(cfg).autoencoder.ae_first


def _ae_utility_calibrator_cfg(*, delta_thresholds=None):
    cfg = _support_free_ae_cfg()
    cfg["autoencoder_proxy"]["utility_calibrator"] = {
        "enabled": True,
        "primary_method": "ae_utility_calibrated_safe_override_v1",
        "model_types": ["ridge_delta"],
        "primary_model_type": "ridge_delta",
        "diagnostic_model_types": ["pairwise_ranker"],
        "fallback_policy": "ae_argmin_zscore",
        "feature_sets_primary": ["ae_core", "ae_quality"],
        "feature_sets_diagnostic": ["ae_metadata", "ae_combined"],
        "delta_thresholds": list(delta_thresholds if delta_thresholds is not None else [0.0, "__inf__"]),
        "margin_thresholds": [0.0, 0.05],
        "ridge_l2": 1.0e-4,
        "risk_gates": {
            "max_top1_drop_vs_ae_argmin_abs": 0.02,
            "max_spearman_drop_vs_ae_argmin_abs": 0.03,
            "max_gap_pct_degradation_vs_ae_argmin": 1.0,
            "max_top1_drop_vs_metadata_abs": 0.02,
            "max_spearman_drop_vs_metadata_abs": 0.03,
            "max_gap_pct_degradation_vs_metadata": 1.0,
        },
    }
    return lu._parse_learned_utility_config(cfg).autoencoder.utility_calibrator


def _ae_utility_precision_v11_cfg(
    *,
    delta_thresholds=None,
    min_active_override_count=10,
    min_strict_lcb=0.60,
    max_worst_gap=1.0,
):
    cfg = _support_free_ae_cfg()
    cfg["autoencoder_proxy"]["utility_calibrator"] = {
        "enabled": True,
        "primary_method": "ae_utility_calibrated_precision_lcb_safe_override_v11",
        "model_types": ["ridge_delta"],
        "primary_model_type": "ridge_delta",
        "diagnostic_model_types": ["pairwise_ranker"],
        "fallback_policy": "ae_argmin_zscore",
        "feature_sets_primary": ["ae_core", "ae_quality"],
        "feature_sets_diagnostic": [],
        "delta_thresholds": list(delta_thresholds if delta_thresholds is not None else [0.0, "__inf__"]),
        "margin_thresholds": [0.0, 0.05],
        "selection_mode": "precision_lcb_selected_v11",
        "ridge_l2": 1.0e-4,
        "precision_selection": {
            "min_strict_improvement_precision": 0.75,
            "min_strict_improvement_precision_lcb": float(min_strict_lcb),
            "min_active_override_count": int(min_active_override_count),
            "min_active_override_rate": 0.10,
            "min_net_gain_vs_ae_argmin": 0.0,
            "neutral_override_gap_pct_band": 0.25,
            "max_worst_pseudo_domain_gap_degradation_pp": float(max_worst_gap),
            "bootstrap_reps": 200,
            "bootstrap_seed": 1337,
            "diagnostic_precision_thresholds": [0.70, 0.75, 0.80, 0.85],
        },
        "risk_gates": {
            "max_top1_drop_vs_ae_argmin_abs": 0.02,
            "max_spearman_drop_vs_ae_argmin_abs": 0.03,
            "max_gap_pct_degradation_vs_ae_argmin": 1.0,
            "max_top1_drop_vs_metadata_abs": 0.02,
            "max_spearman_drop_vs_metadata_abs": 0.03,
            "max_gap_pct_degradation_vs_metadata": 1.0,
        },
    }
    return lu._parse_learned_utility_config(cfg).autoencoder.utility_calibrator


def _ae_utility_precision_v12_cfg(
    *,
    delta_thresholds=None,
    min_active_override_count=12,
    min_strict_lcb=0.60,
    max_worst_gap=1.0,
    min_gap_delta_vs_v1_lcb=-0.25,
    max_harm_ucb=0.30,
):
    cfg = _support_free_ae_cfg()
    cfg["autoencoder_proxy"]["utility_calibrator"] = {
        "enabled": True,
        "primary_method": "ae_utility_calibrated_precision_lcb_v1_guarded_safe_override_v12",
        "model_types": ["ridge_delta"],
        "primary_model_type": "ridge_delta",
        "diagnostic_model_types": ["pairwise_ranker"],
        "fallback_policy": "ae_argmin_zscore",
        "feature_sets_primary": ["ae_core", "ae_quality"],
        "feature_sets_diagnostic": [],
        "delta_thresholds": list(delta_thresholds if delta_thresholds is not None else [0.0, "__inf__"]),
        "margin_thresholds": [0.0, 0.05],
        "selection_mode": "precision_lcb_v1_guarded_v12",
        "ridge_l2": 1.0e-4,
        "precision_selection": {
            "min_strict_improvement_precision": 0.75,
            "min_strict_improvement_precision_lcb": float(min_strict_lcb),
            "min_active_override_count": int(min_active_override_count),
            "min_active_override_rate": 0.10,
            "min_net_gain_vs_ae_argmin": 0.0,
            "neutral_override_gap_pct_band": 0.25,
            "max_worst_pseudo_domain_gap_degradation_pp": float(max_worst_gap),
            "bootstrap_reps": 200,
            "bootstrap_seed": 1337,
            "diagnostic_precision_thresholds": [0.70, 0.75, 0.80, 0.85],
            "v1_guard": {
                "min_gap_delta_vs_v1_lcb_pp": float(min_gap_delta_vs_v1_lcb),
                "max_top1_drop_vs_v1_abs": 0.02,
                "max_spearman_drop_vs_v1_abs": 0.03,
                "max_worst_pseudo_domain_gap_degradation_vs_v1_pp": 1.0,
                "max_harmful_override_rate_ucb": float(max_harm_ucb),
            },
        },
        "risk_gates": {
            "max_top1_drop_vs_ae_argmin_abs": 0.02,
            "max_spearman_drop_vs_ae_argmin_abs": 0.03,
            "max_gap_pct_degradation_vs_ae_argmin": 1.0,
            "max_top1_drop_vs_metadata_abs": 0.02,
            "max_spearman_drop_vs_metadata_abs": 0.03,
            "max_gap_pct_degradation_vs_metadata": 1.0,
        },
    }
    return lu._parse_learned_utility_config(cfg).autoencoder.utility_calibrator


def _ae_utility_harm_veto_v13_cfg(
    *,
    delta_thresholds=None,
    min_active_v1_override_count=12,
    min_veto_count=6,
    min_harmful_count=3,
    min_harm_precision_lcb=0.50,
    max_false_veto_ucb=0.40,
):
    cfg = _support_free_ae_cfg()
    cfg["autoencoder_proxy"]["utility_calibrator"] = {
        "enabled": True,
        "primary_method": "ae_utility_calibrated_v1_harm_veto_safe_override_v13",
        "model_types": ["ridge_delta"],
        "primary_model_type": "ridge_delta",
        "diagnostic_model_types": ["pairwise_ranker"],
        "fallback_policy": "ae_argmin_zscore",
        "feature_sets_primary": ["ae_core", "ae_quality"],
        "feature_sets_diagnostic": [],
        "delta_thresholds": list(delta_thresholds if delta_thresholds is not None else [0.0, "__inf__"]),
        "margin_thresholds": [0.0, 0.05],
        "selection_mode": "v1_harm_veto_v13",
        "ridge_l2": 1.0e-4,
        "harm_veto": {
            "veto_score_model": "logistic_harm_score",
            "veto_thresholds": [0.50, 0.60, "__inf__"],
            "min_active_v1_override_count_source_inner": int(min_active_v1_override_count),
            "min_veto_count_source_inner": int(min_veto_count),
            "min_harmful_v1_override_count_source_inner": int(min_harmful_count),
            "min_strict_harm_prevention_precision_lcb": float(min_harm_precision_lcb),
            "max_false_veto_rate_ucb": float(max_false_veto_ucb),
            "min_retained_v1_override_gain_rate": 0.0,
            "min_active_override_rate_ratio_vs_v1": 0.0,
            "min_gap_delta_vs_v1_lcb_pp": -999.0,
            "neutral_override_gap_pct_band": 0.25,
            "bootstrap_reps": 200,
            "bootstrap_seed": 1337,
        },
        "risk_gates": {
            "max_top1_drop_vs_ae_argmin_abs": 0.02,
            "max_spearman_drop_vs_ae_argmin_abs": 0.03,
            "max_gap_pct_degradation_vs_ae_argmin": 1.0,
            "max_top1_drop_vs_metadata_abs": 0.02,
            "max_spearman_drop_vs_metadata_abs": 0.03,
            "max_gap_pct_degradation_vs_metadata": 1.0,
        },
    }
    return lu._parse_learned_utility_config(cfg).autoencoder.utility_calibrator


def _ae_utility_recall_v15_cfg(
    *,
    budget_rates=None,
    min_recall_count=10,
    max_active_ratio=1.20,
    min_strict_precision=0.0,
    min_strict_lcb=0.0,
    max_harm_ucb=1.0,
):
    cfg = _support_free_ae_cfg()
    cfg["autoencoder_proxy"]["utility_calibrator"] = {
        "enabled": True,
        "primary_method": "ae_utility_calibrated_v1_recall_budget_safe_override_v15",
        "model_types": ["ridge_delta"],
        "primary_model_type": "ridge_delta",
        "diagnostic_model_types": ["pairwise_ranker"],
        "fallback_policy": "ae_argmin_zscore",
        "feature_sets_primary": ["ae_core", "ae_quality"],
        "feature_sets_diagnostic": [],
        "delta_thresholds": [0.0, "__inf__"],
        "margin_thresholds": [0.0, 0.05],
        "selection_mode": "v1_recall_budget_v15",
        "ridge_l2": 1.0e-4,
        "recall_expansion": {
            "scoring_policy": "ridge_delta_best_non_anchor",
            "recall_budget_rates": list(budget_rates if budget_rates is not None else [0.0, 0.50]),
            "budget_scope": "v1_abstentions_per_fold",
            "min_v1_abstention_count_source_inner": 1,
            "min_recall_override_count_source_inner": int(min_recall_count),
            "min_recall_override_count_source_inner_for_pass": 2,
            "min_strict_recall_precision": float(min_strict_precision),
            "min_strict_recall_precision_lcb": float(min_strict_lcb),
            "max_harmful_recall_rate_ucb": float(max_harm_ucb),
            "min_net_gain_vs_v1_source_inner": -999.0,
            "min_gap_delta_vs_v1_lcb_pp": -999.0,
            "min_gap_delta_vs_v1_lcb_pp_for_pass": 0.0,
            "max_active_override_rate_ratio_vs_v1": float(max_active_ratio),
            "diagnostic_active_override_rate_ratio_upper_bound": 1.35,
            "max_worst_pseudo_domain_gap_degradation_vs_v1_pp": 999.0,
            "neutral_override_gap_pct_band": 0.25,
            "bootstrap_reps": 100,
            "bootstrap_seed": 1337,
        },
        "risk_gates": {
            "max_top1_drop_vs_ae_argmin_abs": 0.02,
            "max_spearman_drop_vs_ae_argmin_abs": 0.03,
            "max_gap_pct_degradation_vs_ae_argmin": 1.0,
            "max_top1_drop_vs_metadata_abs": 0.02,
            "max_spearman_drop_vs_metadata_abs": 0.03,
            "max_gap_pct_degradation_vs_metadata": 1.0,
        },
    }
    return lu._parse_learned_utility_config(cfg).autoencoder.utility_calibrator


def _ae_utility_consensus_v2_cfg(*, delta_thresholds=None, consensus_thresholds=None):
    cfg = _support_free_ae_cfg()
    cfg["autoencoder_proxy"]["utility_calibrator"] = {
        "enabled": True,
        "primary_method": "ae_utility_calibrated_consensus_safe_override_v2",
        "model_types": ["ridge_delta_consensus"],
        "primary_model_type": "ridge_delta_consensus",
        "diagnostic_model_types": ["pairwise_ranker"],
        "fallback_policy": "ae_argmin_zscore",
        "feature_sets_primary": ["ae_consensus_core", "ae_consensus_quality"],
        "feature_sets_diagnostic": ["ae_metadata_consensus", "ae_combined_consensus"],
        "delta_thresholds": list(delta_thresholds if delta_thresholds is not None else [0.0, "__inf__"]),
        "margin_thresholds": [0.0, 0.05],
        "consensus_thresholds": list(consensus_thresholds if consensus_thresholds is not None else [0.60, 1.00]),
        "uncertainty_multiplier": 1.0,
        "ensemble_strategy": "source_domain_leave_one_plus_full",
        "abstention_correct_gap_pct_epsilon": 1.0,
        "source_inner_stability_gates": {
            "min_pseudo_domain_positive_rate": 0.80,
            "max_pseudo_domain_gain_share": 0.50,
            "max_source_inner_fold_gain_share": 0.50,
        },
        "ridge_l2": 1.0e-4,
        "risk_gates": {
            "max_top1_drop_vs_ae_argmin_abs": 0.02,
            "max_spearman_drop_vs_ae_argmin_abs": 0.03,
            "max_gap_pct_degradation_vs_ae_argmin": 1.0,
            "max_top1_drop_vs_metadata_abs": 0.02,
            "max_spearman_drop_vs_metadata_abs": 0.03,
            "max_gap_pct_degradation_vs_metadata": 1.0,
        },
    }
    return lu._parse_learned_utility_config(cfg).autoencoder.utility_calibrator


def _run_ae_first_direct(*, thresholds=None, metadata_similarity=None):
    sample_domains, expert_domains, true_nelbo, meta, ae_scores = _fake_payload_ae_first()
    if metadata_similarity is not None:
        meta = metadata_similarity
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=10, expert_domains=expert_domains)
    test_idx = np.where(sample_domains == 10)[0]
    train_idx = np.where(sample_domains != 10)[0]
    return sfa.run_ae_first_methods_for_fold(
        sample_domains=sample_domains,
        expert_domains=expert_domains,
        train_idx=train_idx,
        test_idx=test_idx,
        fold=fold,
        true_nelbo=true_nelbo,
        true_eval=fold.slice_nelbo(true_nelbo, test_idx),
        global_eval=true_nelbo[test_idx],
        metadata_similarity=meta,
        metadata_similarity_eval=meta[test_idx][:, list(fold.candidate_col_indices)],
        ae_scores=ae_scores,
        cfg=_ae_first_cfg(thresholds=thresholds),
        tie_policy="stable_expert_index",
    )


def _run_ae_utility_calibrator_direct(*, delta_thresholds=None):
    sample_domains, expert_domains, true_nelbo, meta, ae_scores = _fake_payload_ae_first()
    embeddings = np.stack([np.asarray([float(i), 0.0]) for i in range(len(sample_domains))])
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=10, expert_domains=expert_domains)
    test_idx = np.where(sample_domains == 10)[0]
    train_idx = np.where(sample_domains != 10)[0]
    return auc.run_ae_utility_calibrator_methods_for_fold(
        embeddings=embeddings,
        sample_domains=sample_domains,
        expert_domains=expert_domains,
        train_idx=train_idx,
        test_idx=test_idx,
        fold=fold,
        true_nelbo=true_nelbo,
        true_eval=fold.slice_nelbo(true_nelbo, test_idx),
        global_eval=true_nelbo[test_idx],
        metadata_similarity=meta,
        metadata_similarity_eval=meta[test_idx][:, list(fold.candidate_col_indices)],
        ae_scores=ae_scores,
        cfg=_ae_utility_calibrator_cfg(delta_thresholds=delta_thresholds),
        seed=7,
        tie_policy="stable_expert_index",
    )


def _run_ae_utility_precision_v11_direct(
    *,
    delta_thresholds=None,
    min_active_override_count=10,
    min_strict_lcb=0.60,
    max_worst_gap=1.0,
):
    sample_domains, expert_domains, true_nelbo, meta, ae_scores = _fake_payload_ae_first()
    embeddings = np.stack([np.asarray([float(i), 0.0]) for i in range(len(sample_domains))])
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=10, expert_domains=expert_domains)
    test_idx = np.where(sample_domains == 10)[0]
    train_idx = np.where(sample_domains != 10)[0]
    return auc.run_ae_utility_calibrator_methods_for_fold(
        embeddings=embeddings,
        sample_domains=sample_domains,
        expert_domains=expert_domains,
        train_idx=train_idx,
        test_idx=test_idx,
        fold=fold,
        true_nelbo=true_nelbo,
        true_eval=fold.slice_nelbo(true_nelbo, test_idx),
        global_eval=true_nelbo[test_idx],
        metadata_similarity=meta,
        metadata_similarity_eval=meta[test_idx][:, list(fold.candidate_col_indices)],
        ae_scores=ae_scores,
        cfg=_ae_utility_precision_v11_cfg(
            delta_thresholds=delta_thresholds,
            min_active_override_count=min_active_override_count,
            min_strict_lcb=min_strict_lcb,
            max_worst_gap=max_worst_gap,
        ),
        seed=7,
        tie_policy="stable_expert_index",
    )


def _run_ae_utility_precision_v12_direct(
    *,
    delta_thresholds=None,
    min_active_override_count=12,
    min_strict_lcb=0.60,
    max_worst_gap=1.0,
    min_gap_delta_vs_v1_lcb=-0.25,
    max_harm_ucb=0.30,
):
    sample_domains, expert_domains, true_nelbo, meta, ae_scores = _fake_payload_ae_first()
    embeddings = np.stack([np.asarray([float(i), 0.0]) for i in range(len(sample_domains))])
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=10, expert_domains=expert_domains)
    test_idx = np.where(sample_domains == 10)[0]
    train_idx = np.where(sample_domains != 10)[0]
    return auc.run_ae_utility_calibrator_methods_for_fold(
        embeddings=embeddings,
        sample_domains=sample_domains,
        expert_domains=expert_domains,
        train_idx=train_idx,
        test_idx=test_idx,
        fold=fold,
        true_nelbo=true_nelbo,
        true_eval=fold.slice_nelbo(true_nelbo, test_idx),
        global_eval=true_nelbo[test_idx],
        metadata_similarity=meta,
        metadata_similarity_eval=meta[test_idx][:, list(fold.candidate_col_indices)],
        ae_scores=ae_scores,
        cfg=_ae_utility_precision_v12_cfg(
            delta_thresholds=delta_thresholds,
            min_active_override_count=min_active_override_count,
            min_strict_lcb=min_strict_lcb,
            max_worst_gap=max_worst_gap,
            min_gap_delta_vs_v1_lcb=min_gap_delta_vs_v1_lcb,
            max_harm_ucb=max_harm_ucb,
        ),
        seed=7,
        tie_policy="stable_expert_index",
    )


def _run_ae_utility_harm_veto_v13_direct(
    *,
    delta_thresholds=None,
    min_active_v1_override_count=12,
    min_veto_count=6,
    min_harmful_count=3,
    min_harm_precision_lcb=0.50,
    max_false_veto_ucb=0.40,
):
    sample_domains, expert_domains, true_nelbo, meta, ae_scores = _fake_payload_ae_first()
    embeddings = np.stack([np.asarray([float(i), 0.0]) for i in range(len(sample_domains))])
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=10, expert_domains=expert_domains)
    test_idx = np.where(sample_domains == 10)[0]
    train_idx = np.where(sample_domains != 10)[0]
    return auc.run_ae_utility_calibrator_methods_for_fold(
        embeddings=embeddings,
        sample_domains=sample_domains,
        expert_domains=expert_domains,
        train_idx=train_idx,
        test_idx=test_idx,
        fold=fold,
        true_nelbo=true_nelbo,
        true_eval=fold.slice_nelbo(true_nelbo, test_idx),
        global_eval=true_nelbo[test_idx],
        metadata_similarity=meta,
        metadata_similarity_eval=meta[test_idx][:, list(fold.candidate_col_indices)],
        ae_scores=ae_scores,
        cfg=_ae_utility_harm_veto_v13_cfg(
            delta_thresholds=delta_thresholds,
            min_active_v1_override_count=min_active_v1_override_count,
            min_veto_count=min_veto_count,
            min_harmful_count=min_harmful_count,
            min_harm_precision_lcb=min_harm_precision_lcb,
            max_false_veto_ucb=max_false_veto_ucb,
        ),
        seed=7,
        tie_policy="stable_expert_index",
    )


def _run_ae_utility_consensus_v2_direct(*, delta_thresholds=None, consensus_thresholds=None):
    sample_domains, expert_domains, true_nelbo, meta, ae_scores = _fake_payload_ae_first()
    embeddings = np.stack([np.asarray([float(i), 0.0]) for i in range(len(sample_domains))])
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=10, expert_domains=expert_domains)
    test_idx = np.where(sample_domains == 10)[0]
    train_idx = np.where(sample_domains != 10)[0]
    return auc.run_ae_utility_calibrator_methods_for_fold(
        embeddings=embeddings,
        sample_domains=sample_domains,
        expert_domains=expert_domains,
        train_idx=train_idx,
        test_idx=test_idx,
        fold=fold,
        true_nelbo=true_nelbo,
        true_eval=fold.slice_nelbo(true_nelbo, test_idx),
        global_eval=true_nelbo[test_idx],
        metadata_similarity=meta,
        metadata_similarity_eval=meta[test_idx][:, list(fold.candidate_col_indices)],
        ae_scores=ae_scores,
        cfg=_ae_utility_consensus_v2_cfg(
            delta_thresholds=delta_thresholds,
            consensus_thresholds=consensus_thresholds,
        ),
        seed=7,
        tie_policy="stable_expert_index",
    )


def _safe_v2_residual_config(**overrides) -> ResidualRoutingConfig:
    values = {
        "enabled": True,
        "residual_policy_version": "metadata_residual_safe_override_v2",
        "models": ("ridge",),
        "thresholds": (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, float("inf")),
        "feature_sets": ("minimal", "latent", "calibrated"),
        "adoption_feature_sets": ("minimal", "latent"),
        "diagnostic_feature_sets": ("calibrated",),
        "allow_calibrated_adoption": False,
        "harmful_override_max": 0.05,
        "gap_regression_max": 2.0,
        "catastrophic_top1_floor": -0.05,
        "selection_metric": "validation_safe_gap_then_top1",
        "unconstrained_reference_method": "linear_regressor",
        "ridge_l2": 1.0e-4,
    }
    values.update(overrides)
    return ResidualRoutingConfig(**values)


def _selection_row(
    *,
    method: str,
    query_domain: int,
    sample_index: int,
    selected_expert: int,
    oracle_expert: int,
    oracle_gap_pct: float,
    top1_hit: int,
) -> dict:
    return {
        "protocol_version": "learned_utility_loqdo_candidate_exclusion_v2",
        "method": method,
        "query_domain": query_domain,
        "fold_query_domain": query_domain,
        "candidate_experts": "100|200",
        "n_candidate_experts": 2,
        "target_expert_excluded": 1,
        "method_role": "learned" if method != "metadata_routing" else "baseline",
        "adoption_eligible": 1,
        "diagnostic_only": 0,
        "routing_uses_eval_nelbo": 0,
        "routing_uses_eval_domain_statistics": 0,
        "selected_expert": selected_expert,
        "candidate_oracle_expert": oracle_expert,
        "oracle_expert": oracle_expert,
        "sample_index": sample_index,
        "top1_oracle_hit": top1_hit,
        "selected_rank": 1.0 if top1_hit else 2.0,
        "oracle_gap": oracle_gap_pct,
        "oracle_gap_pct": oracle_gap_pct,
        "spearman": 1.0,
        "pairwise_auc": 1.0,
        "selected_nelbo": 1.0 + oracle_gap_pct,
        "candidate_oracle_nelbo": 1.0,
    }


def test_residual_target_uses_normalized_metadata_relative_utility() -> None:
    expert_domains = [40, 100, 200, 400]
    sample_domains = np.asarray([100, 100, 200, 200], dtype=np.int64)
    embeddings = np.asarray(
        [[1.0, 0.0], [1.1, 0.0], [2.0, 0.0], [2.1, 0.0]],
        dtype=np.float64,
    )
    true_nelbo = np.asarray(
        [
            [9.0, 1.0, 4.0, 2.0],
            [9.0, 1.0, 5.0, 10.0],
            [9.0, 7.0, 1.0, 3.0],
            [9.0, 8.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    metadata_similarity = np.asarray(
        [
            [0.0, 0.1, 1.0, 0.2],
            [0.0, 0.1, 1.0, 0.2],
            [0.0, 1.0, 0.1, 0.2],
            [0.0, 1.0, 0.1, 0.2],
        ],
        dtype=np.float64,
    )
    context = _feature_context(
        feature_set="minimal",
        embeddings=embeddings,
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        stats_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
    )

    x, y, q = _build_residual_training_rows(
        embeddings=embeddings,
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        metadata_similarity=metadata_similarity,
        outer_heldout_domain=40,
        train_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        context=context,
    )

    assert x.shape[0] == y.shape[0] == q.shape[0]
    assert np.isclose(y[0], 0.0)
    assert np.isclose(y[1], (4.0 - 2.0) / 4.0)
    assert np.isclose(y[2], 0.0)
    assert np.isclose(y[3], (5.0 - 10.0) / 5.0)


def test_tau_inf_fallback_selects_metadata_indices() -> None:
    raw_scores = np.asarray([[10.0, -1.0], [-5.0, 3.0]], dtype=np.float64)
    meta_idx = np.asarray([1, 0], dtype=np.int64)
    assert _selected_from_residual(raw_scores, meta_idx, tau=float("inf")).tolist() == [1, 0]
    assert _selected_from_residual(raw_scores, meta_idx, tau=0.0).tolist() == [0, 1]


def test_safe_v2_harmful_override_veto_rejects_seed43_like_pattern() -> None:
    baseline_rows = [
        _selection_row(
            method="metadata_routing",
            query_domain=40,
            sample_index=0,
            selected_expert=100,
            oracle_expert=100,
            oracle_gap_pct=10.0,
            top1_hit=1,
        )
    ]
    candidate_rows = [
        _selection_row(
            method="metadata_residual_thresholded_safe_v2",
            query_domain=40,
            sample_index=0,
            selected_expert=200,
            oracle_expert=100,
            oracle_gap_pct=5.0,
            top1_hit=0,
        )
    ]
    report = _safe_v2_validation_report(
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
        override_rows=[
            {
                "fold_query_domain": 40,
                "override_rate": 1.0,
                "utility_improving_override_rate": 0.053,
                "harmful_override_rate": 0.947,
            }
        ],
        method="metadata_residual_thresholded_safe_v2",
        residual_cfg=_safe_v2_residual_config(),
    )

    assert report["gap_pct_reduction"] > 0.0
    assert report["safety_pass"] is False
    assert report["max_harmful_override_rate"] == 0.947
    assert "harmful_override" in report["domains_failed_gate"]


def test_safe_v2_zero_override_gate_allows_exact_metadata_match() -> None:
    baseline_rows = [
        _selection_row(
            method="metadata_routing",
            query_domain=40,
            sample_index=0,
            selected_expert=100,
            oracle_expert=100,
            oracle_gap_pct=0.0,
            top1_hit=1,
        )
    ]
    candidate_rows = [
        _selection_row(
            method="metadata_residual_thresholded_safe_v2",
            query_domain=40,
            sample_index=0,
            selected_expert=100,
            oracle_expert=100,
            oracle_gap_pct=0.0,
            top1_hit=1,
        )
    ]
    report = _safe_v2_validation_report(
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
        override_rows=[
            {
                "fold_query_domain": 40,
                "override_rate": 0.0,
                "utility_improving_override_rate": 0.0,
                "harmful_override_rate": 0.0,
            }
        ],
        method="metadata_residual_thresholded_safe_v2",
        residual_cfg=_safe_v2_residual_config(),
    )

    assert report["safety_pass"] is True
    assert report["domains_failed_gate"] == ""


def test_safe_v2_calibrated_excluded_from_adoption_by_default() -> None:
    cfg = _safe_v2_residual_config(
        adoption_feature_sets=("minimal", "latent", "calibrated"),
        allow_calibrated_adoption=False,
    )

    assert _safe_v2_adoption_feature_sets(cfg) == ("minimal", "latent")


def test_residual_artifacts_and_single_inner_selected_adoption_candidate(tmp_path, monkeypatch) -> None:
    def fake_score(**kwargs):
        _ = kwargs
        return _fake_scored_payload()

    monkeypatch.setattr(lu, "_score_experts_batched", fake_score)
    results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={"expert_40": "unused", "expert_100": "unused", "expert_200": "unused"},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg=_residual_cfg(),
        reports_dir=tmp_path,
    )

    assert results["artifacts"]["residual_raw"] == "residual_routing_raw.csv"
    assert (tmp_path / "residual_routing_raw.csv").exists()
    assert (tmp_path / "residual_routing_domain_breakdown.csv").exists()
    assert (tmp_path / "residual_routing_override_diagnostics.csv").exists()
    assert (tmp_path / "residual_routing_policy_audit.md").exists()

    method_summary = _read_csv(tmp_path / "learned_utility_method_summary.csv")
    residual_variants = {
        row["method"]: row
        for row in method_summary
        if row["method"] in {"metadata_residual_thresholded", "metadata_residual_group_robust"}
    }
    assert set(residual_variants) == {"metadata_residual_thresholded", "metadata_residual_group_robust"}
    selected_variants = [
        row
        for row in residual_variants.values()
        if int(row["selected_by_inner_validation"]) == 1 and int(row["diagnostic_only"]) == 1
    ]
    assert len(selected_variants) == 1
    assert all(int(row["adoption_eligible"]) == 0 for row in residual_variants.values())

    selected_method = {row["method"]: row for row in method_summary}["metadata_residual_inner_selected"]
    assert int(selected_method["selected_by_inner_validation"]) == 1
    assert int(selected_method["adoption_eligible"]) == 1
    assert int(selected_method["diagnostic_only"]) == 0
    assert selected_method["decision_policy_version"] == "sign_ci_v2"
    assert selected_method["residual_policy_version"] == "metadata_residual_v1"

    raw_rows = _read_csv(tmp_path / "residual_routing_raw.csv")
    assert raw_rows
    assert all(row["residual_target_scale"] == "delta_u_pct" for row in raw_rows)
    assert all(row["spearman_score_source"] == "raw_residual_pre_threshold" for row in raw_rows)


def test_safe_v2_writes_isolated_artifacts_and_required_policy_fields(tmp_path, monkeypatch) -> None:
    def fake_score(**kwargs):
        _ = kwargs
        return _fake_scored_payload()

    monkeypatch.setattr(lu, "_score_experts_batched", fake_score)
    results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={"expert_40": "unused", "expert_100": "unused", "expert_200": "unused"},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg=_safe_v2_cfg(),
        reports_dir=tmp_path,
    )

    assert results["artifacts"]["residual_raw"] == "residual_safe_v2_raw.csv"
    assert results["artifacts"]["residual_decision_table"] == "residual_safe_v2_decision_table.csv"
    assert results["artifacts"]["residual_selection_policy_audit"] == "residual_safe_v2_selection_policy_audit.md"
    assert (tmp_path / "residual_safe_v2_override_diagnostics.csv").exists()

    raw_rows = _read_csv(tmp_path / "residual_safe_v2_raw.csv")
    assert raw_rows
    required = {
        "decision_policy_version",
        "residual_policy_version",
        "threshold_selection_policy",
        "feature_set",
        "selected_tau",
        "adoption_eligible",
        "diagnostic_only",
        "selected_by_inner_validation",
        "harmful_override_max",
        "allow_calibrated_adoption",
        "fallback_used",
    }
    assert required.issubset(set(raw_rows[0]))
    selected_rows = [
        row
        for row in _read_csv(tmp_path / "residual_safe_v2_policy_audit.csv")
        if int(row["selected_by_inner_validation"]) == 1
    ]
    assert selected_rows
    assert all(row["feature_set"] != "calibrated" for row in selected_rows)


def test_source_inner_self_expert_exclusion() -> None:
    expert_domains = [40, 100, 200, 400]
    outer_heldout = 40
    for pseudo_query_domain in [100, 200, 400]:
        fold = FoldCandidateSet.for_heldout_domain(
            heldout_domain=outer_heldout,
            expert_domains=expert_domains,
            excluded_domains=[pseudo_query_domain],
        )
        assert pseudo_query_domain not in fold.candidate_expert_domains
        assert outer_heldout not in fold.candidate_expert_domains


def test_ae_zscore_uses_source_val_stats_only(monkeypatch) -> None:
    def fake_score(**kwargs):
        _ = kwargs
        return np.asarray([1.0, 3.0], dtype=np.float64)

    monkeypatch.setattr(sfa, "_score_autoencoder", fake_score)
    scores = sfa.build_autoencoder_score_matrices(
        embeddings=np.zeros((2, 2), dtype=np.float64),
        expert_domains=[40],
        autoencoder_artifacts={
            "checkpoints": {"40": "unused.pt"},
            "provenance": {
                "domains": {
                    "40": {
                        "checkpoint": "unused.pt",
                        "source_val_reconstruction_mse": 1.0,
                        "source_val_reconstruction_std": 2.0,
                        "input_dim": 2,
                        "autoencoder_config": {"hidden_dim": 4, "latent_dim": 2},
                    }
                }
            },
        },
        cfg=lu._parse_learned_utility_config(_support_free_ae_cfg()).autoencoder,
    )

    assert np.allclose(scores.zscore_matrix[:, 0], [0.0, 1.0])
    assert scores.quality_rows[0]["source_val_reconstruction_mse_by_domain"] == 1.0
    assert scores.quality_rows[0]["source_val_reconstruction_std_by_domain"] == 2.0


def test_ae_train_val_query_overlap_audit_zero(tmp_path: Path) -> None:
    def write_cache(path: Path, prefix: str) -> None:
        torch.save(
            {
                "embeddings": torch.zeros((2, 2)),
                "metadata": [
                    {"magnification": 40, "sample_id": f"{prefix}_0"},
                    {"magnification": 100, "sample_id": f"{prefix}_1"},
                ],
            },
            path,
        )

    train_cache = tmp_path / "train.pt"
    val_cache = tmp_path / "val.pt"
    test_cache = tmp_path / "test.pt"
    write_cache(train_cache, "train")
    write_cache(val_cache, "val")
    write_cache(test_cache, "test")

    audit = build_support_free_ae_overlap_audit(
        train_cache=train_cache,
        val_cache=val_cache,
        routing_cache=test_cache,
    )
    assert audit["ae_train_query_overlap_count"] == 0
    assert audit["ae_val_query_overlap_count"] == 0
    assert audit["ae_train_cache_hash"]
    assert audit["routing_eval_cache_hash"] == audit["routing_query_cache_hash"]


def test_target_ae_excluded_from_loqdo_candidates() -> None:
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=40, expert_domains=[40, 100, 200])
    outputs = sfa.run_autoencoder_proxy_methods_for_fold(
        sample_domains=np.asarray([40, 40], dtype=np.int64),
        expert_domains=[40, 100, 200],
        test_idx=np.asarray([0, 1], dtype=np.int64),
        fold=fold,
        true_eval=np.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64),
        global_eval=np.asarray([[0.5, 2.0, 1.0], [0.5, 1.0, 2.0]], dtype=np.float64),
        metadata_similarity_eval=np.asarray([[0.8, 0.2], [0.7, 0.3]], dtype=np.float64),
        ae_zscore_matrix=np.asarray([[9.0, 0.1, 0.2], [9.0, 0.2, 0.1]], dtype=np.float64),
        ae_raw_mse_matrix=np.asarray([[9.0, 0.1, 0.2], [9.0, 0.2, 0.1]], dtype=np.float64),
        margin_threshold=0.0,
        tie_policy="stable_expert_index",
    )

    assert outputs.sample_rows
    assert all(int(row["selected_expert"]) != 40 for row in outputs.sample_rows)
    assert all(int(row["expert_domain"]) != 40 for row in outputs.proxy_diag_rows)


def test_ae_argmin_zscore_diagnostic_rows() -> None:
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=40, expert_domains=[40, 100, 200])
    outputs = sfa.run_autoencoder_proxy_methods_for_fold(
        sample_domains=np.asarray([40], dtype=np.int64),
        expert_domains=[40, 100, 200],
        test_idx=np.asarray([0], dtype=np.int64),
        fold=fold,
        true_eval=np.asarray([[2.0, 1.0]], dtype=np.float64),
        global_eval=np.asarray([[0.5, 2.0, 1.0]], dtype=np.float64),
        metadata_similarity_eval=np.asarray([[0.8, 0.2]], dtype=np.float64),
        ae_zscore_matrix=np.asarray([[9.0, 0.2, 0.1]], dtype=np.float64),
        ae_raw_mse_matrix=np.asarray([[9.0, 0.2, 0.1]], dtype=np.float64),
        margin_threshold=0.0,
        tie_policy="stable_expert_index",
    )

    methods = {row["method"] for row in outputs.sample_rows}
    assert {"ae_argmin_zscore", "ae_argmin_margin_gated"}.issubset(methods)
    ae_rows = [row for row in outputs.sample_rows if row["method"] == "ae_argmin_zscore"]
    assert ae_rows[0]["method_role"] == "diagnostic"
    assert int(ae_rows[0]["diagnostic_only"]) == 1


def test_source_prior_fallback_reported_as_named_baseline() -> None:
    outputs = _run_ae_first_direct(thresholds=["__inf__"])
    rows = [row for row in outputs.sample_rows if row["method"] == "source_prior_fallback"]
    assert rows
    assert all(row["method_role"] == "baseline" for row in rows)
    assert all(int(row["adoption_eligible"]) == 1 for row in rows)
    assert all(int(row["routing_uses_query_features"]) == 0 for row in rows)


def test_ae_first_excludes_target_ae_and_expert() -> None:
    outputs = _run_ae_first_direct(thresholds=[0.0, "__inf__"])
    rows = [row for row in outputs.sample_rows if row["method"] == "ae_first_margin_gated_v1"]
    assert rows
    assert all(int(row["selected_expert"]) != 10 for row in rows)
    assert all(int(row["target_ae_excluded"]) == 1 for row in rows)
    assert all("10" not in row["candidate_experts"].split("|") for row in rows)


def test_ae_first_source_inner_self_exclusion() -> None:
    outputs = _run_ae_first_direct(thresholds=[0.0, "__inf__"])
    validation_rows = outputs.source_inner_validation_rows
    assert validation_rows
    for row in validation_rows:
        candidates = {int(v) for v in row["candidate_experts"].split("|") if v}
        assert int(row["pseudo_query_domain"]) not in candidates
        assert int(row["fold_query_domain"]) not in candidates
        assert int(row["source_inner_self_ae_excluded"]) == 1
        assert int(row["source_inner_self_expert_excluded"]) == 1


def test_ae_first_threshold_selection_source_only() -> None:
    outputs = _run_ae_first_direct(thresholds=[0.0, "__inf__"])
    validation_rows = outputs.source_inner_validation_rows
    assert validation_rows
    assert all(row["selection_source"] == "source_inner_only" for row in outputs.sample_rows)
    assert all(row["threshold_selection_policy"] == "source_inner_risk_gated_metadata_gain" for row in validation_rows)
    assert all(int(row["heldout_target_domain_excluded"]) == 1 for row in validation_rows)


def test_ae_first_threshold_objective_risk_gated_metadata_gain() -> None:
    outputs = _run_ae_first_direct(thresholds=[0.0, "__inf__"])
    policy = outputs.policy_audit_rows[0]
    assert policy["threshold_selection_policy"] == "source_inner_risk_gated_metadata_gain"
    assert policy["selected_tau_margin"] == "0"
    assert float(policy["metadata_relative_gain"]) > 0.0
    assert float(policy["source_prior_relative_gain"]) > 0.0


def test_ae_first_no_metadata_anchor() -> None:
    outputs = _run_ae_first_direct(thresholds=[0.0, "__inf__"])
    rows = [row for row in outputs.sample_rows if row["method"] == "ae_first_margin_gated_v1"]
    assert rows
    assert all(row["metadata_role"] == "auxiliary_only" for row in rows)
    assert any(int(row["selected_expert"]) != int(row["metadata_selected_expert"]) for row in rows)


def test_metadata_auxiliary_fields_do_not_change_ae_first_selection() -> None:
    sample_domains, _expert_domains, _true_nelbo, meta, _ae_scores = _fake_payload_ae_first()
    changed_meta = np.zeros_like(meta)
    changed_meta[:, 3] = 1.0
    base = _run_ae_first_direct(thresholds=[0.0], metadata_similarity=meta)
    changed = _run_ae_first_direct(thresholds=[0.0], metadata_similarity=changed_meta)
    base_selected = [
        int(row["selected_expert"])
        for row in base.sample_rows
        if row["method"] == "ae_first_margin_gated_v1"
    ]
    changed_selected = [
        int(row["selected_expert"])
        for row in changed.sample_rows
        if row["method"] == "ae_first_margin_gated_v1"
    ]
    assert sample_domains.tolist()
    assert base_selected == changed_selected


def test_ae_first_tau_zero_matches_ae_argmin_zscore() -> None:
    outputs = _run_ae_first_direct(thresholds=[0.0])
    rows = [row for row in outputs.sample_rows if row["method"] == "ae_first_margin_gated_v1"]
    assert rows
    assert all(int(row["selected_expert"]) == int(row["ae_best_expert"]) for row in rows)
    assert all(int(row["ae_selected_by_gate"]) == 1 for row in rows)


def test_ae_first_tau_inf_matches_source_prior_fallback() -> None:
    outputs = _run_ae_first_direct(thresholds=["__inf__"])
    by_method = {}
    for row in outputs.sample_rows:
        by_method.setdefault(row["method"], []).append(row)
    assert by_method["ae_first_margin_gated_v1"]
    for ae_row, prior_row in zip(by_method["ae_first_margin_gated_v1"], by_method["source_prior_fallback"]):
        assert ae_row["selected_expert"] == prior_row["selected_expert"]
        assert ae_row["selected_nelbo"] == prior_row["selected_nelbo"]
        assert ae_row["oracle_gap"] == prior_row["oracle_gap"]


def test_ae_first_inf_threshold_parsed_consistently() -> None:
    cfg = _ae_first_cfg(thresholds=[0.0, "__inf__"])
    assert np.isinf(cfg.margin_thresholds[-1])


def test_ae_first_reports_coverage_and_fallback_rate() -> None:
    outputs = _run_ae_first_direct(thresholds=[0.0, "__inf__"])
    policy = outputs.policy_audit_rows[0]
    assert "ae_coverage_rate" in policy
    assert "fallback_rate" in policy
    assert np.isclose(float(policy["ae_coverage_rate"]) + float(policy["fallback_rate"]), 1.0)


def test_ae_first_reports_metadata_and_source_prior_relative_gains() -> None:
    outputs = _run_ae_first_direct(thresholds=[0.0, "__inf__"])
    rows = [row for row in outputs.sample_rows if row["method"] == "ae_first_margin_gated_v1"]
    assert rows
    assert {"metadata_relative_gain", "source_prior_relative_gain"}.issubset(rows[0])


def test_ae_first_reports_harmful_improving_against_both_baselines() -> None:
    outputs = _run_ae_first_direct(thresholds=[0.0, "__inf__"])
    rows = [row for row in outputs.sample_rows if row["method"] == "ae_first_margin_gated_v1"]
    required = {
        "harmful_vs_metadata",
        "improving_vs_metadata",
        "harmful_vs_source_prior",
        "improving_vs_source_prior",
    }
    assert required.issubset(rows[0])


def test_ae_first_reports_method_level_decomposition() -> None:
    outputs = _run_ae_first_direct(thresholds=[0.0, "__inf__"])
    policy = outputs.policy_audit_rows[0]
    required = {
        "overall_method_nelbo",
        "fallback_only_nelbo",
        "ae_selected_subset_nelbo",
        "source_prior_on_ae_selected_subset_nelbo",
        "metadata_on_ae_selected_subset_nelbo",
        "oracle_on_ae_selected_subset_nelbo",
    }
    assert required.issubset(policy)


def test_ae_first_reports_oracle_rank_of_ae_best() -> None:
    outputs = _run_ae_first_direct(thresholds=[0.0, "__inf__"])
    policy = outputs.policy_audit_rows[0]
    assert "p_oracle_rank_of_ae_best_eq_1" in policy
    assert "p_oracle_rank_of_ae_best_leq_2" in policy
    assert "mean_oracle_rank_of_ae_best" in policy
    assert all("oracle_rank_of_ae_best" in row for row in outputs.raw_rows)


def test_ae_zscore_uses_global_source_only_sigma_floor(monkeypatch) -> None:
    calls = []

    def fake_score(**kwargs):
        calls.append(kwargs)
        return np.asarray([1.0, 3.0], dtype=np.float64)

    monkeypatch.setattr(sfa, "_score_autoencoder", fake_score)
    cfg_dict = _support_free_ae_cfg()
    cfg_dict["autoencoder_proxy"]["ae_first_routing"] = {
        "enabled": True,
        "margin_thresholds": [0.0, "__inf__"],
        "ae_z_sigma_floor_quantile": 0.5,
    }
    cfg = lu._parse_learned_utility_config(cfg_dict).autoencoder
    scores = sfa.build_autoencoder_score_matrices(
        embeddings=np.zeros((2, 2), dtype=np.float64),
        expert_domains=[40, 100],
        autoencoder_artifacts={
            "checkpoints": {"40": "unused.pt", "100": "unused.pt"},
            "provenance": {
                "domains": {
                    "40": {
                        "checkpoint": "unused.pt",
                        "source_val_reconstruction_mse": 1.0,
                        "source_val_reconstruction_std": 0.01,
                        "input_dim": 2,
                        "autoencoder_config": {"hidden_dim": 4, "latent_dim": 2},
                    },
                    "100": {
                        "checkpoint": "unused.pt",
                        "source_val_reconstruction_mse": 1.0,
                        "source_val_reconstruction_std": 1.0,
                        "input_dim": 2,
                        "autoencoder_config": {"hidden_dim": 4, "latent_dim": 2},
                    },
                }
            },
        },
        cfg=cfg,
    )
    floor = float(scores.quality_rows[0]["ae_z_sigma_floor"])
    assert floor > 0.01
    assert scores.quality_rows[0]["ae_z_sigma_floor_applied"] == 1
    assert scores.quality_rows[1]["ae_z_sigma_floor_applied"] == 0


def test_ae_first_margin_usefulness_diagnostics_present() -> None:
    outputs = _run_ae_first_direct(thresholds=[0.0, "__inf__"])
    assert outputs.margin_bin_rows
    assert {"margin_bin", "harmful_vs_metadata_rate", "mean_oracle_gap_pct", "top1_oracle_hit"}.issubset(
        outputs.margin_bin_rows[0]
    )


def test_ae_first_macro_by_domain_is_primary_summary() -> None:
    outputs = _run_ae_first_direct(thresholds=[0.0, "__inf__"])
    policy = outputs.policy_audit_rows[0]
    assert policy["primary_aggregation"] == "macro_by_domain"
    assert policy["aggregation_unit"] == "seed_x_heldout_domain_x_query_domain"


def test_ae_first_tiny_capped_smoke_run(tmp_path, monkeypatch) -> None:
    sample_domains, expert_domains, true_nelbo, _meta, ae_scores = _fake_payload_ae_first()

    def fake_score(**kwargs):
        _ = kwargs
        embeddings = np.stack([np.asarray([float(i), 0.0]) for i in range(len(sample_domains))])
        metadata = [{"magnification": int(domain), "sample_id": f"s{i}"} for i, domain in enumerate(sample_domains)]
        return embeddings, sample_domains, true_nelbo, expert_domains, metadata

    def fake_ae_scores(**kwargs):
        _ = kwargs
        return ae_scores

    cfg = _support_free_ae_cfg()
    cfg["autoencoder_proxy"]["ae_first_routing"] = {
        "enabled": True,
        "primary_method": "ae_first_margin_gated_v1",
        "fallback_baseline": "source_prior_fallback",
        "margin_thresholds": [0.0, "__inf__"],
        "metadata_auxiliary_features": True,
        "ae_z_sigma_floor_mode": "global_source_val_std_quantile",
        "ae_z_sigma_floor_quantile": 0.05,
    }
    monkeypatch.setattr(lu, "_score_experts_batched", fake_score)
    monkeypatch.setattr(lu, "build_autoencoder_score_matrices", fake_ae_scores)
    results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={f"expert_{d}": "unused" for d in expert_domains},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg=cfg,
        reports_dir=tmp_path,
        autoencoder_artifacts={"dummy": True},
    )

    assert "ae_first_margin_gated_v1" in results["metrics_by_method"]
    assert "source_prior_fallback" in results["metrics_by_method"]
    assert results["artifacts"]["ae_first_raw"] == "ae_first_raw.csv"
    assert (tmp_path / "ae_first_policy_audit.csv").exists()


def test_ae_utility_calibrator_primary_is_metadata_free() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    rows = [row for row in outputs.sample_rows if row["method"] == "ae_utility_calibrated_safe_override_v1"]
    assert rows
    assert all(row["metadata_role"] == "not_used" for row in rows)
    assert outputs.selected_feature_rows[0]["selected_feature_set"] in {"ae_core", "ae_quality"}


def test_ae_metadata_calibrator_reported_as_hybrid_method() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    rows = [row for row in outputs.sample_rows if row["method"] == "ae_metadata_utility_calibrated_safe_override_v1"]
    assert rows
    assert all(row["metadata_role"] == "hybrid_auxiliary_feature" for row in rows)


def test_ae_combined_calibrator_reported_as_hybrid_method() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    rows = [row for row in outputs.sample_rows if row["method"] == "ae_combined_utility_calibrated_safe_override_v1"]
    assert rows
    assert all(row["method_kind"] == "hybrid_combined" for row in rows)


def test_ae_utility_calibrator_model_types_defined() -> None:
    cfg = _ae_utility_calibrator_cfg()
    assert cfg.model_types == ("ridge_delta",)
    assert cfg.primary_model_type == "ridge_delta"
    assert cfg.diagnostic_model_types == ("pairwise_ranker",)


def test_ae_utility_calibrator_pairwise_ranker_diagnostic_only() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    rows = [row for row in outputs.sample_rows if row["method"] == "ae_utility_pairwise_ranker_diagnostic_v1"]
    assert rows
    assert all(int(row["diagnostic_only"]) == 1 for row in rows)
    assert all(int(row["adoption_eligible"]) == 0 for row in rows)


def test_ae_utility_calibrator_margin_threshold_semantics() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    rows = [row for row in outputs.sample_rows if row["method"] == "ae_utility_calibrated_safe_override_v1"]
    assert rows
    assert {"predicted_override_margin", "selected_margin_threshold", "predicted_delta_best_override"}.issubset(rows[0])


def test_ae_utility_calibrator_override_candidates_exclude_anchor() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    raw_rows = [row for row in outputs.raw_rows if row["method"] == "ae_utility_calibrated_safe_override_v1"]
    assert raw_rows
    assert all(int(row["candidate_expert"]) != int(row["ae_anchor_expert"]) for row in raw_rows)


def test_ae_utility_calibrator_label_sign_convention() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    raw_rows = [row for row in outputs.raw_rows if row["method"] == "ae_utility_calibrated_safe_override_v1"]
    assert raw_rows
    assert any(float(row["true_delta_u_ae_pct"]) < 0.0 for row in raw_rows)
    assert all("true_delta_u_ae_pct" in row for row in raw_rows)


def test_ae_utility_calibrator_inf_threshold_matches_ae_argmin() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    rows = [row for row in outputs.sample_rows if row["method"] == "ae_utility_calibrated_safe_override_v1"]
    assert rows
    for row in rows:
        assert int(row["selected_expert"]) == int(row["ae_anchor_expert"])
        assert int(row["override_accepted"]) == 0


def test_ae_utility_calibrator_excludes_target_ae_and_expert() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    assert outputs.policy_audit_rows
    for row in outputs.policy_audit_rows:
        assert int(row["excluded_target_ae"]) == 1
        assert int(row["excluded_target_cvae"]) == 1
        assert int(row["heldout_target_nelbo_used_for_selection"]) == 0


def test_ae_utility_calibrator_source_inner_self_exclusion() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    assert outputs.source_inner_validation_rows
    for row in outputs.source_inner_validation_rows:
        assert int(row["excluded_pseudo_query_ae"]) == 1
        assert int(row["excluded_pseudo_query_cvae"]) == 1


def test_ae_utility_calibrator_threshold_selection_source_only() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    assert outputs.source_inner_validation_rows
    assert all(int(row["heldout_target_nelbo_used_for_selection"]) == 0 for row in outputs.source_inner_validation_rows)


def test_raw_predicted_delta_spearman_non_anchor() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    policy = outputs.policy_audit_rows[0]
    assert "raw_predicted_delta_spearman_non_anchor" in policy


def test_raw_predicted_delta_spearman_with_anchor() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    policy = outputs.policy_audit_rows[0]
    assert "raw_predicted_delta_spearman_with_anchor" in policy


def test_selected_override_precision_nan_when_no_overrides() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    precision = [
        row for row in outputs.override_precision_rows
        if row["method"] == "ae_utility_calibrated_safe_override_v1"
    ][0]
    assert float(precision["active_override_rate"]) == 0.0
    assert np.isnan(float(precision["selected_override_precision"]))


def test_ae_utility_calibrator_reports_oracle_headroom() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    assert outputs.oracle_headroom_rows
    assert {"oracle_headroom_vs_ae_argmin", "ae_argmin_already_oracle", "oracle_best_expert"}.issubset(
        outputs.oracle_headroom_rows[0]
    )


def test_ae_utility_calibrator_reports_override_capture_rate() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    assert outputs.override_precision_rows
    assert "override_capture_rate" in outputs.override_precision_rows[0]


def test_ae_utility_calibrator_policy_audit_provenance_fields() -> None:
    outputs = _run_ae_utility_calibrator_direct(delta_thresholds=["__inf__"])
    policy = outputs.policy_audit_rows[0]
    required = {
        "outer_heldout_domain",
        "source_train_domains",
        "source_inner_pseudo_query_domain",
        "excluded_target_ae",
        "excluded_target_cvae",
        "excluded_pseudo_query_ae",
        "excluded_pseudo_query_cvae",
        "ae_stats_domains_used",
        "threshold_selection_domains_used",
        "model_training_domains_used",
        "heldout_target_nelbo_used_for_selection",
    }
    assert required.issubset(policy)
    assert int(policy["heldout_target_nelbo_used_for_selection"]) == 0


def test_ae_utility_precision_v11_config_parses() -> None:
    cfg = _ae_utility_precision_v11_cfg()
    assert cfg.primary_method == "ae_utility_calibrated_precision_lcb_safe_override_v11"
    assert cfg.selection_mode == "precision_lcb_selected_v11"
    assert cfg.min_strict_improvement_precision == 0.75
    assert cfg.min_strict_improvement_precision_lcb == 0.60
    assert cfg.neutral_override_gap_pct_band == 0.25


def test_ae_utility_precision_v11_primary_is_metadata_free() -> None:
    outputs = _run_ae_utility_precision_v11_direct(delta_thresholds=["__inf__"])
    rows = [
        row for row in outputs.sample_rows
        if row["method"] == "ae_utility_calibrated_precision_lcb_safe_override_v11"
    ]
    assert rows
    assert all(row["metadata_role"] == "not_used" for row in rows)


def test_ae_utility_precision_v11_keeps_v1_behavior_unchanged() -> None:
    outputs = _run_ae_utility_precision_v11_direct(delta_thresholds=["__inf__"])
    v1_rows = [row for row in outputs.sample_rows if row["method"] == "ae_utility_calibrated_safe_override_v1"]
    v11_rows = [
        row for row in outputs.sample_rows
        if row["method"] == "ae_utility_calibrated_precision_lcb_safe_override_v11"
    ]
    assert v1_rows and v11_rows
    assert all(row["method_kind"] == "v1_baseline" for row in v1_rows)
    assert all(int(row["selected_expert"]) == int(row["ae_anchor_expert"]) for row in v1_rows)


def test_ae_utility_precision_v11_reports_strict_and_safe_precision() -> None:
    outputs = _run_ae_utility_precision_v11_direct(delta_thresholds=["__inf__"])
    policy = [
        row for row in outputs.policy_audit_rows
        if row["method"] == "ae_utility_calibrated_precision_lcb_safe_override_v11"
    ][0]
    assert "strict_improvement_precision" in policy
    assert "safe_override_precision" in policy
    assert "strict_improvement_precision_source_inner" in policy
    assert "safe_override_precision_lcb_source_inner" in policy


def test_ae_utility_precision_v11_computes_wilson_precision_lcb() -> None:
    lcb, ucb = auc._wilson_bounds(8, 10)
    assert 0.0 <= lcb < 0.8 < ucb <= 1.0


def test_ae_utility_precision_v11_requires_min_override_count() -> None:
    cfg = _ae_utility_precision_v11_cfg(min_active_override_count=10)
    rows = [
        {
            "active_override_count": 4,
            "active_override_rate": 0.5,
            "improving_override_rate": 1.0,
            "neutral_override_rate": 0.0,
            "harmful_override_rate": 0.0,
            "net_gain_vs_ae_argmin": 1.0,
            "ae_argmin_mean_oracle_gap_pct": 5.0,
            "mean_oracle_gap_pct": 4.0,
        }
    ]
    metrics = auc._precision_lcb_metrics(rows, cfg)
    assert int(metrics["passes_precision_lcb_gates"]) == 0


def test_ae_utility_precision_v11_applies_precision_lcb_before_gap_selection() -> None:
    outputs = _run_ae_utility_precision_v11_direct(delta_thresholds=[0.0, "__inf__"], min_active_override_count=999)
    row = [
        r for r in outputs.selected_feature_rows
        if r["method"] == "ae_utility_calibrated_precision_lcb_safe_override_v11"
    ][0]
    assert str(row["selection_status"]).startswith("fallback_")


def test_ae_utility_precision_v11_macro_gap_lcb_sign_is_positive_for_improvement() -> None:
    cfg = _ae_utility_precision_v11_cfg()
    rows = [
        {"ae_argmin_mean_oracle_gap_pct": 10.0, "mean_oracle_gap_pct": 8.0},
        {"ae_argmin_mean_oracle_gap_pct": 4.0, "mean_oracle_gap_pct": 3.0},
    ]
    assert auc._source_inner_gap_reduction_lcb(rows, cfg) > 0.0


def test_ae_utility_precision_v11_neutral_override_band_classification() -> None:
    true_eval = np.asarray([[1.0, 1.1, 1.102]], dtype=np.float64)
    summary = auc._override_classification_summary(
        selected_idx=np.asarray([2], dtype=np.int64),
        anchor_idx=np.asarray([1], dtype=np.int64),
        true_eval=true_eval,
        neutral_gap_pct_band=0.25,
    )
    assert summary["active_override_count"] == 1
    assert summary["neutral_override_rate"] == 1.0


def test_ae_utility_precision_v11_neutral_overrides_not_counted_as_harmful() -> None:
    true_eval = np.asarray([[1.0, 1.1, 1.102]], dtype=np.float64)
    summary = auc._override_classification_summary(
        selected_idx=np.asarray([2], dtype=np.int64),
        anchor_idx=np.asarray([1], dtype=np.int64),
        true_eval=true_eval,
        neutral_gap_pct_band=0.25,
    )
    assert summary["harmful_override_rate"] == 0.0
    assert summary["safe_override_precision"] == 1.0


def test_ae_utility_precision_v11_rejects_low_lcb_high_raw_precision_config() -> None:
    cfg = _ae_utility_precision_v11_cfg(min_active_override_count=10, min_strict_lcb=0.60)
    rows = [
        {
            "active_override_count": 10,
            "active_override_rate": 0.5,
            "improving_override_rate": 0.8,
            "neutral_override_rate": 0.0,
            "harmful_override_rate": 0.2,
            "net_gain_vs_ae_argmin": 1.0,
            "ae_argmin_mean_oracle_gap_pct": 5.0,
            "mean_oracle_gap_pct": 4.0,
        }
    ]
    metrics = auc._precision_lcb_metrics(rows, cfg)
    assert metrics["strict_improvement_precision_source_inner"] == 0.8
    assert metrics["strict_improvement_precision_lcb_source_inner"] < 0.60
    assert int(metrics["passes_precision_lcb_gates"]) == 0


def test_ae_utility_precision_v11_rejects_worst_pseudo_domain_degradation() -> None:
    cfg = _ae_utility_precision_v11_cfg(min_active_override_count=10, min_strict_lcb=0.0, max_worst_gap=1.0)
    rows = [
        {
            "active_override_count": 10,
            "active_override_rate": 0.5,
            "improving_override_rate": 1.0,
            "neutral_override_rate": 0.0,
            "harmful_override_rate": 0.0,
            "net_gain_vs_ae_argmin": 1.0,
            "ae_argmin_mean_oracle_gap_pct": 5.0,
            "mean_oracle_gap_pct": 7.0,
        }
    ]
    assert int(auc._precision_lcb_metrics(rows, cfg)["passes_precision_lcb_gates"]) == 0


def test_ae_utility_precision_v11_fallback_to_v1_when_no_config_passes() -> None:
    outputs = _run_ae_utility_precision_v11_direct(delta_thresholds=[0.0, "__inf__"], min_active_override_count=999)
    policy = [
        row for row in outputs.policy_audit_rows
        if row["method"] == "ae_utility_calibrated_precision_lcb_safe_override_v11"
    ][0]
    assert str(policy["selection_status"]).startswith("fallback_")
    assert str(policy["fallback_reason"])


def test_ae_utility_precision_v11_selection_uses_no_target_nelbo() -> None:
    outputs = _run_ae_utility_precision_v11_direct(delta_thresholds=["__inf__"])
    rows = [
        row for row in outputs.source_inner_validation_rows
        if row["method"] == "ae_utility_calibrated_precision_lcb_safe_override_v11"
    ]
    assert rows
    assert all(int(row["heldout_target_nelbo_used_for_selection"]) == 0 for row in rows)


def test_ae_utility_precision_v11_heldout_precision_is_report_only() -> None:
    outputs = _run_ae_utility_precision_v11_direct(delta_thresholds=["__inf__"])
    policy = [
        row for row in outputs.policy_audit_rows
        if row["method"] == "ae_utility_calibrated_precision_lcb_safe_override_v11"
    ][0]
    assert int(policy["heldout_precision_report_only"]) == 1


def test_ae_utility_precision_v11_reports_precision_tradeoff() -> None:
    outputs = _run_ae_utility_precision_v11_direct(delta_thresholds=[0.0, "__inf__"])
    rows = [
        row for row in outputs.source_inner_validation_rows
        if row["method"] == "ae_utility_calibrated_precision_lcb_safe_override_v11"
        and row["source_inner_pseudo_query_domain"] == "source_inner_macro"
    ]
    assert rows
    assert "strict_improvement_precision_lcb_source_inner" in rows[0]
    assert "source_inner_macro_gap_reduction_lcb" in rows[0]


def test_ae_utility_precision_v11_inf_threshold_still_matches_ae_argmin() -> None:
    outputs = _run_ae_utility_precision_v11_direct(delta_thresholds=["__inf__"])
    rows = [
        row for row in outputs.sample_rows
        if row["method"] == "ae_utility_calibrated_precision_lcb_safe_override_v11"
    ]
    assert rows
    assert all(int(row["selected_expert"]) == int(row["ae_anchor_expert"]) for row in rows)


def test_ae_utility_precision_v12_config_parses() -> None:
    cfg = _ae_utility_precision_v12_cfg()
    assert cfg.primary_method == "ae_utility_calibrated_precision_lcb_v1_guarded_safe_override_v12"
    assert cfg.selection_mode == "precision_lcb_v1_guarded_v12"
    assert cfg.min_active_override_count == 12
    assert cfg.v1_guard_max_harmful_override_rate_ucb == 0.30


def test_ae_utility_precision_v12_primary_is_metadata_free() -> None:
    outputs = _run_ae_utility_precision_v12_direct(delta_thresholds=["__inf__"])
    rows = [
        row for row in outputs.sample_rows
        if row["method"] == "ae_utility_calibrated_precision_lcb_v1_guarded_safe_override_v12"
    ]
    assert rows
    assert all(row["metadata_role"] == "not_used" for row in rows)


def test_ae_utility_precision_v12_keeps_v1_and_v11_reported() -> None:
    outputs = _run_ae_utility_precision_v12_direct(delta_thresholds=["__inf__"])
    methods = {row["method"] for row in outputs.policy_audit_rows}
    assert "ae_utility_calibrated_safe_override_v1" in methods
    assert "ae_utility_calibrated_precision_lcb_safe_override_v11" in methods
    assert "ae_utility_calibrated_precision_lcb_v1_guarded_safe_override_v12" in methods


def test_ae_utility_precision_v12_uses_v1_as_exact_fallback() -> None:
    outputs = _run_ae_utility_precision_v12_direct(delta_thresholds=[0.0, "__inf__"], min_active_override_count=999)
    v1 = [row for row in outputs.sample_rows if row["method"] == "ae_utility_calibrated_safe_override_v1"]
    v12 = [
        row for row in outputs.sample_rows
        if row["method"] == "ae_utility_calibrated_precision_lcb_v1_guarded_safe_override_v12"
    ]
    assert v1 and v12
    assert all(
        int(a["selected_expert"]) == int(b["selected_expert"])
        for a, b in zip(v1, v12)
    )
    assert any(str(row["selection_status"]).startswith("fallback_to_v1") for row in v12)


def test_ae_utility_precision_v12_gap_delta_vs_v1_sign() -> None:
    cfg = _ae_utility_precision_v12_cfg()
    candidate = [
        {"source_inner_pseudo_query_domain": 1, "mean_oracle_gap_pct": 3.0, "top1_oracle_hit": 0.7, "raw_predicted_delta_spearman_non_anchor": 0.2},
    ]
    v1 = [
        {"source_inner_pseudo_query_domain": 1, "mean_oracle_gap_pct": 5.0, "top1_oracle_hit": 0.7, "raw_predicted_delta_spearman_non_anchor": 0.2},
    ]
    metrics = auc._v1_guard_metrics(candidate, v1, cfg)
    assert metrics["source_inner_gap_delta_vs_v1"] > 0.0


def test_ae_utility_precision_v12_gap_delta_lcb_bootstrap() -> None:
    cfg = _ae_utility_precision_v12_cfg()
    candidate = [
        {"source_inner_pseudo_query_domain": i, "mean_oracle_gap_pct": 3.0, "top1_oracle_hit": 0.7, "raw_predicted_delta_spearman_non_anchor": 0.2}
        for i in range(3)
    ]
    v1 = [
        {"source_inner_pseudo_query_domain": i, "mean_oracle_gap_pct": 5.0, "top1_oracle_hit": 0.7, "raw_predicted_delta_spearman_non_anchor": 0.2}
        for i in range(3)
    ]
    assert auc._v1_guard_metrics(candidate, v1, cfg)["source_inner_gap_delta_vs_v1_lcb"] > 0.0


def test_ae_utility_precision_v12_bootstrap_uses_paired_units() -> None:
    cfg = _ae_utility_precision_v12_cfg()
    candidate = [
        {"source_inner_pseudo_query_domain": "a", "mean_oracle_gap_pct": 4.0, "top1_oracle_hit": 0.7, "raw_predicted_delta_spearman_non_anchor": 0.2},
        {"source_inner_pseudo_query_domain": "b", "mean_oracle_gap_pct": 2.0, "top1_oracle_hit": 0.7, "raw_predicted_delta_spearman_non_anchor": 0.2},
        {"source_inner_pseudo_query_domain": "unpaired", "mean_oracle_gap_pct": 100.0, "top1_oracle_hit": 0.0, "raw_predicted_delta_spearman_non_anchor": -1.0},
    ]
    v1 = [
        {"source_inner_pseudo_query_domain": "a", "mean_oracle_gap_pct": 5.0, "top1_oracle_hit": 0.7, "raw_predicted_delta_spearman_non_anchor": 0.2},
        {"source_inner_pseudo_query_domain": "b", "mean_oracle_gap_pct": 5.0, "top1_oracle_hit": 0.7, "raw_predicted_delta_spearman_non_anchor": 0.2},
    ]
    metrics = auc._v1_guard_metrics(candidate, v1, cfg)
    assert metrics["paired_source_inner_unit_count_vs_v1"] == 2
    assert 1.0 <= metrics["source_inner_gap_delta_vs_v1"] <= 3.0


def test_ae_utility_precision_v12_harm_ucb_gate_feasible_at_min_count() -> None:
    _lcb, ucb = auc._wilson_bounds(0, 12)
    assert ucb <= 0.30


def test_ae_utility_precision_v12_rejects_candidate_with_low_gap_delta_lcb() -> None:
    cfg = _ae_utility_precision_v12_cfg(min_gap_delta_vs_v1_lcb=999.0)
    candidate = [
        {"source_inner_pseudo_query_domain": i, "mean_oracle_gap_pct": 3.0, "top1_oracle_hit": 0.7, "raw_predicted_delta_spearman_non_anchor": 0.2}
        for i in range(3)
    ]
    v1 = [
        {"source_inner_pseudo_query_domain": i, "mean_oracle_gap_pct": 5.0, "top1_oracle_hit": 0.7, "raw_predicted_delta_spearman_non_anchor": 0.2}
        for i in range(3)
    ]
    assert auc._v1_guard_metrics(candidate, v1, cfg)["v1_guard_passed"] == 0


def test_ae_utility_precision_v12_rejects_high_harm_ucb() -> None:
    outputs = _run_ae_utility_precision_v12_direct(
        delta_thresholds=[0.0, "__inf__"],
        min_active_override_count=1,
        min_strict_lcb=0.0,
        max_harm_ucb=0.0,
    )
    policy = [
        row for row in outputs.policy_audit_rows
        if row["method"] == "ae_utility_calibrated_precision_lcb_v1_guarded_safe_override_v12"
    ][0]
    assert str(policy["selection_status"]).startswith("fallback_to_v1")


def test_ae_utility_precision_v12_rejects_worst_pseudo_domain_degradation_vs_v1() -> None:
    cfg = _ae_utility_precision_v12_cfg()
    candidate = [
        {"source_inner_pseudo_query_domain": 1, "mean_oracle_gap_pct": 7.0, "top1_oracle_hit": 0.7, "raw_predicted_delta_spearman_non_anchor": 0.2},
    ]
    v1 = [
        {"source_inner_pseudo_query_domain": 1, "mean_oracle_gap_pct": 5.0, "top1_oracle_hit": 0.7, "raw_predicted_delta_spearman_non_anchor": 0.2},
    ]
    assert auc._v1_guard_metrics(candidate, v1, cfg)["v1_guard_passed"] == 0


def test_ae_utility_precision_v12_tiebreak_prefers_lower_harm_ucb() -> None:
    rows = [
        {"harmful_override_rate_ucb_source_inner": 0.2, "strict_improvement_precision_lcb_source_inner": 0.9, "source_inner_gap_delta_vs_v1_lcb": 0.0, "active_override_rate_source_inner": 0.1, "delta_threshold": 0.0, "margin_threshold": 0.0},
        {"harmful_override_rate_ucb_source_inner": 0.1, "strict_improvement_precision_lcb_source_inner": 0.6, "source_inner_gap_delta_vs_v1_lcb": 0.0, "active_override_rate_source_inner": 0.1, "delta_threshold": 0.0, "margin_threshold": 0.0},
    ]
    chosen = sorted(
        rows,
        key=lambda row: (
            float(row["harmful_override_rate_ucb_source_inner"]),
            -float(row["strict_improvement_precision_lcb_source_inner"]),
            -float(row["source_inner_gap_delta_vs_v1_lcb"]),
            -float(row["active_override_rate_source_inner"]),
            -float(row["delta_threshold"]),
            -float(row["margin_threshold"]),
        ),
    )[0]
    assert chosen["harmful_override_rate_ucb_source_inner"] == 0.1


def test_ae_utility_precision_v12_selection_uses_no_target_nelbo() -> None:
    outputs = _run_ae_utility_precision_v12_direct(delta_thresholds=["__inf__"])
    rows = [
        row for row in outputs.source_inner_validation_rows
        if row["method"] == "ae_utility_calibrated_precision_lcb_v1_guarded_safe_override_v12"
    ]
    assert rows
    assert all(int(row["heldout_target_nelbo_used_for_selection"]) == 0 for row in rows)


def test_ae_utility_precision_v12_heldout_precision_is_report_only() -> None:
    outputs = _run_ae_utility_precision_v12_direct(delta_thresholds=["__inf__"])
    policy = [
        row for row in outputs.policy_audit_rows
        if row["method"] == "ae_utility_calibrated_precision_lcb_v1_guarded_safe_override_v12"
    ][0]
    assert int(policy["heldout_precision_report_only"]) == 1


def test_ae_utility_precision_v12_reports_v1_guard_fields() -> None:
    outputs = _run_ae_utility_precision_v12_direct(delta_thresholds=[0.0, "__inf__"])
    rows = [
        row for row in outputs.source_inner_validation_rows
        if row["method"] == "ae_utility_calibrated_precision_lcb_v1_guarded_safe_override_v12"
        and row["source_inner_pseudo_query_domain"] == "source_inner_macro"
    ]
    assert rows
    for key in [
        "v1_guard_passed",
        "source_inner_gap_delta_vs_v1_lcb",
        "worst_pseudo_domain_gap_degradation_vs_v1_pp",
    ]:
        assert key in rows[0]


def test_ae_utility_precision_v12_reports_active_override_counts() -> None:
    outputs = _run_ae_utility_precision_v12_direct(delta_thresholds=[0.0, "__inf__"])
    policy = [
        row for row in outputs.policy_audit_rows
        if row["method"] == "ae_utility_calibrated_precision_lcb_v1_guarded_safe_override_v12"
    ][0]
    assert "active_override_count_source_inner" in policy
    assert "active_override_count_heldout" in policy


def test_ae_utility_precision_v12_decision_builder_requires_both_datasets_for_cross_dataset_verdict(tmp_path) -> None:
    spec = importlib.util.spec_from_file_location(
        "ae_decision_builder",
        PROJECT_ROOT / "scripts" / "build_ae_utility_calibrator_decision_table.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {
            "dataset": "camelyon17",
            "method": "ae_utility_calibrated_safe_override_v1",
            "top1_oracle_hit": 0.5,
            "raw_spearman": 0.5,
            "mean_oracle_gap_pct": 4.0,
        },
        {
            "dataset": "camelyon17",
            "method": "ae_utility_calibrated_precision_lcb_v1_guarded_safe_override_v12",
            "top1_oracle_hit": 0.5,
            "raw_spearman": 0.5,
            "mean_oracle_gap_pct": 4.0,
        },
        {
            "dataset": "camelyon17",
            "method": "ae_argmin_zscore",
            "top1_oracle_hit": 0.4,
            "raw_spearman": 0.4,
            "mean_oracle_gap_pct": 5.0,
        },
        {
            "dataset": "camelyon17",
            "method": "metadata_routing",
            "top1_oracle_hit": 0.4,
            "raw_spearman": 0.4,
            "mean_oracle_gap_pct": 5.0,
        },
    ]
    _out, summary = module._aggregate(rows, [])
    assert summary["verdicts"]["cross_dataset_local_ae_calibration_verdict"] == "NEEDS EVIDENCE"


def test_ae_utility_harm_veto_v13_config_parses() -> None:
    cfg = _ae_utility_harm_veto_v13_cfg()
    assert cfg.primary_method == "ae_utility_calibrated_v1_harm_veto_safe_override_v13"
    assert cfg.selection_mode == "v1_harm_veto_v13"
    assert cfg.harm_veto_score_model == "logistic_harm_score"
    assert cfg.harm_veto_min_veto_count_source_inner == 6


def test_ae_utility_harm_veto_v13_primary_is_metadata_free() -> None:
    outputs = _run_ae_utility_harm_veto_v13_direct(delta_thresholds=["__inf__"])
    rows = [
        row for row in outputs.sample_rows
        if row["method"] == "ae_utility_calibrated_v1_harm_veto_safe_override_v13"
    ]
    assert rows
    assert all(row["metadata_role"] == "not_used" for row in rows)


def test_ae_utility_harm_veto_v13_keeps_v1_behavior_unchanged() -> None:
    outputs = _run_ae_utility_harm_veto_v13_direct(delta_thresholds=["__inf__"])
    v1 = [row for row in outputs.sample_rows if row["method"] == "ae_utility_calibrated_safe_override_v1"]
    assert v1
    assert all(int(row["selected_expert"]) == int(row["ae_anchor_expert"]) for row in v1)


def test_ae_utility_harm_veto_v13_exact_fallback_matches_v1() -> None:
    outputs = _run_ae_utility_harm_veto_v13_direct(
        delta_thresholds=[0.0, "__inf__"],
        min_active_v1_override_count=999,
    )
    v1 = [row for row in outputs.sample_rows if row["method"] == "ae_utility_calibrated_safe_override_v1"]
    v13 = [
        row for row in outputs.sample_rows
        if row["method"] == "ae_utility_calibrated_v1_harm_veto_safe_override_v13"
    ]
    assert v1 and v13
    assert all(int(a["selected_expert"]) == int(b["selected_expert"]) for a, b in zip(v1, v13))
    assert any(str(row["selection_status"]) == "fallback_to_v1_no_harm_veto_safe_config" for row in v13)


def test_ae_utility_harm_veto_v13_trains_only_on_active_v1_overrides() -> None:
    sample_domains, expert_domains, true_nelbo, meta, ae_scores = _fake_payload_ae_first()
    train_idx = np.where(sample_domains != 10)[0]
    examples = auc._collect_source_inner_harm_examples(
        embeddings=np.stack([np.asarray([float(i), 0.0]) for i in range(len(sample_domains))]),
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        train_idx=train_idx,
        outer_heldout_domain=10,
        excluded_validation_domain=None,
        metadata_similarity=meta,
        ae_scores=ae_scores,
        feature_set="ae_core",
        delta_threshold=0.0,
        margin_threshold=0.0,
        ridge_l2=1.0e-4,
        neutral_gap_pct_band=0.25,
    )
    assert examples["y"].shape[0] <= train_idx.shape[0]


def test_ae_utility_harm_veto_v13_only_scores_active_v1_overrides() -> None:
    outputs = _run_ae_utility_harm_veto_v13_direct(delta_thresholds=[0.0, "__inf__"])
    rows = [
        row for row in outputs.sample_rows
        if row["method"] == "ae_utility_calibrated_v1_harm_veto_safe_override_v13"
    ]
    assert rows
    for row in rows:
        if int(row["active_override"]) == 0:
            assert str(row["harm_score"]) == "nan" or row["harm_score"] == ""


def test_ae_utility_harm_veto_v13_logistic_score_is_bounded() -> None:
    scores = auc._fit_predict_logistic_harm_score(
        train_x=np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64),
        train_y=np.asarray([0, 0, 1, 1], dtype=np.int64),
        eval_x=np.asarray([[-10.0], [10.0]], dtype=np.float64),
        l2=1.0e-4,
    )
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_ae_utility_harm_veto_v13_fallback_when_harm_labels_single_class() -> None:
    scores = auc._fit_predict_logistic_harm_score(
        train_x=np.asarray([[0.0], [1.0]], dtype=np.float64),
        train_y=np.asarray([0, 0], dtype=np.int64),
        eval_x=np.asarray([[0.5]], dtype=np.float64),
        l2=1.0e-4,
    )
    assert np.isnan(scores[0])


def test_ae_utility_harm_veto_v13_fallback_when_too_few_harmful_examples() -> None:
    outputs = _run_ae_utility_harm_veto_v13_direct(
        delta_thresholds=[0.0, "__inf__"],
        min_harmful_count=999,
    )
    policy = [
        row for row in outputs.policy_audit_rows
        if row["method"] == "ae_utility_calibrated_v1_harm_veto_safe_override_v13"
    ][0]
    assert policy["selection_status"] == "fallback_to_v1_no_harm_veto_safe_config"


def test_ae_utility_harm_veto_v13_false_veto_ucb_gate_feasible_at_min_count() -> None:
    _lcb, ucb = auc._wilson_bounds(0, 6)
    assert ucb <= 0.40


def test_ae_utility_harm_veto_v13_reports_harm_label_counts() -> None:
    outputs = _run_ae_utility_harm_veto_v13_direct(delta_thresholds=[0.0, "__inf__"])
    rows = [
        row for row in outputs.source_inner_validation_rows
        if row["method"] == "ae_utility_calibrated_v1_harm_veto_safe_override_v13"
        and row["source_inner_pseudo_query_domain"] == "source_inner_macro"
    ]
    assert rows
    assert "harmful_v1_override_count_source_inner" in rows[0]
    assert "nonharmful_v1_override_count_source_inner" in rows[0]


def test_ae_utility_harm_veto_v13_reports_raw_active_override_counts() -> None:
    outputs = _run_ae_utility_harm_veto_v13_direct(delta_thresholds=[0.0, "__inf__"])
    policy = [
        row for row in outputs.policy_audit_rows
        if row["method"] == "ae_utility_calibrated_v1_harm_veto_safe_override_v13"
    ][0]
    assert "v1_active_override_count_source_inner" in policy
    assert "v13_active_override_count_heldout" in policy


def test_ae_utility_harm_veto_v13_veto_returns_to_ae_argmin() -> None:
    selected, scores, vetoed = auc._apply_harm_veto_policy(
        v1_selected_idx=np.asarray([1, 2]),
        anchor_idx=np.asarray([0, 0]),
        active_sample_positions=np.asarray([0, 1]),
        harm_scores=np.asarray([0.9, 0.1]),
        veto_threshold=0.5,
    )
    assert selected.tolist() == [0, 2]
    assert vetoed.tolist() == [True, False]
    assert scores[0] == 0.9


def test_ae_utility_harm_veto_v13_neutral_veto_not_counted_as_false_veto() -> None:
    true_eval = np.asarray([[1.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    metrics = auc._harm_veto_metrics(
        v1_selected_idx=np.asarray([1, 1]),
        v13_selected_idx=np.asarray([0, 0]),
        anchor_idx=np.asarray([0, 0]),
        true_eval=true_eval,
        neutral_gap_pct_band=0.25,
    )
    assert metrics["vetoed_neutral_count"] == 1
    assert metrics["vetoed_improving_count"] == 1
    assert metrics["false_veto_rate"] == 0.5


def test_ae_utility_harm_veto_v13_retained_gain_is_utility_weighted() -> None:
    true_eval = np.asarray([[2.0, 1.0], [11.0, 1.0]], dtype=np.float64)
    metrics = auc._harm_veto_metrics(
        v1_selected_idx=np.asarray([1, 1]),
        v13_selected_idx=np.asarray([1, 0]),
        anchor_idx=np.asarray([0, 0]),
        true_eval=true_eval,
        neutral_gap_pct_band=0.25,
    )
    assert 0.0 < metrics["retained_v1_override_gain_rate"] < 0.2


def test_ae_utility_harm_veto_v13_selection_uses_no_target_nelbo() -> None:
    outputs = _run_ae_utility_harm_veto_v13_direct(delta_thresholds=["__inf__"])
    rows = [
        row for row in outputs.source_inner_validation_rows
        if row["method"] == "ae_utility_calibrated_v1_harm_veto_safe_override_v13"
    ]
    assert rows
    assert all(int(row["heldout_target_nelbo_used_for_selection"]) == 0 for row in rows)


def test_ae_utility_harm_veto_v13_heldout_precision_is_report_only() -> None:
    outputs = _run_ae_utility_harm_veto_v13_direct(delta_thresholds=["__inf__"])
    policy = [
        row for row in outputs.policy_audit_rows
        if row["method"] == "ae_utility_calibrated_v1_harm_veto_safe_override_v13"
    ][0]
    assert int(policy["heldout_precision_report_only"]) == 1


def test_ae_utility_harm_veto_v13_tiny_capped_smoke_run() -> None:
    outputs = _run_ae_utility_harm_veto_v13_direct(delta_thresholds=[0.0, "__inf__"])
    methods = {row["method"] for row in outputs.policy_audit_rows}
    assert "ae_utility_calibrated_v1_harm_veto_safe_override_v13" in methods


def test_ae_utility_consensus_v2_primary_is_metadata_free() -> None:
    outputs = _run_ae_utility_consensus_v2_direct(delta_thresholds=["__inf__"])
    rows = [row for row in outputs.sample_rows if row["method"] == "ae_utility_calibrated_consensus_safe_override_v2"]
    assert rows
    assert all(row["metadata_role"] == "not_used" for row in rows)
    assert all(row["feature_set"] in {"ae_consensus_core", "ae_consensus_quality"} for row in rows)


def test_ae_utility_consensus_v2_hybrids_are_separate_methods() -> None:
    outputs = _run_ae_utility_consensus_v2_direct(delta_thresholds=["__inf__"])
    methods = {row["method"] for row in outputs.sample_rows}
    assert "ae_metadata_utility_calibrated_consensus_safe_override_v2" in methods
    assert "ae_combined_utility_calibrated_consensus_safe_override_v2" in methods
    hybrid_rows = [
        row for row in outputs.sample_rows
        if row["method"] in {
            "ae_metadata_utility_calibrated_consensus_safe_override_v2",
            "ae_combined_utility_calibrated_consensus_safe_override_v2",
        }
    ]
    assert hybrid_rows
    assert all(row["metadata_role"] == "hybrid_auxiliary_feature" for row in hybrid_rows)


def test_ae_utility_consensus_v2_features_exclude_metadata() -> None:
    cfg = _ae_utility_consensus_v2_cfg()
    assert cfg.feature_sets_primary == ("ae_consensus_core", "ae_consensus_quality")
    assert not any("metadata" in feature_set for feature_set in cfg.feature_sets_primary)


def test_ae_utility_consensus_v2_excludes_target_ae_and_cvae() -> None:
    outputs = _run_ae_utility_consensus_v2_direct(delta_thresholds=["__inf__"])
    assert outputs.policy_audit_rows
    for row in outputs.policy_audit_rows:
        assert int(row["excluded_target_ae"]) == 1
        assert int(row["excluded_target_cvae"]) == 1
        assert int(row["heldout_target_nelbo_used_for_selection"]) == 0


def test_ae_utility_consensus_v2_source_inner_self_exclusion() -> None:
    outputs = _run_ae_utility_consensus_v2_direct(delta_thresholds=["__inf__"])
    assert outputs.source_inner_validation_rows
    for row in outputs.source_inner_validation_rows:
        assert int(row["excluded_pseudo_query_ae"]) == 1
        assert int(row["excluded_pseudo_query_cvae"]) == 1


def test_ae_utility_consensus_v2_ensemble_members_exclude_source_inner_validation_domain() -> None:
    outputs = _run_ae_utility_consensus_v2_direct(delta_thresholds=["__inf__"])
    rows = [
        row for row in outputs.source_inner_validation_rows
        if row["method"] == "ae_utility_calibrated_consensus_safe_override_v2"
    ]
    assert rows
    for row in rows:
        pseudo = str(int(row["source_inner_pseudo_query_domain"]))
        domains = str(row["ensemble_training_domains_used"]).split("|")
        assert int(row["source_inner_validation_domain_excluded_from_ensemble_training"]) == 1
        assert pseudo not in domains


def test_ae_utility_consensus_v2_threshold_selection_source_only() -> None:
    outputs = _run_ae_utility_consensus_v2_direct(delta_thresholds=["__inf__"])
    assert outputs.source_inner_validation_rows
    assert all(int(row["heldout_target_nelbo_used_for_selection"]) == 0 for row in outputs.source_inner_validation_rows)


def test_ae_utility_consensus_v2_source_inner_stability_gates_precede_gap_selection() -> None:
    outputs = _run_ae_utility_consensus_v2_direct(delta_thresholds=["__inf__"])
    rows = [
        row for row in outputs.source_inner_validation_rows
        if row["method"] == "ae_utility_calibrated_consensus_safe_override_v2"
    ]
    assert rows
    assert all(row["threshold_selection_policy"] == "source_inner_stability_then_ae_argmin_gap" for row in rows)


def test_ae_utility_consensus_v2_inf_threshold_matches_ae_argmin() -> None:
    outputs = _run_ae_utility_consensus_v2_direct(delta_thresholds=["__inf__"])
    rows = [row for row in outputs.sample_rows if row["method"] == "ae_utility_calibrated_consensus_safe_override_v2"]
    assert rows
    for row in rows:
        assert int(row["selected_expert"]) == int(row["ae_anchor_expert"])
        assert int(row["override_accepted"]) == 0


def test_ae_utility_consensus_v2_override_candidates_exclude_anchor() -> None:
    outputs = _run_ae_utility_consensus_v2_direct(delta_thresholds=["__inf__"])
    raw_rows = [row for row in outputs.raw_rows if row["method"] == "ae_utility_calibrated_consensus_safe_override_v2"]
    assert raw_rows
    assert all(int(row["candidate_expert"]) != int(row["ae_anchor_expert"]) for row in raw_rows)


def test_ae_utility_consensus_v2_single_override_candidate_margin_behavior() -> None:
    consensus = auc._ConsensusPredictions(
        mean_matrix=np.asarray([[0.0, 0.10]], dtype=np.float64),
        std_matrix=np.asarray([[0.0, 0.01]], dtype=np.float64),
        lower_matrix=np.asarray([[0.0, 0.09]], dtype=np.float64),
        positive_rate_matrix=np.asarray([[0.0, 1.0]], dtype=np.float64),
        n_members_matrix=np.asarray([[2.0, 2.0]], dtype=np.float64),
        n_positive_matrix=np.asarray([[0.0, 2.0]], dtype=np.float64),
        member_labels=("full_source", "leave_domain_20"),
    )
    selected, best, second, lower_best, lower_second, margin, positive = auc._apply_consensus_safe_override_policy(
        consensus=consensus,
        anchor_idx=np.asarray([0], dtype=np.int64),
        delta_threshold=0.05,
        margin_threshold=999.0,
        consensus_threshold=1.0,
    )
    assert int(best[0]) == 1
    assert int(second[0]) == -1
    assert np.isinf(float(margin[0]))
    assert int(selected[0]) == 1
    assert float(lower_best[0]) == 0.09
    assert not np.isfinite(float(lower_second[0]))
    assert float(positive[0]) == 1.0


def test_ae_utility_consensus_v2_lower_confidence_delta_rule() -> None:
    consensus = auc._ConsensusPredictions(
        mean_matrix=np.asarray([[0.0, 0.20, 0.15]], dtype=np.float64),
        std_matrix=np.asarray([[0.0, 0.20, 0.01]], dtype=np.float64),
        lower_matrix=np.asarray([[0.0, 0.00, 0.14]], dtype=np.float64),
        positive_rate_matrix=np.asarray([[0.0, 1.0, 1.0]], dtype=np.float64),
        n_members_matrix=np.ones((1, 3), dtype=np.float64) * 2.0,
        n_positive_matrix=np.ones((1, 3), dtype=np.float64) * 2.0,
        member_labels=("full_source", "leave_domain_20"),
    )
    selected, best, _second, lower_best, _lower_second, _margin, _positive = auc._apply_consensus_safe_override_policy(
        consensus=consensus,
        anchor_idx=np.asarray([0], dtype=np.int64),
        delta_threshold=0.05,
        margin_threshold=0.0,
        consensus_threshold=1.0,
    )
    assert int(best[0]) == 2
    assert int(selected[0]) == 2
    assert float(lower_best[0]) == 0.14


def test_ae_utility_consensus_v2_consensus_threshold_rule() -> None:
    consensus = auc._ConsensusPredictions(
        mean_matrix=np.asarray([[0.0, 0.20]], dtype=np.float64),
        std_matrix=np.asarray([[0.0, 0.01]], dtype=np.float64),
        lower_matrix=np.asarray([[0.0, 0.19]], dtype=np.float64),
        positive_rate_matrix=np.asarray([[0.0, 0.50]], dtype=np.float64),
        n_members_matrix=np.asarray([[2.0, 2.0]], dtype=np.float64),
        n_positive_matrix=np.asarray([[0.0, 1.0]], dtype=np.float64),
        member_labels=("full_source", "leave_domain_20"),
    )
    selected, *_ = auc._apply_consensus_safe_override_policy(
        consensus=consensus,
        anchor_idx=np.asarray([0], dtype=np.int64),
        delta_threshold=0.05,
        margin_threshold=0.0,
        consensus_threshold=0.75,
    )
    assert int(selected[0]) == 0


def test_ae_utility_consensus_v2_reports_candidate_consensus_fields() -> None:
    outputs = _run_ae_utility_consensus_v2_direct(delta_thresholds=["__inf__"])
    row = [row for row in outputs.sample_rows if row["method"] == "ae_utility_calibrated_consensus_safe_override_v2"][0]
    required = {
        "mean_predicted_delta_best",
        "std_predicted_delta_best",
        "lower_confidence_delta_best",
        "positive_consensus_rate_best",
        "mean_predicted_delta_second",
        "lower_confidence_delta_second",
        "predicted_override_margin",
        "n_ensemble_members",
        "n_positive_members",
    }
    assert required.issubset(row)


def test_ae_utility_consensus_v2_reports_coverage_precision_tradeoff() -> None:
    outputs = _run_ae_utility_consensus_v2_direct(delta_thresholds=["__inf__"])
    row = [row for row in outputs.policy_audit_rows if row["method"] == "ae_utility_calibrated_consensus_safe_override_v2"][0]
    required = {
        "active_override_rate",
        "selected_override_precision",
        "net_gain_vs_ae_argmin",
        "captured_oracle_headroom_rate",
    }
    assert required.issubset(row)


def test_ae_utility_consensus_v2_reports_captured_oracle_headroom_rate() -> None:
    outputs = _run_ae_utility_consensus_v2_direct(delta_thresholds=["__inf__"])
    assert outputs.override_precision_rows
    row = [row for row in outputs.override_precision_rows if row["method"] == "ae_utility_calibrated_consensus_safe_override_v2"][0]
    assert "captured_oracle_headroom_rate" in row


def test_ae_utility_consensus_v2_reports_abstention_correctness() -> None:
    outputs = _run_ae_utility_consensus_v2_direct(delta_thresholds=["__inf__"])
    row = [row for row in outputs.override_diagnostic_rows if row["method"] == "ae_utility_calibrated_consensus_safe_override_v2"][0]
    assert {"abstention_rate", "abstention_correct_rate", "abstention_missed_gain"}.issubset(row)


def test_ae_utility_consensus_v2_rejects_source_inner_domain_degradation() -> None:
    cfg = _ae_utility_consensus_v2_cfg()
    good = {
        "ae_argmin_top1_oracle_hit": 0.5,
        "top1_oracle_hit": 0.5,
        "ae_delta_spearman_non_anchor": 0.2,
        "raw_predicted_delta_spearman_non_anchor": 0.2,
        "ae_argmin_mean_oracle_gap_pct": 10.0,
        "mean_oracle_gap_pct": 9.0,
    }
    bad = dict(good, top1_oracle_hit=0.0)
    stability = auc._source_inner_stability([good, bad], cfg)
    assert int(stability["source_inner_material_degradation_count"]) == 1
    assert int(stability["passes_source_inner_stability_gates"]) == 0


def test_ae_utility_consensus_v2_tiny_capped_smoke_run(tmp_path, monkeypatch) -> None:
    sample_domains, expert_domains, true_nelbo, _meta, ae_scores = _fake_payload_ae_first()

    def fake_score(**kwargs):
        _ = kwargs
        embeddings = np.stack([np.asarray([float(i), 0.0]) for i in range(len(sample_domains))])
        metadata = [{"magnification": int(domain), "sample_id": f"s{i}"} for i, domain in enumerate(sample_domains)]
        return embeddings, sample_domains, true_nelbo, expert_domains, metadata

    def fake_ae_scores(**kwargs):
        _ = kwargs
        return ae_scores

    cfg = _support_free_ae_cfg()
    cfg["autoencoder_proxy"]["utility_calibrator"] = {
        "enabled": True,
        "primary_method": "ae_utility_calibrated_consensus_safe_override_v2",
        "model_types": ["ridge_delta_consensus"],
        "primary_model_type": "ridge_delta_consensus",
        "diagnostic_model_types": ["pairwise_ranker"],
        "fallback_policy": "ae_argmin_zscore",
        "feature_sets_primary": ["ae_consensus_core", "ae_consensus_quality"],
        "feature_sets_diagnostic": ["ae_metadata_consensus", "ae_combined_consensus"],
        "delta_thresholds": ["__inf__"],
        "margin_thresholds": [0.0],
        "consensus_thresholds": [1.0],
        "uncertainty_multiplier": 1.0,
        "ensemble_strategy": "source_domain_leave_one_plus_full",
        "abstention_correct_gap_pct_epsilon": 1.0,
        "source_inner_stability_gates": {
            "min_pseudo_domain_positive_rate": 0.80,
            "max_pseudo_domain_gain_share": 0.50,
            "max_source_inner_fold_gain_share": 0.50,
        },
    }
    monkeypatch.setattr(lu, "_score_experts_batched", fake_score)
    monkeypatch.setattr(lu, "build_autoencoder_score_matrices", fake_ae_scores)
    results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={f"expert_{d}": "unused" for d in expert_domains},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg=cfg,
        reports_dir=tmp_path,
        autoencoder_artifacts={"dummy": True},
    )

    assert "ae_utility_calibrated_consensus_safe_override_v2" in results["metrics_by_method"]
    assert results["artifacts"]["ae_utility_calibrator_v2_raw"] == "ae_utility_calibrator_v2_raw.csv"
    assert (tmp_path / "ae_utility_calibrator_v2_policy_audit.csv").exists()


def test_ae_utility_calibrator_tiny_capped_smoke_run(tmp_path, monkeypatch) -> None:
    sample_domains, expert_domains, true_nelbo, _meta, ae_scores = _fake_payload_ae_first()

    def fake_score(**kwargs):
        _ = kwargs
        embeddings = np.stack([np.asarray([float(i), 0.0]) for i in range(len(sample_domains))])
        metadata = [{"magnification": int(domain), "sample_id": f"s{i}"} for i, domain in enumerate(sample_domains)]
        return embeddings, sample_domains, true_nelbo, expert_domains, metadata

    def fake_ae_scores(**kwargs):
        _ = kwargs
        return ae_scores

    cfg = _support_free_ae_cfg()
    cfg["autoencoder_proxy"]["utility_calibrator"] = {
        "enabled": True,
        "primary_method": "ae_utility_calibrated_safe_override_v1",
        "model_types": ["ridge_delta"],
        "primary_model_type": "ridge_delta",
        "diagnostic_model_types": ["pairwise_ranker"],
        "fallback_policy": "ae_argmin_zscore",
        "feature_sets_primary": ["ae_core", "ae_quality"],
        "feature_sets_diagnostic": ["ae_metadata", "ae_combined"],
        "delta_thresholds": ["__inf__"],
        "margin_thresholds": [0.0],
    }
    monkeypatch.setattr(lu, "_score_experts_batched", fake_score)
    monkeypatch.setattr(lu, "build_autoencoder_score_matrices", fake_ae_scores)
    results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={f"expert_{d}": "unused" for d in expert_domains},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg=cfg,
        reports_dir=tmp_path,
        autoencoder_artifacts={"dummy": True},
    )

    assert "ae_utility_calibrated_safe_override_v1" in results["metrics_by_method"]
    assert results["artifacts"]["ae_utility_calibrator_raw"] == "ae_utility_calibrator_raw.csv"
    assert (tmp_path / "ae_utility_calibrator_policy_audit.csv").exists()


def test_ae_utility_precision_v11_tiny_capped_smoke_run(tmp_path, monkeypatch) -> None:
    sample_domains, expert_domains, true_nelbo, _meta, ae_scores = _fake_payload_ae_first()

    def fake_score(**kwargs):
        _ = kwargs
        embeddings = np.stack([np.asarray([float(i), 0.0]) for i in range(len(sample_domains))])
        metadata = [{"magnification": int(domain), "sample_id": f"s{i}"} for i, domain in enumerate(sample_domains)]
        return embeddings, sample_domains, true_nelbo, expert_domains, metadata

    def fake_ae_scores(**kwargs):
        _ = kwargs
        return ae_scores

    cfg = _support_free_ae_cfg()
    cfg["autoencoder_proxy"]["utility_calibrator"] = {
        "enabled": True,
        "primary_method": "ae_utility_calibrated_precision_lcb_safe_override_v11",
        "model_types": ["ridge_delta"],
        "primary_model_type": "ridge_delta",
        "diagnostic_model_types": ["pairwise_ranker"],
        "fallback_policy": "ae_argmin_zscore",
        "feature_sets_primary": ["ae_core", "ae_quality"],
        "feature_sets_diagnostic": [],
        "delta_thresholds": ["__inf__"],
        "margin_thresholds": [0.0],
        "selection_mode": "precision_lcb_selected_v11",
        "precision_selection": {
            "min_strict_improvement_precision": 0.75,
            "min_strict_improvement_precision_lcb": 0.60,
            "min_active_override_count": 10,
            "min_active_override_rate": 0.10,
            "min_net_gain_vs_ae_argmin": 0.0,
            "neutral_override_gap_pct_band": 0.25,
            "max_worst_pseudo_domain_gap_degradation_pp": 1.0,
            "bootstrap_reps": 10,
            "bootstrap_seed": 1337,
            "diagnostic_precision_thresholds": [0.70, 0.75, 0.80, 0.85],
        },
    }
    monkeypatch.setattr(lu, "_score_experts_batched", fake_score)
    monkeypatch.setattr(lu, "build_autoencoder_score_matrices", fake_ae_scores)
    results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={f"expert_{d}": "unused" for d in expert_domains},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg=cfg,
        reports_dir=tmp_path,
        autoencoder_artifacts={"dummy": True},
    )

    assert "ae_utility_calibrated_precision_lcb_safe_override_v11" in results["metrics_by_method"]
    assert (
        results["artifacts"]["ae_utility_calibrator_precision_v11_policy_audit"]
        == "ae_utility_calibrator_precision_v11_policy_audit.csv"
    )
    assert (tmp_path / "ae_utility_calibrator_precision_v11_precision_tradeoff.csv").exists()


def test_ae_utility_precision_v12_tiny_capped_smoke_run(tmp_path, monkeypatch) -> None:
    sample_domains, expert_domains, true_nelbo, _meta, ae_scores = _fake_payload_ae_first()

    def fake_score(**kwargs):
        _ = kwargs
        embeddings = np.stack([np.asarray([float(i), 0.0]) for i in range(len(sample_domains))])
        metadata = [{"magnification": int(domain), "sample_id": f"s{i}"} for i, domain in enumerate(sample_domains)]
        return embeddings, sample_domains, true_nelbo, expert_domains, metadata

    def fake_ae_scores(**kwargs):
        _ = kwargs
        return ae_scores

    cfg = _support_free_ae_cfg()
    cfg["autoencoder_proxy"]["utility_calibrator"] = {
        "enabled": True,
        "primary_method": "ae_utility_calibrated_precision_lcb_v1_guarded_safe_override_v12",
        "model_types": ["ridge_delta"],
        "primary_model_type": "ridge_delta",
        "diagnostic_model_types": ["pairwise_ranker"],
        "fallback_policy": "ae_argmin_zscore",
        "feature_sets_primary": ["ae_core", "ae_quality"],
        "feature_sets_diagnostic": [],
        "delta_thresholds": ["__inf__"],
        "margin_thresholds": [0.0],
        "selection_mode": "precision_lcb_v1_guarded_v12",
        "precision_selection": {
            "min_strict_improvement_precision": 0.75,
            "min_strict_improvement_precision_lcb": 0.60,
            "min_active_override_count": 12,
            "min_active_override_rate": 0.10,
            "min_net_gain_vs_ae_argmin": 0.0,
            "neutral_override_gap_pct_band": 0.25,
            "max_worst_pseudo_domain_gap_degradation_pp": 1.0,
            "bootstrap_reps": 10,
            "bootstrap_seed": 1337,
            "diagnostic_precision_thresholds": [0.70, 0.75, 0.80, 0.85],
            "v1_guard": {
                "min_gap_delta_vs_v1_lcb_pp": -0.25,
                "max_top1_drop_vs_v1_abs": 0.02,
                "max_spearman_drop_vs_v1_abs": 0.03,
                "max_worst_pseudo_domain_gap_degradation_vs_v1_pp": 1.0,
                "max_harmful_override_rate_ucb": 0.30,
            },
        },
    }
    monkeypatch.setattr(lu, "_score_experts_batched", fake_score)
    monkeypatch.setattr(lu, "build_autoencoder_score_matrices", fake_ae_scores)
    results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={f"expert_{d}": "unused" for d in expert_domains},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg=cfg,
        reports_dir=tmp_path,
        autoencoder_artifacts={"dummy": True},
    )

    method = "ae_utility_calibrated_precision_lcb_v1_guarded_safe_override_v12"
    assert method in results["metrics_by_method"]
    assert (
        results["artifacts"]["ae_utility_calibrator_precision_v12_policy_audit"]
        == "ae_utility_calibrator_precision_v12_policy_audit.csv"
    )
    assert (tmp_path / "ae_utility_calibrator_precision_v12_precision_tradeoff.csv").exists()


def test_safe_override_falls_back_to_metadata_with_tau_inf(tmp_path, monkeypatch) -> None:
    def fake_score(**kwargs):
        _ = kwargs
        return _fake_scored_payload()

    def fake_ae_scores(**kwargs):
        _ = kwargs
        return _fake_ae_scores()

    monkeypatch.setattr(lu, "_score_experts_batched", fake_score)
    monkeypatch.setattr(lu, "build_autoencoder_score_matrices", fake_ae_scores)
    results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={"expert_40": "unused", "expert_100": "unused", "expert_200": "unused"},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg=_support_free_ae_cfg(thresholds=["inf"]),
        reports_dir=tmp_path,
        autoencoder_artifacts={"dummy": True},
    )

    assert results["artifacts"]["residual_raw"] == "residual_safe_v2_raw.csv"
    override_rows = _read_csv(tmp_path / "residual_safe_v2_override_diagnostics.csv")
    ae_rows = [r for r in override_rows if r["method"] == "metadata_ae_residual_safe_override_v1"]
    assert ae_rows
    assert all(row["selected_tau"] == "inf" for row in ae_rows)
    assert all(float(row["override_rate"]) == 0.0 for row in ae_rows)
    assert all(float(row["safe_fallback_rate"]) == 1.0 for row in ae_rows)


def test_tau_inf_outputs_identical_to_metadata_routing(tmp_path, monkeypatch) -> None:
    def fake_score(**kwargs):
        _ = kwargs
        return _fake_scored_payload()

    def fake_ae_scores(**kwargs):
        _ = kwargs
        return _fake_ae_scores()

    monkeypatch.setattr(lu, "_score_experts_batched", fake_score)
    monkeypatch.setattr(lu, "build_autoencoder_score_matrices", fake_ae_scores)
    lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={"expert_40": "unused", "expert_100": "unused", "expert_200": "unused"},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg=_support_free_ae_cfg(thresholds=["inf"]),
        reports_dir=tmp_path,
        autoencoder_artifacts={"dummy": True},
    )

    rows = _read_csv(tmp_path / "learned_utility_sample_selections.csv")
    metadata = {
        int(row["sample_index"]): row
        for row in rows
        if row["method"] == "metadata_routing"
    }
    ae_safe = [
        row for row in rows if row["method"] == "metadata_ae_residual_safe_override_v1"
    ]
    assert ae_safe
    for row in ae_safe:
        base = metadata[int(row["sample_index"])]
        assert row["selected_expert"] == base["selected_expert"]
        assert row["selected_nelbo"] == base["selected_nelbo"]
        assert row["oracle_gap"] == base["oracle_gap"]


def test_ae_utility_recall_v15_config_parses() -> None:
    cfg = _ae_utility_recall_v15_cfg()
    assert cfg.primary_method == "ae_utility_calibrated_v1_recall_budget_safe_override_v15"
    assert cfg.selection_mode == "v1_recall_budget_v15"
    assert cfg.recall_budget_rates == (0.0, 0.50)
    assert cfg.recall_scoring_policy == "ridge_delta_best_non_anchor"


def test_ae_utility_recall_v15_primary_is_metadata_free() -> None:
    cfg = _ae_utility_recall_v15_cfg()
    assert cfg.feature_sets_primary == ("ae_core", "ae_quality")
    assert cfg.feature_sets_diagnostic == ()


def test_ae_utility_recall_v15_keeps_v1_active_overrides_unchanged() -> None:
    pred = np.asarray([[0.0, 0.6, 0.3], [0.0, 0.2, 0.4], [0.0, 0.8, 0.7]], dtype=np.float64)
    anchor = np.asarray([0, 0, 0], dtype=np.int64)
    v1_selected = np.asarray([2, 0, 0], dtype=np.int64)
    selected, info = auc._apply_recall_budget_policy(
        v1_selected_idx=v1_selected,
        anchor_idx=anchor,
        pred_delta_matrix=pred,
        ae_zscore_eval=np.asarray([[0.0, 1.0, 2.0], [0.0, 2.0, 1.0], [0.0, 1.0, 2.0]], dtype=np.float64),
        candidate_expert_domains=[0, 1, 2],
        sample_indices=[10, 11, 12],
        delta_threshold=0.5,
        margin_threshold=0.0,
        recall_budget_rate=1.0,
    )
    assert int(selected[0]) == int(v1_selected[0])
    assert not bool(info["recall_applied"][0])


def test_ae_utility_recall_v15_budget_zero_matches_v1() -> None:
    pred = np.asarray([[0.0, 0.9, 0.1], [0.0, 0.8, 0.2]], dtype=np.float64)
    anchor = np.asarray([0, 0], dtype=np.int64)
    v1_selected = np.asarray([0, 0], dtype=np.int64)
    selected, info = auc._apply_recall_budget_policy(
        v1_selected_idx=v1_selected,
        anchor_idx=anchor,
        pred_delta_matrix=pred,
        ae_zscore_eval=np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float64),
        candidate_expert_domains=[0, 1, 2],
        sample_indices=[1, 2],
        delta_threshold=1.0,
        margin_threshold=0.0,
        recall_budget_rate=0.0,
    )
    assert selected.tolist() == v1_selected.tolist()
    assert int(info["recall_budget_count"]) == 0


def test_ae_utility_recall_v15_only_scores_v1_abstentions() -> None:
    pred = np.asarray([[0.0, 0.9], [0.0, 0.8]], dtype=np.float64)
    anchor = np.asarray([0, 0], dtype=np.int64)
    v1_selected = np.asarray([1, 0], dtype=np.int64)
    selected, info = auc._apply_recall_budget_policy(
        v1_selected_idx=v1_selected,
        anchor_idx=anchor,
        pred_delta_matrix=pred,
        ae_zscore_eval=np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        candidate_expert_domains=[0, 1],
        sample_indices=[1, 2],
        delta_threshold=1.0,
        margin_threshold=0.0,
        recall_budget_rate=1.0,
    )
    assert selected.tolist() == [1, 1]
    assert info["abstention_reason"][0] == "v1_active_override"
    assert bool(info["recall_applied"][1])


def test_ae_utility_recall_v15_excludes_anchor_from_recall_candidates() -> None:
    pred = np.asarray([[99.0, 0.2, 0.1]], dtype=np.float64)
    selected, info = auc._apply_recall_budget_policy(
        v1_selected_idx=np.asarray([0], dtype=np.int64),
        anchor_idx=np.asarray([0], dtype=np.int64),
        pred_delta_matrix=pred,
        ae_zscore_eval=np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64),
        candidate_expert_domains=[0, 1, 2],
        sample_indices=[1],
        delta_threshold=1.0,
        margin_threshold=0.0,
        recall_budget_rate=1.0,
    )
    assert int(selected[0]) == 1
    assert int(info["best_idx"][0]) == 1


def test_ae_utility_recall_v15_excludes_nonfinite_candidate_scores() -> None:
    selected, info = auc._apply_recall_budget_policy(
        v1_selected_idx=np.asarray([0], dtype=np.int64),
        anchor_idx=np.asarray([0], dtype=np.int64),
        pred_delta_matrix=np.asarray([[0.0, float("nan"), float("-inf")]], dtype=np.float64),
        ae_zscore_eval=np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64),
        candidate_expert_domains=[0, 1, 2],
        sample_indices=[1],
        delta_threshold=0.0,
        margin_threshold=0.0,
        recall_budget_rate=1.0,
    )
    assert int(selected[0]) == 0
    assert info["abstention_reason"][0] == "no_positive_candidate"


def test_ae_utility_recall_v15_requires_positive_predicted_delta() -> None:
    selected, info = auc._apply_recall_budget_policy(
        v1_selected_idx=np.asarray([0], dtype=np.int64),
        anchor_idx=np.asarray([0], dtype=np.int64),
        pred_delta_matrix=np.asarray([[0.0, -0.1, -0.2]], dtype=np.float64),
        ae_zscore_eval=np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64),
        candidate_expert_domains=[0, 1, 2],
        sample_indices=[1],
        delta_threshold=0.0,
        margin_threshold=0.0,
        recall_budget_rate=1.0,
    )
    assert int(selected[0]) == 0
    assert int(info["eligible_recall_count"]) == 0


def test_ae_utility_recall_v15_reports_abstention_reason() -> None:
    selected, info = auc._apply_recall_budget_policy(
        v1_selected_idx=np.asarray([0, 0, 0, 1], dtype=np.int64),
        anchor_idx=np.asarray([0, 0, 0, 0], dtype=np.int64),
        pred_delta_matrix=np.asarray(
            [
                [0.0, 0.40, 0.30],
                [0.0, 0.60, 0.59],
                [0.0, -0.10, -0.20],
                [0.0, 0.90, 0.10],
            ],
            dtype=np.float64,
        ),
        ae_zscore_eval=np.asarray(
            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            dtype=np.float64,
        ),
        candidate_expert_domains=[0, 1, 2],
        sample_indices=[1, 2, 3, 4],
        delta_threshold=0.50,
        margin_threshold=0.05,
        recall_budget_rate=1.0,
    )
    assert selected.tolist()[:3] == [1, 1, 0]
    assert info["abstention_reason"][0] == "below_delta_threshold"
    assert info["abstention_reason"][1] == "below_margin_threshold"
    assert info["abstention_reason"][2] == "no_positive_candidate"
    assert info["abstention_reason"][3] == "v1_active_override"


def test_ae_utility_recall_v15_budget_count_ceil_semantics() -> None:
    pred = np.asarray([[0.0, 0.9], [0.0, 0.8], [0.0, 0.7]], dtype=np.float64)
    selected, info = auc._apply_recall_budget_policy(
        v1_selected_idx=np.asarray([0, 0, 0], dtype=np.int64),
        anchor_idx=np.asarray([0, 0, 0], dtype=np.int64),
        pred_delta_matrix=pred,
        ae_zscore_eval=np.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        candidate_expert_domains=[0, 1],
        sample_indices=[1, 2, 3],
        delta_threshold=1.0,
        margin_threshold=0.0,
        recall_budget_rate=0.34,
    )
    assert int(info["recall_budget_count"]) == 2
    assert int(np.sum(selected != 0)) == 2


def test_ae_utility_recall_v15_rank_tiebreak_is_stable() -> None:
    pred = np.asarray([[0.0, 0.5], [0.0, 0.5]], dtype=np.float64)
    selected, info = auc._apply_recall_budget_policy(
        v1_selected_idx=np.asarray([0, 0], dtype=np.int64),
        anchor_idx=np.asarray([0, 0], dtype=np.int64),
        pred_delta_matrix=pred,
        ae_zscore_eval=np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        candidate_expert_domains=[0, 1],
        sample_indices=[20, 10],
        delta_threshold=1.0,
        margin_threshold=0.0,
        recall_budget_rate=0.5,
    )
    assert int(info["recall_budget_count"]) == 1
    assert selected.tolist() == [0, 1]


def test_ae_utility_recall_v15_recall_classes_are_anchor_relative() -> None:
    metrics = auc._recall_budget_metrics(
        v1_selected_idx=np.asarray([0, 0, 0], dtype=np.int64),
        v15_selected_idx=np.asarray([1, 1, 1], dtype=np.int64),
        anchor_idx=np.asarray([0, 0, 0], dtype=np.int64),
        recall_applied=np.asarray([True, True, True]),
        true_eval=np.asarray([[10.0, 8.0], [10.0, 12.0], [10.0, 10.01]], dtype=np.float64),
        neutral_gap_pct_band=0.25,
    )
    assert int(metrics["recall_improving_count"]) == 1
    assert int(metrics["recall_harmful_count"]) == 1
    assert int(metrics["recall_neutral_count"]) == 1


def test_ae_utility_recall_v15_rejects_excess_active_rate_ratio() -> None:
    cfg = _ae_utility_recall_v15_cfg(max_active_ratio=1.1)
    summaries = [
        {
            "recall_improving_count": 10,
            "recall_harmful_count": 0,
            "recall_neutral_count": 0,
            "gap_delta_vs_v1": 0.1,
            "net_gain_vs_v1": 1.0,
            "v1_active_override_count": 10,
            "v15_active_override_count": 20,
            "v1_abstention_count": 100,
            "top1_oracle_hit": 1.0,
            "raw_predicted_delta_spearman_non_anchor": 1.0,
        }
    ]
    metrics = auc._aggregate_recall_budget_metrics(summaries=summaries, v1_summaries=summaries, cfg=cfg)
    assert int(metrics["passes_recall_budget_gates"]) == 0


def test_ae_utility_recall_v15_rejects_low_precision_budget() -> None:
    cfg = _ae_utility_recall_v15_cfg(min_strict_precision=0.70)
    summaries = [
        {
            "recall_improving_count": 6,
            "recall_harmful_count": 4,
            "recall_neutral_count": 0,
            "gap_delta_vs_v1": 0.20,
            "net_gain_vs_v1": 1.0,
            "v1_active_override_count": 100,
            "v15_active_override_count": 110,
            "v1_abstention_count": 100,
            "top1_oracle_hit": 1.0,
            "raw_predicted_delta_spearman_non_anchor": 1.0,
        }
    ]
    metrics = auc._aggregate_recall_budget_metrics(summaries=summaries, v1_summaries=summaries, cfg=cfg)
    assert float(metrics["strict_recall_precision"]) < 0.70
    assert int(metrics["passes_recall_budget_gates"]) == 0


def test_ae_utility_recall_v15_rejects_high_harm_budget() -> None:
    cfg = _ae_utility_recall_v15_cfg(max_harm_ucb=0.35)
    summaries = [
        {
            "recall_improving_count": 10,
            "recall_harmful_count": 4,
            "recall_neutral_count": 0,
            "gap_delta_vs_v1": 0.20,
            "net_gain_vs_v1": 1.0,
            "v1_active_override_count": 100,
            "v15_active_override_count": 110,
            "v1_abstention_count": 100,
            "top1_oracle_hit": 1.0,
            "raw_predicted_delta_spearman_non_anchor": 1.0,
        }
    ]
    metrics = auc._aggregate_recall_budget_metrics(summaries=summaries, v1_summaries=summaries, cfg=cfg)
    assert float(metrics["harmful_recall_rate_ucb"]) > 0.35
    assert int(metrics["passes_recall_budget_gates"]) == 0
