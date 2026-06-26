from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from config import parse_config
from component_union_tailrisk_anchored_mass_bagged import PRIMARY_TAILRISK_METHOD
from covariance_prior import parse_covariance_prior_config
from covariance_shrinkage import parse_covariance_shrinkage_config
from covariance_viability import parse_covariance_viability_config
from preservation import parse_preservation_config
from preservation_repair import parse_repair_config
from preservation_sampling import parse_sampling_config
from prior_calibration import parse_prior_calibration_config
from source_union_balanced_gmm_prior import parse_source_union_balanced_gmm_prior_config
from source_union_gmm_prior import parse_source_union_gmm_prior_config
from source_union_k24_gmm_prior import parse_source_union_k24_gmm_prior_config


def _tiny_config(tmp_path: Path):
    return parse_config(
        {
            "experiment": {
                "name": "target_support32_calibrated_unlabeled_marginal_nelbo_top2_geom_virchow2_cvae_pca256_v1",
                "artifact_root": str(tmp_path / "artifacts"),
                "primary_method": "support_nelbo_top2_geom",
            },
            "inputs": {
                "feature_cache_root": str(tmp_path / "cache" / "virchow2"),
                "backbone": "virchow2",
            },
            "run_matrix": {
                "experiment_seeds": [42],
                "heldout_centers": ["0", "1", "2", "3", "4"],
                "support_size": 32,
                "support_seeds": [17],
                "generation_seeds": [17],
                "classifier_seeds": [17],
                "candidate_count_per_cell": 4,
            },
            "feature_frame": {"pca_dim": 256, "fit_scope": "per_expert_source_train"},
            "model": {
                "hidden_dim": 512,
                "latent_dim": 64,
                "num_hidden_layers": 2,
                "train_epochs": 1,
                "batch_size": 16,
                "learning_rate": 0.001,
                "class_conditioning": "encoder_decoder_one_hot",
            },
            "routing": {
                "primary_score": "calibrated_marginal_support_nelbo",
                "support_sampler": "random_unlabeled_sample_ids",
            },
            "generation": {
                "mode": "class_stratified_reference_posterior",
                "synthetic_per_class_total": 128,
            },
            "downstream": {
                "classifier": "sklearn_logistic_regression",
                "aggregation": "geometric_probability_pooling",
                "eps": 1e-12,
            },
        },
        base_dir=tmp_path,
    )


def _tiny_preservation_config(tmp_path: Path):
    return parse_preservation_config(
        {
            "experiment": {
                "name": "virchow2_cvae_preservation_diagnosis_v1",
                "artifact_root": str(tmp_path / "preservation_artifacts"),
            },
            "inputs": {
                "feature_cache_root": str(tmp_path / "preservation_cache" / "virchow2"),
                "backbone": "virchow2",
            },
            "run_matrix": {
                "experiment_seeds": [42],
                "heldout_centers": ["0", "1", "2", "3", "4"],
                "replicate_seeds": [17],
                "candidate_count_per_cell": 4,
            },
            "feature_frame": {"pca_dim": 256, "fit_scope": "per_expert_source_train"},
            "model": {
                "hidden_dim": 512,
                "latent_dim": 64,
                "num_hidden_layers": 2,
                "train_epochs": 1,
                "batch_size": 16,
                "learning_rate": 0.001,
                "class_conditioning": "encoder_decoder_one_hot",
            },
            "generation": {
                "synthetic_per_class_total": 128,
                "class_prior_for_generation": "uniform",
            },
            "classifier": {
                "type": "sklearn_logistic_regression",
                "solver": "lbfgs",
                "C": 1.0,
                "max_iter": 2000,
                "class_weight": "balanced",
                "classifier_seed": None,
            },
        },
        base_dir=tmp_path,
    )


def _tiny_repair_config(tmp_path: Path):
    variants = [
        ("current_pca200_beta1_reference", "per_source", 256, 64, 1.0, 0.0, "legacy_sum_mse_kl", "reference_only", "adam"),
        ("pca64_beta001", "per_source", 64, 16, 0.001, 0.0, "normalized_repair", "primary", "adamw"),
        ("pca128_beta001", "per_source", 128, 32, 0.001, 0.0, "normalized_repair", "diagnostic_only", "adamw"),
        ("pca64_beta001_probe025", "per_source", 64, 16, 0.001, 0.25, "normalized_repair", "diagnostic_only", "adamw"),
        ("pca128_beta001_probe025", "per_source", 128, 32, 0.001, 0.25, "normalized_repair", "diagnostic_only", "adamw"),
        (
            "source_union_pca64_beta001_diagnostic",
            "source_union_excluding_target",
            64,
            16,
            0.001,
            0.0,
            "normalized_repair",
            "diagnostic_only",
            "adamw",
        ),
        (
            "source_union_pca64_beta001_probe025_diagnostic",
            "source_union_excluding_target",
            64,
            16,
            0.001,
            0.25,
            "normalized_repair",
            "diagnostic_only",
            "adamw",
        ),
    ]
    return parse_repair_config(
        {
            "experiment": {
                "name": "virchow2_cvae_preservation_repair_v1",
                "artifact_root": str(tmp_path / "repair_artifacts"),
                "primary_variant": "pca64_beta001",
                "min_decision_rows": 1,
            },
            "inputs": {
                "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
                "backbone": "virchow2",
            },
            "run_matrix": {
                "experiment_seeds": [42],
                "heldout_centers": ["0", "1", "2"],
                "replicate_seeds": [17],
            },
            "generation": {
                "synthetic_per_class_total": 128,
            },
            "classifier": {
                "type": "sklearn_logistic_regression",
                "solver": "lbfgs",
                "C": 1.0,
                "max_iter": 2000,
                "class_weight": "balanced",
                "classifier_seed": None,
            },
            "source_probe": {
                "type": "torch_linear_classifier",
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "epochs": 1,
                "batch_size": 16,
                "class_weight": "balanced",
                "early_stopping": False,
            },
            "variants": [
                {
                    "variant_id": variant_id,
                    "expert_pool_type": pool,
                    "requested_pca_dim": pca,
                    "hidden_dim": 512,
                    "latent_dim": latent,
                    "num_hidden_layers": 2,
                    "train_epochs": 1,
                    "batch_size": 16,
                    "learning_rate": 0.001,
                    "optimizer": optimizer,
                    "weight_decay": 0.0 if optimizer == "adam" else 0.0001,
                    "gradient_clip_norm": 5.0,
                    "beta_final": beta,
                    "kl_warmup_epochs": 1 if variant_id == "current_pca200_beta1_reference" else 2,
                    "probe_ce_weight": probe,
                    "loss_style": loss_style,
                    "selection_source": selection,
                }
                for variant_id, pool, pca, latent, beta, probe, loss_style, selection, optimizer in variants
            ],
        },
        base_dir=tmp_path,
    )


def _tiny_sampling_config(tmp_path: Path, repair_root: Path):
    return parse_sampling_config(_tiny_sampling_payload(tmp_path, repair_root), base_dir=tmp_path)


def _tiny_sampling_payload(tmp_path: Path, repair_root: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_pca64_sampling_continuation_v1",
            "artifact_root": str(tmp_path / "sampling_artifacts"),
            "primary_variant": "pca64_beta001",
            "min_decision_cells": 1,
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(repair_root),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "sampling": {
            "posterior_temperatures_primary": [1.0],
            "posterior_temperatures_diagnostic": [0.25, 0.5],
            "prior_scales_primary": [1.0],
            "prior_scales_diagnostic": [0.25, 0.5],
            "empirical_posterior_temperature": 1.0,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_prior_calibration_config(tmp_path: Path, repair_root: Path, sampling_root: Path):
    return parse_prior_calibration_config(
        _tiny_prior_calibration_payload(tmp_path, repair_root, sampling_root),
        base_dir=tmp_path,
    )


def _tiny_prior_calibration_payload(tmp_path: Path, repair_root: Path, sampling_root: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_latent_prior_calibration_v1",
            "artifact_root": str(tmp_path / "prior_calibration_artifacts"),
            "primary_variant": "pca64_beta001",
            "min_decision_cells": 9,
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(repair_root),
            "sampling_artifact_root": str(sampling_root),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "prior_calibration": {
            "primary_method": "cvae_cc_diag_gaussian_prior_sample",
            "min_prior_fit_records_per_class": 8,
            "variance_floor": 1.0e-4,
            "variance_ddof": 0,
            "shrinkage_alphas": [0.25, 0.5],
            "standard_prior_repro_abs_tol_bacc": 1.0,
            "full_cov_min_records_per_class": 32,
            "full_cov_shrinkage_alpha": 0.1,
            "full_cov_eigenvalue_floor": 1.0e-4,
            "full_cov_fallback_if_singular": "diag",
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_covariance_prior_config(tmp_path: Path, repair_root: Path, sampling_root: Path, prior_root: Path):
    return parse_covariance_prior_config(
        _tiny_covariance_prior_payload(tmp_path, repair_root, sampling_root, prior_root),
        base_dir=tmp_path,
    )


def _tiny_covariance_prior_payload(tmp_path: Path, repair_root: Path, sampling_root: Path, prior_root: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_covariance_prior_confirmation_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_covariance_prior_confirmation_v1"),
            "primary_variant": "pca64_beta001",
            "min_decision_cells": 9,
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(repair_root),
            "sampling_artifact_root": str(sampling_root),
            "prior_calibration_artifact_root": str(prior_root),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "covariance_prior": {
            "primary_method": "cvae_cc_cov_shrinkage_prior_sample",
            "covariance_shrinkage_alpha": 0.10,
            "covariance_eigenvalue_floor": 1.0e-4,
            "full_cov_min_records_per_class": 32,
            "fallback_if_under_ranked": "diag",
            "standard_prior_repro_abs_tol_bacc": 1.0,
            "diag_prior_repro_abs_tol_bacc": 1.0,
            "full_cov_diagnostic_repro_abs_tol_bacc": 1.0,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_covariance_viability_config(tmp_path: Path, covariance_root: Path):
    return parse_covariance_viability_config(
        _tiny_covariance_viability_payload(tmp_path, covariance_root),
        base_dir=tmp_path,
    )


def _tiny_covariance_viability_payload(tmp_path: Path, covariance_root: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_covariance_prior_viability_audit_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_covariance_prior_viability_audit_v1"),
        },
        "inputs": {
            "covariance_confirmation_artifact_root": str(covariance_root),
        },
        "viability_audit": {
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "min_viable_cells": 30,
            "min_viable_cells_per_center": 3,
            "min_viable_seeds_per_center": 2,
            "high_real_threshold": 0.80,
            "viable_real_threshold": 0.75,
            "borderline_real_threshold": 0.65,
            "global_center_equal_mean_bacc_min": 0.85,
            "mean_clipped_preservation_gap_max": 0.08,
            "mean_preservation_ratio_min": 0.92,
            "seed_std_max": 0.07,
            "delta_bacc_vs_standard_prior_min": 0.05,
            "delta_bacc_vs_diag_prior_min": 0.03,
            "covariance_beats_diag_cell_fraction_min": 0.70,
            "covariance_beats_diag_center_fraction_min": 0.75,
            "worst_delta_vs_diag_prior_min": -0.05,
            "min_cell_bacc_min": 0.60,
            "min_center_mean_bacc_min": 0.75,
        },
    }


def _tiny_covariance_shrinkage_config(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
    viability_root: Path,
):
    return parse_covariance_shrinkage_config(
        _tiny_covariance_shrinkage_payload(tmp_path, repair_root, sampling_root, prior_root, covariance_root, viability_root),
        base_dir=tmp_path,
    )


def _tiny_covariance_shrinkage_payload(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
    viability_root: Path,
):
    return {
        "experiment": {
            "name": "virchow2_cvae_covariance_shrinkage_stability_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_covariance_shrinkage_stability_v1"),
            "primary_variant": "pca64_beta001",
            "min_decision_cells": 9,
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(repair_root),
            "sampling_artifact_root": str(sampling_root),
            "prior_calibration_artifact_root": str(prior_root),
            "covariance_confirmation_artifact_root": str(covariance_root),
            "covariance_viability_artifact_root": str(viability_root),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "covariance_shrinkage": {
            "primary_method": "cvae_cc_cov_diag_shrinkage075_prior_sample",
            "primary_covariance_shrinkage_alpha": 0.75,
            "diagnostic_covariance_shrinkage_alphas": [0.50, 0.90],
            "reference_covariance_shrinkage_alpha": 0.10,
            "diagonal_reference_alpha": 1.00,
            "covariance_eigenvalue_floor": 1.0e-4,
            "full_cov_min_records_per_class": 32,
            "fallback_if_under_ranked": "diag",
            "standard_prior_repro_abs_tol_bacc": 1.0,
            "diag_prior_repro_abs_tol_bacc": 1.0,
            "alpha010_repro_abs_tol_bacc": 1.0,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_source_union_gmm_config(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
):
    return parse_source_union_gmm_prior_config(
        _tiny_source_union_gmm_payload(tmp_path, repair_root, sampling_root, prior_root, covariance_root),
        base_dir=tmp_path,
    )


def _tiny_source_union_gmm_payload(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
):
    return {
        "experiment": {
            "name": "virchow2_cvae_source_union_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_source_union_gmm_prior_v1"),
            "primary_variant": "source_union_pca64_beta001_diagnostic",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(repair_root),
            "sampling_artifact_root": str(sampling_root),
            "prior_calibration_artifact_root": str(prior_root),
            "covariance_confirmation_artifact_root": str(covariance_root),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "gmm_prior": {
            "primary_method": "source_union_cc_diag_gmm_k8_prior_sample",
            "gmm_components": 8,
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 2,
            "gmm_max_iter": 200,
            "gmm_weight_floor": 0.01,
            "min_class_train_count": 8,
            "min_effective_gmm_components": 1,
            "posterior_noise_scale": 0.0,
            "diagnostic_gmm_components": [4, 16],
            "diagnostic_posterior_noise_scales": [0.25],
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_source_union_balanced_gmm_config(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
    source_union_gmm_root: Path,
):
    return parse_source_union_balanced_gmm_prior_config(
        _tiny_source_union_balanced_gmm_payload(
            tmp_path,
            repair_root,
            sampling_root,
            prior_root,
            covariance_root,
            source_union_gmm_root,
        ),
        base_dir=tmp_path,
    )


def _tiny_source_union_balanced_gmm_payload(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
    source_union_gmm_root: Path,
):
    return {
        "experiment": {
            "name": "virchow2_cvae_source_union_center_balanced_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_source_union_center_balanced_gmm_prior_v1"),
            "primary_variant": "source_union_pca64_beta001_diagnostic",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(repair_root),
            "sampling_artifact_root": str(sampling_root),
            "prior_calibration_artifact_root": str(prior_root),
            "covariance_confirmation_artifact_root": str(covariance_root),
            "source_union_gmm_artifact_root": str(source_union_gmm_root),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "balanced_gmm_prior": {
            "primary_method": "source_union_center_balanced_cc_diag_gmm_k16_prior_sample",
            "gmm_components": 16,
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 2,
            "gmm_max_iter": 200,
            "gmm_weight_floor": 0.005,
            "min_source_center_class_count": 8,
            "min_effective_gmm_components": 1,
            "balanced_fit_samples_per_center_class": 8,
            "max_center_class_replacement_rate": 1.0,
            "mean_center_class_replacement_rate": 1.0,
            "posterior_noise_scale": 0.0,
            "diagnostic_gmm_components": [8, 24],
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_k16_gmm_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_k16_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_k16_gmm_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "decentralized_k16_prior": {
            "primary_method": "decentralized_exported_k4x4_cc_diag_gmm_k16_late_geom",
            "local_gmm_components_per_source_class": 4,
            "composed_components_per_class": 16,
            "source_weighting": "equal_source_mass",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_count_for_k4": 8,
            "min_component_weight": 0.001,
            "variance_floor": 1.0e-5,
            "primary_pooling": "geometric",
        },
        "support_nelbo_diagnostic": {
            "enabled": False,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_adaptive_gmm_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_adaptive_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_adaptive_gmm_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "strict_d1_artifact_root": str(tmp_path / "missing_decentralized_k16"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "adaptive_gmm_prior": {
            "primary_method": "decentralized_exported_adaptive_k_cc_diag_gmm_late_geom",
            "bic_method": "decentralized_exported_bic_selected_cc_diag_gmm_late_geom",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "equal_source_mass",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.001,
            "variance_floor": 1.0e-5,
            "primary_pooling": "geometric",
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_reliability_weighted_gmm_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "min_per_source_per_class": 8,
        },
        "reliability_weighted_gmm_prior": {
            "primary_method": "decentralized_exported_adaptive_k_source_reliability_weighted_geom",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "source_local_reliability",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.001,
            "variance_floor": 1.0e-5,
            "primary_pooling": "weighted_geometric",
            "reliability_floor_score": 0.05,
            "softmax_tau": 1.0,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_component_union_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_component_union_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_component_union_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "d1_2_artifact_root": str(tmp_path / "missing_d1_2"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "budget_diagnostic_per_class_total": 256,
            "min_per_source_per_class": 8,
        },
        "component_union_prior": {
            "primary_method": "decentralized_component_union_uniform_gmm",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "uniform_source_component_union",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "variance_ceiling_multiplier": 16.0,
            "primary_pooling": "pooled_raw_logistic",
            "reliability_floor_score": 0.05,
            "shrink_lambdas": [0.25, 0.5],
            "prototype_candidate_counts_per_source_class": [4, 3, 2, 1],
            "prototype_min_samples_per_component": 12,
            "prototype_variance_floor": 1.0e-5,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_support_calibrated_component_union_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_support8_calibrated_component_union_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_support8_calibrated_component_union_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "strict_full_run_matrix": False,
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
            "support_seeds": [17],
            "support_size": 8,
            "support_size_diagnostics": [16, 32],
            "nested_support_max_size": 32,
        },
        "generation": {
            "synthetic_per_class_total": 32,
            "min_per_source_per_class": 2,
        },
        "support_calibrated_component_union_prior": {
            "primary_method": "support8_calibrated_component_union_softmax_shrink050",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "support_calibrated_component_union_softmax_shrink050",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "variance_ceiling_multiplier": 16.0,
            "primary_pooling": "pooled_raw_logistic",
            "support_nelbo_tau": 1.0,
            "support_shrink_lambda": 0.5,
            "reliability_floor_score": 0.05,
            "shrink_lambdas": [0.25, 0.5],
            "matched_shuffled_support_null_permutations": 2,
            "random_mass_bag_control_size": 3,
            "anchor_repro_tolerance": 1.0e-4,
            "prototype_candidate_counts_per_source_class": [4, 3, 2, 1],
            "prototype_min_samples_per_component": 12,
            "prototype_variance_floor": 1.0e-5,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_target_support_regime_risk_gate_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_target_support32_regime_risk_gated_component_union_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_target_support32_regime_risk_gated_component_union_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "strict_full_run_matrix": False,
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "support_seeds": [17],
            "support_size": 32,
            "support_size_diagnostics": [8, 16],
            "nested_support_max_size": 32,
        },
        "generation": {
            "synthetic_per_class_total": 32,
            "min_per_source_per_class": 2,
        },
        "target_support_regime_risk_gate": {
            "primary_method": "target_support32_regime_risk_gated_random_bag_tail_safe_policy_v1",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 2,
            "source_weighting": "target_support32_regime_risk_policy_gate",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "variance_ceiling_multiplier": 16.0,
            "primary_pooling": "policy_level_gate",
            "reliability_floor_score": 0.05,
            "reliability_epsilon": 1.0e-8,
            "support_nelbo_tau": 1.0,
            "random_mass_bag_size": 3,
            "random_mass_bag_alpha": 4.0,
            "risk_low_threshold": 0.60,
            "risk_high_threshold": 0.75,
            "threshold_sensitivity_pairs": [[0.50, 0.70], [0.60, 0.75], [0.70, 0.85]],
            "min_gate_train_episodes": 8,
            "tail_risk_bacc_threshold": 0.80,
            "safer_policy_gain_threshold": 0.025,
            "gate_c": 0.25,
            "reconstruction_probability_tolerance": 1.0e-6,
        },
        "memory": {
            "skip_nearest_neighbor_audit": True,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_labeled_support_policy_calibration_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_labeled_support16_random_vs_dense_policy_calibration_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_labeled_support16_random_vs_dense_policy_calibration_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "strict_full_run_matrix": False,
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "support_seeds": [17],
            "primary_labeled_support_size": 16,
            "diagnostic_labeled_support_sizes": [8, 32],
            "nested_support_max_size": 32,
        },
        "generation": {
            "synthetic_per_class_total": 32,
            "min_per_source_per_class": 2,
        },
        "labeled_support_policy_calibration": {
            "primary_method": "labeled_support16_random_default_dense_switch_v1",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 2,
            "source_weighting": "labeled_support16_random_default_dense_switch",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "variance_ceiling_multiplier": 16.0,
            "primary_pooling": "labeled_support_random_default_dense_switch",
            "reliability_floor_score": 0.05,
            "reliability_epsilon": 1.0e-8,
            "random_mass_bag_size": 3,
            "random_mass_bag_alpha": 4.0,
            "primary_switch_quantum": 0.0625,
            "support_quantum_by_size": {8: 0.125, 16: 0.0625, 32: 0.03125},
        },
        "memory": {
            "skip_nearest_neighbor_audit": True,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_source_inner_validated_hybrid_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_source_inner_validated_dense_component_hybrid_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_source_inner_validated_dense_component_hybrid_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "d1_2_artifact_root": str(tmp_path / "missing_d1_2"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "strict_full_run_matrix": False,
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "min_per_source_per_class": 8,
        },
        "source_inner_validated_dense_component_hybrid": {
            "primary_method": "source_inner_validated_dense_component_binary_gate",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "source_inner_validated_dense_component_binary_gate",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "variance_ceiling_multiplier": 16.0,
            "primary_pooling": "binary_gate",
            "reliability_floor_score": 0.05,
            "reliability_epsilon": 1.0e-8,
            "component_shrink_lambda": 0.25,
            "matched_shuffled_gate_null_permutations": 2,
            "gate_mean_gain_min": 0.005,
            "gate_min_degradation_floor": -0.005,
            "gate_std_increase_max": 0.015,
            "gate_abs_ablation_ceiling": 0.15,
            "gate_abs_ablation_slack": 0.05,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_mass_bagged_component_union_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_component_union_mass_bagged_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_component_union_mass_bagged_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "paired_dense_artifact_root": str(tmp_path / "missing_paired_dense"),
            "component_union_v2_artifact_root": str(tmp_path / "missing_component_union_v2"),
            "hybrid_artifact_root": str(tmp_path / "missing_hybrid"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "strict_full_run_matrix": False,
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 32,
            "min_per_source_per_class": 2,
        },
        "mass_bagged_component_union": {
            "primary_method": "decentralized_component_union_mass_uncertainty_bagged_v1",
            "primary_bag_members": [
                "uniform_source_mass",
                "reliability_shrink_0.25",
                "reliability_shrink_0.50",
                "dirichlet_uniform_alpha4_perm000",
            ],
            "control_bag_size": 2,
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 2,
            "source_weighting": "mass_uncertainty_bagged_source_component_union",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "variance_ceiling_multiplier": 16.0,
            "primary_pooling": "arithmetic_probability_ensemble",
            "reliability_floor_score": 0.05,
            "reliability_epsilon": 1.0e-8,
            "shrink_lambdas": [0.25, 0.5],
            "anchor_repro_tolerance": 1.0e-4,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_tailrisk_anchored_component_union_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_component_union_tailrisk_anchored_mass_bagged_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_component_union_tailrisk_anchored_mass_bagged_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "paired_dense_artifact_root": str(tmp_path / "missing_paired_dense"),
            "mass_bagged_artifact_root": str(tmp_path / "missing_mass_bagged"),
            "support_calibrated_artifact_root": str(tmp_path / "missing_support_calibrated"),
            "shrink050_artifact_root": str(tmp_path / "missing_shrink050"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "strict_full_run_matrix": False,
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
            "fresh_replicate_seeds": [101],
        },
        "generation": {
            "synthetic_per_class_total": 32,
            "min_per_source_per_class": 2,
        },
        "tailrisk_anchored_component_union": {
            "primary_method": "component_union_tailrisk_anchored_shrink050_random_mass_bag_blend050",
            "primary_shrink_lambda": 0.5,
            "random_mass_bag_size": 3,
            "random_mass_bag_alpha": 4.0,
            "blend_alpha": 0.5,
            "matched_shuffled_reliability_null_permutations": 2,
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 2,
            "source_weighting": "tailrisk_anchored_shrink050_random_mass_bag_blend050",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "variance_ceiling_multiplier": 16.0,
            "primary_pooling": "fixed_arithmetic_probability_blend",
            "reliability_floor_score": 0.05,
            "reliability_epsilon": 1.0e-8,
            "anchor_repro_tolerance": 1.0e-4,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_multipanel_tailrisk_component_union_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "paired_dense_artifact_root": str(tmp_path / "missing_paired_dense"),
            "mass_bagged_artifact_root": str(tmp_path / "missing_mass_bagged"),
            "shrink050_artifact_root": str(tmp_path / "missing_shrink050"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "prior_tailrisk_artifact_root": str(tmp_path / "prior_tailrisk"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "strict_full_run_matrix": False,
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17, 23, 31],
            "fresh_replicate_seeds": [101, 103, 107, 109, 113, 127],
        },
        "generation": {
            "synthetic_per_class_total": 32,
            "min_per_source_per_class": 2,
        },
        "tailrisk_multipanel_component_union": {
            "primary_method": "component_union_tailrisk_multipanel_shrink050_random_mass_bag_blend050",
            "primary_shrink_lambda": 0.5,
            "random_mass_bag_size": 2,
            "random_mass_bag_alpha": 4.0,
            "blend_alpha": 0.5,
            "matched_shuffled_reliability_null_permutations": 0,
            "panel_seed_groups": {
                "canonical": [17, 23, 31],
                "fresh_a": [101, 103, 107],
                "fresh_b": [109, 113, 127],
            },
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 2,
            "source_weighting": "tailrisk_multipanel_shrink050_random_mass_bag_blend050",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "variance_ceiling_multiplier": 16.0,
            "primary_pooling": "seed_blend_then_equal_probability_pool",
            "reliability_floor_score": 0.05,
            "reliability_epsilon": 1.0e-8,
            "anchor_repro_tolerance": 1.0e-4,
            "primary_noninferiority_margin": 0.005,
            "weak_pass_noninferiority_margin": 0.010,
            "tailrisk_transfer_threshold": -0.010,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_source_inner_positive_union_payload(tmp_path: Path):
    payload = _tiny_multipanel_tailrisk_component_union_payload(tmp_path)
    payload["experiment"]["name"] = "virchow2_cvae_source_inner_class_conditional_positive_union_v1"
    payload["experiment"]["artifact_root"] = str(tmp_path / "virchow2_cvae_source_inner_class_conditional_positive_union_v1")
    section = payload.pop("tailrisk_multipanel_component_union")
    section.update(
        {
            "primary_method": "source_inner_class_conditional_positive_union_v1",
            "source_weighting": "source_inner_class_conditional_positive_union",
            "primary_pooling": "source_inner_selected_class_conditional_positive_union",
            "candidate_pooling_rules": [
                "arithmetic_mean",
                "positive_union_beta025",
                "positive_union_beta050",
                "positive_union_beta100",
            ],
            "positive_label": 1,
            "prediction_threshold": 0.5,
            "min_source_inner_positive_count": 5,
            "positive_union_eps": 1.0e-8,
            "source_inner_bacc_noninferiority_margin": 0.010,
            "source_inner_class0_recall_margin": 0.015,
            "source_inner_predicted_positive_rate_delta": 0.050,
            "beta100_class0_recall_margin": 0.005,
            "beta100_precision_margin": 0.010,
        }
    )
    payload["source_inner_class_conditional_positive_union"] = section
    return payload


def _tiny_fixed_beta050_positive_union_payload(tmp_path: Path):
    payload = _tiny_multipanel_tailrisk_component_union_payload(tmp_path)
    payload["experiment"]["name"] = "virchow2_cvae_fixed_beta050_positive_union_confirmation_v1"
    payload["experiment"]["artifact_root"] = str(tmp_path / "virchow2_cvae_fixed_beta050_positive_union_confirmation_v1")
    payload["inputs"]["development_positive_union_artifact_root"] = str(tmp_path / "virchow2_cvae_source_inner_class_conditional_positive_union_v1")
    payload["run_matrix"]["experiment_seeds"] = [45]
    section = payload.pop("tailrisk_multipanel_component_union")
    section.update(
        {
            "primary_method": "fixed_beta050_positive_union_confirmation_v1",
            "source_weighting": "fixed_beta050_positive_union_confirmation",
            "primary_pooling": "fixed_global_positive_union_beta050",
            "candidate_pooling_rules": [
                "arithmetic_mean",
                "positive_union_beta025",
                "positive_union_beta050",
                "positive_union_beta100",
            ],
            "fixed_pooling_rule": "positive_union_beta050",
            "fixed_beta": 0.5,
            "development_experiment_seeds": [42, 43, 44],
            "primary_confirmation_experiment_seeds": [45],
            "positive_label": 1,
            "prediction_threshold": 0.5,
            "min_source_inner_positive_count": 5,
            "positive_union_eps": 1.0e-8,
            "rare_positive_count_threshold": 10,
            "rare_positive_prevalence_threshold": 0.05,
            "source_inner_bacc_noninferiority_margin": 0.010,
            "source_inner_class0_recall_margin": 0.015,
            "source_inner_predicted_positive_rate_delta": 0.050,
            "beta100_class0_recall_margin": 0.005,
            "beta100_precision_margin": 0.010,
        }
    )
    payload["fixed_beta050_positive_union_confirmation"] = section
    return payload


def _tiny_harm_gated_positive_union_payload(tmp_path: Path):
    payload = _tiny_multipanel_tailrisk_component_union_payload(tmp_path)
    payload["experiment"]["name"] = "virchow2_cvae_source_inner_harm_gated_positive_union_v1"
    payload["experiment"]["artifact_root"] = str(tmp_path / "virchow2_cvae_source_inner_harm_gated_positive_union_v1")
    payload["run_matrix"]["experiment_seeds"] = [50, 55]
    section = payload.pop("tailrisk_multipanel_component_union")
    section.update(
        {
            "primary_method": "source_inner_harm_gated_positive_union_v1",
            "source_weighting": "source_inner_harm_gated_positive_union",
            "primary_pooling": "source_inner_harm_gated_positive_union",
            "candidate_pooling_rules": [
                "arithmetic_mean",
                "positive_union_beta025",
                "positive_union_beta050",
                "positive_union_beta100",
            ],
            "primary_selectable_rules": [
                "arithmetic_mean",
                "positive_union_beta025",
                "positive_union_beta050",
            ],
            "beta100_primary_selectable": False,
            "development_experiment_seeds": [42, 43, 44, 45, 46, 47, 48, 49],
            "primary_requested_experiment_seeds": [50],
            "reserve_experiment_seeds": [55],
            "reserve_seed_policy": "replace_incomplete_primary_seed_whole_seed_lowest_available_reserve",
            "cell_level_reserve_stitching_allowed": False,
            "selector_thresholds_frozen_before_primary": True,
            "selector_threshold_source": "retrospective_development_only",
            "selector_thresholds_may_be_changed_after_primary": False,
            "positive_label": 1,
            "prediction_threshold": 0.5,
            "min_source_inner_positive_count": 5,
            "beta050_min_source_inner_positive_count": 10,
            "positive_union_eps": 1.0e-8,
            "harm_gate_bacc_noninferiority_margin": 0.005,
            "beta025_class0_recall_margin": 0.020,
            "beta025_predicted_positive_rate_delta": 0.040,
            "beta050_class0_recall_margin": 0.015,
            "beta050_precision_margin": 0.020,
            "beta050_predicted_positive_rate_delta": 0.060,
            "rare_positive_count_threshold": 10,
            "rare_positive_prevalence_threshold": 0.05,
            "source_inner_bacc_noninferiority_margin": 0.005,
            "source_inner_class0_recall_margin": 0.015,
            "source_inner_predicted_positive_rate_delta": 0.060,
            "beta100_class0_recall_margin": 0.005,
            "beta100_precision_margin": 0.010,
        }
    )
    payload["source_inner_harm_gated_positive_union"] = section
    payload["memory"] = {"skip_nearest_neighbor_audit": True}
    return payload


def _write_tiny_prior_tailrisk_matrix(root: Path | None) -> None:
    assert root is not None
    path = root / "tables" / "tailrisk_downstream_matrix.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    centers = ["0", "1", "2", "3", "4"]
    rows = []
    for idx, center in enumerate(centers):
        rows.append(
            {
                "experiment_seed": 42,
                "heldout_center": center,
                "replicate_seed": 17,
                "prior_method": PRIMARY_TAILRISK_METHOD,
                "status": "ok",
                "bacc": 0.60 + 0.02 * idx,
                "macro_f1": 0.60 + 0.02 * idx,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _tiny_dense_tailshield_random_mass_bag_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "paired_dense_artifact_root": str(tmp_path / "missing_paired_dense"),
            "mass_bagged_artifact_root": str(tmp_path / "missing_mass_bagged"),
            "shrink050_artifact_root": str(tmp_path / "missing_shrink050"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "strict_full_run_matrix": False,
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
            "fresh_replicate_seeds": [101],
        },
        "generation": {
            "synthetic_per_class_total": 32,
            "min_per_source_per_class": 2,
        },
        "dense_tailshield_random_mass_bag": {
            "primary_method": "dense_reliability_tailshield_random_mass_bag_blend25_75",
            "random_mass_bag_size": 3,
            "random_mass_bag_alpha": 4.0,
            "dense_blend_alpha": 0.25,
            "bag_blend_alpha": 0.75,
            "alpha_curve_dense_values": [0.0, 0.25, 1.0],
            "reconstruction_probability_tolerance": 1.0e-6,
            "nontrivial_rescue_threshold": 0.02,
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 2,
            "source_weighting": "dense_reliability_tailshield_random_mass_bag_blend25_75",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "variance_ceiling_multiplier": 16.0,
            "primary_pooling": "fixed_arithmetic_probability_blend",
            "reliability_floor_score": 0.05,
            "reliability_epsilon": 1.0e-8,
            "anchor_repro_tolerance": 1.0e-4,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_harmful_source_suppression_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "paired_dense_artifact_root": str(tmp_path / "missing_paired_dense"),
            "dense_tailshield_artifact_root": str(tmp_path / "missing_dense_tailshield"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "strict_full_run_matrix": False,
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
            "fresh_replicate_seeds": [101],
        },
        "generation": {
            "synthetic_per_class_total": 32,
            "min_per_source_per_class": 2,
        },
        "harmful_source_suppression": {
            "primary_method": "source_inner_harm_suppressed_random_mass_bag_component_union_v1",
            "random_mass_bag_size": 3,
            "random_mass_bag_alpha": 4.0,
            "dirichlet_total_concentration": 16.0,
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 2,
            "source_weighting": "source_inner_harm_suppressed_random_mass_bag_component_union",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "variance_ceiling_multiplier": 16.0,
            "primary_pooling": "arithmetic_probability_ensemble",
            "reliability_floor_score": 0.05,
            "reliability_epsilon": 1.0e-8,
            "anchor_repro_tolerance": 1.0e-4,
            "min_harmfulness_observations": 6,
            "moderate_hit_rate_min": 0.50,
            "moderate_gain_min": 0.015,
            "moderate_helpful_loss_max": 0.020,
            "severe_hit_rate_min": 0.75,
            "severe_gain_min": 0.025,
            "severe_helpful_loss_max": 0.010,
            "max_suppressed_sources": 2,
            "suppression_rate_low": 0.05,
            "suppression_rate_high": 0.80,
            "oracle_harm_delta_threshold": 0.02,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_pruned_adaptive_equal_all4_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_pruned_adaptive_equal_all4_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_pruned_adaptive_equal_all4_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "d1_2_artifact_root": str(tmp_path / "missing_d1_2"),
            "component_union_artifact_root": str(tmp_path / "missing_component_union"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "min_per_source_per_class": 8,
        },
        "pruned_adaptive_equal_all4_prior": {
            "primary_method": "decentralized_pruned_adaptive_k_equal_all4_late_geom",
            "unpruned_fixed_k": 4,
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "equal_source_mass",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "variance_ceiling_multiplier": 16.0,
            "primary_pooling": "geometric",
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_paired_dense_all4_reliability_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_paired_dense_all4_reliability_confirmation_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_paired_dense_all4_reliability_confirmation_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "d1_2_artifact_root": str(tmp_path / "missing_d1_2_context"),
            "d1_4_artifact_root": str(tmp_path / "missing_d1_4_context"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "min_per_source_per_class": 8,
        },
        "paired_dense_all4_reliability": {
            "primary_method": "paired_reliability_all4_shrink050_geom",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "heldout_excluded_source_local_reliability_dense_all4",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 5,
            "gmm_max_iter": 500,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "reliability_floor_score": 0.05,
            "reliability_epsilon": 1.0e-8,
            "shrinkage_values": [0.25, 0.50],
            "primary_pooling": "weighted_geometric",
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_paired_component_coverage_audit_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_paired_component_coverage_audit_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_paired_component_coverage_audit_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "paired_reliability_artifact_root": str(tmp_path / "missing_paired_reliability_context"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "diagnostic_synthetic_per_class_total": 256,
            "min_per_source_per_class": 8,
        },
        "paired_component_coverage_audit": {
            "primary_method": "paired_reliability_all4_weighted_component_stratified128_geom",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "heldout_excluded_source_local_reliability_dense_all4",
            "component_sampling_rules": ["multinomial", "stratified_largest_remainder"],
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 5,
            "gmm_max_iter": 500,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "reliability_floor_score": 0.05,
            "reliability_epsilon": 1.0e-8,
            "primary_pooling": "weighted_geometric",
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_reliability_top3_gmm_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_reliability_top3_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_reliability_top3_gmm_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "d1_3_1_artifact_root": str(tmp_path / "missing_d1_3_1_context"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "min_per_source_per_class": 8,
            "top_k_sources": 3,
        },
        "reliability_top3_gmm_prior": {
            "primary_method": "decentralized_reliability_top3_geom_confirmation",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "source_local_reliability_top3",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.001,
            "variance_floor": 1.0e-5,
            "primary_pooling": "geometric",
            "reliability_floor_score": 0.05,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_source_inner_transfer_top3_gmm_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "min_per_source_per_class": 8,
            "top_k_sources": 3,
        },
        "source_inner_transfer_top3_gmm_prior": {
            "primary_method": "decentralized_source_inner_transfer_top3_geom_confirmation",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "source_inner_transfer_top3",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.001,
            "variance_floor": 1.0e-5,
            "primary_pooling": "geometric",
            "reliability_floor_score": 0.05,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_support_nelbo_reliability_gmm_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_support_nelbo_reliability_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_support_nelbo_reliability_gmm_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
            "support_seeds": [17],
            "support_size": 32,
            "support_size_diagnostics": [8, 16, 64],
            "align_support_and_generation_seed": True,
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "min_per_source_per_class": 8,
        },
        "support_nelbo_reliability_gmm_prior": {
            "primary_method": "decentralized_exported_adaptive_k_support_nelbo_x_reliability_weighted_geom",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "support_nelbo_x_source_local_reliability",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.001,
            "variance_floor": 1.0e-5,
            "primary_pooling": "weighted_geometric",
            "reliability_floor_score": 0.05,
            "support_nelbo_tau": 1.0,
            "tau_diagnostics": [0.5, 2.0],
            "support_alpha": 1.0,
            "reliability_alpha": 1.0,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_support8_top3_tau05_gmm_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_support8_top3_tau05_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_support8_top3_tau05_gmm_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "d1_3_artifact_root": str(tmp_path / "missing_d1_3_context"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
            "support_seeds": [17],
            "support_size": 8,
            "align_support_and_generation_seed": True,
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "min_per_source_per_class": 8,
        },
        "support8_top3_tau05_gmm_prior": {
            "primary_method": "decentralized_support8_top3_tau05_support_nelbo_x_reliability_geom",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "support_nelbo_x_source_local_reliability_top3",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.001,
            "variance_floor": 1.0e-5,
            "primary_pooling": "weighted_geometric",
            "reliability_floor_score": 0.05,
            "support_nelbo_tau": 0.5,
            "top_k_sources": 3,
            "support_alpha": 1.0,
            "reliability_alpha": 1.0,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_source_union_k24_gmm_config(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
    source_union_gmm_root: Path,
    balanced_gmm_root: Path,
):
    return parse_source_union_k24_gmm_prior_config(
        _tiny_source_union_k24_gmm_payload(
            tmp_path,
            repair_root,
            sampling_root,
            prior_root,
            covariance_root,
            source_union_gmm_root,
            balanced_gmm_root,
        ),
        base_dir=tmp_path,
    )


def _tiny_source_union_k24_gmm_payload(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
    source_union_gmm_root: Path,
    balanced_gmm_root: Path,
):
    return {
        "experiment": {
            "name": "virchow2_cvae_source_union_k24_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_source_union_k24_gmm_prior_v1"),
            "primary_variant": "source_union_pca64_beta001_diagnostic",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(repair_root),
            "sampling_artifact_root": str(sampling_root),
            "prior_calibration_artifact_root": str(prior_root),
            "covariance_confirmation_artifact_root": str(covariance_root),
            "source_union_gmm_artifact_root": str(source_union_gmm_root),
            "balanced_gmm_artifact_root": str(balanced_gmm_root),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "k24_gmm_prior": {
            "primary_method": "source_union_cc_diag_gmm_k24_prior_sample",
            "gmm_components": 24,
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 2,
            "gmm_max_iter": 200,
            "gmm_weight_floor": 0.005,
            "min_class_train_count": 24,
            "min_effective_gmm_components": 1,
            "min_train_count_per_effective_component": 1,
            "posterior_noise_scale": 0.0,
            "diagnostic_gmm_components": [20, 32],
            "budget256_synthetic_per_class_total": 256,
            "center_cap_samples_per_center_class": 8,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _write_tiny_cache(root: Path, *, seed: int, mono_test_centers: set[str] | None = None) -> None:
    rng = np.random.default_rng(123)
    mono_test_centers = set(mono_test_centers or set())
    for split, per_class in (("train", 24), ("test", 26)):
        embeddings = []
        metadata = []
        for center_idx, center in enumerate(["0", "1", "2", "3", "4"]):
            for label in (0, 1):
                mean = np.array([center_idx * 0.5, label * 1.2, center_idx * 0.1, label * 0.3])
                for idx in range(per_class):
                    embeddings.append(mean + rng.normal(0.0, 0.05, size=4))
                    observed_label = 0 if split == "test" and center in mono_test_centers else label
                    metadata.append(
                        {
                            "sample_id": f"{split}_c{center}_y{observed_label}_{label}_{idx}",
                            "center": center,
                            "label": observed_label,
                            "split": split,
                        }
                    )
        path = root / f"seed{seed}" / "embeddings" / f"{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            embeddings=np.asarray(embeddings, dtype=float),
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
