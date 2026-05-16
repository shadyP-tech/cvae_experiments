from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Tuple

from src.eval.evaluators.learned_utility_proxies import _DEFAULT_ALPHA_GRID
from src.eval.evaluators.support_response_routing import (
    SupportResponseConfig,
    parse_support_response_config,
)


@dataclass(frozen=True)
class HybridConfig:
    enabled: bool
    alphas: Tuple[float, ...]
    primary_norm_policy: str
    sensitivity_norm_policy: str
    run_sensitivity: bool
    tie_policy: str
    min_rank_improvement_abs: float
    min_gap_pct_improvement_abs: float
    max_top1_drop_abs: float


@dataclass(frozen=True)
class CompatibilityResearchConfig:
    enable_random_rank_floor: bool
    enable_random_score_floor: bool
    run_expert_label_permutation: bool
    run_metadata_permutation: bool
    permutation_repeats: int
    save_distribution_plots: bool
    uplift_reference_method: str
    strong_spearman_uplift: float
    strong_top1_uplift: float
    strong_gap_reduction: float
    weak_spearman_uplift: float
    weak_top1_uplift: float
    weak_gap_reduction: float
    decision_policy_version: str
    instability_std_threshold: float
    top1_uplift_std_threshold: float
    spearman_uplift_std_threshold: float
    gap_pct_reduction_std_threshold: float
    instability_sign_inconsistency_min_count: int
    min_positive_fraction: float
    ci_level: float
    ci_bootstrap_reps: int
    ci_bootstrap_seed: int
    allow_missing_domain_breakdown_as_diagnostic: bool


@dataclass(frozen=True)
class ResidualRoutingConfig:
    enabled: bool
    residual_policy_version: str
    models: Tuple[str, ...]
    thresholds: Tuple[float, ...]
    feature_sets: Tuple[str, ...]
    adoption_feature_sets: Tuple[str, ...]
    diagnostic_feature_sets: Tuple[str, ...]
    allow_calibrated_adoption: bool
    harmful_override_max: float
    gap_regression_max: float
    catastrophic_top1_floor: float
    selection_metric: str
    unconstrained_reference_method: str
    ridge_l2: float


@dataclass(frozen=True)
class AutoencoderProxyConfig:
    enabled: bool
    hidden_dim: int
    latent_dim: int
    learning_rate: float
    epochs: int
    patience: int
    batch_size: int
    score_normalization: str
    score_normalization_eps: float
    margin_threshold: float
    run_diagnostics: bool
    ae_first: "AEFirstRoutingConfig"
    utility_calibrator: "AEUtilityCalibratorConfig"


@dataclass(frozen=True)
class AEFirstRoutingConfig:
    enabled: bool
    primary_method: str
    fallback_baseline: str
    margin_thresholds: Tuple[float, ...]
    metadata_auxiliary_features: bool
    ae_z_sigma_floor_mode: str
    ae_z_sigma_floor_quantile: float
    min_ae_coverage_rate_for_weak_pass: float
    min_ae_coverage_rate_for_pass: float
    max_top1_drop_abs: float
    max_raw_spearman_drop_abs: float
    max_gap_pct_degradation: float


@dataclass(frozen=True)
class AEUtilityCalibratorConfig:
    enabled: bool
    primary_method: str
    model_types: Tuple[str, ...]
    primary_model_type: str
    diagnostic_model_types: Tuple[str, ...]
    fallback_policy: str
    feature_sets_primary: Tuple[str, ...]
    feature_sets_diagnostic: Tuple[str, ...]
    delta_thresholds: Tuple[float, ...]
    margin_thresholds: Tuple[float, ...]
    consensus_thresholds: Tuple[float, ...]
    uncertainty_multiplier: float
    ensemble_strategy: str
    abstention_correct_gap_pct_epsilon: float
    min_pseudo_domain_positive_rate: float
    max_pseudo_domain_gain_share: float
    max_source_inner_fold_gain_share: float
    max_top1_drop_vs_ae_argmin_abs: float
    max_spearman_drop_vs_ae_argmin_abs: float
    max_gap_pct_degradation_vs_ae_argmin: float
    max_top1_drop_vs_metadata_abs: float
    max_spearman_drop_vs_metadata_abs: float
    max_gap_pct_degradation_vs_metadata: float
    ridge_l2: float
    selection_mode: str
    min_strict_improvement_precision: float
    min_strict_improvement_precision_lcb: float
    min_active_override_count: int
    min_active_override_rate: float
    min_net_gain_vs_ae_argmin: float
    neutral_override_gap_pct_band: float
    max_worst_pseudo_domain_gap_degradation_pp: float
    precision_bootstrap_reps: int
    precision_bootstrap_seed: int
    diagnostic_precision_thresholds: Tuple[float, ...]
    v1_guard_min_gap_delta_vs_v1_lcb_pp: float
    v1_guard_max_top1_drop_vs_v1_abs: float
    v1_guard_max_spearman_drop_vs_v1_abs: float
    v1_guard_max_worst_pseudo_domain_gap_degradation_vs_v1_pp: float
    v1_guard_max_harmful_override_rate_ucb: float
    harm_veto_score_model: str
    harm_veto_thresholds: Tuple[float, ...]
    harm_veto_min_active_v1_override_count_source_inner: int
    harm_veto_min_veto_count_source_inner: int
    harm_veto_min_harmful_v1_override_count_source_inner: int
    harm_veto_min_strict_harm_prevention_precision_lcb: float
    harm_veto_max_false_veto_rate_ucb: float
    harm_veto_min_retained_v1_override_gain_rate: float
    harm_veto_min_active_override_rate_ratio_vs_v1: float
    harm_veto_min_gap_delta_vs_v1_lcb_pp: float
    recall_scoring_policy: str
    recall_budget_rates: Tuple[float, ...]
    recall_budget_scope: str
    recall_min_v1_abstention_count_source_inner: int
    recall_min_recall_override_count_source_inner: int
    recall_min_recall_override_count_source_inner_for_pass: int
    recall_min_strict_recall_precision: float
    recall_min_strict_recall_precision_lcb: float
    recall_max_harmful_recall_rate_ucb: float
    recall_min_net_gain_vs_v1_source_inner: float
    recall_min_gap_delta_vs_v1_lcb_pp: float
    recall_min_gap_delta_vs_v1_lcb_pp_for_pass: float
    recall_max_active_override_rate_ratio_vs_v1: float
    recall_diagnostic_active_override_rate_ratio_upper_bound: float
    recall_max_worst_pseudo_domain_gap_degradation_vs_v1_pp: float


@dataclass(frozen=True)
class SourceReliabilityConfig:
    enabled: bool
    primary_method: str
    fallback_method: str
    candidate_methods: Tuple[str, ...]
    group_key_candidates: Tuple[str, ...]
    pseudo_domain_strategy: str
    n_pseudo_domains_per_source: int
    min_pseudo_domains_per_source: int
    min_groups_per_pseudo_domain: int
    min_samples_per_pseudo_domain: int
    min_candidate_pool_size: int
    pca_dim: int
    kmeans_iterations: int
    aggregation_unit: str
    min_source_inner_units: int
    min_parent_domains: int
    min_units_per_parent_for_gain_share: int
    max_top1_drop_abs: float
    max_spearman_drop_abs: float
    max_gap_pct_degradation: float
    max_worst_unit_gap_degradation: float
    min_gap_reduction_vs_fallback: float
    min_positive_unit_rate: float
    min_positive_parent_rate: float
    max_positive_gain_share: float
    require_parent_holdout_guard: bool


@dataclass(frozen=True)
class LearnedUtilityConfig:
    pair_batch_size: int
    include_metadata_features: bool
    predictors: Tuple[str, ...]
    mlp_cfg: Dict[str, Any]
    pairwise_cfg: Dict[str, Any]
    hybrid: HybridConfig
    compatibility: CompatibilityResearchConfig
    residual: ResidualRoutingConfig
    autoencoder: AutoencoderProxyConfig
    source_reliability: SourceReliabilityConfig
    support_response: SupportResponseConfig


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_threshold(value: Any) -> float:
    if isinstance(value, str) and value.strip().lower() in {"inf", "__inf__"}:
        return float("inf")
    parsed = float(value)
    if not math.isfinite(parsed):
        return float("inf")
    return parsed


def _parse_learned_utility_config(learned_cfg: Dict[str, Any]) -> LearnedUtilityConfig:
    pair_batch_size = int(learned_cfg.get("scoring", {}).get("pair_batch_size", 4096))
    include_metadata_features = bool(learned_cfg.get("pair_features", {}).get("include_metadata_features", False))
    predictors = tuple(str(v) for v in learned_cfg.get("predictors", ["linear_regressor", "mlp_regressor"]))

    predictor_params = learned_cfg.get("predictor_params", {})
    predictor_params_dict = _as_dict(predictor_params)
    mlp_cfg = predictor_params_dict.get("mlp", {}) if isinstance(predictor_params, dict) else {}
    pairwise_cfg = predictor_params_dict.get("pairwise_ranker", {}) if isinstance(predictor_params, dict) else {}

    hybrid_cfg = _as_dict(learned_cfg.get("hybrid_scoring", {}))
    hybrid_accept_cfg = _as_dict(hybrid_cfg.get("acceptance", {}))
    hybrid = HybridConfig(
        enabled=bool((hybrid_cfg or {}).get("enabled", False)),
        alphas=tuple(float(v) for v in (hybrid_cfg or {}).get("alphas", _DEFAULT_ALPHA_GRID)),
        primary_norm_policy=str((hybrid_cfg or {}).get("normalization_primary", "per_query_zscore")).strip().lower(),
        sensitivity_norm_policy=str((hybrid_cfg or {}).get("normalization_sensitivity", "per_query_minmax")).strip().lower(),
        run_sensitivity=bool((hybrid_cfg or {}).get("run_sensitivity", True)),
        tie_policy=str((hybrid_cfg or {}).get("tie_policy", "stable_expert_index")).strip().lower(),
        min_rank_improvement_abs=float((hybrid_accept_cfg or {}).get("min_mean_rank_improvement_abs", 0.05)),
        min_gap_pct_improvement_abs=float(
            (hybrid_accept_cfg or {}).get("min_mean_oracle_gap_pct_improvement_abs", 0.50)
        ),
        max_top1_drop_abs=float((hybrid_accept_cfg or {}).get("max_top1_drop_abs", 0.0)),
    )

    compatibility_cfg = learned_cfg.get("compatibility_research", {}) if isinstance(learned_cfg, dict) else {}
    if compatibility_cfg is None:
        compatibility_cfg = {}
    compatibility_cfg = _as_dict(compatibility_cfg)
    floors_cfg = _as_dict(compatibility_cfg.get("floors", {}))
    permutation_cfg = _as_dict(compatibility_cfg.get("permutation_tests", {}))
    diagnostics_cfg = _as_dict(compatibility_cfg.get("diagnostics", {}))
    gate_cfg = _as_dict(compatibility_cfg.get("gate", {}))
    strong_gate = _as_dict(gate_cfg.get("strong", {}))
    weak_gate = _as_dict(gate_cfg.get("weak", {}))
    instability_gate = _as_dict(gate_cfg.get("instability", {}))

    compatibility = CompatibilityResearchConfig(
        enable_random_rank_floor=bool((floors_cfg or {}).get("random_rank_floor", True)),
        enable_random_score_floor=bool((floors_cfg or {}).get("random_score_floor", True)),
        run_expert_label_permutation=bool((permutation_cfg or {}).get("expert_label_permutation", True)),
        run_metadata_permutation=bool((permutation_cfg or {}).get("metadata_permutation", True)),
        permutation_repeats=int((permutation_cfg or {}).get("repeats", 200)),
        save_distribution_plots=bool((diagnostics_cfg or {}).get("save_distribution_plots", True)),
        uplift_reference_method=str((gate_cfg or {}).get("uplift_reference_method", "metadata_routing")),
        strong_spearman_uplift=float((strong_gate or {}).get("spearman_uplift_min", 0.05)),
        strong_top1_uplift=float((strong_gate or {}).get("top1_uplift_min", 0.10)),
        strong_gap_reduction=float((strong_gate or {}).get("oracle_gap_pct_reduction_min", 5.0)),
        weak_spearman_uplift=float((weak_gate or {}).get("spearman_uplift_min", 0.025)),
        weak_top1_uplift=float((weak_gate or {}).get("top1_uplift_min", 0.05)),
        weak_gap_reduction=float((weak_gate or {}).get("oracle_gap_pct_reduction_min", 2.5)),
        decision_policy_version=str((gate_cfg or {}).get("decision_policy_version", "sign_ci_v2")),
        instability_std_threshold=float((instability_gate or {}).get("std_threshold", 0.05)),
        top1_uplift_std_threshold=float((instability_gate or {}).get("top1_uplift_std_threshold", 0.05)),
        spearman_uplift_std_threshold=float(
            (instability_gate or {}).get("spearman_uplift_std_threshold", 0.05)
        ),
        gap_pct_reduction_std_threshold=float(
            (instability_gate or {}).get("gap_pct_reduction_std_threshold", 3.0)
        ),
        instability_sign_inconsistency_min_count=int(
            (instability_gate or {}).get("sign_inconsistency_min_count", 2)
        ),
        min_positive_fraction=float((instability_gate or {}).get("min_positive_fraction", 0.67)),
        ci_level=float((instability_gate or {}).get("ci_level", 0.95)),
        ci_bootstrap_reps=int((instability_gate or {}).get("ci_bootstrap_reps", 10000)),
        ci_bootstrap_seed=int((instability_gate or {}).get("ci_bootstrap_seed", 1337)),
        allow_missing_domain_breakdown_as_diagnostic=bool(
            (instability_gate or {}).get("allow_missing_domain_breakdown_as_diagnostic", False)
        ),
    )

    residual_cfg = _as_dict(learned_cfg.get("residual_routing", {}))
    residual_policy_version = str(
        (residual_cfg or {}).get("residual_policy_version", "metadata_residual_v1")
    )
    feature_sets = tuple(
        str(v).strip().lower()
        for v in (residual_cfg or {}).get("feature_sets", ["minimal", "latent", "calibrated"])
    )
    residual = ResidualRoutingConfig(
        enabled=bool((residual_cfg or {}).get("enabled", False)),
        residual_policy_version=str(residual_policy_version),
        models=tuple(str(v) for v in (residual_cfg or {}).get("models", ["ridge"])),
        thresholds=tuple(
            _parse_threshold(v)
            for v in (residual_cfg or {}).get("thresholds", [0, 0.01, 0.05, 0.10, 0.25, 0.50, "inf"])
        ),
        feature_sets=feature_sets,
        adoption_feature_sets=tuple(
            str(v).strip().lower()
            for v in (residual_cfg or {}).get(
                "adoption_feature_sets",
                ["minimal", "latent"] if residual_policy_version == "metadata_residual_safe_override_v2" else feature_sets,
            )
        ),
        diagnostic_feature_sets=tuple(
            str(v).strip().lower()
            for v in (residual_cfg or {}).get(
                "diagnostic_feature_sets",
                ["calibrated"] if residual_policy_version == "metadata_residual_safe_override_v2" else (),
            )
        ),
        allow_calibrated_adoption=bool((residual_cfg or {}).get("allow_calibrated_adoption", False)),
        harmful_override_max=float((residual_cfg or {}).get("harmful_override_max", 0.05)),
        gap_regression_max=float((residual_cfg or {}).get("gap_regression_max", 2.0)),
        catastrophic_top1_floor=float((residual_cfg or {}).get("catastrophic_top1_floor", -0.05)),
        selection_metric=str(
            (residual_cfg or {}).get("selection_metric", "validation_safe_gap_then_top1")
        ).strip().lower(),
        unconstrained_reference_method=str(
            (residual_cfg or {}).get("unconstrained_reference_method", "pairwise_ranker_metadata_only")
        ),
        ridge_l2=float((residual_cfg or {}).get("ridge_l2", 1e-4)),
    )

    autoencoder_cfg = _as_dict(learned_cfg.get("autoencoder_proxy", {}))
    ae_first_cfg = _as_dict((autoencoder_cfg or {}).get("ae_first_routing", {}))
    ae_first_risk = _as_dict((ae_first_cfg or {}).get("risk_gates", {}))
    ae_first = AEFirstRoutingConfig(
        enabled=bool((ae_first_cfg or {}).get("enabled", False)),
        primary_method=str((ae_first_cfg or {}).get("primary_method", "ae_first_margin_gated_v1")),
        fallback_baseline=str((ae_first_cfg or {}).get("fallback_baseline", "source_prior_fallback")),
        margin_thresholds=tuple(
            _parse_threshold(v)
            for v in (ae_first_cfg or {}).get(
                "margin_thresholds",
                [0.0, 0.05, 0.10, 0.25, 0.50, 1.0, "__inf__"],
            )
        ),
        metadata_auxiliary_features=bool((ae_first_cfg or {}).get("metadata_auxiliary_features", True)),
        ae_z_sigma_floor_mode=str(
            (ae_first_cfg or {}).get("ae_z_sigma_floor_mode", "global_source_val_std_quantile")
        ).strip().lower(),
        ae_z_sigma_floor_quantile=float((ae_first_cfg or {}).get("ae_z_sigma_floor_quantile", 0.05)),
        min_ae_coverage_rate_for_weak_pass=float(
            (ae_first_cfg or {}).get("min_ae_coverage_rate_for_weak_pass", 0.10)
        ),
        min_ae_coverage_rate_for_pass=float((ae_first_cfg or {}).get("min_ae_coverage_rate_for_pass", 0.20)),
        max_top1_drop_abs=float((ae_first_risk or {}).get("max_top1_drop_abs", 0.02)),
        max_raw_spearman_drop_abs=float((ae_first_risk or {}).get("max_raw_spearman_drop_abs", 0.03)),
        max_gap_pct_degradation=float((ae_first_risk or {}).get("max_gap_pct_degradation", 1.0)),
    )
    utility_calibrator_cfg = _as_dict((autoencoder_cfg or {}).get("utility_calibrator", {}))
    utility_calibrator_risk = _as_dict((utility_calibrator_cfg or {}).get("risk_gates", {}))
    precision_selection_cfg = _as_dict((utility_calibrator_cfg or {}).get("precision_selection", {}))
    v1_guard_cfg = _as_dict((precision_selection_cfg or {}).get("v1_guard", {}))
    harm_veto_cfg = _as_dict((utility_calibrator_cfg or {}).get("harm_veto", {}))
    recall_expansion_cfg = _as_dict((utility_calibrator_cfg or {}).get("recall_expansion", {}))
    utility_calibrator = AEUtilityCalibratorConfig(
        enabled=bool((utility_calibrator_cfg or {}).get("enabled", False)),
        primary_method=str(
            (utility_calibrator_cfg or {}).get(
                "primary_method",
                "ae_utility_calibrated_safe_override_v1",
            )
        ),
        model_types=tuple(str(v).strip().lower() for v in (utility_calibrator_cfg or {}).get("model_types", ["ridge_delta"])),
        primary_model_type=str((utility_calibrator_cfg or {}).get("primary_model_type", "ridge_delta")).strip().lower(),
        diagnostic_model_types=tuple(
            str(v).strip().lower()
            for v in (utility_calibrator_cfg or {}).get("diagnostic_model_types", ["pairwise_ranker"])
        ),
        fallback_policy=str((utility_calibrator_cfg or {}).get("fallback_policy", "ae_argmin_zscore")).strip(),
        feature_sets_primary=tuple(
            str(v).strip().lower()
            for v in (utility_calibrator_cfg or {}).get("feature_sets_primary", ["ae_core", "ae_quality"])
        ),
        feature_sets_diagnostic=tuple(
            str(v).strip().lower()
            for v in (utility_calibrator_cfg or {}).get("feature_sets_diagnostic", ["ae_metadata", "ae_combined"])
        ),
        delta_thresholds=tuple(
            _parse_threshold(v)
            for v in (utility_calibrator_cfg or {}).get(
                "delta_thresholds",
                [0.0, 0.01, 0.025, 0.05, 0.10, "__inf__"],
            )
        ),
        margin_thresholds=tuple(
            _parse_threshold(v)
            for v in (utility_calibrator_cfg or {}).get(
                "margin_thresholds",
                [0.0, 0.05, 0.10, 0.25],
            )
        ),
        consensus_thresholds=tuple(
            float(v)
            for v in (utility_calibrator_cfg or {}).get(
                "consensus_thresholds",
                [0.60, 0.75, 1.00],
            )
        ),
        uncertainty_multiplier=float((utility_calibrator_cfg or {}).get("uncertainty_multiplier", 1.0)),
        ensemble_strategy=str(
            (utility_calibrator_cfg or {}).get("ensemble_strategy", "source_domain_leave_one_plus_full")
        ).strip().lower(),
        abstention_correct_gap_pct_epsilon=float(
            (utility_calibrator_cfg or {}).get("abstention_correct_gap_pct_epsilon", 1.0)
        ),
        min_pseudo_domain_positive_rate=float(
            _as_dict((utility_calibrator_cfg or {}).get("source_inner_stability_gates", {})).get(
                "min_pseudo_domain_positive_rate",
                0.80,
            )
        ),
        max_pseudo_domain_gain_share=float(
            _as_dict((utility_calibrator_cfg or {}).get("source_inner_stability_gates", {})).get(
                "max_pseudo_domain_gain_share",
                0.50,
            )
        ),
        max_source_inner_fold_gain_share=float(
            _as_dict((utility_calibrator_cfg or {}).get("source_inner_stability_gates", {})).get(
                "max_source_inner_fold_gain_share",
                0.50,
            )
        ),
        max_top1_drop_vs_ae_argmin_abs=float(
            (utility_calibrator_risk or {}).get("max_top1_drop_vs_ae_argmin_abs", 0.02)
        ),
        max_spearman_drop_vs_ae_argmin_abs=float(
            (utility_calibrator_risk or {}).get("max_spearman_drop_vs_ae_argmin_abs", 0.03)
        ),
        max_gap_pct_degradation_vs_ae_argmin=float(
            (utility_calibrator_risk or {}).get("max_gap_pct_degradation_vs_ae_argmin", 1.0)
        ),
        max_top1_drop_vs_metadata_abs=float(
            (utility_calibrator_risk or {}).get("max_top1_drop_vs_metadata_abs", 0.02)
        ),
        max_spearman_drop_vs_metadata_abs=float(
            (utility_calibrator_risk or {}).get("max_spearman_drop_vs_metadata_abs", 0.03)
        ),
        max_gap_pct_degradation_vs_metadata=float(
            (utility_calibrator_risk or {}).get("max_gap_pct_degradation_vs_metadata", 1.0)
        ),
        ridge_l2=float((utility_calibrator_cfg or {}).get("ridge_l2", 1e-4)),
        selection_mode=str((utility_calibrator_cfg or {}).get("selection_mode", "")).strip().lower(),
        min_strict_improvement_precision=float(
            (precision_selection_cfg or {}).get("min_strict_improvement_precision", 0.75)
        ),
        min_strict_improvement_precision_lcb=float(
            (precision_selection_cfg or {}).get("min_strict_improvement_precision_lcb", 0.60)
        ),
        min_active_override_count=int((precision_selection_cfg or {}).get("min_active_override_count", 10)),
        min_active_override_rate=float((precision_selection_cfg or {}).get("min_active_override_rate", 0.10)),
        min_net_gain_vs_ae_argmin=float((precision_selection_cfg or {}).get("min_net_gain_vs_ae_argmin", 0.0)),
        neutral_override_gap_pct_band=float(
            (precision_selection_cfg or {}).get(
                "neutral_override_gap_pct_band",
                (harm_veto_cfg or {}).get(
                    "neutral_override_gap_pct_band",
                    (recall_expansion_cfg or {}).get("neutral_override_gap_pct_band", 0.25),
                ),
            )
        ),
        max_worst_pseudo_domain_gap_degradation_pp=float(
            (precision_selection_cfg or {}).get("max_worst_pseudo_domain_gap_degradation_pp", 1.0)
        ),
        precision_bootstrap_reps=int(
            (precision_selection_cfg or {}).get(
                "bootstrap_reps",
                (harm_veto_cfg or {}).get("bootstrap_reps", (recall_expansion_cfg or {}).get("bootstrap_reps", 2000)),
            )
        ),
        precision_bootstrap_seed=int(
            (precision_selection_cfg or {}).get(
                "bootstrap_seed",
                (harm_veto_cfg or {}).get("bootstrap_seed", (recall_expansion_cfg or {}).get("bootstrap_seed", 1337)),
            )
        ),
        diagnostic_precision_thresholds=tuple(
            float(v)
            for v in (precision_selection_cfg or {}).get(
                "diagnostic_precision_thresholds",
                [0.70, 0.75, 0.80, 0.85],
            )
        ),
        v1_guard_min_gap_delta_vs_v1_lcb_pp=float((v1_guard_cfg or {}).get("min_gap_delta_vs_v1_lcb_pp", -0.25)),
        v1_guard_max_top1_drop_vs_v1_abs=float((v1_guard_cfg or {}).get("max_top1_drop_vs_v1_abs", 0.02)),
        v1_guard_max_spearman_drop_vs_v1_abs=float((v1_guard_cfg or {}).get("max_spearman_drop_vs_v1_abs", 0.03)),
        v1_guard_max_worst_pseudo_domain_gap_degradation_vs_v1_pp=float(
            (v1_guard_cfg or {}).get("max_worst_pseudo_domain_gap_degradation_vs_v1_pp", 1.0)
        ),
        v1_guard_max_harmful_override_rate_ucb=float((v1_guard_cfg or {}).get("max_harmful_override_rate_ucb", 0.30)),
        harm_veto_score_model=str((harm_veto_cfg or {}).get("veto_score_model", "logistic_harm_score")).strip().lower(),
        harm_veto_thresholds=tuple(
            _parse_threshold(v)
            for v in (harm_veto_cfg or {}).get(
                "veto_thresholds",
                [0.50, 0.60, 0.70, 0.80, 0.90, "__inf__"],
            )
        ),
        harm_veto_min_active_v1_override_count_source_inner=int(
            (harm_veto_cfg or {}).get("min_active_v1_override_count_source_inner", 12)
        ),
        harm_veto_min_veto_count_source_inner=int((harm_veto_cfg or {}).get("min_veto_count_source_inner", 6)),
        harm_veto_min_harmful_v1_override_count_source_inner=int(
            (harm_veto_cfg or {}).get("min_harmful_v1_override_count_source_inner", 3)
        ),
        harm_veto_min_strict_harm_prevention_precision_lcb=float(
            (harm_veto_cfg or {}).get("min_strict_harm_prevention_precision_lcb", 0.50)
        ),
        harm_veto_max_false_veto_rate_ucb=float((harm_veto_cfg or {}).get("max_false_veto_rate_ucb", 0.40)),
        harm_veto_min_retained_v1_override_gain_rate=float(
            (harm_veto_cfg or {}).get("min_retained_v1_override_gain_rate", 0.85)
        ),
        harm_veto_min_active_override_rate_ratio_vs_v1=float(
            (harm_veto_cfg or {}).get("min_active_override_rate_ratio_vs_v1", 0.80)
        ),
        harm_veto_min_gap_delta_vs_v1_lcb_pp=float((harm_veto_cfg or {}).get("min_gap_delta_vs_v1_lcb_pp", -0.10)),
        recall_scoring_policy=str(
            (recall_expansion_cfg or {}).get("scoring_policy", "ridge_delta_best_non_anchor")
        ).strip().lower(),
        recall_budget_rates=tuple(
            float(v)
            for v in (recall_expansion_cfg or {}).get(
                "recall_budget_rates",
                [0.0, 0.005, 0.01, 0.02, 0.05, 0.10],
            )
        ),
        recall_budget_scope=str(
            (recall_expansion_cfg or {}).get("budget_scope", "v1_abstentions_per_fold")
        ).strip().lower(),
        recall_min_v1_abstention_count_source_inner=int(
            (recall_expansion_cfg or {}).get("min_v1_abstention_count_source_inner", 50)
        ),
        recall_min_recall_override_count_source_inner=int(
            (recall_expansion_cfg or {}).get("min_recall_override_count_source_inner", 10)
        ),
        recall_min_recall_override_count_source_inner_for_pass=int(
            (recall_expansion_cfg or {}).get("min_recall_override_count_source_inner_for_pass", 20)
        ),
        recall_min_strict_recall_precision=float(
            (recall_expansion_cfg or {}).get("min_strict_recall_precision", 0.70)
        ),
        recall_min_strict_recall_precision_lcb=float(
            (recall_expansion_cfg or {}).get("min_strict_recall_precision_lcb", 0.50)
        ),
        recall_max_harmful_recall_rate_ucb=float(
            (recall_expansion_cfg or {}).get("max_harmful_recall_rate_ucb", 0.35)
        ),
        recall_min_net_gain_vs_v1_source_inner=float(
            (recall_expansion_cfg or {}).get("min_net_gain_vs_v1_source_inner", 0.0)
        ),
        recall_min_gap_delta_vs_v1_lcb_pp=float(
            (recall_expansion_cfg or {}).get("min_gap_delta_vs_v1_lcb_pp", -0.05)
        ),
        recall_min_gap_delta_vs_v1_lcb_pp_for_pass=float(
            (recall_expansion_cfg or {}).get("min_gap_delta_vs_v1_lcb_pp_for_pass", 0.0)
        ),
        recall_max_active_override_rate_ratio_vs_v1=float(
            (recall_expansion_cfg or {}).get("max_active_override_rate_ratio_vs_v1", 1.20)
        ),
        recall_diagnostic_active_override_rate_ratio_upper_bound=float(
            (recall_expansion_cfg or {}).get("diagnostic_active_override_rate_ratio_upper_bound", 1.35)
        ),
        recall_max_worst_pseudo_domain_gap_degradation_vs_v1_pp=float(
            (recall_expansion_cfg or {}).get("max_worst_pseudo_domain_gap_degradation_vs_v1_pp", 1.0)
        ),
    )
    autoencoder = AutoencoderProxyConfig(
        enabled=bool((autoencoder_cfg or {}).get("enabled", False)),
        hidden_dim=int((autoencoder_cfg or {}).get("hidden_dim", 256)),
        latent_dim=int((autoencoder_cfg or {}).get("latent_dim", 32)),
        learning_rate=float((autoencoder_cfg or {}).get("learning_rate", 1e-3)),
        epochs=int((autoencoder_cfg or {}).get("epochs", 25)),
        patience=int((autoencoder_cfg or {}).get("patience", 5)),
        batch_size=int((autoencoder_cfg or {}).get("batch_size", 512)),
        score_normalization=str((autoencoder_cfg or {}).get("score_normalization", "source_val_zscore"))
        .strip()
        .lower(),
        score_normalization_eps=float((autoencoder_cfg or {}).get("score_normalization_eps", 1e-6)),
        margin_threshold=float((autoencoder_cfg or {}).get("margin_threshold", 0.0)),
        run_diagnostics=bool((autoencoder_cfg or {}).get("run_diagnostics", True)),
        ae_first=ae_first,
        utility_calibrator=utility_calibrator,
    )

    source_reliability_cfg = _as_dict(learned_cfg.get("source_reliability", {}))
    reliability_selection_cfg = _as_dict((source_reliability_cfg or {}).get("reliability_selection", {}))
    source_reliability = SourceReliabilityConfig(
        enabled=bool((source_reliability_cfg or {}).get("enabled", False)),
        primary_method=str(
            (source_reliability_cfg or {}).get(
                "primary_method",
                "source_subdomain_reliability_selected_router_v1",
            )
        ),
        fallback_method=str((source_reliability_cfg or {}).get("fallback_method", "ae_argmin_zscore")),
        candidate_methods=tuple(
            str(v).strip()
            for v in (source_reliability_cfg or {}).get(
                "candidate_methods",
                ["pairwise_ranker_ae_only", "pairwise_ranker_ae_combined"],
            )
        ),
        group_key_candidates=tuple(
            str(v).strip()
            for v in (source_reliability_cfg or {}).get(
                "group_key_candidates",
                ["patient_id", "slide_id", "case_id"],
            )
        ),
        pseudo_domain_strategy=str(
            (source_reliability_cfg or {}).get(
                "pseudo_domain_strategy",
                "per_parent_group_embedding_kmeans",
            )
        ).strip().lower(),
        n_pseudo_domains_per_source=int((source_reliability_cfg or {}).get("n_pseudo_domains_per_source", 4)),
        min_pseudo_domains_per_source=int((source_reliability_cfg or {}).get("min_pseudo_domains_per_source", 2)),
        min_groups_per_pseudo_domain=int((source_reliability_cfg or {}).get("min_groups_per_pseudo_domain", 3)),
        min_samples_per_pseudo_domain=int((source_reliability_cfg or {}).get("min_samples_per_pseudo_domain", 25)),
        min_candidate_pool_size=int((source_reliability_cfg or {}).get("min_candidate_pool_size", 2)),
        pca_dim=int((source_reliability_cfg or {}).get("pca_dim", 16)),
        kmeans_iterations=int((source_reliability_cfg or {}).get("kmeans_iterations", 50)),
        aggregation_unit=str(
            (reliability_selection_cfg or {}).get(
                "aggregation_unit",
                "parent_domain_x_pseudo_domain_macro",
            )
        ).strip().lower(),
        min_source_inner_units=int((reliability_selection_cfg or {}).get("min_source_inner_units", 6)),
        min_parent_domains=int((reliability_selection_cfg or {}).get("min_parent_domains", 2)),
        min_units_per_parent_for_gain_share=int(
            (reliability_selection_cfg or {}).get("min_units_per_parent_for_gain_share", 2)
        ),
        max_top1_drop_abs=float((reliability_selection_cfg or {}).get("max_top1_drop_abs", 0.02)),
        max_spearman_drop_abs=float((reliability_selection_cfg or {}).get("max_spearman_drop_abs", 0.03)),
        max_gap_pct_degradation=float((reliability_selection_cfg or {}).get("max_gap_pct_degradation", 1.0)),
        max_worst_unit_gap_degradation=float(
            (reliability_selection_cfg or {}).get("max_worst_unit_gap_degradation", 2.0)
        ),
        min_gap_reduction_vs_fallback=float(
            (reliability_selection_cfg or {}).get("min_gap_reduction_vs_fallback", 0.0)
        ),
        min_positive_unit_rate=float((reliability_selection_cfg or {}).get("min_positive_unit_rate", 0.60)),
        min_positive_parent_rate=float((reliability_selection_cfg or {}).get("min_positive_parent_rate", 0.50)),
        max_positive_gain_share=float((reliability_selection_cfg or {}).get("max_positive_gain_share", 0.60)),
        require_parent_holdout_guard=bool(
            (reliability_selection_cfg or {}).get("require_parent_holdout_guard", True)
        ),
    )

    return LearnedUtilityConfig(
        pair_batch_size=pair_batch_size,
        include_metadata_features=include_metadata_features,
        predictors=predictors,
        mlp_cfg=mlp_cfg,
        pairwise_cfg=pairwise_cfg,
        hybrid=hybrid,
        compatibility=compatibility,
        residual=residual,
        autoencoder=autoencoder,
        source_reliability=source_reliability,
        support_response=parse_support_response_config(learned_cfg),
    )
