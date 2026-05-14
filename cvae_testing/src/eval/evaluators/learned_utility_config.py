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
        support_response=parse_support_response_config(learned_cfg),
    )
