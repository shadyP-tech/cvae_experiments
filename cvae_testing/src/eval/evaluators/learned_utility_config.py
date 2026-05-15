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
class FallbackBenefitGateConfig:
    enabled: bool
    method_name: str
    predictor: str
    feature_set: str
    diagnostic_feature_sets: Tuple[str, ...]
    calibration_policy: str
    ridge_l2: float
    predicted_delta_pct_thresholds: Tuple[float, ...]
    target_clip_delta_pct: Tuple[float, float]
    feature_standardization: str
    max_sparse_mix_activation_rate: float
    max_fallback_harm_rate_active_only: float
    min_fallback_help_minus_harm_active_only: float
    min_source_inner_gap_reduction_pct: float
    min_source_inner_active_rows: int
    min_source_inner_active_domains: int
    min_source_inner_validation_domains: int


@dataclass(frozen=True)
class PairwiseTournamentConfig:
    enabled: bool
    policy_name: str
    base_methods: Tuple[str, ...]
    diagnostic_base_methods: Tuple[str, ...]
    margin_thresholds: Tuple[float, ...]
    sparse_mix_topk_values: Tuple[int, ...]
    sparse_mix_weighting: str
    score_temperature: float
    temperature_policy: str
    calibration_policy: str
    max_sparse_mix_activation_rate: float
    fallback_benefit_gate: FallbackBenefitGateConfig
    pairprob_tournament: "PairprobTournamentConfig"


@dataclass(frozen=True)
class ConformalRegretSetConfig:
    enabled: bool = False
    method_name: str = "conformal_pairprob_regret_set_router_v1"
    base_method: str = "pairwise_group_robust_pairprob_tournament_v1"
    feature_set: str = "pairprob_latent_only_v1"
    calibration_policy: str = "source_inner_oof_conformal_margin_v1"
    alpha_values: Tuple[float, ...] = (0.05, 0.10, 0.20)
    robust_lambda_values: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    nonconformity: str = "top_win_minus_expert_win"
    selection_rule: str = "source_inner_worst_regret_penalized_selection_v1"
    near_oracle_gap_pct_values: Tuple[float, ...] = (1.0, 2.0)
    primary_near_oracle_gap_pct: float = 2.0
    target_primary_near_oracle_in_set_rate: float = 0.80
    max_mean_set_size: float = 2.5
    max_set_size_gt3_rate: float = 0.20
    min_oracle_in_set_rate: float = 0.70
    min_source_inner_regret_rows_per_expert: int = 3
    max_quantile_clipped_fold_rate: float = 0.25
    absolute_high_regret_gap_pct: float = 5.0
    catastrophic_regression_vs_pairprob_hard_gap_pct: float = 5.0
    topwin_diagnostic_method: str = "conformal_pairprob_topwin_set_diagnostic_v1"
    oracle_diagnostic_method: str = "oracle_conformal_regret_set_diagnostic_v1"


@dataclass(frozen=True)
class JackknifeLCBTournamentConfig:
    enabled: bool = False
    method_name: str = "pairwise_jackknife_lcb_pairprob_tournament_v1"
    mean_method_name: str = "pairwise_jackknife_mean_pairprob_tournament_v1"
    base_method: str = "pairwise_group_robust_pairprob_tournament_v1"
    adoption_feature_family: str = "pairprob_latent_only_v1"
    calibration_policy: str = "source_inner_oof_jackknife_lcb_v1"
    lambda_values: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    uncertainty_stat: str = "std_win_across_source_jackknife"
    score_rule: str = "mean_win_minus_lambda_std_win"
    allow_lcb_penalty_auc_min: float = 0.60
    allow_lcb_penalty_spearman_min: float = 0.20
    min_jackknife_models: int = 3
    min_source_inner_validation_domains: int = 2
    max_override_rate: float = 0.20
    absolute_high_regret_gap_pct: float = 5.0
    catastrophic_regression_vs_pairprob_hard_gap_pct: float = 5.0


@dataclass(frozen=True)
class PairprobTournamentConfig:
    enabled: bool
    policy_name: str
    predictor: str
    ridge_l2_values: Tuple[float, ...]
    probability_calibration: str
    adoption_feature_set: str
    diagnostic_feature_sets: Tuple[str, ...]
    direct_method: str
    direct_adoption_method: str
    group_robust_method: str
    combined_diagnostic_method: str
    near_tie_delta_pct: float
    margin_weight_scale_pct: float
    margin_weight_clip: Tuple[float, float]
    min_pairwise_train_pairs: int
    min_pairwise_validation_pairs: int
    min_source_inner_validation_domains: int
    min_non_tie_pairs_per_inner_domain: int
    absolute_high_regret_gap_pct: float
    catastrophic_regression_vs_hard_gap_pct: float
    selection_policy: str
    conformal_regret_set: ConformalRegretSetConfig = ConformalRegretSetConfig()
    jackknife_lcb_tournament: JackknifeLCBTournamentConfig = JackknifeLCBTournamentConfig()


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
    pairwise_tournament: PairwiseTournamentConfig
    support_response: SupportResponseConfig


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_threshold(value: Any) -> float:
    if isinstance(value, str) and value.strip().lower() == "inf":
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

    tournament_cfg = _as_dict(learned_cfg.get("pairwise_tournament", {}))
    fallback_gate_cfg = _as_dict(tournament_cfg.get("fallback_benefit_gate", {}))
    target_clip_values = tuple(
        float(v) for v in fallback_gate_cfg.get("target_clip_delta_pct", [-50.0, 50.0])
    )
    if len(target_clip_values) != 2:
        raise ValueError("learned_utility.pairwise_tournament.fallback_benefit_gate.target_clip_delta_pct must have length 2")
    fallback_gate = FallbackBenefitGateConfig(
        enabled=bool((fallback_gate_cfg or {}).get("enabled", False)),
        method_name=str(
            (fallback_gate_cfg or {}).get(
                "method_name",
                "pairwise_tournament_delta_gated_sparse_mix_v1",
            )
        ),
        predictor=str((fallback_gate_cfg or {}).get("predictor", "ridge_delta_pct")).strip().lower(),
        feature_set=str(
            (fallback_gate_cfg or {}).get("feature_set", "tournament_uncertainty_latent_only_v1")
        ).strip(),
        diagnostic_feature_sets=tuple(
            str(v).strip()
            for v in (fallback_gate_cfg or {}).get(
                "diagnostic_feature_sets",
                ["tournament_uncertainty_combined_diagnostic_v1"],
            )
        ),
        calibration_policy=str(
            (fallback_gate_cfg or {}).get(
                "calibration_policy",
                "source_inner_leave_query_domain_out_crossfit_delta_gate_v1",
            )
        ).strip().lower(),
        ridge_l2=float((fallback_gate_cfg or {}).get("ridge_l2", 1e-4)),
        predicted_delta_pct_thresholds=tuple(
            float(v)
            for v in (fallback_gate_cfg or {}).get(
                "predicted_delta_pct_thresholds",
                [-10.0, -5.0, -2.5, -1.0, -0.5],
            )
        ),
        target_clip_delta_pct=(float(target_clip_values[0]), float(target_clip_values[1])),
        feature_standardization=str(
            (fallback_gate_cfg or {}).get("feature_standardization", "source_inner_train_only")
        ).strip().lower(),
        max_sparse_mix_activation_rate=float((fallback_gate_cfg or {}).get("max_sparse_mix_activation_rate", 0.40)),
        max_fallback_harm_rate_active_only=float(
            (fallback_gate_cfg or {}).get("max_fallback_harm_rate_active_only", 0.45)
        ),
        min_fallback_help_minus_harm_active_only=float(
            (fallback_gate_cfg or {}).get("min_fallback_help_minus_harm_active_only", 0.05)
        ),
        min_source_inner_gap_reduction_pct=float(
            (fallback_gate_cfg or {}).get("min_source_inner_gap_reduction_pct", 0.25)
        ),
        min_source_inner_active_rows=int((fallback_gate_cfg or {}).get("min_source_inner_active_rows", 10)),
        min_source_inner_active_domains=int((fallback_gate_cfg or {}).get("min_source_inner_active_domains", 2)),
        min_source_inner_validation_domains=int(
            (fallback_gate_cfg or {}).get("min_source_inner_validation_domains", 2)
        ),
    )
    pairprob_cfg = _as_dict(tournament_cfg.get("pairprob_tournament", {}))
    pairprob_methods_cfg = _as_dict(pairprob_cfg.get("methods", {}))
    conformal_cfg = _as_dict(pairprob_cfg.get("conformal_regret_set", {}))
    jackknife_cfg = _as_dict(pairprob_cfg.get("jackknife_lcb_tournament", {}))
    margin_clip_values = tuple(
        float(v) for v in pairprob_cfg.get("margin_weight_clip", [0.25, 3.0])
    )
    if len(margin_clip_values) != 2:
        raise ValueError("learned_utility.pairwise_tournament.pairprob_tournament.margin_weight_clip must have length 2")
    conformal_regret_set = ConformalRegretSetConfig(
        enabled=bool((conformal_cfg or {}).get("enabled", False)),
        method_name=str(
            (conformal_cfg or {}).get("method_name", "conformal_pairprob_regret_set_router_v1")
        ),
        base_method=str(
            (conformal_cfg or {}).get("base_method", "pairwise_group_robust_pairprob_tournament_v1")
        ),
        feature_set=str((conformal_cfg or {}).get("feature_set", "pairprob_latent_only_v1")).strip(),
        calibration_policy=str(
            (conformal_cfg or {}).get("calibration_policy", "source_inner_oof_conformal_margin_v1")
        ).strip().lower(),
        alpha_values=tuple(float(v) for v in (conformal_cfg or {}).get("alpha_values", [0.05, 0.10, 0.20])),
        robust_lambda_values=tuple(
            float(v) for v in (conformal_cfg or {}).get("robust_lambda_values", [0.0, 0.25, 0.5, 1.0])
        ),
        nonconformity=str((conformal_cfg or {}).get("nonconformity", "top_win_minus_expert_win")).strip().lower(),
        selection_rule=str(
            (conformal_cfg or {}).get("selection_rule", "source_inner_worst_regret_penalized_selection_v1")
        ).strip().lower(),
        near_oracle_gap_pct_values=tuple(
            float(v) for v in (conformal_cfg or {}).get("near_oracle_gap_pct_values", [1.0, 2.0])
        ),
        primary_near_oracle_gap_pct=float(
            (conformal_cfg or {}).get("primary_near_oracle_gap_pct", 2.0)
        ),
        target_primary_near_oracle_in_set_rate=float(
            (conformal_cfg or {}).get("target_primary_near_oracle_in_set_rate", 0.80)
        ),
        max_mean_set_size=float((conformal_cfg or {}).get("max_mean_set_size", 2.5)),
        max_set_size_gt3_rate=float((conformal_cfg or {}).get("max_set_size_gt3_rate", 0.20)),
        min_oracle_in_set_rate=float((conformal_cfg or {}).get("min_oracle_in_set_rate", 0.70)),
        min_source_inner_regret_rows_per_expert=int(
            (conformal_cfg or {}).get("min_source_inner_regret_rows_per_expert", 3)
        ),
        max_quantile_clipped_fold_rate=float(
            (conformal_cfg or {}).get("max_quantile_clipped_fold_rate", 0.25)
        ),
        absolute_high_regret_gap_pct=float(
            (conformal_cfg or {}).get("absolute_high_regret_gap_pct", 5.0)
        ),
        catastrophic_regression_vs_pairprob_hard_gap_pct=float(
            (conformal_cfg or {}).get("catastrophic_regression_vs_pairprob_hard_gap_pct", 5.0)
        ),
        topwin_diagnostic_method=str(
            (conformal_cfg or {}).get(
                "topwin_diagnostic_method",
                "conformal_pairprob_topwin_set_diagnostic_v1",
            )
        ),
        oracle_diagnostic_method=str(
            (conformal_cfg or {}).get(
                "oracle_diagnostic_method",
                "oracle_conformal_regret_set_diagnostic_v1",
            )
        ),
    )
    jackknife_lcb_tournament = JackknifeLCBTournamentConfig(
        enabled=bool((jackknife_cfg or {}).get("enabled", False)),
        method_name=str(
            (jackknife_cfg or {}).get("method_name", "pairwise_jackknife_lcb_pairprob_tournament_v1")
        ),
        mean_method_name=str(
            (jackknife_cfg or {}).get("mean_method_name", "pairwise_jackknife_mean_pairprob_tournament_v1")
        ),
        base_method=str(
            (jackknife_cfg or {}).get("base_method", "pairwise_group_robust_pairprob_tournament_v1")
        ),
        adoption_feature_family=str(
            (jackknife_cfg or {}).get("adoption_feature_family", "pairprob_latent_only_v1")
        ).strip(),
        calibration_policy=str(
            (jackknife_cfg or {}).get("calibration_policy", "source_inner_oof_jackknife_lcb_v1")
        ).strip().lower(),
        lambda_values=tuple(float(v) for v in (jackknife_cfg or {}).get("lambda_values", [0.0, 0.25, 0.5, 1.0])),
        uncertainty_stat=str(
            (jackknife_cfg or {}).get("uncertainty_stat", "std_win_across_source_jackknife")
        ).strip().lower(),
        score_rule=str(
            (jackknife_cfg or {}).get("score_rule", "mean_win_minus_lambda_std_win")
        ).strip().lower(),
        allow_lcb_penalty_auc_min=float((jackknife_cfg or {}).get("allow_lcb_penalty_auc_min", 0.60)),
        allow_lcb_penalty_spearman_min=float(
            (jackknife_cfg or {}).get("allow_lcb_penalty_spearman_min", 0.20)
        ),
        min_jackknife_models=int((jackknife_cfg or {}).get("min_jackknife_models", 3)),
        min_source_inner_validation_domains=int(
            (jackknife_cfg or {}).get("min_source_inner_validation_domains", 2)
        ),
        max_override_rate=float((jackknife_cfg or {}).get("max_override_rate", 0.20)),
        absolute_high_regret_gap_pct=float(
            (jackknife_cfg or {}).get("absolute_high_regret_gap_pct", 5.0)
        ),
        catastrophic_regression_vs_pairprob_hard_gap_pct=float(
            (jackknife_cfg or {}).get("catastrophic_regression_vs_pairprob_hard_gap_pct", 5.0)
        ),
    )
    pairprob_tournament = PairprobTournamentConfig(
        enabled=bool((pairprob_cfg or {}).get("enabled", False)),
        policy_name=str(
            (pairprob_cfg or {}).get(
                "policy_name",
                "pairwise_group_robust_pairprob_tournament_v1",
            )
        ),
        predictor=str((pairprob_cfg or {}).get("predictor", "logistic_ridge_pairprob")).strip().lower(),
        ridge_l2_values=tuple(
            float(v) for v in (pairprob_cfg or {}).get("ridge_l2_values", [1.0e-4, 1.0e-3, 1.0e-2])
        ),
        probability_calibration=str(
            (pairprob_cfg or {}).get("probability_calibration", "none_v1")
        ).strip().lower(),
        adoption_feature_set=str(
            (pairprob_cfg or {}).get("adoption_feature_set", "pairprob_latent_only_v1")
        ).strip(),
        diagnostic_feature_sets=tuple(
            str(v).strip()
            for v in (pairprob_cfg or {}).get(
                "diagnostic_feature_sets",
                ["pairprob_combined_diagnostic_v1"],
            )
        ),
        direct_method=str(
            (pairprob_methods_cfg or {}).get("direct", "pairwise_direct_pairprob_tournament_v1")
        ),
        direct_adoption_method=str(
            (pairprob_methods_cfg or {}).get(
                "direct_adoption",
                "",
            )
        ),
        group_robust_method=str(
            (pairprob_methods_cfg or {}).get(
                "group_robust",
                "pairwise_group_robust_pairprob_tournament_v1",
            )
        ),
        combined_diagnostic_method=str(
            (pairprob_methods_cfg or {}).get(
                "combined_diagnostic",
                "pairwise_pairprob_combined_diagnostic_v1",
            )
        ),
        near_tie_delta_pct=float((pairprob_cfg or {}).get("near_tie_delta_pct", 0.5)),
        margin_weight_scale_pct=float((pairprob_cfg or {}).get("margin_weight_scale_pct", 5.0)),
        margin_weight_clip=(float(margin_clip_values[0]), float(margin_clip_values[1])),
        min_pairwise_train_pairs=int((pairprob_cfg or {}).get("min_pairwise_train_pairs", 20)),
        min_pairwise_validation_pairs=int((pairprob_cfg or {}).get("min_pairwise_validation_pairs", 10)),
        min_source_inner_validation_domains=int(
            (pairprob_cfg or {}).get("min_source_inner_validation_domains", 2)
        ),
        min_non_tie_pairs_per_inner_domain=int(
            (pairprob_cfg or {}).get("min_non_tie_pairs_per_inner_domain", 3)
        ),
        absolute_high_regret_gap_pct=float(
            (pairprob_cfg or {}).get("absolute_high_regret_gap_pct", 5.0)
        ),
        catastrophic_regression_vs_hard_gap_pct=float(
            (pairprob_cfg or {}).get("catastrophic_regression_vs_hard_gap_pct", 5.0)
        ),
        selection_policy=str(
            (pairprob_cfg or {}).get(
                "selection_policy",
                "source_inner_group_robust_worst_gap_then_catastrophic_then_mean_gap_v1",
            )
        ).strip().lower(),
        conformal_regret_set=conformal_regret_set,
        jackknife_lcb_tournament=jackknife_lcb_tournament,
    )
    tournament = PairwiseTournamentConfig(
        enabled=bool((tournament_cfg or {}).get("enabled", False)),
        policy_name=str(
            (tournament_cfg or {}).get(
                "policy_name",
                "pairwise_tournament_margin_sparse_mix_v1",
            )
        ),
        base_methods=tuple(
            str(v)
            for v in (tournament_cfg or {}).get(
                "base_methods",
                ["pairwise_ranker_latent_only", "pairwise_ranker_combined"],
            )
        ),
        diagnostic_base_methods=tuple(
            str(v)
            for v in (tournament_cfg or {}).get(
                "diagnostic_base_methods",
                [],
            )
        ),
        margin_thresholds=tuple(
            float(v) for v in (tournament_cfg or {}).get("margin_thresholds", [0.0, 0.1, 0.2, 0.3, 0.5])
        ),
        sparse_mix_topk_values=tuple(
            int(v) for v in (tournament_cfg or {}).get("sparse_mix_topk_values", [2, 3])
        ),
        sparse_mix_weighting=str((tournament_cfg or {}).get("sparse_mix_weighting", "uniform")).strip().lower(),
        score_temperature=float((tournament_cfg or {}).get("score_temperature", 1.0)),
        temperature_policy=str(
            (tournament_cfg or {}).get("temperature_policy", "fixed_temperature_not_selected")
        ).strip().lower(),
        calibration_policy=str(
            (tournament_cfg or {}).get("calibration_policy", "inner_leave_query_domain_out_gap_then_top1")
        ).strip().lower(),
        max_sparse_mix_activation_rate=float(
            (tournament_cfg or {}).get("max_sparse_mix_activation_rate", 0.80)
        ),
        fallback_benefit_gate=fallback_gate,
        pairprob_tournament=pairprob_tournament,
    )
    if tournament.enabled:
        if tournament.sparse_mix_weighting != "uniform":
            raise ValueError("learned_utility.pairwise_tournament.sparse_mix_weighting must be 'uniform'")
        if tournament.score_temperature <= 0.0:
            raise ValueError("learned_utility.pairwise_tournament.score_temperature must be > 0")
        if not tournament.base_methods:
            raise ValueError("learned_utility.pairwise_tournament.base_methods must be non-empty")
        if not tournament.margin_thresholds:
            raise ValueError("learned_utility.pairwise_tournament.margin_thresholds must be non-empty")
        if any(int(v) < 1 for v in tournament.sparse_mix_topk_values):
            raise ValueError("learned_utility.pairwise_tournament.sparse_mix_topk_values must be >= 1")
        if tournament.fallback_benefit_gate.enabled:
            gate = tournament.fallback_benefit_gate
            if gate.predictor != "ridge_delta_pct":
                raise ValueError(
                    "learned_utility.pairwise_tournament.fallback_benefit_gate.predictor must be 'ridge_delta_pct'"
                )
            if gate.feature_standardization != "source_inner_train_only":
                raise ValueError(
                    "learned_utility.pairwise_tournament.fallback_benefit_gate.feature_standardization "
                    "must be 'source_inner_train_only'"
                )
            if not gate.predicted_delta_pct_thresholds:
                raise ValueError(
                    "learned_utility.pairwise_tournament.fallback_benefit_gate."
                    "predicted_delta_pct_thresholds must be non-empty"
                )
            if gate.target_clip_delta_pct[0] >= gate.target_clip_delta_pct[1]:
                raise ValueError(
                    "learned_utility.pairwise_tournament.fallback_benefit_gate.target_clip_delta_pct "
                    "must be ordered [low, high]"
                )
            if not gate.feature_set:
                raise ValueError("learned_utility.pairwise_tournament.fallback_benefit_gate.feature_set is required")
        if tournament.pairprob_tournament.enabled:
            pairprob = tournament.pairprob_tournament
            if pairprob.predictor != "logistic_ridge_pairprob":
                raise ValueError(
                    "learned_utility.pairwise_tournament.pairprob_tournament.predictor must be "
                    "'logistic_ridge_pairprob'"
                )
            if pairprob.direct_adoption_method:
                if pairprob.direct_adoption_method != "pairwise_direct_pairprob_adoption_v1":
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.methods."
                        "direct_adoption must be 'pairwise_direct_pairprob_adoption_v1'"
                    )
                if pairprob.direct_method != "pairwise_direct_pairprob_tournament_v1":
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.methods."
                        "direct must be 'pairwise_direct_pairprob_tournament_v1' when direct_adoption is enabled"
                    )
                if pairprob.adoption_feature_set != "pairprob_latent_only_v1":
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.adoption_feature_set "
                        "must be 'pairprob_latent_only_v1' when direct_adoption is enabled"
                    )
            if pairprob.probability_calibration != "none_v1":
                raise ValueError(
                    "learned_utility.pairwise_tournament.pairprob_tournament.probability_calibration "
                    "must be 'none_v1'"
                )
            if not pairprob.ridge_l2_values:
                raise ValueError(
                    "learned_utility.pairwise_tournament.pairprob_tournament.ridge_l2_values must be non-empty"
                )
            if pairprob.margin_weight_scale_pct <= 0.0:
                raise ValueError(
                    "learned_utility.pairwise_tournament.pairprob_tournament.margin_weight_scale_pct must be > 0"
                )
            if pairprob.margin_weight_clip[0] <= 0.0 or pairprob.margin_weight_clip[0] > pairprob.margin_weight_clip[1]:
                raise ValueError(
                    "learned_utility.pairwise_tournament.pairprob_tournament.margin_weight_clip must be positive and ordered"
                )
            if pairprob.adoption_feature_set != "pairprob_latent_only_v1":
                raise ValueError(
                    "learned_utility.pairwise_tournament.pairprob_tournament.adoption_feature_set "
                    "must be 'pairprob_latent_only_v1'"
                )
            conformal = pairprob.conformal_regret_set
            if conformal.enabled:
                if conformal.base_method != pairprob.group_robust_method:
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.conformal_regret_set."
                        "base_method must match methods.group_robust"
                    )
                if conformal.feature_set != pairprob.adoption_feature_set:
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.conformal_regret_set."
                        "feature_set must match adoption_feature_set"
                    )
                if conformal.calibration_policy != "source_inner_oof_conformal_margin_v1":
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.conformal_regret_set."
                        "calibration_policy must be 'source_inner_oof_conformal_margin_v1'"
                    )
                if conformal.nonconformity != "top_win_minus_expert_win":
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.conformal_regret_set."
                        "nonconformity must be 'top_win_minus_expert_win'"
                    )
                if conformal.selection_rule != "source_inner_worst_regret_penalized_selection_v1":
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.conformal_regret_set."
                        "selection_rule must be 'source_inner_worst_regret_penalized_selection_v1'"
                    )
                if not conformal.alpha_values or any(v <= 0.0 or v >= 1.0 for v in conformal.alpha_values):
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.conformal_regret_set."
                        "alpha_values must be in (0, 1)"
                    )
                if not conformal.robust_lambda_values or any(v < 0.0 for v in conformal.robust_lambda_values):
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.conformal_regret_set."
                        "robust_lambda_values must be non-negative"
                    )
                if not conformal.near_oracle_gap_pct_values:
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.conformal_regret_set."
                        "near_oracle_gap_pct_values must be non-empty"
                    )
                if conformal.max_mean_set_size < 1.0:
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.conformal_regret_set."
                        "max_mean_set_size must be >= 1"
                    )
            jackknife = pairprob.jackknife_lcb_tournament
            if jackknife.enabled:
                if jackknife.base_method != pairprob.group_robust_method:
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.jackknife_lcb_tournament."
                        "base_method must match methods.group_robust"
                    )
                if jackknife.adoption_feature_family != pairprob.adoption_feature_set:
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.jackknife_lcb_tournament."
                        "adoption_feature_family must match adoption_feature_set"
                    )
                if jackknife.adoption_feature_family != "pairprob_latent_only_v1":
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.jackknife_lcb_tournament."
                        "adoption_feature_family must be 'pairprob_latent_only_v1'"
                    )
                if jackknife.calibration_policy != "source_inner_oof_jackknife_lcb_v1":
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.jackknife_lcb_tournament."
                        "calibration_policy must be 'source_inner_oof_jackknife_lcb_v1'"
                    )
                if jackknife.uncertainty_stat != "std_win_across_source_jackknife":
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.jackknife_lcb_tournament."
                        "uncertainty_stat must be 'std_win_across_source_jackknife'"
                    )
                if jackknife.score_rule != "mean_win_minus_lambda_std_win":
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.jackknife_lcb_tournament."
                        "score_rule must be 'mean_win_minus_lambda_std_win'"
                    )
                if not jackknife.lambda_values or any(v < 0.0 for v in jackknife.lambda_values):
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.jackknife_lcb_tournament."
                        "lambda_values must be non-empty and non-negative"
                    )
                if 0.0 not in {float(v) for v in jackknife.lambda_values}:
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.jackknife_lcb_tournament."
                        "lambda_values must include 0.0"
                    )
                if jackknife.min_jackknife_models < 2:
                    raise ValueError(
                        "learned_utility.pairwise_tournament.pairprob_tournament.jackknife_lcb_tournament."
                        "min_jackknife_models must be >= 2"
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
        pairwise_tournament=tournament,
        support_response=parse_support_response_config(learned_cfg),
    )
