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
class PairwiseTournamentConfig:
    enabled: bool
    policy_name: str
    base_methods: Tuple[str, ...]
    margin_thresholds: Tuple[float, ...]
    sparse_mix_topk_values: Tuple[int, ...]
    sparse_mix_weighting: str
    score_temperature: float
    temperature_policy: str
    calibration_policy: str
    max_sparse_mix_activation_rate: float


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
