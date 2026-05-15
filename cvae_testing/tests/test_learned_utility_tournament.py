from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluators.learned_utility_config import (
    ConformalRegretSetConfig,
    FallbackBenefitGateConfig,
    GroupOOFHardpairBoostConfig,
    JackknifeLCBTournamentConfig,
    PairprobTournamentConfig,
    Top2MarginRerankerConfig,
)
from src.eval.evaluators.learned_utility_models import _LogisticRidgePairprob
from src.eval.evaluators.learned_utility_pairs import _build_fold_training_pair_features
from src.eval.evaluators.learned_utility_pairprob import (
    ConformalRegretSetSelection,
    JackknifeCalibrationBlock,
    JackknifeLCBSelection,
    GroupOOFHardpairBoostObservation,
    GroupOOFHardpairBoostSelection,
    PairprobPolicySelection,
    Top2RerankSelection,
    build_pairprob_training_data,
    build_group_oof_hardpair_observations,
    build_top2_rerank_training_data,
    clone_direct_pairprob_adoption_rows,
    conformal_pairprob_route_rows,
    conformal_quantile,
    fit_pairprob_model,
    jackknife_pairprob_route_rows,
    pairprob_feature_names,
    pairprob_probability_matrix,
    pairprob_route_rows,
    hardpair_weight_multipliers_from_observations,
    hardpair_boost_route_rows,
    select_jackknife_lcb_policy,
    select_pairprob_policy,
    top2_rerank_feature_names,
    top2_rerank_route_rows,
)
from src.eval.evaluators import learned_utility as lu
from src.eval.evaluators.learned_utility_protocol import (
    FoldCandidateSet,
    _aggregate_metrics_from_sample_rows,
    _method_protocol,
)
from src.eval.evaluators.learned_utility_tournament import (
    DeltaGatePolicySelection,
    delta_gate_feature_matrix,
    delta_gate_feature_names,
    delta_gate_route_rows,
    fallback_delta_pct_arrays,
    oracle_confidence_set_rows,
    select_delta_gate_policy,
    tournament_route_rows,
    tournament_win_scores,
)


def _gate_cfg(**overrides: object) -> FallbackBenefitGateConfig:
    values = {
        "enabled": True,
        "method_name": "pairwise_tournament_delta_gated_sparse_mix_v1",
        "predictor": "ridge_delta_pct",
        "feature_set": "tournament_uncertainty_latent_only_v1",
        "diagnostic_feature_sets": ("tournament_uncertainty_combined_diagnostic_v1",),
        "calibration_policy": "source_inner_leave_query_domain_out_crossfit_delta_gate_v1",
        "ridge_l2": 1.0e-4,
        "predicted_delta_pct_thresholds": (-5.0,),
        "target_clip_delta_pct": (-50.0, 50.0),
        "feature_standardization": "source_inner_train_only",
        "max_sparse_mix_activation_rate": 1.0,
        "max_fallback_harm_rate_active_only": 0.45,
        "min_fallback_help_minus_harm_active_only": 0.0,
        "min_source_inner_gap_reduction_pct": 0.1,
        "min_source_inner_active_rows": 1,
        "min_source_inner_active_domains": 1,
        "min_source_inner_validation_domains": 2,
    }
    values.update(overrides)
    return FallbackBenefitGateConfig(**values)


def _pairprob_cfg(**overrides: object) -> PairprobTournamentConfig:
    values = {
        "enabled": True,
        "policy_name": "pairwise_group_robust_pairprob_tournament_v1",
        "predictor": "logistic_ridge_pairprob",
        "ridge_l2_values": (1.0e-3,),
        "probability_calibration": "none_v1",
        "adoption_feature_set": "pairprob_latent_only_v1",
        "diagnostic_feature_sets": ("pairprob_combined_diagnostic_v1",),
        "direct_method": "pairwise_direct_pairprob_tournament_v1",
        "direct_adoption_method": "pairwise_direct_pairprob_adoption_v1",
        "group_robust_method": "pairwise_group_robust_pairprob_tournament_v1",
        "combined_diagnostic_method": "pairwise_pairprob_combined_diagnostic_v1",
        "near_tie_delta_pct": 0.5,
        "margin_weight_scale_pct": 5.0,
        "margin_weight_clip": (0.25, 3.0),
        "min_pairwise_train_pairs": 1,
        "min_pairwise_validation_pairs": 1,
        "min_source_inner_validation_domains": 1,
        "min_non_tie_pairs_per_inner_domain": 1,
        "absolute_high_regret_gap_pct": 5.0,
        "catastrophic_regression_vs_hard_gap_pct": 5.0,
        "selection_policy": "source_inner_group_robust_worst_gap_then_catastrophic_then_mean_gap_v1",
    }
    values.update(overrides)
    return PairprobTournamentConfig(**values)


def _conformal_cfg(**overrides: object) -> ConformalRegretSetConfig:
    values = {
        "enabled": True,
        "method_name": "conformal_pairprob_regret_set_router_v1",
        "base_method": "pairwise_group_robust_pairprob_tournament_v1",
        "feature_set": "pairprob_latent_only_v1",
        "calibration_policy": "source_inner_oof_conformal_margin_v1",
        "alpha_values": (0.1,),
        "robust_lambda_values": (0.0,),
        "nonconformity": "top_win_minus_expert_win",
        "selection_rule": "source_inner_worst_regret_penalized_selection_v1",
        "near_oracle_gap_pct_values": (1.0, 2.0),
        "primary_near_oracle_gap_pct": 2.0,
        "target_primary_near_oracle_in_set_rate": 0.80,
        "max_mean_set_size": 3.0,
        "max_set_size_gt3_rate": 1.0,
        "min_oracle_in_set_rate": 0.0,
        "min_source_inner_regret_rows_per_expert": 1,
        "max_quantile_clipped_fold_rate": 1.0,
        "absolute_high_regret_gap_pct": 5.0,
        "catastrophic_regression_vs_pairprob_hard_gap_pct": 5.0,
        "topwin_diagnostic_method": "conformal_pairprob_topwin_set_diagnostic_v1",
        "oracle_diagnostic_method": "oracle_conformal_regret_set_diagnostic_v1",
    }
    values.update(overrides)
    return ConformalRegretSetConfig(**values)


def _jackknife_cfg(**overrides: object) -> JackknifeLCBTournamentConfig:
    values = {
        "enabled": True,
        "method_name": "pairwise_jackknife_lcb_pairprob_tournament_v1",
        "mean_method_name": "pairwise_jackknife_mean_pairprob_tournament_v1",
        "base_method": "pairwise_group_robust_pairprob_tournament_v1",
        "adoption_feature_family": "pairprob_latent_only_v1",
        "calibration_policy": "source_inner_oof_jackknife_lcb_v1",
        "lambda_values": (0.0, 0.5),
        "uncertainty_stat": "std_win_across_source_jackknife",
        "score_rule": "mean_win_minus_lambda_std_win",
        "allow_lcb_penalty_auc_min": 0.60,
        "allow_lcb_penalty_spearman_min": 0.20,
        "min_jackknife_models": 2,
        "min_source_inner_validation_domains": 1,
        "max_override_rate": 1.0,
        "absolute_high_regret_gap_pct": 5.0,
        "catastrophic_regression_vs_pairprob_hard_gap_pct": 5.0,
    }
    values.update(overrides)
    return JackknifeLCBTournamentConfig(**values)


def _top2_cfg(**overrides: object) -> Top2MarginRerankerConfig:
    values = {
        "enabled": True,
        "method_name": "pairwise_direct_top2_margin_reranker_v1",
        "base_method": "pairwise_direct_pairprob_adoption_v1",
        "diagnostic_oracle_method_name": "oracle_top2_margin_reranker_diagnostic_v1",
        "feature_set": "top2_rerank_latent_context_v1",
        "base_feature_set": "pairprob_latent_only_v1",
        "predictor": "logistic_ridge_pairprob",
        "calibration_policy": "source_inner_oof_top2_margin_rerank_v1",
        "margin_thresholds": (0.20,),
        "reranker_l2_values": (1.0e-3,),
        "decision_threshold": 0.50,
        "near_tie_delta_pct": 0.5,
        "margin_weight_scale_pct": 5.0,
        "margin_weight_clip": (0.25, 3.0),
        "max_rerank_activation_rate": 1.0,
        "max_rerank_switch_rate": 1.0,
        "min_source_inner_rerank_rows": 1,
        "min_source_inner_positive_rows": 1,
        "min_source_inner_negative_rows": 1,
        "min_source_inner_active_domains": 1,
        "min_source_inner_validation_domains": 1,
        "min_source_inner_gap_reduction_abs_pct_points": 0.0,
        "min_low_margin_high_regret_enrichment": 0.0,
        "max_switch_harm_rate_active_only": 1.0,
        "min_oracle_top2_recoverable_error_rate": 0.0,
        "min_oracle_top2_recoverable_gap_mass_pct_points": 0.0,
        "absolute_high_regret_gap_pct": 5.0,
        "catastrophic_regression_vs_direct_gap_pct": 5.0,
    }
    values.update(overrides)
    return Top2MarginRerankerConfig(**values)


def _group_oof_cfg(**overrides: object) -> GroupOOFHardpairBoostConfig:
    values = {
        "enabled": True,
        "method_name": "pairwise_direct_group_oof_hardpair_boosted_pairprob_v1",
        "base_method": "pairwise_direct_pairprob_adoption_v1",
        "miss_only_diagnostic_method_name": "pairwise_direct_group_oof_hardpair_miss_boosted_pairprob_v1_diagnostic",
        "random_control_method_name": "pairwise_direct_random_low_margin_boost_pairprob_v1_diagnostic",
        "oracle_top2_diagnostic_method_name": "oracle_top2_margin_reranker_diagnostic_v1",
        "feature_set": "pairprob_latent_only_v1",
        "calibration_policy": "source_inner_group_oof_hardpair_boost_v1",
        "ridge_l2_values": (1.0e-3,),
        "hardpair_margin_thresholds": (0.10,),
        "hardpair_miss_boost_weights": (2.0,),
        "hardpair_confirm_boost_weights": (1.0,),
        "max_pair_weight": 8.0,
        "group_oof_folds": 3,
        "min_group_oof_folds": 3,
        "min_group_oof_train_domains_per_fold": 3,
        "min_group_oof_rows_per_domain": 1,
        "require_group_id_for_adoption": True,
        "max_group_oof_same_slide_leakage_rate": 0.0,
        "near_tie_delta_pct": 0.0,
        "margin_weight_scale_pct": 5.0,
        "margin_weight_clip": (0.25, 3.0),
        "min_source_inner_hardpair_rows": 1,
        "min_source_inner_switch_rows": 0,
        "min_source_inner_keep_rows": 0,
        "min_source_inner_active_domains": 1,
        "min_low_margin_high_regret_enrichment": 0.0,
        "min_oracle_in_base_top2_rate_among_low_margin_high_regret_rows": 0.0,
        "min_source_inner_gap_reduction_abs_pct_points": 0.0,
        "max_source_inner_domain_regression_abs_pct_points": 1.0,
        "tie_mean_gap_tolerance_pct_points": 0.05,
        "absolute_high_regret_gap_pct": 5.0,
        "catastrophic_regression_vs_direct_gap_pct": 5.0,
    }
    values.update(overrides)
    return GroupOOFHardpairBoostConfig(**values)


def test_tournament_win_scores_exclude_self_comparisons() -> None:
    scores = np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64)
    wins = tournament_win_scores(scores, temperature=1.0)

    expected_best = 0.5 * (
        (1.0 / (1.0 + np.exp(-1.0)))
        + (1.0 / (1.0 + np.exp(-2.0)))
    )
    expected_middle = 0.5 * (
        (1.0 / (1.0 + np.exp(1.0)))
        + (1.0 / (1.0 + np.exp(-1.0)))
    )

    assert wins.shape == (1, 3)
    assert np.isclose(wins[0, 0], expected_best)
    assert np.isclose(wins[0, 1], expected_middle)
    assert wins[0, 0] > wins[0, 1] > wins[0, 2]


def test_logistic_pairprob_model_outputs_probabilities() -> None:
    model = _LogisticRidgePairprob(l2=1.0e-3, max_iter=20, device="cpu")
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    model.fit(x, y, np.ones_like(y))
    p = model.predict_proba(x)

    assert p.shape == (4,)
    assert np.all(p >= 0.0)
    assert np.all(p <= 1.0)


def test_pairprob_training_data_filters_near_ties_and_clips_weights() -> None:
    # Two samples, three experts per sample. The first pair in sample 0 is a near tie.
    x_rows = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    q = np.asarray([10, 10, 10, 20, 20, 20], dtype=np.int64)
    e = np.asarray([1, 2, 3, 1, 2, 3], dtype=np.int64)
    s = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    y = np.asarray([100.0, 100.1, 130.0, 90.0, 120.0, 300.0], dtype=np.float64)

    data = build_pairprob_training_data(
        x_rows=x_rows,
        q_rows=q,
        e_rows=e,
        s_rows=s,
        y_rows=y,
        embedding_dim=2,
        expert_feature_dim=3,
        feature_set="pairprob_latent_only_v1",
        near_tie_delta_pct=0.5,
        margin_weight_scale_pct=5.0,
        margin_weight_clip=(0.25, 3.0),
    )

    assert data.total_pairs == 6
    assert data.dropped_near_tie == 1
    assert data.x.shape[0] == 5
    assert np.all(data.weight >= 0.25)
    assert np.all(data.weight <= 3.0)
    assert set(data.kept_by_domain) == {10, 20}


def test_pairprob_feature_sets_preserve_metadata_free_adoption_boundary() -> None:
    latent = pairprob_feature_names(
        "pairprob_latent_only_v1",
        embedding_dim=2,
        expert_feature_dim=3,
        metadata_dim=4,
    )
    combined = pairprob_feature_names(
        "pairprob_combined_diagnostic_v1",
        embedding_dim=2,
        expert_feature_dim=3,
        metadata_dim=4,
    )
    combined_protocol = _method_protocol("pairwise_pairprob_combined_diagnostic_v1")

    assert not any("metadata" in name for name in latent)
    assert any("metadata" in name for name in combined)
    assert combined_protocol.adoption_eligible == 0
    assert combined_protocol.diagnostic_only == 1


def test_pairprob_canonical_orientation_enforces_reverse_complement() -> None:
    x_rows = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    q = np.asarray([10, 10, 20, 20], dtype=np.int64)
    e = np.asarray([1, 2, 1, 2], dtype=np.int64)
    s = np.asarray([0, 0, 1, 1], dtype=np.int64)
    y = np.asarray([1.0, 3.0, 3.0, 1.0], dtype=np.float64)
    train = build_pairprob_training_data(
        x_rows=x_rows,
        q_rows=q,
        e_rows=e,
        s_rows=s,
        y_rows=y,
        embedding_dim=2,
        expert_feature_dim=2,
        feature_set="pairprob_latent_only_v1",
        near_tie_delta_pct=0.0,
        margin_weight_scale_pct=5.0,
        margin_weight_clip=(0.25, 3.0),
    )
    bundle = fit_pairprob_model(
        train_data=train,
        feature_set="pairprob_latent_only_v1",
        ridge_l2=1.0e-3,
        device="cpu",
    )

    probs = pairprob_probability_matrix(
        bundle=bundle,
        x_rows=x_rows,
        expert_domains=[1, 2],
        embedding_dim=2,
        expert_feature_dim=2,
    )

    assert probs.shape == (2, 2, 2)
    assert np.allclose(probs[:, 0, 1] + probs[:, 1, 0], 1.0)
    assert np.allclose(np.diagonal(probs, axis1=1, axis2=2), 0.5)


def test_pairprob_route_rows_use_win_tournament_and_cycle_na_for_two_candidates() -> None:
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=[0, 1, 2])
    prob = np.asarray([[[0.5, 0.8], [0.2, 0.5]]], dtype=np.float64)
    true = np.asarray([[10.0, 20.0]], dtype=np.float64)
    selection = PairprobPolicySelection(
        method="pairwise_group_robust_pairprob_tournament_v1",
        feature_set="pairprob_latent_only_v1",
        ridge_l2=1.0e-3,
        selected_by_inner_validation=True,
    )

    rows = pairprob_route_rows(
        method="pairwise_group_robust_pairprob_tournament_v1",
        fold=fold,
        query_domains=np.asarray([0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        prob_matrix=prob,
        true_nelbo_matrix=true,
        global_true_nelbo_matrix=np.asarray([[5.0, 10.0, 20.0]], dtype=np.float64),
        global_expert_domains=[0, 1, 2],
        policy_name="pairwise_group_robust_pairprob_tournament_v1",
        selection=selection,
        hard_oracle_gap_pct=np.asarray([0.0], dtype=np.float64),
    )

    row = rows[0]
    assert row["selected_expert"] == 1
    assert row["route_experts"] == "1"
    assert np.isclose(row["pairprob_win_top1"], 0.8)
    assert np.isnan(row["pairwise_cycle_rate"])


def test_group_oof_hardpair_marking_preserves_camelyon17_pairprob_geometry() -> None:
    sample_domains = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    embeddings = np.eye(9, 2, dtype=np.float64)
    expert_domains = [0, 1, 2, 3, 4]
    train_idx = np.arange(9, dtype=np.int64)
    x_rows, q_rows, e_rows, s_rows = _build_fold_training_pair_features(
        sample_embeddings=embeddings,
        sample_domains=sample_domains,
        train_indices=train_idx,
        expert_domains=expert_domains,
        outer_heldout_domain=4,
        include_metadata_features=False,
        extra_excluded_domains=[3],
    )
    true_nelbo = np.asarray(
        [
            [5.0, 3.0, 4.0, 9.0, 9.0],
            [5.0, 4.0, 3.0, 9.0, 9.0],
            [5.0, 3.5, 4.0, 9.0, 9.0],
            [3.0, 5.0, 4.0, 9.0, 9.0],
            [4.0, 5.0, 3.0, 9.0, 9.0],
            [3.5, 5.0, 4.0, 9.0, 9.0],
            [3.0, 4.0, 5.0, 9.0, 9.0],
            [4.0, 3.0, 5.0, 9.0, 9.0],
            [3.5, 4.0, 5.0, 9.0, 9.0],
        ],
        dtype=np.float64,
    )
    y_rows = true_nelbo[s_rows, e_rows]
    observations, diag = build_group_oof_hardpair_observations(
        x_rows=x_rows,
        q_rows=q_rows,
        e_rows=e_rows,
        s_rows=s_rows,
        y_rows=y_rows,
        source_domains=[0, 1, 2],
        feature_set="pairprob_latent_only_v1",
        ridge_l2=1.0e-3,
        cfg=_group_oof_cfg(min_source_inner_hardpair_rows=1),
        embedding_dim=2,
        expert_feature_dim=5,
        device="cpu",
    )

    assert diag.reason == ""
    assert diag.grouping_level == "sample"
    assert diag.folds_used == 3
    assert diag.train_domains_per_fold_min == 3
    assert diag.candidate_experts_per_fold_min >= 2
    assert observations
    assert all(obs.base_top1_domain != obs.base_top2_domain for obs in observations)


def test_hardpair_weight_overrides_apply_only_low_margin_top1_top2_pairs() -> None:
    obs = [
        # Low-margin miss: canonical pair gets the miss multiplier.
        dict(
            sample_index=1,
            query_domain=0,
            domain_a=1,
            domain_b=2,
            base_top1_domain=1,
            base_top2_domain=2,
            margin=0.04,
            delta_pct=2.0,
            top2_better=True,
            base_top1_better=False,
            direct_gap_pct=6.0,
            direct_high_regret=True,
            oracle_in_base_top2=True,
            oracle_is_base_top2=True,
        ),
        # High-margin pair is outside the boost region.
        dict(
            sample_index=2,
            query_domain=0,
            domain_a=1,
            domain_b=2,
            base_top1_domain=1,
            base_top2_domain=2,
            margin=0.30,
            delta_pct=2.0,
            top2_better=True,
            base_top1_better=False,
            direct_gap_pct=6.0,
            direct_high_regret=True,
            oracle_in_base_top2=True,
            oracle_is_base_top2=True,
        ),
    ]
    observations = [GroupOOFHardpairBoostObservation(**item) for item in obs]
    multipliers, stats = hardpair_weight_multipliers_from_observations(
        observations,
        margin_threshold=0.10,
        miss_boost_weight=4.0,
        confirm_boost_weight=1.0,
        max_pair_weight=8.0,
    )

    assert multipliers == {(1, 1, 2): 4.0}
    assert stats["hardpair_oof_low_margin_rows"] == 1.0
    assert stats["hardpair_oof_switch_rows"] == 1.0
    assert stats["low_margin_high_regret_oracle_in_base_top2_rate"] == 1.0


def test_hardpair_boost_guard_failure_routes_exactly_as_direct_pairprob() -> None:
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=[0, 1, 2])
    direct_prob = np.asarray([[[0.5, 0.8], [0.2, 0.5]]], dtype=np.float64)
    boosted_prob = np.asarray([[[0.5, 0.1], [0.9, 0.5]]], dtype=np.float64)
    true = np.asarray([[10.0, 20.0]], dtype=np.float64)
    selection = GroupOOFHardpairBoostSelection(
        method="pairwise_direct_group_oof_hardpair_boosted_pairprob_v1",
        base_method="pairwise_direct_pairprob_adoption_v1",
        feature_set="pairprob_latent_only_v1",
        ridge_l2=1.0e-3,
        margin_threshold=0.10,
        miss_boost_weight=2.0,
        confirm_boost_weight=1.0,
        selected_by_inner_validation=True,
        diagnostic_only_reason="insufficient_source_inner_hardpair_rows",
        noop=True,
        guard_status="failed_guards_noop",
    )

    rows = hardpair_boost_route_rows(
        method="pairwise_direct_group_oof_hardpair_boosted_pairprob_v1",
        fold=fold,
        query_domains=np.asarray([0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        prob_matrix=boosted_prob,
        direct_prob_matrix=direct_prob,
        true_nelbo_matrix=true,
        global_true_nelbo_matrix=np.asarray([[5.0, 10.0, 20.0]], dtype=np.float64),
        global_expert_domains=[0, 1, 2],
        policy_name="pairwise_direct_group_oof_hardpair_boosted_pairprob_v1",
        selection=selection,
        cfg=_group_oof_cfg(),
    )

    row = rows[0]
    assert row["selected_expert"] == 1
    assert row["route_experts"] == "1"
    assert row["diagnostic_only"] == 1
    assert row["sign_ci_candidate"] == 0
    assert row["hardpair_boost_guard_status"] == "failed_guards_noop"


def test_direct_pairprob_adoption_alias_clones_diagnostic_route() -> None:
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=[0, 1, 2])
    prob = np.asarray([[[0.5, 0.8], [0.2, 0.5]]], dtype=np.float64)
    true = np.asarray([[10.0, 20.0]], dtype=np.float64)
    selection = PairprobPolicySelection(
        method="pairwise_direct_pairprob_tournament_v1",
        feature_set="pairprob_latent_only_v1",
        ridge_l2=1.0e-3,
        selected_by_inner_validation=True,
    )
    direct_rows = pairprob_route_rows(
        method="pairwise_direct_pairprob_tournament_v1",
        fold=fold,
        query_domains=np.asarray([0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        prob_matrix=prob,
        true_nelbo_matrix=true,
        global_true_nelbo_matrix=np.asarray([[5.0, 10.0, 20.0]], dtype=np.float64),
        global_expert_domains=[0, 1, 2],
        policy_name="pairwise_direct_pairprob_adoption_v1",
        selection=selection,
        hard_oracle_gap_pct=np.asarray([0.0], dtype=np.float64),
    )
    adoption_rows = clone_direct_pairprob_adoption_rows(direct_rows)

    direct = direct_rows[0]
    adoption = adoption_rows[0]
    assert direct["excluded_from_sign_ci_selection"] == 1
    assert direct["sign_ci_candidate"] == 0
    assert adoption["method"] == "pairwise_direct_pairprob_adoption_v1"
    assert adoption["adoption_eligible"] == 1
    assert adoption["diagnostic_only"] == 0
    assert adoption["sign_ci_candidate"] == 1
    assert adoption["direct_adoption_is_alias_of"] == "pairwise_direct_pairprob_tournament_v1"
    assert adoption["direct_adoption_audit_failure_reason"] == "none"
    assert adoption["direct_adoption_same_route_as_direct"] == 1
    assert adoption["direct_adoption_route_hash"] == direct["direct_diagnostic_route_hash"]
    for key in [
        "selected_expert",
        "route_experts",
        "route_weights",
        "selected_nelbo",
        "oracle_gap_pct",
        "top1_oracle_hit",
        "spearman",
        "selected_rank",
    ]:
        assert str(adoption[key]) == str(direct[key])


def test_direct_pairprob_adoption_alias_fails_on_diagnostic_evidence_failure() -> None:
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=[0, 1, 2])
    prob = np.asarray([[[0.5, 0.8], [0.2, 0.5]]], dtype=np.float64)
    true = np.asarray([[10.0, 20.0]], dtype=np.float64)
    selection = PairprobPolicySelection(
        method="pairwise_direct_pairprob_tournament_v1",
        feature_set="pairprob_latent_only_v1",
        ridge_l2=1.0e-3,
        selected_by_inner_validation=True,
        diagnostic_only_reason="insufficient_pairwise_evidence",
    )
    direct_rows = pairprob_route_rows(
        method="pairwise_direct_pairprob_tournament_v1",
        fold=fold,
        query_domains=np.asarray([0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        prob_matrix=prob,
        true_nelbo_matrix=true,
        global_true_nelbo_matrix=np.asarray([[5.0, 10.0, 20.0]], dtype=np.float64),
        global_expert_domains=[0, 1, 2],
        policy_name="pairwise_direct_pairprob_adoption_v1",
        selection=selection,
        hard_oracle_gap_pct=np.asarray([0.0], dtype=np.float64),
    )
    adoption = clone_direct_pairprob_adoption_rows(direct_rows)[0]

    assert adoption["adoption_eligible"] == 0
    assert adoption["diagnostic_only"] == 1
    assert adoption["sign_ci_candidate"] == 0
    assert adoption["direct_adoption_audit_failure_reason"] == "source_only_audit_failed"


def test_top2_rerank_features_are_metadata_free_and_oriented() -> None:
    names = top2_rerank_feature_names(embedding_dim=2, expert_feature_dim=3)

    assert not any("metadata" in name for name in names)
    assert "top1_minus_top2_0" in names
    assert names[-1] == "p_top1_beats_top2"


def test_top2_rerank_training_data_orients_keep_label_and_class_counts() -> None:
    x_rows = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    prob = np.asarray(
        [
            [[0.5, 0.8], [0.2, 0.5]],
            [[0.5, 0.7], [0.3, 0.5]],
        ],
        dtype=np.float64,
    )
    true = np.asarray([[10.0, 20.0], [20.0, 10.0]], dtype=np.float64)
    data = build_top2_rerank_training_data(
        x_rows=x_rows,
        query_domains=np.asarray([0, 1], dtype=np.int64),
        expert_domains=[1, 2],
        prob_matrix=prob,
        true_nelbo_matrix=true,
        embedding_dim=2,
        expert_feature_dim=2,
        margin_threshold=1.0,
        near_tie_delta_pct=0.0,
        margin_weight_scale_pct=5.0,
        margin_weight_clip=(0.25, 3.0),
    )

    assert data.x.shape[0] == 2
    assert data.positive_rows == 1
    assert data.negative_rows == 1
    assert data.y.tolist() == [1.0, 0.0]
    assert np.isclose(data.switch_candidate_rate, 0.5)


def test_top2_rerank_noop_routes_exactly_as_direct_pairprob() -> None:
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=[0, 1, 2])
    x_rows = np.asarray([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    prob = np.asarray([[[0.5, 0.9], [0.1, 0.5]]], dtype=np.float64)
    true = np.asarray([[20.0, 10.0]], dtype=np.float64)
    selection = Top2RerankSelection(
        method="pairwise_direct_top2_margin_reranker_v1",
        oracle_method="oracle_top2_margin_reranker_diagnostic_v1",
        base_method="pairwise_direct_pairprob_adoption_v1",
        feature_set="top2_rerank_latent_context_v1",
        base_feature_set="pairprob_latent_only_v1",
        base_ridge_l2=1.0e-3,
        reranker_l2=1.0e-3,
        margin_threshold=1.0,
        decision_threshold=0.5,
        selected_by_inner_validation=True,
        diagnostic_only_reason="insufficient_source_inner_rerank_rows",
        noop=True,
        guard_status="failed_guards_noop",
    )

    rows = top2_rerank_route_rows(
        method="pairwise_direct_top2_margin_reranker_v1",
        fold=fold,
        query_domains=np.asarray([0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        x_rows=x_rows,
        prob_matrix=prob,
        true_nelbo_matrix=true,
        global_true_nelbo_matrix=np.asarray([[5.0, 20.0, 10.0]], dtype=np.float64),
        global_expert_domains=[0, 1, 2],
        policy_name="pairwise_direct_top2_margin_reranker_v1",
        selection=selection,
        reranker_bundle=None,
        pairprob_direct_gap_pct=np.asarray([100.0], dtype=np.float64),
        metadata_oracle_gap_pct=None,
        embedding_dim=2,
        expert_feature_dim=2,
        cfg=_top2_cfg(),
        keep_prob_override=np.asarray([0.0], dtype=np.float64),
    )

    assert rows[0]["selected_expert"] == 1
    assert rows[0]["top2_rerank_active"] == 0
    assert rows[0]["top2_rerank_switched"] == 0
    assert rows[0]["diagnostic_only"] == 1


def test_oracle_top2_rerank_diagnostic_uses_eval_nelbo_and_is_not_adoption_eligible() -> None:
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=[0, 1, 2])
    x_rows = np.asarray([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    prob = np.asarray([[[0.5, 0.51], [0.49, 0.5]]], dtype=np.float64)
    true = np.asarray([[20.0, 10.0]], dtype=np.float64)
    selection = Top2RerankSelection(
        method="pairwise_direct_top2_margin_reranker_v1",
        oracle_method="oracle_top2_margin_reranker_diagnostic_v1",
        base_method="pairwise_direct_pairprob_adoption_v1",
        feature_set="top2_rerank_latent_context_v1",
        base_feature_set="pairprob_latent_only_v1",
        base_ridge_l2=1.0e-3,
        reranker_l2=1.0e-3,
        margin_threshold=0.1,
        decision_threshold=0.5,
        selected_by_inner_validation=True,
    )

    rows = top2_rerank_route_rows(
        method="oracle_top2_margin_reranker_diagnostic_v1",
        fold=fold,
        query_domains=np.asarray([0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        x_rows=x_rows,
        prob_matrix=prob,
        true_nelbo_matrix=true,
        global_true_nelbo_matrix=np.asarray([[5.0, 20.0, 10.0]], dtype=np.float64),
        global_expert_domains=[0, 1, 2],
        policy_name="pairwise_direct_top2_margin_reranker_v1",
        selection=selection,
        reranker_bundle=None,
        pairprob_direct_gap_pct=np.asarray([100.0], dtype=np.float64),
        metadata_oracle_gap_pct=None,
        embedding_dim=2,
        expert_feature_dim=2,
        cfg=_top2_cfg(),
        oracle_diagnostic=True,
    )
    protocol = _method_protocol("oracle_top2_margin_reranker_diagnostic_v1")

    assert rows[0]["selected_expert"] == 2
    assert rows[0]["routing_uses_eval_nelbo"] == 1
    assert rows[0]["adoption_eligible"] == 0
    assert protocol.routing_uses_eval_nelbo == 1


def test_jackknife_lambda_zero_routes_as_mean_ensemble_and_lcb_can_override() -> None:
    cfg = _jackknife_cfg()
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=[0, 1, 2, 3])
    mean_win = np.asarray([[0.60, 0.59, 0.10]], dtype=np.float64)
    std_win = np.asarray([[0.50, 0.00, 0.01]], dtype=np.float64)
    true = np.asarray([[10.0, 9.0, 30.0]], dtype=np.float64)
    selection = JackknifeLCBSelection(
        method="pairwise_jackknife_lcb_pairprob_tournament_v1",
        mean_method="pairwise_jackknife_mean_pairprob_tournament_v1",
        base_method="pairwise_group_robust_pairprob_tournament_v1",
        feature_set="pairprob_latent_only_v1",
        ridge_l2=1.0e-3,
        jackknife_lambda=0.5,
        selected_by_inner_validation=True,
    )

    mean_rows = jackknife_pairprob_route_rows(
        method="pairwise_jackknife_mean_pairprob_tournament_v1",
        fold=fold,
        query_domains=np.asarray([0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        mean_win=mean_win,
        std_win=std_win,
        n_models=3,
        candidate_pool_consistent=True,
        true_nelbo_matrix=true,
        global_true_nelbo_matrix=np.asarray([[99.0, 10.0, 9.0, 30.0]], dtype=np.float64),
        global_expert_domains=[0, 1, 2, 3],
        policy_name="pairwise_jackknife_lcb_pairprob_tournament_v1",
        selection=selection,
        pairprob_hard_win=mean_win,
        pairprob_hard_selected_idx=np.asarray([0], dtype=np.int64),
        pairprob_hard_oracle_gap_pct=np.asarray([100.0 / 9.0], dtype=np.float64),
        metadata_oracle_gap_pct=np.asarray([0.0], dtype=np.float64),
        cfg=cfg,
        force_lambda=0.0,
    )
    lcb_rows = jackknife_pairprob_route_rows(
        method="pairwise_jackknife_lcb_pairprob_tournament_v1",
        fold=fold,
        query_domains=np.asarray([0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        mean_win=mean_win,
        std_win=std_win,
        n_models=3,
        candidate_pool_consistent=True,
        true_nelbo_matrix=true,
        global_true_nelbo_matrix=np.asarray([[99.0, 10.0, 9.0, 30.0]], dtype=np.float64),
        global_expert_domains=[0, 1, 2, 3],
        policy_name="pairwise_jackknife_lcb_pairprob_tournament_v1",
        selection=selection,
        pairprob_hard_win=mean_win,
        pairprob_hard_selected_idx=np.asarray([0], dtype=np.int64),
        pairprob_hard_oracle_gap_pct=np.asarray([100.0 / 9.0], dtype=np.float64),
        metadata_oracle_gap_pct=np.asarray([0.0], dtype=np.float64),
        cfg=cfg,
    )

    assert mean_rows[0]["selected_expert"] == 1
    assert mean_rows[0]["diagnostic_only"] == 1
    assert lcb_rows[0]["selected_expert"] == 2
    assert lcb_rows[0]["lcb_override_vs_jackknife_mean"] == 1
    assert lcb_rows[0]["paired_gap_delta_vs_pairprob_hard"] < 0.0


def test_jackknife_selection_forces_zero_when_uncertainty_precondition_fails() -> None:
    cfg = _jackknife_cfg(lambda_values=(0.0, 0.5), allow_lcb_penalty_auc_min=0.60)
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=[0, 1, 2])
    base = PairprobPolicySelection(
        method="pairwise_group_robust_pairprob_tournament_v1",
        feature_set="pairprob_latent_only_v1",
        ridge_l2=1.0e-3,
        selected_by_inner_validation=True,
    )
    blocks = [
        JackknifeCalibrationBlock(
            validation_domain=1,
            query_domains=np.asarray([1], dtype=np.int64),
            expert_domains=fold.candidate_expert_domains,
            mean_win=np.asarray([[0.60, 0.40]], dtype=np.float64),
            std_win=np.asarray([[0.50, 0.10]], dtype=np.float64),
            n_models=2,
            candidate_pool_consistent=True,
            true_nelbo_matrix=np.asarray([[10.0, 20.0]], dtype=np.float64),
            global_true_nelbo_matrix=np.asarray([[99.0, 10.0, 20.0]], dtype=np.float64),
            fold=fold,
            pairprob_hard_win=np.asarray([[0.60, 0.40]], dtype=np.float64),
            pairprob_hard_selected_idx=np.asarray([0], dtype=np.int64),
            pairprob_hard_oracle_gap_pct=np.asarray([0.0], dtype=np.float64),
        ),
        JackknifeCalibrationBlock(
            validation_domain=2,
            query_domains=np.asarray([2], dtype=np.int64),
            expert_domains=fold.candidate_expert_domains,
            mean_win=np.asarray([[0.51, 0.49]], dtype=np.float64),
            std_win=np.asarray([[0.10, 0.00]], dtype=np.float64),
            n_models=2,
            candidate_pool_consistent=True,
            true_nelbo_matrix=np.asarray([[20.0, 10.0]], dtype=np.float64),
            global_true_nelbo_matrix=np.asarray([[99.0, 20.0, 10.0]], dtype=np.float64),
            fold=fold,
            pairprob_hard_win=np.asarray([[0.51, 0.49]], dtype=np.float64),
            pairprob_hard_selected_idx=np.asarray([0], dtype=np.int64),
            pairprob_hard_oracle_gap_pct=np.asarray([100.0], dtype=np.float64),
        ),
    ]

    selected = select_jackknife_lcb_policy(
        blocks=blocks,
        base_selection=base,
        global_expert_domains=[0, 1, 2],
        cfg=cfg,
    )

    assert selected is not None
    assert selected.jackknife_lambda == 0.0
    assert selected.lambda_stability_status == "forced_zero_uncertainty_failed"
    assert "forced_zero_uncertainty_failed" in selected.diagnostic_only_reason


def test_jackknife_noop_routes_exactly_as_pairprob_hard() -> None:
    cfg = _jackknife_cfg()
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=[0, 1, 2, 3])
    selection = JackknifeLCBSelection(
        method="pairwise_jackknife_lcb_pairprob_tournament_v1",
        mean_method="pairwise_jackknife_mean_pairprob_tournament_v1",
        base_method="pairwise_group_robust_pairprob_tournament_v1",
        feature_set="pairprob_latent_only_v1",
        ridge_l2=1.0e-3,
        jackknife_lambda=1.0,
        selected_by_inner_validation=False,
        diagnostic_only_reason="source_inner_evidence_insufficient",
        noop=True,
    )

    rows = jackknife_pairprob_route_rows(
        method="pairwise_jackknife_lcb_pairprob_tournament_v1",
        fold=fold,
        query_domains=np.asarray([0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        mean_win=np.asarray([[0.60, 0.59, 0.10]], dtype=np.float64),
        std_win=np.asarray([[0.90, 0.00, 0.00]], dtype=np.float64),
        n_models=1,
        candidate_pool_consistent=False,
        true_nelbo_matrix=np.asarray([[10.0, 9.0, 30.0]], dtype=np.float64),
        global_true_nelbo_matrix=np.asarray([[99.0, 10.0, 9.0, 30.0]], dtype=np.float64),
        global_expert_domains=[0, 1, 2, 3],
        policy_name="pairwise_jackknife_lcb_pairprob_tournament_v1",
        selection=selection,
        pairprob_hard_win=np.asarray([[0.60, 0.59, 0.10]], dtype=np.float64),
        pairprob_hard_selected_idx=np.asarray([0], dtype=np.int64),
        pairprob_hard_oracle_gap_pct=np.asarray([100.0 / 9.0], dtype=np.float64),
        metadata_oracle_gap_pct=None,
        cfg=cfg,
    )

    assert rows[0]["selected_expert"] == 1
    assert rows[0]["lcb_override_vs_pairprob_hard"] == 0
    assert rows[0]["diagnostic_only"] == 1


def test_conformal_quantile_reports_clipping() -> None:
    tau, n, k, clipped = conformal_quantile([0.0, 0.1, 0.2], alpha=0.1)

    assert n == 3
    assert k == 4
    assert clipped == 1
    assert np.isclose(tau, 0.2)


def test_conformal_route_can_override_top1_with_source_inner_penalty() -> None:
    cfg = _conformal_cfg()
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=[0, 1, 2, 3])
    prob = np.asarray(
        [
            [
                [0.5, 0.9, 0.7],
                [0.1, 0.5, 0.6],
                [0.3, 0.4, 0.5],
            ]
        ],
        dtype=np.float64,
    )
    true = np.asarray([[11.0, 10.0, 10.1]], dtype=np.float64)
    selection = ConformalRegretSetSelection(
        method="conformal_pairprob_regret_set_router_v1",
        base_method="pairwise_group_robust_pairprob_tournament_v1",
        feature_set="pairprob_latent_only_v1",
        ridge_l2=1.0e-3,
        alpha=0.1,
        robust_lambda=1.0,
        tau=1.0,
        selected_by_inner_validation=True,
        conformal_calibration_n=3,
        conformal_quantile_k=2,
        conformal_quantile_clipped=0,
        normalized_worst_regret_by_expert={1: 1.0, 2: 0.0, 3: 0.0},
        mean_regret_by_expert={1: 1.0, 2: 0.0, 3: 1.0},
    )

    rows = conformal_pairprob_route_rows(
        method="conformal_pairprob_regret_set_router_v1",
        fold=fold,
        query_domains=np.asarray([0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        prob_matrix=prob,
        true_nelbo_matrix=true,
        global_true_nelbo_matrix=np.asarray([[50.0, 11.0, 10.0, 10.1]], dtype=np.float64),
        global_expert_domains=[0, 1, 2, 3],
        policy_name="conformal_pairprob_regret_set_router_v1",
        selection=selection,
        cfg=cfg,
        pairprob_baseline_gap_pct=np.asarray([10.0], dtype=np.float64),
        scalar_hard_oracle_gap_pct=np.asarray([10.0], dtype=np.float64),
        metadata_oracle_gap_pct=np.asarray([20.0], dtype=np.float64),
    )

    row = rows[0]
    assert row["selected_expert"] == 2
    assert row["regret_set_override_active"] == 1
    assert row["override_delta_gap_pct_vs_pairprob_top1"] < 0.0
    assert row["paired_gap_delta_vs_pairprob_hard"] < 0.0
    assert row["primary_near_oracle_in_conformal_set"] == 1


def test_conformal_topwin_diagnostic_routes_only_top1_not_set() -> None:
    cfg = _conformal_cfg()
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=[0, 1, 2, 3])
    prob = np.asarray(
        [
            [
                [0.5, 0.9, 0.7],
                [0.1, 0.5, 0.6],
                [0.3, 0.4, 0.5],
            ]
        ],
        dtype=np.float64,
    )
    true = np.asarray([[11.0, 10.0, 10.1]], dtype=np.float64)
    selection = ConformalRegretSetSelection(
        method="conformal_pairprob_regret_set_router_v1",
        base_method="pairwise_group_robust_pairprob_tournament_v1",
        feature_set="pairprob_latent_only_v1",
        ridge_l2=1.0e-3,
        alpha=0.1,
        robust_lambda=1.0,
        tau=1.0,
        selected_by_inner_validation=True,
        normalized_worst_regret_by_expert={1: 1.0, 2: 0.0, 3: 0.0},
        mean_regret_by_expert={1: 1.0, 2: 0.0, 3: 1.0},
    )

    rows = conformal_pairprob_route_rows(
        method="conformal_pairprob_topwin_set_diagnostic_v1",
        fold=fold,
        query_domains=np.asarray([0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        prob_matrix=prob,
        true_nelbo_matrix=true,
        global_true_nelbo_matrix=np.asarray([[50.0, 11.0, 10.0, 10.1]], dtype=np.float64),
        global_expert_domains=[0, 1, 2, 3],
        policy_name="conformal_pairprob_regret_set_router_v1",
        selection=selection,
        cfg=cfg,
        pairprob_baseline_gap_pct=np.asarray([10.0], dtype=np.float64),
        scalar_hard_oracle_gap_pct=np.asarray([10.0], dtype=np.float64),
        topwin_diagnostic=True,
    )

    row = rows[0]
    assert row["selected_expert"] == 1
    assert row["route_experts"] == "1"
    assert row["conformal_set_experts"] == "1|2|3"
    assert row["adoption_eligible"] == 0
    assert row["diagnostic_only"] == 1


def test_pairprob_group_robust_selection_prioritizes_worst_domain_gap() -> None:
    cfg = _pairprob_cfg()
    key_a = ("pairwise_group_robust_pairprob_tournament_v1", "pairprob_latent_only_v1", 1.0e-4)
    key_b = ("pairwise_group_robust_pairprob_tournament_v1", "pairprob_latent_only_v1", 1.0e-3)
    base = {
        "spearman": 0.5,
        "top1_oracle_hit": 1,
        "relative_catastrophic_regression_vs_hard_gt_5": 0,
        "absolute_high_regret_gap_gt_5": 0,
    }
    rows_by_key = {
        key_a: [
            {**base, "query_domain": 1, "oracle_gap_pct": 1.0},
            {**base, "query_domain": 2, "oracle_gap_pct": 10.0},
        ],
        key_b: [
            {**base, "query_domain": 1, "oracle_gap_pct": 4.0},
            {**base, "query_domain": 2, "oracle_gap_pct": 4.0},
        ],
    }

    selected = select_pairprob_policy(
        rows_by_key=rows_by_key,
        method="pairwise_group_robust_pairprob_tournament_v1",
        cfg=cfg,
        selection_mode="group_robust",
        evidence_by_key={
            key_a: {"pairwise_train_pairs_after_filter": 20},
            key_b: {"pairwise_train_pairs_after_filter": 20},
        },
    )

    assert selected is not None
    assert np.isclose(selected.ridge_l2, 1.0e-3)


def test_pairprob_evidence_guard_demotes_insufficient_validation_domains() -> None:
    cfg = _pairprob_cfg(min_source_inner_validation_domains=2)
    key = ("pairwise_group_robust_pairprob_tournament_v1", "pairprob_latent_only_v1", 1.0e-4)
    rows_by_key = {
        key: [
            {
                "query_domain": 1,
                "oracle_gap_pct": 1.0,
                "spearman": 1.0,
                "top1_oracle_hit": 1,
                "relative_catastrophic_regression_vs_hard_gt_5": 0,
                "absolute_high_regret_gap_gt_5": 0,
            }
        ]
    }

    selected = select_pairprob_policy(
        rows_by_key=rows_by_key,
        method="pairwise_group_robust_pairprob_tournament_v1",
        cfg=cfg,
        selection_mode="group_robust",
        evidence_by_key={key: {"pairwise_train_pairs_after_filter": 20}},
    )

    assert selected is not None
    assert selected.diagnostic_only_reason == "insufficient_pairwise_evidence"


def test_sparse_mix_rows_use_uniform_expected_nelbo_and_strict_top1() -> None:
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=[0, 1, 2, 3])
    score = np.asarray([[0.0, 0.05, 1.0]], dtype=np.float64)
    true = np.asarray([[100.0, 90.0, 150.0]], dtype=np.float64)

    rows = tournament_route_rows(
        method="pairwise_tournament_topk_uniform",
        fold=fold,
        query_domains=np.asarray([0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        score_matrix=score,
        true_nelbo_matrix=true,
        global_true_nelbo_matrix=np.asarray([[80.0, 100.0, 90.0, 150.0]], dtype=np.float64),
        global_expert_domains=[0, 1, 2, 3],
        policy_name="pairwise_tournament_margin_sparse_mix_v1",
        base_method="pairwise_ranker_latent_only",
        threshold=float("inf"),
        topk=2,
        temperature=1.0,
        temperature_policy="fixed_temperature_not_selected",
        selected_by_inner_validation=True,
        threshold_selection_policy="test",
        diagnostic_only_reason="diagnostic_only_sparse_mix_always_active",
    )

    row = rows[0]
    assert row["route_experts"] == "1|2"
    assert row["route_size"] == 2
    assert row["sparse_mix_active"] == 1
    assert row["selected_expert"] == 1
    assert row["candidate_oracle_expert"] == 2
    assert row["top1_oracle_hit"] == 0
    assert np.isclose(row["selected_nelbo"], 95.0)
    assert np.isclose(row["oracle_gap"], 5.0)
    assert np.isclose(row["fallback_delta"], -5.0)
    assert row["fallback_help"] == 1
    assert row["fallback_harm"] == 0
    assert row["oracle_in_route_set"] == 1


def test_oracle_confidence_set_only_uses_sparse_mix_when_true_nelbo_improves() -> None:
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=[0, 1, 2, 3])
    score = np.asarray(
        [
            [0.0, 0.05, 1.0],
            [0.0, 0.05, 1.0],
        ],
        dtype=np.float64,
    )
    true = np.asarray(
        [
            [100.0, 90.0, 150.0],
            [100.0, 130.0, 150.0],
        ],
        dtype=np.float64,
    )
    rows = oracle_confidence_set_rows(
        fold=fold,
        query_domains=np.asarray([0, 0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        score_matrix=score,
        true_nelbo_matrix=true,
        global_true_nelbo_matrix=np.asarray(
            [
                [80.0, 100.0, 90.0, 150.0],
                [80.0, 100.0, 130.0, 150.0],
            ],
            dtype=np.float64,
        ),
        global_expert_domains=[0, 1, 2, 3],
        policy_name="pairwise_tournament_margin_sparse_mix_v1",
        base_method="pairwise_ranker_latent_only",
        topk=2,
        temperature=1.0,
        temperature_policy="fixed_temperature_not_selected",
    )

    assert rows[0]["sparse_mix_active"] == 1
    assert rows[0]["route_experts"] == "1|2"
    assert rows[1]["sparse_mix_active"] == 0
    assert rows[1]["route_experts"] == "1"
    assert rows[0]["routing_uses_eval_nelbo"] == 1
    assert rows[0]["adoption_eligible"] == 0
    assert rows[0]["diagnostic_only"] == 1


def test_delta_gate_target_clipping_and_metadata_free_feature_sets() -> None:
    score = np.asarray([[0.0, 0.1], [0.0, 0.1]], dtype=np.float64)
    true = np.asarray([[100.0, 80.0], [1.0, 100.0]], dtype=np.float64)
    _features, _win, orders, _margins, feature_names = delta_gate_feature_matrix(
        score_matrix=score,
        expert_domains=[1, 2],
        temperature=1.0,
        topk=2,
        feature_set="tournament_uncertainty_latent_only_v1",
    )
    raw, clipped, _top1, _topk = fallback_delta_pct_arrays(
        true_nelbo_matrix=true,
        tournament_orders=orders,
        topk=2,
        clip_bounds=(-50.0, 50.0),
    )

    assert "latent_combined_top1_agreement" not in feature_names
    assert "latent_combined_topk_jaccard" not in feature_names
    assert np.isclose(raw[0], -10.0)
    assert raw[1] > 1000.0
    assert np.isclose(clipped[1], 50.0)


def test_delta_gate_combined_feature_set_is_explicitly_diagnostic() -> None:
    latent_names = delta_gate_feature_names("tournament_uncertainty_latent_only_v1")
    combined_names = delta_gate_feature_names("tournament_uncertainty_combined_diagnostic_v1")
    protocol = _method_protocol("pairwise_tournament_delta_gated_sparse_mix_combined_diagnostic_v1")

    assert "latent_combined_top1_agreement" not in latent_names
    assert "latent_combined_top1_agreement" in combined_names
    assert protocol.adoption_eligible == 0
    assert protocol.diagnostic_only == 1


def test_delta_gate_top3_feature_is_nan_when_fewer_than_three_candidates() -> None:
    features, _win, _orders, _margins, feature_names = delta_gate_feature_matrix(
        score_matrix=np.asarray([[0.0, 0.2]], dtype=np.float64),
        expert_domains=[1, 2],
        temperature=1.0,
        topk=2,
        feature_set="tournament_uncertainty_latent_only_v1",
    )
    top3_idx = feature_names.index("score_gap_top1_top3")

    assert np.isnan(features[0, top3_idx])


def test_delta_gate_selection_guards_are_deterministic_and_use_paired_gap() -> None:
    rows = []
    for validation_domain in [1, 2]:
        for _ in range(3):
            rows.append(
                {
                    "validation_domain": validation_domain,
                    "query_domain": validation_domain,
                    "base_method": "pairwise_ranker_latent_only",
                    "feature_set": "tournament_uncertainty_latent_only_v1",
                    "feature_names": ("x",),
                    "features": np.asarray([-10.0], dtype=np.float64),
                    "fallback_delta_pct_raw": -10.0,
                    "fallback_delta_pct_clipped_for_training": -10.0,
                    "hard_oracle_gap_pct": 10.0,
                    "topk_oracle_gap_pct": 5.0,
                    "hard_high_regret_selection": 1,
                    "topk_high_regret_selection": 1,
                    "hard_top1_oracle_hit": 0,
                    "topk_oracle_in_route_set": 1,
                }
            )
    selection = select_delta_gate_policy(
        rows_by_key={("pairwise_ranker_latent_only", "tournament_uncertainty_latent_only_v1", 2): rows},
        gate_cfg=_gate_cfg(),
    )

    assert selection is not None
    assert selection.selection_status == "selected"
    assert selection.diagnostic_only_reason == ""
    assert selection.source_inner_paired_gap_reduction_vs_hard > 0.0
    assert selection.model is not None


def test_delta_gate_insufficient_evidence_noop_routes_exactly_as_hard_tournament() -> None:
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=[0, 1, 2, 3])
    score = np.asarray([[0.0, 0.05, 1.0], [0.4, 0.1, 0.7]], dtype=np.float64)
    true = np.asarray([[100.0, 90.0, 150.0], [110.0, 100.0, 120.0]], dtype=np.float64)
    global_true = np.asarray(
        [[80.0, 100.0, 90.0, 150.0], [80.0, 110.0, 100.0, 120.0]],
        dtype=np.float64,
    )
    selection = DeltaGatePolicySelection(
        base_method="pairwise_ranker_latent_only",
        feature_set="tournament_uncertainty_latent_only_v1",
        threshold=-5.0,
        topk=2,
        selected_by_inner_validation=False,
        selection_status="insufficient_evidence_noop",
        diagnostic_only_reason="insufficient_active_rows",
    )
    gated = delta_gate_route_rows(
        method="pairwise_tournament_delta_gated_sparse_mix_v1",
        fold=fold,
        query_domains=np.asarray([0, 0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        score_matrix=score,
        true_nelbo_matrix=true,
        global_true_nelbo_matrix=global_true,
        global_expert_domains=[0, 1, 2, 3],
        policy_name="pairwise_tournament_delta_gated_sparse_mix_v1",
        selection=selection,
        temperature=1.0,
        temperature_policy="fixed_temperature_not_selected",
        gate_cfg=_gate_cfg(),
    )
    hard = tournament_route_rows(
        method="pairwise_tournament_hard",
        fold=fold,
        query_domains=np.asarray([0, 0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        score_matrix=score,
        true_nelbo_matrix=true,
        global_true_nelbo_matrix=global_true,
        global_expert_domains=[0, 1, 2, 3],
        policy_name="pairwise_tournament_delta_gated_sparse_mix_v1",
        base_method="pairwise_ranker_latent_only",
        threshold=0.0,
        topk=1,
        temperature=1.0,
        temperature_policy="fixed_temperature_not_selected",
        selected_by_inner_validation=False,
        threshold_selection_policy="test",
    )

    assert [r["route_experts"] for r in gated] == [r["route_experts"] for r in hard]
    assert [r["selected_nelbo"] for r in gated] == [r["selected_nelbo"] for r in hard]
    assert all(int(r["delta_gate_active"]) == 0 for r in gated)
    assert all(r["delta_gate_selection_status"] == "insufficient_evidence_noop" for r in gated)
    assert all(int(r["adoption_eligible"]) == 0 for r in gated)


def test_margin_diagnostics_and_high_activation_guard_are_aggregated() -> None:
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=[0, 1, 2, 3])
    rows = tournament_route_rows(
        method="pairwise_tournament_inner_selected",
        fold=fold,
        query_domains=np.asarray([0, 0], dtype=np.int64),
        expert_domains=fold.candidate_expert_domains,
        score_matrix=np.asarray([[0.0, 1.0, 2.0], [0.0, 0.1, 0.2]], dtype=np.float64),
        true_nelbo_matrix=np.asarray([[90.0, 100.0, 110.0], [100.0, 90.0, 120.0]], dtype=np.float64),
        global_true_nelbo_matrix=np.asarray(
            [[80.0, 90.0, 100.0, 110.0], [80.0, 100.0, 90.0, 120.0]],
            dtype=np.float64,
        ),
        global_expert_domains=[0, 1, 2, 3],
        policy_name="pairwise_tournament_margin_sparse_mix_v1",
        base_method="pairwise_ranker_latent_only",
        threshold=float("inf"),
        topk=2,
        temperature=1.0,
        temperature_policy="fixed_temperature_not_selected",
        selected_by_inner_validation=True,
        threshold_selection_policy="test",
        diagnostic_only_reason="diagnostic_only_high_fallback_rate",
    )
    metrics = _aggregate_metrics_from_sample_rows(rows)["pairwise_tournament_inner_selected"]

    assert metrics["adoption_eligible"] == 0.0
    assert metrics["diagnostic_only"] == 1.0
    assert metrics["diagnostic_only_reason"] == "diagnostic_only_high_fallback_rate"
    assert metrics["sparse_mix_active_rate"] == 1.0
    assert np.isfinite(metrics["mean_margin_when_top1_correct"])
    assert np.isfinite(metrics["mean_margin_when_top1_wrong"])


def test_source_inner_training_features_exclude_outer_and_inner_validation_experts() -> None:
    embeddings = np.arange(6 * 2, dtype=np.float64).reshape(6, 2)
    sample_domains = np.asarray([0, 1, 1, 2, 2, 3], dtype=np.int64)
    _x, q, e, _s = _build_fold_training_pair_features(
        sample_embeddings=embeddings,
        sample_domains=sample_domains,
        train_indices=np.asarray([1, 2, 3, 4, 5], dtype=np.int64),
        expert_domains=[0, 1, 2, 3],
        outer_heldout_domain=0,
        include_metadata_features=True,
        extra_excluded_domains=[2],
    )

    assert all(int(expert) != 0 for expert in e.tolist())
    assert all(int(expert) != 2 for expert in e.tolist())
    assert all(int(expert) != int(query) for query, expert in zip(q.tolist(), e.tolist()))


def test_pairwise_tournament_protocol_flags() -> None:
    learned = _method_protocol("pairwise_tournament_inner_selected")
    oracle = _method_protocol("oracle_confidence_set_diagnostic")
    group_pairprob = _method_protocol("pairwise_group_robust_pairprob_tournament_v1")
    direct_pairprob = _method_protocol("pairwise_direct_pairprob_tournament_v1")
    direct_adoption = _method_protocol("pairwise_direct_pairprob_adoption_v1")
    combined_pairprob = _method_protocol("pairwise_pairprob_combined_diagnostic_v1")
    conformal = _method_protocol("conformal_pairprob_regret_set_router_v1")
    conformal_topwin = _method_protocol("conformal_pairprob_topwin_set_diagnostic_v1")
    conformal_oracle = _method_protocol("oracle_conformal_regret_set_diagnostic_v1")
    jackknife_lcb = _method_protocol("pairwise_jackknife_lcb_pairprob_tournament_v1")
    jackknife_mean = _method_protocol("pairwise_jackknife_mean_pairprob_tournament_v1")
    top2_rerank = _method_protocol("pairwise_direct_top2_margin_reranker_v1")
    top2_oracle = _method_protocol("oracle_top2_margin_reranker_diagnostic_v1")
    hardpair_boost = _method_protocol("pairwise_direct_group_oof_hardpair_boosted_pairprob_v1")
    hardpair_miss_only = _method_protocol(
        "pairwise_direct_group_oof_hardpair_miss_boosted_pairprob_v1_diagnostic"
    )
    random_low_margin = _method_protocol("pairwise_direct_random_low_margin_boost_pairprob_v1_diagnostic")

    assert learned.method_role == "learned"
    assert learned.adoption_eligible == 1
    assert learned.routing_uses_eval_nelbo == 0
    assert group_pairprob.method_role == "learned"
    assert group_pairprob.adoption_eligible == 1
    assert group_pairprob.diagnostic_only == 0
    assert group_pairprob.routing_uses_eval_nelbo == 0
    assert direct_pairprob.method_role == "diagnostic"
    assert direct_pairprob.adoption_eligible == 0
    assert direct_pairprob.diagnostic_only == 1
    assert direct_adoption.method_role == "learned"
    assert direct_adoption.adoption_eligible == 1
    assert direct_adoption.diagnostic_only == 0
    assert direct_adoption.routing_uses_eval_nelbo == 0
    assert combined_pairprob.method_role == "diagnostic"
    assert combined_pairprob.adoption_eligible == 0
    assert combined_pairprob.diagnostic_only == 1
    assert conformal.method_role == "learned"
    assert conformal.adoption_eligible == 1
    assert conformal.routing_uses_eval_nelbo == 0
    assert jackknife_lcb.method_role == "learned"
    assert jackknife_lcb.adoption_eligible == 1
    assert jackknife_lcb.routing_uses_eval_nelbo == 0
    assert jackknife_mean.method_role == "diagnostic"
    assert jackknife_mean.adoption_eligible == 0
    assert conformal_topwin.method_role == "diagnostic"
    assert conformal_topwin.adoption_eligible == 0
    assert conformal_oracle.method_role == "diagnostic"
    assert conformal_oracle.adoption_eligible == 0
    assert conformal_oracle.routing_uses_eval_nelbo == 1
    assert top2_rerank.method_role == "learned"
    assert top2_rerank.adoption_eligible == 1
    assert top2_rerank.routing_uses_eval_nelbo == 0
    assert top2_oracle.method_role == "diagnostic"
    assert top2_oracle.adoption_eligible == 0
    assert top2_oracle.routing_uses_eval_nelbo == 1
    assert hardpair_boost.method_role == "learned"
    assert hardpair_boost.adoption_eligible == 1
    assert hardpair_boost.routing_uses_eval_nelbo == 0
    assert hardpair_miss_only.method_role == "diagnostic"
    assert hardpair_miss_only.adoption_eligible == 0
    assert random_low_margin.method_role == "diagnostic"
    assert random_low_margin.adoption_eligible == 0
    assert oracle.method_role == "diagnostic"
    assert oracle.adoption_eligible == 0
    assert oracle.diagnostic_only == 1
    assert oracle.routing_uses_eval_nelbo == 1


def test_learned_utility_eval_emits_tournament_methods(tmp_path, monkeypatch) -> None:
    expert_domains = [0, 1, 2, 3, 4]
    sample_domains = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    embeddings = np.stack(
        [
            np.asarray([float(domain), float(i % 2)], dtype=np.float64)
            for i, domain in enumerate(sample_domains.tolist())
        ],
        axis=0,
    )
    nelbo = np.zeros((len(sample_domains), len(expert_domains)), dtype=np.float64)
    for i, query_domain in enumerate(sample_domains.tolist()):
        for j, expert_domain in enumerate(expert_domains):
            nelbo[i, j] = abs(float(query_domain) - float(expert_domain)) + 0.1 * float(j)
    metadata = [
        {"magnification": int(domain), "sample_id": f"s{i}"}
        for i, domain in enumerate(sample_domains.tolist())
    ]

    def fake_score(**kwargs):
        _ = kwargs
        return embeddings, sample_domains, nelbo, expert_domains, metadata

    monkeypatch.setattr(lu, "_score_experts_batched", fake_score)

    results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={f"expert_{domain}": "unused" for domain in expert_domains},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg={
            "predictors": ["pairwise_ranker"],
            "pair_features": {"include_metadata_features": True},
            "scoring": {"pair_batch_size": 2},
            "predictor_params": {
                "pairwise_ranker": {
                    "hidden_dim": 8,
                    "epochs": 1,
                    "lr": 1.0e-3,
                    "batch_size": 64,
                    "device": "cpu",
                    "margin": 1.0,
                    "near_tie_delta": 0.0,
                    "hard_pair_fraction": 1.0,
                    "random_pair_fraction": 0.0,
                    "max_pairs_per_sample": 6,
                    "max_pairs_per_domain": 100,
                    "run_ablations": True,
                }
            },
            "pairwise_tournament": {
                "enabled": True,
                "base_methods": ["pairwise_ranker_latent_only", "pairwise_ranker_combined"],
                "margin_thresholds": [0.0, 0.5],
                "sparse_mix_topk_values": [2],
                "score_temperature": 1.0,
                "max_sparse_mix_activation_rate": 0.80,
            },
            "hybrid_scoring": {"enabled": False, "tie_policy": "stable_expert_index"},
            "compatibility_research": {
                "floors": {"random_rank_floor": False, "random_score_floor": False},
                "permutation_tests": {
                    "expert_label_permutation": False,
                    "metadata_permutation": False,
                    "repeats": 1,
                },
                "diagnostics": {"save_distribution_plots": False},
                "gate": {"uplift_reference_method": "metadata_routing"},
            },
        },
        reports_dir=tmp_path,
    )

    metrics = results["metrics_by_method"]
    assert "pairwise_tournament_hard" in metrics
    assert "pairwise_tournament_topk_uniform" in metrics
    assert "pairwise_tournament_inner_selected" in metrics
    assert "oracle_confidence_set_diagnostic" in metrics
    assert metrics["pairwise_tournament_hard"]["routing_uses_eval_nelbo"] == 0.0
    assert metrics["oracle_confidence_set_diagnostic"]["adoption_eligible"] == 0.0
    assert metrics["oracle_confidence_set_diagnostic"]["routing_uses_eval_nelbo"] == 1.0

    delta_results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={f"expert_{domain}": "unused" for domain in expert_domains},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg={
            "predictors": ["pairwise_ranker"],
            "pair_features": {"include_metadata_features": True},
            "scoring": {"pair_batch_size": 2},
            "predictor_params": {
                "pairwise_ranker": {
                    "hidden_dim": 8,
                    "epochs": 1,
                    "lr": 1.0e-3,
                    "batch_size": 64,
                    "device": "cpu",
                    "margin": 1.0,
                    "near_tie_delta": 0.0,
                    "hard_pair_fraction": 1.0,
                    "random_pair_fraction": 0.0,
                    "max_pairs_per_sample": 6,
                    "max_pairs_per_domain": 100,
                    "run_ablations": True,
                }
            },
            "pairwise_tournament": {
                "enabled": True,
                "policy_name": "pairwise_tournament_delta_gated_sparse_mix_v1",
                "base_methods": ["pairwise_ranker_latent_only"],
                "diagnostic_base_methods": ["pairwise_ranker_combined"],
                "sparse_mix_topk_values": [2],
                "score_temperature": 1.0,
                "max_sparse_mix_activation_rate": 0.80,
                "fallback_benefit_gate": {
                    "enabled": True,
                    "feature_set": "tournament_uncertainty_latent_only_v1",
                    "diagnostic_feature_sets": ["tournament_uncertainty_combined_diagnostic_v1"],
                    "predicted_delta_pct_thresholds": [-5.0],
                    "min_source_inner_active_rows": 1,
                    "min_source_inner_active_domains": 1,
                    "min_source_inner_validation_domains": 2,
                    "max_sparse_mix_activation_rate": 1.0,
                    "max_fallback_harm_rate_active_only": 1.0,
                    "min_fallback_help_minus_harm_active_only": -1.0,
                    "min_source_inner_gap_reduction_pct": -100.0,
                },
            },
            "hybrid_scoring": {"enabled": False, "tie_policy": "stable_expert_index"},
            "compatibility_research": {
                "floors": {"random_rank_floor": False, "random_score_floor": False},
                "permutation_tests": {
                    "expert_label_permutation": False,
                    "metadata_permutation": False,
                    "repeats": 1,
                },
                "diagnostics": {"save_distribution_plots": False},
                "gate": {"uplift_reference_method": "metadata_routing"},
            },
        },
        reports_dir=tmp_path / "delta",
    )
    delta_metrics = delta_results["metrics_by_method"]
    assert "pairwise_tournament_delta_gated_sparse_mix_v1" in delta_metrics
    assert "pairwise_tournament_delta_gated_sparse_mix_combined_diagnostic_v1" in delta_metrics
    assert "oracle_confidence_set_diagnostic" in delta_metrics
    assert delta_metrics["pairwise_tournament_delta_gated_sparse_mix_v1"]["routing_uses_eval_nelbo"] == 0.0
    assert delta_metrics["pairwise_tournament_delta_gated_sparse_mix_combined_diagnostic_v1"]["diagnostic_only"] == 1.0
    assert delta_metrics["oracle_confidence_set_diagnostic"]["routing_uses_eval_nelbo"] == 1.0

    conformal_results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={f"expert_{domain}": "unused" for domain in expert_domains},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg={
            "predictors": ["pairwise_ranker"],
            "pair_features": {"include_metadata_features": True},
            "scoring": {"pair_batch_size": 2},
            "predictor_params": {
                "pairwise_ranker": {
                    "hidden_dim": 8,
                    "epochs": 1,
                    "lr": 1.0e-3,
                    "batch_size": 64,
                    "device": "cpu",
                    "margin": 1.0,
                    "near_tie_delta": 0.0,
                    "hard_pair_fraction": 1.0,
                    "random_pair_fraction": 0.0,
                    "max_pairs_per_sample": 6,
                    "max_pairs_per_domain": 100,
                    "run_ablations": True,
                }
            },
            "pairwise_tournament": {
                "enabled": True,
                "policy_name": "conformal_pairprob_regret_set_router_v1",
                "base_methods": ["pairwise_ranker_latent_only"],
                "diagnostic_base_methods": ["pairwise_ranker_combined"],
                "margin_thresholds": [0.0],
                "sparse_mix_topk_values": [2],
                "score_temperature": 1.0,
                "pairprob_tournament": {
                    "enabled": True,
                    "methods": {
                        "direct": "pairwise_direct_pairprob_tournament_v1",
                        "direct_adoption": "pairwise_direct_pairprob_adoption_v1",
                        "group_robust": "pairwise_group_robust_pairprob_tournament_v1",
                        "combined_diagnostic": "pairwise_pairprob_combined_diagnostic_v1",
                    },
                    "ridge_l2_values": [1.0e-3],
                    "min_pairwise_train_pairs": 1,
                    "min_pairwise_validation_pairs": 1,
                    "min_source_inner_validation_domains": 1,
                    "min_non_tie_pairs_per_inner_domain": 1,
                    "conformal_regret_set": {
                        "enabled": True,
                        "alpha_values": [0.1],
                        "robust_lambda_values": [0.0, 1.0],
                        "max_mean_set_size": 5.0,
                        "max_set_size_gt3_rate": 1.0,
                        "min_oracle_in_set_rate": 0.0,
                        "min_source_inner_regret_rows_per_expert": 1,
                        "max_quantile_clipped_fold_rate": 1.0,
                    },
                },
            },
            "hybrid_scoring": {"enabled": False, "tie_policy": "stable_expert_index"},
            "compatibility_research": {
                "floors": {"random_rank_floor": False, "random_score_floor": False},
                "permutation_tests": {
                    "expert_label_permutation": False,
                    "metadata_permutation": False,
                    "repeats": 1,
                },
                "diagnostics": {"save_distribution_plots": False},
                "gate": {"uplift_reference_method": "metadata_routing"},
            },
        },
        reports_dir=tmp_path / "conformal",
    )
    conformal_metrics = conformal_results["metrics_by_method"]
    assert "pairwise_direct_pairprob_tournament_v1" in conformal_metrics
    assert "pairwise_direct_pairprob_adoption_v1" in conformal_metrics
    assert "pairwise_group_robust_pairprob_tournament_v1" in conformal_metrics
    assert "conformal_pairprob_regret_set_router_v1" in conformal_metrics
    assert "conformal_pairprob_topwin_set_diagnostic_v1" in conformal_metrics
    assert "oracle_conformal_regret_set_diagnostic_v1" in conformal_metrics
    assert conformal_metrics["conformal_pairprob_regret_set_router_v1"]["routing_uses_eval_nelbo"] == 0.0
    assert conformal_metrics["pairwise_direct_pairprob_tournament_v1"]["diagnostic_only"] == 1.0
    assert conformal_metrics["pairwise_direct_pairprob_tournament_v1"]["excluded_from_sign_ci_selection"] == "1"
    assert conformal_metrics["pairwise_direct_pairprob_adoption_v1"]["adoption_eligible"] == 1.0
    assert conformal_metrics["pairwise_direct_pairprob_adoption_v1"]["diagnostic_only"] == 0.0
    assert conformal_metrics["pairwise_direct_pairprob_adoption_v1"]["sign_ci_candidate"] == "1"
    assert (
        conformal_metrics["pairwise_direct_pairprob_adoption_v1"]["direct_adoption_same_route_as_direct"]
        == "1"
    )
    assert (
        conformal_metrics["pairwise_direct_pairprob_adoption_v1"]["direct_adoption_audit_failure_reason"]
        == "none"
    )
    assert (
        conformal_metrics["pairwise_direct_pairprob_adoption_v1"]["direct_adoption_route_hash"]
        == conformal_metrics["pairwise_direct_pairprob_tournament_v1"]["direct_diagnostic_route_hash"]
    )
    assert "mean_gap_delta_vs_group_robust_pairprob" in conformal_metrics[
        "pairwise_direct_pairprob_adoption_v1"
    ]
    assert conformal_metrics["conformal_pairprob_topwin_set_diagnostic_v1"]["diagnostic_only"] == 1.0
    assert conformal_metrics["oracle_conformal_regret_set_diagnostic_v1"]["routing_uses_eval_nelbo"] == 1.0

    jackknife_results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={f"expert_{domain}": "unused" for domain in expert_domains},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg={
            "predictors": ["pairwise_ranker"],
            "pair_features": {"include_metadata_features": True},
            "scoring": {"pair_batch_size": 2},
            "predictor_params": {
                "pairwise_ranker": {
                    "hidden_dim": 8,
                    "epochs": 1,
                    "lr": 1.0e-3,
                    "batch_size": 64,
                    "device": "cpu",
                    "margin": 1.0,
                    "near_tie_delta": 0.0,
                    "hard_pair_fraction": 1.0,
                    "random_pair_fraction": 0.0,
                    "max_pairs_per_sample": 6,
                    "max_pairs_per_domain": 100,
                    "run_ablations": True,
                }
            },
            "pairwise_tournament": {
                "enabled": True,
                "policy_name": "pairwise_jackknife_lcb_pairprob_tournament_v1",
                "base_methods": ["pairwise_ranker_latent_only"],
                "diagnostic_base_methods": ["pairwise_ranker_combined"],
                "margin_thresholds": [0.0],
                "sparse_mix_topk_values": [2],
                "score_temperature": 1.0,
                "pairprob_tournament": {
                    "enabled": True,
                    "ridge_l2_values": [1.0e-3],
                    "min_pairwise_train_pairs": 1,
                    "min_pairwise_validation_pairs": 1,
                    "min_source_inner_validation_domains": 1,
                    "min_non_tie_pairs_per_inner_domain": 1,
                    "jackknife_lcb_tournament": {
                        "enabled": True,
                        "lambda_values": [0.0, 0.5],
                        "min_jackknife_models": 2,
                        "min_source_inner_validation_domains": 1,
                    },
                },
            },
            "hybrid_scoring": {"enabled": False, "tie_policy": "stable_expert_index"},
            "compatibility_research": {
                "floors": {"random_rank_floor": False, "random_score_floor": False},
                "permutation_tests": {
                    "expert_label_permutation": False,
                    "metadata_permutation": False,
                    "repeats": 1,
                },
                "diagnostics": {"save_distribution_plots": False},
                "gate": {"uplift_reference_method": "metadata_routing"},
            },
        },
        reports_dir=tmp_path / "jackknife",
    )
    jackknife_metrics = jackknife_results["metrics_by_method"]
    assert "pairwise_group_robust_pairprob_tournament_v1" in jackknife_metrics
    assert "pairwise_jackknife_mean_pairprob_tournament_v1" in jackknife_metrics
    assert "pairwise_jackknife_lcb_pairprob_tournament_v1" in jackknife_metrics
    assert jackknife_metrics["pairwise_jackknife_mean_pairprob_tournament_v1"]["diagnostic_only"] == 1.0
    assert jackknife_metrics["pairwise_jackknife_lcb_pairprob_tournament_v1"]["routing_uses_eval_nelbo"] == 0.0
    assert (
        jackknife_metrics["pairwise_jackknife_lcb_pairprob_tournament_v1"]["adoption_feature_family"]
        == "pairprob_latent_only_v1"
    )
