from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluators.learned_utility_config import FallbackBenefitGateConfig, PairprobTournamentConfig
from src.eval.evaluators.learned_utility_models import _LogisticRidgePairprob
from src.eval.evaluators.learned_utility_pairs import _build_fold_training_pair_features
from src.eval.evaluators.learned_utility_pairprob import (
    PairprobPolicySelection,
    build_pairprob_training_data,
    fit_pairprob_model,
    pairprob_feature_names,
    pairprob_probability_matrix,
    pairprob_route_rows,
    select_pairprob_policy,
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
    combined_pairprob = _method_protocol("pairwise_pairprob_combined_diagnostic_v1")

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
    assert combined_pairprob.method_role == "diagnostic"
    assert combined_pairprob.adoption_eligible == 0
    assert combined_pairprob.diagnostic_only == 1
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
