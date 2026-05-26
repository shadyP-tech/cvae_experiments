import math
from types import SimpleNamespace

from cvae_rebuild.downstream import (
    PredictionBundle,
    evaluate_probability_predictions,
    geometric_probability_pool,
    weighted_arithmetic_probability_pool,
    weighted_geometric_probability_pool,
)
from cvae_rebuild.decentralized_reliability_weighted_gmm_prior import SourceReliability
from cvae_rebuild.decentralized_support_nelbo_reliability_gmm_prior import _combined_weight_plan, _strongest_control
from cvae_rebuild.decentralized_support8_top3_tau05_gmm_prior import _topk_existing_plan
from cvae_rebuild.decentralized_reliability_top3_gmm_prior import _select_topk_reliable
from cvae_rebuild.decentralized_source_inner_transfer_top3_gmm_prior import (
    SubsetScore,
    _best_subset,
    _method_stats,
)
from cvae_rebuild.protocol import ProtocolError
from cvae_rebuild.splits import candidate_experts, random_unlabeled_support_eval_split
from cvae_rebuild.support_nelbo import (
    SupportScore,
    annotate_selection_fraction,
    calibration_stats,
    calibrate,
    rank_support_scores,
    selected_experts,
)


def test_support_sampler_is_unlabeled_and_disjoint() -> None:
    metadata = [
        {"sample_id": f"c0_{idx}", "center": "0", "label": idx % 2}
        for idx in range(40)
    ]
    metadata += [
        {"sample_id": f"c1_{idx}", "center": "1", "label": idx % 2}
        for idx in range(10)
    ]
    split = random_unlabeled_support_eval_split(
        metadata,
        heldout_center="0",
        support_size=32,
        support_seed=17,
    )
    assert split.support_labels_used is False
    assert len(split.support_indices) == 32
    assert set(split.support_sample_ids).isdisjoint(split.eval_sample_ids)


def test_candidate_experts_are_four_source_centers() -> None:
    assert candidate_experts(["0", "1", "2", "3", "4"], "3") == ("0", "1", "2", "4")


def test_calibrated_support_nelbo_rank_selection_flags() -> None:
    stats = calibration_stats("1", [10.0, 12.0, 14.0])
    assert math.isclose(calibrate(12.0, stats), 0.0)
    scores = [
        SupportScore(42, "0", 17, 32, "1", 50.0, 0.5),
        SupportScore(42, "0", 17, 32, "2", 50.0, -1.0),
        SupportScore(42, "0", 17, 32, "3", 50.0, 0.0),
        SupportScore(42, "0", 17, 32, "4", 50.0, 2.0),
    ]
    ranked = rank_support_scores(scores)
    assert [row.expert_id for row in ranked] == ["2", "3", "1", "4"]
    assert selected_experts(ranked, 2) == ("2", "3")
    annotated = annotate_selection_fraction(ranked, k=2)
    selected = [row for row in annotated if row.expert_id in {"2", "3"}]
    assert all(row.selected_expert_count == 2 for row in selected)
    assert all(row.selected_fraction == 0.5 for row in selected)
    assert ranked[0].selected_top1
    assert ranked[1].selected_top2
    assert ranked[2].selected_top3


def test_geometric_pooling_aggregates_probabilities() -> None:
    first = PredictionBundle("1", probabilities=((0.9, 0.1), (0.2, 0.8)))
    second = PredictionBundle("2", probabilities=((0.8, 0.2), (0.3, 0.7)))
    pooled = geometric_probability_pool([first, second])
    assert len(pooled) == 2
    result = evaluate_probability_predictions("top2", pooled, [0, 1])
    assert result.bacc == 1.0
    assert result.macro_f1 == 1.0


def test_weighted_geometric_pooling_equal_weights_matches_unweighted() -> None:
    first = PredictionBundle("1", probabilities=((0.9, 0.1), (0.2, 0.8)))
    second = PredictionBundle("2", probabilities=((0.8, 0.2), (0.3, 0.7)))
    assert weighted_geometric_probability_pool([first, second], [1.0, 1.0]) == geometric_probability_pool([first, second])


def test_weighted_geometric_pooling_rejects_invalid_weights() -> None:
    first = PredictionBundle("1", probabilities=((0.9, 0.1),))
    second = PredictionBundle("2", probabilities=((0.8, 0.2),))
    for weights in ([1.0], [1.0, -1.0], [0.0, 0.0]):
        try:
            weighted_geometric_probability_pool([first, second], weights)
        except ProtocolError:
            pass
        else:
            raise AssertionError(f"Expected invalid weights to fail: {weights}")


def test_weighted_arithmetic_pooling_rejects_class_order_mismatch() -> None:
    first = PredictionBundle("1", probabilities=((0.9, 0.1),), classes=(0, 1))
    second = PredictionBundle("2", probabilities=((0.8, 0.2),), classes=(1, 0))
    try:
        weighted_arithmetic_probability_pool([first, second], [1.0, 1.0])
    except ProtocolError:
        pass
    else:
        raise AssertionError("Expected class-order mismatch to fail.")


def test_weighted_arithmetic_pooling_averages_member_probabilities() -> None:
    first = PredictionBundle("1", probabilities=((0.9, 0.1), (0.2, 0.8)))
    second = PredictionBundle("2", probabilities=((0.7, 0.3), (0.6, 0.4)))
    pooled = weighted_arithmetic_probability_pool([first, second], [1.0, 1.0])
    assert math.isclose(pooled[0][0], 0.8)
    assert math.isclose(pooled[0][1], 0.2)
    assert math.isclose(pooled[1][0], 0.4)
    assert math.isclose(pooled[1][1], 0.6)


def test_support_nelbo_weight_plan_prefers_lower_calibrated_nelbo_when_reliability_is_equal() -> None:
    cfg = SimpleNamespace(
        support_alpha=1.0,
        reliability_alpha=1.0,
        synthetic_per_class_total=128,
        min_per_source_per_class=8,
    )
    rels = {
        source: SourceReliability(42, 17, source, 0.75, 0.75, 0.5, "ok", "", 20, "", "")
        for source in ("1", "2", "3", "4")
    }
    ranked = rank_support_scores(
        [
            SupportScore(42, "0", 17, 32, "1", 10.0, 1.0),
            SupportScore(42, "0", 17, 32, "2", 10.0, -2.0),
            SupportScore(42, "0", 17, 32, "3", 10.0, 0.0),
            SupportScore(42, "0", 17, 32, "4", 10.0, 2.0),
        ]
    )
    plan = _combined_weight_plan(
        cfg,
        ("1", "2", "3", "4"),
        rels,
        ranked,
        tau=1.0,
        use_calibrated=True,
        include_reliability=True,
    )
    assert plan["weights"]["2"] > plan["weights"]["3"] > plan["weights"]["1"] > plan["weights"]["4"]
    assert sum(plan["budgets"].values()) == 128


def test_support_nelbo_decision_audit_uses_strongest_negative_control() -> None:
    method, value = _strongest_control(
        {
            "decentralized_support_nelbo_shuffled_support_control": {"center_equal_mean_bacc": 0.846},
            "decentralized_support_nelbo_shuffled_summary_control": {"center_equal_mean_bacc": 0.17},
            "decentralized_support_nelbo_shuffled_label_control": {"center_equal_mean_bacc": 0.51},
        }
    )
    assert method == "decentralized_support_nelbo_shuffled_support_control"
    assert value == 0.846


def test_support8_top3_plan_uses_combined_scores_and_tie_breaks() -> None:
    cfg = SimpleNamespace(
        synthetic_per_class_total=128,
        min_per_source_per_class=8,
        support_nelbo_tau=0.5,
    )
    plan = {
        "sources": ("1", "2", "3", "4"),
        "weights": {"1": 0.25, "2": 0.25, "3": 0.25, "4": 0.25},
        "scores": {"1": 0.4, "2": 0.8, "3": 0.8, "4": 0.2},
        "support_weights": {"1": 0.25, "2": 0.25, "3": 0.25, "4": 0.25},
        "support_scores": {"1": -1.0, "2": -2.0, "3": -2.0, "4": -0.5},
        "raw_support_nelbo": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0},
        "calibrated_support_nelbo": {"1": 0.0, "2": -0.2, "3": -0.5, "4": 1.0},
        "reliability_scores": {"1": 0.4, "2": 0.8, "3": 0.8, "4": 0.2},
        "use_calibrated_support_nelbo": True,
        "include_reliability": True,
        "support_nelbo_tau": 0.5,
    }
    ranked = rank_support_scores(
        [
            SupportScore(42, "0", 17, 8, "1", 10.0, 0.0),
            SupportScore(42, "0", 17, 8, "2", 10.0, -0.2),
            SupportScore(42, "0", 17, 8, "3", 10.0, -0.5),
            SupportScore(42, "0", 17, 8, "4", 10.0, 1.0),
        ]
    )
    out = _topk_existing_plan(cfg, plan, ranked, k=3)
    assert out["sources"] == ("3", "2", "1")
    assert abs(sum(out["weights"].values()) - 1.0) < 1.0e-8
    assert sum(out["budgets"].values()) == 128
    assert all(value >= 8 for value in out["budgets"].values())


def test_reliability_top3_selection_uses_raw_bacc_and_source_tie_breaks() -> None:
    rels = {
        "1": SourceReliability(42, 17, "1", 0.75, 0.75, 0.6, "ok", "", 20, "", ""),
        "2": SourceReliability(42, 17, "2", 0.80, 0.75, 0.6, "ok", "", 20, "", ""),
        "3": SourceReliability(42, 17, "3", 0.70, 0.75, 0.4, "ok", "", 20, "", ""),
        "4": SourceReliability(42, 17, "4", 0.75, 0.75, 0.6, "ok", "", 20, "", ""),
    }
    assert _select_topk_reliable(("1", "2", "3", "4"), rels, k=3) == ("2", "1", "4")


def test_source_inner_subset_selection_uses_lexicographic_tie_breaks() -> None:
    rows = [
        SubsetScore(42, "0", 17, ("1", "2", "3"), "4", 3, 0.80, 0.90, 0.70, 4),
        SubsetScore(42, "0", 17, ("1", "2", "4"), "3", 3, 0.80, 0.91, 0.60, 4),
        SubsetScore(42, "0", 17, ("1", "3", "4"), "2", 3, 0.79, 0.99, 0.99, 4),
    ]
    assert _best_subset(rows).subset_sources == ("1", "2", "4")
    tied = [
        SubsetScore(42, "0", 17, ("1", "2", "3"), "4", 3, 0.80, 0.90, 0.70, 4),
        SubsetScore(42, "0", 17, ("1", "2", "4"), "3", 3, 0.80, 0.90, 0.70, 4),
    ]
    assert _best_subset(tied).dropped_source == "3"


def test_source_inner_stats_names_are_not_seed_equal_misnamed() -> None:
    rows = [
        {"status": "ok", "experiment_seed": "1", "heldout_center": "A", "bacc": 0.0, "macro_f1": 0.0},
        {"status": "ok", "experiment_seed": "1", "heldout_center": "B", "bacc": 1.0, "macro_f1": 1.0},
        {"status": "ok", "experiment_seed": "2", "heldout_center": "A", "bacc": 0.0, "macro_f1": 0.0},
    ]
    stats = _method_stats(rows)
    assert stats["center_equal_mean_bacc"] == 0.5
    assert stats["seed_equal_mean_bacc"] == 0.25
    assert stats["replicate_row_mean_bacc"] == (1.0 / 3.0)
    assert stats["eligible_seed_center_cells"] == 3
