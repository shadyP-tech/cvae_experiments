from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluators.learned_utility_pairs import _build_fold_training_pair_features
from src.eval.evaluators import learned_utility as lu
from src.eval.evaluators.learned_utility_protocol import (
    FoldCandidateSet,
    _aggregate_metrics_from_sample_rows,
    _method_protocol,
)
from src.eval.evaluators.learned_utility_tournament import (
    oracle_confidence_set_rows,
    tournament_route_rows,
    tournament_win_scores,
)


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

    assert learned.method_role == "learned"
    assert learned.adoption_eligible == 1
    assert learned.routing_uses_eval_nelbo == 0
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
