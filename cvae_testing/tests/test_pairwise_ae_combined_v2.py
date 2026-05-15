from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.load_config import load_config  # noqa: E402
from src.config.schema import validate_config  # noqa: E402
from src.eval.evaluators.learned_utility_protocol import FoldCandidateSet, _method_protocol  # noqa: E402
from src.eval.evaluators import pairwise_ae_combined_v2 as v2  # noqa: E402


def _payload() -> dict:
    expert_domains = [0, 1, 2, 3, 4]
    sample_domains = np.asarray([domain for domain in expert_domains for _ in range(3)], dtype=np.int64)
    embeddings = np.asarray(
        [[float(domain), float(i), float(domain * 10 + i)] for domain in expert_domains for i in range(3)],
        dtype=np.float64,
    )
    true_nelbo = np.asarray(
        [[abs(float(query - expert)) + 0.1 for expert in expert_domains] for query in sample_domains],
        dtype=np.float64,
    )
    ae_z = np.asarray(
        [[abs(float(query - expert)) for expert in expert_domains] for query in sample_domains],
        dtype=np.float64,
    )
    heldout = 0
    train_idx = np.where(sample_domains != heldout)[0].astype(np.int64)
    test_idx = np.where(sample_domains == heldout)[0].astype(np.int64)
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=heldout, expert_domains=expert_domains)
    return {
        "expert_domains": expert_domains,
        "sample_domains": sample_domains,
        "embeddings": embeddings,
        "true_nelbo": true_nelbo,
        "ae_z": ae_z,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "fold": fold,
        "domain_to_idx": {domain: idx for idx, domain in enumerate(expert_domains)},
        "global_eval": true_nelbo[test_idx],
    }


def _pairwise_cfg() -> dict:
    return {
        "hidden_dim": 4,
        "epochs": 1,
        "lr": 1.0e-3,
        "batch_size": 64,
        "device": "cpu",
        "margin": 1.0,
        "near_tie_delta": 0.0,
        "hard_pair_fraction": 0.5,
        "random_pair_fraction": 0.5,
        "max_pairs_per_sample": 8,
        "max_pairs_per_domain": 500,
        "run_utility_weighted_v2": True,
        "utility_weighted_v2": {
            "enabled": True,
            "primary_method": v2.PRIMARY_METHOD,
            "hard_pair_fraction": 0.40,
            "utility_pair_fraction": 0.40,
            "random_pair_fraction": 0.20,
            "pair_weight_alpha": 4.0,
            "pair_weight_delta_clip": 0.50,
            "pair_weight_min": 1.0,
            "pair_weight_max": 3.0,
        },
    }


def test_pairwise_ae_combined_v2_method_protocol_is_adoption_clean() -> None:
    protocol = _method_protocol(v2.PRIMARY_METHOD)
    assert protocol.adoption_eligible == 1
    assert protocol.diagnostic_only == 0
    assert protocol.routing_uses_eval_nelbo == 0
    assert protocol.routing_uses_eval_domain_statistics == 0


def test_pairwise_ae_combined_v2_config_validates() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "experiments" / "camelyon17" / "learned_utility_pairwise_ae_combined_v2.yaml")
    validate_config(cfg)
    pairwise = cfg["learned_utility"]["predictor_params"]["pairwise_ranker"]
    assert pairwise["run_utility_weighted_v2"] is True
    assert pairwise["utility_weighted_v2"]["primary_method"] == v2.PRIMARY_METHOD


def test_pairwise_ae_combined_v2_selection_is_per_seed_outer_center(monkeypatch) -> None:
    payload = _payload()

    def fake_eval(**kwargs):
        method = kwargs["method"]
        assert int(kwargs["outer_heldout_domain"]) == 0
        gap = 1.0 if method == v2.RANK_MARGIN_UNWEIGHTED else 2.0
        return {"mean_oracle_gap_pct": gap, "top1_oracle_hit": 0.5, "spearman": 0.1}, []

    monkeypatch.setattr(v2, "_evaluate_variant_on_indices", fake_eval)
    selected, rows = v2._source_inner_selection(
        embeddings=payload["embeddings"],
        sample_domains=payload["sample_domains"],
        true_nelbo=payload["true_nelbo"],
        expert_domains=payload["expert_domains"],
        domain_to_idx=payload["domain_to_idx"],
        train_idx=payload["train_idx"],
        outer_fold=payload["fold"],
        embedding_feature_dim=3,
        expert_feature_dim=5,
        ae_zscore_matrix=payload["ae_z"],
        pairwise_cfg=_pairwise_cfg(),
        seed=11,
        tie_policy="stable_expert_index",
    )
    assert selected == v2.RANK_MARGIN_UNWEIGHTED
    assert rows
    assert {int(row["seed"]) for row in rows} == {11}
    assert {int(row["outer_heldout_center"]) for row in rows} == {0}


def test_pairwise_ae_combined_v2_inner_selection_excludes_outer_target_rows(monkeypatch) -> None:
    payload = _payload()

    def fake_eval(**kwargs):
        train_domains = set(payload["sample_domains"][kwargs["train_idx"]].tolist())
        eval_domains = set(payload["sample_domains"][kwargs["eval_idx"]].tolist())
        assert 0 not in train_domains
        assert 0 not in eval_domains
        assert len(eval_domains) == 1
        assert next(iter(eval_domains)) not in train_domains
        return {"mean_oracle_gap_pct": 1.0, "top1_oracle_hit": 0.5, "spearman": 0.1}, []

    monkeypatch.setattr(v2, "_evaluate_variant_on_indices", fake_eval)
    v2._source_inner_selection(
        embeddings=payload["embeddings"],
        sample_domains=payload["sample_domains"],
        true_nelbo=payload["true_nelbo"],
        expert_domains=payload["expert_domains"],
        domain_to_idx=payload["domain_to_idx"],
        train_idx=payload["train_idx"],
        outer_fold=payload["fold"],
        embedding_feature_dim=3,
        expert_feature_dim=5,
        ae_zscore_matrix=payload["ae_z"],
        pairwise_cfg=_pairwise_cfg(),
        seed=11,
        tie_policy="stable_expert_index",
    )


def test_pairwise_ae_combined_v2_inner_validation_candidate_pool_excludes_outer_target_and_query_self(monkeypatch) -> None:
    payload = _payload()

    def fake_eval(**kwargs):
        fold = kwargs["eval_fold"]
        inner_domain = int(payload["sample_domains"][kwargs["eval_idx"][0]])
        assert 0 not in set(fold.candidate_expert_domains)
        assert inner_domain not in set(fold.candidate_expert_domains)
        return {"mean_oracle_gap_pct": 1.0, "top1_oracle_hit": 0.5, "spearman": 0.1}, []

    monkeypatch.setattr(v2, "_evaluate_variant_on_indices", fake_eval)
    v2._source_inner_selection(
        embeddings=payload["embeddings"],
        sample_domains=payload["sample_domains"],
        true_nelbo=payload["true_nelbo"],
        expert_domains=payload["expert_domains"],
        domain_to_idx=payload["domain_to_idx"],
        train_idx=payload["train_idx"],
        outer_fold=payload["fold"],
        embedding_feature_dim=3,
        expert_feature_dim=5,
        ae_zscore_matrix=payload["ae_z"],
        pairwise_cfg=_pairwise_cfg(),
        seed=11,
        tie_policy="stable_expert_index",
    )


def test_pairwise_ae_combined_v2_recomputes_ae_features_inside_inner_fold(monkeypatch) -> None:
    payload = _payload()
    seen_train_domains: list[set[int]] = []

    def fake_train_predict(**kwargs):
        seen_train_domains.append(set(payload["sample_domains"][kwargs["train_idx"]].tolist()))
        n_eval = int(kwargs["eval_idx"].shape[0])
        n_candidates = len(kwargs["eval_candidate_domains"])
        return np.zeros((n_eval, n_candidates), dtype=np.float64), np.ones(1), [], [], []

    monkeypatch.setattr(v2, "_train_predict_variant", fake_train_predict)
    metrics, _rows = v2._evaluate_variant_on_indices(
        method=v2.RANK_MARGIN_UNWEIGHTED,
        embeddings=payload["embeddings"],
        sample_domains=payload["sample_domains"],
        true_nelbo=payload["true_nelbo"],
        expert_domains=payload["expert_domains"],
        domain_to_idx=payload["domain_to_idx"],
        train_idx=payload["train_idx"][payload["sample_domains"][payload["train_idx"]] != 1],
        eval_idx=payload["train_idx"][payload["sample_domains"][payload["train_idx"]] == 1],
        outer_heldout_domain=0,
        globally_excluded_domains=[1],
        eval_fold=FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=payload["expert_domains"], excluded_domains=[1]),
        global_eval=payload["true_nelbo"][payload["train_idx"][payload["sample_domains"][payload["train_idx"]] == 1]],
        embedding_feature_dim=3,
        expert_feature_dim=5,
        ae_zscore_matrix=payload["ae_z"],
        pairwise_cfg=_pairwise_cfg(),
        seed=11,
        tie_policy="stable_expert_index",
    )
    assert metrics["top1_oracle_hit"] >= 0.0
    assert seen_train_domains
    assert all(0 not in domains and 1 not in domains for domains in seen_train_domains)


def test_pairwise_ae_combined_v2_ae_rank_features_are_query_relative() -> None:
    ae_z = np.asarray([[0.3, 0.1, 0.2], [1.0, 3.0, 2.0]], dtype=np.float64)
    features, names = v2._ae_rank_margin_features(
        ae_zscore_matrix=ae_z,
        sample_indices=np.asarray([0, 1], dtype=np.int64),
        candidate_domains=[0, 1, 2],
        expert_domains=[0, 1, 2],
    )
    rank_col = names.index("candidate_ae_rank")
    assert features[:3, rank_col].tolist() == [3.0, 1.0, 2.0]
    assert features[3:, rank_col].tolist() == [1.0, 3.0, 2.0]


def test_pairwise_ae_combined_v2_query_level_features_do_not_silently_cancel() -> None:
    ae_z = np.asarray([[0.1, 0.4, 0.9]], dtype=np.float64)
    features, names = v2._ae_rank_margin_features(
        ae_zscore_matrix=ae_z,
        sample_indices=np.asarray([0], dtype=np.int64),
        candidate_domains=[0, 1, 2],
        expert_domains=[0, 1, 2],
    )
    best_margin = names.index("ae_margin_if_best")
    second_margin = names.index("ae_margin_if_second")
    pair_diff = features[0] - features[1]
    assert abs(float(pair_diff[best_margin])) > 0.0
    assert abs(float(pair_diff[second_margin])) > 0.0


def test_pairwise_ae_combined_v2_pair_weights_follow_source_inner_scaled_formula() -> None:
    cfg = _pairwise_cfg()["utility_weighted_v2"]
    weight = v2._pair_weight(better_nelbo=1.0, worse_nelbo=1.25, source_inner_median_abs_nelbo=1.0, cfg=cfg)
    assert weight == 2.0
    clipped = v2._pair_weight(better_nelbo=1.0, worse_nelbo=3.0, source_inner_median_abs_nelbo=1.0, cfg=cfg)
    assert clipped == 3.0


def test_pairwise_ae_combined_v2_weight_scaling_is_source_inner_only() -> None:
    cfg = _pairwise_cfg()["utility_weighted_v2"]
    pairs, weights, rows = v2._build_utility_weighted_training_pairs(
        y_train=np.asarray([1.0, 1.2, 3.0, 3.5], dtype=np.float64),
        q_train=np.asarray([1, 1, 2, 2], dtype=np.int64),
        s_train=np.asarray([10, 10, 11, 11], dtype=np.int64),
        experts_per_sample=2,
        near_tie_delta=0.0,
        hard_pair_fraction=0.0,
        utility_pair_fraction=1.0,
        random_pair_fraction=0.0,
        max_pairs_per_sample=1,
        max_pairs_per_domain=10,
        seed=1,
        cfg=cfg,
    )
    assert pairs
    assert weights.shape[0] == len(pairs)
    assert all(float(row["source_inner_median_abs_nelbo"]) == 2.1 for row in rows)


def test_pairwise_ae_combined_v2_pair_sampling_not_domain_dominated() -> None:
    cfg = _pairwise_cfg()["utility_weighted_v2"]
    _pairs, _weights, rows = v2._build_utility_weighted_training_pairs(
        y_train=np.asarray([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float64),
        q_train=np.asarray([1, 1, 1, 1, 2, 2], dtype=np.int64),
        s_train=np.asarray([10, 10, 11, 11, 12, 12], dtype=np.int64),
        experts_per_sample=2,
        near_tie_delta=0.0,
        hard_pair_fraction=0.0,
        utility_pair_fraction=1.0,
        random_pair_fraction=0.0,
        max_pairs_per_sample=1,
        max_pairs_per_domain=1,
        seed=1,
        cfg=cfg,
    )
    selected_by_domain = {}
    for row in rows:
        selected_by_domain[int(row["query_domain"])] = selected_by_domain.get(int(row["query_domain"]), 0) + int(row["n_selected"])
    assert selected_by_domain[1] == 1
    assert selected_by_domain[2] == 1


def test_pairwise_ae_combined_v2_inner_selection_table_emitted(monkeypatch) -> None:
    payload = _payload()

    def fake_eval(**kwargs):
        method = kwargs["method"]
        gap = 0.5 if method == v2.RAW_AE_WEIGHTED else 1.0
        return {"mean_oracle_gap_pct": gap, "top1_oracle_hit": 0.5, "spearman": 0.1}, []

    monkeypatch.setattr(v2, "_evaluate_variant_on_indices", fake_eval)
    _selected, rows = v2._source_inner_selection(
        embeddings=payload["embeddings"],
        sample_domains=payload["sample_domains"],
        true_nelbo=payload["true_nelbo"],
        expert_domains=payload["expert_domains"],
        domain_to_idx=payload["domain_to_idx"],
        train_idx=payload["train_idx"],
        outer_fold=payload["fold"],
        embedding_feature_dim=3,
        expert_feature_dim=5,
        ae_zscore_matrix=payload["ae_z"],
        pairwise_cfg=_pairwise_cfg(),
        seed=11,
        tie_policy="stable_expert_index",
    )
    assert rows
    assert "heldout_target_nelbo_used_for_selection" in rows[0]
    assert all(int(row["heldout_target_nelbo_used_for_selection"]) == 0 for row in rows)


def test_pairwise_ae_combined_v2_inner_selection_can_fallback_to_baseline(monkeypatch) -> None:
    payload = _payload()

    def fake_eval(**kwargs):
        method = kwargs["method"]
        gap = 1.0 if method == v2.BASELINE_METHOD else 2.5
        return {"mean_oracle_gap_pct": gap, "top1_oracle_hit": 0.5, "spearman": 0.1}, []

    monkeypatch.setattr(v2, "_evaluate_variant_on_indices", fake_eval)
    selected, rows = v2._source_inner_selection(
        embeddings=payload["embeddings"],
        sample_domains=payload["sample_domains"],
        true_nelbo=payload["true_nelbo"],
        expert_domains=payload["expert_domains"],
        domain_to_idx=payload["domain_to_idx"],
        train_idx=payload["train_idx"],
        outer_fold=payload["fold"],
        embedding_feature_dim=3,
        expert_feature_dim=5,
        ae_zscore_matrix=payload["ae_z"],
        pairwise_cfg=_pairwise_cfg(),
        seed=11,
        tie_policy="stable_expert_index",
    )
    assert selected == v2.BASELINE_METHOD
    assert all(int(row["fallback_to_baseline"]) == 1 for row in rows)


def test_pairwise_ae_combined_v2_reports_selected_method_counts(tmp_path: Path) -> None:
    decision_rows = [
        {"selected_method": v2.BASELINE_METHOD, "seed": 11, "outer_heldout_center": 0, "delta_gap_vs_baseline": 0.0},
        {"selected_method": v2.RAW_AE_WEIGHTED, "seed": 11, "outer_heldout_center": 0, "delta_gap_vs_baseline": 1.0},
    ]
    artifacts = v2.write_pairwise_ae_combined_v2_artifacts(
        reports_dir=tmp_path,
        training_rows=[],
        feature_rows=[{"feature_name": "__all__", "feature_nonzero_rate_after_pairwise_difference": 0.5}],
        inner_selection_rows=[],
        pair_prediction_rows=[],
        decision_rows=decision_rows,
    )
    assert artifacts
    summary = json.loads((tmp_path / "pairwise_ae_combined_v2_decision_summary.json").read_text(encoding="utf-8"))
    assert summary["selected_method_count_total"][v2.BASELINE_METHOD] == 1
    assert summary["selected_method_count_by_seed"]["11"][v2.RAW_AE_WEIGHTED] == 1


def test_pairwise_ae_combined_v2_reports_fallback_and_adoption_rates(tmp_path: Path) -> None:
    decision_rows = [
        {"selected_method": v2.BASELINE_METHOD, "seed": 11, "outer_heldout_center": 0, "delta_gap_vs_baseline": 0.0},
        {"selected_method": v2.RANK_MARGIN_WEIGHTED, "seed": 11, "outer_heldout_center": 0, "delta_gap_vs_baseline": 0.5},
    ]
    v2.write_pairwise_ae_combined_v2_artifacts(
        reports_dir=tmp_path,
        training_rows=[],
        feature_rows=[],
        inner_selection_rows=[],
        pair_prediction_rows=[],
        decision_rows=decision_rows,
    )
    summary = json.loads((tmp_path / "pairwise_ae_combined_v2_decision_summary.json").read_text(encoding="utf-8"))
    assert summary["fallback_to_baseline_rate"] == 0.5
    assert summary["v2_adoption_rate"] == 0.5


def test_pairwise_ae_combined_v2_tiny_capped_smoke_run(monkeypatch) -> None:
    payload = _payload()

    def fake_selection(**_kwargs):
        return v2.BASELINE_METHOD, [
            {
                "seed": 11,
                "outer_heldout_center": 0,
                "candidate_method": v2.BASELINE_METHOD,
                "selected_method": v2.BASELINE_METHOD,
                "heldout_target_nelbo_used_for_selection": 0,
            }
        ]

    def fake_train_predict(**kwargs):
        n_eval = int(kwargs["eval_idx"].shape[0])
        n_candidates = len(kwargs["eval_candidate_domains"])
        pred = np.tile(np.arange(n_candidates, dtype=np.float64), (n_eval, 1))
        return pred, np.ones(1), [], [{"method": kwargs["method"], "feature_name": "__all__", "feature_nonzero_rate_after_pairwise_difference": 1.0}], []

    monkeypatch.setattr(v2, "_source_inner_selection", fake_selection)
    monkeypatch.setattr(v2, "_train_predict_variant", fake_train_predict)
    out = v2.run_pairwise_ae_combined_v2_for_fold(
        embeddings=payload["embeddings"],
        sample_domains=payload["sample_domains"],
        true_nelbo=payload["true_nelbo"],
        expert_domains=payload["expert_domains"],
        domain_to_idx=payload["domain_to_idx"],
        train_idx=payload["train_idx"],
        test_idx=payload["test_idx"],
        fold=payload["fold"],
        global_eval=payload["global_eval"],
        pairwise_cfg=_pairwise_cfg(),
        seed=11,
        embedding_feature_dim=3,
        expert_feature_dim=5,
        tie_policy="stable_expert_index",
        ae_zscore_matrix=payload["ae_z"],
    )
    assert out.sample_rows
    assert any(row["method"] == v2.PRIMARY_METHOD for row in out.sample_rows)
    assert out.decision_rows


def test_pairwise_ae_combined_v2_required_artifact_csv_is_readable(tmp_path: Path) -> None:
    rows = [{"selected_method": v2.BASELINE_METHOD, "seed": 11, "outer_heldout_center": 0, "delta_gap_vs_baseline": 0.0}]
    v2.write_pairwise_ae_combined_v2_artifacts(
        reports_dir=tmp_path,
        training_rows=[],
        feature_rows=[],
        inner_selection_rows=[],
        pair_prediction_rows=[],
        decision_rows=rows,
    )
    with (tmp_path / "pairwise_ae_combined_v2_decision_table.csv").open("r", encoding="utf-8", newline="") as f:
        assert list(csv.DictReader(f))[0]["selected_method"] == v2.BASELINE_METHOD
