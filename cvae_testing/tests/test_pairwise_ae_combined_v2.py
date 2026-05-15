from __future__ import annotations

import csv
import importlib.util
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


def _load_decision_builder():
    path = PROJECT_ROOT / "scripts" / "build_pairwise_ae_combined_v2_decision_table.py"
    spec = importlib.util.spec_from_file_location("build_pairwise_ae_combined_v2_decision_table", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _pairwise_cfg(
    *,
    strict: bool = False,
    target_batch: bool = False,
    v31: bool = False,
    strict_overrides: dict | None = None,
    target_batch_overrides: dict | None = None,
) -> dict:
    utility = {
        "enabled": True,
        "primary_method": v2.TARGET_BATCH_AGREEMENT_V31_PRIMARY_METHOD
        if v31
        else v2.TARGET_BATCH_AGREEMENT_PRIMARY_METHOD
        if target_batch
        else v2.STRICT_PRIMARY_METHOD
        if strict
        else v2.PRIMARY_METHOD,
        "hard_pair_fraction": 0.40,
        "utility_pair_fraction": 0.40,
        "random_pair_fraction": 0.20,
        "pair_weight_alpha": 4.0,
        "pair_weight_delta_clip": 0.50,
        "pair_weight_min": 1.0,
        "pair_weight_max": 3.0,
    }
    if strict or target_batch or v31:
        strict_cfg = {
            "min_macro_gap_reduction_pp": 0.5,
            "max_top1_drop_abs": 0.02,
            "max_spearman_drop_abs": 0.03,
            "max_worst_inner_center_gap_degradation_pp": 0.25,
            "min_positive_inner_center_rate": 0.75,
            "min_non_degrading_inner_center_rate": 1.0,
            "min_passing_inner_centers": 2,
        }
        strict_cfg.update(strict_overrides or {})
        utility.update(
            {
                "selection_mode": "target_batch_agreement_gated" if target_batch or v31 else "strict_adoption",
                "fallback_method": v2.BASELINE_METHOD,
                "strict_adoption": strict_cfg,
            }
        )
        if target_batch or v31:
            agreement = {
                "agreement_threshold": 0.60,
                "agreement_threshold_source": "predeclared_development_seed_diagnostic",
                "reference_method": v2.RAW_AE_WEIGHTED,
                "group_key_candidates": ["patient_id", "slide_id", "case_id"],
                "min_query_count": 1,
                "min_group_count": 1,
            }
            if v31:
                agreement["gate_scope"] = "all_nonbaseline"
            agreement.update(target_batch_overrides or {})
            utility["target_batch_agreement"] = agreement
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
        "utility_weighted_v2": utility,
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


def test_pairwise_ae_combined_v2_strict_config_validates() -> None:
    cfg = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "camelyon17"
        / "learned_utility_pairwise_ae_combined_v2_strict.yaml"
    )
    validate_config(cfg)
    pairwise = cfg["learned_utility"]["predictor_params"]["pairwise_ranker"]
    utility = pairwise["utility_weighted_v2"]
    assert utility["primary_method"] == v2.STRICT_PRIMARY_METHOD
    assert utility["selection_mode"] == "strict_adoption"
    assert utility["fallback_method"] == v2.BASELINE_METHOD


def test_pairwise_ae_combined_v3_target_batch_config_validates() -> None:
    cfg = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "camelyon17"
        / "learned_utility_pairwise_ae_combined_v3_target_batch_agreement.yaml"
    )
    validate_config(cfg)
    utility = cfg["learned_utility"]["predictor_params"]["pairwise_ranker"]["utility_weighted_v2"]
    assert utility["primary_method"] == v2.TARGET_BATCH_AGREEMENT_PRIMARY_METHOD
    assert utility["selection_mode"] == "target_batch_agreement_gated"
    assert utility["target_batch_agreement"]["agreement_threshold"] == 0.60


def test_pairwise_ae_combined_v31_config_validates_all_nonbaseline_gate() -> None:
    cfg = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "camelyon17"
        / "learned_utility_pairwise_ae_combined_v31_target_batch_agreement.yaml"
    )
    validate_config(cfg)
    utility = cfg["learned_utility"]["predictor_params"]["pairwise_ranker"]["utility_weighted_v2"]
    assert utility["primary_method"] == v2.TARGET_BATCH_AGREEMENT_V31_PRIMARY_METHOD
    assert utility["selection_mode"] == "target_batch_agreement_gated"
    assert utility["target_batch_agreement"]["gate_scope"] == "all_nonbaseline"


def test_pairwise_ae_combined_v3_primary_method_must_match_mode() -> None:
    cfg = _pairwise_cfg(target_batch=True)
    cfg["utility_weighted_v2"]["primary_method"] = v2.STRICT_PRIMARY_METHOD
    try:
        v2._primary_method(cfg)
    except Exception as exc:
        assert v2.TARGET_BATCH_AGREEMENT_PRIMARY_METHOD in str(exc)
    else:
        raise AssertionError("target_batch_agreement_gated accepted the wrong primary method")


def _run_strict_selection(monkeypatch, payload, metric_fn, *, strict_overrides: dict | None = None):
    def fake_eval(**kwargs):
        inner_domain = int(payload["sample_domains"][kwargs["eval_idx"][0]])
        return metric_fn(str(kwargs["method"]), inner_domain), []

    monkeypatch.setattr(v2, "_evaluate_variant_on_indices", fake_eval)
    return v2._source_inner_selection(
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
        pairwise_cfg=_pairwise_cfg(strict=True, strict_overrides=strict_overrides),
        seed=11,
        tie_policy="stable_expert_index",
    )


def test_pairwise_ae_combined_v2_strict_uses_pairwise_ae_combined_fallback(monkeypatch) -> None:
    payload = _payload()

    def metrics(method: str, _inner: int) -> dict:
        gap = 1.0 if method == v2.BASELINE_METHOD else 1.25
        return {"mean_oracle_gap_pct": gap, "top1_oracle_hit": 0.8, "spearman": 0.7}

    selected, rows = _run_strict_selection(monkeypatch, payload, metrics)
    assert selected == v2.BASELINE_METHOD
    assert rows
    assert all(int(row["fallback_used"]) == 1 for row in rows)


def test_pairwise_ae_combined_v2_strict_rejects_worst_inner_center_degradation(monkeypatch) -> None:
    payload = _payload()

    def metrics(method: str, inner: int) -> dict:
        if method == v2.BASELINE_METHOD:
            return {"mean_oracle_gap_pct": 1.0, "top1_oracle_hit": 0.8, "spearman": 0.7}
        gap = 0.3 if inner != 1 else 1.4
        return {"mean_oracle_gap_pct": gap, "top1_oracle_hit": 0.8, "spearman": 0.7}

    selected, rows = _run_strict_selection(monkeypatch, payload, metrics)
    assert selected == v2.BASELINE_METHOD
    candidate_rows = [row for row in rows if row["candidate_method"] == v2.RANK_MARGIN_UNWEIGHTED]
    assert candidate_rows
    assert all(int(row["passed_worst_center_gate"]) == 0 for row in candidate_rows)


def test_pairwise_ae_combined_v2_strict_requires_positive_inner_center_rate(monkeypatch) -> None:
    payload = _payload()

    def metrics(method: str, inner: int) -> dict:
        if method == v2.BASELINE_METHOD:
            return {"mean_oracle_gap_pct": 1.0, "top1_oracle_hit": 0.8, "spearman": 0.7}
        gap = 0.4 if inner in {1, 2} else 1.0
        return {"mean_oracle_gap_pct": gap, "top1_oracle_hit": 0.8, "spearman": 0.7}

    selected, rows = _run_strict_selection(monkeypatch, payload, metrics)
    assert selected == v2.BASELINE_METHOD
    assert any(float(row["positive_inner_center_rate"]) == 0.5 for row in rows if row["candidate_method"] != v2.BASELINE_METHOD)


def test_pairwise_ae_combined_v2_strict_requires_min_macro_gap_reduction(monkeypatch) -> None:
    payload = _payload()

    def metrics(method: str, _inner: int) -> dict:
        gap = 1.0 if method == v2.BASELINE_METHOD else 0.75
        return {"mean_oracle_gap_pct": gap, "top1_oracle_hit": 0.8, "spearman": 0.7}

    selected, rows = _run_strict_selection(monkeypatch, payload, metrics)
    assert selected == v2.BASELINE_METHOD
    assert all(int(row["passed_macro_gap_gate"]) == 0 for row in rows if row["candidate_method"] != v2.BASELINE_METHOD)


def test_pairwise_ae_combined_v2_strict_gap_reduction_sign_is_baseline_minus_candidate(monkeypatch) -> None:
    payload = _payload()

    def metrics(method: str, _inner: int) -> dict:
        gap = 2.0 if method == v2.BASELINE_METHOD else 1.0
        return {"mean_oracle_gap_pct": gap, "top1_oracle_hit": 0.8, "spearman": 0.7}

    selected, rows = _run_strict_selection(monkeypatch, payload, metrics)
    assert selected in {v2.RANK_MARGIN_UNWEIGHTED, v2.RAW_AE_WEIGHTED, v2.RANK_MARGIN_WEIGHTED}
    candidate_row = next(row for row in rows if row["candidate_method"] == selected)
    assert float(candidate_row["gap_reduction_pp"]) == 1.0
    assert float(candidate_row["inner_center_gap_degradation_pp"]) == -1.0


def test_pairwise_ae_combined_v2_strict_tiebreak_prefers_simpler_method(monkeypatch) -> None:
    payload = _payload()

    def metrics(method: str, _inner: int) -> dict:
        gap = 2.0 if method == v2.BASELINE_METHOD else 1.0
        return {"mean_oracle_gap_pct": gap, "top1_oracle_hit": 0.8, "spearman": 0.7}

    selected, _rows = _run_strict_selection(monkeypatch, payload, metrics)
    assert selected == v2.RANK_MARGIN_UNWEIGHTED


def test_pairwise_ae_combined_v2_strict_selection_uses_no_target_nelbo(monkeypatch) -> None:
    payload = _payload()

    def metrics(method: str, _inner: int) -> dict:
        gap = 2.0 if method == v2.BASELINE_METHOD else 1.0
        return {"mean_oracle_gap_pct": gap, "top1_oracle_hit": 0.8, "spearman": 0.7}

    _selected, rows = _run_strict_selection(monkeypatch, payload, metrics)
    assert rows
    assert all(int(row["heldout_target_nelbo_used_for_selection"]) == 0 for row in rows)


def test_pairwise_ae_combined_v2_strict_recomputes_inner_fold_feature_normalization(monkeypatch) -> None:
    payload = _payload()
    seen_train_domains: list[set[int]] = []

    def fake_train_predict(**kwargs):
        seen_train_domains.append(set(payload["sample_domains"][kwargs["train_idx"]].tolist()))
        n_eval = int(kwargs["eval_idx"].shape[0])
        n_candidates = len(kwargs["eval_candidate_domains"])
        return np.zeros((n_eval, n_candidates), dtype=np.float64), np.ones(1), [], [], []

    monkeypatch.setattr(v2, "_train_predict_variant", fake_train_predict)
    v2._evaluate_variant_on_indices(
        method=v2.RANK_MARGIN_WEIGHTED,
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
        pairwise_cfg=_pairwise_cfg(strict=True),
        seed=11,
        tie_policy="stable_expert_index",
    )
    assert seen_train_domains
    assert all(0 not in domains and 1 not in domains for domains in seen_train_domains)


def test_pairwise_ae_combined_v2_strict_inner_candidate_pool_excludes_outer_target_and_query_self(monkeypatch) -> None:
    payload = _payload()

    def fake_eval(**kwargs):
        fold = kwargs["eval_fold"]
        inner_domain = int(payload["sample_domains"][kwargs["eval_idx"][0]])
        assert 0 not in set(fold.candidate_expert_domains)
        assert inner_domain not in set(fold.candidate_expert_domains)
        return {"mean_oracle_gap_pct": 1.0, "top1_oracle_hit": 0.8, "spearman": 0.7}, []

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
        pairwise_cfg=_pairwise_cfg(strict=True),
        seed=11,
        tie_policy="stable_expert_index",
    )


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


def test_pairwise_ae_combined_v2_strict_fallback_rows_match_baseline(monkeypatch) -> None:
    payload = _payload()

    def fake_selection(**_kwargs):
        return v2.BASELINE_METHOD, [
            {
                "seed": 11,
                "outer_heldout_center": 0,
                "candidate_method": v2.BASELINE_METHOD,
                "selected_method": v2.BASELINE_METHOD,
                "selected_variant": v2.BASELINE_METHOD,
                "fallback_used": 1,
                "heldout_target_nelbo_used_for_selection": 0,
            }
        ]

    def fake_train_predict(**kwargs):
        n_eval = int(kwargs["eval_idx"].shape[0])
        n_candidates = len(kwargs["eval_candidate_domains"])
        pred = np.tile(np.arange(n_candidates, dtype=np.float64), (n_eval, 1))
        if kwargs["method"] != v2.BASELINE_METHOD:
            pred = pred[:, ::-1]
        return pred, np.ones(1), [], [], []

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
        pairwise_cfg=_pairwise_cfg(strict=True),
        seed=11,
        embedding_feature_dim=3,
        expert_feature_dim=5,
        tie_policy="stable_expert_index",
        ae_zscore_matrix=payload["ae_z"],
    )
    assert out.decision_rows
    assert all(row["selected_expert"] == row["baseline_selected_expert"] for row in out.decision_rows)
    assert all(row["primary_method"] == v2.STRICT_PRIMARY_METHOD for row in out.decision_rows)
    assert all(int(row["fallback_used"]) == 1 for row in out.decision_rows)


def test_pairwise_ae_combined_v2_strict_reports_adoption_and_fallback_rates(tmp_path: Path) -> None:
    decision_rows = [
        {
            "primary_method": v2.STRICT_PRIMARY_METHOD,
            "selected_method": v2.BASELINE_METHOD,
            "seed": 11,
            "outer_heldout_center": 0,
            "delta_gap_vs_baseline_pp": 0.0,
            "top1_oracle_hit": 1,
            "baseline_top1_oracle_hit": 1,
        },
        {
            "primary_method": v2.STRICT_PRIMARY_METHOD,
            "selected_method": v2.RANK_MARGIN_UNWEIGHTED,
            "seed": 11,
            "outer_heldout_center": 1,
            "delta_gap_vs_baseline_pp": 0.5,
            "top1_oracle_hit": 1,
            "baseline_top1_oracle_hit": 1,
        },
    ]
    artifacts = v2.write_pairwise_ae_combined_v2_artifacts(
        reports_dir=tmp_path,
        training_rows=[],
        feature_rows=[],
        inner_selection_rows=[],
        pair_prediction_rows=[],
        decision_rows=decision_rows,
    )
    summary = json.loads((tmp_path / "pairwise_ae_combined_v2_strict_decision_summary.json").read_text(encoding="utf-8"))
    assert artifacts["pairwise_ae_combined_v2_strict_decision_table"] == "pairwise_ae_combined_v2_strict_decision_table.csv"
    assert summary["strict_v2_adoption_rate"] == 0.5
    assert summary["fallback_to_pairwise_ae_combined_rate"] == 0.5
    assert summary["always_baseline_fallback"] is False


def test_pairwise_ae_combined_v2_strict_decision_table_builder(tmp_path: Path) -> None:
    builder = _load_decision_builder()
    reports = tmp_path / "outputs" / "camelyon17" / "learned_utility_pairwise_ae_combined_v2_strict" / "run_seed11" / "reports"
    reports.mkdir(parents=True)
    result_path = reports / "learned_utility_results.json"
    result_path.write_text("{}", encoding="utf-8")
    v2._write_csv(
        reports / "pairwise_ae_combined_v2_strict_decision_table.csv",
        [
            {
                "seed": 11,
                "outer_heldout_center": 0,
                "selected_method": v2.RANK_MARGIN_UNWEIGHTED,
                "delta_gap_vs_baseline_pp": 0.5,
            }
        ],
    )
    v2._write_csv(
        reports / "learned_utility_domain_breakdown.csv",
        [
            {"method": v2.STRICT_PRIMARY_METHOD, "query_domain": 0, "mean_oracle_gap_pct": 1.0, "top1_oracle_hit": 0.8, "spearman": 0.7},
            {"method": v2.BASELINE_METHOD, "query_domain": 0, "mean_oracle_gap_pct": 1.5, "top1_oracle_hit": 0.8, "spearman": 0.7},
            {"method": "ae_argmin_zscore", "query_domain": 0, "mean_oracle_gap_pct": 2.0, "top1_oracle_hit": 0.7, "spearman": 0.6},
            {"method": "metadata_routing", "query_domain": 0, "mean_oracle_gap_pct": 2.5, "top1_oracle_hit": 0.6, "spearman": 0.5},
        ],
    )
    decisions, seed_domain_rows, any_strict = builder._aggregate_decisions([result_path])
    summary = builder._verdict(seed_domain_rows)
    assert any_strict is True
    assert decisions[0]["decision_artifact"].endswith("pairwise_ae_combined_v2_strict_decision_table.csv")
    assert summary["selected_nonbaseline_v2_at_least_once"] is True


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


def test_pairwise_ae_combined_v2_strict_tiny_capped_smoke_run(monkeypatch) -> None:
    payload = _payload()

    def fake_selection(**_kwargs):
        return v2.RAW_AE_WEIGHTED, [
            {
                "seed": 11,
                "outer_heldout_center": 0,
                "candidate_method": v2.RAW_AE_WEIGHTED,
                "selected_method": v2.RAW_AE_WEIGHTED,
                "selected_variant": v2.RAW_AE_WEIGHTED,
                "fallback_used": 0,
                "heldout_target_nelbo_used_for_selection": 0,
            }
        ]

    def fake_train_predict(**kwargs):
        n_eval = int(kwargs["eval_idx"].shape[0])
        n_candidates = len(kwargs["eval_candidate_domains"])
        pred = np.tile(np.arange(n_candidates, dtype=np.float64), (n_eval, 1))
        return pred, np.ones(1), [], [], []

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
        pairwise_cfg=_pairwise_cfg(strict=True),
        seed=11,
        embedding_feature_dim=3,
        expert_feature_dim=5,
        tie_policy="stable_expert_index",
        ae_zscore_matrix=payload["ae_z"],
    )
    assert any(row["method"] == v2.STRICT_PRIMARY_METHOD for row in out.sample_rows)
    assert all(row["primary_method"] == v2.STRICT_PRIMARY_METHOD for row in out.decision_rows)


def test_pairwise_ae_combined_v3_gate_records_target_batch_usage() -> None:
    payload = _payload()
    n = int(payload["test_idx"].shape[0])
    preds = {
        v2.RANK_MARGIN_WEIGHTED: np.tile(np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64), (n, 1)),
        v2.RAW_AE_WEIGHTED: np.tile(np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64), (n, 1)),
    }
    metadata = [{"patient_id": f"p{i % 2}"} for i in range(len(payload["sample_domains"]))]
    deployed, fields = v2._target_batch_agreement_policy(
        source_inner_selected_method=v2.RANK_MARGIN_WEIGHTED,
        method_predictions=preds,
        candidate_domains=payload["fold"].candidate_expert_domains,
        test_idx=payload["test_idx"],
        sample_metadata=metadata,
        pairwise_cfg=_pairwise_cfg(target_batch=True),
    )
    assert deployed == v2.RANK_MARGIN_WEIGHTED
    assert int(fields["used_target_embeddings_for_gate"]) == 1
    assert int(fields["used_target_group_ids_for_gate"]) == 1
    assert int(fields["used_target_labels_for_gate"]) == 0
    assert int(fields["used_target_nelbo_for_gate"]) == 0
    assert int(fields["used_target_support_for_gate"]) == 0
    assert int(fields["used_target_fitting_for_gate"]) == 0
    assert int(fields["heldout_target_nelbo_used_for_selection"]) == 0


def test_pairwise_ae_combined_v3_low_agreement_falls_back_exactly_to_baseline(monkeypatch) -> None:
    payload = _payload()
    metadata = [{"patient_id": f"p{i % 2}"} for i in range(len(payload["sample_domains"]))]

    def fake_selection(**_kwargs):
        return v2.RANK_MARGIN_WEIGHTED, [
            {
                "seed": 11,
                "outer_heldout_center": 0,
                "candidate_method": v2.RANK_MARGIN_WEIGHTED,
                "selected_method": v2.RANK_MARGIN_WEIGHTED,
                "heldout_target_nelbo_used_for_selection": 0,
            }
        ]

    def fake_train_predict(**kwargs):
        n_eval = int(kwargs["eval_idx"].shape[0])
        n_candidates = len(kwargs["eval_candidate_domains"])
        if kwargs["method"] == v2.BASELINE_METHOD:
            pred = np.tile(np.arange(n_candidates, dtype=np.float64), (n_eval, 1))
        elif kwargs["method"] == v2.RANK_MARGIN_WEIGHTED:
            pred = np.tile(np.arange(n_candidates, dtype=np.float64), (n_eval, 1))
        elif kwargs["method"] == v2.RAW_AE_WEIGHTED:
            pred = np.tile(np.arange(n_candidates, dtype=np.float64)[::-1], (n_eval, 1))
        else:
            pred = np.tile(np.arange(n_candidates, dtype=np.float64), (n_eval, 1))
        return pred, np.ones(1), [], [], []

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
        pairwise_cfg=_pairwise_cfg(target_batch=True),
        seed=101,
        embedding_feature_dim=3,
        expert_feature_dim=5,
        tie_policy="stable_expert_index",
        ae_zscore_matrix=payload["ae_z"],
        sample_metadata=metadata,
    )
    assert out.decision_rows
    assert all(row["deployed_method"] == v2.BASELINE_METHOD for row in out.decision_rows)
    assert all(row["selected_expert"] == row["baseline_selected_expert"] for row in out.decision_rows)
    assert all(int(row["agreement_gate_applied"]) == 1 for row in out.decision_rows)
    assert all(int(row["agreement_gate_passed"]) == 0 for row in out.decision_rows)


def test_pairwise_ae_combined_v3_high_agreement_deploys_selected_v2(monkeypatch) -> None:
    payload = _payload()
    metadata = [{"patient_id": f"p{i % 2}"} for i in range(len(payload["sample_domains"]))]

    def fake_selection(**_kwargs):
        return v2.RANK_MARGIN_UNWEIGHTED, []

    def fake_train_predict(**kwargs):
        n_eval = int(kwargs["eval_idx"].shape[0])
        n_candidates = len(kwargs["eval_candidate_domains"])
        base = np.tile(np.arange(n_candidates, dtype=np.float64), (n_eval, 1))
        if kwargs["method"] == v2.BASELINE_METHOD:
            return base[:, ::-1], np.ones(1), [], [], []
        return base, np.ones(1), [], [], []

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
        pairwise_cfg=_pairwise_cfg(target_batch=True),
        seed=101,
        embedding_feature_dim=3,
        expert_feature_dim=5,
        tie_policy="stable_expert_index",
        ae_zscore_matrix=payload["ae_z"],
        sample_metadata=metadata,
    )
    assert out.decision_rows
    assert all(row["deployed_method"] == v2.RANK_MARGIN_UNWEIGHTED for row in out.decision_rows)
    assert all(int(row["agreement_gate_passed"]) == 1 for row in out.decision_rows)


def test_pairwise_ae_combined_v3_group_macro_agreement_enforced() -> None:
    payload = _payload()
    preds = {
        v2.RANK_MARGIN_WEIGHTED: np.asarray([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]]),
        v2.RAW_AE_WEIGHTED: np.asarray([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]]),
    }
    metadata = [{"patient_id": "a"} for _ in range(len(payload["sample_domains"]))]
    for idx in payload["test_idx"][1:]:
        metadata[int(idx)] = {"patient_id": "b"}
    deployed, fields = v2._target_batch_agreement_policy(
        source_inner_selected_method=v2.RANK_MARGIN_WEIGHTED,
        method_predictions=preds,
        candidate_domains=payload["fold"].candidate_expert_domains,
        test_idx=payload["test_idx"],
        sample_metadata=metadata,
        pairwise_cfg=_pairwise_cfg(target_batch=True, target_batch_overrides={"agreement_threshold": 0.60}),
    )
    assert deployed == v2.BASELINE_METHOD
    assert float(fields["selected_vs_raw_agreement_rate_query_weighted"]) >= 0.0
    assert float(fields["selected_vs_raw_agreement_rate_group_macro"]) < 0.60


def test_pairwise_ae_combined_v3_small_batches_fallback_to_baseline() -> None:
    payload = _payload()
    n = int(payload["test_idx"].shape[0])
    pred = np.tile(np.arange(len(payload["fold"].candidate_expert_domains), dtype=np.float64), (n, 1))
    metadata = [{"patient_id": f"p{i % 2}"} for i in range(len(payload["sample_domains"]))]
    deployed, fields = v2._target_batch_agreement_policy(
        source_inner_selected_method=v2.RANK_MARGIN_WEIGHTED,
        method_predictions={v2.RANK_MARGIN_WEIGHTED: pred, v2.RAW_AE_WEIGHTED: pred},
        candidate_domains=payload["fold"].candidate_expert_domains,
        test_idx=payload["test_idx"],
        sample_metadata=metadata,
        pairwise_cfg=_pairwise_cfg(target_batch=True, target_batch_overrides={"min_query_count": 100}),
    )
    assert deployed == v2.BASELINE_METHOD
    assert int(fields["agreement_gate_skipped_due_to_small_batch"]) == 1


def test_pairwise_ae_combined_v31_raw_selected_low_peer_agreement_falls_back() -> None:
    payload = _payload()
    n = int(payload["test_idx"].shape[0])
    raw = np.tile(np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64), (n, 1))
    peer = np.tile(np.asarray([3.0, 2.0, 1.0, 0.0], dtype=np.float64), (n, 1))
    metadata = [{"patient_id": f"p{i % 2}"} for i in range(len(payload["sample_domains"]))]
    deployed, fields = v2._target_batch_agreement_policy(
        source_inner_selected_method=v2.RAW_AE_WEIGHTED,
        method_predictions={
            v2.RAW_AE_WEIGHTED: raw,
            v2.RANK_MARGIN_UNWEIGHTED: peer,
            v2.RANK_MARGIN_WEIGHTED: peer,
        },
        candidate_domains=payload["fold"].candidate_expert_domains,
        test_idx=payload["test_idx"],
        sample_metadata=metadata,
        pairwise_cfg=_pairwise_cfg(v31=True),
    )
    assert deployed == v2.BASELINE_METHOD
    assert int(fields["agreement_gate_applied"]) == 1
    assert int(fields["agreement_gate_passed"]) == 0
    assert fields["gate_scope"] == "all_nonbaseline"


def test_pairwise_ae_combined_v31_raw_selected_high_peer_agreement_deploys_raw() -> None:
    payload = _payload()
    n = int(payload["test_idx"].shape[0])
    raw = np.tile(np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64), (n, 1))
    disagree = np.tile(np.asarray([3.0, 2.0, 1.0, 0.0], dtype=np.float64), (n, 1))
    metadata = [{"patient_id": f"p{i % 2}"} for i in range(len(payload["sample_domains"]))]
    deployed, fields = v2._target_batch_agreement_policy(
        source_inner_selected_method=v2.RAW_AE_WEIGHTED,
        method_predictions={
            v2.RAW_AE_WEIGHTED: raw,
            v2.RANK_MARGIN_UNWEIGHTED: raw,
            v2.RANK_MARGIN_WEIGHTED: disagree,
        },
        candidate_domains=payload["fold"].candidate_expert_domains,
        test_idx=payload["test_idx"],
        sample_metadata=metadata,
        pairwise_cfg=_pairwise_cfg(v31=True),
    )
    assert deployed == v2.RAW_AE_WEIGHTED
    assert int(fields["agreement_gate_applied"]) == 1
    assert int(fields["agreement_gate_passed"]) == 1
    assert fields["agreement_reference_best_method"] == v2.RANK_MARGIN_UNWEIGHTED


def test_pairwise_ae_combined_v31_raw_selected_reports_best_mean_min_peer_agreement() -> None:
    payload = _payload()
    raw = np.asarray([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    half_peer = np.asarray([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
    bad_peer = np.asarray([[3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]])
    metadata = [{"patient_id": f"p{i % 2}"} for i in range(len(payload["sample_domains"]))]
    _deployed, fields = v2._target_batch_agreement_policy(
        source_inner_selected_method=v2.RAW_AE_WEIGHTED,
        method_predictions={
            v2.RAW_AE_WEIGHTED: raw,
            v2.RANK_MARGIN_UNWEIGHTED: half_peer,
            v2.RANK_MARGIN_WEIGHTED: bad_peer,
        },
        candidate_domains=payload["fold"].candidate_expert_domains,
        test_idx=payload["test_idx"],
        sample_metadata=metadata,
        pairwise_cfg=_pairwise_cfg(v31=True, target_batch_overrides={"agreement_threshold": 0.0}),
    )
    assert "selected_vs_reference_best_agreement" in fields
    assert "selected_vs_reference_mean_agreement" in fields
    assert "selected_vs_reference_min_agreement" in fields
    assert fields["raw_peer_agreement_with_rank_margin_unweighted"] > fields["raw_peer_agreement_with_rank_margin_weighted"]


def test_pairwise_ae_combined_v31_rank_margin_gate_preserves_v3_raw_reference_behavior() -> None:
    payload = _payload()
    n = int(payload["test_idx"].shape[0])
    pred = np.tile(np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64), (n, 1))
    metadata = [{"patient_id": f"p{i % 2}"} for i in range(len(payload["sample_domains"]))]
    deployed, fields = v2._target_batch_agreement_policy(
        source_inner_selected_method=v2.RANK_MARGIN_WEIGHTED,
        method_predictions={v2.RANK_MARGIN_WEIGHTED: pred, v2.RAW_AE_WEIGHTED: pred},
        candidate_domains=payload["fold"].candidate_expert_domains,
        test_idx=payload["test_idx"],
        sample_metadata=metadata,
        pairwise_cfg=_pairwise_cfg(v31=True),
    )
    assert deployed == v2.RANK_MARGIN_WEIGHTED
    assert fields["agreement_reference_best_method"] == v2.RAW_AE_WEIGHTED
    assert float(fields["selected_vs_raw_agreement_rate_query_weighted"]) == 1.0


def test_pairwise_ae_combined_v31_missing_peer_predictions_fallback_to_baseline() -> None:
    payload = _payload()
    n = int(payload["test_idx"].shape[0])
    pred = np.tile(np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64), (n, 1))
    deployed, fields = v2._target_batch_agreement_policy(
        source_inner_selected_method=v2.RAW_AE_WEIGHTED,
        method_predictions={v2.RAW_AE_WEIGHTED: pred},
        candidate_domains=payload["fold"].candidate_expert_domains,
        test_idx=payload["test_idx"],
        sample_metadata=[{"patient_id": f"p{i % 2}"} for i in range(len(payload["sample_domains"]))],
        pairwise_cfg=_pairwise_cfg(v31=True),
    )
    assert deployed == v2.BASELINE_METHOD
    assert fields["agreement_gate_reason"] == "missing_selected_or_reference_predictions"


def test_pairwise_ae_combined_v31_all_nonbaseline_deployments_have_gate_decision(monkeypatch) -> None:
    payload = _payload()
    metadata = [{"patient_id": f"p{i % 2}"} for i in range(len(payload["sample_domains"]))]

    def fake_selection(**_kwargs):
        return v2.RAW_AE_WEIGHTED, []

    def fake_train_predict(**kwargs):
        n_eval = int(kwargs["eval_idx"].shape[0])
        n_candidates = len(kwargs["eval_candidate_domains"])
        pred = np.tile(np.arange(n_candidates, dtype=np.float64), (n_eval, 1))
        return pred, np.ones(1), [], [], []

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
        pairwise_cfg=_pairwise_cfg(v31=True),
        seed=101,
        embedding_feature_dim=3,
        expert_feature_dim=5,
        tie_policy="stable_expert_index",
        ae_zscore_matrix=payload["ae_z"],
        sample_metadata=metadata,
    )
    deployed_nonbaseline = [row for row in out.decision_rows if row["deployed_method"] != v2.BASELINE_METHOD]
    assert deployed_nonbaseline
    assert all(int(row["agreement_gate_applied"]) == 1 for row in deployed_nonbaseline)


def test_pairwise_ae_combined_v3_threshold_sweep_rows_are_diagnostic_only(tmp_path: Path) -> None:
    decision_rows = [
        {
            "primary_method": v2.TARGET_BATCH_AGREEMENT_PRIMARY_METHOD,
            "selected_method": v2.RANK_MARGIN_WEIGHTED,
            "source_inner_selected_method": v2.RANK_MARGIN_WEIGHTED,
            "seed": 101,
            "outer_heldout_center": 0,
            "delta_gap_vs_baseline_pp": 0.3,
            "top1_oracle_hit": 1,
            "baseline_top1_oracle_hit": 1,
            "agreement_gate_applied": 1,
            "agreement_gate_passed": 1,
            "selected_vs_raw_agreement_rate_query_weighted": 0.75,
            "selected_vs_raw_agreement_rate_group_macro": 0.75,
        }
    ]
    v2.write_pairwise_ae_combined_v2_artifacts(
        reports_dir=tmp_path,
        training_rows=[],
        feature_rows=[],
        inner_selection_rows=[],
        pair_prediction_rows=[],
        decision_rows=decision_rows,
    )
    with (tmp_path / "pairwise_ae_combined_v3_threshold_sensitivity.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert all(int(row["posthoc_diagnostic_only"]) == 1 for row in rows)
    assert all(int(row["used_for_selection"]) == 0 for row in rows)


def test_pairwise_ae_combined_v31_no_ungated_nonbaseline_deployments(tmp_path: Path) -> None:
    rows = [
        {
            "primary_method": v2.TARGET_BATCH_AGREEMENT_V31_PRIMARY_METHOD,
            "selected_method": v2.RAW_AE_WEIGHTED,
            "deployed_method": v2.RAW_AE_WEIGHTED,
            "seed": 101,
            "outer_heldout_center": 0,
            "delta_gap_vs_baseline_pp": 0.2,
            "agreement_gate_applied": 1,
            "agreement_gate_passed": 1,
            "top1_oracle_hit": 1,
            "baseline_top1_oracle_hit": 1,
        }
    ]
    v2.write_pairwise_ae_combined_v2_artifacts(
        reports_dir=tmp_path,
        training_rows=[],
        feature_rows=[],
        inner_selection_rows=[],
        pair_prediction_rows=[],
        decision_rows=rows,
    )
    summary = json.loads((tmp_path / "pairwise_ae_combined_v31_decision_summary.json").read_text(encoding="utf-8"))
    assert summary["ungated_nonbaseline_count"] == 0
    assert summary["harmful_nonbaseline_bypass"] == 0


def test_pairwise_ae_combined_v31_gate_accounting_matches_decision_rows(tmp_path: Path) -> None:
    rows = [
        {
            "primary_method": v2.TARGET_BATCH_AGREEMENT_V31_PRIMARY_METHOD,
            "selected_method": v2.RAW_AE_WEIGHTED,
            "deployed_method": v2.RAW_AE_WEIGHTED,
            "seed": 101,
            "outer_heldout_center": 0,
            "sample_index": idx,
            "delta_gap_vs_baseline_pp": 0.2,
            "agreement_gate_applied": 1,
            "agreement_gate_passed": 1,
            "top1_oracle_hit": 1,
            "baseline_top1_oracle_hit": 1,
        }
        for idx in range(3)
    ]
    v2.write_pairwise_ae_combined_v2_artifacts(
        reports_dir=tmp_path,
        training_rows=[],
        feature_rows=[],
        inner_selection_rows=[],
        pair_prediction_rows=[],
        decision_rows=rows,
    )
    summary = json.loads((tmp_path / "pairwise_ae_combined_v31_decision_summary.json").read_text(encoding="utf-8"))
    assert summary["nonbaseline_deployment_count"] == 3
    assert summary["gated_nonbaseline_count"] == 3
    assert summary["ungated_nonbaseline_count"] == 0


def test_pairwise_ae_combined_v31_false_allow_counts_gate_applied_only(tmp_path: Path) -> None:
    rows = [
        {
            "primary_method": v2.TARGET_BATCH_AGREEMENT_PRIMARY_METHOD,
            "selected_method": v2.RAW_AE_WEIGHTED,
            "deployed_method": v2.RAW_AE_WEIGHTED,
            "seed": 101,
            "outer_heldout_center": 0,
            "delta_gap_vs_baseline_pp": -0.1,
            "agreement_gate_applied": 0,
            "agreement_gate_passed": 1,
            "false_allow": 0,
            "harmful_nonbaseline_bypass": 1,
            "nonbaseline_bypass_delta_gap_vs_baseline": 0.1,
            "top1_oracle_hit": 1,
            "baseline_top1_oracle_hit": 1,
        }
    ]
    v2.write_pairwise_ae_combined_v2_artifacts(
        reports_dir=tmp_path,
        training_rows=[],
        feature_rows=[],
        inner_selection_rows=[],
        pair_prediction_rows=[],
        decision_rows=rows,
    )
    with (tmp_path / "pairwise_ae_combined_v3_agreement_policy.csv").open("r", encoding="utf-8", newline="") as f:
        policy = list(csv.DictReader(f))[0]
    assert int(policy["false_allow"]) == 0
    assert int(policy["harmful_nonbaseline_bypass"]) == 1


def test_pairwise_ae_combined_v31_summary_compares_against_v3(tmp_path: Path) -> None:
    rows = [
        {
            "primary_method": v2.TARGET_BATCH_AGREEMENT_V31_PRIMARY_METHOD,
            "selected_method": v2.BASELINE_METHOD,
            "deployed_method": v2.BASELINE_METHOD,
            "seed": 101,
            "outer_heldout_center": 0,
            "delta_gap_vs_baseline_pp": 0.0,
            "agreement_gate_applied": 1,
            "agreement_gate_passed": 0,
            "v31_additional_blocks_over_v3": 1,
            "v31_additional_harm_prevented_over_v3": 1,
            "delta_gap_v31_vs_v3": 0.5,
            "delta_top1_v31_vs_v3": 0.0,
            "top1_oracle_hit": 1,
            "baseline_top1_oracle_hit": 1,
        }
    ]
    v2.write_pairwise_ae_combined_v2_artifacts(
        reports_dir=tmp_path,
        training_rows=[],
        feature_rows=[],
        inner_selection_rows=[],
        pair_prediction_rows=[],
        decision_rows=rows,
    )
    summary = json.loads((tmp_path / "pairwise_ae_combined_v31_decision_summary.json").read_text(encoding="utf-8"))
    assert summary["delta_gap_v31_vs_v3"] == 0.5
    assert summary["v31_additional_blocks_over_v3"] == 1
    assert summary["v31_additional_harm_prevented_over_v3"] == 1


def test_pairwise_ae_combined_v3_decision_table_builder(tmp_path: Path) -> None:
    builder = _load_decision_builder()
    reports = tmp_path / "outputs" / "camelyon17" / "learned_utility_pairwise_ae_combined_v3" / "run_seed101" / "reports"
    reports.mkdir(parents=True)
    result_path = reports / "learned_utility_results.json"
    result_path.write_text("{}", encoding="utf-8")
    v2._write_csv(
        reports / "pairwise_ae_combined_v3_decision_table.csv",
        [
            {
                "seed": 101,
                "outer_heldout_center": 0,
                "primary_method": v2.TARGET_BATCH_AGREEMENT_PRIMARY_METHOD,
                "selected_method": v2.RANK_MARGIN_UNWEIGHTED,
                "delta_gap_vs_baseline_pp": 0.3,
                "agreement_gate_applied": 1,
                "agreement_gate_passed": 1,
            }
        ],
    )
    v2._write_csv(
        reports / "learned_utility_domain_breakdown.csv",
        [
            {"method": v2.TARGET_BATCH_AGREEMENT_PRIMARY_METHOD, "query_domain": 0, "mean_oracle_gap_pct": 1.0, "top1_oracle_hit": 0.8, "spearman": 0.7},
            {"method": v2.BASELINE_METHOD, "query_domain": 0, "mean_oracle_gap_pct": 1.3, "top1_oracle_hit": 0.8, "spearman": 0.7},
            {"method": "ae_argmin_zscore", "query_domain": 0, "mean_oracle_gap_pct": 2.0, "top1_oracle_hit": 0.7, "spearman": 0.6},
            {"method": "metadata_routing", "query_domain": 0, "mean_oracle_gap_pct": 2.5, "top1_oracle_hit": 0.6, "spearman": 0.5},
        ],
    )
    decisions, seed_domain_rows, any_special = builder._aggregate_decisions([result_path])
    summary = builder._verdict(seed_domain_rows)
    summary.update(builder._v3_summary(decisions))
    assert any_special is True
    assert decisions[0]["decision_artifact"].endswith("pairwise_ae_combined_v3_decision_table.csv")
    assert summary["gate_activation_count"] == 1


def test_pairwise_ae_combined_v31_decision_table_builder(tmp_path: Path) -> None:
    builder = _load_decision_builder()
    reports = tmp_path / "outputs" / "camelyon17" / "learned_utility_pairwise_ae_combined_v31" / "run_seed101" / "reports"
    reports.mkdir(parents=True)
    result_path = reports / "learned_utility_results.json"
    result_path.write_text("{}", encoding="utf-8")
    v2._write_csv(
        reports / "pairwise_ae_combined_v31_decision_table.csv",
        [
            {
                "seed": 101,
                "outer_heldout_center": 0,
                "primary_method": v2.TARGET_BATCH_AGREEMENT_V31_PRIMARY_METHOD,
                "selected_method": v2.RAW_AE_WEIGHTED,
                "deployed_method": v2.RAW_AE_WEIGHTED,
                "delta_gap_vs_baseline_pp": 0.3,
                "agreement_gate_applied": 1,
                "agreement_gate_passed": 1,
            }
        ],
    )
    v2._write_csv(
        reports / "learned_utility_domain_breakdown.csv",
        [
            {"method": v2.TARGET_BATCH_AGREEMENT_V31_PRIMARY_METHOD, "query_domain": 0, "mean_oracle_gap_pct": 1.0, "top1_oracle_hit": 0.8, "spearman": 0.7},
            {"method": v2.BASELINE_METHOD, "query_domain": 0, "mean_oracle_gap_pct": 1.3, "top1_oracle_hit": 0.8, "spearman": 0.7},
            {"method": "ae_argmin_zscore", "query_domain": 0, "mean_oracle_gap_pct": 2.0, "top1_oracle_hit": 0.7, "spearman": 0.6},
            {"method": "metadata_routing", "query_domain": 0, "mean_oracle_gap_pct": 2.5, "top1_oracle_hit": 0.6, "spearman": 0.5},
        ],
    )
    decisions, seed_domain_rows, any_special = builder._aggregate_decisions([result_path])
    summary = builder._verdict(seed_domain_rows)
    summary.update(builder._v3_summary(decisions))
    assert any_special is True
    assert decisions[0]["decision_artifact"].endswith("pairwise_ae_combined_v31_decision_table.csv")
    assert summary["nonbaseline_deployment_count"] == 1


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
