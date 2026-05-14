from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluators import learned_utility as lu
from src.eval.evaluators import support_free_ae as sfa
from src.eval.evaluators.learned_utility_config import ResidualRoutingConfig
from src.eval.evaluators.learned_utility_protocol import FoldCandidateSet
from src.eval.evaluators.learned_utility_residual import (
    _build_residual_training_rows,
    _feature_context,
    _safe_v2_adoption_feature_sets,
    _safe_v2_validation_report,
    _selected_from_residual,
)
from src.eval.evaluators.support_free_ae import AutoencoderScoreMatrices
from src.train.train_autoencoders import build_support_free_ae_overlap_audit


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fake_scored_payload():
    expert_domains = [40, 100, 200]
    sample_domains = np.asarray([40, 40, 100, 100, 200, 200], dtype=np.int64)
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
            [2.0, 0.0],
            [2.1, 0.0],
        ],
        dtype=np.float64,
    )
    nelbo = np.asarray(
        [
            [0.1, 0.5, 0.8],
            [0.1, 0.4, 0.9],
            [0.6, 0.1, 0.7],
            [0.8, 0.1, 0.5],
            [0.9, 0.4, 0.1],
            [0.8, 0.3, 0.1],
        ],
        dtype=np.float64,
    )
    metadata = [
        {"magnification": int(domain), "sample_id": f"s{i}"}
        for i, domain in enumerate(sample_domains.tolist())
    ]
    return embeddings, sample_domains, nelbo, expert_domains, metadata


def _residual_cfg() -> dict:
    return {
        "predictors": ["linear_regressor"],
        "pair_features": {"include_metadata_features": True},
        "scoring": {"pair_batch_size": 2},
        "hybrid_scoring": {
            "enabled": False,
            "tie_policy": "stable_expert_index",
        },
        "residual_routing": {
            "enabled": True,
            "models": ["ridge"],
            "thresholds": [0, 0.01, "inf"],
            "feature_sets": ["minimal", "latent"],
            "selection_metric": "validation_safe_gap_then_top1",
            "unconstrained_reference_method": "linear_regressor",
            "ridge_l2": 1.0e-4,
        },
        "compatibility_research": {
            "floors": {"random_rank_floor": False, "random_score_floor": False},
            "permutation_tests": {
                "expert_label_permutation": False,
                "metadata_permutation": False,
                "repeats": 1,
            },
            "diagnostics": {"save_distribution_plots": False},
            "gate": {"decision_policy_version": "sign_ci_v2", "uplift_reference_method": "metadata_routing"},
        },
    }


def _safe_v2_cfg() -> dict:
    cfg = _residual_cfg()
    cfg["residual_routing"] = {
        **cfg["residual_routing"],
        "residual_policy_version": "metadata_residual_safe_override_v2",
        "feature_sets": ["minimal", "latent", "calibrated"],
        "adoption_feature_sets": ["minimal", "latent"],
        "diagnostic_feature_sets": ["calibrated"],
        "allow_calibrated_adoption": False,
        "harmful_override_max": 0.05,
        "gap_regression_max": 2.0,
        "catastrophic_top1_floor": -0.05,
    }
    return cfg


def _support_free_ae_cfg(*, thresholds=None) -> dict:
    cfg = _residual_cfg()
    cfg["autoencoder_proxy"] = {
        "enabled": True,
        "hidden_dim": 8,
        "latent_dim": 2,
        "learning_rate": 1.0e-3,
        "epochs": 2,
        "patience": 1,
        "batch_size": 8,
        "score_normalization": "source_val_zscore",
        "score_normalization_eps": 1.0e-6,
        "margin_threshold": 0.0,
        "run_diagnostics": True,
    }
    cfg["residual_routing"] = {
        **cfg["residual_routing"],
        "residual_policy_version": "metadata_ae_residual_safe_override_v1",
        "thresholds": list(thresholds if thresholds is not None else ["inf"]),
        "feature_sets": ["ae"],
        "adoption_feature_sets": ["ae"],
        "diagnostic_feature_sets": [],
        "allow_calibrated_adoption": False,
        "harmful_override_max": 0.50,
        "gap_regression_max": 1.0,
        "catastrophic_top1_floor": -0.02,
        "unconstrained_reference_method": "linear_regressor",
    }
    return cfg


def _fake_ae_scores() -> AutoencoderScoreMatrices:
    z = np.asarray(
        [
            [0.0, 0.8, 1.4],
            [0.0, 0.7, 1.5],
            [1.2, 0.0, 0.6],
            [1.3, 0.0, 0.4],
            [1.5, 0.5, 0.0],
            [1.4, 0.4, 0.0],
        ],
        dtype=np.float64,
    )
    return AutoencoderScoreMatrices(
        raw_mse_matrix=z.copy(),
        zscore_matrix=z,
        quality_rows=[
            {
                "source_domain": d,
                "source_val_reconstruction_mse_by_domain": 0.0,
                "source_val_reconstruction_std_by_domain": 1.0,
                "ae_training_converged": 1,
                "ae_best_epoch": 0,
                "ae_val_loss": 0.0,
            }
            for d in [40, 100, 200]
        ],
        provenance_rows=[],
        overlap_rows=[
            {
                "ae_train_query_overlap_count": 0,
                "ae_val_query_overlap_count": 0,
                "ae_train_cache_hash": "train",
                "ae_val_cache_hash": "val",
                "routing_query_cache_hash": "test",
                "routing_eval_cache_hash": "test",
            }
        ],
        provenance={
            "overlap_audit": {
                "ae_train_query_overlap_count": 0,
                "ae_val_query_overlap_count": 0,
                "ae_train_cache_hash": "train",
                "ae_val_cache_hash": "val",
                "routing_query_cache_hash": "test",
                "routing_eval_cache_hash": "test",
            }
        },
    )


def _safe_v2_residual_config(**overrides) -> ResidualRoutingConfig:
    values = {
        "enabled": True,
        "residual_policy_version": "metadata_residual_safe_override_v2",
        "models": ("ridge",),
        "thresholds": (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, float("inf")),
        "feature_sets": ("minimal", "latent", "calibrated"),
        "adoption_feature_sets": ("minimal", "latent"),
        "diagnostic_feature_sets": ("calibrated",),
        "allow_calibrated_adoption": False,
        "harmful_override_max": 0.05,
        "gap_regression_max": 2.0,
        "catastrophic_top1_floor": -0.05,
        "selection_metric": "validation_safe_gap_then_top1",
        "unconstrained_reference_method": "linear_regressor",
        "ridge_l2": 1.0e-4,
    }
    values.update(overrides)
    return ResidualRoutingConfig(**values)


def _selection_row(
    *,
    method: str,
    query_domain: int,
    sample_index: int,
    selected_expert: int,
    oracle_expert: int,
    oracle_gap_pct: float,
    top1_hit: int,
) -> dict:
    return {
        "protocol_version": "learned_utility_loqdo_candidate_exclusion_v2",
        "method": method,
        "query_domain": query_domain,
        "fold_query_domain": query_domain,
        "candidate_experts": "100|200",
        "n_candidate_experts": 2,
        "target_expert_excluded": 1,
        "method_role": "learned" if method != "metadata_routing" else "baseline",
        "adoption_eligible": 1,
        "diagnostic_only": 0,
        "routing_uses_eval_nelbo": 0,
        "routing_uses_eval_domain_statistics": 0,
        "selected_expert": selected_expert,
        "candidate_oracle_expert": oracle_expert,
        "oracle_expert": oracle_expert,
        "sample_index": sample_index,
        "top1_oracle_hit": top1_hit,
        "selected_rank": 1.0 if top1_hit else 2.0,
        "oracle_gap": oracle_gap_pct,
        "oracle_gap_pct": oracle_gap_pct,
        "spearman": 1.0,
        "pairwise_auc": 1.0,
        "selected_nelbo": 1.0 + oracle_gap_pct,
        "candidate_oracle_nelbo": 1.0,
    }


def test_residual_target_uses_normalized_metadata_relative_utility() -> None:
    expert_domains = [40, 100, 200, 400]
    sample_domains = np.asarray([100, 100, 200, 200], dtype=np.int64)
    embeddings = np.asarray(
        [[1.0, 0.0], [1.1, 0.0], [2.0, 0.0], [2.1, 0.0]],
        dtype=np.float64,
    )
    true_nelbo = np.asarray(
        [
            [9.0, 1.0, 4.0, 2.0],
            [9.0, 1.0, 5.0, 10.0],
            [9.0, 7.0, 1.0, 3.0],
            [9.0, 8.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    metadata_similarity = np.asarray(
        [
            [0.0, 0.1, 1.0, 0.2],
            [0.0, 0.1, 1.0, 0.2],
            [0.0, 1.0, 0.1, 0.2],
            [0.0, 1.0, 0.1, 0.2],
        ],
        dtype=np.float64,
    )
    context = _feature_context(
        feature_set="minimal",
        embeddings=embeddings,
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        stats_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
    )

    x, y, q = _build_residual_training_rows(
        embeddings=embeddings,
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        metadata_similarity=metadata_similarity,
        outer_heldout_domain=40,
        train_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        context=context,
    )

    assert x.shape[0] == y.shape[0] == q.shape[0]
    assert np.isclose(y[0], 0.0)
    assert np.isclose(y[1], (4.0 - 2.0) / 4.0)
    assert np.isclose(y[2], 0.0)
    assert np.isclose(y[3], (5.0 - 10.0) / 5.0)


def test_tau_inf_fallback_selects_metadata_indices() -> None:
    raw_scores = np.asarray([[10.0, -1.0], [-5.0, 3.0]], dtype=np.float64)
    meta_idx = np.asarray([1, 0], dtype=np.int64)
    assert _selected_from_residual(raw_scores, meta_idx, tau=float("inf")).tolist() == [1, 0]
    assert _selected_from_residual(raw_scores, meta_idx, tau=0.0).tolist() == [0, 1]


def test_safe_v2_harmful_override_veto_rejects_seed43_like_pattern() -> None:
    baseline_rows = [
        _selection_row(
            method="metadata_routing",
            query_domain=40,
            sample_index=0,
            selected_expert=100,
            oracle_expert=100,
            oracle_gap_pct=10.0,
            top1_hit=1,
        )
    ]
    candidate_rows = [
        _selection_row(
            method="metadata_residual_thresholded_safe_v2",
            query_domain=40,
            sample_index=0,
            selected_expert=200,
            oracle_expert=100,
            oracle_gap_pct=5.0,
            top1_hit=0,
        )
    ]
    report = _safe_v2_validation_report(
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
        override_rows=[
            {
                "fold_query_domain": 40,
                "override_rate": 1.0,
                "utility_improving_override_rate": 0.053,
                "harmful_override_rate": 0.947,
            }
        ],
        method="metadata_residual_thresholded_safe_v2",
        residual_cfg=_safe_v2_residual_config(),
    )

    assert report["gap_pct_reduction"] > 0.0
    assert report["safety_pass"] is False
    assert report["max_harmful_override_rate"] == 0.947
    assert "harmful_override" in report["domains_failed_gate"]


def test_safe_v2_zero_override_gate_allows_exact_metadata_match() -> None:
    baseline_rows = [
        _selection_row(
            method="metadata_routing",
            query_domain=40,
            sample_index=0,
            selected_expert=100,
            oracle_expert=100,
            oracle_gap_pct=0.0,
            top1_hit=1,
        )
    ]
    candidate_rows = [
        _selection_row(
            method="metadata_residual_thresholded_safe_v2",
            query_domain=40,
            sample_index=0,
            selected_expert=100,
            oracle_expert=100,
            oracle_gap_pct=0.0,
            top1_hit=1,
        )
    ]
    report = _safe_v2_validation_report(
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
        override_rows=[
            {
                "fold_query_domain": 40,
                "override_rate": 0.0,
                "utility_improving_override_rate": 0.0,
                "harmful_override_rate": 0.0,
            }
        ],
        method="metadata_residual_thresholded_safe_v2",
        residual_cfg=_safe_v2_residual_config(),
    )

    assert report["safety_pass"] is True
    assert report["domains_failed_gate"] == ""


def test_safe_v2_calibrated_excluded_from_adoption_by_default() -> None:
    cfg = _safe_v2_residual_config(
        adoption_feature_sets=("minimal", "latent", "calibrated"),
        allow_calibrated_adoption=False,
    )

    assert _safe_v2_adoption_feature_sets(cfg) == ("minimal", "latent")


def test_residual_artifacts_and_single_inner_selected_adoption_candidate(tmp_path, monkeypatch) -> None:
    def fake_score(**kwargs):
        _ = kwargs
        return _fake_scored_payload()

    monkeypatch.setattr(lu, "_score_experts_batched", fake_score)
    results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={"expert_40": "unused", "expert_100": "unused", "expert_200": "unused"},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg=_residual_cfg(),
        reports_dir=tmp_path,
    )

    assert results["artifacts"]["residual_raw"] == "residual_routing_raw.csv"
    assert (tmp_path / "residual_routing_raw.csv").exists()
    assert (tmp_path / "residual_routing_domain_breakdown.csv").exists()
    assert (tmp_path / "residual_routing_override_diagnostics.csv").exists()
    assert (tmp_path / "residual_routing_policy_audit.md").exists()

    method_summary = _read_csv(tmp_path / "learned_utility_method_summary.csv")
    residual_variants = {
        row["method"]: row
        for row in method_summary
        if row["method"] in {"metadata_residual_thresholded", "metadata_residual_group_robust"}
    }
    assert set(residual_variants) == {"metadata_residual_thresholded", "metadata_residual_group_robust"}
    selected_variants = [
        row
        for row in residual_variants.values()
        if int(row["selected_by_inner_validation"]) == 1 and int(row["diagnostic_only"]) == 1
    ]
    assert len(selected_variants) == 1
    assert all(int(row["adoption_eligible"]) == 0 for row in residual_variants.values())

    selected_method = {row["method"]: row for row in method_summary}["metadata_residual_inner_selected"]
    assert int(selected_method["selected_by_inner_validation"]) == 1
    assert int(selected_method["adoption_eligible"]) == 1
    assert int(selected_method["diagnostic_only"]) == 0
    assert selected_method["decision_policy_version"] == "sign_ci_v2"
    assert selected_method["residual_policy_version"] == "metadata_residual_v1"

    raw_rows = _read_csv(tmp_path / "residual_routing_raw.csv")
    assert raw_rows
    assert all(row["residual_target_scale"] == "delta_u_pct" for row in raw_rows)
    assert all(row["spearman_score_source"] == "raw_residual_pre_threshold" for row in raw_rows)


def test_safe_v2_writes_isolated_artifacts_and_required_policy_fields(tmp_path, monkeypatch) -> None:
    def fake_score(**kwargs):
        _ = kwargs
        return _fake_scored_payload()

    monkeypatch.setattr(lu, "_score_experts_batched", fake_score)
    results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={"expert_40": "unused", "expert_100": "unused", "expert_200": "unused"},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg=_safe_v2_cfg(),
        reports_dir=tmp_path,
    )

    assert results["artifacts"]["residual_raw"] == "residual_safe_v2_raw.csv"
    assert results["artifacts"]["residual_decision_table"] == "residual_safe_v2_decision_table.csv"
    assert results["artifacts"]["residual_selection_policy_audit"] == "residual_safe_v2_selection_policy_audit.md"
    assert (tmp_path / "residual_safe_v2_override_diagnostics.csv").exists()

    raw_rows = _read_csv(tmp_path / "residual_safe_v2_raw.csv")
    assert raw_rows
    required = {
        "decision_policy_version",
        "residual_policy_version",
        "threshold_selection_policy",
        "feature_set",
        "selected_tau",
        "adoption_eligible",
        "diagnostic_only",
        "selected_by_inner_validation",
        "harmful_override_max",
        "allow_calibrated_adoption",
        "fallback_used",
    }
    assert required.issubset(set(raw_rows[0]))
    selected_rows = [
        row
        for row in _read_csv(tmp_path / "residual_safe_v2_policy_audit.csv")
        if int(row["selected_by_inner_validation"]) == 1
    ]
    assert selected_rows
    assert all(row["feature_set"] != "calibrated" for row in selected_rows)


def test_source_inner_self_expert_exclusion() -> None:
    expert_domains = [40, 100, 200, 400]
    outer_heldout = 40
    for pseudo_query_domain in [100, 200, 400]:
        fold = FoldCandidateSet.for_heldout_domain(
            heldout_domain=outer_heldout,
            expert_domains=expert_domains,
            excluded_domains=[pseudo_query_domain],
        )
        assert pseudo_query_domain not in fold.candidate_expert_domains
        assert outer_heldout not in fold.candidate_expert_domains


def test_ae_zscore_uses_source_val_stats_only(monkeypatch) -> None:
    def fake_score(**kwargs):
        _ = kwargs
        return np.asarray([1.0, 3.0], dtype=np.float64)

    monkeypatch.setattr(sfa, "_score_autoencoder", fake_score)
    scores = sfa.build_autoencoder_score_matrices(
        embeddings=np.zeros((2, 2), dtype=np.float64),
        expert_domains=[40],
        autoencoder_artifacts={
            "checkpoints": {"40": "unused.pt"},
            "provenance": {
                "domains": {
                    "40": {
                        "checkpoint": "unused.pt",
                        "source_val_reconstruction_mse": 1.0,
                        "source_val_reconstruction_std": 2.0,
                        "input_dim": 2,
                        "autoencoder_config": {"hidden_dim": 4, "latent_dim": 2},
                    }
                }
            },
        },
        cfg=lu._parse_learned_utility_config(_support_free_ae_cfg()).autoencoder,
    )

    assert np.allclose(scores.zscore_matrix[:, 0], [0.0, 1.0])
    assert scores.quality_rows[0]["source_val_reconstruction_mse_by_domain"] == 1.0
    assert scores.quality_rows[0]["source_val_reconstruction_std_by_domain"] == 2.0


def test_ae_train_val_query_overlap_audit_zero(tmp_path: Path) -> None:
    def write_cache(path: Path, prefix: str) -> None:
        torch.save(
            {
                "embeddings": torch.zeros((2, 2)),
                "metadata": [
                    {"magnification": 40, "sample_id": f"{prefix}_0"},
                    {"magnification": 100, "sample_id": f"{prefix}_1"},
                ],
            },
            path,
        )

    train_cache = tmp_path / "train.pt"
    val_cache = tmp_path / "val.pt"
    test_cache = tmp_path / "test.pt"
    write_cache(train_cache, "train")
    write_cache(val_cache, "val")
    write_cache(test_cache, "test")

    audit = build_support_free_ae_overlap_audit(
        train_cache=train_cache,
        val_cache=val_cache,
        routing_cache=test_cache,
    )
    assert audit["ae_train_query_overlap_count"] == 0
    assert audit["ae_val_query_overlap_count"] == 0
    assert audit["ae_train_cache_hash"]
    assert audit["routing_eval_cache_hash"] == audit["routing_query_cache_hash"]


def test_target_ae_excluded_from_loqdo_candidates() -> None:
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=40, expert_domains=[40, 100, 200])
    outputs = sfa.run_autoencoder_proxy_methods_for_fold(
        sample_domains=np.asarray([40, 40], dtype=np.int64),
        expert_domains=[40, 100, 200],
        test_idx=np.asarray([0, 1], dtype=np.int64),
        fold=fold,
        true_eval=np.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64),
        global_eval=np.asarray([[0.5, 2.0, 1.0], [0.5, 1.0, 2.0]], dtype=np.float64),
        metadata_similarity_eval=np.asarray([[0.8, 0.2], [0.7, 0.3]], dtype=np.float64),
        ae_zscore_matrix=np.asarray([[9.0, 0.1, 0.2], [9.0, 0.2, 0.1]], dtype=np.float64),
        ae_raw_mse_matrix=np.asarray([[9.0, 0.1, 0.2], [9.0, 0.2, 0.1]], dtype=np.float64),
        margin_threshold=0.0,
        tie_policy="stable_expert_index",
    )

    assert outputs.sample_rows
    assert all(int(row["selected_expert"]) != 40 for row in outputs.sample_rows)
    assert all(int(row["expert_domain"]) != 40 for row in outputs.proxy_diag_rows)


def test_ae_argmin_zscore_diagnostic_rows() -> None:
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=40, expert_domains=[40, 100, 200])
    outputs = sfa.run_autoencoder_proxy_methods_for_fold(
        sample_domains=np.asarray([40], dtype=np.int64),
        expert_domains=[40, 100, 200],
        test_idx=np.asarray([0], dtype=np.int64),
        fold=fold,
        true_eval=np.asarray([[2.0, 1.0]], dtype=np.float64),
        global_eval=np.asarray([[0.5, 2.0, 1.0]], dtype=np.float64),
        metadata_similarity_eval=np.asarray([[0.8, 0.2]], dtype=np.float64),
        ae_zscore_matrix=np.asarray([[9.0, 0.2, 0.1]], dtype=np.float64),
        ae_raw_mse_matrix=np.asarray([[9.0, 0.2, 0.1]], dtype=np.float64),
        margin_threshold=0.0,
        tie_policy="stable_expert_index",
    )

    methods = {row["method"] for row in outputs.sample_rows}
    assert {"ae_argmin_zscore", "ae_argmin_margin_gated"}.issubset(methods)
    ae_rows = [row for row in outputs.sample_rows if row["method"] == "ae_argmin_zscore"]
    assert ae_rows[0]["method_role"] == "diagnostic"
    assert int(ae_rows[0]["diagnostic_only"]) == 1


def test_safe_override_falls_back_to_metadata_with_tau_inf(tmp_path, monkeypatch) -> None:
    def fake_score(**kwargs):
        _ = kwargs
        return _fake_scored_payload()

    def fake_ae_scores(**kwargs):
        _ = kwargs
        return _fake_ae_scores()

    monkeypatch.setattr(lu, "_score_experts_batched", fake_score)
    monkeypatch.setattr(lu, "build_autoencoder_score_matrices", fake_ae_scores)
    results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={"expert_40": "unused", "expert_100": "unused", "expert_200": "unused"},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg=_support_free_ae_cfg(thresholds=["inf"]),
        reports_dir=tmp_path,
        autoencoder_artifacts={"dummy": True},
    )

    assert results["artifacts"]["residual_raw"] == "residual_safe_v2_raw.csv"
    override_rows = _read_csv(tmp_path / "residual_safe_v2_override_diagnostics.csv")
    ae_rows = [r for r in override_rows if r["method"] == "metadata_ae_residual_safe_override_v1"]
    assert ae_rows
    assert all(row["selected_tau"] == "inf" for row in ae_rows)
    assert all(float(row["override_rate"]) == 0.0 for row in ae_rows)
    assert all(float(row["safe_fallback_rate"]) == 1.0 for row in ae_rows)


def test_tau_inf_outputs_identical_to_metadata_routing(tmp_path, monkeypatch) -> None:
    def fake_score(**kwargs):
        _ = kwargs
        return _fake_scored_payload()

    def fake_ae_scores(**kwargs):
        _ = kwargs
        return _fake_ae_scores()

    monkeypatch.setattr(lu, "_score_experts_batched", fake_score)
    monkeypatch.setattr(lu, "build_autoencoder_score_matrices", fake_ae_scores)
    lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={"expert_40": "unused", "expert_100": "unused", "expert_200": "unused"},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg=_support_free_ae_cfg(thresholds=["inf"]),
        reports_dir=tmp_path,
        autoencoder_artifacts={"dummy": True},
    )

    rows = _read_csv(tmp_path / "learned_utility_sample_selections.csv")
    metadata = {
        int(row["sample_index"]): row
        for row in rows
        if row["method"] == "metadata_routing"
    }
    ae_safe = [
        row for row in rows if row["method"] == "metadata_ae_residual_safe_override_v1"
    ]
    assert ae_safe
    for row in ae_safe:
        base = metadata[int(row["sample_index"])]
        assert row["selected_expert"] == base["selected_expert"]
        assert row["selected_nelbo"] == base["selected_nelbo"]
        assert row["oracle_gap"] == base["oracle_gap"]
