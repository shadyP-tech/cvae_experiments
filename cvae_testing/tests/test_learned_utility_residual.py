from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluators import learned_utility as lu
from src.eval.evaluators.learned_utility_protocol import FoldCandidateSet
from src.eval.evaluators.learned_utility_residual import (
    _build_residual_training_rows,
    _feature_context,
    _selected_from_residual,
)


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
