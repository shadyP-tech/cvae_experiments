from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluators.domain_query_oracle_gap import (
    aggregate_domain_query_oracle_gap_rows,
    evaluate_domain_query_oracle_gap_from_arrays,
)
from src.eval.evaluators.support_set_calibration import SupportSetRunMeta


def _metadata(n_target: int = 4) -> list[dict]:
    rows: list[dict] = []
    for i in range(n_target):
        rows.append(
            {
                "sample_id": f"target-{i}",
                "image_path": f"/tmp/target-{i}.png",
                "magnification": 40,
                "label": i % 2,
            }
        )
    for domain in [100, 200, 400]:
        rows.append(
            {
                "sample_id": f"{domain}-0",
                "image_path": f"/tmp/{domain}-0.png",
                "magnification": domain,
                "label": 0,
            }
        )
    return rows


def _embeddings(n: int) -> np.ndarray:
    values = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    return values / 10.0


def _run_meta() -> SupportSetRunMeta:
    return SupportSetRunMeta(
        dataset_name="breakhis",
        seed=42,
        backbone_type="dinov2_vitb14",
        run_id="run1",
        variant="B",
    )


def test_fixed_domain_and_per_query_oracle_gap_metrics() -> None:
    metadata = _metadata()
    embeddings = _embeddings(len(metadata))
    expert_domains = [40, 100, 200, 400]
    nelbo = np.full((len(metadata), len(expert_domains)), 10.0, dtype=np.float64)

    target_scores = [
        [0.0, 1.0, 5.0, 6.0],
        [0.0, 1.0, 2.0, 8.0],
        [0.0, 9.0, 1.0, 8.0],
        [0.0, 9.0, 1.0, 8.0],
    ]
    for idx, row in enumerate(target_scores):
        nelbo[idx, :] = row

    fold_rows, sample_rows = evaluate_domain_query_oracle_gap_from_arrays(
        embeddings=embeddings,
        metadata=metadata,
        nelbo_matrix=nelbo,
        expert_domains=expert_domains,
        run_meta=_run_meta(),
        bootstrap_reps=25,
        bootstrap_seed=7,
    )

    row = [r for r in fold_rows if int(r["target_domain"]) == 40][0]
    assert row["target_expert_excluded"] == 1
    assert row["candidate_experts"] == "100|200|400"
    assert row["fixed_domain_oracle_expert"] == 200
    assert row["worst_fixed_expert"] == 400
    assert row["fixed_to_query_gap_invariant_ok"] == 1
    assert row["fixed_domain_oracle_nelbo"] >= row["per_query_oracle_nelbo"] - 1e-12
    assert np.isclose(float(row["fixed_domain_oracle_nelbo"]), 2.25)
    assert np.isclose(float(row["per_query_oracle_nelbo"]), 1.0)
    assert np.isclose(float(row["worst_fixed_expert_nelbo"]), 7.5)
    assert np.isclose(float(row["fixed_to_query_oracle_gap"]), 1.25)
    assert np.isclose(float(row["normalized_fixed_to_query_oracle_gap"]), 1.25 / 6.5)

    counts = json.loads(str(row["per_query_selected_expert_counts_json"]))
    assert counts == {"100": 2, "200": 2, "400": 0}
    assert np.isclose(float(row["per_query_expert_entropy"]), np.log(2.0))
    assert np.isclose(float(row["per_query_expert_entropy_normalized"]), np.log(2.0) / np.log(3.0))
    assert row["per_query_oracle_modal_share"] == 0.5
    assert row["per_query_oracle_switch_rate"] == 0.5
    assert np.isclose(float(row["per_query_oracle_margin_mean"]), 4.75)
    assert np.isclose(float(row["per_query_oracle_margin_median"]), 5.5)
    assert row["low_margin_share"] == 0.0
    assert np.isclose(float(row["fixed_domain_oracle_sample_rank_mean"]), 1.5)
    assert np.isclose(float(row["fixed_domain_oracle_sample_rank_std"]), 0.5)
    assert "fixed_to_query_oracle_gap_ci_low" in row
    assert "fixed_to_query_oracle_gap_ci_high" in row

    target_sample_rows = [r for r in sample_rows if int(r["target_domain"]) == 40]
    assert len(target_sample_rows) == 4
    assert [int(r["fixed_domain_oracle_sample_rank"]) for r in target_sample_rows] == [2, 2, 1, 1]
    assert [int(r["per_query_oracle_expert"]) for r in target_sample_rows] == [100, 100, 200, 200]
    assert np.isclose(float(target_sample_rows[0]["per_query_oracle_margin"]), 4.0)
    assert np.isclose(float(target_sample_rows[0]["fixed_to_query_sample_gap"]), 4.0)
    assert json.loads(str(target_sample_rows[0]["nelbo_by_expert_json"])) == {
        "100": 1.0,
        "200": 5.0,
        "400": 6.0,
    }


def test_identical_candidate_nelbo_degenerate_tie_behavior() -> None:
    metadata = _metadata(n_target=3)
    embeddings = _embeddings(len(metadata))
    expert_domains = [40, 100, 200, 400]
    nelbo = np.full((len(metadata), len(expert_domains)), 7.0, dtype=np.float64)
    nelbo[:3, expert_domains.index(40)] = 0.0

    fold_rows, sample_rows = evaluate_domain_query_oracle_gap_from_arrays(
        embeddings=embeddings,
        metadata=metadata,
        nelbo_matrix=nelbo,
        expert_domains=expert_domains,
        run_meta=_run_meta(),
        bootstrap_reps=0,
    )

    row = [r for r in fold_rows if int(r["target_domain"]) == 40][0]
    assert row["fixed_domain_oracle_expert"] == 100
    assert row["fixed_domain_oracle_nelbo"] == 7.0
    assert row["per_query_oracle_nelbo"] == 7.0
    assert row["fixed_to_query_oracle_gap"] == 0.0
    assert row["normalized_fixed_to_query_oracle_gap"] == 0.0
    assert row["per_query_oracle_margin_mean"] == 0.0
    assert row["per_query_oracle_margin_median"] == 0.0
    assert row["low_margin_share"] == 1.0
    assert row["per_query_oracle_switch_rate"] == 0.0
    assert row["per_query_expert_entropy"] == 0.0
    assert json.loads(str(row["per_query_selected_expert_counts_json"])) == {"100": 3, "200": 0, "400": 0}

    target_sample_rows = [r for r in sample_rows if int(r["target_domain"]) == 40]
    assert [int(r["per_query_oracle_expert"]) for r in target_sample_rows] == [100, 100, 100]
    assert [int(r["fixed_domain_oracle_sample_rank"]) for r in target_sample_rows] == [1, 1, 1]


def test_aggregate_rows_adds_all_backbone_and_interpretation() -> None:
    metadata = _metadata()
    embeddings = _embeddings(len(metadata))
    expert_domains = [40, 100, 200, 400]
    nelbo = np.full((len(metadata), len(expert_domains)), 10.0, dtype=np.float64)
    nelbo[:4, :] = np.asarray(
        [
            [0.0, 1.0, 5.0, 6.0],
            [0.0, 1.0, 2.0, 8.0],
            [0.0, 9.0, 1.0, 8.0],
            [0.0, 9.0, 1.0, 8.0],
        ],
        dtype=np.float64,
    )
    fold_rows, _ = evaluate_domain_query_oracle_gap_from_arrays(
        embeddings=embeddings,
        metadata=metadata,
        nelbo_matrix=nelbo,
        expert_domains=expert_domains,
        run_meta=_run_meta(),
        bootstrap_reps=0,
    )

    stats = aggregate_domain_query_oracle_gap_rows(fold_rows, bootstrap_reps=10, bootstrap_seed=11)
    keys = {(r["dataset_name"], r["backbone_type"], r["variant"]) for r in stats}
    assert ("breakhis", "dinov2_vitb14", "B") in keys
    assert ("breakhis", "all", "B") in keys
    all_row = [r for r in stats if r["backbone_type"] == "all"][0]
    assert all_row["thresholds_are_descriptive_heuristics"] == 1
    assert "normalized_fixed_to_query_oracle_gap_ci_low" in all_row
    assert "interpretation_pattern" in all_row
