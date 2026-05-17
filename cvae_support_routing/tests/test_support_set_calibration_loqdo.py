from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


SUPPORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUPPORT_ROOT.parent
CVAE_TESTING_ROOT = REPO_ROOT / "cvae_testing"
if str(CVAE_TESTING_ROOT) not in sys.path:
    sys.path.insert(0, str(CVAE_TESTING_ROOT))

from src.eval.evaluators.support_set_calibration import (
    SupportSetRunMeta,
    calibration_mae_from_vectors,
    evaluate_support_set_calibration_from_arrays,
    make_support_eval_split,
    normalized_oracle_gap,
)


def _metadata() -> list[dict]:
    rows: list[dict] = []
    for i in range(8):
        rows.append(
            {
                "sample_id": f"target-{i}",
                "magnification": 40,
                "label": i % 2,
            }
        )
    for domain in [100, 200, 400]:
        for i in range(3):
            rows.append(
                {
                    "sample_id": f"{domain}-{i}",
                    "magnification": domain,
                    "label": i % 2,
                }
            )
    return rows


def _embeddings(n: int) -> np.ndarray:
    values = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    return values / 10.0


def test_support_split_exact_disjoint_nested_and_fallback() -> None:
    idxs = list(range(10))
    labels = {i: i % 2 for i in idxs}

    small = make_support_eval_split(
        target_domain=40,
        target_indices=idxs,
        labels_by_index=labels,
        support_size=4,
        sampling_policy="random",
        support_seed=17,
    )
    large = make_support_eval_split(
        target_domain=40,
        target_indices=idxs,
        labels_by_index=labels,
        support_size=8,
        sampling_policy="random",
        support_seed=17,
    )
    assert small.support_size_actual == 4
    assert small.eval_size == 6
    assert set(small.support_indices).isdisjoint(set(small.eval_indices))
    assert small.support_indices == large.support_indices[:4]

    fallback = make_support_eval_split(
        target_domain=40,
        target_indices=idxs,
        labels_by_index={i: 0 for i in idxs},
        support_size=4,
        sampling_policy="class_balanced",
        support_seed=17,
    )
    assert fallback.split_status == "ok"
    assert fallback.sampling_policy_effective == "random_fallback"
    assert fallback.support_labels_used == 0
    assert fallback.no_data_reason == ""

    skipped = make_support_eval_split(
        target_domain=40,
        target_indices=[0, 1, 2, 3],
        labels_by_index={0: 0, 1: 0, 2: 1, 3: 1},
        support_size=4,
        sampling_policy="class_balanced",
        support_seed=17,
    )
    assert skipped.split_status == "skipped_insufficient_samples"
    assert skipped.no_data_reason == "fewer_than_k_plus_one_target_samples"


def test_support_set_top1_uses_support_nelbo_and_eval_metrics_use_q_only() -> None:
    metadata = _metadata()
    embeddings = _embeddings(len(metadata))
    expert_domains = [40, 100, 200, 400]
    nelbo = np.full((len(metadata), len(expert_domains)), 9.0, dtype=np.float64)

    labels = {i: int(m["label"]) for i, m in enumerate(metadata)}
    split = make_support_eval_split(
        target_domain=40,
        target_indices=list(range(8)),
        labels_by_index=labels,
        support_size=4,
        sampling_policy="class_balanced",
        support_seed=17,
    )
    for idx in split.support_indices:
        nelbo[idx, expert_domains.index(100)] = 1.0
        nelbo[idx, expert_domains.index(200)] = 2.0
        nelbo[idx, expert_domains.index(400)] = 3.0
    for idx in split.eval_indices:
        nelbo[idx, expert_domains.index(100)] = 3.0
        nelbo[idx, expert_domains.index(200)] = 1.0
        nelbo[idx, expert_domains.index(400)] = 4.0

    rows = evaluate_support_set_calibration_from_arrays(
        embeddings=embeddings,
        metadata=metadata,
        nelbo_matrix=nelbo,
        expert_domains=expert_domains,
        run_meta=SupportSetRunMeta(
            dataset_name="breakhis",
            seed=42,
            backbone_type="dinov2_vitb14",
            run_id="run1",
            variant="B",
        ),
        support_sizes=[4],
        support_seeds=[17],
        sampling_policies=["class_balanced"],
    )
    target_rows = [r for r in rows if int(r.get("target_domain", -1)) == 40]
    support_row = [r for r in target_rows if r.get("method") == "support_set_calibration_top1"][0]

    assert support_row["selected_expert"] == 100
    assert support_row["oracle_expert"] == 200
    assert support_row["target_expert_excluded"] == 1
    assert support_row["support_eval_disjoint"] == 1
    assert support_row["support_size_requested"] == 4
    assert support_row["support_size_actual"] == 4
    assert support_row["eval_size"] == 4
    assert support_row["routing_uses_eval_nelbo"] == 0
    assert support_row["routing_uses_eval_indices"] == 0
    assert support_row["support_mean_nelbo"] == 1.0
    assert support_row["selected_eval_nelbo"] == 3.0
    assert support_row["oracle_eval_nelbo"] == 1.0
    assert support_row["worst_eval_nelbo"] == 4.0
    assert support_row["normalized_oracle_gap"] == normalized_oracle_gap(3.0, 1.0, 4.0)

    support_map = json.loads(str(support_row["support_nelbo_by_expert_json"]))
    eval_map = json.loads(str(support_row["eval_nelbo_by_expert_json"]))
    assert set(support_map) == {"100", "200", "400"}
    assert set(eval_map) == {"100", "200", "400"}
    assert support_map["100"] == 1.0
    assert eval_map["200"] == 1.0

    expected_cal = calibration_mae_from_vectors([1.0, 2.0, 3.0], [3.0, 1.0, 4.0])
    assert abs(float(support_row["calibration_mae"]) - expected_cal) < 1e-12


def test_oracle_and_exploratory_methods_are_not_deployable() -> None:
    metadata = _metadata()
    embeddings = _embeddings(len(metadata))
    expert_domains = [40, 100, 200, 400]
    nelbo = np.ones((len(metadata), len(expert_domains)), dtype=np.float64)
    rows = evaluate_support_set_calibration_from_arrays(
        embeddings=embeddings,
        metadata=metadata,
        nelbo_matrix=nelbo,
        expert_domains=expert_domains,
        run_meta=SupportSetRunMeta(
            dataset_name="breakhis",
            seed=42,
            backbone_type="dinov2_vitb14",
            run_id="run1",
            variant="B",
        ),
        support_sizes=[4],
        support_seeds=[17],
        sampling_policies=["random"],
        topk_values=[2],
        softmax_temperatures=[1.0],
    )
    target_rows = [r for r in rows if int(r.get("target_domain", -1)) == 40]
    domain_oracle = [r for r in target_rows if r.get("method") == "domain_oracle"][0]
    per_query_oracle = [r for r in target_rows if r.get("method") == "per_query_oracle"][0]
    topk = [r for r in target_rows if str(r.get("method", "")).startswith("support_set_topk")][0]
    soft = [r for r in target_rows if str(r.get("method", "")).startswith("support_set_softmax")][0]

    assert domain_oracle["adoption_eligible"] == 0
    assert domain_oracle["diagnostic_only"] == 1
    assert domain_oracle["routing_uses_eval_nelbo"] == 1
    assert per_query_oracle["adoption_eligible"] == 0
    assert per_query_oracle["diagnostic_only"] == 1
    assert topk["adoption_eligible"] == 0
    assert topk["exploratory_only"] == 1
    assert soft["adoption_eligible"] == 0
    assert soft["exploratory_only"] == 1
