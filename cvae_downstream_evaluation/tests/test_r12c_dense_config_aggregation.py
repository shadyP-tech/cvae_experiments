from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.r12c_dense_config_aggregation import (  # noqa: E402
    ELIGIBILITY_AUDIT_ONLY,
    LABEL_READY_REBUILD,
    PredictionBundle,
    R12CDenseConfig,
    ROW_CROSS_BACKBONE,
    ROW_PRIMARY,
    aggregate_member_predictions,
    compute_r12c_decision_labels,
    load_r12c_config,
    rank_candidates,
    robust_score_from_vector,
    select_source_k_setting,
    write_leakage_report,
)


def test_r12c_config_loads_locked_template() -> None:
    config = load_r12c_config(ROOT / "configs" / "experiments" / "r12c_virchow2_dense_config_aggregation.yaml")
    assert config.primary_backbone == "virchow2"
    assert config.fixed_k_values == (1, 3, 5, 10)
    assert config.primary_k_values == (3, 5, 10)
    assert config.primary_calibration_rules == ("none",)


def test_robust_score_penalizes_std_and_weak_center() -> None:
    row = {"source_inner_lodo_center_bacc_vector": '{"1": 0.9, "2": 0.8, "3": 0.85}'}
    score = robust_score_from_vector(
        row,
        rank_centers=("1", "2", "3"),
        weak_center_threshold=0.85,
        std_weight=0.25,
        weak_penalty_weight=0.50,
    )
    assert round(score["mean_inner_bacc"], 6) == 0.85
    assert score["min_inner_bacc"] == 0.8
    assert score["robust_score"] < score["mean_inner_bacc"]


def test_rank_candidates_respects_inner_center_exclusion() -> None:
    config = R12CDenseConfig()
    collapses_on_3 = _candidate("a", {"1": 0.92, "2": 0.92, "3": 0.50})
    stable = _candidate("b", {"1": 0.85, "2": 0.85, "3": 0.85})

    ranked_without_3 = rank_candidates(config, [collapses_on_3, stable], rank_centers=("1", "2"))
    ranked_with_3 = rank_candidates(config, [collapses_on_3, stable], rank_centers=("1", "2", "3"))

    assert ranked_without_3[0]["config_id"].startswith("backbone=virchow2|representation=raw|C=0.01")
    assert ranked_with_3[0]["config_id"].startswith("backbone=virchow2|representation=PCA64|C=0.01")


def test_select_source_k_setting_prefers_geometric_and_smaller_k_on_tie() -> None:
    rows = [
        _k_row(k=10, rule="arithmetic", mean=0.9),
        _k_row(k=5, rule="geometric", mean=0.9),
        _k_row(k=3, rule="geometric", mean=0.9),
    ]
    selected = select_source_k_setting(rows)
    assert selected["k"] == 3
    assert selected["aggregation_rule"] == "geometric"


def test_aggregate_predictions_requires_sample_and_class_order_alignment() -> None:
    good = PredictionBundle(
        config_id="a",
        sample_ids=("s1", "s2"),
        y_true=(0, 1),
        proba=_array([[0.8, 0.2], [0.2, 0.8]]),
        pred=(0, 1),
        class_order=(0, 1),
        n_train=10,
        class_balance_train={"0": 5, "1": 5},
    )
    bad_order = PredictionBundle(
        config_id="b",
        sample_ids=("s1", "s2"),
        y_true=(0, 1),
        proba=_array([[0.8, 0.2], [0.2, 0.8]]),
        pred=(0, 1),
        class_order=(1, 0),
        n_train=10,
        class_balance_train={"0": 5, "1": 5},
    )
    bad_samples = PredictionBundle(
        config_id="c",
        sample_ids=("s2", "s1"),
        y_true=(1, 0),
        proba=_array([[0.2, 0.8], [0.8, 0.2]]),
        pred=(1, 0),
        class_order=(0, 1),
        n_train=10,
        class_balance_train={"0": 5, "1": 5},
    )

    proba, sample_ids, y_true = aggregate_member_predictions(
        [good],
        aggregation_rule="geometric",
        calibration_rule="none",
        temperature=1.0,
    )
    assert sample_ids == ("s1", "s2")
    assert y_true == (0, 1)
    assert proba.shape == (2, 2)
    with pytest.raises(Exception, match="class-order"):
        aggregate_member_predictions([good, bad_order], aggregation_rule="geometric", calibration_rule="none", temperature=1.0)
    with pytest.raises(Exception, match="sample_id"):
        aggregate_member_predictions([good, bad_samples], aggregation_rule="geometric", calibration_rule="none", temperature=1.0)


def test_rebuild_gate_ignores_cross_backbone_and_source_temperature_rows(tmp_path: Path) -> None:
    config = R12CDenseConfig()
    labels = compute_r12c_decision_labels(
        config=config,
        dense_rows=[],
        center_rows=[],
        cross_rows=[{"row_role": ROW_CROSS_BACKBONE, "bacc": 1.0, "eligibility": ELIGIBILITY_AUDIT_ONLY}],
    )
    assert LABEL_READY_REBUILD not in labels

    report = tmp_path / "leakage.json"
    write_leakage_report(
        report,
        labels=[],
        dense_rows=[{"row_role": ROW_PRIMARY, "calibration_rule": "source_temperature"}],
        cross_rows=[],
    )
    assert "source_temperature_used_in_primary_row" in report.read_text(encoding="utf-8")


def _candidate(name: str, vector: dict[str, float]) -> dict[str, object]:
    rep = "raw" if name == "a" else "PCA64"
    return {
        "row_id": name,
        "experiment_seed": 42,
        "heldout_center": "0",
        "backbone_name": "virchow2",
        "representation": rep,
        "C": 0.01,
        "class_weight": "none",
        "source_inner_lodo_center_bacc_vector": __import__("json").dumps(vector),
        "status": "ok",
    }


def _k_row(*, k: int, rule: str, mean: float) -> dict[str, object]:
    return {
        "k": k,
        "aggregation_rule": rule,
        "mean_inner_bacc": mean,
        "min_inner_bacc": mean,
        "std_inner_bacc": 0.0,
        "status": "ok",
    }


def _array(values):
    import numpy as np

    return np.asarray(values, dtype=float)
