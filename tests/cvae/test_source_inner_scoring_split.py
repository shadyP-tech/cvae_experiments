from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.routing import source_inner_utility
from midogpp_thesis.cvae.routing.source_inner_utility import (
    bundle,
    metric_scoring,
    prediction,
    prediction_io,
    runner,
    scoring,
)
from midogpp_thesis.cvae.routing.source_inner_utility.contracts import EvaluationRow


def _prediction_pass() -> prediction.PredictionPass:
    row = EvaluationRow(
        row_ordinal=0,
        manifest_row_index=11,
        sample_id="sample-0",
        case_id="case-0",
        center="0",
        split="val",
        cache_shard_path="embeddings/by_center/center_0.pt",
        cache_row_index=3,
    )
    return prediction.PredictionPass(
        evaluation_rows=(row,),
        fit_rows=({"fit_ordinal": 0, "fit_id": "fit-0"},),
        y_pred=np.asarray([[1]], dtype=np.uint8),
        prob_pos=np.asarray([[0.75]], dtype=np.float32),
    )


def test_split_modules_own_implementations_and_facades_preserve_identity() -> None:
    assert prediction.PredictionPass.__module__.endswith(".prediction")
    assert prediction.run_label_free_prediction_pass.__module__.endswith(".prediction")
    assert metric_scoring.score_prediction_pass.__module__.endswith(".metric_scoring")
    assert prediction_io.write_prediction_arrays.__module__.endswith(".prediction_io")

    assert scoring.PredictionPass is prediction.PredictionPass
    assert scoring.run_label_free_prediction_pass is prediction.run_label_free_prediction_pass
    assert scoring.array_sha256 is prediction.array_sha256
    assert scoring.generated_block_sha256 is prediction.generated_block_sha256
    assert scoring.FIT_COLUMNS is prediction.FIT_COLUMNS
    assert scoring.score_prediction_pass is metric_scoring.score_prediction_pass
    assert (
        scoring.reconstruct_metrics_from_case_confusions
        is metric_scoring.reconstruct_metrics_from_case_confusions
    )
    assert scoring.UTILITY_COLUMNS is metric_scoring.UTILITY_COLUMNS
    assert scoring.CASE_CONFUSION_COLUMNS is metric_scoring.CASE_CONFUSION_COLUMNS

    assert bundle.write_prediction_arrays is prediction_io.write_prediction_arrays
    assert bundle.read_prediction_arrays is prediction_io.read_prediction_arrays
    assert bundle.prediction_index_payload is prediction_io.prediction_index_payload
    assert bundle.evaluation_row_table is prediction_io.evaluation_row_table
    assert bundle.PREDICTION_ARRAY_MEMBER == prediction_io.PREDICTION_ARRAY_MEMBER

    assert runner.run_label_free_prediction_pass is prediction.run_label_free_prediction_pass
    assert runner.score_prediction_pass is metric_scoring.score_prediction_pass
    assert runner.write_prediction_arrays is prediction_io.write_prediction_arrays
    assert source_inner_utility.PredictionPass is prediction.PredictionPass


def test_prediction_io_preserves_npz_and_index_schemas(tmp_path: Path) -> None:
    predictions = _prediction_pass()
    path = tmp_path / "nested/candidate_predictions.npz"

    prediction_io.write_prediction_arrays(path, predictions)
    persisted_y_pred, persisted_prob_pos = bundle.read_prediction_arrays(path)

    assert np.array_equal(persisted_y_pred, predictions.y_pred)
    assert np.array_equal(persisted_prob_pos, predictions.prob_pos)
    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == {"y_pred", "prob_pos"}

    direct_index = prediction_io.prediction_index_payload(
        predictions,
        prediction_file_sha256="a" * 64,
    )
    assert direct_index == bundle.prediction_index_payload(
        predictions,
        prediction_file_sha256="a" * 64,
    )
    assert direct_index["array_member"] == "arrays/candidate_predictions.npz"
    assert direct_index["fit_index_member"] == "tables/classifier_fits.csv"
    assert direct_index["evaluation_row_index_member"] == "tables/evaluation_rows.csv"
    assert direct_index["allowed_array_keys"] == ["y_pred", "prob_pos"]
    assert direct_index["labels_stored"] is False

    row_table = prediction_io.evaluation_row_table(predictions)
    assert row_table == bundle.evaluation_row_table(predictions)
    assert tuple(row_table[0]) == prediction_io.EVALUATION_ROW_COLUMNS
    assert row_table[0]["label_present"] is False


def test_metric_reconstruction_remains_available_through_scoring_facade() -> None:
    rows = (
        {"tn": 3, "fp": 1, "fn": 2, "tp": 4},
        {"tn": 1, "fp": 0, "fn": 0, "tp": 2},
    )

    direct = metric_scoring.reconstruct_metrics_from_case_confusions(rows)

    assert direct == scoring.reconstruct_metrics_from_case_confusions(rows)
    assert direct[0] == pytest.approx(0.775)
    assert direct[1] == pytest.approx(0.7636363636363637)
