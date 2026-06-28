"""Diagnostics for source-inner learned utility estimators."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from ..downstream import spearman
from ..protocol import ProtocolError


def estimator_diagnostics(
    rows: Sequence[Mapping[str, object]],
    *,
    prediction_column: str = "predicted_primary_utility",
    label_column: str = "source_inner_heldout_bacc",
) -> dict[str, float]:
    if not rows:
        raise ProtocolError("Cannot compute estimator diagnostics with no rows.")
    predictions = [float(row[prediction_column]) for row in rows if prediction_column in row]
    labels = [float(row[label_column]) for row in rows if label_column in row]
    if len(predictions) != len(rows) or len(labels) != len(rows):
        raise ProtocolError("Every diagnostic row must contain prediction and label columns.")
    errors = [pred - label for pred, label in zip(predictions, labels)]
    return {
        "n_rows": float(len(rows)),
        "mae": sum(abs(err) for err in errors) / float(len(errors)),
        "rmse": math.sqrt(sum(err * err for err in errors) / float(len(errors))),
        "spearman_predicted_vs_observed": spearman(predictions, labels),
        "pairwise_preference_accuracy": pairwise_preference_accuracy(predictions, labels),
    }


def pairwise_preference_accuracy(predictions: Sequence[float], labels: Sequence[float]) -> float:
    if len(predictions) != len(labels):
        raise ValueError("Pairwise preference inputs must have equal length.")
    correct = 0
    total = 0
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            label_delta = float(labels[i]) - float(labels[j])
            pred_delta = float(predictions[i]) - float(predictions[j])
            if label_delta == 0.0:
                continue
            total += 1
            if pred_delta == 0.0:
                correct += 0.5
            elif (pred_delta > 0.0) == (label_delta > 0.0):
                correct += 1
    return float(correct) / float(total) if total else math.nan
