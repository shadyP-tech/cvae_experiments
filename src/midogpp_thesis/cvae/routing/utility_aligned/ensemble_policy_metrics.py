"""Routing metric vectors and deterministic numeric helpers."""

from __future__ import annotations

import numpy as np

from ...metrics import spearman
from ...protocol import ProtocolError


def routing_metric_vectors(
    keys: tuple[tuple[str, str, str], ...],
    predictions: np.ndarray,
    truth: np.ndarray,
) -> dict[str, np.ndarray]:
    if predictions.shape != truth.shape or predictions.shape != (len(keys),):
        raise ProtocolError("Ensemble routing metric arrays are misaligned.")
    by_query: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        by_query.setdefault(key[1], []).append(index)
    top1: list[float] = []
    correlations: list[float] = []
    regrets: list[float] = []
    for query, indices in sorted(by_query.items()):
        sources = [keys[index][2] for index in indices]
        pred_by_source = {source: float(predictions[index]) for source, index in zip(sources, indices)}
        truth_by_source = {source: float(truth[index]) for source, index in zip(sources, indices)}
        selected = min(pred_by_source, key=lambda source: (-pred_by_source[source], source))
        maximum = max(truth_by_source.values())
        minimum = min(truth_by_source.values())
        oracle = {source for source, value in truth_by_source.items() if value == maximum}
        top1.append(1.0 if selected in oracle else 0.0)
        correlation = float(
            spearman(
                [pred_by_source[source] for source in sorted(sources)],
                [truth_by_source[source] for source in sorted(sources)],
            )
        )
        correlations.append(0.0 if not np.isfinite(correlation) else correlation)
        denominator = maximum - minimum
        regrets.append(
            0.0
            if denominator <= 0.0
            else (maximum - truth_by_source[selected]) / denominator
        )
    output = {
        "top1": np.asarray(top1, dtype=np.float64),
        "spearman": np.asarray(correlations, dtype=np.float64),
        "normalized_gap": np.asarray(regrets, dtype=np.float64),
    }
    for values in output.values():
        values.setflags(write=False)
    return output


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


__all__ = ("quantile", "routing_metric_vectors")

