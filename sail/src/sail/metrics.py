"""Small metric helpers used by the extracted pipeline."""

from __future__ import annotations

import math
from typing import Sequence


def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    classes = sorted(set(int(value) for value in y_true))
    if not classes:
        raise ValueError("Cannot compute balanced accuracy with no labels.")
    recalls = []
    for cls in classes:
        total = sum(1 for value in y_true if int(value) == cls)
        correct = sum(
            1
            for truth, pred in zip(y_true, y_pred)
            if int(truth) == cls and int(pred) == cls
        )
        recalls.append(float(correct) / float(total) if total else 0.0)
    return sum(recalls) / float(len(recalls))


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    classes = sorted(set(int(value) for value in y_true).union(int(value) for value in y_pred))
    if not classes:
        raise ValueError("Cannot compute macro-F1 with no labels.")
    scores = []
    for cls in classes:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if int(truth) == cls and int(pred) == cls)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if int(truth) != cls and int(pred) == cls)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if int(truth) == cls and int(pred) != cls)
        denom = (2 * tp) + fp + fn
        scores.append(float(2 * tp) / float(denom) if denom else 0.0)
    return sum(scores) / float(len(scores))


def binary_auroc(y_true: Sequence[int], prob_pos: Sequence[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        return float(roc_auc_score(list(y_true), list(prob_pos)))
    except Exception:
        return math.nan


def nanmean(values: Sequence[object]) -> float:
    vals = [_float(value) for value in values]
    vals = [value for value in vals if not math.isnan(value)]
    return sum(vals) / float(len(vals)) if vals else math.nan


def nanmin(values: Sequence[object]) -> float:
    vals = [_float(value) for value in values]
    vals = [value for value in vals if not math.isnan(value)]
    return min(vals) if vals else math.nan


def nanstd(values: Sequence[object]) -> float:
    vals = [_float(value) for value in values]
    vals = [value for value in vals if not math.isnan(value)]
    if not vals:
        return math.nan
    mu = nanmean(vals)
    return math.sqrt(sum((value - mu) ** 2 for value in vals) / len(vals))


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Spearman inputs must have equal length.")
    if len(xs) < 2:
        return math.nan
    rx = _rank(xs)
    ry = _rank(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((left - mx) * (right - my) for left, right in zip(rx, ry))
    den_x = math.sqrt(sum((value - mx) ** 2 for value in rx))
    den_y = math.sqrt(sum((value - my) ** 2 for value in ry))
    if den_x == 0.0 or den_y == 0.0:
        return math.nan
    return float(num / (den_x * den_y))


def _rank(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (float(i + 1) + float(j)) / 2.0
        for pos in range(i, j):
            ranks[indexed[pos][0]] = avg
        i = j
    return ranks


def _float(value: object) -> float:
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan
