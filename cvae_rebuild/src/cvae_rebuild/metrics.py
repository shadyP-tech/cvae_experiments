from __future__ import annotations

import math
from typing import Sequence


def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    classes = sorted(set(int(v) for v in y_true))
    if not classes:
        raise ValueError("Cannot compute balanced accuracy without labels.")
    recalls = []
    for cls in classes:
        total = sum(1 for v in y_true if int(v) == cls)
        correct = sum(1 for t, p in zip(y_true, y_pred) if int(t) == cls and int(p) == cls)
        recalls.append(float(correct) / float(total) if total else 0.0)
    return sum(recalls) / float(len(recalls))


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    classes = sorted(set(int(v) for v in y_true).union(int(v) for v in y_pred))
    if not classes:
        raise ValueError("Cannot compute macro-F1 without labels.")
    values = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if int(t) == cls and int(p) == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if int(t) != cls and int(p) == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if int(t) == cls and int(p) != cls)
        denom = (2 * tp) + fp + fn
        values.append(float(2 * tp) / float(denom) if denom else 0.0)
    return sum(values) / float(len(values))


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Spearman inputs must have equal length.")
    if len(xs) < 2:
        return math.nan
    rx = _rank(xs)
    ry = _rank(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in rx))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ry))
    if den_x == 0.0 or den_y == 0.0:
        return math.nan
    return float(num / (den_x * den_y))


def nanmean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return sum(vals) / float(len(vals)) if vals else math.nan


def _rank(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(float(v) for v in values), key=lambda item: item[1])
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
