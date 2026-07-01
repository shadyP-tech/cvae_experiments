"""Metric boundary for transfer diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn


def predicted_positive_rate(counts: ConfusionCounts) -> float:
    if counts.n == 0:
        return float("nan")
    return float((counts.tp + counts.fp) / counts.n)


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> ConfusionCounts:
    return ConfusionCounts(
        tp=sum(1 for truth, pred in zip(y_true, y_pred) if int(truth) == 1 and int(pred) == 1),
        fp=sum(1 for truth, pred in zip(y_true, y_pred) if int(truth) == 0 and int(pred) == 1),
        tn=sum(1 for truth, pred in zip(y_true, y_pred) if int(truth) == 0 and int(pred) == 0),
        fn=sum(1 for truth, pred in zip(y_true, y_pred) if int(truth) == 1 and int(pred) == 0),
    )


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int], prob_pos: Sequence[float]) -> dict[str, float]:
    counts = confusion_counts(y_true, y_pred)
    n_pos = counts.tp + counts.fn
    n_neg = counts.tn + counts.fp
    sensitivity = float(counts.tp / n_pos) if n_pos else math.nan
    specificity = float(counts.tn / n_neg) if n_neg else math.nan
    precision = float(counts.tp / (counts.tp + counts.fp)) if (counts.tp + counts.fp) else 0.0
    f1_pos = float(2 * precision * sensitivity / (precision + sensitivity)) if precision + sensitivity else 0.0
    f1_neg = _class_f1(counts.tn, counts.fn, counts.fp)
    return {
        "target_prevalence": float(n_pos / counts.n) if counts.n else math.nan,
        "predicted_positive_rate": predicted_positive_rate(counts),
        "tp": float(counts.tp),
        "fp": float(counts.fp),
        "tn": float(counts.tn),
        "fn": float(counts.fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "macro_f1": float((f1_pos + f1_neg) / 2.0),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0)
        if not math.isnan(sensitivity) and not math.isnan(specificity)
        else math.nan,
        "auroc": auroc(y_true, prob_pos),
        "pr_auc": pr_auc(y_true, prob_pos),
        "pr_auc_baseline": float(n_pos / counts.n) if counts.n else math.nan,
    }


def auroc(y_true: Sequence[int], prob_pos: Sequence[float]) -> float:
    if len(set(int(value) for value in y_true)) < 2:
        return math.nan
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        return float(roc_auc_score(list(y_true), list(prob_pos)))
    except Exception:
        return _rank_auc(y_true, prob_pos)


def pr_auc(y_true: Sequence[int], prob_pos: Sequence[float]) -> float:
    if len(set(int(value) for value in y_true)) < 2:
        return math.nan
    try:
        from sklearn.metrics import average_precision_score  # type: ignore

        return float(average_precision_score(list(y_true), list(prob_pos)))
    except Exception:
        return math.nan


def _class_f1(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp) + fp + fn
    return float((2 * tp) / denom) if denom else 0.0


def _rank_auc(y_true: Sequence[int], prob_pos: Sequence[float]) -> float:
    positives = [float(score) for truth, score in zip(y_true, prob_pos) if int(truth) == 1]
    negatives = [float(score) for truth, score in zip(y_true, prob_pos) if int(truth) == 0]
    if not positives or not negatives:
        return math.nan
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return float(wins / total) if total else math.nan
