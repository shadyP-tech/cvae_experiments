from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .metrics import balanced_accuracy, macro_f1
from .protocol import ProtocolError


@dataclass(frozen=True)
class PredictionBundle:
    expert_id: str
    probabilities: tuple[tuple[float, ...], ...]
    classes: tuple[int, ...] = (0, 1)


@dataclass(frozen=True)
class DownstreamResult:
    method: str
    bacc: float
    macro_f1: float
    n_target_eval: int


def geometric_probability_pool(
    bundles: Sequence[PredictionBundle],
    *,
    eps: float = 1.0e-12,
) -> tuple[tuple[float, ...], ...]:
    if not bundles:
        raise ProtocolError("At least one prediction bundle is required.")
    classes = bundles[0].classes
    n_rows = len(bundles[0].probabilities)
    for bundle in bundles:
        if bundle.classes != classes:
            raise ProtocolError("Class order mismatch in geometric pooling.")
        if len(bundle.probabilities) != n_rows:
            raise ProtocolError("Prediction row count mismatch in geometric pooling.")
    pooled: list[tuple[float, ...]] = []
    for row_idx in range(n_rows):
        logs = [0.0 for _ in classes]
        for bundle in bundles:
            row = bundle.probabilities[row_idx]
            if len(row) != len(classes):
                raise ProtocolError("Probability width mismatch in geometric pooling.")
            for cls_idx, prob in enumerate(row):
                logs[cls_idx] += math.log(max(float(prob), float(eps)))
        logs = [value / float(len(bundles)) for value in logs]
        max_log = max(logs)
        exp_vals = [math.exp(value - max_log) for value in logs]
        denom = sum(exp_vals)
        pooled.append(tuple(value / denom for value in exp_vals))
    return tuple(pooled)


def predict_from_probabilities(probabilities: Sequence[Sequence[float]], classes: Sequence[int] = (0, 1)) -> tuple[int, ...]:
    preds = []
    for row in probabilities:
        best_idx = max(range(len(row)), key=lambda idx: (float(row[idx]), -idx))
        preds.append(int(classes[best_idx]))
    return tuple(preds)


def evaluate_probability_predictions(
    method: str,
    probabilities: Sequence[Sequence[float]],
    target_labels: Sequence[int],
    *,
    classes: Sequence[int] = (0, 1),
) -> DownstreamResult:
    preds = predict_from_probabilities(probabilities, classes=classes)
    return DownstreamResult(
        method=str(method),
        bacc=balanced_accuracy(target_labels, preds),
        macro_f1=macro_f1(target_labels, preds),
        n_target_eval=len(target_labels),
    )


def fit_locked_logistic_classifier(
    synthetic_embeddings: Sequence[Sequence[float]],
    synthetic_labels: Sequence[int],
    target_embeddings: Sequence[Sequence[float]],
    *,
    classifier_seed: int,
    expert_id: str,
) -> PredictionBundle:
    try:
        import numpy as np  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Downstream classifier requires numpy and scikit-learn.") from exc

    x_syn = np.asarray(synthetic_embeddings, dtype=float)
    y_syn = np.asarray(synthetic_labels, dtype=int)
    x_eval = np.asarray(target_embeddings, dtype=float)
    if x_syn.ndim != 2 or x_eval.ndim != 2:
        raise ValueError("Synthetic and target embeddings must be 2D arrays.")
    if x_syn.shape[1] != x_eval.shape[1]:
        raise ValueError("Synthetic and target embeddings must share a feature frame.")
    if sorted(set(int(v) for v in y_syn.tolist())) != [0, 1]:
        raise ValueError("Locked downstream classifier requires binary synthetic labels 0/1.")
    scaler = StandardScaler()
    x_syn_scaled = scaler.fit_transform(x_syn)
    x_eval_scaled = scaler.transform(x_eval)
    clf = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=2000,
        class_weight=None,
        random_state=int(classifier_seed),
    )
    clf.fit(x_syn_scaled, y_syn)
    classes = tuple(int(v) for v in clf.classes_.tolist())
    return PredictionBundle(
        expert_id=str(expert_id),
        probabilities=tuple(tuple(float(v) for v in row) for row in clf.predict_proba(x_eval_scaled)),
        classes=classes,
    )
