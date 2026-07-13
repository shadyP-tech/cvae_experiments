"""Frozen-spec representation scoring and preservation ratios."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from ...real_features.classifier_reference.classifiers import ClassifierSpec, fit_logistic_classifier
from ..metrics import balanced_accuracy, macro_f1


@dataclass(frozen=True)
class RepresentationScore:
    bacc: float
    macro_f1: float
    predictions: tuple[int, ...]
    probabilities_positive: tuple[float, ...]
    converged: bool
    classifier_spec_hash: str


def score_representation(
    train_embeddings: Sequence[Sequence[float]],
    train_labels: Sequence[int],
    eval_embeddings: Sequence[Sequence[float]],
    eval_labels: Sequence[int],
    *,
    spec: ClassifierSpec,
) -> RepresentationScore:
    fitted = fit_logistic_classifier(train_embeddings, train_labels, eval_embeddings, spec=spec)
    predictions = tuple(int(value) for value in fitted.predictions.tolist())
    probabilities = fitted.probabilities.tolist()
    return RepresentationScore(
        bacc=balanced_accuracy(eval_labels, predictions),
        macro_f1=macro_f1(eval_labels, predictions),
        predictions=predictions,
        probabilities_positive=tuple(float(row[1]) for row in probabilities),
        converged=bool(fitted.converged),
        classifier_spec_hash=spec.config_hash,
    )


def chance_normalized_preservation(
    generated_bacc: float,
    real_reference_bacc: float,
    *,
    minimum_real_bacc: float = 0.55,
) -> float:
    if not math.isfinite(float(generated_bacc)) or not math.isfinite(float(real_reference_bacc)):
        raise ValueError("Preservation ratio inputs must be finite.")
    if float(real_reference_bacc) < float(minimum_real_bacc):
        raise ValueError("Real-reference BACC is below the predeclared denominator floor.")
    return (float(generated_bacc) - 0.5) / (float(real_reference_bacc) - 0.5)
