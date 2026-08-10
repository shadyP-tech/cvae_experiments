"""Exact pooled BACC from whole-case sufficient statistics only."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from ...protocol import ProtocolError
from .contracts import (
    BinaryLabel,
    CaseConfusionCounts,
    PairedClusterEstimate,
    PooledExactBacc,
    PredictionRow,
)
from .scientific_constants import CONFIDENCE_MULTIPLIER, VARIANCE_FLOOR


def score_case_confusions(
    predictions: Sequence[PredictionRow],
    labels: Sequence[BinaryLabel],
) -> tuple[CaseConfusionCounts, ...]:
    prediction_rows = tuple(predictions)
    label_rows = tuple(labels)
    if not prediction_rows or not label_rows:
        raise ProtocolError("Exact scoring inputs must be non-empty.")
    methods = {row.method_id for row in prediction_rows}
    if len(methods) != 1:
        raise ProtocolError("Score one method at a time before pooling.")
    method = next(iter(methods))
    prediction_by_sample: dict[tuple[str, str, str], int] = {}
    for row in prediction_rows:
        if row.sample_key in prediction_by_sample:
            raise ProtocolError("Prediction surface contains duplicate sample rows.")
        prediction_by_sample[row.sample_key] = row.hard_prediction
    label_by_sample: dict[tuple[str, str, str], int] = {}
    for row in label_rows:
        if row.sample_key in label_by_sample:
            raise ProtocolError("Label surface contains duplicate sample rows.")
        label_by_sample[row.sample_key] = row.label
    if set(prediction_by_sample) != set(label_by_sample):
        raise ProtocolError("Exact prediction and label surfaces are not aligned.")
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for key in sorted(label_by_sample):
        target, case, _sample = key
        grouped[(target, case)].append((label_by_sample[key], prediction_by_sample[key]))
    return tuple(
        CaseConfusionCounts(
            method_id=method,
            target_center=target,
            case_id=case,
            n_positive=sum(label == 1 for label, _ in values),
            true_positive=sum(label == prediction == 1 for label, prediction in values),
            n_negative=sum(label == 0 for label, _ in values),
            true_negative=sum(label == prediction == 0 for label, prediction in values),
        )
        for (target, case), values in sorted(grouped.items())
    )


def pooled_exact_bacc(rows: Sequence[CaseConfusionCounts]) -> PooledExactBacc:
    values = tuple(rows)
    if not values:
        raise ProtocolError("Cannot pool an empty whole-case scope.")
    methods = {row.method_id for row in values}
    cases = {row.case_key for row in values}
    if len(methods) != 1 or len(cases) != len(values):
        raise ProtocolError("Pooled exact-BACC rows must be one method and unique cases.")
    n_positive = sum(row.n_positive for row in values)
    n_negative = sum(row.n_negative for row in values)
    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("Pooled exact BACC requires both classes in the legal scope.")
    true_positive = sum(row.true_positive for row in values)
    true_negative = sum(row.true_negative for row in values)
    sensitivity = true_positive / n_positive
    specificity = true_negative / n_negative
    return PooledExactBacc(
        method_id=next(iter(methods)),
        case_count=len(values),
        n_positive=n_positive,
        true_positive=true_positive,
        n_negative=n_negative,
        true_negative=true_negative,
        sensitivity=sensitivity,
        specificity=specificity,
        exact_bacc=0.5 * (sensitivity + specificity),
    )


def paired_whole_case_cluster_lcb(
    challenger_rows: Sequence[CaseConfusionCounts],
    reference_rows: Sequence[CaseConfusionCounts],
    *,
    confidence_multiplier: float = CONFIDENCE_MULTIPLIER,
    variance_floor: float = VARIANCE_FLOOR,
) -> PairedClusterEstimate:
    challenger, reference = _aligned_pair(challenger_rows, reference_rows)
    multiplier = float(confidence_multiplier)
    floor = float(variance_floor)
    if multiplier <= 0.0 or not math.isfinite(multiplier):
        raise ProtocolError("Confidence multiplier must be finite and positive.")
    if floor <= 0.0 or not math.isfinite(floor):
        raise ProtocolError("Cluster variance floor must be finite and positive.")
    if len(challenger) < 2:
        raise ProtocolError("Paired whole-case uncertainty needs at least two cases.")
    n_positive = sum(row.n_positive for row in challenger)
    n_negative = sum(row.n_negative for row in challenger)
    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("Paired whole-case exact BACC requires both pooled classes.")
    positive_difference = (
        sum(left.true_positive - right.true_positive for left, right in zip(challenger, reference))
        / n_positive
    )
    negative_difference = (
        sum(left.true_negative - right.true_negative for left, right in zip(challenger, reference))
        / n_negative
    )
    difference = 0.5 * (positive_difference + negative_difference)
    influences: list[tuple[str, str, float]] = []
    for left, right in zip(challenger, reference):
        positive_term = 0.0
        if left.n_positive:
            case_difference = (left.true_positive - right.true_positive) / left.n_positive
            positive_term = left.n_positive / n_positive * (case_difference - positive_difference)
        negative_term = 0.0
        if left.n_negative:
            case_difference = (left.true_negative - right.true_negative) / left.n_negative
            negative_term = left.n_negative / n_negative * (case_difference - negative_difference)
        influences.append((left.target_center, left.case_id, 0.5 * (positive_term + negative_term)))
    count = len(influences)
    variance = max(
        count / (count - 1) * math.fsum(value * value for _h, _case, value in influences),
        floor,
    )
    standard_error = math.sqrt(variance)
    return PairedClusterEstimate(
        challenger_method=challenger[0].method_id,
        reference_method=reference[0].method_id,
        case_count=count,
        difference=difference,
        standard_error=standard_error,
        confidence_multiplier=multiplier,
        lower_bound=difference - multiplier * standard_error,
        case_influences=tuple(influences),
    )


def _aligned_pair(
    challenger_rows: Sequence[CaseConfusionCounts],
    reference_rows: Sequence[CaseConfusionCounts],
) -> tuple[tuple[CaseConfusionCounts, ...], tuple[CaseConfusionCounts, ...]]:
    challenger = tuple(sorted(challenger_rows, key=lambda row: row.case_key))
    reference = tuple(sorted(reference_rows, key=lambda row: row.case_key))
    if not challenger or len(challenger) != len(reference):
        raise ProtocolError("Paired whole-case rows must be non-empty and aligned.")
    if tuple(row.case_key for row in challenger) != tuple(row.case_key for row in reference):
        raise ProtocolError("Paired whole-case methods cover different cases.")
    if challenger[0].method_id == reference[0].method_id:
        raise ProtocolError("Paired contrast requires distinct methods.")
    for left, right in zip(challenger, reference):
        if (left.n_positive, left.n_negative) != (right.n_positive, right.n_negative):
            raise ProtocolError("Paired methods drifted from common label counts.")
    return challenger, reference


__all__ = (
    "paired_whole_case_cluster_lcb",
    "pooled_exact_bacc",
    "score_case_confusions",
)
