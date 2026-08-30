"""Post-seal HARP replay metrics; this module never changes routing."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import numpy as np

from ...protocol import ProtocolError
from .capability import HarpReplayCapability
from .sealing import FrozenHarpPredictionSeal


@dataclass(frozen=True)
class HarpReplayMetrics:
    aggregation_unit: str
    row_count: int
    baseline_balanced_accuracy: float
    routed_balanced_accuracy: float
    balanced_accuracy_delta: float
    baseline_brier: float
    routed_brier: float
    brier_delta: float
    baseline_log_loss: float
    routed_log_loss: float
    log_loss_delta: float
    route_rate: float


@dataclass(frozen=True)
class HarpReplayResult:
    prediction_seal_hash: str
    metrics: HarpReplayMetrics
    center_metrics: tuple[tuple[str, HarpReplayMetrics], ...]
    descriptive_row_metrics: HarpReplayMetrics


def _balanced_accuracy(truth: np.ndarray, probability: np.ndarray) -> float:
    prediction = probability >= 0.5
    if set(int(value) for value in truth) != {0, 1}:
        raise ProtocolError("A descriptive HARP replay surface must contain both truth classes.")
    tpr = float(np.mean(prediction[truth == 1]))
    tnr = float(np.mean(~prediction[truth == 0]))
    return 0.5 * (tpr + tnr)


def _log_loss(truth: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return float(np.mean(-(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped))))


def _row_metrics(rows: tuple[object, ...], truth_by_key: dict[tuple[str, str, str], int], *, unit: str) -> HarpReplayMetrics:
    truth = np.asarray([truth_by_key[row.row_key] for row in rows], dtype=np.int64)
    baseline = np.asarray([struct.unpack("<d", row.baseline_probability_bytes)[0] for row in rows], dtype=np.float64)
    output = np.asarray([struct.unpack("<d", row.output_probability_bytes)[0] for row in rows], dtype=np.float64)
    baseline_bacc = _balanced_accuracy(truth, baseline)
    output_bacc = _balanced_accuracy(truth, output)
    baseline_brier = float(np.mean((baseline - truth) ** 2))
    output_brier = float(np.mean((output - truth) ** 2))
    baseline_loss = _log_loss(truth, baseline)
    output_loss = _log_loss(truth, output)
    return HarpReplayMetrics(unit, len(rows), baseline_bacc, output_bacc, output_bacc - baseline_bacc, baseline_brier, output_brier, output_brier - baseline_brier, baseline_loss, output_loss, output_loss - baseline_loss, float(np.mean([row.routed for row in rows])))


def _case_equal_mean(values: np.ndarray, case_ids: np.ndarray) -> float:
    cases = tuple(sorted(set(str(value) for value in case_ids)))
    if not cases:
        raise ProtocolError("HARP replay contains no independent cases.")
    return float(np.mean([
        float(np.mean(values[case_ids == case], dtype=np.float64))
        for case in cases
    ], dtype=np.float64))


def _case_equal_balanced_accuracy(
    truth: np.ndarray,
    probability: np.ndarray,
    case_ids: np.ndarray,
) -> float:
    """Balance classes, then give each supporting case equal mass.

    A case need not contain both classes.  For each class, recalls are averaged
    across the independent cases that contain that class; the two class means
    are then averaged.  This is the same estimand as the fresh HARP endpoint.
    """

    prediction = probability >= 0.5
    if set(int(value) for value in truth) != {0, 1}:
        raise ProtocolError("Every HARP replay target center must contain both truth classes.")
    recalls: list[float] = []
    for label in (0, 1):
        class_cases = tuple(sorted(set(str(value) for value in case_ids[truth == label])))
        if not class_cases:
            raise ProtocolError("HARP replay class/case support is empty.")
        recalls.append(float(np.mean([
            float(np.mean(prediction[(truth == label) & (case_ids == case)] == bool(label)))
            for case in class_cases
        ], dtype=np.float64)))
    return 0.5 * (recalls[0] + recalls[1])


def _center_metrics(
    rows: tuple[object, ...],
    truth_by_key: dict[tuple[str, str, str], int],
) -> HarpReplayMetrics:
    truth = np.asarray([truth_by_key[row.row_key] for row in rows], dtype=np.int64)
    baseline = np.asarray([struct.unpack("<d", row.baseline_probability_bytes)[0] for row in rows], dtype=np.float64)
    output = np.asarray([struct.unpack("<d", row.output_probability_bytes)[0] for row in rows], dtype=np.float64)
    cases = np.asarray([row.case_id for row in rows], dtype=str)
    routed = np.asarray([row.routed for row in rows], dtype=np.float64)
    baseline_bacc = _case_equal_balanced_accuracy(truth, baseline, cases)
    output_bacc = _case_equal_balanced_accuracy(truth, output, cases)
    baseline_brier = _case_equal_mean((baseline - truth) ** 2, cases)
    output_brier = _case_equal_mean((output - truth) ** 2, cases)
    baseline_loss = _case_equal_mean(
        -(truth * np.log(np.clip(baseline, 1e-7, 1 - 1e-7)) + (1 - truth) * np.log1p(-np.clip(baseline, 1e-7, 1 - 1e-7))),
        cases,
    )
    output_loss = _case_equal_mean(
        -(truth * np.log(np.clip(output, 1e-7, 1 - 1e-7)) + (1 - truth) * np.log1p(-np.clip(output, 1e-7, 1 - 1e-7))),
        cases,
    )
    return HarpReplayMetrics(
        "equal_case_within_target", len(rows), baseline_bacc, output_bacc,
        output_bacc - baseline_bacc, baseline_brier, output_brier,
        output_brier - baseline_brier, baseline_loss, output_loss,
        output_loss - baseline_loss, _case_equal_mean(routed, cases),
    )


def _equal_mean(values: tuple[HarpReplayMetrics, ...], *, unit: str) -> HarpReplayMetrics:
    if not values:
        raise ProtocolError("HARP hierarchical replay aggregation is empty.")
    names = (
        "baseline_balanced_accuracy", "routed_balanced_accuracy", "balanced_accuracy_delta",
        "baseline_brier", "routed_brier", "brier_delta", "baseline_log_loss",
        "routed_log_loss", "log_loss_delta", "route_rate",
    )
    means = [float(np.mean([getattr(value, name) for value in values])) for name in names]
    return HarpReplayMetrics(unit, sum(value.row_count for value in values), *means)


def evaluate_harp_replay(seal: FrozenHarpPredictionSeal, capability: HarpReplayCapability) -> HarpReplayResult:
    if not isinstance(seal, FrozenHarpPredictionSeal) or not isinstance(capability, HarpReplayCapability):
        raise ProtocolError("HARP replay requires a frozen seal and one-shot capability.")
    truth_by_key = capability.consume(seal)
    rows = tuple(seal.decisions)
    by_center: dict[str, list[object]] = {}
    for row in rows:
        by_center.setdefault(row.outer_target_id, []).append(row)
    center_metrics = tuple(
        (center, _center_metrics(tuple(by_center[center]), truth_by_key))
        for center in sorted(by_center)
    )
    primary = _equal_mean(tuple(value for _center, value in center_metrics), unit="equal_target_center")
    descriptive = _row_metrics(rows, truth_by_key, unit="descriptive_raw_row")
    return HarpReplayResult(seal.seal_hash, primary, center_metrics, descriptive)


__all__ = ("HarpReplayMetrics", "HarpReplayResult", "evaluate_harp_replay")
