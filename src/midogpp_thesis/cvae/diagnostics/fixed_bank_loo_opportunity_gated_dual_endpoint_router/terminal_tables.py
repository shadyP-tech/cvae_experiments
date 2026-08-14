"""Terminal table construction from sealed probabilities and labels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import B_ACTION_ID, CENTERS, METHOD_IDS, a1_action_id
from .identification_metrics import calibration_parameters
from .probability_surfaces import ProbabilityIndex
from .response_products import BinaryLabel
from .terminal_products import DirectionalOracleDecision


PredictionView = dict[str, object]


def preterminal_prediction_views(rows: Sequence[object]) -> tuple[PredictionView, ...]:
    return tuple(
        {
            "target_center": str(getattr(row, "target_center")),
            "case_id": str(getattr(row, "case_id")),
            "sample_id": str(getattr(row, "sample_id")),
            "method_id": str(getattr(row, "method_id")),
            "probability": float(getattr(row, "probability")),
            "hard_prediction": int(getattr(row, "hard_prediction")),
        }
        for row in rows
    )


def oracle_prediction_views(
    surface: object,
    case_oracles: Sequence[DirectionalOracleDecision],
    static_oracles: Sequence[DirectionalOracleDecision],
) -> tuple[PredictionView, ...]:
    index = ProbabilityIndex(surface)
    case_index = {row.key: row for row in case_oracles}
    static_index = {(row.target_center, row.direction): row for row in static_oracles}
    output: list[PredictionView] = []
    for target, case, sample, action in index:
        if action != B_ACTION_ID:
            continue
        baseline = index[(target, case, sample, B_ACTION_ID)].probability_mean
        direction = "zero_to_one" if baseline < 0.5 else "one_to_zero"
        for method_id, oracle in (
            ("O_DIRECTIONAL_STATIC", static_index[(target, direction)]),
            ("O_CASE_DIRECTIONAL", case_index[(target, case, direction)]),
        ):
            source = oracle.selected_source
            probability = (
                baseline
                if source is None
                else index[(target, case, sample, a1_action_id(source))].probability_mean
            )
            output.append(
                {
                    "target_center": target,
                    "case_id": case,
                    "sample_id": sample,
                    "method_id": method_id,
                    "probability": probability,
                    "hard_prediction": int(probability >= 0.5),
                }
            )
    return tuple(output)


def terminal_case_confusions(
    predictions: Sequence[PredictionView], labels: Sequence[BinaryLabel]
) -> tuple[dict[str, object], ...]:
    truth = {row.key: row.value for row in labels}
    grouped: dict[tuple[str, str, str], list[PredictionView]] = defaultdict(list)
    for row in predictions:
        grouped[(str(row["method_id"]), str(row["target_center"]), str(row["case_id"]))].append(row)
    output: list[dict[str, object]] = []
    for (method, target, case), rows in sorted(grouped.items(), key=lambda item: (METHOD_IDS.index(item[0][0]), CENTERS.index(item[0][1]), item[0][2])):
        samples = tuple(sorted(rows, key=lambda row: str(row["sample_id"])))
        y = np.asarray([truth[(target, case, str(row["sample_id"]))] for row in samples], dtype=np.int8)
        hard = np.asarray([int(row["hard_prediction"]) for row in samples], dtype=np.int8)
        positive, negative = y == 1, y == 0
        output.append(
            {
                "method_id": method,
                "target_center": target,
                "case_id": case,
                "n_positive": int(np.sum(positive, dtype=np.int64)),
                "true_positive": int(np.sum(positive & (hard == 1), dtype=np.int64)),
                "n_negative": int(np.sum(negative, dtype=np.int64)),
                "true_negative": int(np.sum(negative & (hard == 0), dtype=np.int64)),
            }
        )
    return tuple(output)


def center_metric_rows(
    case_confusions: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    output: list[dict[str, object]] = []
    for method in METHOD_IDS:
        for center in CENTERS:
            rows = tuple(row for row in case_confusions if row["method_id"] == method and row["target_center"] == center)
            positive = sum(int(row["n_positive"]) for row in rows)
            negative = sum(int(row["n_negative"]) for row in rows)
            tp = sum(int(row["true_positive"]) for row in rows)
            tn = sum(int(row["true_negative"]) for row in rows)
            if not rows or positive <= 0 or negative <= 0:
                raise ProtocolError("OGDE terminal center metric lacks cases or a class.")
            sensitivity, specificity = tp / positive, tn / negative
            output.append(
                {
                    "method_id": method,
                    "target_center": center,
                    "case_count": len(rows),
                    "n_positive": positive,
                    "true_positive": tp,
                    "n_negative": negative,
                    "true_negative": tn,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "bacc": 0.5 * (sensitivity + specificity),
                }
            )
    return tuple(output)


def method_and_calibration_rows(
    predictions: Sequence[PredictionView],
    labels: Sequence[BinaryLabel],
    center_metrics: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    truth = {row.key: row.value for row in labels}
    baseline = {
        (str(row["target_center"]), str(row["case_id"]), str(row["sample_id"])): row
        for row in predictions
        if row["method_id"] == "B"
    }
    methods: list[dict[str, object]] = []
    calibration: list[dict[str, object]] = []
    for method in METHOD_IDS:
        rows = tuple(row for row in predictions if row["method_id"] == method)
        indexed = {
            (str(row["target_center"]), str(row["case_id"]), str(row["sample_id"])): row
            for row in rows
        }
        if set(indexed) != set(truth):
            raise ProtocolError("OGDE terminal method predictions are not full-surface aligned.")
        keys = sorted(indexed)
        y = np.asarray([truth[key] for key in keys], dtype=np.float64)
        p = np.asarray([float(indexed[key]["probability"]) for key in keys], dtype=np.float64)
        hard = np.asarray([int(indexed[key]["hard_prediction"]) for key in keys], dtype=np.int8)
        brier = float(np.mean((p - y) ** 2, dtype=np.float64))
        clipped = np.clip(p, 1.0e-12, 1.0 - 1.0e-12)
        log_loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped), dtype=np.float64))
        intercept, slope = calibration_parameters(y, p)
        bacc = float(
            np.mean(
                [float(row["bacc"]) for row in center_metrics if row["method_id"] == method],
                dtype=np.float64,
            )
        )
        methods.append(
            {
                "method_id": method,
                "sample_count": len(rows),
                "center_count": len(CENTERS),
                "equal_center_bacc": bacc,
                "brier_score": brier,
                "log_loss": log_loss,
                "calibration_intercept": intercept,
                "calibration_slope": slope,
            }
        )
        baseline_hard = np.asarray([int(baseline[key]["hard_prediction"]) for key in keys], dtype=np.int8)
        changed = hard != baseline_hard
        helpful = int(np.sum(changed & (hard == y), dtype=np.int64))
        harmful = int(np.sum(changed & (baseline_hard == y), dtype=np.int64))
        calibration.append(
            {
                "method_id": method,
                "brier_score": brier,
                "log_loss": log_loss,
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "threshold_crossings_vs_B": int(np.sum(changed, dtype=np.int64)),
                "helpful_crossings_vs_B": helpful,
                "harmful_crossings_vs_B": harmful,
                "net_helpful_crossings_vs_B": helpful - harmful,
            }
        )
    return tuple(methods), tuple(calibration)


def contrast_rows(center_metrics: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    pairs = (
        ("OGDE_PORTFOLIO", "B"),
        ("OGDE_PORTFOLIO", "U"),
        ("OGDE_PORTFOLIO", "R_NINE_ARM_ROBUST"),
        ("OGDE_PORTFOLIO", "CALIBRATION_ONLY_B_R"),
    )
    lookup = {(str(row["method_id"]), str(row["target_center"])): float(row["bacc"]) for row in center_metrics}
    output = []
    for candidate, reference in pairs:
        differences = np.asarray([lookup[(candidate, center)] - lookup[(reference, center)] for center in CENTERS], dtype=np.float64)
        mean = float(np.mean(differences, dtype=np.float64))
        sd = float(np.std(differences, ddof=1, dtype=np.float64))
        half = 2.306004135204166 * sd / np.sqrt(len(CENTERS))
        output.append(
            {
                "contrast_id": f"{candidate}-{reference}",
                "candidate_method": candidate,
                "reference_method": reference,
                "center_count": len(CENTERS),
                "mean_gain": mean,
                "sd_gain": sd,
                "descriptive_t8_lower": mean - half,
                "descriptive_t8_upper": mean + half,
                "nominal_inference_claimed": False,
            }
        )
    return tuple(output)


__all__ = (
    "center_metric_rows",
    "contrast_rows",
    "method_and_calibration_rows",
    "oracle_prediction_views",
    "preterminal_prediction_views",
    "terminal_case_confusions",
)
