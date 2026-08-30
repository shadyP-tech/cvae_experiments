"""Case-equal binary metrics shared by HARP policy and oracle reports."""

from __future__ import annotations

import numpy as np

from ...protocol import ProtocolError


def binary_log_loss(truth: np.ndarray, probability: np.ndarray) -> np.ndarray:
    labels = np.asarray(truth, dtype=np.int64)
    values = np.asarray(probability, dtype=np.float64)
    if (
        labels.ndim != 1
        or values.shape != labels.shape
        or not np.isin(labels, (0, 1)).all()
        or not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise ProtocolError("HARP log-loss inputs are malformed.")
    clipped = np.clip(values, 1.0e-7, 1.0 - 1.0e-7)
    return -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))


def case_equal_mean(values: np.ndarray, case_ids: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    cases = np.asarray(case_ids, dtype=object)
    if array.ndim != 1 or cases.shape != array.shape or not np.isfinite(array).all():
        raise ProtocolError("HARP case-equal values are malformed.")
    identities = tuple(sorted(set(str(value) for value in cases.tolist())))
    if not identities:
        raise ProtocolError("HARP case-equal aggregation requires cases.")
    return float(
        np.mean(
            [float(np.mean(array[cases == case], dtype=np.float64)) for case in identities],
            dtype=np.float64,
        )
    )


def case_equal_balanced_accuracy(
    truth: np.ndarray,
    probability: np.ndarray,
    case_ids: np.ndarray,
) -> float:
    labels = np.asarray(truth, dtype=np.int64)
    values = np.asarray(probability, dtype=np.float64)
    cases = np.asarray(case_ids, dtype=object)
    if (
        labels.ndim != 1
        or values.shape != labels.shape
        or cases.shape != labels.shape
        or set(int(value) for value in labels.tolist()) != {0, 1}
        or not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise ProtocolError(
            "Every HARP center requires aligned probabilities and both truth classes."
        )
    prediction = values >= 0.5
    recalls: list[float] = []
    for label in (0, 1):
        class_cases = tuple(sorted(set(str(value) for value in cases[labels == label])))
        if not class_cases:
            raise ProtocolError("HARP class/case support is empty.")
        recalls.append(
            float(
                np.mean(
                    [
                        float(
                            np.mean(
                                prediction[(labels == label) & (cases == case)]
                                == bool(label)
                            )
                        )
                        for case in class_cases
                    ],
                    dtype=np.float64,
                )
            )
        )
    return 0.5 * (recalls[0] + recalls[1])


__all__ = (
    "binary_log_loss",
    "case_equal_balanced_accuracy",
    "case_equal_mean",
)
