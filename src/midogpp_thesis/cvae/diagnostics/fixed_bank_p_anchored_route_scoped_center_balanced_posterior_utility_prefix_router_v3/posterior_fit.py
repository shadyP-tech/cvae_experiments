"""One-fit H-c posterior kernel used by identity and cyclic controls."""

from __future__ import annotations

from collections.abc import Sequence
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from ...protocol import ProtocolError
from .constants import (
    TARGET_POSTERIOR_C,
    TARGET_POSTERIOR_MAX_ITER,
    TARGET_POSTERIOR_PROBABILITY_CLIP,
    TARGET_POSTERIOR_RANDOM_STATE,
    TARGET_POSTERIOR_SOLVER,
    TARGET_POSTERIOR_TOLERANCE,
)
from .contracts import BinaryLabel
from .hashing import canonical_hash
from .posterior_contracts import (
    CasePosteriorPrediction,
    PhysicalFingerprintSurface,
    TargetLocalPosteriorModel,
)


def fit_route_posterior(
    fingerprint: PhysicalFingerprintSurface,
    *,
    held_case_id: object,
    support_labels: Sequence[BinaryLabel],
) -> tuple[TargetLocalPosteriorModel, CasePosteriorPrediction]:
    """Fit once on every sample in H-c, then predict c.

    The capability is verified by exact identity equality.  No validation fold,
    OOF reliability statistic, or target-case label enters the fit.
    """

    held = str(held_case_id)
    if held not in fingerprint.cases:
        raise ProtocolError("CBPUPR posterior held case is absent.")
    train_positions = np.asarray(
        [index for index, case in enumerate(fingerprint.case_ids) if case != held],
        dtype=np.int64,
    )
    held_positions = fingerprint.positions(held)
    expected = {
        (fingerprint.center, fingerprint.case_ids[index], fingerprint.sample_ids[index])
        for index in train_positions
    }
    labels = tuple(support_labels)
    label_by_key = {row.key: row for row in labels}
    expected_scope = f"outer_support::H={fingerprint.center}::excluded_c={held}"
    if (
        not len(train_positions)
        or len(label_by_key) != len(labels)
        or set(label_by_key) != expected
        or any(row.case_id == held or row.center != fingerprint.center for row in labels)
        or {row.scope for row in labels} != {expected_scope}
    ):
        raise ProtocolError("CBPUPR posterior escaped exact H-c support.")

    x = fingerprint.feature_values[train_positions]
    y = np.asarray(
        [
            label_by_key[
                (fingerprint.center, fingerprint.case_ids[index], fingerprint.sample_ids[index])
            ].value
            for index in train_positions
        ],
        dtype=np.int64,
    )
    n_positive = int(np.sum(y == 1, dtype=np.int64))
    n_negative = int(np.sum(y == 0, dtype=np.int64))
    if min(n_positive, n_negative) <= 0:
        raise ProtocolError("CBPUPR H-c posterior requires both classes.")

    feature_mean = np.mean(x, axis=0, dtype=np.float64)
    feature_scale = np.std(x, axis=0, ddof=0, dtype=np.float64)
    feature_scale = np.where(feature_scale > 1.0e-12, feature_scale, 1.0)
    estimator = LogisticRegression(
        C=TARGET_POSTERIOR_C,
        penalty="l2",
        class_weight="balanced",
        solver=TARGET_POSTERIOR_SOLVER,
        max_iter=TARGET_POSTERIOR_MAX_ITER,
        random_state=TARGET_POSTERIOR_RANDOM_STATE,
        tol=TARGET_POSTERIOR_TOLERANCE,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit((x - feature_mean) / feature_scale, y)
    if any(issubclass(row.category, ConvergenceWarning) for row in captured):
        raise ProtocolError("CBPUPR route posterior failed to converge.")
    iterations = int(estimator.n_iter_[0])
    if iterations >= TARGET_POSTERIOR_MAX_ITER:
        raise ProtocolError("CBPUPR route posterior exhausted max_iter.")

    training_cases = tuple(
        sorted({fingerprint.case_ids[index] for index in train_positions})
    )
    identity_hash = canonical_hash([list(key) for key in sorted(expected)])
    model = TargetLocalPosteriorModel(
        fingerprint.center,
        held,
        fingerprint.control_id,
        training_cases,
        fingerprint.feature_names,
        tuple(float(value) for value in feature_mean),
        tuple(float(value) for value in feature_scale),
        tuple(float(value) for value in estimator.coef_[0]),
        float(estimator.intercept_[0]),
        len(y),
        n_positive,
        n_negative,
        fingerprint.fingerprint_hash,
        identity_hash,
        iterations,
        True,
    )

    prediction = predict_route_posterior(fingerprint, model)
    return model, prediction


def predict_route_posterior(
    fingerprint: PhysicalFingerprintSurface,
    model: TargetLocalPosteriorModel,
) -> CasePosteriorPrediction:
    """Replay one fitted model on its whole held case without labels or refits."""

    if (
        model.target_center != fingerprint.center
        or model.control_id != fingerprint.control_id
        or model.fingerprint_hash != fingerprint.fingerprint_hash
        or model.held_case_id not in fingerprint.cases
        or tuple(model.feature_names) != tuple(fingerprint.feature_names)
    ):
        raise ProtocolError("CBPUPR posterior replay lineage drifted.")
    held_positions = fingerprint.positions(model.held_case_id)
    held_x = fingerprint.feature_values[held_positions]
    feature_mean = np.asarray(model.feature_mean, dtype=np.float64)
    feature_scale = np.asarray(model.feature_scale, dtype=np.float64)
    standardized = (held_x - feature_mean) / feature_scale
    logits = standardized @ np.asarray(model.coefficients, dtype=np.float64)
    logits += model.intercept
    balanced = np.clip(
        _sigmoid(logits),
        TARGET_POSTERIOR_PROBABILITY_CLIP,
        1.0 - TARGET_POSTERIOR_PROBABILITY_CLIP,
    )
    prevalence = model.training_n_positive / model.training_row_count
    numerator = prevalence * balanced
    denominator = numerator + (1.0 - prevalence) * (1.0 - balanced)
    natural = np.clip(
        numerator / denominator,
        TARGET_POSTERIOR_PROBABILITY_CLIP,
        1.0 - TARGET_POSTERIOR_PROBABILITY_CLIP,
    )
    prediction = CasePosteriorPrediction(
        fingerprint.center,
        model.held_case_id,
        fingerprint.control_id,
        tuple(fingerprint.sample_ids[index] for index in held_positions),
        tuple(float(value) for value in natural),
        model.model_hash,
        fingerprint.fingerprint_hash,
    )
    return prediction


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


__all__ = ("fit_route_posterior", "predict_route_posterior")
