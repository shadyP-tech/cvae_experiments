"""Deterministic whole-case support cross-fitting for route-local posteriors."""

from __future__ import annotations

from collections.abc import Sequence
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from ...protocol import ProtocolError
from .constants import (
    SUPPORT_CROSSFIT_FOLD_COUNT,
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
    RoutePosteriorEnsemble,
    SupportFoldPlan,
    TargetLocalPosteriorModel,
)


def build_support_fold_plans(
    fingerprint: PhysicalFingerprintSurface,
    *,
    held_case_id: object,
) -> tuple[SupportFoldPlan, ...]:
    """Partition H-c into five nonempty label-free whole-case folds."""

    held = str(held_case_id)
    support_cases = tuple(case for case in fingerprint.cases if case != held)
    if held not in fingerprint.cases or len(support_cases) < SUPPORT_CROSSFIT_FOLD_COUNT:
        raise ProtocolError("PUMR route lacks five support cases for grouped cross-fit.")
    ordered = tuple(
        sorted(
            support_cases,
            key=lambda case: (
                canonical_hash(
                    {
                        "schema_version": "fixed_bank_pumr_fold_order_v1",
                        "center": fingerprint.center,
                        "held_case_id": held,
                        "support_case_id": case,
                    }
                ),
                case,
            ),
        )
    )
    validation_by_fold = tuple(
        tuple(sorted(ordered[index::SUPPORT_CROSSFIT_FOLD_COUNT]))
        for index in range(SUPPORT_CROSSFIT_FOLD_COUNT)
    )
    if any(not cases for cases in validation_by_fold):
        raise ProtocolError("PUMR support cross-fit contains an empty fold.")
    plans = tuple(
        SupportFoldPlan(
            fingerprint.center,
            held,
            fold_id,
            tuple(case for case in support_cases if case not in validation_cases),
            validation_cases,
            fingerprint.fingerprint_hash,
        )
        for fold_id, validation_cases in enumerate(validation_by_fold)
    )
    validation_union = {
        case for plan in plans for case in plan.validation_case_ids
    }
    if (
        validation_union != set(support_cases)
        or sum(len(plan.validation_case_ids) for plan in plans)
        != len(support_cases)
        or any(
            set(plan.training_case_ids) | set(plan.validation_case_ids)
            != set(support_cases)
            for plan in plans
        )
    ):
        raise ProtocolError("PUMR support-fold partition is not exact H-c.")
    return plans


def fit_fold_posterior(
    fingerprint: PhysicalFingerprintSurface,
    plan: SupportFoldPlan,
    *,
    support_labels: Sequence[BinaryLabel],
) -> TargetLocalPosteriorModel:
    """Fit one balanced logistic posterior without c or the validation fold."""

    labels = tuple(support_labels)
    label_by_key = {row.key: row for row in labels}
    expected_scope = (
        f"outer_support::H={fingerprint.center}::excluded_c={plan.held_case_id}"
    )
    expected_support_positions = np.flatnonzero(
        np.asarray(fingerprint.case_ids, dtype=object) != plan.held_case_id
    )
    expected_support_keys = {
        (
            fingerprint.center,
            fingerprint.case_ids[index],
            fingerprint.sample_ids[index],
        )
        for index in expected_support_positions
    }
    if (
        plan.target_center != fingerprint.center
        or plan.fingerprint_hash != fingerprint.fingerprint_hash
        or len(label_by_key) != len(labels)
        or set(label_by_key) != expected_support_keys
        or any(row.scope != expected_scope for row in labels)
        or any(row.case_id == plan.held_case_id for row in labels)
    ):
        raise ProtocolError("PUMR fold posterior escaped its exact H-c capability.")

    training_cases = set(plan.training_case_ids)
    positions = np.asarray(
        [
            index
            for index in expected_support_positions
            if fingerprint.case_ids[index] in training_cases
        ],
        dtype=np.int64,
    )
    expected_training_keys = {
        (
            fingerprint.center,
            fingerprint.case_ids[index],
            fingerprint.sample_ids[index],
        )
        for index in positions
    }
    if not positions.size or {
        fingerprint.case_ids[index] for index in positions
    } != training_cases:
        raise ProtocolError("PUMR fold posterior lacks its exact training cases.")
    x = fingerprint.feature_values[positions]
    y = np.asarray(
        [
            label_by_key[
                (
                    fingerprint.center,
                    fingerprint.case_ids[index],
                    fingerprint.sample_ids[index],
                )
            ].value
            for index in positions
        ],
        dtype=np.int64,
    )
    n_positive = int(np.sum(y == 1, dtype=np.int64))
    n_negative = int(np.sum(y == 0, dtype=np.int64))
    if min(n_positive, n_negative) <= 0:
        raise ProtocolError("PUMR support-fold training must contain both classes.")

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
        raise ProtocolError("PUMR support-fold posterior failed to converge.")
    iterations = int(estimator.n_iter_[0])
    if iterations >= TARGET_POSTERIOR_MAX_ITER:
        raise ProtocolError("PUMR support-fold posterior exhausted max_iter.")
    return TargetLocalPosteriorModel(
        fingerprint.center,
        plan.held_case_id,
        plan.fold_id,
        plan.training_case_ids,
        plan.validation_case_ids,
        fingerprint.feature_names,
        tuple(float(value) for value in feature_mean),
        tuple(float(value) for value in feature_scale),
        tuple(float(value) for value in estimator.coef_[0]),
        float(estimator.intercept_[0]),
        len(y),
        n_positive,
        n_negative,
        fingerprint.fingerprint_hash,
        canonical_hash([list(key) for key in sorted(expected_training_keys)]),
        plan.plan_hash,
        iterations,
        True,
    )


def predict_case_posterior(
    model: TargetLocalPosteriorModel,
    fingerprint: PhysicalFingerprintSurface,
    *,
    case_id: object,
    prediction_role: str,
) -> CasePosteriorPrediction:
    """Predict one excluded case and undo balanced-prior distortion."""

    case = str(case_id)
    if (
        model.target_center != fingerprint.center
        or model.fingerprint_hash != fingerprint.fingerprint_hash
        or model.feature_names != fingerprint.feature_names
        or (
            prediction_role == "HELD_ROUTE"
            and case != model.held_case_id
        )
        or (
            prediction_role == "SUPPORT_OOF"
            and case not in model.validation_case_ids
        )
        or case in model.training_case_ids
    ):
        raise ProtocolError("PUMR posterior prediction escaped an excluded case.")
    positions = fingerprint.positions(case)
    x = fingerprint.feature_values[positions]
    standardized = (
        x - np.asarray(model.feature_mean, dtype=np.float64)
    ) / np.asarray(model.feature_scale, dtype=np.float64)
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
    return CasePosteriorPrediction(
        model.target_center,
        model.held_case_id,
        case,
        model.fold_id,
        prediction_role,
        tuple(fingerprint.sample_ids[index] for index in positions),
        tuple(float(value) for value in balanced),
        tuple(float(value) for value in natural),
        model.model_hash,
        fingerprint.fingerprint_hash,
    )


def fit_route_posterior_ensemble(
    fingerprint: PhysicalFingerprintSurface,
    *,
    held_case_id: object,
    support_labels: Sequence[BinaryLabel],
) -> tuple[
    tuple[TargetLocalPosteriorModel, ...],
    tuple[CasePosteriorPrediction, ...],
    RoutePosteriorEnsemble,
]:
    """Fit five support folds and return held predictions plus OOF reliability."""

    held = str(held_case_id)
    labels = tuple(support_labels)
    plans = build_support_fold_plans(fingerprint, held_case_id=held)
    models: list[TargetLocalPosteriorModel] = []
    held_predictions: list[CasePosteriorPrediction] = []
    oof_predictions: list[CasePosteriorPrediction] = []
    for plan in plans:
        model = fit_fold_posterior(
            fingerprint, plan, support_labels=labels
        )
        models.append(model)
        held_predictions.append(
            predict_case_posterior(
                model,
                fingerprint,
                case_id=held,
                prediction_role="HELD_ROUTE",
            )
        )
        oof_predictions.extend(
            predict_case_posterior(
                model,
                fingerprint,
                case_id=case,
                prediction_role="SUPPORT_OOF",
            )
            for case in plan.validation_case_ids
        )

    label_by_key = {row.key: row.value for row in labels}
    oof_by_key = {
        (prediction.target_center, prediction.predicted_case_id, sample_id): value
        for prediction in oof_predictions
        for sample_id, value in zip(
            prediction.sample_ids,
            prediction.natural_probabilities,
            strict=True,
        )
    }
    if set(oof_by_key) != set(label_by_key) or len(oof_by_key) != len(label_by_key):
        raise ProtocolError("PUMR OOF posterior does not cover exact H-c once.")
    ordered_keys = tuple(sorted(label_by_key))
    y = np.asarray([label_by_key[key] for key in ordered_keys], dtype=np.int8)
    probabilities = np.asarray([oof_by_key[key] for key in ordered_keys], dtype=np.float64)
    n_positive = int(np.sum(y == 1, dtype=np.int64))
    n_negative = int(np.sum(y == 0, dtype=np.int64))
    if min(n_positive, n_negative) <= 0:
        raise ProtocolError("PUMR route support must contain both classes.")
    prevalence = n_positive / len(y)
    brier = float(np.mean((probabilities - y) ** 2, dtype=np.float64))
    prevalence_brier = float(prevalence * (1.0 - prevalence))
    held_samples = held_predictions[0].sample_ids
    if any(row.sample_ids != held_samples for row in held_predictions):
        raise ProtocolError("PUMR held posterior fold predictions are misaligned.")
    ensemble = RoutePosteriorEnsemble(
        fingerprint.center,
        held,
        fingerprint.control_id,
        tuple(plan.plan_hash for plan in plans),
        tuple(model.model_hash for model in models),
        tuple(row.prediction_hash for row in held_predictions),
        held_samples,
        tuple(row.natural_probabilities for row in held_predictions),
        len(y),
        n_positive,
        n_negative,
        len(y),
        _auc(y, probabilities),
        brier,
        prevalence_brier,
        prevalence_brier - brier,
        canonical_hash([list(key) for key in ordered_keys]),
        canonical_hash(
            {
                "keys": [list(key) for key in ordered_keys],
                "probabilities": probabilities.tolist(),
            }
        ),
    )
    return tuple(models), tuple(held_predictions), ensemble


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = int(np.sum(labels == 1, dtype=np.int64))
    negative = int(np.sum(labels == 0, dtype=np.int64))
    if min(positive, negative) <= 0:
        raise ProtocolError("PUMR OOF AUC requires both classes.")
    ranks = _rank(scores) + 1.0
    return float(
        (
            np.sum(ranks[labels == 1], dtype=np.float64)
            - positive * (positive + 1) / 2
        )
        / (positive * negative)
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


__all__ = (
    "build_support_fold_plans",
    "fit_fold_posterior",
    "fit_route_posterior_ensemble",
    "predict_case_posterior",
)
