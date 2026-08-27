"""Deterministic fixed-ridge whole-group OOF local residual models."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..identity import RIDGE_ALPHA
from ..influence.contracts import ActionDescriptor, ActionMetricVector
from ..protocol import ProtocolError
from ..support_folds import SupportFoldPlan, fold_index_for_member
from .contracts import (
    LocalCrossfitResult,
    LocalResidualModel,
    LocalResidualRecord,
    OOFResidualPrediction,
)


SCALE_FLOOR = 1.0e-12
PINV_RCOND = 1.0e-12


def _ordered_records(records: object) -> tuple[LocalResidualRecord, ...]:
    rows = tuple(records)  # type: ignore[arg-type]
    if not rows or any(not isinstance(row, LocalResidualRecord) for row in rows):
        raise ProtocolError("SCALE-BP local-residual training rows drifted.")
    ordered = tuple(sorted(rows, key=lambda row: (row.member_id, row.action_id, row.record_hash)))
    if len({row.record_hash for row in ordered}) != len(ordered):
        raise ProtocolError("SCALE-BP local-residual record is duplicated.")
    if len({row.center_id for row in ordered}) != 1:
        raise ProtocolError("SCALE-BP local residuals crossed target centers.")
    if len({row.route_scope_hash for row in ordered}) != 1:
        raise ProtocolError("SCALE-BP local residuals crossed route scopes.")
    feature_names = ordered[0].descriptor.feature_names
    if any(row.descriptor.feature_names != feature_names for row in ordered):
        raise ProtocolError("SCALE-BP local-residual feature identity drifted.")
    return ordered


def _case_balanced_weights(rows: tuple[LocalResidualRecord, ...]) -> np.ndarray:
    counts = Counter(row.member_id for row in rows)
    n_members = len(counts)
    weights = np.ascontiguousarray(
        [1.0 / (n_members * counts[row.member_id]) for row in rows], dtype=np.float64
    )
    if not np.isclose(np.sum(weights, dtype=np.float64), 1.0, atol=1.0e-12, rtol=0.0):
        raise ProtocolError("SCALE-BP case-balanced ridge weights drifted.")
    return weights


def fit_local_residual_model(
    records: object,
    *,
    excluded_fold_index: int | None = None,
    ridge_alpha: float = RIDGE_ALPHA,
) -> LocalResidualModel:
    """Fit three residual outputs with fold-training-only standardization."""

    rows = _ordered_records(records)
    alpha = float(ridge_alpha)
    if alpha != RIDGE_ALPHA:
        raise ProtocolError("SCALE-BP local ridge alpha is not the frozen value.")
    x = np.ascontiguousarray([row.descriptor.values for row in rows], dtype=np.float64)
    y = np.ascontiguousarray(
        [row.observed_residual.as_tuple() for row in rows], dtype=np.float64
    )
    if x.ndim != 2 or y.shape != (len(rows), 3) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ProtocolError("SCALE-BP local ridge matrix drifted.")
    weights = _case_balanced_weights(rows)
    mean = np.sum(weights[:, None] * x, axis=0, dtype=np.float64)
    variance = np.sum(weights[:, None] * (x - mean) ** 2, axis=0, dtype=np.float64)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(scale > SCALE_FLOOR, scale, 1.0)
    standardized = np.ascontiguousarray((x - mean) / scale, dtype=np.float64)
    design = np.ascontiguousarray(
        np.column_stack((np.ones(len(rows), dtype=np.float64), standardized)),
        dtype=np.float64,
    )
    penalty = np.diag(np.asarray([0.0, *([alpha] * x.shape[1])], dtype=np.float64))
    system = design.T @ (weights[:, None] * design) + penalty
    target = design.T @ (weights[:, None] * y)
    try:
        fitted = np.linalg.solve(system, target)
        solver = "solve"
    except np.linalg.LinAlgError:
        fitted = np.linalg.pinv(system, rcond=PINV_RCOND) @ target
        solver = "pinv"
    if fitted.shape != (x.shape[1] + 1, 3) or not np.isfinite(fitted).all():
        raise ProtocolError("SCALE-BP local ridge produced invalid coefficients.")
    return LocalResidualModel(
        route_scope_hash=rows[0].route_scope_hash,
        excluded_fold_index=excluded_fold_index,
        feature_names=rows[0].descriptor.feature_names,
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        intercepts=ActionMetricVector.from_iterable(fitted[0, :]),
        coefficients=tuple(
            tuple(float(value) for value in fitted[1:, metric_index])
            for metric_index in range(3)
        ),
        training_member_ids=tuple(sorted({row.member_id for row in rows})),
        training_group_ids=tuple(sorted({row.group_id for row in rows})),
        training_patient_ids=tuple(sorted({row.patient_id for row in rows})),
        training_slide_ids=tuple(sorted({row.slide_id for row in rows})),
        training_record_hashes=tuple(sorted(row.record_hash for row in rows)),
        ridge_alpha=alpha,
        solver=solver,
    )


def predict_local_residual(
    model: LocalResidualModel,
    descriptor: ActionDescriptor,
) -> ActionMetricVector:
    if descriptor.feature_names != model.feature_names:
        raise ProtocolError("SCALE-BP local residual descriptor identity drifted.")
    values = np.asarray(descriptor.values, dtype=np.float64)
    mean = np.asarray(model.feature_mean, dtype=np.float64)
    scale = np.asarray(model.feature_scale, dtype=np.float64)
    coefficients = np.asarray(model.coefficients, dtype=np.float64)
    standardized = (values - mean) / scale
    predicted = np.asarray(model.intercepts.as_tuple(), dtype=np.float64) + (
        coefficients @ standardized
    )
    if predicted.shape != (3,) or not np.isfinite(predicted).all():
        raise ProtocolError("SCALE-BP local residual replay produced invalid output.")
    return ActionMetricVector.from_iterable(predicted)


def _validate_plan_records(
    rows: tuple[LocalResidualRecord, ...], plan: SupportFoldPlan
) -> None:
    members = {member.member_id: member for member in plan.members}
    if {row.member_id for row in rows} != set(members):
        raise ProtocolError("SCALE-BP residual population does not equal the fold plan.")
    for row in rows:
        member = members[row.member_id]
        if (
            row.center_id != plan.held_center
            or row.center_id != member.center_id
            or row.case_id != member.case_id
            or row.group_id != member.group_id
            or row.patient_id != member.patient_id
            or row.slide_id != member.slide_id
            or row.route_scope_hash != plan.route_scope_hash
            or row.support_member_hash != member.member_hash
            or row.case_id == plan.held_case_id
            or row.group_id == plan.held_group_id
            or row.patient_id == plan.held_patient_id
            or row.slide_id == plan.held_slide_id
        ):
            raise ProtocolError("SCALE-BP residual row escaped its H\\c support identity.")
        fold = plan.folds[fold_index_for_member(plan, row.member_id)]
        if row.group_id not in fold.group_ids:
            raise ProtocolError("SCALE-BP residual group crossed an OOF boundary.")
        if row.patient_id not in fold.patient_ids or row.slide_id not in fold.slide_ids:
            raise ProtocolError("SCALE-BP residual patient/slide crossed an OOF boundary.")


def crossfit_local_residuals(
    records: object,
    plan: SupportFoldPlan,
    *,
    ridge_alpha: float = RIDGE_ALPHA,
) -> LocalCrossfitResult:
    """Produce case/group-block OOF residual predictions for all support actions."""

    rows = _ordered_records(records)
    if float(ridge_alpha) != RIDGE_ALPHA:
        raise ProtocolError("SCALE-BP local crossfit hyperparameter drifted.")
    _validate_plan_records(rows, plan)
    predictions: list[OOFResidualPrediction] = []
    models: list[LocalResidualModel] = []
    for fold in plan.folds:
        scored_member_ids = set(fold.member_ids)
        scored_group_ids = set(fold.group_ids)
        scored_patient_ids = set(fold.patient_ids)
        scored_slide_ids = set(fold.slide_ids)
        training = tuple(row for row in rows if row.member_id not in scored_member_ids)
        scoring = tuple(row for row in rows if row.member_id in scored_member_ids)
        if (
            not training
            or not scoring
            or any(row.group_id in scored_group_ids for row in training)
            or any(row.patient_id in scored_patient_ids for row in training)
            or any(row.slide_id in scored_slide_ids for row in training)
            or any(row.group_id not in scored_group_ids for row in scoring)
            or any(row.patient_id not in scored_patient_ids for row in scoring)
            or any(row.slide_id not in scored_slide_ids for row in scoring)
        ):
            raise ProtocolError("SCALE-BP whole-group OOF partition drifted.")
        model = fit_local_residual_model(
            training,
            excluded_fold_index=fold.fold_index,
            ridge_alpha=ridge_alpha,
        )
        if scored_member_ids.intersection(model.training_member_ids) or scored_group_ids.intersection(
            model.training_group_ids
        ) or scored_patient_ids.intersection(model.training_patient_ids) or scored_slide_ids.intersection(
            model.training_slide_ids
        ):
            raise ProtocolError("SCALE-BP scored fold entered its local residual model.")
        models.append(model)
        for row in scoring:
            predictions.append(
                OOFResidualPrediction(
                    route_scope_hash=row.route_scope_hash,
                    record_hash=row.record_hash,
                    member_id=row.member_id,
                    case_id=row.case_id,
                    group_id=row.group_id,
                    patient_id=row.patient_id,
                    slide_id=row.slide_id,
                    action_id=row.action_id,
                    fold_index=fold.fold_index,
                    model_hash=model.model_hash,
                    training_member_ids=model.training_member_ids,
                    training_group_ids=model.training_group_ids,
                    training_patient_ids=model.training_patient_ids,
                    training_slide_ids=model.training_slide_ids,
                    predicted_residual=predict_local_residual(model, row.descriptor),
                    observed_residual=row.observed_residual,
                )
            )
    ordered_predictions = tuple(
        sorted(
            predictions,
            key=lambda row: (row.fold_index, row.member_id, row.action_id, row.record_hash),
        )
    )
    if {row.record_hash for row in ordered_predictions} != {row.record_hash for row in rows}:
        raise ProtocolError("SCALE-BP crossfit did not score every support action once.")
    return LocalCrossfitResult(
        plan.route_scope_hash,
        plan.plan_hash,
        ordered_predictions,
        tuple(models),
    )


__all__ = (
    "PINV_RCOND",
    "SCALE_FLOOR",
    "crossfit_local_residuals",
    "fit_local_residual_model",
    "predict_local_residual",
)
