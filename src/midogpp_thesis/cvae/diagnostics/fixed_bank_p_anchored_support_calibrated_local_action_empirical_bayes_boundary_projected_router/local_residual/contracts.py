"""Primitive contracts for case-block OOF target-local residual correction."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ..hashing import canonical_hash
from ..identity import RIDGE_ALPHA, SUPPORT_FOLD_COUNT
from ..influence.contracts import ActionDescriptor, ActionMetricVector, require_sha256
from ..protocol import ProtocolError


@dataclass(frozen=True, slots=True)
class LocalResidualRecord:
    """One support-only case/action aggregate; raw labels never persist."""

    member_id: str
    center_id: str
    case_id: str
    group_id: str
    patient_id: str
    slide_id: str
    route_scope_hash: str
    support_member_hash: str
    descriptor: ActionDescriptor
    donor_metrics: ActionMetricVector
    realized_metrics: ActionMetricVector
    record_hash: str = field(init=False)

    def __post_init__(self) -> None:
        member_id = str(self.member_id)
        center_id = str(self.center_id)
        case_id = str(self.case_id)
        group_id = str(self.group_id)
        patient_id = str(self.patient_id)
        slide_id = str(self.slide_id)
        route_scope_hash = require_sha256(
            self.route_scope_hash, "route-scope hash"
        )
        support_member_hash = require_sha256(
            self.support_member_hash, "support-member hash"
        )
        if (
            not member_id
            or not center_id
            or not case_id
            or not group_id
            or not patient_id
            or not slide_id
            or self.descriptor.case_id != case_id
        ):
            raise ProtocolError("SCALE-BP local-residual record identity drifted.")
        payload = {
            "schema_version": "scale_bp_local_residual_record_v1",
            "member_id": member_id,
            "center_id": center_id,
            "case_id": case_id,
            "group_id": group_id,
            "patient_id": patient_id,
            "slide_id": slide_id,
            "route_scope_hash": route_scope_hash,
            "support_member_hash": support_member_hash,
            "action_id": self.descriptor.action_id,
            "descriptor_hash": self.descriptor.descriptor_hash,
            "donor_metrics": self.donor_metrics.to_payload(),
            "realized_metrics": self.realized_metrics.to_payload(),
            "raw_labels_persisted": False,
        }
        object.__setattr__(self, "member_id", member_id)
        object.__setattr__(self, "center_id", center_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "patient_id", patient_id)
        object.__setattr__(self, "slide_id", slide_id)
        object.__setattr__(self, "route_scope_hash", route_scope_hash)
        object.__setattr__(self, "support_member_hash", support_member_hash)
        object.__setattr__(self, "record_hash", canonical_hash(payload))

    @property
    def action_id(self) -> str:
        return self.descriptor.action_id

    @property
    def observed_residual(self) -> ActionMetricVector:
        return self.realized_metrics.minus(self.donor_metrics)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_local_residual_record_v1",
            "member_id": self.member_id,
            "center_id": self.center_id,
            "case_id": self.case_id,
            "group_id": self.group_id,
            "patient_id": self.patient_id,
            "slide_id": self.slide_id,
            "route_scope_hash": self.route_scope_hash,
            "support_member_hash": self.support_member_hash,
            "action_id": self.action_id,
            "descriptor_hash": self.descriptor.descriptor_hash,
            "donor_metrics": self.donor_metrics.to_payload(),
            "realized_metrics": self.realized_metrics.to_payload(),
            "raw_labels_persisted": False,
            "record_hash": self.record_hash,
        }


@dataclass(frozen=True, slots=True)
class LocalResidualModel:
    """Serialized three-output ridge with training-only feature scaling."""

    route_scope_hash: str
    excluded_fold_index: int | None
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    intercepts: ActionMetricVector
    coefficients: tuple[tuple[float, ...], ...]
    training_member_ids: tuple[str, ...]
    training_group_ids: tuple[str, ...]
    training_patient_ids: tuple[str, ...]
    training_slide_ids: tuple[str, ...]
    training_record_hashes: tuple[str, ...]
    ridge_alpha: float
    solver: str
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        excluded = None if self.excluded_fold_index is None else int(self.excluded_fold_index)
        route_scope_hash = require_sha256(
            self.route_scope_hash, "local-residual route-scope hash"
        )
        names = tuple(str(value) for value in self.feature_names)
        mean = tuple(float(value) for value in self.feature_mean)
        scale = tuple(float(value) for value in self.feature_scale)
        coefficients = tuple(tuple(float(value) for value in row) for row in self.coefficients)
        member_ids = tuple(str(value) for value in self.training_member_ids)
        group_ids = tuple(str(value) for value in self.training_group_ids)
        patient_ids = tuple(str(value) for value in self.training_patient_ids)
        slide_ids = tuple(str(value) for value in self.training_slide_ids)
        record_hashes = tuple(str(value) for value in self.training_record_hashes)
        alpha = float(self.ridge_alpha)
        if (
            (excluded is not None and excluded not in range(SUPPORT_FOLD_COUNT))
            or not names
            or len(names) != len(set(names))
            or len(mean) != len(names)
            or len(scale) != len(names)
            or not all(math.isfinite(value) for value in mean)
            or not all(math.isfinite(value) and value > 0.0 for value in scale)
            or len(coefficients) != 3
            or any(len(row) != len(names) for row in coefficients)
            or not all(math.isfinite(value) for row in coefficients for value in row)
            or not member_ids
            or member_ids != tuple(sorted(set(member_ids)))
            or not group_ids
            or group_ids != tuple(sorted(set(group_ids)))
            or not patient_ids
            or patient_ids != tuple(sorted(set(patient_ids)))
            or not slide_ids
            or slide_ids != tuple(sorted(set(slide_ids)))
            or not record_hashes
            or record_hashes != tuple(sorted(set(record_hashes)))
            or not math.isclose(alpha, RIDGE_ALPHA, rel_tol=0.0, abs_tol=0.0)
            or self.solver not in {"solve", "pinv"}
        ):
            raise ProtocolError("SCALE-BP local-residual model drifted.")
        for digest in record_hashes:
            require_sha256(digest, "local-residual training-record hash")
        payload = {
            "schema_version": "scale_bp_local_residual_model_v1",
            "route_scope_hash": route_scope_hash,
            "excluded_fold_index": excluded,
            "feature_names": names,
            "feature_mean": mean,
            "feature_scale": scale,
            "intercepts": self.intercepts.to_payload(),
            "coefficients": coefficients,
            "training_member_ids": member_ids,
            "training_group_ids": group_ids,
            "training_patient_ids": patient_ids,
            "training_slide_ids": slide_ids,
            "training_record_hashes": record_hashes,
            "ridge_alpha": alpha,
            "solver": self.solver,
            "training_only_scaling": True,
        }
        object.__setattr__(self, "route_scope_hash", route_scope_hash)
        object.__setattr__(self, "excluded_fold_index", excluded)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "training_member_ids", member_ids)
        object.__setattr__(self, "training_group_ids", group_ids)
        object.__setattr__(self, "training_patient_ids", patient_ids)
        object.__setattr__(self, "training_slide_ids", slide_ids)
        object.__setattr__(self, "training_record_hashes", record_hashes)
        object.__setattr__(self, "ridge_alpha", alpha)
        object.__setattr__(self, "model_hash", canonical_hash(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_local_residual_model_v1",
            "route_scope_hash": self.route_scope_hash,
            "excluded_fold_index": self.excluded_fold_index,
            "feature_names": self.feature_names,
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "intercepts": self.intercepts.to_payload(),
            "coefficients": self.coefficients,
            "training_member_ids": self.training_member_ids,
            "training_group_ids": self.training_group_ids,
            "training_patient_ids": self.training_patient_ids,
            "training_slide_ids": self.training_slide_ids,
            "training_record_hashes": self.training_record_hashes,
            "ridge_alpha": self.ridge_alpha,
            "solver": self.solver,
            "training_only_scaling": True,
            "model_hash": self.model_hash,
        }


@dataclass(frozen=True, slots=True)
class OOFResidualPrediction:
    route_scope_hash: str
    record_hash: str
    member_id: str
    case_id: str
    group_id: str
    patient_id: str
    slide_id: str
    action_id: str
    fold_index: int
    model_hash: str
    training_member_ids: tuple[str, ...]
    training_group_ids: tuple[str, ...]
    training_patient_ids: tuple[str, ...]
    training_slide_ids: tuple[str, ...]
    predicted_residual: ActionMetricVector
    observed_residual: ActionMetricVector
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        route_scope_hash = require_sha256(
            self.route_scope_hash, "OOF route-scope hash"
        )
        record_hash = require_sha256(self.record_hash, "OOF record hash")
        model_hash = require_sha256(self.model_hash, "OOF model hash")
        member_id = str(self.member_id)
        case_id = str(self.case_id)
        group_id = str(self.group_id)
        patient_id = str(self.patient_id)
        slide_id = str(self.slide_id)
        action_id = str(self.action_id)
        fold_index = int(self.fold_index)
        training_ids = tuple(str(value) for value in self.training_member_ids)
        training_groups = tuple(str(value) for value in self.training_group_ids)
        training_patients = tuple(str(value) for value in self.training_patient_ids)
        training_slides = tuple(str(value) for value in self.training_slide_ids)
        if (
            not member_id
            or not case_id
            or not group_id
            or not patient_id
            or not slide_id
            or not action_id
            or fold_index not in range(SUPPORT_FOLD_COUNT)
            or not training_ids
            or training_ids != tuple(sorted(set(training_ids)))
            or member_id in training_ids
            or not training_groups
            or training_groups != tuple(sorted(set(training_groups)))
            or group_id in training_groups
            or not training_patients
            or training_patients != tuple(sorted(set(training_patients)))
            or patient_id in training_patients
            or not training_slides
            or training_slides != tuple(sorted(set(training_slides)))
            or slide_id in training_slides
        ):
            raise ProtocolError("SCALE-BP OOF residual prediction drifted.")
        payload = {
            "schema_version": "scale_bp_oof_residual_prediction_v1",
            "route_scope_hash": route_scope_hash,
            "record_hash": record_hash,
            "member_id": member_id,
            "case_id": case_id,
            "group_id": group_id,
            "patient_id": patient_id,
            "slide_id": slide_id,
            "action_id": action_id,
            "fold_index": fold_index,
            "model_hash": model_hash,
            "training_member_ids": training_ids,
            "training_group_ids": training_groups,
            "training_patient_ids": training_patients,
            "training_slide_ids": training_slides,
            "predicted_residual": self.predicted_residual.to_payload(),
            "observed_residual": self.observed_residual.to_payload(),
            "raw_labels_persisted": False,
        }
        object.__setattr__(self, "route_scope_hash", route_scope_hash)
        object.__setattr__(self, "record_hash", record_hash)
        object.__setattr__(self, "model_hash", model_hash)
        object.__setattr__(self, "member_id", member_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "patient_id", patient_id)
        object.__setattr__(self, "slide_id", slide_id)
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "fold_index", fold_index)
        object.__setattr__(self, "training_member_ids", training_ids)
        object.__setattr__(self, "training_group_ids", training_groups)
        object.__setattr__(self, "training_patient_ids", training_patients)
        object.__setattr__(self, "training_slide_ids", training_slides)
        object.__setattr__(self, "prediction_hash", canonical_hash(payload))

    @property
    def residual_error(self) -> ActionMetricVector:
        return self.observed_residual.minus(self.predicted_residual)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_oof_residual_prediction_v1",
            "route_scope_hash": self.route_scope_hash,
            "record_hash": self.record_hash,
            "member_id": self.member_id,
            "case_id": self.case_id,
            "group_id": self.group_id,
            "patient_id": self.patient_id,
            "slide_id": self.slide_id,
            "action_id": self.action_id,
            "fold_index": self.fold_index,
            "model_hash": self.model_hash,
            "training_member_ids": self.training_member_ids,
            "training_group_ids": self.training_group_ids,
            "training_patient_ids": self.training_patient_ids,
            "training_slide_ids": self.training_slide_ids,
            "predicted_residual": self.predicted_residual.to_payload(),
            "observed_residual": self.observed_residual.to_payload(),
            "raw_labels_persisted": False,
            "prediction_hash": self.prediction_hash,
        }


@dataclass(frozen=True, slots=True)
class LocalCrossfitResult:
    route_scope_hash: str
    plan_hash: str
    predictions: tuple[OOFResidualPrediction, ...]
    fold_models: tuple[LocalResidualModel, ...]
    crossfit_hash: str = field(init=False)

    def __post_init__(self) -> None:
        route_scope_hash = require_sha256(
            self.route_scope_hash, "crossfit route-scope hash"
        )
        plan_hash = require_sha256(self.plan_hash, "crossfit plan hash")
        predictions = tuple(self.predictions)
        models = tuple(self.fold_models)
        if (
            not predictions
            or len({row.record_hash for row in predictions}) != len(predictions)
            or len(models) != SUPPORT_FOLD_COUNT
            or tuple(model.excluded_fold_index for model in models)
            != tuple(range(SUPPORT_FOLD_COUNT))
            or any(model.route_scope_hash != route_scope_hash for model in models)
            or any(row.route_scope_hash != route_scope_hash for row in predictions)
        ):
            raise ProtocolError("SCALE-BP local crossfit result drifted.")
        payload = {
            "schema_version": "scale_bp_local_crossfit_result_v1",
            "route_scope_hash": route_scope_hash,
            "plan_hash": plan_hash,
            "prediction_hashes": tuple(row.prediction_hash for row in predictions),
            "model_hashes": tuple(model.model_hash for model in models),
            "whole_group_oof": True,
            "training_only_scaling": True,
        }
        object.__setattr__(self, "route_scope_hash", route_scope_hash)
        object.__setattr__(self, "plan_hash", plan_hash)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "fold_models", models)
        object.__setattr__(self, "crossfit_hash", canonical_hash(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_local_crossfit_result_v1",
            "route_scope_hash": self.route_scope_hash,
            "plan_hash": self.plan_hash,
            "predictions": tuple(row.to_payload() for row in self.predictions),
            "fold_models": tuple(model.to_payload() for model in self.fold_models),
            "whole_group_oof": True,
            "training_only_scaling": True,
            "crossfit_hash": self.crossfit_hash,
        }


__all__ = (
    "LocalCrossfitResult",
    "LocalResidualModel",
    "LocalResidualRecord",
    "OOFResidualPrediction",
)
