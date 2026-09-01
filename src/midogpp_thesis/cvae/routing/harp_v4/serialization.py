"""Canonical, hash-bound serialization for HARP v4 fits and decisions."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .calibration import (
    ConservativeBounds,
    DonorResidualCalibration,
)
from .compatibility import GeometryAssessment, GeometryCalibration
from .contracts import OUTCOME_NAMES, ActionKind, Comparison, EffectVector, SupportSummary
from .fitting import (
    AlphaFoldScore,
    AlphaSelection,
    DeleteDonorFit,
    HarpV4Fit,
)
from .policy import ActionAudit, CaseRoutingDecision
from .ridge import SharedDesignRidge
from .scoring import ConservativeScore
from .metrics import PRIMARY_ESTIMAND


FIT_SCHEMA = "midogpp_harp_v4_fit_v4"
FIT_COLLECTION_SCHEMA = "midogpp_harp_v4_fit_collection_v4"
DECISION_SCHEMA = "midogpp_harp_v4_case_decision_v2"


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("HARP v4 value is not canonically serializable.") from exc


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require_keys(raw: object, keys: set[str], *, name: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != keys:
        raise ProtocolError(f"Serialized HARP v4 {name} keys drifted.")
    return raw


def _array_payload(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value, dtype="<f8", order="C")
    return {
        "dtype": "<f8",
        "shape": list(array.shape),
        "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def _array_from_payload(raw: object) -> np.ndarray:
    item = _require_keys(raw, {"dtype", "shape", "data_base64"}, name="array")
    if item["dtype"] != "<f8" or not isinstance(item["shape"], list):
        raise ProtocolError("Serialized HARP v4 array dtype or shape drifted.")
    try:
        shape = tuple(int(value) for value in item["shape"])
        data = base64.b64decode(str(item["data_base64"]), validate=True)
        array = np.frombuffer(data, dtype="<f8").reshape(shape).copy()
    except (ValueError, TypeError) as exc:
        raise ProtocolError("Serialized HARP v4 array bytes are malformed.") from exc
    return array


def _effect_payload(value: EffectVector) -> list[float]:
    return list(value.as_tuple())


def _effect_from_payload(raw: object) -> EffectVector:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ProtocolError("Serialized HARP v4 effect vector is malformed.")
    try:
        return EffectVector(*(float(value) for value in raw))
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Serialized HARP v4 effect values are malformed.") from exc


def _model_payload(model: SharedDesignRidge) -> dict[str, object]:
    return {
        "feature_names": list(model.feature_names),
        "query_levels": list(model.query_levels),
        "candidate_levels": list(model.candidate_levels),
        "comparison_levels": list(model.comparison_levels),
        "feature_mean": _array_payload(model.feature_mean),
        "feature_scale": _array_payload(model.feature_scale),
        "coefficients": _array_payload(model.coefficients),
        "normal_inverse": _array_payload(model.normal_inverse),
        "alpha": model.alpha,
        "training_query_ids": list(model.training_query_ids),
        "training_candidate_ids": list(model.training_candidate_ids),
        "training_case_ids": list(model.training_case_ids),
        "excluded_center_ids": list(model.excluded_center_ids),
    }


def _model_from_payload(raw: object) -> SharedDesignRidge:
    keys = {
        "feature_names", "query_levels", "candidate_levels", "comparison_levels",
        "feature_mean", "feature_scale", "coefficients", "normal_inverse", "alpha",
        "training_query_ids", "training_candidate_ids", "training_case_ids",
        "excluded_center_ids",
    }
    item = _require_keys(raw, keys, name="shared ridge")
    try:
        return SharedDesignRidge(
            feature_names=tuple(str(value) for value in item["feature_names"]),
            query_levels=tuple(str(value) for value in item["query_levels"]),
            candidate_levels=tuple(str(value) for value in item["candidate_levels"]),
            comparison_levels=tuple(str(value) for value in item["comparison_levels"]),
            feature_mean=_array_from_payload(item["feature_mean"]),
            feature_scale=_array_from_payload(item["feature_scale"]),
            coefficients=_array_from_payload(item["coefficients"]),
            normal_inverse=_array_from_payload(item["normal_inverse"]),
            alpha=float(item["alpha"]),
            training_query_ids=tuple(str(value) for value in item["training_query_ids"]),
            training_candidate_ids=tuple(str(value) for value in item["training_candidate_ids"]),
            training_case_ids=tuple(str(value) for value in item["training_case_ids"]),
            excluded_center_ids=tuple(str(value) for value in item["excluded_center_ids"]),
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Serialized HARP v4 shared ridge values are malformed.") from exc


def _fold_payload(value: AlphaFoldScore) -> dict[str, object]:
    return {
        "heldout_donor_id": value.heldout_donor_id,
        "alpha": value.alpha,
        "standardized_mse": value.standardized_mse,
        "training_query_ids": list(value.training_query_ids),
        "training_candidate_ids": list(value.training_candidate_ids),
    }


def _fold_from_payload(raw: object) -> AlphaFoldScore:
    item = _require_keys(
        raw,
        {"heldout_donor_id", "alpha", "standardized_mse", "training_query_ids", "training_candidate_ids"},
        name="alpha fold",
    )
    return AlphaFoldScore(
        str(item["heldout_donor_id"]),
        float(item["alpha"]),
        float(item["standardized_mse"]),
        tuple(str(value) for value in item["training_query_ids"]),
        tuple(str(value) for value in item["training_candidate_ids"]),
    )


def _selection_payload(value: AlphaSelection) -> dict[str, object]:
    return {
        "selected_alpha": value.selected_alpha,
        "alpha_grid": list(value.alpha_grid),
        "fold_scores": [_fold_payload(item) for item in value.fold_scores],
    }


def _selection_from_payload(raw: object) -> AlphaSelection:
    item = _require_keys(raw, {"selected_alpha", "alpha_grid", "fold_scores"}, name="alpha selection")
    return AlphaSelection(
        float(item["selected_alpha"]),
        tuple(float(value) for value in item["alpha_grid"]),
        tuple(_fold_from_payload(value) for value in item["fold_scores"]),
    )


def _geometry_calibration_payload(value: GeometryCalibration) -> dict[str, object]:
    return {
        "comparison": value.comparison.value,
        "quantile_level": value.quantile_level,
        "reference_median": value.reference_median,
        "reference_quantile": value.reference_quantile,
        "heldout_raw_leverages": list(value.heldout_raw_leverages),
        "heldout_donor_ids": list(value.heldout_donor_ids),
        "heldout_raw_block_ids": list(value.heldout_raw_block_ids),
        "heldout_block_ids": list(value.heldout_block_ids),
        "heldout_block_donor_ids": list(value.heldout_block_donor_ids),
        "heldout_block_maxima": list(value.heldout_block_maxima),
        "heldout_block_sizes": list(value.heldout_block_sizes),
        "source_donor_ids": list(value.source_donor_ids),
        "calibration_method": value.calibration_method,
        "ensemble_cardinality_rule": value.ensemble_cardinality_rule,
        "formal_conformal_claimed": value.formal_conformal_claimed,
    }


def _geometry_calibration_from_payload(raw: object) -> GeometryCalibration:
    item = _require_keys(
        raw,
        {
            "comparison", "quantile_level", "reference_median",
            "reference_quantile", "heldout_raw_leverages",
            "heldout_donor_ids", "heldout_raw_block_ids",
            "heldout_block_ids", "heldout_block_donor_ids",
            "heldout_block_maxima", "heldout_block_sizes",
            "source_donor_ids", "calibration_method",
            "ensemble_cardinality_rule", "formal_conformal_claimed",
        },
        name="geometry calibration",
    )
    return GeometryCalibration(
        comparison=Comparison(str(item["comparison"])),
        quantile_level=float(item["quantile_level"]),
        reference_median=float(item["reference_median"]),
        reference_quantile=float(item["reference_quantile"]),
        heldout_raw_leverages=tuple(
            float(value) for value in item["heldout_raw_leverages"]
        ),
        heldout_donor_ids=tuple(str(value) for value in item["heldout_donor_ids"]),
        heldout_raw_block_ids=tuple(
            str(value) for value in item["heldout_raw_block_ids"]
        ),
        heldout_block_ids=tuple(str(value) for value in item["heldout_block_ids"]),
        heldout_block_donor_ids=tuple(
            str(value) for value in item["heldout_block_donor_ids"]
        ),
        heldout_block_maxima=tuple(
            float(value) for value in item["heldout_block_maxima"]
        ),
        heldout_block_sizes=tuple(int(value) for value in item["heldout_block_sizes"]),
        source_donor_ids=tuple(str(value) for value in item["source_donor_ids"]),
        calibration_method=str(item["calibration_method"]),
        ensemble_cardinality_rule=str(item["ensemble_cardinality_rule"]),
        formal_conformal_claimed=bool(item["formal_conformal_claimed"]),
    )


def _residual_payload(value: DonorResidualCalibration) -> dict[str, object]:
    return {
        "comparison": value.comparison.value,
        "quantile_level": value.quantile_level,
        "endpoint_scales": list(value.endpoint_scales),
        "joint_harm_quantile": value.joint_harm_quantile,
        "calibration_row_count": value.calibration_row_count,
        "calibration_case_block_count": value.calibration_case_block_count,
        "donor_ids": list(value.donor_ids),
        "donor_case_counts": list(value.donor_case_counts),
        "donor_joint_harm_quantiles": list(value.donor_joint_harm_quantiles),
        "calibration_method": value.calibration_method,
        "finite_sample_rule": value.finite_sample_rule,
    }


def _residual_from_payload(raw: object) -> DonorResidualCalibration:
    item = _require_keys(
        raw,
        {
            "comparison", "quantile_level", "endpoint_scales",
            "joint_harm_quantile", "calibration_row_count",
            "calibration_case_block_count", "donor_ids", "donor_case_counts",
            "donor_joint_harm_quantiles", "calibration_method",
            "finite_sample_rule",
        },
        name="residual calibration",
    )
    return DonorResidualCalibration(
        comparison=Comparison(str(item["comparison"])),
        quantile_level=float(item["quantile_level"]),
        endpoint_scales=tuple(float(value) for value in item["endpoint_scales"]),
        joint_harm_quantile=float(item["joint_harm_quantile"]),
        calibration_row_count=int(item["calibration_row_count"]),
        calibration_case_block_count=int(item["calibration_case_block_count"]),
        donor_ids=tuple(str(value) for value in item["donor_ids"]),
        donor_case_counts=tuple(int(value) for value in item["donor_case_counts"]),
        donor_joint_harm_quantiles=tuple(
            float(value) for value in item["donor_joint_harm_quantiles"]
        ),
        calibration_method=str(item["calibration_method"]),
        finite_sample_rule=str(item["finite_sample_rule"]),
    )


def _support_payload(value: SupportSummary) -> dict[str, object]:
    return {
        "comparison": value.comparison.value,
        "candidate_source_id": value.candidate_source_id,
        "donor_ids": list(value.donor_ids),
        "paired_case_count": value.paired_case_count,
        "class_counts": list(value.class_counts),
    }


def _support_from_payload(raw: object) -> SupportSummary:
    item = _require_keys(raw, {"comparison", "candidate_source_id", "donor_ids", "paired_case_count", "class_counts"}, name="support")
    return SupportSummary(
        comparison=Comparison(str(item["comparison"])),
        candidate_source_id=None if item["candidate_source_id"] is None else str(item["candidate_source_id"]),
        donor_ids=tuple(str(value) for value in item["donor_ids"]),
        paired_case_count=int(item["paired_case_count"]),
        class_counts=tuple(int(value) for value in item["class_counts"]),
    )


def fit_to_payload(fit: HarpV4Fit) -> dict[str, object]:
    if not isinstance(fit, HarpV4Fit):
        raise ProtocolError("HARP v4 serialization requires one typed fit.")
    payload: dict[str, object] = {
        "schema_version": FIT_SCHEMA,
        "outcome_names": list(OUTCOME_NAMES),
        "primary_estimand": PRIMARY_ESTIMAND,
        "outer_target_id": fit.outer_target_id,
        "feature_names": list(fit.feature_names),
        "alpha_selection": _selection_payload(fit.alpha_selection),
        "full_model": _model_payload(fit.full_model),
        "delete_donor_fits": [
            {
                "donor_id": value.donor_id,
                "model": _model_payload(value.model),
                "inner_selection": _selection_payload(value.inner_selection),
            }
            for value in fit.delete_donor_fits
        ],
        "geometry_calibrations": [_geometry_calibration_payload(value) for value in fit.geometry_calibrations],
        "residual_calibrations": [_residual_payload(value) for value in fit.residual_calibrations],
        "support_summaries": [_support_payload(value) for value in fit.support_summaries],
        "residual_quantile": fit.residual_quantile,
        "geometry_quantile": fit.geometry_quantile,
    }
    payload["fit_hash"] = canonical_hash(payload)
    return payload


def fit_from_payload(raw: object) -> HarpV4Fit:
    keys = {
        "schema_version", "outcome_names", "primary_estimand", "outer_target_id",
        "feature_names", "alpha_selection",
        "full_model", "delete_donor_fits", "geometry_calibrations",
        "residual_calibrations", "support_summaries", "residual_quantile",
        "geometry_quantile", "fit_hash",
    }
    item = _require_keys(raw, keys, name="fit")
    unsigned = {key: value for key, value in item.items() if key != "fit_hash"}
    if (
        item["schema_version"] != FIT_SCHEMA
        or item["outcome_names"] != list(OUTCOME_NAMES)
        or item["primary_estimand"] != PRIMARY_ESTIMAND
        or item["fit_hash"] != canonical_hash(unsigned)
    ):
        raise ProtocolError("Serialized HARP v4 fit schema or hash drifted.")
    delete_rows = []
    for value in item["delete_donor_fits"]:
        row = _require_keys(value, {"donor_id", "model", "inner_selection"}, name="delete donor")
        delete_rows.append(
            DeleteDonorFit(
                str(row["donor_id"]),
                _model_from_payload(row["model"]),
                _selection_from_payload(row["inner_selection"]),
            )
        )
    fit = HarpV4Fit(
        outer_target_id=str(item["outer_target_id"]),
        feature_names=tuple(str(value) for value in item["feature_names"]),
        alpha_selection=_selection_from_payload(item["alpha_selection"]),
        full_model=_model_from_payload(item["full_model"]),
        delete_donor_fits=tuple(delete_rows),
        geometry_calibrations=tuple(_geometry_calibration_from_payload(value) for value in item["geometry_calibrations"]),
        residual_calibrations=tuple(_residual_from_payload(value) for value in item["residual_calibrations"]),
        support_summaries=tuple(_support_from_payload(value) for value in item["support_summaries"]),
        residual_quantile=float(item["residual_quantile"]),
        geometry_quantile=float(item["geometry_quantile"]),
    )
    if fit_to_payload(fit)["fit_hash"] != item["fit_hash"]:
        raise ProtocolError("Reconstructed HARP v4 fit changed identity.")
    return fit


def serialize_fit(fit: HarpV4Fit) -> str:
    return canonical_bytes(fit_to_payload(fit)).decode("utf-8")


def deserialize_fit(text: str) -> HarpV4Fit:
    try:
        raw = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Serialized HARP v4 fit is invalid JSON.") from exc
    return fit_from_payload(raw)


def fit_collection_to_payload(fits: Sequence[HarpV4Fit]) -> dict[str, object]:
    values = tuple(sorted(fits, key=lambda value: value.outer_target_id))
    if not values or len({value.outer_target_id for value in values}) != len(values):
        raise ProtocolError("HARP v4 fit collection requires unique outer targets.")
    payload: dict[str, object] = {
        "schema_version": FIT_COLLECTION_SCHEMA,
        "outer_target_ids": [value.outer_target_id for value in values],
        "fits": [fit_to_payload(value) for value in values],
    }
    payload["collection_hash"] = canonical_hash(payload)
    return payload


def fit_collection_from_payload(raw: object) -> tuple[HarpV4Fit, ...]:
    item = _require_keys(raw, {"schema_version", "outer_target_ids", "fits", "collection_hash"}, name="fit collection")
    unsigned = {key: value for key, value in item.items() if key != "collection_hash"}
    if item["schema_version"] != FIT_COLLECTION_SCHEMA or item["collection_hash"] != canonical_hash(unsigned):
        raise ProtocolError("Serialized HARP v4 fit collection drifted.")
    fits = tuple(fit_from_payload(value) for value in item["fits"])
    if [value.outer_target_id for value in fits] != item["outer_target_ids"]:
        raise ProtocolError("HARP v4 fit collection target order drifted.")
    return fits


def serialize_fit_collection(fits: Sequence[HarpV4Fit]) -> str:
    return canonical_bytes(fit_collection_to_payload(fits)).decode("utf-8")


def deserialize_fit_collection(text: str) -> tuple[HarpV4Fit, ...]:
    try:
        raw = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Serialized HARP v4 fit collection is invalid JSON.") from exc
    return fit_collection_from_payload(raw)


def _bounds_payload(value: ConservativeBounds) -> dict[str, object]:
    return {
        "prediction_center": _effect_payload(value.prediction_center),
        "case_equal_bacc_contribution_gain_lower": (
            value.case_equal_bacc_contribution_gain_lower
        ),
        "brier_upper": value.brier_upper,
        "log_loss_upper": value.log_loss_upper,
        "calibration_method": value.calibration_method,
    }


def _bounds_from_payload(raw: object) -> ConservativeBounds:
    item = _require_keys(
        raw,
        {
            "prediction_center", "case_equal_bacc_contribution_gain_lower",
            "brier_upper", "log_loss_upper", "calibration_method",
        },
        name="bounds",
    )
    return ConservativeBounds(
        _effect_from_payload(item["prediction_center"]),
        float(item["case_equal_bacc_contribution_gain_lower"]),
        float(item["brier_upper"]),
        float(item["log_loss_upper"]),
        str(item["calibration_method"]),
    )


def _assessment_payload(value: GeometryAssessment) -> dict[str, object]:
    return {
        "raw_leverages": list(value.raw_leverages),
        "calibrated_ratios": list(value.calibrated_ratios),
        "maximum_ratio": value.maximum_ratio,
        "empirical_percentile": value.empirical_percentile,
        "empirical_tail_probability": value.empirical_tail_probability,
        "finite_sample_tail_floor": value.finite_sample_tail_floor,
        "calibration_block_count": value.calibration_block_count,
        "compatibility_shrinkage": value.compatibility_shrinkage,
        "reference_median": value.reference_median,
        "reference_quantile": value.reference_quantile,
        "quantile_level": value.quantile_level,
        "calibration_method": value.calibration_method,
        "ensemble_cardinality_rule": value.ensemble_cardinality_rule,
        "formal_conformal_claimed": value.formal_conformal_claimed,
    }


def _assessment_from_payload(raw: object) -> GeometryAssessment:
    item = _require_keys(
        raw,
        {
            "raw_leverages", "calibrated_ratios", "maximum_ratio",
            "empirical_percentile", "empirical_tail_probability",
            "finite_sample_tail_floor", "calibration_block_count",
            "compatibility_shrinkage", "reference_median",
            "reference_quantile", "quantile_level", "calibration_method",
            "ensemble_cardinality_rule", "formal_conformal_claimed",
        },
        name="geometry assessment",
    )
    return GeometryAssessment(
        raw_leverages=tuple(float(value) for value in item["raw_leverages"]),
        calibrated_ratios=tuple(
            float(value) for value in item["calibrated_ratios"]
        ),
        maximum_ratio=float(item["maximum_ratio"]),
        empirical_percentile=float(item["empirical_percentile"]),
        empirical_tail_probability=float(item["empirical_tail_probability"]),
        finite_sample_tail_floor=float(item["finite_sample_tail_floor"]),
        calibration_block_count=int(item["calibration_block_count"]),
        compatibility_shrinkage=float(item["compatibility_shrinkage"]),
        reference_median=float(item["reference_median"]),
        reference_quantile=float(item["reference_quantile"]),
        quantile_level=float(item["quantile_level"]),
        calibration_method=str(item["calibration_method"]),
        ensemble_cardinality_rule=str(item["ensemble_cardinality_rule"]),
        formal_conformal_claimed=bool(item["formal_conformal_claimed"]),
    )


def _score_payload(value: ConservativeScore) -> dict[str, object]:
    return {
        "action_id": value.action_id,
        "comparison": value.comparison.value,
        "delete_donor_ids": list(value.delete_donor_ids),
        "donor_predictions": [_effect_payload(item) for item in value.donor_predictions],
        "geometry": _assessment_payload(value.geometry),
        "support": _support_payload(value.support),
        "source_only_bounds": _bounds_payload(value.source_only_bounds),
        "geometry_adjusted_bounds": _bounds_payload(value.geometry_adjusted_bounds),
        "eligible": value.eligible,
        "rejection_reasons": list(value.rejection_reasons),
    }


def _score_from_payload(raw: object) -> ConservativeScore:
    item = _require_keys(raw, {"action_id", "comparison", "delete_donor_ids", "donor_predictions", "geometry", "support", "source_only_bounds", "geometry_adjusted_bounds", "eligible", "rejection_reasons"}, name="score")
    return ConservativeScore(
        str(item["action_id"]),
        Comparison(str(item["comparison"])),
        tuple(str(value) for value in item["delete_donor_ids"]),
        tuple(_effect_from_payload(value) for value in item["donor_predictions"]),
        _assessment_from_payload(item["geometry"]),
        _support_from_payload(item["support"]),
        _bounds_from_payload(item["source_only_bounds"]),
        _bounds_from_payload(item["geometry_adjusted_bounds"]),
        bool(item["eligible"]),
        tuple(str(value) for value in item["rejection_reasons"]),
    )


def _audit_payload(value: ActionAudit) -> dict[str, object]:
    return {
        "action_id": value.action_id,
        "action_kind": value.action_kind.value,
        "candidate_source_id": value.candidate_source_id,
        "comparison_scores": [_score_payload(item) for item in value.comparison_scores],
        "eligible": value.eligible,
        "rejection_reasons": list(value.rejection_reasons),
    }


def _audit_from_payload(raw: object) -> ActionAudit:
    item = _require_keys(raw, {"action_id", "action_kind", "candidate_source_id", "comparison_scores", "eligible", "rejection_reasons"}, name="action audit")
    return ActionAudit(
        str(item["action_id"]),
        ActionKind(str(item["action_kind"])),
        None if item["candidate_source_id"] is None else str(item["candidate_source_id"]),
        tuple(_score_from_payload(value) for value in item["comparison_scores"]),
        bool(item["eligible"]),
        tuple(str(value) for value in item["rejection_reasons"]),
    )


def _bytes_payload(values: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(value).decode("ascii") for value in values]


def _bytes_from_payload(raw: object) -> tuple[bytes, ...]:
    if not isinstance(raw, list):
        raise ProtocolError("Serialized HARP v4 probability bytes are malformed.")
    try:
        values = tuple(base64.b64decode(str(value), validate=True) for value in raw)
    except (ValueError, TypeError) as exc:
        raise ProtocolError("Serialized HARP v4 probability bytes are malformed.") from exc
    if any(len(value) != 4 for value in values):
        raise ProtocolError("Serialized HARP v4 probabilities lost float32 bytes.")
    return values


def decision_to_payload(decision: CaseRoutingDecision) -> dict[str, object]:
    if not isinstance(decision, CaseRoutingDecision):
        raise ProtocolError("HARP v4 serialization requires a typed decision.")
    payload: dict[str, object] = {
        "schema_version": DECISION_SCHEMA,
        "outer_target_id": decision.outer_target_id,
        "case_id": decision.case_id,
        "sample_ids": list(decision.sample_ids),
        "baseline_probability_bytes": _bytes_payload(decision.baseline_probability_bytes),
        "output_probability_bytes": _bytes_payload(decision.output_probability_bytes),
        "selected_kind": decision.selected_kind.value,
        "selected_source_id": decision.selected_source_id,
        "reason": decision.reason,
        "prediction_seal_hash": decision.prediction_seal_hash,
        "action_audits": [_audit_payload(value) for value in decision.action_audits],
    }
    payload["decision_hash"] = canonical_hash(payload)
    return payload


def decision_from_payload(raw: object) -> CaseRoutingDecision:
    keys = {"schema_version", "outer_target_id", "case_id", "sample_ids", "baseline_probability_bytes", "output_probability_bytes", "selected_kind", "selected_source_id", "reason", "prediction_seal_hash", "action_audits", "decision_hash"}
    item = _require_keys(raw, keys, name="decision")
    unsigned = {key: value for key, value in item.items() if key != "decision_hash"}
    if item["schema_version"] != DECISION_SCHEMA or item["decision_hash"] != canonical_hash(unsigned):
        raise ProtocolError("Serialized HARP v4 decision schema or hash drifted.")
    decision = CaseRoutingDecision(
        outer_target_id=str(item["outer_target_id"]),
        case_id=str(item["case_id"]),
        sample_ids=tuple(str(value) for value in item["sample_ids"]),
        baseline_probability_bytes=_bytes_from_payload(item["baseline_probability_bytes"]),
        output_probability_bytes=_bytes_from_payload(item["output_probability_bytes"]),
        selected_kind=ActionKind(str(item["selected_kind"])),
        selected_source_id=None if item["selected_source_id"] is None else str(item["selected_source_id"]),
        reason=str(item["reason"]),
        prediction_seal_hash=str(item["prediction_seal_hash"]),
        action_audits=tuple(_audit_from_payload(value) for value in item["action_audits"]),
    )
    if decision_to_payload(decision)["decision_hash"] != item["decision_hash"]:
        raise ProtocolError("Reconstructed HARP v4 decision changed identity.")
    return decision


def serialize_decision(decision: CaseRoutingDecision) -> str:
    return canonical_bytes(decision_to_payload(decision)).decode("utf-8")


def deserialize_decision(text: str) -> CaseRoutingDecision:
    try:
        raw = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Serialized HARP v4 decision is invalid JSON.") from exc
    return decision_from_payload(raw)


__all__ = (
    "canonical_bytes", "canonical_hash", "decision_from_payload",
    "decision_to_payload", "deserialize_decision", "deserialize_fit",
    "deserialize_fit_collection", "fit_collection_from_payload",
    "fit_collection_to_payload", "fit_from_payload", "fit_to_payload",
    "serialize_decision", "serialize_fit", "serialize_fit_collection",
)
