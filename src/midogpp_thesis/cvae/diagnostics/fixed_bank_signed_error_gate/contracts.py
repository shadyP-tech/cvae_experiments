"""Immutable records for the signed sample-level correction surface."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ...protocol import ProtocolError
from ..fixed_bank_hierarchical_residual_stacker.core_hashing import canonical_hash
from ..fixed_bank_hierarchical_residual_stacker.scientific_constants import (
    MIDOGPP_CENTERS,
)
from .constants import FEATURE_NAMES
from .constants import INTERCEPT_GRID, LAMBDA_GRID


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite.")
    return result


def _identity(center: str, case: str, sample: str) -> None:
    if center not in MIDOGPP_CENTERS:
        raise ProtocolError("Signed-error row uses an unknown MIDOG++ center.")
    if not str(case).strip() or not str(sample).strip():
        raise ProtocolError("Signed-error row identities must be non-empty.")


@dataclass(frozen=True, order=True)
class SignedFeatureRow:
    target_center: str
    case_id: str
    sample_id: str
    values: tuple[float, ...]
    candidate_source_ids: tuple[str, ...]
    context_excluded_centers: tuple[str, ...] = ()
    feature_origin_center: str | None = None
    feature_origin_case_id: str | None = None
    feature_origin_sample_id: str | None = None
    feature_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        _identity(self.target_center, self.case_id, self.sample_id)
        values = tuple(_finite(value, "feature") for value in self.values)
        if len(values) != len(FEATURE_NAMES):
            raise ProtocolError("Signed-error feature width drifted.")
        if values[0] != 1.0:
            raise ProtocolError("Signed-error intercept must be exactly one.")
        candidates = tuple(sorted(set(str(value) for value in self.candidate_source_ids)))
        exclusions = tuple(sorted(set(str(value) for value in self.context_excluded_centers)))
        if (
            not candidates
            or any(value not in MIDOGPP_CENTERS for value in candidates)
            or any(value not in MIDOGPP_CENTERS for value in exclusions)
            or self.target_center in candidates
            or set(candidates).intersection(exclusions)
        ):
            raise ProtocolError("Signed-error candidate/exclusion context is invalid.")
        origin_center = (
            self.target_center
            if self.feature_origin_center is None
            else str(self.feature_origin_center)
        )
        origin_case = (
            self.case_id
            if self.feature_origin_case_id is None
            else str(self.feature_origin_case_id)
        )
        origin_sample = (
            self.sample_id
            if self.feature_origin_sample_id is None
            else str(self.feature_origin_sample_id)
        )
        if (
            origin_center != self.target_center
            or not origin_case
            or not origin_sample
        ):
            raise ProtocolError("Feature-origin sample identity must be non-empty.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "candidate_source_ids", candidates)
        object.__setattr__(self, "context_excluded_centers", exclusions)
        object.__setattr__(self, "feature_origin_center", origin_center)
        object.__setattr__(self, "feature_origin_case_id", origin_case)
        object.__setattr__(self, "feature_origin_sample_id", origin_sample)
        object.__setattr__(self, "feature_hash", canonical_hash(self._unhashed()))

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return (self.target_center, self.case_id, self.sample_id)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_signed_error_feature_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "candidate_source_ids": list(self.candidate_source_ids),
            "context_excluded_centers": list(self.context_excluded_centers),
            "feature_origin_center": self.feature_origin_center,
            "feature_origin_case_id": self.feature_origin_case_id,
            "feature_origin_sample_id": self.feature_origin_sample_id,
            "feature_names": list(FEATURE_NAMES),
            "values": list(self.values),
            "label_free": True,
            "baseline_predicted_class_branch_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "feature_hash": self.feature_hash}


@dataclass(frozen=True, order=True)
class GradientTargetRow:
    target_center: str
    case_id: str
    sample_id: str
    label: int
    baseline_probability: float
    class_balance_weight: float
    negative_log_loss_gradient: float

    def __post_init__(self) -> None:
        _identity(self.target_center, self.case_id, self.sample_id)
        if isinstance(self.label, bool) or self.label not in (0, 1):
            raise ProtocolError("Gradient target label must be binary.")
        probability = _finite(self.baseline_probability, "baseline_probability")
        weight = _finite(self.class_balance_weight, "class_balance_weight")
        gradient = _finite(
            self.negative_log_loss_gradient, "negative_log_loss_gradient"
        )
        if not 0.0 <= probability <= 1.0 or weight <= 0.0:
            raise ProtocolError("Gradient target probability/weight is invalid.")
        object.__setattr__(self, "baseline_probability", probability)
        object.__setattr__(self, "class_balance_weight", weight)
        object.__setattr__(self, "negative_log_loss_gradient", gradient)

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return (self.target_center, self.case_id, self.sample_id)


@dataclass(frozen=True)
class Standardization:
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        means = tuple(_finite(value, "standardization mean") for value in self.means)
        scales = tuple(_finite(value, "standardization scale") for value in self.scales)
        if len(means) != len(FEATURE_NAMES) - 1 or len(scales) != len(
            FEATURE_NAMES
        ) - 1:
            raise ProtocolError("Signed-error standardization width drifted.")
        if any(value <= 0.0 for value in scales):
            raise ProtocolError("Signed-error standardization scales must be positive.")
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "scales", scales)


@dataclass(frozen=True)
class SignedGateModel:
    target_center: str
    family: str
    ridge_alpha: float
    coefficients: tuple[float, ...]
    standardization: Standardization
    donor_centers: tuple[str, ...]
    nested_model_hashes: tuple[str, ...]
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS:
            raise ProtocolError("Signed gate model has an unknown held-out center.")
        if self.family not in ("G", "R", "P"):
            raise ProtocolError("Signed gate model family must be G, R, or P.")
        alpha = _finite(self.ridge_alpha, "ridge_alpha")
        coefficients = tuple(_finite(value, "coefficient") for value in self.coefficients)
        if alpha <= 0.0 or len(coefficients) != len(FEATURE_NAMES):
            raise ProtocolError("Signed gate model parameters are invalid.")
        donors = tuple(self.donor_centers)
        if (
            self.target_center in donors
            or donors != tuple(sorted(set(donors)))
            or any(value not in MIDOGPP_CENTERS for value in donors)
            or len(donors) not in (len(MIDOGPP_CENTERS) - 1, len(MIDOGPP_CENTERS) - 2)
        ):
            raise ProtocolError("Signed gate model donor-center exclusion drifted.")
        hashes = tuple(str(value) for value in self.nested_model_hashes)
        if (
            len(hashes) not in (0, len(MIDOGPP_CENTERS) - 1)
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
        ):
            raise ProtocolError("Signed gate nested-model hash contract drifted.")
        object.__setattr__(self, "ridge_alpha", alpha)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "donor_centers", donors)
        object.__setattr__(self, "nested_model_hashes", hashes)
        unhashed = {
            "schema_version": "fixed_bank_signed_error_model_v1",
            "target_center": self.target_center,
            "family": self.family,
            "ridge_alpha": alpha,
            "coefficients": list(coefficients),
            "means": list(self.standardization.means),
            "scales": list(self.standardization.scales),
            "donor_centers": list(donors),
            "nested_model_hashes": list(hashes),
            "response": "class_balanced_rescaled_negative_log_loss_logit_gradient",
            "ridge_objective": "unweighted_mse_on_rescaled_gradient_target",
            "target_labels_used": False,
        }
        object.__setattr__(self, "model_hash", canonical_hash(unhashed))


@dataclass(frozen=True, order=True)
class CorrectionRow:
    target_center: str
    case_id: str
    sample_id: str
    family: str
    raw_correction: float
    correction_standard_error: float
    safe_correction: float
    uncertainty_admitted: bool
    correction_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        _identity(self.target_center, self.case_id, self.sample_id)
        if self.family not in ("G", "R", "P"):
            raise ProtocolError("Unknown signed-correction family.")
        raw = _finite(self.raw_correction, "raw_correction")
        standard_error = _finite(
            self.correction_standard_error, "correction_standard_error"
        )
        safe = _finite(self.safe_correction, "safe_correction")
        if standard_error < 0.0 or type(self.uncertainty_admitted) is not bool:
            raise ProtocolError("Correction standard error cannot be negative.")
        if (not self.uncertainty_admitted and safe != 0.0) or (
            self.uncertainty_admitted and safe != raw
        ):
            raise ProtocolError("Safe correction does not match its uncertainty gate.")
        object.__setattr__(self, "raw_correction", raw)
        object.__setattr__(self, "correction_standard_error", standard_error)
        object.__setattr__(self, "safe_correction", safe)
        unhashed = {
            "schema_version": "fixed_bank_signed_error_correction_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "family": self.family,
            "raw_correction": raw,
            "correction_standard_error": standard_error,
            "safe_correction": safe,
            "uncertainty_admitted": self.uncertainty_admitted,
        }
        object.__setattr__(self, "correction_hash", canonical_hash(unhashed))

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return (self.target_center, self.case_id, self.sample_id)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_signed_error_correction_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "family": self.family,
            "raw_correction": self.raw_correction,
            "correction_standard_error": self.correction_standard_error,
            "safe_correction": self.safe_correction,
            "uncertainty_admitted": self.uncertainty_admitted,
            "correction_hash": self.correction_hash,
        }


@dataclass(frozen=True)
class LambdaPathRow:
    residual_scale: float
    support_loss: float
    loss_delta_vs_zero: float
    threshold_crossing_count: int

    def __post_init__(self) -> None:
        scale = _finite(self.residual_scale, "residual_scale")
        loss = _finite(self.support_loss, "support_loss")
        delta = _finite(self.loss_delta_vs_zero, "loss_delta_vs_zero")
        if (
            scale not in LAMBDA_GRID
            or loss < 0.0
            or type(self.threshold_crossing_count) is not int
            or self.threshold_crossing_count < 0
        ):
            raise ProtocolError("Signed-error lambda-path row is invalid.")
        object.__setattr__(self, "residual_scale", scale)
        object.__setattr__(self, "support_loss", loss)
        object.__setattr__(self, "loss_delta_vs_zero", delta)

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class SignedGateDecision:
    intercept: float
    proposed_scale: float
    selected_scale: float
    support_bacc_lcb: float
    fallback_reason: str | None
    lambda_path: tuple[LambdaPathRow, ...]

    def __post_init__(self) -> None:
        intercept = _finite(self.intercept, "intercept")
        proposed = _finite(self.proposed_scale, "proposed_scale")
        selected = _finite(self.selected_scale, "selected_scale")
        lower_bound = _finite(self.support_bacc_lcb, "support_bacc_lcb")
        path = tuple(self.lambda_path)
        if (
            intercept not in INTERCEPT_GRID
            or proposed not in LAMBDA_GRID
            or selected not in LAMBDA_GRID
            or tuple(row.residual_scale for row in path) != LAMBDA_GRID
            or (selected == 0.0) != (self.fallback_reason is not None)
            or (self.fallback_reason is not None and not str(self.fallback_reason).strip())
        ):
            raise ProtocolError("Signed-error support decision is invalid.")
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "proposed_scale", proposed)
        object.__setattr__(self, "selected_scale", selected)
        object.__setattr__(self, "support_bacc_lcb", lower_bound)
        object.__setattr__(self, "lambda_path", path)

    def to_payload(self) -> dict[str, object]:
        return {
            "intercept": self.intercept,
            "proposed_scale": self.proposed_scale,
            "selected_scale": self.selected_scale,
            "support_bacc_lcb": self.support_bacc_lcb,
            "fallback_reason": self.fallback_reason,
            "lambda_path": [row.to_payload() for row in self.lambda_path],
        }


__all__ = (
    "CorrectionRow",
    "GradientTargetRow",
    "LambdaPathRow",
    "SignedFeatureRow",
    "SignedGateDecision",
    "SignedGateModel",
    "Standardization",
)
