"""Immutable scientific records for the residual-stacker mechanism audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ...protocol import ProtocolError
from .core_hashing import canonical_hash, finite_float, nonempty_text
from .scientific_constants import (
    BASELINE_ACTION_ID,
    DESIGN_TERMS,
    HARD_THRESHOLD,
    MAX_RESIDUAL_SCALE,
    MIDOGPP_CENTERS,
    MODEL_RANK,
    PHI_NAMES,
    PROBABILITY_EPSILON,
    RIDGE_GRID,
    SOFTMAX_TEMPERATURE,
    SPARSE_SOURCE_BUDGET,
    candidate_sources,
)

LabelScope = Literal["loco_donor", "target_support", "terminal_evaluation"]


def _center(value: object, name: str = "target_center") -> str:
    center = nonempty_text(value, name)
    if center not in MIDOGPP_CENTERS:
        raise ProtocolError(f"{name} is not a locked MIDOG++ center.")
    return center


def _source(target_center: str, value: object) -> str:
    source = nonempty_text(value, "source_id")
    if source not in candidate_sources(target_center):
        if source == target_center:
            raise ProtocolError("The held-out target expert cannot be a candidate source.")
        raise ProtocolError("Candidate source is outside the fixed non-target bank.")
    return source


def _tuple_floats(values: object, *, length: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(finite_float(value, name) for value in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProtocolError(f"{name} must be a numeric sequence.") from exc
    if len(result) != length:
        raise ProtocolError(f"{name} must contain exactly {length} values.")
    return result


@dataclass(frozen=True, order=True)
class SampleActionProbability:
    target_center: str
    case_id: str
    sample_id: str
    action_id: str
    probability: float
    row_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        target = _center(self.target_center)
        nonempty_text(self.case_id, "case_id")
        nonempty_text(self.sample_id, "sample_id")
        action = nonempty_text(self.action_id, "action_id")
        if action not in (BASELINE_ACTION_ID, *candidate_sources(target)):
            raise ProtocolError("Probability row uses an illegal target action.")
        probability = finite_float(self.probability, "probability")
        if not 0.0 <= probability <= 1.0:
            raise ProtocolError("Probability must lie in [0, 1].")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return (self.target_center, self.case_id, self.sample_id)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_probability_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "action_id": self.action_id,
            "probability": self.probability,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True, order=True)
class BinaryLabel:
    target_center: str
    case_id: str
    sample_id: str
    label: int
    label_scope: LabelScope

    def __post_init__(self) -> None:
        _center(self.target_center)
        nonempty_text(self.case_id, "case_id")
        nonempty_text(self.sample_id, "sample_id")
        if isinstance(self.label, bool) or self.label not in (0, 1):
            raise ProtocolError("Binary label must be integer zero or one.")
        if self.label_scope not in ("loco_donor", "target_support", "terminal_evaluation"):
            raise ProtocolError("Unknown label capability scope.")

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return (self.target_center, self.case_id, self.sample_id)


@dataclass(frozen=True, order=True)
class CaseFeatureRow:
    target_center: str
    case_id: str
    source_id: str
    sample_count: int
    phi: tuple[float, float, float, float]
    feature_origin_source_id: str | None = None
    feature_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        target = _center(self.target_center)
        nonempty_text(self.case_id, "case_id")
        _source(target, self.source_id)
        origin = self.source_id if self.feature_origin_source_id is None else str(self.feature_origin_source_id)
        _source(target, origin)
        if isinstance(self.sample_count, bool) or self.sample_count <= 0:
            raise ProtocolError("Case feature sample_count must be positive.")
        values = _tuple_floats(self.phi, length=len(PHI_NAMES), name="phi")
        if values[1] < 0.0 or values[2] < 0.0 or not 0.0 <= values[3] <= 1.0:
            raise ProtocolError("Case residual feature values are outside their domains.")
        object.__setattr__(self, "phi", values)
        object.__setattr__(self, "feature_origin_source_id", origin)
        object.__setattr__(self, "feature_hash", canonical_hash(self._unhashed()))

    @property
    def case_key(self) -> tuple[str, str]:
        return (self.target_center, self.case_id)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_case_phi_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "source_id": self.source_id,
            "feature_origin_source_id": self.feature_origin_source_id,
            "sample_count": self.sample_count,
            "phi_names": list(PHI_NAMES),
            "phi": list(self.phi),
            "label_free": True,
            "case_conditioning": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "feature_hash": self.feature_hash}


@dataclass(frozen=True, order=True)
class SourceControlRow:
    target_center: str
    source_id: str
    excluded_query_center: str | None
    donor_query_centers: tuple[str, ...]
    global_source_control: float
    context_excluded_centers: tuple[str, ...] = ()
    control_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        target = _center(self.target_center)
        source = _source(target, self.source_id)
        excluded = self.excluded_query_center
        if excluded is not None:
            excluded = _center(excluded, "excluded_query_center")
            if excluded in (target, source):
                raise ProtocolError("Nested query exclusion must differ from H and e.")
        context_exclusions = tuple(sorted(set(str(value) for value in self.context_excluded_centers)))
        if any(_center(value, "context_excluded_center") in (target, source) for value in context_exclusions):
            raise ProtocolError("Context exclusions redundantly contain H or the described source.")
        donors = tuple(self.donor_query_centers)
        if not donors or donors != tuple(sorted(set(donors))):
            raise ProtocolError("Source-control donors must be non-empty, unique, and sorted.")
        forbidden = {target, source}
        if excluded is not None:
            forbidden.add(excluded)
        forbidden.update(context_exclusions)
        if any(_center(value, "donor_query_center") in forbidden for value in donors):
            raise ProtocolError("Source control violated the H/e/q exclusion.")
        value = finite_float(self.global_source_control, "global_source_control")
        if value < 0.0:
            raise ProtocolError("Global source control must be nonnegative.")
        object.__setattr__(self, "excluded_query_center", excluded)
        object.__setattr__(self, "donor_query_centers", donors)
        object.__setattr__(self, "global_source_control", value)
        object.__setattr__(self, "context_excluded_centers", context_exclusions)
        object.__setattr__(self, "control_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_source_control_v1",
            "target_center": self.target_center,
            "source_id": self.source_id,
            "excluded_query_center": self.excluded_query_center,
            "donor_query_centers": list(self.donor_query_centers),
            "global_source_control": self.global_source_control,
            "context_excluded_centers": list(self.context_excluded_centers),
            "label_free": True,
            "metadata_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "control_hash": self.control_hash}


@dataclass(frozen=True, order=True)
class DonorResponseRow:
    donor_center: str
    case_id: str
    source_id: str
    class_side: int
    sample_count: int
    smooth_response: float
    response_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        donor = _center(self.donor_center, "donor_center")
        nonempty_text(self.case_id, "case_id")
        source = _center(self.source_id, "source_id")
        if source == donor:
            raise ProtocolError("A donor response cannot use its target expert.")
        if self.class_side not in (0, 1) or isinstance(self.class_side, bool):
            raise ProtocolError("class_side must be integer zero or one.")
        if isinstance(self.sample_count, bool) or self.sample_count <= 0:
            raise ProtocolError("Donor response sample_count must be positive.")
        response = finite_float(self.smooth_response, "smooth_response")
        if not -1.0 <= response <= 1.0:
            raise ProtocolError("Smooth donor response must lie in [-1, 1].")
        object.__setattr__(self, "smooth_response", response)
        object.__setattr__(self, "response_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_donor_response_v1",
            "donor_center": self.donor_center,
            "case_id": self.case_id,
            "source_id": self.source_id,
            "class_side": self.class_side,
            "sample_count": self.sample_count,
            "smooth_response": self.smooth_response,
            "terminal_metric": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "response_hash": self.response_hash}


@dataclass(frozen=True)
class Standardization:
    means: tuple[float, float, float, float, float]
    scales: tuple[float, float, float, float, float]

    def __post_init__(self) -> None:
        means = _tuple_floats(self.means, length=5, name="standardization.means")
        scales = _tuple_floats(self.scales, length=5, name="standardization.scales")
        if any(value <= 0.0 for value in scales):
            raise ProtocolError("Standardization scales must be positive.")
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "scales", scales)

    def to_payload(self) -> dict[str, object]:
        return {"means": list(self.means), "scales": list(self.scales)}


@dataclass(frozen=True)
class CandidateClassModel:
    target_center: str
    heldout_source_id: str
    class_side: int
    ridge_alpha: float
    coefficients: tuple[float, ...]
    standardization: Standardization
    training_row_count: int
    donor_centers: tuple[str, ...]
    nested_validation_mse: tuple[tuple[float, float], ...]
    model_family: str = "R"
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = _center(self.target_center)
        _source(target, self.heldout_source_id)
        if self.class_side not in (0, 1) or isinstance(self.class_side, bool):
            raise ProtocolError("class_side must be integer zero or one.")
        if self.model_family not in ("G", "R", "P"):
            raise ProtocolError("Candidate model family must be G, R, or P.")
        alpha = finite_float(self.ridge_alpha, "ridge_alpha")
        if alpha not in RIDGE_GRID:
            raise ProtocolError("Ridge alpha left the frozen donor-only grid.")
        coefficients = _tuple_floats(
            self.coefficients, length=len(DESIGN_TERMS), name="coefficients"
        )
        if isinstance(self.training_row_count, bool) or self.training_row_count <= 0:
            raise ProtocolError("Model needs at least one donor training row.")
        donors = tuple(self.donor_centers)
        if not donors or donors != tuple(sorted(set(donors))):
            raise ProtocolError("Model donor centers must be non-empty, unique, and sorted.")
        if target in donors or self.heldout_source_id in donors:
            raise ProtocolError("Final model violated strict H/e donor exclusion.")
        validation = tuple(
            (finite_float(a, "validation alpha"), finite_float(mse, "validation mse"))
            for a, mse in self.nested_validation_mse
        )
        if tuple(a for a, _ in validation) != RIDGE_GRID or any(mse < 0.0 for _, mse in validation):
            raise ProtocolError("Nested validation does not cover the exact ridge grid.")
        object.__setattr__(self, "ridge_alpha", alpha)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "donor_centers", donors)
        object.__setattr__(self, "nested_validation_mse", validation)
        object.__setattr__(self, "model_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_candidate_model_v1",
            "target_center": self.target_center,
            "heldout_source_id": self.heldout_source_id,
            "class_side": self.class_side,
            "ridge_alpha": self.ridge_alpha,
            "coefficients": list(self.coefficients),
            "design_terms": list(DESIGN_TERMS),
            "source_id_term_present": False,
            "rank": MODEL_RANK,
            "standardization": self.standardization.to_payload(),
            "training_row_count": self.training_row_count,
            "donor_centers": list(self.donor_centers),
            "nested_validation_mse": [list(value) for value in self.nested_validation_mse],
            "model_family": self.model_family,
            "uses_local_case_features": self.model_family != "G",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "model_hash": self.model_hash}


@dataclass(frozen=True)
class HierarchicalResidualModel:
    target_center: str
    candidate_models: tuple[CandidateClassModel, ...]
    feature_surface_hash: str
    response_surface_hash: str
    model_family: str = "R"
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = _center(self.target_center)
        if self.model_family not in ("G", "R", "P"):
            raise ProtocolError("Hierarchical model family must be G, R, or P.")
        models = tuple(self.candidate_models)
        expected = {(source, side) for source in candidate_sources(target) for side in (0, 1)}
        observed = {(row.heldout_source_id, row.class_side) for row in models}
        if observed != expected or len(models) != len(expected):
            raise ProtocolError("Hierarchical model must contain both classes for eight sources.")
        if models != tuple(sorted(models, key=lambda row: (row.heldout_source_id, row.class_side))):
            raise ProtocolError("Candidate models must be canonically ordered.")
        if any(row.model_family != self.model_family for row in models):
            raise ProtocolError("Candidate models drifted from their model family.")
        for name in ("feature_surface_hash", "response_surface_hash"):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64:
                raise ProtocolError(f"{name} must be a full SHA-256.")
        object.__setattr__(self, "candidate_models", models)
        object.__setattr__(self, "model_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_model_v1",
            "target_center": self.target_center,
            "candidate_models": [row.to_payload() for row in self.candidate_models],
            "feature_surface_hash": self.feature_surface_hash,
            "response_surface_hash": self.response_surface_hash,
            "model_family": self.model_family,
            "permutation_control": self.model_family == "P",
            "rank": MODEL_RANK,
            "target_labels_used": False,
            "evaluation_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "model_hash": self.model_hash}

    def candidate(self, source_id: str, class_side: int) -> CandidateClassModel:
        for row in self.candidate_models:
            if row.heldout_source_id == source_id and row.class_side == class_side:
                return row
        raise ProtocolError("Candidate class model is missing.")


@dataclass(frozen=True, order=True)
class CaseClassWeights:
    target_center: str
    case_id: str
    class_side: int
    weights: tuple[tuple[str, float], ...]
    predicted_gains: tuple[tuple[str, float], ...]
    weight_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        target = _center(self.target_center)
        nonempty_text(self.case_id, "case_id")
        if self.class_side not in (0, 1) or isinstance(self.class_side, bool):
            raise ProtocolError("class_side must be integer zero or one.")
        weights = tuple((str(key), finite_float(value, "weight")) for key, value in self.weights)
        gains = tuple((str(key), finite_float(value, "predicted_gain")) for key, value in self.predicted_gains)
        if weights != tuple(sorted(weights)) or gains != tuple(sorted(gains)):
            raise ProtocolError("Weights and gains must be source-ID ordered.")
        if set(key for key, _ in gains) != set(candidate_sources(target)):
            raise ProtocolError("Predicted gains must cover all legal sources.")
        if not 0 <= len(weights) <= SPARSE_SOURCE_BUDGET:
            raise ProtocolError("Weights violate the fixed sparse-source budget.")
        if any(key not in candidate_sources(target) or value <= 0.0 for key, value in weights):
            raise ProtocolError("Sparse weights contain an illegal source or value.")
        if weights and abs(sum(value for _, value in weights) - 1.0) > 1.0e-12:
            raise ProtocolError("Sparse source weights must sum exactly to one numerically.")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "predicted_gains", gains)
        object.__setattr__(self, "weight_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_case_weights_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "class_side": self.class_side,
            "weights": [list(value) for value in self.weights],
            "predicted_gains": [list(value) for value in self.predicted_gains],
            "temperature": SOFTMAX_TEMPERATURE,
            "top_k": SPARSE_SOURCE_BUDGET,
            "case_conditioning": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "weight_hash": self.weight_hash}


@dataclass(frozen=True, order=True)
class PredictionRow:
    method_id: str
    target_center: str
    case_id: str
    sample_id: str
    probability: float
    hard_prediction: int
    prediction_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        nonempty_text(self.method_id, "method_id")
        _center(self.target_center)
        nonempty_text(self.case_id, "case_id")
        nonempty_text(self.sample_id, "sample_id")
        probability = finite_float(self.probability, "probability")
        if not 0.0 <= probability <= 1.0:
            raise ProtocolError("Composed probability must lie in [0, 1].")
        if self.hard_prediction != int(probability >= HARD_THRESHOLD):
            raise ProtocolError("Hard prediction drifted from the fixed 0.5 threshold.")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "prediction_hash", canonical_hash(self._unhashed()))

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return (self.target_center, self.case_id, self.sample_id)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_prediction_v1",
            "method_id": self.method_id,
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "probability": self.probability,
            "hard_prediction": self.hard_prediction,
            "threshold": HARD_THRESHOLD,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prediction_hash": self.prediction_hash}


@dataclass(frozen=True, order=True)
class CaseConfusionCounts:
    method_id: str
    target_center: str
    case_id: str
    n_positive: int
    true_positive: int
    n_negative: int
    true_negative: int

    def __post_init__(self) -> None:
        nonempty_text(self.method_id, "method_id")
        _center(self.target_center)
        nonempty_text(self.case_id, "case_id")
        for name in ("n_positive", "true_positive", "n_negative", "true_negative"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProtocolError(f"{name} must be a nonnegative integer.")
        if self.n_positive + self.n_negative <= 0:
            raise ProtocolError("A whole-case count row cannot be empty.")
        if self.true_positive > self.n_positive or self.true_negative > self.n_negative:
            raise ProtocolError("Correct counts exceed class counts.")

    @property
    def case_key(self) -> tuple[str, str]:
        return (self.target_center, self.case_id)


@dataclass(frozen=True)
class PooledExactBacc:
    method_id: str
    case_count: int
    n_positive: int
    true_positive: int
    n_negative: int
    true_negative: int
    sensitivity: float
    specificity: float
    exact_bacc: float
    metric_hash: str = field(init=False)

    def __post_init__(self) -> None:
        nonempty_text(self.method_id, "method_id")
        if self.case_count <= 0 or self.n_positive <= 0 or self.n_negative <= 0:
            raise ProtocolError("Pooled exact BACC requires cases and both pooled classes.")
        sensitivity = finite_float(self.sensitivity, "sensitivity")
        specificity = finite_float(self.specificity, "specificity")
        exact = finite_float(self.exact_bacc, "exact_bacc")
        if abs(sensitivity - self.true_positive / self.n_positive) > 1.0e-12:
            raise ProtocolError("Sensitivity drifted from pooled counts.")
        if abs(specificity - self.true_negative / self.n_negative) > 1.0e-12:
            raise ProtocolError("Specificity drifted from pooled counts.")
        if abs(exact - 0.5 * (sensitivity + specificity)) > 1.0e-12:
            raise ProtocolError("Exact BACC drifted from pooled counts.")
        object.__setattr__(self, "sensitivity", sensitivity)
        object.__setattr__(self, "specificity", specificity)
        object.__setattr__(self, "exact_bacc", exact)
        object.__setattr__(self, "metric_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_pooled_exact_bacc_v1",
            "method_id": self.method_id,
            "case_count": self.case_count,
            "n_positive": self.n_positive,
            "true_positive": self.true_positive,
            "n_negative": self.n_negative,
            "true_negative": self.true_negative,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "exact_bacc": self.exact_bacc,
            "per_case_bacc_used": False,
            "smooth_response_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "metric_hash": self.metric_hash}


@dataclass(frozen=True)
class PairedClusterEstimate:
    challenger_method: str
    reference_method: str
    case_count: int
    difference: float
    standard_error: float
    confidence_multiplier: float
    lower_bound: float
    case_influences: tuple[tuple[str, str, float], ...]
    estimate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.challenger_method == self.reference_method:
            raise ProtocolError("Paired contrast needs distinct methods.")
        if self.case_count < 2 or len(self.case_influences) != self.case_count:
            raise ProtocolError("Paired whole-case contrast needs every case exactly once.")
        if len({(h, case) for h, case, _ in self.case_influences}) != self.case_count:
            raise ProtocolError("Paired whole-case influences contain duplicate cases.")
        for name in ("difference", "standard_error", "confidence_multiplier", "lower_bound"):
            object.__setattr__(self, name, finite_float(getattr(self, name), name))
        if self.standard_error < 0.0 or self.confidence_multiplier <= 0.0:
            raise ProtocolError("Paired cluster uncertainty is invalid.")
        if abs(self.lower_bound - (self.difference - self.confidence_multiplier * self.standard_error)) > 1.0e-12:
            raise ProtocolError("Paired lower confidence bound identity drifted.")
        object.__setattr__(self, "estimate_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_case_cluster_lcb_v1",
            "challenger_method": self.challenger_method,
            "reference_method": self.reference_method,
            "case_count": self.case_count,
            "difference": self.difference,
            "standard_error": self.standard_error,
            "confidence_multiplier": self.confidence_multiplier,
            "lower_bound": self.lower_bound,
            "case_influences": [list(value) for value in self.case_influences],
            "uncertainty_unit": "paired_whole_case_cluster",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "estimate_hash": self.estimate_hash}


@dataclass(frozen=True)
class CalibrationChoice:
    method_id: str
    intercept: float
    residual_scale: float
    objective_value: float
    support_case_count: int
    lcb_gain_over_baseline_calibrated: float | None
    selected_before_evaluation_labels: bool = True
    calibration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        nonempty_text(self.method_id, "method_id")
        intercept = finite_float(self.intercept, "intercept")
        scale = finite_float(self.residual_scale, "residual_scale")
        objective = finite_float(self.objective_value, "objective_value")
        if not 0.0 <= scale <= MAX_RESIDUAL_SCALE:
            raise ProtocolError("Residual scale left the frozen [0, 0.25] interval.")
        if self.support_case_count <= 0:
            raise ProtocolError("Calibration needs target-support whole cases.")
        lcb = self.lcb_gain_over_baseline_calibrated
        if lcb is not None:
            lcb = finite_float(lcb, "lcb_gain_over_baseline_calibrated")
        if self.selected_before_evaluation_labels is not True:
            raise ProtocolError("Calibration must be sealed before evaluation-label access.")
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "residual_scale", scale)
        object.__setattr__(self, "objective_value", objective)
        object.__setattr__(self, "lcb_gain_over_baseline_calibrated", lcb)
        object.__setattr__(self, "calibration_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_calibration_v1",
            "method_id": self.method_id,
            "intercept": self.intercept,
            "residual_scale": self.residual_scale,
            "objective_value": self.objective_value,
            "support_case_count": self.support_case_count,
            "lcb_gain_over_baseline_calibrated": self.lcb_gain_over_baseline_calibrated,
            "selected_before_evaluation_labels": True,
            "threshold": HARD_THRESHOLD,
            "probability_epsilon": PROBABILITY_EPSILON,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "calibration_hash": self.calibration_hash}


__all__ = tuple(name for name in globals() if not name.startswith("_"))
