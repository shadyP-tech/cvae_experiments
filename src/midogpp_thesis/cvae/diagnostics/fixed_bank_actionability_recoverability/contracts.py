"""Immutable contracts for the actionability/recoverability scientific core."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .constants import (
    A1_OTHER_SAMPLE_WEIGHT,
    A1_SELECTED_SAMPLE_WEIGHT,
    BINARY_CLASSES,
    B_ACTION_ID,
    B_COUNT_PER_SOURCE_CLASS,
    CASE_ACTION_FEATURE_NAMES,
    GEOMETRY_IDS,
    HARD_THRESHOLD,
    MIDOGPP_CENTERS,
    OTHER_COUNT_PER_CLASS,
    PRE_SUPPORT_METHOD_IDS,
    RIDGE_ALPHA,
    SEED_PAIR_ORDINALS,
    SELECTED_COUNT_PER_CLASS,
    SUPPORT_METHOD_ID,
    TERMINAL_ORACLE_METHOD_IDS,
    U_ACTION_ID,
    U_COUNT_PER_SOURCE_CLASS,
    candidate_sources,
    geometry_action_id,
)
from .hashing import canonical_hash, finite, nonempty_text, require_sha256


def _identity(target_center: object, case_id: object, sample_id: object | None = None) -> tuple[str, str, str | None]:
    target = str(target_center)
    if target not in MIDOGPP_CENTERS:
        raise ProtocolError("Scientific row uses an unknown MIDOG++ center.")
    case = nonempty_text(case_id, "case_id")
    sample = None if sample_id is None else nonempty_text(sample_id, "sample_id")
    return target, case, sample


def _canonical_counts(
    value: Mapping[int, Mapping[str, int]],
    *,
    expected_sources: tuple[str, ...],
) -> Mapping[int, Mapping[str, int]]:
    if set(value) != set(BINARY_CLASSES):
        raise ProtocolError("Action counts must cover binary classes zero and one.")
    output: dict[int, Mapping[str, int]] = {}
    for label in BINARY_CLASSES:
        raw = value[label]
        if set(raw) != set(expected_sources):
            raise ProtocolError("Action counts drifted from the target's eight-source pool.")
        counts: dict[str, int] = {}
        for source in expected_sources:
            count = raw[source]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ProtocolError("Action source counts must be positive integers.")
            counts[source] = count
        output[label] = MappingProxyType(counts)
    return MappingProxyType(output)


def _canonical_weights(
    value: Mapping[str, float], *, expected_sources: tuple[str, ...]
) -> Mapping[str, float]:
    if set(value) != set(expected_sources):
        raise ProtocolError("Action weights drifted from the target's eight-source pool.")
    output = {source: finite(value[source], f"sample weight {source}") for source in expected_sources}
    if any(weight <= 0.0 for weight in output.values()):
        raise ProtocolError("Action sample weights must be positive.")
    return MappingProxyType(output)


@dataclass(frozen=True)
class ActionSpec:
    """One deterministic physical action; seed repetitions are never actions."""

    target_center: str
    action_id: str
    geometry_id: str | None
    selected_source: str | None
    counts_by_class: Mapping[int, Mapping[str, int]]
    sample_weight_by_source: Mapping[str, float]
    physical_fit_required: bool
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target, _case, _sample = _identity(self.target_center, "action-spec")
        action = nonempty_text(self.action_id, "action_id")
        sources = candidate_sources(target)
        counts = _canonical_counts(self.counts_by_class, expected_sources=sources)
        weights = _canonical_weights(self.sample_weight_by_source, expected_sources=sources)
        geometry = self.geometry_id
        selected = self.selected_source
        if action == B_ACTION_ID:
            if geometry is not None or selected is not None or self.physical_fit_required is not True:
                raise ProtocolError("B must be a recomputed, geometry-free baseline fit.")
            expected_count = {source: B_COUNT_PER_SOURCE_CLASS for source in sources}
            expected_weight = {source: 1.0 for source in sources}
        elif action == U_ACTION_ID:
            if geometry is not None or selected is not None or self.physical_fit_required is not True:
                raise ProtocolError("U must be the common physical uniform action.")
            expected_count = {source: U_COUNT_PER_SOURCE_CLASS for source in sources}
            expected_weight = {source: 1.0 for source in sources}
        else:
            if geometry not in GEOMETRY_IDS or selected not in sources:
                raise ProtocolError("Source action has an invalid geometry/source context.")
            if action != geometry_action_id(geometry, selected):
                raise ProtocolError("Source action identifier is not canonical.")
            if self.physical_fit_required is not True:
                raise ProtocolError("Every A0/A1 action requires its own physical fit.")
            expected_count = {
                source: SELECTED_COUNT_PER_CLASS if source == selected else OTHER_COUNT_PER_CLASS
                for source in sources
            }
            expected_weight = {
                source: (
                    1.0
                    if geometry == "A0"
                    else A1_SELECTED_SAMPLE_WEIGHT
                    if source == selected
                    else A1_OTHER_SAMPLE_WEIGHT
                )
                for source in sources
            }
        for label in BINARY_CLASSES:
            if dict(counts[label]) != expected_count:
                raise ProtocolError("Action row counts drifted from the frozen geometry.")
        if any(not math.isclose(weights[source], expected_weight[source], rel_tol=0.0, abs_tol=0.0) for source in sources):
            raise ProtocolError("Action sample weights drifted from the frozen geometry.")
        expected_total = 1024 if action == B_ACTION_ID else 1152
        for label in BINARY_CLASSES:
            effective = math.fsum(counts[label][source] * weights[source] for source in sources)
            if not math.isclose(effective, expected_total, rel_tol=0.0, abs_tol=1.0e-12):
                raise ProtocolError("Action effective class mass violates its budget lock.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "counts_by_class", counts)
        object.__setattr__(self, "sample_weight_by_source", weights)
        object.__setattr__(self, "action_hash", canonical_hash(self._unhashed()))

    @property
    def sample_weights_by_source(self) -> Mapping[str, float]:
        """Compatibility alias for callers that use the plural field spelling."""

        return self.sample_weight_by_source

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_action_spec_v1",
            "target_center": self.target_center,
            "action_id": self.action_id,
            "geometry_id": self.geometry_id,
            "selected_source": self.selected_source,
            "counts_by_class": {
                str(label): dict(self.counts_by_class[label]) for label in BINARY_CLASSES
            },
            "sample_weight_by_source": dict(self.sample_weight_by_source),
            "physical_fit_required": self.physical_fit_required,
            "target_expert_excluded": True,
            "seed_repetitions_selectable": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "action_hash": self.action_hash}


@dataclass(frozen=True, order=True)
class SeedProbabilityRow:
    target_center: str
    case_id: str
    sample_id: str
    action_id: str
    seed_pair_ordinal: int
    probability: float
    probability_store_hash: str

    def __post_init__(self) -> None:
        _identity(self.target_center, self.case_id, self.sample_id)
        nonempty_text(self.action_id, "action_id")
        if self.seed_pair_ordinal not in SEED_PAIR_ORDINALS or isinstance(self.seed_pair_ordinal, bool):
            raise ProtocolError("Seed-pair ordinal must be exactly one of zero through eight.")
        probability = finite(self.probability, "probability")
        if probability < 0.0 or probability > 1.0:
            raise ProtocolError("Probability must lie in [0, 1].")
        require_sha256(self.probability_store_hash, "probability_store_hash")
        object.__setattr__(self, "probability", probability)

    @property
    def row_key(self) -> tuple[str, str, str, str]:
        return self.target_center, self.case_id, self.sample_id, self.action_id


@dataclass(frozen=True, order=True)
class AggregatedProbabilityRow:
    target_center: str
    case_id: str
    sample_id: str
    action_id: str
    probability_mean: float
    probability_sd: float
    seed_pair_count: int
    seed_probability_hash: str
    row_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        _identity(self.target_center, self.case_id, self.sample_id)
        nonempty_text(self.action_id, "action_id")
        mean = finite(self.probability_mean, "probability_mean")
        sd = finite(self.probability_sd, "probability_sd")
        if not 0.0 <= mean <= 1.0 or sd < 0.0 or self.seed_pair_count != 9:
            raise ProtocolError("Aggregated probability moments are not exact-nine.")
        require_sha256(self.seed_probability_hash, "seed_probability_hash")
        object.__setattr__(self, "probability_mean", mean)
        object.__setattr__(self, "probability_sd", sd)
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    @property
    def row_key(self) -> tuple[str, str, str, str]:
        return self.target_center, self.case_id, self.sample_id, self.action_id

    def _unhashed(self) -> dict[str, object]:
        return {
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "action_id": self.action_id,
            "probability_mean": self.probability_mean,
            "probability_sd": self.probability_sd,
            "seed_pair_count": self.seed_pair_count,
            "seed_probability_hash": self.seed_probability_hash,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True)
class ExactNineProbabilitySurface:
    rows: tuple[AggregatedProbabilityRow, ...]
    probability_store_hash: str
    surface_hash: str
    predictions_sealed_before_labels: bool = True

    def __post_init__(self) -> None:
        require_sha256(self.probability_store_hash, "probability_store_hash")
        require_sha256(self.surface_hash, "surface_hash")
        rows = tuple(self.rows)
        if not rows or rows != tuple(sorted(rows, key=lambda row: row.row_key)):
            raise ProtocolError("Probability surface rows must be non-empty and canonical.")
        if len({row.row_key for row in rows}) != len(rows):
            raise ProtocolError("Probability surface contains duplicate action rows.")
        if self.predictions_sealed_before_labels is not True:
            raise ProtocolError("Action probabilities must be sealed before label access.")
        expected = canonical_hash(
            {
                "schema_version": "fixed_bank_actionability_exact_nine_surface_v1",
                "probability_store_hash": self.probability_store_hash,
                "rows": [row.to_payload() for row in rows],
                "predictions_sealed_before_labels": True,
            }
        )
        if expected != self.surface_hash:
            raise ProtocolError("Exact-nine probability surface hash drifted.")
        object.__setattr__(self, "rows", rows)


@dataclass(frozen=True, order=True)
class CaseActionFeatureRow:
    query_center: str
    case_id: str
    geometry_id: str
    selected_source: str
    values: tuple[float, ...]
    feature_origin_source: str | None = None
    context_excluded_centers: tuple[str, ...] = ()
    feature_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        _identity(self.query_center, self.case_id)
        if self.geometry_id not in GEOMETRY_IDS or self.selected_source not in candidate_sources(self.query_center):
            raise ProtocolError("Case-action feature uses an invalid geometry/source.")
        values = tuple(finite(value, "case-action feature") for value in self.values)
        if len(values) != len(CASE_ACTION_FEATURE_NAMES) or values[0] != 1.0:
            raise ProtocolError("Case-action feature width/intercept drifted.")
        origin = self.selected_source if self.feature_origin_source is None else str(self.feature_origin_source)
        exclusions = tuple(sorted(set(str(value) for value in self.context_excluded_centers)))
        if origin not in MIDOGPP_CENTERS or any(value not in MIDOGPP_CENTERS for value in exclusions):
            raise ProtocolError("Feature origin/exclusion context is invalid.")
        if origin in exclusions or self.selected_source in exclusions:
            raise ProtocolError("Excluded candidate entered a feature context.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "feature_origin_source", origin)
        object.__setattr__(self, "context_excluded_centers", exclusions)
        object.__setattr__(self, "feature_hash", canonical_hash(self._unhashed()))

    @property
    def action_id(self) -> str:
        return geometry_action_id(self.geometry_id, self.selected_source)

    @property
    def row_key(self) -> tuple[str, str, str, str]:
        return self.query_center, self.case_id, self.geometry_id, self.selected_source

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_case_action_feature_v1",
            "query_center": self.query_center,
            "case_id": self.case_id,
            "geometry_id": self.geometry_id,
            "selected_source": self.selected_source,
            "feature_origin_source": self.feature_origin_source,
            "context_excluded_centers": list(self.context_excluded_centers),
            "feature_names": list(CASE_ACTION_FEATURE_NAMES),
            "values": list(self.values),
            "labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "action_id": self.action_id, "feature_hash": self.feature_hash}


@dataclass(frozen=True, order=True)
class UtilityTargetRow:
    query_center: str
    case_id: str
    geometry_id: str
    selected_source: str
    response: float
    response_kind: str = "class_balanced_proper_loss_gain_vs_u"

    def __post_init__(self) -> None:
        _identity(self.query_center, self.case_id)
        if self.geometry_id not in GEOMETRY_IDS or self.selected_source not in candidate_sources(self.query_center):
            raise ProtocolError("Utility target uses an invalid query/source context.")
        if self.response_kind not in (
            "class_balanced_proper_loss_gain_vs_u",
            "pooled_bacc_additive_gain_vs_u",
        ):
            raise ProtocolError("Utility response must be a predeclared dense/additive target.")
        object.__setattr__(self, "response", finite(self.response, "utility response"))

    @property
    def row_key(self) -> tuple[str, str, str, str]:
        return self.query_center, self.case_id, self.geometry_id, self.selected_source


@dataclass(frozen=True)
class RidgeActionModel:
    outer_target_center: str
    heldout_donor_center: str | None
    geometry_id: str
    selected_source: str
    family: str
    ridge_alpha: float
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    training_query_centers: tuple[str, ...]
    response_kind: str
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = str(self.outer_target_center)
        if outer not in MIDOGPP_CENTERS or self.geometry_id not in GEOMETRY_IDS:
            raise ProtocolError("Ridge model has an invalid outer/geometry context.")
        if self.selected_source not in candidate_sources(outer) or self.family not in ("G", "R", "P"):
            raise ProtocolError("Ridge model has an invalid source/family.")
        heldout = self.heldout_donor_center
        if heldout is not None and (heldout not in MIDOGPP_CENTERS or heldout in (outer, self.selected_source)):
            raise ProtocolError("Nested donor-q exclusion is invalid.")
        alpha = finite(self.ridge_alpha, "ridge_alpha")
        if not math.isclose(alpha, RIDGE_ALPHA, rel_tol=0.0, abs_tol=0.0):
            raise ProtocolError("Ridge alpha drifted from the frozen value.")
        names = tuple(self.feature_names)
        means = tuple(finite(value, "feature mean") for value in self.means)
        scales = tuple(finite(value, "feature scale") for value in self.scales)
        coefficients = tuple(finite(value, "ridge coefficient") for value in self.coefficients)
        if not names or len(names) != len(means) or len(names) != len(scales) or len(names) != len(coefficients):
            raise ProtocolError("Ridge model dimensions are inconsistent.")
        if names[0] != "intercept" or means[0] != 0.0 or scales[0] != 1.0 or any(value <= 0.0 for value in scales):
            raise ProtocolError("Ridge standardization contract drifted.")
        legal = set(MIDOGPP_CENTERS).difference((outer, self.selected_source))
        if heldout is not None:
            legal.discard(heldout)
        training = tuple(self.training_query_centers)
        if training != tuple(center for center in MIDOGPP_CENTERS if center in legal):
            raise ProtocolError("Ridge model violated outer-H/donor-q/candidate-e exclusion.")
        object.__setattr__(self, "ridge_alpha", alpha)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "training_query_centers", training)
        object.__setattr__(self, "model_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_ridge_model_v1",
            "outer_target_center": self.outer_target_center,
            "heldout_donor_center": self.heldout_donor_center,
            "geometry_id": self.geometry_id,
            "selected_source": self.selected_source,
            "family": self.family,
            "ridge_alpha": self.ridge_alpha,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "training_query_centers": list(self.training_query_centers),
            "response_kind": self.response_kind,
            "target_labels_used": False,
            "exclusion_rule": "q_not_in_{outer_H,heldout_donor_q,candidate_e}",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "model_hash": self.model_hash}


@dataclass(frozen=True, order=True)
class ActionScoreRow:
    target_center: str
    case_id: str
    geometry_id: str
    selected_source: str
    family: str
    predicted_gain: float
    model_hash: str

    def __post_init__(self) -> None:
        _identity(self.target_center, self.case_id)
        if self.geometry_id not in GEOMETRY_IDS or self.selected_source not in candidate_sources(self.target_center):
            raise ProtocolError("Action score uses an invalid target/geometry/source.")
        if self.family not in ("G", "R", "P"):
            raise ProtocolError("Action score family must be G, R, or P.")
        object.__setattr__(self, "predicted_gain", finite(self.predicted_gain, "predicted_gain"))
        require_sha256(self.model_hash, "model_hash")

    @property
    def row_key(self) -> tuple[str, str, str, str, str]:
        return self.target_center, self.case_id, self.geometry_id, self.family, self.selected_source


@dataclass(frozen=True, order=True)
class MethodDecision:
    target_center: str
    case_id: str
    method_id: str
    action_id: str
    geometry_id: str | None
    predicted_gain: float | None
    decision_source: str
    evaluation_labels_used: bool = False

    def __post_init__(self) -> None:
        _identity(self.target_center, self.case_id)
        allowed = (*PRE_SUPPORT_METHOD_IDS, SUPPORT_METHOD_ID)
        if self.method_id not in allowed:
            raise ProtocolError("Decision method is outside the predeclared surface.")
        if self.method_id == "B":
            if self.geometry_id is not None or self.action_id != B_ACTION_ID:
                raise ProtocolError("Global B decision cannot be geometry-specific.")
        else:
            if self.geometry_id not in GEOMETRY_IDS:
                raise ProtocolError("Non-B decisions must remain within one geometry.")
        if self.method_id == "U" and self.action_id != U_ACTION_ID:
            raise ProtocolError("U decision must use the uniform action.")
        if self.evaluation_labels_used is not False:
            raise ProtocolError("Routing decisions cannot use evaluation labels.")
        if self.predicted_gain is not None:
            object.__setattr__(self, "predicted_gain", finite(self.predicted_gain, "predicted_gain"))


@dataclass(frozen=True, order=True)
class BinaryLabelRow:
    target_center: str
    case_id: str
    sample_id: str
    label: int

    def __post_init__(self) -> None:
        _identity(self.target_center, self.case_id, self.sample_id)
        if isinstance(self.label, bool) or self.label not in BINARY_CLASSES:
            raise ProtocolError("Label row must contain integer zero or one.")

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.sample_id


@dataclass(frozen=True, order=True)
class BinaryPredictionRow:
    target_center: str
    case_id: str
    sample_id: str
    action_id: str
    hard_prediction: int

    def __post_init__(self) -> None:
        _identity(self.target_center, self.case_id, self.sample_id)
        nonempty_text(self.action_id, "action_id")
        if isinstance(self.hard_prediction, bool) or self.hard_prediction not in BINARY_CLASSES:
            raise ProtocolError("Hard prediction must contain integer zero or one.")

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.sample_id


@dataclass(frozen=True, order=True)
class CaseConfusionCounts:
    target_center: str
    case_id: str
    action_id: str
    n_positive: int
    true_positive: int
    n_negative: int
    true_negative: int

    def __post_init__(self) -> None:
        _identity(self.target_center, self.case_id)
        nonempty_text(self.action_id, "action_id")
        values = (self.n_positive, self.true_positive, self.n_negative, self.true_negative)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ProtocolError("Confusion counts must be non-negative integers.")
        if self.true_positive > self.n_positive or self.true_negative > self.n_negative:
            raise ProtocolError("Correct counts cannot exceed their class totals.")

    @property
    def case_key(self) -> tuple[str, str]:
        return self.target_center, self.case_id


@dataclass(frozen=True)
class PooledBacc:
    action_or_method_id: str
    case_count: int
    n_positive: int
    true_positive: int
    n_negative: int
    true_negative: int
    sensitivity: float
    specificity: float
    exact_bacc: float

    def __post_init__(self) -> None:
        nonempty_text(self.action_or_method_id, "action_or_method_id")
        if self.case_count <= 0 or self.n_positive <= 0 or self.n_negative <= 0:
            raise ProtocolError("Pooled BACC requires cases and both classes.")
        for name in ("sensitivity", "specificity", "exact_bacc"):
            value = finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ProtocolError("Pooled BACC components must lie in [0, 1].")


@dataclass(frozen=True)
class TerminalOracleResult:
    target_center: str
    geometry_id: str
    oracle_method: str
    selected_action_by_case: tuple[tuple[str, str], ...]
    pooled_bacc: PooledBacc
    diagnostic_only: bool = True

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS or self.geometry_id not in GEOMETRY_IDS:
            raise ProtocolError("Terminal oracle has an invalid context.")
        if self.oracle_method not in TERMINAL_ORACLE_METHOD_IDS or self.diagnostic_only is not True:
            raise ProtocolError("Oracle rows are terminal diagnostic-only surfaces.")
        selections = tuple(self.selected_action_by_case)
        if not selections or selections != tuple(sorted(selections)) or len(dict(selections)) != len(selections):
            raise ProtocolError("Oracle case selections must be canonical and unique.")
        object.__setattr__(self, "selected_action_by_case", selections)


__all__ = (
    "ActionScoreRow",
    "ActionSpec",
    "AggregatedProbabilityRow",
    "BinaryLabelRow",
    "BinaryPredictionRow",
    "CaseActionFeatureRow",
    "CaseConfusionCounts",
    "ExactNineProbabilitySurface",
    "MethodDecision",
    "PooledBacc",
    "RidgeActionModel",
    "SeedProbabilityRow",
    "TerminalOracleResult",
    "UtilityTargetRow",
)
