"""Experiment-owned immutable products for the abstention router.

The DTOs deliberately duplicate no product type from another Stage-90
diagnostic.  Label-bearing products are capability-local; every product that
may cross a pre-terminal persistence boundary carries a canonical hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    DESCRIPTIVE_METHOD_IDS,
    DIRECTION_IDS,
    FEATURE_NAMES,
    IRLS_CONVERGENCE_TOLERANCE,
    IRLS_MAX_ITERATIONS,
    PRE_TERMINAL_METHOD_IDS,
    RIDGE_ALPHA,
    TIE_TOLERANCE,
    candidate_sources,
)
from .hashing import canonical_hash, require_sha256


def _text(value: object, role: str) -> str:
    result = str(value)
    if not result:
        raise ProtocolError(f"Abstention-router {role} is empty.")
    return result


def _finite(value: object, role: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"Abstention-router {role} is not finite.")
    return result


def _count(value: object, role: str) -> int:
    if isinstance(value, bool):
        raise ProtocolError(f"Abstention-router {role} is not an integer count.")
    result = int(value)
    if result != value or result < 0:
        raise ProtocolError(f"Abstention-router {role} is not a nonnegative count.")
    return result


def _direction(value: object) -> str:
    result = str(value)
    if result not in DIRECTION_IDS:
        raise ProtocolError("Abstention-router direction identity drifted.")
    return result


def _source(target_center: object, value: object) -> str:
    target = str(target_center)
    result = str(value)
    if result not in candidate_sources(target):
        raise ProtocolError("Abstention-router candidate source drifted.")
    return result


@dataclass(frozen=True, order=True)
class BinaryLabel:
    """Ephemeral scoped label; intentionally has no persistence payload API."""

    target_center: str
    case_id: str
    sample_id: str
    value: int
    label_scope: str

    def __post_init__(self) -> None:
        target = str(self.target_center)
        value = _count(self.value, "binary label")
        if target not in CENTERS or value not in (0, 1):
            raise ProtocolError("Abstention-router binary label drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(self, "sample_id", _text(self.sample_id, "sample_id"))
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "label_scope", _text(self.label_scope, "label_scope"))

    @property
    def key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.sample_id


@dataclass(frozen=True, order=True)
class LabelFreeDirectionalFeatures:
    target_center: str
    case_id: str
    source: str
    direction: str
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    directional_flip_count: int
    case_size: int
    feature_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        names = tuple(str(value) for value in self.feature_names)
        values = tuple(_finite(value, "feature value") for value in self.values)
        flips = _count(self.directional_flip_count, "directional_flip_count")
        size = _count(self.case_size, "case_size")
        if (
            target not in CENTERS
            or names != FEATURE_NAMES
            or len(values) != len(FEATURE_NAMES)
            or size <= 0
            or flips > size
        ):
            raise ProtocolError("Abstention-router label-free feature schema drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(self, "source", _source(target, self.source))
        object.__setattr__(self, "direction", _direction(self.direction))
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "directional_flip_count", flips)
        object.__setattr__(self, "case_size", size)
        object.__setattr__(self, "feature_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.target_center, self.case_id, self.source, self.direction

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cdca_label_free_directional_features_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "source": self.source,
            "direction": self.direction,
            "feature_names": list(self.feature_names),
            "values": list(self.values),
            "directional_flip_count": self.directional_flip_count,
            "case_size": self.case_size,
            "labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "feature_hash": self.feature_hash}


@dataclass(frozen=True, order=True)
class DirectionalCorrectnessObservation:
    target_center: str
    route_case_id: str
    support_case_id: str
    source: str
    direction: str
    feature_values: tuple[float, ...]
    successes: int
    trials: int
    observation_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        route_case = _text(self.route_case_id, "route_case_id")
        support_case = _text(self.support_case_id, "support_case_id")
        values = tuple(_finite(value, "observation feature") for value in self.feature_values)
        successes = _count(self.successes, "successes")
        trials = _count(self.trials, "trials")
        if (
            target not in CENTERS
            or route_case == support_case
            or len(values) != len(FEATURE_NAMES)
            or successes > trials
        ):
            raise ProtocolError("Abstention-router correctness observation drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "route_case_id", route_case)
        object.__setattr__(self, "support_case_id", support_case)
        object.__setattr__(self, "source", _source(target, self.source))
        object.__setattr__(self, "direction", _direction(self.direction))
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "successes", successes)
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "observation_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.target_center,
            self.route_case_id,
            self.support_case_id,
            self.source,
            self.direction,
        )

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cdca_correctness_observation_v1",
            "target_center": self.target_center,
            "route_case_id": self.route_case_id,
            "support_case_id": self.support_case_id,
            "source": self.source,
            "direction": self.direction,
            "feature_names": list(FEATURE_NAMES),
            "feature_values": list(self.feature_values),
            "successes": self.successes,
            "trials": self.trials,
            "support_labels_only": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "observation_hash": self.observation_hash}


@dataclass(frozen=True, order=True)
class SupportClassDenominators:
    target_center: str
    route_case_id: str
    n_positive: int
    n_negative: int
    support_case_ids: tuple[str, ...]
    denominator_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        cases = tuple(sorted(str(value) for value in self.support_case_ids))
        positive = _count(self.n_positive, "n_positive")
        negative = _count(self.n_negative, "n_negative")
        if (
            self.target_center not in CENTERS
            or not cases
            or self.route_case_id in cases
            or len(cases) != len(set(cases))
            or positive <= 0
            or negative <= 0
        ):
            raise ProtocolError("Abstention-router support denominators drifted.")
        object.__setattr__(self, "support_case_ids", cases)
        object.__setattr__(self, "n_positive", positive)
        object.__setattr__(self, "n_negative", negative)
        object.__setattr__(self, "denominator_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cdca_support_denominators_v1",
            "target_center": self.target_center,
            "route_case_id": self.route_case_id,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "support_case_ids": list(self.support_case_ids),
            "held_case_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "denominator_hash": self.denominator_hash}


@dataclass(frozen=True, order=True)
class DirectionalGain:
    query_center: str
    source: str
    direction: str
    favorable_count: int
    adverse_count: int
    n_positive: int
    n_negative: int
    numerator: int
    denominator: int
    value: float
    gain_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        query = str(self.query_center)
        favorable = _count(self.favorable_count, "favorable_count")
        adverse = _count(self.adverse_count, "adverse_count")
        n_positive = _count(self.n_positive, "n_positive")
        n_negative = _count(self.n_negative, "n_negative")
        numerator = int(self.numerator)
        denominator = _count(self.denominator, "denominator")
        if query not in CENTERS or self.source == query or denominator <= 0:
            raise ProtocolError("Abstention-router directional gain identity drifted.")
        if denominator <= 0:
            raise ProtocolError("Abstention-router directional gain denominator drifted.")
        exact = Fraction(numerator, denominator)
        value = _finite(self.value, "directional gain")
        if value != float(exact):
            raise ProtocolError("Abstention-router directional gain fraction drifted.")
        object.__setattr__(self, "query_center", query)
        object.__setattr__(self, "source", _source(query, self.source))
        object.__setattr__(self, "direction", _direction(self.direction))
        object.__setattr__(self, "favorable_count", favorable)
        object.__setattr__(self, "adverse_count", adverse)
        object.__setattr__(self, "n_positive", n_positive)
        object.__setattr__(self, "n_negative", n_negative)
        object.__setattr__(self, "numerator", exact.numerator)
        object.__setattr__(self, "denominator", exact.denominator)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gain_hash", canonical_hash(self._unhashed()))

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cdca_directional_gain_v1",
            "query_center": self.query_center,
            "source": self.source,
            "direction": self.direction,
            "favorable_count": self.favorable_count,
            "adverse_count": self.adverse_count,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "gain_hash": self.gain_hash}


@dataclass(frozen=True, order=True)
class DonorDirectionalPrior:
    heldout_center: str
    source: str
    direction: str
    query_centers: tuple[str, ...]
    query_gain_hashes: tuple[str, ...]
    numerator: int
    denominator: int
    value: float
    prior_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        heldout = str(self.heldout_center)
        source = _source(heldout, self.source)
        queries = tuple(str(value) for value in self.query_centers)
        hashes = tuple(require_sha256(value, "query_gain_hash") for value in self.query_gain_hashes)
        expected_queries = tuple(center for center in CENTERS if center not in {heldout, source})
        if queries != expected_queries or len(queries) != len(hashes):
            raise ProtocolError("Abstention-router donor query scope drifted.")
        denominator = _count(self.denominator, "prior denominator")
        if denominator <= 0:
            raise ProtocolError("Abstention-router donor prior denominator drifted.")
        exact = Fraction(int(self.numerator), denominator)
        value = _finite(self.value, "donor prior")
        if value != float(exact):
            raise ProtocolError("Abstention-router donor prior fraction drifted.")
        object.__setattr__(self, "heldout_center", heldout)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "direction", _direction(self.direction))
        object.__setattr__(self, "query_centers", queries)
        object.__setattr__(self, "query_gain_hashes", hashes)
        object.__setattr__(self, "numerator", exact.numerator)
        object.__setattr__(self, "denominator", exact.denominator)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "prior_hash", canonical_hash(self._unhashed()))

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cdca_donor_directional_prior_v1",
            "heldout_center": self.heldout_center,
            "source": self.source,
            "direction": self.direction,
            "query_centers": list(self.query_centers),
            "query_gain_hashes": list(self.query_gain_hashes),
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "query_excludes_heldout_and_source": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prior_hash": self.prior_hash}


@dataclass(frozen=True, order=True)
class DirectionalCorrectnessModel:
    target_center: str
    case_id: str
    source: str
    direction: str
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    training_case_ids: tuple[str, ...]
    training_trial_count: int
    valid_observation_count: int
    converged: bool
    iterations: int
    alpha: float = RIDGE_ALPHA
    max_iterations: int = IRLS_MAX_ITERATIONS
    tolerance: float = IRLS_CONVERGENCE_TOLERANCE
    training_observation_hashes: tuple[str, ...] = ()
    model_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        names = tuple(str(value) for value in self.feature_names)
        mean = tuple(_finite(value, "feature mean") for value in self.feature_mean)
        scale = tuple(_finite(value, "feature scale") for value in self.feature_scale)
        coefficients = tuple(_finite(value, "coefficient") for value in self.coefficients)
        cases = tuple(sorted(str(value) for value in self.training_case_ids))
        observation_hashes = tuple(
            require_sha256(value, "training_observation_hash")
            for value in self.training_observation_hashes
        )
        iterations = _count(self.iterations, "iterations")
        if (
            target not in CENTERS
            or names != FEATURE_NAMES
            or len(mean) != len(names)
            or len(scale) != len(names)
            or len(coefficients) != len(names) + 1
            or any(value <= 0.0 for value in scale)
            or not cases
            or self.case_id in cases
            or len(cases) != len(set(cases))
            or (observation_hashes and len(observation_hashes) != len(cases))
            or iterations > int(self.max_iterations)
            or _finite(self.alpha, "alpha") != RIDGE_ALPHA
            or int(self.max_iterations) != IRLS_MAX_ITERATIONS
            or _finite(self.tolerance, "tolerance") != IRLS_CONVERGENCE_TOLERANCE
        ):
            raise ProtocolError("Abstention-router model contract drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(self, "source", _source(target, self.source))
        object.__setattr__(self, "direction", _direction(self.direction))
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "training_case_ids", cases)
        object.__setattr__(self, "training_observation_hashes", observation_hashes)
        object.__setattr__(self, "training_trial_count", _count(self.training_trial_count, "training_trial_count"))
        object.__setattr__(self, "valid_observation_count", _count(self.valid_observation_count, "valid_observation_count"))
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "model_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.target_center, self.case_id, self.source, self.direction

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cdca_directional_correctness_model_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "source": self.source,
            "direction": self.direction,
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "training_case_ids": list(self.training_case_ids),
            "training_observation_hashes": list(self.training_observation_hashes),
            "training_trial_count": self.training_trial_count,
            "valid_observation_count": self.valid_observation_count,
            "converged": bool(self.converged),
            "iterations": self.iterations,
            "alpha": self.alpha,
            "intercept_penalized": False,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "held_case_excluded": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "model_hash": self.model_hash}


@dataclass(frozen=True, order=True)
class CandidateDirectionalScore:
    target_center: str
    case_id: str
    direction: str
    source: str | None
    predicted_correctness: float
    directional_flip_count: int
    case_proxy: float
    donor_prior: float
    final_score: float
    model_hash: str | None
    score_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        source = None if self.source is None else _source(target, self.source)
        model_hash = None if self.model_hash is None else require_sha256(self.model_hash, "model_hash")
        values = tuple(
            _finite(value, role)
            for value, role in (
                (self.predicted_correctness, "predicted_correctness"),
                (self.case_proxy, "case_proxy"),
                (self.donor_prior, "donor_prior"),
                (self.final_score, "final_score"),
            )
        )
        if target not in CENTERS or not 0.0 <= values[0] <= 1.0:
            raise ProtocolError("Abstention-router predicted correctness drifted.")
        if source is None and (any(value != 0.0 for value in values) or model_hash is not None):
            raise ProtocolError("Abstention-router OFF score must be the exact zero baseline.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(self, "direction", _direction(self.direction))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "predicted_correctness", values[0])
        object.__setattr__(self, "directional_flip_count", _count(self.directional_flip_count, "directional_flip_count"))
        object.__setattr__(self, "case_proxy", values[1])
        object.__setattr__(self, "donor_prior", values[2])
        object.__setattr__(self, "final_score", values[3])
        object.__setattr__(self, "model_hash", model_hash)
        object.__setattr__(self, "score_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cdca_candidate_directional_score_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "direction": self.direction,
            "source": self.source,
            "predicted_correctness": self.predicted_correctness,
            "directional_flip_count": self.directional_flip_count,
            "case_proxy": self.case_proxy,
            "donor_prior": self.donor_prior,
            "final_score": self.final_score,
            "model_hash": self.model_hash,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "score_hash": self.score_hash}


@dataclass(frozen=True, order=True)
class DirectionalAbstentionDecision:
    method_id: str
    target_center: str
    case_id: str
    direction: str
    candidate_scores: tuple[CandidateDirectionalScore, ...]
    selected_source: str | None
    decision_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        scores = tuple(self.candidate_scores)
        sources = tuple(score.source for score in scores)
        expected = (None, *candidate_sources(self.target_center))
        selected = None if self.selected_source is None else str(self.selected_source)
        if (
            self.method_id not in PRE_TERMINAL_METHOD_IDS
            and self.method_id not in {
                "G_directional_matched",
                "CDCA_case_proxy_only",
                *DESCRIPTIVE_METHOD_IDS,
            }
        ):
            raise ProtocolError("Abstention-router decision method drifted.")
        if sources != expected or selected not in expected:
            raise ProtocolError("Abstention-router all-eight-plus-OFF topology drifted.")
        if any(
            (score.target_center, score.case_id, score.direction)
            != (self.target_center, self.case_id, self.direction)
            for score in scores
        ):
            raise ProtocolError("Abstention-router decision score identity drifted.")
        maximum = max(score.final_score for score in scores)
        expected_selected = next(
            score.source
            for score in scores
            if maximum - score.final_score <= TIE_TOLERANCE
        )
        if selected != expected_selected:
            raise ProtocolError(
                "Abstention-router decision violates OFF-first numeric selection."
            )
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(self, "direction", _direction(self.direction))
        object.__setattr__(self, "candidate_scores", scores)
        object.__setattr__(self, "selected_source", selected)
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cdca_directional_abstention_decision_v1",
            "method_id": self.method_id,
            "target_center": self.target_center,
            "case_id": self.case_id,
            "direction": self.direction,
            "candidate_scores": [score.to_payload() for score in self.candidate_scores],
            "selected_source": self.selected_source,
            "selection_order": "OFF_then_numeric_source",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_hash": self.decision_hash}


@dataclass(frozen=True, order=True)
class CaseAbstentionDecision:
    method_id: str
    target_center: str
    case_id: str
    zero_to_one: DirectionalAbstentionDecision
    one_to_zero: DirectionalAbstentionDecision
    decision_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        expected = (self.method_id, self.target_center, self.case_id)
        if (
            (self.zero_to_one.method_id, self.zero_to_one.target_center, self.zero_to_one.case_id) != expected
            or (self.one_to_zero.method_id, self.one_to_zero.target_center, self.one_to_zero.case_id) != expected
            or self.zero_to_one.direction != "zero_to_one"
            or self.one_to_zero.direction != "one_to_zero"
        ):
            raise ProtocolError("Abstention-router paired decision drifted.")
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str]:
        return self.method_id, self.target_center, self.case_id

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cdca_case_abstention_decision_v1",
            "method_id": self.method_id,
            "target_center": self.target_center,
            "case_id": self.case_id,
            "zero_to_one": self.zero_to_one.to_payload(),
            "one_to_zero": self.one_to_zero.to_payload(),
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_hash": self.decision_hash}


@dataclass(frozen=True, order=True)
class MethodPrediction:
    target_center: str
    case_id: str
    sample_id: str
    method_id: str
    probability: float
    hard_prediction: int
    baseline_hard_prediction: int
    selected_source: str | None
    prediction_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        probability = _finite(self.probability, "method probability")
        hard = _count(self.hard_prediction, "hard_prediction")
        baseline = _count(self.baseline_hard_prediction, "baseline_hard_prediction")
        if (
            target not in CENTERS
            or not 0.0 <= probability <= 1.0
            or hard not in (0, 1)
            or baseline not in (0, 1)
            or hard != int(probability >= 0.5)
        ):
            raise ProtocolError("Abstention-router method prediction drifted.")
        source = None if self.selected_source is None else _source(self.target_center, self.selected_source)
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(self, "sample_id", _text(self.sample_id, "sample_id"))
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "hard_prediction", hard)
        object.__setattr__(self, "baseline_hard_prediction", baseline)
        object.__setattr__(self, "selected_source", source)
        object.__setattr__(self, "prediction_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.method_id, self.target_center, self.case_id, self.sample_id

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cdca_method_prediction_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "method_id": self.method_id,
            "probability": self.probability,
            "hard_prediction": self.hard_prediction,
            "baseline_hard_prediction": self.baseline_hard_prediction,
            "selected_source": self.selected_source,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prediction_hash": self.prediction_hash}


__all__ = (
    "BinaryLabel",
    "CandidateDirectionalScore",
    "CaseAbstentionDecision",
    "DirectionalAbstentionDecision",
    "DirectionalCorrectnessModel",
    "DirectionalCorrectnessObservation",
    "DirectionalGain",
    "DonorDirectionalPrior",
    "LabelFreeDirectionalFeatures",
    "MethodPrediction",
    "SupportClassDenominators",
)
