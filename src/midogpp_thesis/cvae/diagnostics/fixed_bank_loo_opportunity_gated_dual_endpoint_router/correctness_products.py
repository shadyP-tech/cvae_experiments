"""Immutable DTOs for the opportunity-gated correctness endpoint."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    DIRECTION_IDS,
    FEATURE_NAMES,
    IRLS_CONVERGENCE_TOLERANCE,
    IRLS_ETA_CLIP,
    IRLS_MAX_ITERATIONS,
    IRLS_PROBABILITY_CLIP,
    RIDGE_ALPHA,
    candidate_sources,
)
from .hashing import canonical_hash, require_sha256


def finite_tuple(values: Sequence[object], role: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ProtocolError(f"OGDE {role} contains nonfinite values.")
    return result


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
        names = tuple(str(value) for value in self.feature_names)
        values = finite_tuple(self.values, "feature vector")
        flips = int(self.directional_flip_count)
        size = int(self.case_size)
        if (
            self.source not in candidate_sources(self.target_center)
            or self.direction not in DIRECTION_IDS
            or not self.case_id
            or names != FEATURE_NAMES
            or len(values) != len(FEATURE_NAMES)
            or flips < 0
            or size <= 0
            or flips > size
        ):
            raise ProtocolError("OGDE label-free directional feature drifted.")
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
            "schema_version": "fixed_bank_ogde_label_free_directional_features_v1",
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
        values = finite_tuple(self.feature_values, "correctness observation")
        successes = int(self.successes)
        trials = int(self.trials)
        if (
            self.source not in candidate_sources(self.target_center)
            or self.direction not in DIRECTION_IDS
            or not self.route_case_id
            or not self.support_case_id
            or self.route_case_id == self.support_case_id
            or len(values) != len(FEATURE_NAMES)
            or successes < 0
            or trials < 0
            or successes > trials
        ):
            raise ProtocolError("OGDE correctness observation drifted.")
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "successes", successes)
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "observation_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return self.target_center, self.route_case_id, self.support_case_id, self.source, self.direction

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_correctness_observation_v1",
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
        positive = int(self.n_positive)
        negative = int(self.n_negative)
        if (
            self.target_center not in CENTERS
            or not self.route_case_id
            or not cases
            or self.route_case_id in cases
            or len(cases) != len(set(cases))
            or positive <= 0
            or negative <= 0
        ):
            raise ProtocolError("OGDE support denominators drifted.")
        object.__setattr__(self, "support_case_ids", cases)
        object.__setattr__(self, "n_positive", positive)
        object.__setattr__(self, "n_negative", negative)
        object.__setattr__(self, "denominator_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_support_denominators_v1",
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
class DirectionalCorrectnessModel:
    target_center: str
    case_id: str
    source: str
    direction: str
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    training_case_ids: tuple[str, ...]
    training_observation_hashes: tuple[str, ...]
    training_trial_count: int
    valid_observation_count: int
    converged: bool
    iterations: int
    model_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        mean = finite_tuple(self.feature_mean, "model mean")
        scale = finite_tuple(self.feature_scale, "model scale")
        coefficients = finite_tuple(self.coefficients, "model coefficients")
        cases = tuple(sorted(str(value) for value in self.training_case_ids))
        hashes = tuple(require_sha256(value, "training_observation_hash") for value in self.training_observation_hashes)
        if (
            self.source not in candidate_sources(self.target_center)
            or self.direction not in DIRECTION_IDS
            or not self.case_id
            or len(mean) != len(FEATURE_NAMES)
            or len(scale) != len(FEATURE_NAMES)
            or len(coefficients) != len(FEATURE_NAMES) + 1
            or any(value <= 0.0 for value in scale)
            or not cases
            or self.case_id in cases
            or len(cases) != len(set(cases))
            or len(hashes) != len(cases)
            or int(self.training_trial_count) < 0
            or int(self.valid_observation_count) < 0
            or not 0 <= int(self.iterations) <= IRLS_MAX_ITERATIONS
        ):
            raise ProtocolError("OGDE correctness model contract drifted.")
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "training_case_ids", cases)
        object.__setattr__(self, "training_observation_hashes", hashes)
        object.__setattr__(self, "training_trial_count", int(self.training_trial_count))
        object.__setattr__(self, "valid_observation_count", int(self.valid_observation_count))
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "model_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.target_center, self.case_id, self.source, self.direction

    @property
    def is_valid(self) -> bool:
        return bool(self.converged and self.training_trial_count > 0)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_directional_correctness_model_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "source": self.source,
            "direction": self.direction,
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "training_case_ids": list(self.training_case_ids),
            "training_observation_hashes": list(self.training_observation_hashes),
            "training_trial_count": self.training_trial_count,
            "valid_observation_count": self.valid_observation_count,
            "converged": bool(self.converged),
            "iterations": self.iterations,
            "alpha": RIDGE_ALPHA,
            "intercept_penalized": False,
            "max_iterations": IRLS_MAX_ITERATIONS,
            "tolerance": IRLS_CONVERGENCE_TOLERANCE,
            "eta_clip": IRLS_ETA_CLIP,
            "probability_clip": IRLS_PROBABILITY_CLIP,
            "held_case_excluded": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "model_hash": self.model_hash}


__all__ = (
    "DirectionalCorrectnessModel",
    "DirectionalCorrectnessObservation",
    "LabelFreeDirectionalFeatures",
    "SupportClassDenominators",
    "finite_tuple",
)
