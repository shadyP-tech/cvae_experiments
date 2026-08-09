"""Exact-nine probability endpoint and label-free support-shift contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from ..residual_topup.hashing import array_sha256, canonical_sha256
from .row_contracts import (
    _bounded_utility,
    _canonical_text,
    _nonnegative,
    _seed,
)


ENSEMBLE_SEED_KEYS = tuple(product(TRAINING_SEEDS, GENERATION_SEEDS))
ENSEMBLE_SEED_PAIR_COUNT = len(ENSEMBLE_SEED_KEYS)
ENSEMBLE_THRESHOLD = 0.5
ENSEMBLE_ENDPOINT_SEMANTICS = (
    "arithmetic_mean_of_exact_nine_positive_class_probability_vectors_"
    "thresholded_at_0_5_then_balanced_accuracy"
)
ENSEMBLE_UTILITY_SEMANTICS = (
    "candidate_probability_ensemble_bacc_minus_base_probability_ensemble_bacc"
)
SUPPORT_ACTION_PROBABILITY_SHIFT_NAME = (
    "mean_support_row_absolute_exact_nine_ensemble_probability_shift_v2"
)
SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA = (
    "midogpp_utility_aligned_support_action_probability_shift_v2"
)
SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS = (
    "label_free_mean_over_support_rows_of_absolute_difference_between_exact_"
    "nine_seed_cell_mean_tail_and_base_positive_class_probabilities_v2"
)
SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS = (
    "descriptive_only_per_seed_cell_mean_over_support_rows_of_absolute_tail_"
    "minus_base_positive_class_probability_v1"
)


@dataclass(frozen=True)
class SeedProbabilityVector:
    """One canonical seed cell of positive-class probabilities.

    ``row_identity_hash`` binds order and membership of the scored rows;
    ``prediction_provenance_hash`` binds the upstream action/checkpoint/seal.
    Raw vectors are copied to immutable float64 arrays so later caller mutation
    cannot alter an already hashed contract.
    """

    training_seed: int
    generation_seed: int
    row_identity_hash: str
    prediction_provenance_hash: str
    positive_class_probabilities: np.ndarray
    probability_hash: str = field(init=False)
    vector_hash: str = field(init=False)

    def __post_init__(self) -> None:
        training_seed = _seed(self.training_seed, TRAINING_SEEDS, "training_seed")
        generation_seed = _seed(
            self.generation_seed, GENERATION_SEEDS, "generation_seed"
        )
        row_hash = _canonical_text(self.row_identity_hash, "row_identity_hash")
        provenance_hash = _canonical_text(
            self.prediction_provenance_hash, "prediction_provenance_hash"
        )
        probabilities = np.asarray(
            self.positive_class_probabilities, dtype=np.float64
        ).copy()
        if (
            probabilities.ndim != 1
            or not len(probabilities)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
        ):
            raise ProtocolError(
                "Positive-class probabilities must be a nonempty finite vector in [0, 1]."
            )
        probabilities.setflags(write=False)
        probability_hash = array_sha256(probabilities)
        payload = {
            "schema_version": "midogpp_utility_aligned_seed_probability_vector_v1",
            "training_seed": training_seed,
            "generation_seed": generation_seed,
            "row_identity_hash": row_hash,
            "prediction_provenance_hash": provenance_hash,
            "row_count": len(probabilities),
            "probability_sha256": probability_hash,
        }
        object.__setattr__(self, "training_seed", training_seed)
        object.__setattr__(self, "generation_seed", generation_seed)
        object.__setattr__(self, "row_identity_hash", row_hash)
        object.__setattr__(self, "prediction_provenance_hash", provenance_hash)
        object.__setattr__(self, "positive_class_probabilities", probabilities)
        object.__setattr__(self, "probability_hash", probability_hash)
        object.__setattr__(self, "vector_hash", canonical_sha256(payload))

    @property
    def seed_key(self) -> tuple[int, int]:
        return self.training_seed, self.generation_seed

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_seed_probability_vector_v1",
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "row_identity_hash": self.row_identity_hash,
            "prediction_provenance_hash": self.prediction_provenance_hash,
            "row_count": len(self.positive_class_probabilities),
            "probability_sha256": self.probability_hash,
            "vector_hash": self.vector_hash,
        }


@dataclass(frozen=True)
class ProbabilityEnsembleEndpoint:
    """The primary endpoint for one action after exact-nine averaging."""

    row_identity_hash: str
    label_hash: str
    seed_keys: tuple[tuple[int, int], ...]
    component_vector_hashes: tuple[str, ...]
    mean_positive_probabilities: np.ndarray
    predictions: np.ndarray
    balanced_accuracy: float
    endpoint_hash: str
    threshold: float = ENSEMBLE_THRESHOLD
    endpoint_semantics: str = ENSEMBLE_ENDPOINT_SEMANTICS

    @property
    def row_count(self) -> int:
        return len(self.predictions)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                "midogpp_utility_aligned_probability_ensemble_endpoint_v1"
            ),
            "row_identity_hash": self.row_identity_hash,
            "label_sha256": self.label_hash,
            "seed_pair_count": len(self.seed_keys),
            "seed_keys": [list(key) for key in self.seed_keys],
            "component_vector_hashes": list(self.component_vector_hashes),
            "row_count": self.row_count,
            "mean_probability_sha256": array_sha256(
                self.mean_positive_probabilities
            ),
            "prediction_sha256": array_sha256(self.predictions),
            "balanced_accuracy": self.balanced_accuracy,
            "threshold": self.threshold,
            "endpoint_semantics": self.endpoint_semantics,
            "endpoint_hash": self.endpoint_hash,
        }


@dataclass(frozen=True)
class SupportActionProbabilityShift:
    """One label-free ensemble-first scalar plus technical-seed diagnostics."""

    row_identity_hash: str
    seed_keys: tuple[tuple[int, int], ...]
    base_component_vector_hashes: tuple[str, ...]
    tail_component_vector_hashes: tuple[str, ...]
    per_seed_mean_absolute_shifts: tuple[float, ...]
    base_ensemble_probability_hash: str
    tail_ensemble_probability_hash: str
    ensemble_absolute_difference_hash: str
    value: float
    seed_standard_deviation: float
    seed_minimum: float
    seed_maximum: float
    seed_range: float
    shift_hash: str
    scalar_name: str = SUPPORT_ACTION_PROBABILITY_SHIFT_NAME
    scalar_semantics: str = SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS

    def __post_init__(self) -> None:
        row_identity_hash = _canonical_text(
            self.row_identity_hash, "row_identity_hash"
        )
        try:
            seed_keys = tuple(
                (int(training_seed), int(generation_seed))
                for training_seed, generation_seed in self.seed_keys
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError(
                "Support action shift seed keys are invalid."
            ) from exc
        if seed_keys != ENSEMBLE_SEED_KEYS:
            raise ProtocolError(
                "Support action shift requires canonical exact-nine seeds."
            )
        base_hashes = tuple(
            _canonical_text(value, "base_component_vector_hash")
            for value in self.base_component_vector_hashes
        )
        tail_hashes = tuple(
            _canonical_text(value, "tail_component_vector_hash")
            for value in self.tail_component_vector_hashes
        )
        if (
            len(base_hashes) != ENSEMBLE_SEED_PAIR_COUNT
            or len(tail_hashes) != ENSEMBLE_SEED_PAIR_COUNT
            or len(set(base_hashes)) != ENSEMBLE_SEED_PAIR_COUNT
            or len(set(tail_hashes)) != ENSEMBLE_SEED_PAIR_COUNT
        ):
            raise ProtocolError(
                "Support action shift requires exact-nine unique component hashes."
            )
        shifts = tuple(
            _bounded_utility(value, "per_seed_mean_absolute_shift")
            for value in self.per_seed_mean_absolute_shifts
        )
        if len(shifts) != ENSEMBLE_SEED_PAIR_COUNT:
            raise ProtocolError(
                "Support action shift requires exactly nine per-seed values."
            )
        shift_values = np.asarray(shifts, dtype=np.float64)
        expected_statistics = {
            "seed_standard_deviation": float(
                np.std(shift_values, ddof=0, dtype=np.float64)
            ),
            "seed_minimum": float(np.min(shift_values)),
            "seed_maximum": float(np.max(shift_values)),
        }
        expected_statistics["seed_range"] = (
            expected_statistics["seed_maximum"]
            - expected_statistics["seed_minimum"]
        )
        supplied_statistics = {
            "seed_standard_deviation": _nonnegative(
                self.seed_standard_deviation, "seed_standard_deviation"
            ),
            "seed_minimum": _bounded_utility(self.seed_minimum, "seed_minimum"),
            "seed_maximum": _bounded_utility(self.seed_maximum, "seed_maximum"),
            "seed_range": _nonnegative(self.seed_range, "seed_range"),
        }
        if supplied_statistics != expected_statistics:
            raise ProtocolError(
                "Support action shift summary statistics cannot be reconstructed."
            )
        value = _bounded_utility(self.value, "value")
        # Jensen's inequality is a useful fail-closed check that does not need
        # the raw vectors: |mean(delta)| cannot exceed mean(|delta|).
        if value > float(np.mean(shift_values, dtype=np.float64)) + 1.0e-12:
            raise ProtocolError(
                "Ensemble-first support action shift exceeds its technical-seed bound."
            )
        base_ensemble_hash = _canonical_text(
            self.base_ensemble_probability_hash,
            "base_ensemble_probability_hash",
        )
        tail_ensemble_hash = _canonical_text(
            self.tail_ensemble_probability_hash,
            "tail_ensemble_probability_hash",
        )
        difference_hash = _canonical_text(
            self.ensemble_absolute_difference_hash,
            "ensemble_absolute_difference_hash",
        )
        if (
            self.scalar_name != SUPPORT_ACTION_PROBABILITY_SHIFT_NAME
            or self.scalar_semantics != SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS
        ):
            raise ProtocolError("Support action shift scalar semantics drifted.")
        shift_hash = _canonical_text(self.shift_hash, "shift_hash")
        object.__setattr__(self, "row_identity_hash", row_identity_hash)
        object.__setattr__(self, "seed_keys", seed_keys)
        object.__setattr__(self, "base_component_vector_hashes", base_hashes)
        object.__setattr__(self, "tail_component_vector_hashes", tail_hashes)
        object.__setattr__(self, "per_seed_mean_absolute_shifts", shifts)
        object.__setattr__(
            self, "base_ensemble_probability_hash", base_ensemble_hash
        )
        object.__setattr__(
            self, "tail_ensemble_probability_hash", tail_ensemble_hash
        )
        object.__setattr__(
            self, "ensemble_absolute_difference_hash", difference_hash
        )
        object.__setattr__(self, "value", value)
        for name, value in supplied_statistics.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "shift_hash", shift_hash)
        unhashed = self.to_payload()
        unhashed.pop("shift_hash")
        if canonical_sha256(unhashed) != shift_hash:
            raise ProtocolError("Support action shift aggregate hash drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA
            ),
            "row_identity_hash": self.row_identity_hash,
            "seed_pair_count": len(self.seed_keys),
            "seed_keys": [list(key) for key in self.seed_keys],
            "base_component_vector_hashes": list(
                self.base_component_vector_hashes
            ),
            "tail_component_vector_hashes": list(
                self.tail_component_vector_hashes
            ),
            "per_seed_mean_absolute_shifts": list(
                self.per_seed_mean_absolute_shifts
            ),
            "technical_seed_spread_semantics": (
                SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS
            ),
            "technical_seed_values_may_feed_model": False,
            "base_ensemble_probability_sha256": (
                self.base_ensemble_probability_hash
            ),
            "tail_ensemble_probability_sha256": (
                self.tail_ensemble_probability_hash
            ),
            "ensemble_absolute_difference_sha256": (
                self.ensemble_absolute_difference_hash
            ),
            "value": self.value,
            "seed_standard_deviation": self.seed_standard_deviation,
            "seed_minimum": self.seed_minimum,
            "seed_maximum": self.seed_maximum,
            "seed_range": self.seed_range,
            "scalar_name": self.scalar_name,
            "scalar_semantics": self.scalar_semantics,
            "labels_used": False,
            "shift_hash": self.shift_hash,
        }





__all__ = (
    "ENSEMBLE_ENDPOINT_SEMANTICS",
    "ENSEMBLE_SEED_KEYS",
    "ENSEMBLE_SEED_PAIR_COUNT",
    "ENSEMBLE_THRESHOLD",
    "ENSEMBLE_UTILITY_SEMANTICS",
    "SUPPORT_ACTION_PROBABILITY_SHIFT_NAME",
    "SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA",
    "SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS",
    "SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS",
    "ProbabilityEnsembleEndpoint",
    "SeedProbabilityVector",
    "SupportActionProbabilityShift",
)
