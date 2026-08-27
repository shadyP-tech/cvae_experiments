"""Read-only adapter over the neutral 810-cell fixed-bank prediction store."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import Mapping

import numpy as np

from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_contracts import (
    PredictionStore,
)

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from .contracts import (
    CENTERS,
    GENERATION_SEEDS,
    PHYSICAL_CELL_COUNT,
    TRAINING_SEEDS,
    array_sha256,
    physical_action_ids,
    probability_vector,
)


SEED_PAIRS = tuple(
    (training_seed, generation_seed)
    for training_seed in TRAINING_SEEDS
    for generation_seed in GENERATION_SEEDS
)


@dataclass(frozen=True, slots=True, eq=False)
class ExactNineActionView:
    target_center: str
    action_id: str
    sample_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    seed_probabilities: np.ndarray
    mean_probability: np.ndarray
    seed_standard_deviation: np.ndarray
    positive_vote_fraction: np.ndarray
    view_hash: str

    def __post_init__(self) -> None:
        sample_ids = tuple(str(value) for value in self.sample_ids)
        case_ids = tuple(str(value) for value in self.case_ids)
        seeds = np.ascontiguousarray(self.seed_probabilities, dtype=np.float64)
        mean = probability_vector(self.mean_probability, expected_length=len(sample_ids))
        standard_deviation = np.ascontiguousarray(
            self.seed_standard_deviation, dtype=np.float64
        )
        votes = probability_vector(
            self.positive_vote_fraction, expected_length=len(sample_ids)
        )
        if (
            self.target_center not in CENTERS
            or self.action_id not in physical_action_ids(self.target_center)
            or not sample_ids
            or len(sample_ids) != len(case_ids)
            or len(set(sample_ids)) != len(sample_ids)
            or seeds.shape != (len(SEED_PAIRS), len(sample_ids))
            or standard_deviation.shape != (len(sample_ids),)
            or not np.isfinite(seeds).all()
            or not np.isfinite(standard_deviation).all()
            or np.any(standard_deviation < 0.0)
            or not np.array_equal(mean, np.mean(seeds, axis=0, dtype=np.float64))
            or not np.array_equal(
                standard_deviation, np.std(seeds, axis=0, ddof=0, dtype=np.float64)
            )
            or not np.array_equal(
                votes, np.mean(seeds >= 0.5, axis=0, dtype=np.float64)
            )
        ):
            raise GovernanceError("SCALE-BP v2 exact-nine view drifted.")
        seeds.setflags(write=False)
        standard_deviation.setflags(write=False)
        payload = {
            "schema_version": "scale_bp_v2_exact_nine_action_view_v1",
            "target_center": self.target_center,
            "action_id": self.action_id,
            "sample_ids": sample_ids,
            "case_ids": case_ids,
            "seed_probability_sha256": array_sha256(seeds),
            "mean_probability_sha256": array_sha256(mean),
            "seed_standard_deviation_sha256": array_sha256(standard_deviation),
            "positive_vote_fraction_sha256": array_sha256(votes),
            "label_free": True,
        }
        expected_hash = canonical_hash(payload)
        if str(self.view_hash) != expected_hash:
            raise GovernanceError("SCALE-BP v2 exact-nine view hash drifted.")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "case_ids", case_ids)
        object.__setattr__(self, "seed_probabilities", seeds)
        object.__setattr__(self, "mean_probability", mean)
        object.__setattr__(self, "seed_standard_deviation", standard_deviation)
        object.__setattr__(self, "positive_vote_fraction", votes)


@dataclass(frozen=True)
class PhysicalStoreAdapter:
    """Zero-I/O validated view; arrays remain owned by the neutral store."""

    store: PredictionStore
    adapter_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.store, PredictionStore):
            raise GovernanceError("SCALE-BP v2 requires the neutral PredictionStore.")
        expected = tuple(
            (target, action, training_seed, generation_seed)
            for target in CENTERS
            for training_seed in TRAINING_SEEDS
            for generation_seed in GENERATION_SEEDS
            for action in physical_action_ids(target)
        )
        actual = tuple(cell.key for cell in self.store.cells)
        if (
            len(actual) != PHYSICAL_CELL_COUNT
            or len(set(actual)) != PHYSICAL_CELL_COUNT
            or actual != expected
            or tuple(self.store.rows_by_center) != CENTERS
            or tuple(self.store.case_ids_by_center) != CENTERS
        ):
            raise GovernanceError("SCALE-BP v2 physical 810-cell topology drifted.")
        object.__setattr__(
            self,
            "adapter_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_physical_store_adapter_v1",
                    "neutral_store_hash": self.store.store_hash,
                    "physical_cell_count": PHYSICAL_CELL_COUNT,
                    "target_expert_excluded": True,
                    "labels_consumed": False,
                }
            ),
        )

    @cached_property
    def case_indices_by_center(self) -> Mapping[str, Mapping[str, tuple[int, ...]]]:
        outer: dict[str, Mapping[str, tuple[int, ...]]] = {}
        for center in CENTERS:
            grouped: dict[str, list[int]] = {}
            for index, case_id in enumerate(self.store.case_ids_by_center[center]):
                grouped.setdefault(str(case_id), []).append(index)
            outer[center] = MappingProxyType(
                {case_id: tuple(indices) for case_id, indices in grouped.items()}
            )
        return MappingProxyType(outer)

    def case_ids(self, target_center: object) -> tuple[str, ...]:
        target = str(target_center)
        try:
            return tuple(self.case_indices_by_center[target])
        except KeyError as exc:
            raise GovernanceError("SCALE-BP v2 physical target is unknown.") from exc

    def case_indices(self, target_center: object, case_id: object) -> tuple[int, ...]:
        target, case = str(target_center), str(case_id)
        try:
            return self.case_indices_by_center[target][case]
        except KeyError as exc:
            raise GovernanceError("SCALE-BP v2 physical case is absent.") from exc

    def exact_nine_view(
        self,
        target_center: object,
        action_id: object,
        *,
        case_id: object | None = None,
    ) -> ExactNineActionView:
        target, action = str(target_center), str(action_id)
        if action not in physical_action_ids(target):
            raise GovernanceError("SCALE-BP v2 physical action is outside the target menu.")
        if case_id is None:
            indices = tuple(range(len(self.store.rows_by_center[target])))
        else:
            indices = self.case_indices(target, case_id)
        index_array = np.asarray(indices, dtype=np.int64)
        seeds = np.stack(
            [
                self.store.probabilities(target, action, training, generation)[
                    index_array
                ]
                for training, generation in SEED_PAIRS
            ],
            axis=0,
        ).astype(np.float64, copy=False)
        mean = np.mean(seeds, axis=0, dtype=np.float64)
        standard_deviation = np.std(seeds, axis=0, ddof=0, dtype=np.float64)
        votes = np.mean(seeds >= 0.5, axis=0, dtype=np.float64)
        sample_ids = tuple(self.store.rows_by_center[target][index] for index in indices)
        case_ids = tuple(self.store.case_ids_by_center[target][index] for index in indices)
        payload = {
            "schema_version": "scale_bp_v2_exact_nine_action_view_v1",
            "target_center": target,
            "action_id": action,
            "sample_ids": sample_ids,
            "case_ids": case_ids,
            "seed_probability_sha256": array_sha256(seeds),
            "mean_probability_sha256": array_sha256(mean),
            "seed_standard_deviation_sha256": array_sha256(standard_deviation),
            "positive_vote_fraction_sha256": array_sha256(votes),
            "label_free": True,
        }
        return ExactNineActionView(
            target,
            action,
            sample_ids,
            case_ids,
            seeds,
            mean,
            standard_deviation,
            votes,
            canonical_hash(payload),
        )


def adapt_prediction_store(store: PredictionStore) -> PhysicalStoreAdapter:
    return PhysicalStoreAdapter(store)


__all__ = (
    "ExactNineActionView",
    "PhysicalStoreAdapter",
    "SEED_PAIRS",
    "adapt_prediction_store",
)
