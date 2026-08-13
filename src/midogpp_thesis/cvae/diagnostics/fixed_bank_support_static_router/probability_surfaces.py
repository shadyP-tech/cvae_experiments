"""Exact-nine probability aggregation over the neutral 810-cell store."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .actions import actions_for_target
from .experiment_contracts import (
    CENTERS,
    GENERATION_SEEDS,
    HARD_THRESHOLD,
    SEED_PAIR_COUNT,
    TRAINING_SEEDS,
)
from .input_contracts import canonical_hash
from .products import BinaryPredictionRow


@dataclass(frozen=True)
class ExactNineProbabilitySurface:
    """Read-only exact-nine reductions, never a new classifier fit."""

    prediction: object
    action_ids_by_target: Mapping[str, tuple[str, ...]]
    surface_hash: str

    def __post_init__(self) -> None:
        store = getattr(self.prediction, "store")
        actions = {
            str(target): tuple(str(value) for value in values)
            for target, values in self.action_ids_by_target.items()
        }
        expected = {
            target: tuple(action.action_id for action in actions_for_target(target))
            for target in CENTERS
        }
        unhashed = {
            "schema_version": "fixed_bank_support_static_router_probability_surface_v1",
            "global_prediction_seal_hash": str(getattr(self.prediction, "seal_hash")),
            "probability_store_hash": str(store.store_hash),
            "action_ids_by_target": {key: list(value) for key, value in actions.items()},
            "seed_pair_count": SEED_PAIR_COUNT,
            "aggregation": "float64_arithmetic_mean_over_exact_3x3_seed_grid",
            "threshold": HARD_THRESHOLD,
            "labels_used": False,
        }
        if (
            actions != expected
            or len(getattr(store, "cells")) != 810
            or self.surface_hash != canonical_hash(unhashed)
        ):
            raise ProtocolError("S4 exact-nine probability surface drifted.")
        object.__setattr__(self, "action_ids_by_target", MappingProxyType(actions))

    @cached_property
    def _cache(self) -> dict[tuple[str, str], np.ndarray]:
        return {}

    @property
    def probability_store_hash(self) -> str:
        return str(getattr(self.prediction, "store").store_hash)

    def probabilities(self, target_center: str, action_id: str) -> np.ndarray:
        key = (str(target_center), str(action_id))
        if key[0] not in CENTERS or key[1] not in self.action_ids_by_target[key[0]]:
            raise ProtocolError("S4 exact-nine action key is absent.")
        cached = self._cache.get(key)
        if cached is None:
            store = getattr(self.prediction, "store")
            cells = np.stack(
                [
                    store.probabilities(key[0], key[1], training, generation)
                    for training in TRAINING_SEEDS
                    for generation in GENERATION_SEEDS
                ]
            ).astype(np.float64, copy=False)
            if cells.shape[0] != SEED_PAIR_COUNT or not np.isfinite(cells).all():
                raise ProtocolError("S4 exact-nine seed surface drifted.")
            values = np.mean(cells, axis=0, dtype=np.float64)
            values.setflags(write=False)
            self._cache[key] = values
            cached = values
        return cached

    def predictions(self, target_center: str, action_id: str) -> np.ndarray:
        return np.ascontiguousarray(
            self.probabilities(target_center, action_id) >= HARD_THRESHOLD,
            dtype=np.uint8,
        )

    def case_probabilities(
        self, target_center: str, case_id: str, action_id: str
    ) -> np.ndarray:
        store = getattr(self.prediction, "store")
        cases = np.asarray(store.case_ids_by_center[str(target_center)], dtype=object)
        mask = cases == str(case_id)
        if not np.any(mask):
            raise ProtocolError("S4 probability case is absent.")
        return np.ascontiguousarray(
            self.probabilities(target_center, action_id)[mask], dtype=np.float64
        )


@dataclass(frozen=True)
class PredictionRowIndex:
    """Label-free case/sample index over the exact 10-row physical menu."""

    rows_by_sample: Mapping[tuple[str, str, str], tuple[BinaryPredictionRow, ...]]
    surface_hash: str

    def __post_init__(self) -> None:
        rows = {
            tuple(str(part) for part in key): tuple(value)
            for key, value in self.rows_by_sample.items()
        }
        if (
            not rows
            or any(len(value) != 10 for value in rows.values())
            or any(
                tuple(row.sample_key for row in value) != (key,) * 10
                or tuple(row.action_id for row in value)
                != tuple(action.action_id for action in actions_for_target(key[0]))
                or any(row.probability_surface_hash != self.surface_hash for row in value)
                for key, value in rows.items()
            )
        ):
            raise ProtocolError("S4 prediction-row index is incomplete or misordered.")
        object.__setattr__(self, "rows_by_sample", MappingProxyType(rows))

    def for_labels(self, labels: Sequence[object]) -> tuple[BinaryPredictionRow, ...]:
        result: list[BinaryPredictionRow] = []
        seen: set[tuple[str, str, str]] = set()
        for label in labels:
            key = (
                str(getattr(label, "target_center")),
                str(getattr(label, "case_id")),
                str(getattr(label, "sample_id")),
            )
            if key in seen:
                raise ProtocolError("S4 prediction-index grant labels are duplicated.")
            seen.add(key)
            try:
                result.extend(self.rows_by_sample[key])
            except KeyError as exc:
                raise ProtocolError("S4 prediction-index grant row is absent.") from exc
        if len(result) != 10 * len(seen):
            raise ProtocolError("S4 prediction-index grant coverage drifted.")
        return tuple(result)


def build_exact_nine_surface(prediction: object) -> ExactNineProbabilitySurface:
    actions = {
        target: tuple(action.action_id for action in actions_for_target(target))
        for target in CENTERS
    }
    unhashed = {
        "schema_version": "fixed_bank_support_static_router_probability_surface_v1",
        "global_prediction_seal_hash": str(getattr(prediction, "seal_hash")),
        "probability_store_hash": str(getattr(prediction, "store").store_hash),
        "action_ids_by_target": {key: list(value) for key, value in actions.items()},
        "seed_pair_count": SEED_PAIR_COUNT,
        "aggregation": "float64_arithmetic_mean_over_exact_3x3_seed_grid",
        "threshold": HARD_THRESHOLD,
        "labels_used": False,
    }
    return ExactNineProbabilitySurface(prediction, actions, canonical_hash(unhashed))


def probability_surface_seal_payload(
    surface: ExactNineProbabilitySurface,
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_support_static_router_probability_seal_v1",
        "global_prediction_seal_hash": str(getattr(surface.prediction, "seal_hash")),
        "probability_store_hash": surface.probability_store_hash,
        "probability_surface_hash": surface.surface_hash,
        "prediction_cell_count": len(getattr(surface.prediction, "store").cells),
        "seed_pair_count": SEED_PAIR_COUNT,
        "exact_nine_ensemble_first": True,
        "probability_storage_dtype": "float32",
        "scientific_reduction_dtype": "float64",
        "labels_used": False,
        "sealed_before_fold_plans_and_label_capabilities": True,
    }


def prediction_rows(
    surface: ExactNineProbabilitySurface,
) -> tuple[BinaryPredictionRow, ...]:
    """Expose the exact-nine surface to role-scoped count scoring."""

    store = getattr(surface.prediction, "store")
    rows: list[BinaryPredictionRow] = []
    for target in CENTERS:
        sample_ids = store.rows_by_center[target]
        case_ids = store.case_ids_by_center[target]
        for action_id in surface.action_ids_by_target[target]:
            probabilities = surface.probabilities(target, action_id)
            rows.extend(
                BinaryPredictionRow(
                    target,
                    str(case_id),
                    str(sample_id),
                    action_id,
                    float(probability),
                    surface.surface_hash,
                )
                for sample_id, case_id, probability in zip(
                    sample_ids, case_ids, probabilities, strict=True
                )
            )
    return tuple(rows)


def build_prediction_row_index(
    rows: Sequence[BinaryPredictionRow], *, surface_hash: str
) -> PredictionRowIndex:
    grouped: dict[tuple[str, str, str], list[BinaryPredictionRow]] = {}
    for row in rows:
        grouped.setdefault(row.sample_key, []).append(row)
    return PredictionRowIndex(
        {key: tuple(value) for key, value in grouped.items()}, surface_hash
    )


__all__ = (
    "ExactNineProbabilitySurface",
    "PredictionRowIndex",
    "build_exact_nine_surface",
    "build_prediction_row_index",
    "prediction_rows",
    "probability_surface_seal_payload",
)
