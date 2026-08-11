"""Strict H/q source-development actions.

These actions exist only to create source-OOF development predictions.  The
query centre ``q`` is removed from every synthetic training composition, as is
the outer target ``H``.  A deterministic logistic-only mass normalization
makes each per-class effective mass match the corresponding q=H target action;
the StandardScaler remains unweighted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
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
    CENTERS,
    GEOMETRY_IDS,
    OTHER_COUNT_PER_CLASS,
    SELECTED_COUNT_PER_CLASS,
    U_ACTION_ID,
    U_COUNT_PER_SOURCE_CLASS,
    geometry_action_id,
)
from .hashing import canonical_hash


DEVELOPMENT_ACTION_COUNT_PER_TASK = 16
DEVELOPMENT_ORIENTED_CONTEXT_COUNT = len(CENTERS) * (len(CENTERS) - 1) * 3 * 3
DEVELOPMENT_PHYSICAL_TASK_COUNT = (len(CENTERS) * (len(CENTERS) - 1) // 2) * 3 * 3
DEVELOPMENT_CLASSIFIER_FIT_COUNT = (
    DEVELOPMENT_PHYSICAL_TASK_COUNT * DEVELOPMENT_ACTION_COUNT_PER_TASK
)
DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT = (
    DEVELOPMENT_ORIENTED_CONTEXT_COUNT * DEVELOPMENT_ACTION_COUNT_PER_TASK
)

MASS_NORMALIZATION_BY_ACTION_KIND = MappingProxyType(
    {
        "B": 8.0 / 7.0,
        "U": 8.0 / 7.0,
        "A0": 9.0 / 8.0,
        "A1": 72.0 / 65.0,
    }
)
TARGET_EFFECTIVE_MASS_BY_ACTION_KIND = MappingProxyType(
    {"B": 1_024.0, "U": 1_152.0, "A0": 1_152.0, "A1": 1_152.0}
)


def development_candidate_sources(
    outer_target: object, query_center: object
) -> tuple[str, ...]:
    target, query = str(outer_target), str(query_center)
    if target not in CENTERS or query not in CENTERS or query == target:
        raise ProtocolError("Strict source-OOF H/q identity is invalid.")
    return tuple(center for center in CENTERS if center not in (target, query))


@dataclass(frozen=True)
class DevelopmentActionSpec:
    outer_target: str
    query_center: str
    action_id: str
    geometry_id: str | None
    selected_source: str | None
    counts_by_class: Mapping[int, Mapping[str, int]]
    base_sample_weight_by_source: Mapping[str, float]
    logistic_mass_normalization: float
    scaler_fit_used_sample_weight: bool = False
    action_hash: str = field(init=False)
    orientation_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target, query = str(self.outer_target), str(self.query_center)
        sources = development_candidate_sources(target, query)
        counts = {
            int(label): MappingProxyType(
                {str(source): int(value) for source, value in values.items()}
            )
            for label, values in self.counts_by_class.items()
        }
        base_weights = MappingProxyType(
            {
                str(source): float(value)
                for source, value in self.base_sample_weight_by_source.items()
            }
        )
        if self.action_id == B_ACTION_ID:
            kind, uniform_count = "B", B_COUNT_PER_SOURCE_CLASS
        elif self.action_id == U_ACTION_ID:
            kind, uniform_count = "U", U_COUNT_PER_SOURCE_CLASS
        else:
            kind, uniform_count = str(self.geometry_id), None
        valid_identity = (
            (kind in ("B", "U") and self.geometry_id is None and self.selected_source is None)
            or (
                kind in GEOMETRY_IDS
                and self.selected_source in sources
                and self.action_id == geometry_action_id(kind, self.selected_source)
            )
        )
        expected_normalization = MASS_NORMALIZATION_BY_ACTION_KIND.get(kind)
        if (
            not valid_identity
            or tuple(counts) != BINARY_CLASSES
            or any(tuple(counts[label]) != sources for label in BINARY_CLASSES)
            or tuple(base_weights) != sources
            or expected_normalization is None
            or not math.isclose(
                float(self.logistic_mass_normalization),
                expected_normalization,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or self.scaler_fit_used_sample_weight is not False
        ):
            raise ProtocolError("Strict source-OOF action topology drifted.")
        for label in BINARY_CLASSES:
            expected_counts = {
                source: (
                    uniform_count
                    if uniform_count is not None
                    else SELECTED_COUNT_PER_CLASS
                    if source == self.selected_source
                    else OTHER_COUNT_PER_CLASS
                )
                for source in sources
            }
            expected_base_weights = {
                source: (
                    A1_SELECTED_SAMPLE_WEIGHT
                    if kind == "A1" and source == self.selected_source
                    else A1_OTHER_SAMPLE_WEIGHT
                    if kind == "A1"
                    else 1.0
                )
                for source in sources
            }
            effective_mass = math.fsum(
                counts[label][source]
                * base_weights[source]
                * expected_normalization
                for source in sources
            )
            if (
                dict(counts[label]) != expected_counts
                or dict(base_weights) != expected_base_weights
                or not math.isclose(
                    effective_mass,
                    TARGET_EFFECTIVE_MASS_BY_ACTION_KIND[kind],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ProtocolError("Strict source-OOF action mass drifted.")
        object.__setattr__(self, "outer_target", target)
        object.__setattr__(self, "query_center", query)
        object.__setattr__(self, "counts_by_class", MappingProxyType(counts))
        object.__setattr__(self, "base_sample_weight_by_source", base_weights)
        object.__setattr__(
            self, "action_hash", canonical_hash(self._physical_unhashed_payload())
        )
        object.__setattr__(
            self,
            "orientation_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_strict_source_oof_action_orientation_v1",
                    "outer_target": target,
                    "query_center": query,
                    "physical_action_hash": self.action_hash,
                }
            ),
        )

    @property
    def sample_weight_by_source(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                source: weight * self.logistic_mass_normalization
                for source, weight in self.base_sample_weight_by_source.items()
            }
        )

    @property
    def excluded_pair(self) -> tuple[str, str]:
        return tuple(sorted((self.outer_target, self.query_center)))  # type: ignore[return-value]

    def _physical_unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_strict_source_oof_action_v1",
            "excluded_pair": list(self.excluded_pair),
            "action_id": self.action_id,
            "geometry_id": self.geometry_id,
            "selected_source": self.selected_source,
            "counts_by_class": {
                str(label): dict(self.counts_by_class[label])
                for label in BINARY_CLASSES
            },
            "base_sample_weight_by_source": dict(self.base_sample_weight_by_source),
            "logistic_mass_normalization": self.logistic_mass_normalization,
            "sample_weight_by_source": dict(self.sample_weight_by_source),
            "sample_weight_scope": "logistic_regression_fit_only",
            "scaler_fit_used_sample_weight": False,
            "outer_target_excluded": True,
            "query_center_excluded": True,
            "target_effective_mass_per_class": TARGET_EFFECTIVE_MASS_BY_ACTION_KIND[
                "B" if self.action_id == B_ACTION_ID else "U" if self.action_id == U_ACTION_ID else str(self.geometry_id)
            ],
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._physical_unhashed_payload(),
            "outer_target": self.outer_target,
            "query_center": self.query_center,
            "action_hash": self.action_hash,
            "orientation_hash": self.orientation_hash,
        }


def _counts(
    sources: tuple[str, ...], selected: str | None, selected_count: int, other_count: int
) -> dict[int, dict[str, int]]:
    values = {
        source: selected_count if source == selected else other_count
        for source in sources
    }
    return {label: dict(values) for label in BINARY_CLASSES}


@lru_cache(maxsize=len(CENTERS) * (len(CENTERS) - 1))
def development_actions_for(
    outer_target: object, query_center: object
) -> tuple[DevelopmentActionSpec, ...]:
    target, query = str(outer_target), str(query_center)
    sources = development_candidate_sources(target, query)
    actions: list[DevelopmentActionSpec] = []
    for action_id, count, kind in (
        (B_ACTION_ID, B_COUNT_PER_SOURCE_CLASS, "B"),
        (U_ACTION_ID, U_COUNT_PER_SOURCE_CLASS, "U"),
    ):
        actions.append(
            DevelopmentActionSpec(
                target,
                query,
                action_id,
                None,
                None,
                _counts(sources, None, count, count),
                {source: 1.0 for source in sources},
                MASS_NORMALIZATION_BY_ACTION_KIND[kind],
            )
        )
    for geometry in GEOMETRY_IDS:
        for selected in sources:
            actions.append(
                DevelopmentActionSpec(
                    target,
                    query,
                    geometry_action_id(geometry, selected),
                    geometry,
                    selected,
                    _counts(
                        sources,
                        selected,
                        SELECTED_COUNT_PER_CLASS,
                        OTHER_COUNT_PER_CLASS,
                    ),
                    {
                        source: (
                            A1_SELECTED_SAMPLE_WEIGHT
                            if geometry == "A1" and source == selected
                            else A1_OTHER_SAMPLE_WEIGHT
                            if geometry == "A1"
                            else 1.0
                        )
                        for source in sources
                    },
                    MASS_NORMALIZATION_BY_ACTION_KIND[geometry],
                )
            )
    result = tuple(actions)
    if (
        len(result) != DEVELOPMENT_ACTION_COUNT_PER_TASK
        or len({action.action_id for action in result}) != len(result)
    ):
        raise AssertionError("Strict source-OOF action coverage drifted.")
    return result


@lru_cache(maxsize=1)
def development_action_library_payload() -> dict[str, object]:
    tasks = [
        {
            "outer_target": target,
            "query_center": query,
            "actions": [
                action.to_payload() for action in development_actions_for(target, query)
            ],
        }
        for target in CENTERS
        for query in CENTERS
        if query != target
    ]
    unhashed = {
        "schema_version": "midogpp_strict_source_oof_action_library_v1",
        "tasks": tasks,
        "query_excluded_from_every_composition": True,
        "outer_target_excluded_from_every_composition": True,
        "sample_weight_scope": "logistic_regression_fit_only",
        "scaler_fit_used_sample_weight": False,
        "mass_normalization_by_action_kind": dict(MASS_NORMALIZATION_BY_ACTION_KIND),
        "task_count_per_seed_pair": len(CENTERS) * (len(CENTERS) - 1),
        "physical_task_count_per_seed_pair": len(CENTERS) * (len(CENTERS) - 1) // 2,
        "action_count_per_task": DEVELOPMENT_ACTION_COUNT_PER_TASK,
        "physical_fit_count": DEVELOPMENT_CLASSIFIER_FIT_COUNT,
        "logical_prediction_cell_count": DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT,
        "unordered_excluded_pair_fit_reuse": True,
    }
    return {**unhashed, "action_library_hash": canonical_hash(unhashed)}


__all__ = (
    "DEVELOPMENT_ACTION_COUNT_PER_TASK",
    "DEVELOPMENT_CLASSIFIER_FIT_COUNT",
    "DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT",
    "DEVELOPMENT_ORIENTED_CONTEXT_COUNT",
    "DEVELOPMENT_PHYSICAL_TASK_COUNT",
    "DevelopmentActionSpec",
    "MASS_NORMALIZATION_BY_ACTION_KIND",
    "TARGET_EFFECTIVE_MASS_BY_ACTION_KIND",
    "development_action_library_payload",
    "development_actions_for",
    "development_candidate_sources",
)
