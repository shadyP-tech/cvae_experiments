"""Deterministic B/U/A0/A1 action library owned by this diagnostic."""

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
    PHYSICAL_ACTION_COUNT_PER_TARGET,
    SELECTED_COUNT_PER_CLASS,
    U_ACTION_ID,
    U_COUNT_PER_SOURCE_CLASS,
    candidate_sources,
    geometry_action_id,
)
from .hashing import canonical_hash


@dataclass(frozen=True)
class ActionSpec:
    target_center: str
    action_id: str
    geometry_id: str | None
    selected_source: str | None
    counts_by_class: Mapping[int, Mapping[str, int]]
    sample_weight_by_source: Mapping[str, float]
    physical_fit_required: bool = True
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        sources = candidate_sources(target)
        counts = {
            int(label): MappingProxyType(
                {str(source): int(value) for source, value in values.items()}
            )
            for label, values in self.counts_by_class.items()
        }
        weights = MappingProxyType(
            {
                str(source): float(value)
                for source, value in self.sample_weight_by_source.items()
            }
        )
        if tuple(counts) != BINARY_CLASSES or any(
            tuple(counts[label]) != sources for label in BINARY_CLASSES
        ) or tuple(weights) != sources:
            raise ProtocolError("Prediction-only action composition drifted.")
        if self.action_id == B_ACTION_ID:
            expected_count = B_COUNT_PER_SOURCE_CLASS
            expected_weights = {source: 1.0 for source in sources}
            valid_identity = self.geometry_id is None and self.selected_source is None
            expected_mass = 1_024.0
        elif self.action_id == U_ACTION_ID:
            expected_count = U_COUNT_PER_SOURCE_CLASS
            expected_weights = {source: 1.0 for source in sources}
            valid_identity = self.geometry_id is None and self.selected_source is None
            expected_mass = 1_152.0
        else:
            expected_count = None
            valid_identity = (
                self.geometry_id in GEOMETRY_IDS
                and self.selected_source in sources
                and self.action_id
                == geometry_action_id(self.geometry_id, self.selected_source)
            )
            expected_weights = {
                source: (
                    1.0
                    if self.geometry_id == "A0"
                    else A1_SELECTED_SAMPLE_WEIGHT
                    if source == self.selected_source
                    else A1_OTHER_SAMPLE_WEIGHT
                )
                for source in sources
            }
            expected_mass = 1_152.0
        if not valid_identity or self.physical_fit_required is not True:
            raise ProtocolError("Prediction-only action identity drifted.")
        for label in BINARY_CLASSES:
            expected_counts = {
                source: (
                    expected_count
                    if expected_count is not None
                    else SELECTED_COUNT_PER_CLASS
                    if source == self.selected_source
                    else OTHER_COUNT_PER_CLASS
                )
                for source in sources
            }
            if dict(counts[label]) != expected_counts:
                raise ProtocolError("Prediction-only action row counts drifted.")
            effective = math.fsum(
                counts[label][source] * weights[source] for source in sources
            )
            if not math.isclose(effective, expected_mass, rel_tol=0.0, abs_tol=1e-12):
                raise ProtocolError("Prediction-only action effective mass drifted.")
        if dict(weights) != expected_weights:
            raise ProtocolError("Prediction-only action sample weights drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "counts_by_class", MappingProxyType(counts))
        object.__setattr__(self, "sample_weight_by_source", weights)
        object.__setattr__(self, "action_hash", canonical_hash(self._unhashed_payload()))

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_prediction_only_action_spec_v1",
            "target_center": self.target_center,
            "action_id": self.action_id,
            "geometry_id": self.geometry_id,
            "selected_source": self.selected_source,
            "counts_by_class": {
                str(label): dict(self.counts_by_class[label])
                for label in BINARY_CLASSES
            },
            "sample_weight_by_source": dict(self.sample_weight_by_source),
            "physical_fit_required": True,
            "outer_target_expert_excluded": True,
            "seed_repetitions_selectable": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "action_hash": self.action_hash}


def _counts(
    sources: tuple[str, ...], selected: str | None, selected_count: int, other_count: int
) -> dict[int, dict[str, int]]:
    values = {
        source: selected_count if source == selected else other_count
        for source in sources
    }
    return {label: dict(values) for label in BINARY_CLASSES}


@lru_cache(maxsize=len(CENTERS))
def actions_for_target(target: object) -> tuple[ActionSpec, ...]:
    target_id = str(target)
    sources = candidate_sources(target_id)
    actions: list[ActionSpec] = [
        ActionSpec(
            target_id,
            B_ACTION_ID,
            None,
            None,
            _counts(sources, None, B_COUNT_PER_SOURCE_CLASS, B_COUNT_PER_SOURCE_CLASS),
            {source: 1.0 for source in sources},
        ),
        ActionSpec(
            target_id,
            U_ACTION_ID,
            None,
            None,
            _counts(sources, None, U_COUNT_PER_SOURCE_CLASS, U_COUNT_PER_SOURCE_CLASS),
            {source: 1.0 for source in sources},
        ),
    ]
    for geometry in GEOMETRY_IDS:
        for source in sources:
            actions.append(
                ActionSpec(
                    target_id,
                    geometry_action_id(geometry, source),
                    geometry,
                    source,
                    _counts(
                        sources,
                        source,
                        SELECTED_COUNT_PER_CLASS,
                        OTHER_COUNT_PER_CLASS,
                    ),
                    {
                        candidate: (
                            1.0
                            if geometry == "A0"
                            else A1_SELECTED_SAMPLE_WEIGHT
                            if candidate == source
                            else A1_OTHER_SAMPLE_WEIGHT
                        )
                        for candidate in sources
                    },
                )
            )
    result = tuple(actions)
    if len(result) != PHYSICAL_ACTION_COUNT_PER_TARGET or len(
        {action.action_id for action in result}
    ) != len(result):
        raise AssertionError("Prediction-only action topology drifted.")
    return result


@lru_cache(maxsize=1)
def build_action_library() -> tuple[ActionSpec, ...]:
    return tuple(
        action for target in CENTERS for action in actions_for_target(target)
    )


def action_library_payload() -> dict[str, object]:
    rows = {
        target: [action.to_payload() for action in actions_for_target(target)]
        for target in CENTERS
    }
    unhashed = {
        "schema_version": "fixed_bank_prediction_only_action_library_v1",
        "targets": rows,
        "geometry_selection_used": False,
        "fit_count_per_seed_pair": len(CENTERS) * PHYSICAL_ACTION_COUNT_PER_TARGET,
    }
    return {**unhashed, "action_library_hash": canonical_hash(unhashed)}


__all__ = (
    "ActionSpec",
    "action_library_payload",
    "actions_for_target",
    "build_action_library",
)
