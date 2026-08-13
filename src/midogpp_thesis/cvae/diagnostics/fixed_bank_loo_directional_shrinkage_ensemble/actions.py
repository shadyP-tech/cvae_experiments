"""DCSE-local descriptors for physical B, U, and eight A1 actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .constants import (
    ACTION_COUNT_PER_TARGET,
    A1_OTHER_SAMPLE_WEIGHT,
    A1_SELECTED_SAMPLE_WEIGHT,
    B_ACTION_ID,
    B_COUNT_PER_SOURCE_CLASS,
    BINARY_CLASSES,
    CENTERS,
    OTHER_COUNT_PER_CLASS,
    SELECTED_COUNT_PER_CLASS,
    U_ACTION_ID,
    U_COUNT_PER_SOURCE_CLASS,
    a1_action_id,
    candidate_sources,
)
from .hashing import canonical_hash


ACTION_SCHEMA = "fixed_bank_loo_directional_shrinkage_ensemble_action_v1"


@dataclass(frozen=True)
class ActionSpec:
    target_center: str
    action_id: str
    selected_source: str | None
    counts_by_class: Mapping[int, Mapping[str, int]]
    sample_weight_by_source: Mapping[str, float]
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        sources = candidate_sources(target)
        if tuple(self.counts_by_class) != BINARY_CLASSES:
            raise ProtocolError("DCSE action class order drifted.")
        counts: dict[int, Mapping[str, int]] = {}
        for label in BINARY_CLASSES:
            raw = self.counts_by_class[label]
            if tuple(raw) != sources:
                raise ProtocolError("DCSE action source order drifted.")
            values = {source: int(raw[source]) for source in sources}
            if any(isinstance(raw[source], bool) or values[source] <= 0 for source in sources):
                raise ProtocolError("DCSE action counts must be positive integers.")
            counts[label] = MappingProxyType(values)
        weights = {source: float(self.sample_weight_by_source[source]) for source in sources}
        if tuple(self.sample_weight_by_source) != sources or any(value <= 0.0 for value in weights.values()):
            raise ProtocolError("DCSE action weights/order drifted.")
        selected = None if self.selected_source is None else str(self.selected_source)
        if self.action_id == B_ACTION_ID:
            expected_counts = {source: B_COUNT_PER_SOURCE_CLASS for source in sources}
            expected_weights = {source: 1.0 for source in sources}
            if selected is not None:
                raise ProtocolError("B cannot select a source.")
        elif self.action_id == U_ACTION_ID:
            expected_counts = {source: U_COUNT_PER_SOURCE_CLASS for source in sources}
            expected_weights = {source: 1.0 for source in sources}
            if selected is not None:
                raise ProtocolError("U cannot select a source.")
        else:
            if selected not in sources or self.action_id != a1_action_id(selected):
                raise ProtocolError("A1 action/source identity drifted.")
            expected_counts = {
                source: SELECTED_COUNT_PER_CLASS if source == selected else OTHER_COUNT_PER_CLASS
                for source in sources
            }
            expected_weights = {
                source: A1_SELECTED_SAMPLE_WEIGHT if source == selected else A1_OTHER_SAMPLE_WEIGHT
                for source in sources
            }
        if any(dict(counts[label]) != expected_counts for label in BINARY_CLASSES):
            raise ProtocolError("DCSE physical action counts drifted.")
        if weights != expected_weights:
            raise ProtocolError("DCSE physical action weights drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "selected_source", selected)
        object.__setattr__(self, "counts_by_class", MappingProxyType(counts))
        object.__setattr__(self, "sample_weight_by_source", MappingProxyType(weights))
        object.__setattr__(self, "action_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": ACTION_SCHEMA,
            "target_center": self.target_center,
            "action_id": self.action_id,
            "selected_source": self.selected_source,
            "geometry_id": None if self.action_id in {B_ACTION_ID, U_ACTION_ID} else "A1",
            "counts_by_class": {
                str(label): dict(self.counts_by_class[label]) for label in BINARY_CLASSES
            },
            "sample_weight_by_source": dict(self.sample_weight_by_source),
            "physical_fit_required": True,
            "exact_nine_seed_pairs_required": True,
            "target_expert_excluded": True,
            "reused_stage90_predictions": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "action_hash": self.action_hash}


def _action(target: str, action_id: str, selected: str | None) -> ActionSpec:
    sources = candidate_sources(target)
    if selected is None:
        count = B_COUNT_PER_SOURCE_CLASS if action_id == B_ACTION_ID else U_COUNT_PER_SOURCE_CLASS
        counts = {label: {source: count for source in sources} for label in BINARY_CLASSES}
        weights = {source: 1.0 for source in sources}
    else:
        counts = {
            label: {
                source: SELECTED_COUNT_PER_CLASS if source == selected else OTHER_COUNT_PER_CLASS
                for source in sources
            }
            for label in BINARY_CLASSES
        }
        weights = {
            source: A1_SELECTED_SAMPLE_WEIGHT if source == selected else A1_OTHER_SAMPLE_WEIGHT
            for source in sources
        }
    return ActionSpec(target, action_id, selected, counts, weights)


@lru_cache(maxsize=len(CENTERS))
def actions_for_target(target: object) -> tuple[ActionSpec, ...]:
    center = str(target)
    sources = candidate_sources(center)
    result = (
        _action(center, B_ACTION_ID, None),
        _action(center, U_ACTION_ID, None),
        *(_action(center, a1_action_id(source), source) for source in sources),
    )
    if len(result) != ACTION_COUNT_PER_TARGET:
        raise ProtocolError("DCSE physical action count drifted.")
    return result


@lru_cache(maxsize=1)
def build_action_library() -> tuple[ActionSpec, ...]:
    return tuple(action for target in CENTERS for action in actions_for_target(target))


def action_library_by_target() -> Mapping[str, tuple[ActionSpec, ...]]:
    return MappingProxyType({target: actions_for_target(target) for target in CENTERS})


__all__ = (
    "ACTION_SCHEMA",
    "ActionSpec",
    "action_library_by_target",
    "actions_for_target",
    "build_action_library",
)
