"""Deterministic B, U, and eight legal A1 physical actions per target."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .constants import (
    A1_OTHER_SAMPLE_WEIGHT,
    A1_SELECTED_SAMPLE_WEIGHT,
    ACTION_COUNT_PER_TARGET,
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
        selected = None if self.selected_source is None else str(self.selected_source)
        try:
            counts = {
                label: {source: int(self.counts_by_class[label][source]) for source in sources}
                for label in BINARY_CLASSES
            }
            weights = {
                source: float(self.sample_weight_by_source[source]) for source in sources
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Action counts or weights are incomplete.") from exc
        if any(tuple(self.counts_by_class[label]) != sources for label in BINARY_CLASSES):
            raise ProtocolError("Action count source order drifted.")
        if tuple(self.sample_weight_by_source) != sources:
            raise ProtocolError("Action weight source order drifted.")
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
                raise ProtocolError("A1 action must name one legal non-target source.")
            expected_counts = {
                source: SELECTED_COUNT_PER_CLASS if source == selected else OTHER_COUNT_PER_CLASS
                for source in sources
            }
            expected_weights = {
                source: A1_SELECTED_SAMPLE_WEIGHT if source == selected else A1_OTHER_SAMPLE_WEIGHT
                for source in sources
            }
        if any(counts[label] != expected_counts for label in BINARY_CLASSES):
            raise ProtocolError("Action sample counts drifted from the frozen bank.")
        if weights != expected_weights:
            raise ProtocolError("Action sample weights drifted from the frozen bank.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "selected_source", selected)
        object.__setattr__(
            self,
            "counts_by_class",
            MappingProxyType(
                {label: MappingProxyType(counts[label]) for label in BINARY_CLASSES}
            ),
        )
        object.__setattr__(self, "sample_weight_by_source", MappingProxyType(weights))
        object.__setattr__(self, "action_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_support_static_router_action_v1",
            "target_center": self.target_center,
            "action_id": self.action_id,
            "selected_source": self.selected_source,
            "geometry_id": None if self.action_id in {B_ACTION_ID, U_ACTION_ID} else "A1",
            "counts_by_class": {
                str(label): dict(self.counts_by_class[label]) for label in BINARY_CLASSES
            },
            "sample_weight_by_source": dict(self.sample_weight_by_source),
            "physical_fit_required": True,
            "target_expert_excluded": True,
            "seed_repetitions_selectable": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "action_hash": self.action_hash}


def _action(target: str, action_id: str, selected_source: str | None) -> ActionSpec:
    sources = candidate_sources(target)
    if action_id == B_ACTION_ID:
        count = B_COUNT_PER_SOURCE_CLASS
    elif action_id == U_ACTION_ID:
        count = U_COUNT_PER_SOURCE_CLASS
    else:
        count = None
    counts = {
        label: {
            source: (
                count
                if count is not None
                else SELECTED_COUNT_PER_CLASS
                if source == selected_source
                else OTHER_COUNT_PER_CLASS
            )
            for source in sources
        }
        for label in BINARY_CLASSES
    }
    weights = {
        source: (
            1.0
            if selected_source is None
            else A1_SELECTED_SAMPLE_WEIGHT
            if source == selected_source
            else A1_OTHER_SAMPLE_WEIGHT
        )
        for source in sources
    }
    return ActionSpec(target, action_id, selected_source, counts, weights)


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
        raise ProtocolError("Action library does not contain exactly ten actions per target.")
    return result


@lru_cache(maxsize=1)
def build_action_library() -> Mapping[str, tuple[ActionSpec, ...]]:
    """Return the canonical target-keyed library expected by prediction runtime."""

    return MappingProxyType({center: actions_for_target(center) for center in CENTERS})


def flatten_action_library() -> tuple[ActionSpec, ...]:
    return tuple(action for center in CENTERS for action in actions_for_target(center))


__all__ = (
    "ActionSpec",
    "actions_for_target",
    "build_action_library",
    "flatten_action_library",
)
