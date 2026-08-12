"""Deterministic B, U, and eight A1 physical actions per target."""

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
        counts: dict[int, Mapping[str, int]] = {}
        for label in BINARY_CLASSES:
            raw = self.counts_by_class[label]
            if tuple(raw) != sources:
                raise ProtocolError("Flip-router action source order drifted.")
            counts[label] = MappingProxyType(
                {source: int(raw[source]) for source in sources}
            )
        weights = {source: float(self.sample_weight_by_source[source]) for source in sources}
        if tuple(self.sample_weight_by_source) != sources:
            raise ProtocolError("Flip-router action weight order drifted.")
        selected = self.selected_source
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
                raise ProtocolError("A1 action source identity drifted.")
            expected_counts = {
                source: SELECTED_COUNT_PER_CLASS if source == selected else OTHER_COUNT_PER_CLASS
                for source in sources
            }
            expected_weights = {
                source: A1_SELECTED_SAMPLE_WEIGHT if source == selected else A1_OTHER_SAMPLE_WEIGHT
                for source in sources
            }
        if any(dict(counts[label]) != expected_counts for label in BINARY_CLASSES):
            raise ProtocolError("Flip-router action counts drifted.")
        if weights != expected_weights:
            raise ProtocolError("Flip-router A1 fit weights drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "counts_by_class", MappingProxyType(counts))
        object.__setattr__(self, "sample_weight_by_source", MappingProxyType(weights))
        object.__setattr__(self, "action_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_labeled_support_flip_action_v1",
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


def _action(
    target: str,
    action_id: str,
    selected: str | None,
    selected_count: int,
    other_count: int,
) -> ActionSpec:
    sources = candidate_sources(target)
    counts = {
        label: {
            source: selected_count if selected is None or source == selected else other_count
            for source in sources
        }
        for label in BINARY_CLASSES
    }
    weights = {
        source: (
            1.0
            if selected is None
            else A1_SELECTED_SAMPLE_WEIGHT
            if source == selected
            else A1_OTHER_SAMPLE_WEIGHT
        )
        for source in sources
    }
    return ActionSpec(target, action_id, selected, counts, weights)


@lru_cache(maxsize=len(CENTERS))
def actions_for_target(target: object) -> tuple[ActionSpec, ...]:
    center = str(target)
    sources = candidate_sources(center)
    result = (
        _action(center, B_ACTION_ID, None, B_COUNT_PER_SOURCE_CLASS, B_COUNT_PER_SOURCE_CLASS),
        _action(center, U_ACTION_ID, None, U_COUNT_PER_SOURCE_CLASS, U_COUNT_PER_SOURCE_CLASS),
        *(
            _action(center, a1_action_id(source), source, SELECTED_COUNT_PER_CLASS, OTHER_COUNT_PER_CLASS)
            for source in sources
        ),
    )
    if len(result) != ACTION_COUNT_PER_TARGET:
        raise ProtocolError("Flip-router physical action count drifted.")
    return tuple(result)


@lru_cache(maxsize=1)
def build_action_library() -> tuple[ActionSpec, ...]:
    return tuple(action for target in CENTERS for action in actions_for_target(target))


def action_library_by_target() -> Mapping[str, tuple[ActionSpec, ...]]:
    return MappingProxyType({target: actions_for_target(target) for target in CENTERS})


__all__ = ("ActionSpec", "action_library_by_target", "actions_for_target", "build_action_library")
