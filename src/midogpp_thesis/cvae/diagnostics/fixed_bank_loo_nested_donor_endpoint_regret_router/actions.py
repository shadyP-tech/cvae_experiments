"""Successor-owned B/U/eight-A1 action specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .constants import (
    A1_OTHER_ROWS_PER_CLASS,
    A1_OTHER_ROW_WEIGHT,
    A1_SELECTED_ROWS_PER_CLASS,
    A1_SELECTED_ROW_WEIGHT,
    B_ACTION_ID,
    B_ROWS_PER_SOURCE_CLASS,
    CENTERS,
    U_ACTION_ID,
    U_ROWS_PER_SOURCE_CLASS,
    a1_action_id,
    candidate_sources,
)


@dataclass(frozen=True)
class ActionSpec:
    target_center: str
    action_id: str
    selected_source: str | None
    counts_by_class: Mapping[str, Mapping[str, int]]
    sample_weight_by_source: Mapping[str, float]
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        sources = candidate_sources(target)
        selected = None if self.selected_source is None else str(self.selected_source)
        counts = {
            label: {source: int(self.counts_by_class[label][source]) for source in sources}
            for label in ("0", "1")
        }
        weights = {
            source: float(self.sample_weight_by_source[source]) for source in sources
        }
        if self.action_id == B_ACTION_ID:
            expected_count = B_ROWS_PER_SOURCE_CLASS
            expected_weights = {source: 1.0 for source in sources}
            if selected is not None:
                raise ProtocolError("Baseline action cannot select a source.")
        elif self.action_id == U_ACTION_ID:
            expected_count = U_ROWS_PER_SOURCE_CLASS
            expected_weights = {source: 1.0 for source in sources}
            if selected is not None:
                raise ProtocolError("Uniform action cannot select a source.")
        else:
            if selected not in sources or self.action_id != a1_action_id(selected):
                raise ProtocolError("A1 action must select one legal source.")
            expected_count = -1
            expected_weights = {
                source: (
                    A1_SELECTED_ROW_WEIGHT
                    if source == selected
                    else A1_OTHER_ROW_WEIGHT
                )
                for source in sources
            }
        expected_counts = {
            source: (
                expected_count
                if expected_count >= 0
                else (
                    A1_SELECTED_ROWS_PER_CLASS
                    if source == selected
                    else A1_OTHER_ROWS_PER_CLASS
                )
            )
            for source in sources
        }
        if any(counts[label] != expected_counts for label in ("0", "1")) or weights != expected_weights:
            raise ProtocolError("Action counts or sample weights drifted.")
        unhashed = {
            "schema_version": "fixed_bank_nested_regret_action_v1",
            "target_center": target,
            "action_id": self.action_id,
            "selected_source": selected,
            "counts_by_class": counts,
            "sample_weight_by_source": weights,
            "target_expert_excluded": True,
            "labels_consumed": False,
        }
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "selected_source", selected)
        object.__setattr__(
            self,
            "counts_by_class",
            MappingProxyType(
                {label: MappingProxyType(counts[label]) for label in ("0", "1")}
            ),
        )
        object.__setattr__(self, "sample_weight_by_source", MappingProxyType(weights))
        object.__setattr__(self, "action_hash", stable_hash(unhashed))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_nested_regret_action_v1",
            "target_center": self.target_center,
            "action_id": self.action_id,
            "selected_source": self.selected_source,
            "counts_by_class": {
                label: dict(self.counts_by_class[label]) for label in ("0", "1")
            },
            "sample_weight_by_source": dict(self.sample_weight_by_source),
            "target_expert_excluded": True,
            "labels_consumed": False,
            "action_hash": self.action_hash,
        }


def _action(target: str, action_id: str, selected: str | None) -> ActionSpec:
    sources = candidate_sources(target)
    counts = {
        label: {
            source: (
                B_ROWS_PER_SOURCE_CLASS
                if action_id == B_ACTION_ID
                else U_ROWS_PER_SOURCE_CLASS
                if action_id == U_ACTION_ID
                else A1_SELECTED_ROWS_PER_CLASS
                if source == selected
                else A1_OTHER_ROWS_PER_CLASS
            )
            for source in sources
        }
        for label in ("0", "1")
    }
    weights = {
        source: (
            1.0
            if selected is None
            else A1_SELECTED_ROW_WEIGHT
            if source == selected
            else A1_OTHER_ROW_WEIGHT
        )
        for source in sources
    }
    return ActionSpec(target, action_id, selected, counts, weights)


@lru_cache(maxsize=len(CENTERS))
def actions_for_target(target_center: object) -> tuple[ActionSpec, ...]:
    target = str(target_center)
    if target not in CENTERS:
        raise ProtocolError("Action target center is unknown.")
    return (
        _action(target, B_ACTION_ID, None),
        _action(target, U_ACTION_ID, None),
        *(
            _action(target, a1_action_id(source), source)
            for source in candidate_sources(target)
        ),
    )


def action_library_by_target() -> Mapping[str, tuple[ActionSpec, ...]]:
    return MappingProxyType(
        {target: actions_for_target(target) for target in CENTERS}
    )


__all__ = ("ActionSpec", "action_library_by_target", "actions_for_target")
