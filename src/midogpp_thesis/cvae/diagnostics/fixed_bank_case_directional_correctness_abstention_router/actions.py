"""The fixed B/U/eight-A1 physical action library."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .constants import (
    ACTION_COUNT_PER_TARGET,
    B_ACTION_ID,
    CENTERS,
    U_ACTION_ID,
    a1_action_id,
    candidate_sources,
)
from .experiment_contracts import (
    A1_OTHER_ROWS_PER_CLASS,
    A1_OTHER_ROW_WEIGHT,
    A1_SELECTED_ROWS_PER_CLASS,
    A1_SELECTED_ROW_WEIGHT,
    BASE_ROWS_PER_SOURCE_CLASS,
    UNIFORM_ROWS_PER_SOURCE_CLASS,
)
from .hashing import canonical_hash


BINARY_CLASSES = (0, 1)
ACTION_SCHEMA = "fixed_bank_cdca_action_v1"


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
        if target not in CENTERS:
            raise ProtocolError("Abstention-router action target drifted.")
        sources = candidate_sources(target)
        selected = None if self.selected_source is None else str(self.selected_source)
        try:
            counts = {
                label: {
                    source: int(self.counts_by_class[label][source])
                    for source in sources
                }
                for label in BINARY_CLASSES
            }
            weights = {
                source: float(self.sample_weight_by_source[source])
                for source in sources
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(
                "Abstention-router action counts or weights are incomplete."
            ) from exc
        if any(
            tuple(self.counts_by_class[label]) != sources
            for label in BINARY_CLASSES
        ) or tuple(self.sample_weight_by_source) != sources:
            raise ProtocolError("Abstention-router action source order drifted.")
        if self.action_id == B_ACTION_ID:
            expected_counts = {
                source: BASE_ROWS_PER_SOURCE_CLASS for source in sources
            }
            expected_weights = {source: 1.0 for source in sources}
            if selected is not None:
                raise ProtocolError("B cannot select a source.")
        elif self.action_id == U_ACTION_ID:
            expected_counts = {
                source: UNIFORM_ROWS_PER_SOURCE_CLASS for source in sources
            }
            expected_weights = {source: 1.0 for source in sources}
            if selected is not None:
                raise ProtocolError("U cannot select a source.")
        else:
            if selected not in sources or self.action_id != a1_action_id(selected):
                raise ProtocolError("A1 action must name one legal non-target source.")
            expected_counts = {
                source: (
                    A1_SELECTED_ROWS_PER_CLASS
                    if source == selected
                    else A1_OTHER_ROWS_PER_CLASS
                )
                for source in sources
            }
            expected_weights = {
                source: (
                    A1_SELECTED_ROW_WEIGHT
                    if source == selected
                    else A1_OTHER_ROW_WEIGHT
                )
                for source in sources
            }
        if any(counts[label] != expected_counts for label in BINARY_CLASSES):
            raise ProtocolError("Abstention-router physical action counts drifted.")
        if weights != expected_weights:
            raise ProtocolError("Abstention-router physical action weights drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "selected_source", selected)
        object.__setattr__(
            self,
            "counts_by_class",
            MappingProxyType(
                {
                    label: MappingProxyType(counts[label])
                    for label in BINARY_CLASSES
                }
            ),
        )
        object.__setattr__(
            self, "sample_weight_by_source", MappingProxyType(weights)
        )
        object.__setattr__(self, "action_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": ACTION_SCHEMA,
            "target_center": self.target_center,
            "action_id": self.action_id,
            "selected_source": self.selected_source,
            "geometry_id": (
                None if self.action_id in {B_ACTION_ID, U_ACTION_ID} else "A1"
            ),
            "counts_by_class": {
                str(label): dict(self.counts_by_class[label])
                for label in BINARY_CLASSES
            },
            "sample_weight_by_source": dict(self.sample_weight_by_source),
            "physical_fit_required": True,
            "exact_nine_seed_pairs_required": True,
            "target_expert_excluded": True,
            "labels_used": False,
            "reused_stage90_predictions": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "action_hash": self.action_hash}


def _action(target: str, action_id: str, selected: str | None) -> ActionSpec:
    sources = candidate_sources(target)
    if selected is None:
        count = (
            BASE_ROWS_PER_SOURCE_CLASS
            if action_id == B_ACTION_ID
            else UNIFORM_ROWS_PER_SOURCE_CLASS
        )
        counts = {
            label: {source: count for source in sources}
            for label in BINARY_CLASSES
        }
        weights = {source: 1.0 for source in sources}
    else:
        counts = {
            label: {
                source: (
                    A1_SELECTED_ROWS_PER_CLASS
                    if source == selected
                    else A1_OTHER_ROWS_PER_CLASS
                )
                for source in sources
            }
            for label in BINARY_CLASSES
        }
        weights = {
            source: (
                A1_SELECTED_ROW_WEIGHT
                if source == selected
                else A1_OTHER_ROW_WEIGHT
            )
            for source in sources
        }
    return ActionSpec(target, action_id, selected, counts, weights)


@lru_cache(maxsize=len(CENTERS))
def actions_for_target(target_center: object) -> tuple[ActionSpec, ...]:
    target = str(target_center)
    sources = candidate_sources(target)
    result = (
        _action(target, B_ACTION_ID, None),
        _action(target, U_ACTION_ID, None),
        *(
            _action(target, a1_action_id(source), source)
            for source in sources
        ),
    )
    if len(result) != ACTION_COUNT_PER_TARGET:
        raise ProtocolError(
            "Abstention-router action library must contain ten actions per target."
        )
    return result


@lru_cache(maxsize=1)
def build_action_library() -> tuple[ActionSpec, ...]:
    return tuple(
        action for target in CENTERS for action in actions_for_target(target)
    )


def action_library_by_target() -> Mapping[str, tuple[ActionSpec, ...]]:
    """Return the exact target-keyed contract consumed by neutral runtime."""

    return MappingProxyType(
        {target: actions_for_target(target) for target in CENTERS}
    )


__all__ = (
    "ACTION_SCHEMA",
    "ActionSpec",
    "action_library_by_target",
    "actions_for_target",
    "build_action_library",
)
