"""Canonical B/U/eight-A1 library consumed by the neutral prediction runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from .contracts import candidate_sources, physical_action_ids


B_COUNT_PER_SOURCE_CLASS = 128
U_COUNT_PER_SOURCE_CLASS = 144
A1_SELECTED_ROWS_PER_CLASS = 256
A1_OTHER_ROWS_PER_CLASS = 128
A1_SELECTED_ROW_WEIGHT = 23.0 / 16.0
A1_OTHER_ROW_WEIGHT = 7.0 / 8.0


@dataclass(frozen=True, slots=True)
class PhysicalActionSpec:
    target_center: str
    action_id: str
    selected_source: str | None
    counts_by_class: Mapping[int, Mapping[str, int]]
    sample_weight_by_source: Mapping[str, float]
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        sources = candidate_sources(self.target_center)
        selected = None if self.selected_source is None else str(self.selected_source)
        try:
            counts = {
                label: {
                    source: int(self.counts_by_class[label][source])
                    for source in sources
                }
                for label in (0, 1)
            }
            weights = {
                source: float(self.sample_weight_by_source[source]) for source in sources
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise GovernanceError("SCALE-BP v2 physical action is incomplete.") from exc
        if self.action_id == "B":
            expected_counts = {source: B_COUNT_PER_SOURCE_CLASS for source in sources}
            expected_weights = {source: 1.0 for source in sources}
            valid_selected = selected is None
        elif self.action_id == "U":
            expected_counts = {source: U_COUNT_PER_SOURCE_CLASS for source in sources}
            expected_weights = {source: 1.0 for source in sources}
            valid_selected = selected is None
        else:
            valid_selected = (
                selected in sources and self.action_id == f"A1::source={selected}"
            )
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
        if (
            self.action_id not in physical_action_ids(self.target_center)
            or not valid_selected
            or any(counts[label] != expected_counts for label in (0, 1))
            or weights != expected_weights
        ):
            raise GovernanceError("SCALE-BP v2 physical action semantics drifted.")
        frozen_counts = MappingProxyType(
            {label: MappingProxyType(counts[label]) for label in (0, 1)}
        )
        object.__setattr__(self, "selected_source", selected)
        object.__setattr__(self, "counts_by_class", frozen_counts)
        object.__setattr__(self, "sample_weight_by_source", MappingProxyType(weights))
        object.__setattr__(self, "action_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_v2_physical_action_v1",
            "target_center": self.target_center,
            "action_id": self.action_id,
            "selected_source": self.selected_source,
            "geometry_id": None if self.action_id in {"B", "U"} else "A1",
            "counts_by_class": {
                str(label): dict(self.counts_by_class[label]) for label in (0, 1)
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


def _build_action(
    target: str, action_id: str, selected: str | None
) -> PhysicalActionSpec:
    sources = candidate_sources(target)
    if selected is None:
        count = B_COUNT_PER_SOURCE_CLASS if action_id == "B" else U_COUNT_PER_SOURCE_CLASS
        counts = {label: {source: count for source in sources} for label in (0, 1)}
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
            for label in (0, 1)
        }
        weights = {
            source: (
                A1_SELECTED_ROW_WEIGHT
                if source == selected
                else A1_OTHER_ROW_WEIGHT
            )
            for source in sources
        }
    return PhysicalActionSpec(target, action_id, selected, counts, weights)


@lru_cache(maxsize=9)
def actions_for_target(target_center: object) -> tuple[PhysicalActionSpec, ...]:
    target = str(target_center)
    result = (
        _build_action(target, "B", None),
        _build_action(target, "U", None),
        *(
            _build_action(target, f"A1::source={source}", source)
            for source in candidate_sources(target)
        ),
    )
    if tuple(action.action_id for action in result) != physical_action_ids(target):
        raise GovernanceError("SCALE-BP v2 target action library drifted.")
    return result


def action_library_by_target() -> Mapping[str, tuple[PhysicalActionSpec, ...]]:
    from .contracts import CENTERS

    return MappingProxyType({target: actions_for_target(target) for target in CENTERS})


def build_action_library() -> tuple[PhysicalActionSpec, ...]:
    from .contracts import CENTERS

    return tuple(action for target in CENTERS for action in actions_for_target(target))


__all__ = (
    "PhysicalActionSpec",
    "action_library_by_target",
    "actions_for_target",
    "build_action_library",
)
