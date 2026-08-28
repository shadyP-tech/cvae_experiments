"""V3-owned B/U/A1 physical action specifications.

These values are the immutable composition menu consumed by the neutral
fixed-bank A1 probability runtime.  No predecessor Stage-90 package is
imported, and the target expert H is excluded from every action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

from ....protocol import ProtocolError
from ..hashing import canonical_hash
from ..identity import CENTERS


B_ACTION_ID = "B"
U_ACTION_ID = "U"
A1_ACTION_PREFIX = "A1::source="
B_ROWS_PER_SOURCE_CLASS = 128
U_ROWS_PER_SOURCE_CLASS = 144
A1_SELECTED_ROWS_PER_CLASS = 256
A1_OTHER_ROWS_PER_CLASS = 128
A1_SELECTED_ROW_WEIGHT = 23.0 / 16.0
A1_OTHER_ROW_WEIGHT = 7.0 / 8.0


def candidate_sources(target_center: object) -> tuple[str, ...]:
    target = str(target_center)
    if target not in CENTERS:
        raise ProtocolError("OE-PPUR v3 physical target center is unknown.")
    return tuple(center for center in CENTERS if center != target)


def a1_action_id(source_center: object) -> str:
    source = str(source_center)
    if source not in CENTERS:
        raise ProtocolError("OE-PPUR v3 physical source center is unknown.")
    return f"{A1_ACTION_PREFIX}{source}"


@dataclass(frozen=True, slots=True)
class PhysicalActionSpec:
    target_center: str
    action_id: str
    selected_source: str | None
    counts_by_class: Mapping[str, Mapping[str, int]]
    sample_weight_by_source: Mapping[str, float]
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        sources = candidate_sources(target)
        selected = (
            None if self.selected_source is None else str(self.selected_source)
        )
        try:
            counts = {
                label: {
                    source: int(self.counts_by_class[label][source])
                    for source in sources
                }
                for label in ("0", "1")
            }
            weights = {
                source: float(self.sample_weight_by_source[source])
                for source in sources
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(
                "OE-PPUR v3 physical action mapping drifted."
            ) from exc
        if self.action_id == B_ACTION_ID:
            expected_counts = {
                source: B_ROWS_PER_SOURCE_CLASS for source in sources
            }
            expected_weights = {source: 1.0 for source in sources}
            valid_selection = selected is None
        elif self.action_id == U_ACTION_ID:
            expected_counts = {
                source: U_ROWS_PER_SOURCE_CLASS for source in sources
            }
            expected_weights = {source: 1.0 for source in sources}
            valid_selection = selected is None
        else:
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
            valid_selection = (
                selected in sources and self.action_id == a1_action_id(selected)
            )
        if (
            not valid_selection
            or any(counts[label] != expected_counts for label in ("0", "1"))
            or weights != expected_weights
        ):
            raise ProtocolError("OE-PPUR v3 physical action contract drifted.")
        body = {
            "schema_version": "oe_ppur_v3_physical_action_v1",
            "target_center": target,
            "action_id": self.action_id,
            "selected_source": selected,
            "counts_by_class": counts,
            "sample_weight_by_source": weights,
            "target_expert_excluded": True,
            "labels_consumed": False,
            "bank_lock_hash_bound_by_admission": True,
            "generation_lock_hash_bound_by_admission": True,
        }
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "selected_source", selected)
        object.__setattr__(
            self,
            "counts_by_class",
            MappingProxyType(
                {
                    label: MappingProxyType(counts[label])
                    for label in ("0", "1")
                }
            ),
        )
        object.__setattr__(
            self,
            "sample_weight_by_source",
            MappingProxyType(weights),
        )
        object.__setattr__(self, "action_hash", canonical_hash(body))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_physical_action_v1",
            "target_center": self.target_center,
            "action_id": self.action_id,
            "selected_source": self.selected_source,
            "counts_by_class": {
                label: dict(self.counts_by_class[label])
                for label in ("0", "1")
            },
            "sample_weight_by_source": dict(self.sample_weight_by_source),
            "target_expert_excluded": True,
            "labels_consumed": False,
            "bank_lock_hash_bound_by_admission": True,
            "generation_lock_hash_bound_by_admission": True,
            "action_hash": self.action_hash,
        }


def _build_action(
    target: str,
    action_id: str,
    selected: str | None,
) -> PhysicalActionSpec:
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
    return PhysicalActionSpec(target, action_id, selected, counts, weights)


@lru_cache(maxsize=len(CENTERS))
def actions_for_target(target_center: object) -> tuple[PhysicalActionSpec, ...]:
    target = str(target_center)
    sources = candidate_sources(target)
    return (
        _build_action(target, B_ACTION_ID, None),
        _build_action(target, U_ACTION_ID, None),
        *(
            _build_action(target, a1_action_id(source), source)
            for source in sources
        ),
    )


def action_library_by_target() -> Mapping[str, tuple[PhysicalActionSpec, ...]]:
    return MappingProxyType(
        {target: actions_for_target(target) for target in CENTERS}
    )


__all__ = (
    "B_ACTION_ID",
    "PhysicalActionSpec",
    "U_ACTION_ID",
    "a1_action_id",
    "action_library_by_target",
    "actions_for_target",
    "candidate_sources",
)
