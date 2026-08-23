"""Frozen B/U/A1 action definitions for the P-DCAPS physical bank."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from .identity import canonical_hash


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
    """Return the canonical source order with the target expert excluded."""

    target = str(target_center)
    if target not in CENTERS:
        raise ProtocolError("P-DCAPS physical target center is unknown.")
    return tuple(center for center in CENTERS if center != target)


def a1_action_id(source_center: object) -> str:
    source = str(source_center)
    if source not in CENTERS:
        raise ProtocolError("P-DCAPS physical source center is unknown.")
    return f"{A1_ACTION_PREFIX}{source}"


@dataclass(frozen=True)
class PhysicalActionSpec:
    """Immutable topology and weighting contract for one physical action."""

    target_center: str
    action_id: str
    selected_source: str | None
    counts_by_class: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    sample_weight_by_source: tuple[tuple[str, float], ...]
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        sources = candidate_sources(target)
        selected = None if self.selected_source is None else str(self.selected_source)
        counts = {
            str(label): {str(source): int(value) for source, value in rows}
            for label, rows in self.counts_by_class
        }
        weights = {
            str(source): float(value) for source, value in self.sample_weight_by_source
        }
        if tuple(counts) != ("0", "1") or tuple(weights) != sources:
            raise ProtocolError("P-DCAPS physical action topology drifted.")
        expected_action = (
            B_ACTION_ID
            if selected is None and self.action_id == B_ACTION_ID
            else U_ACTION_ID
            if selected is None and self.action_id == U_ACTION_ID
            else a1_action_id(selected)
            if selected in sources
            else ""
        )
        if self.action_id != expected_action:
            raise ProtocolError("P-DCAPS physical action identity drifted.")
        expected_counts = {
            source: (
                B_ROWS_PER_SOURCE_CLASS
                if self.action_id == B_ACTION_ID
                else U_ROWS_PER_SOURCE_CLASS
                if self.action_id == U_ACTION_ID
                else A1_SELECTED_ROWS_PER_CLASS
                if source == selected
                else A1_OTHER_ROWS_PER_CLASS
            )
            for source in sources
        }
        expected_weights = {
            source: (
                1.0
                if selected is None
                else A1_SELECTED_ROW_WEIGHT
                if source == selected
                else A1_OTHER_ROW_WEIGHT
            )
            for source in sources
        }
        if (
            any(counts[label] != expected_counts for label in ("0", "1"))
            or weights != expected_weights
        ):
            raise ProtocolError("P-DCAPS physical action counts or weights drifted.")
        object.__setattr__(
            self,
            "counts_by_class",
            tuple(
                (label, tuple((source, counts[label][source]) for source in sources))
                for label in ("0", "1")
            ),
        )
        object.__setattr__(
            self,
            "sample_weight_by_source",
            tuple((source, weights[source]) for source in sources),
        )
        object.__setattr__(
            self,
            "action_hash",
            canonical_hash(self.identity_payload(include_hash=False)),
        )

    def identity_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = {
            "schema_version": "pdcaps_physical_action_v1",
            "target_center": self.target_center,
            "action_id": self.action_id,
            "selected_source": self.selected_source,
            "counts_by_class": {
                label: {source: value for source, value in rows}
                for label, rows in self.counts_by_class
            },
            "sample_weight_by_source": dict(self.sample_weight_by_source),
            "target_expert_excluded": True,
            "labels_consumed": False,
        }
        if include_hash:
            payload["action_hash"] = self.action_hash
        return payload

    def to_payload(self) -> dict[str, object]:
        return self.identity_payload()


def _physical_action(
    target: str,
    action_id: str,
    selected: str | None,
) -> PhysicalActionSpec:
    sources = candidate_sources(target)
    counts = tuple(
        (
            label,
            tuple(
                (
                    source,
                    B_ROWS_PER_SOURCE_CLASS
                    if action_id == B_ACTION_ID
                    else U_ROWS_PER_SOURCE_CLASS
                    if action_id == U_ACTION_ID
                    else A1_SELECTED_ROWS_PER_CLASS
                    if source == selected
                    else A1_OTHER_ROWS_PER_CLASS,
                )
                for source in sources
            ),
        )
        for label in ("0", "1")
    )
    weights = tuple(
        (
            source,
            1.0
            if selected is None
            else A1_SELECTED_ROW_WEIGHT
            if source == selected
            else A1_OTHER_ROW_WEIGHT,
        )
        for source in sources
    )
    return PhysicalActionSpec(target, action_id, selected, counts, weights)


def action_library_by_target() -> dict[str, tuple[PhysicalActionSpec, ...]]:
    """Build the deterministic B/U/eight-A1 library for every target center."""

    return {
        target: (
            _physical_action(target, B_ACTION_ID, None),
            _physical_action(target, U_ACTION_ID, None),
            *(
                _physical_action(target, a1_action_id(source), source)
                for source in candidate_sources(target)
            ),
        )
        for target in CENTERS
    }


__all__ = (
    "A1_ACTION_PREFIX",
    "B_ACTION_ID",
    "PhysicalActionSpec",
    "U_ACTION_ID",
    "a1_action_id",
    "action_library_by_target",
    "candidate_sources",
)
