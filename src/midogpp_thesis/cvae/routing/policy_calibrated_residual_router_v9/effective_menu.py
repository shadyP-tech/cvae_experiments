"""Independent label-free physical-menu filtering for HARP v9."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import LabelFreeAction
from .hashing import canonical_hash


@dataclass(frozen=True, slots=True)
class EffectiveMenu:
    outer_target_id: str
    query_center_id: str
    case_id: str
    feature_names: tuple[str, ...]
    baseline_probability_hex: tuple[str, ...]
    actions: tuple[LabelFreeAction, ...]
    dropped_noop_action_ids: tuple[str, ...]
    duplicate_representatives: tuple[tuple[str, str], ...]
    menu_hash: str = field(init=False)

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if any(
            row.outer_target_id != self.outer_target_id
            or row.query_center_id != self.query_center_id
            or row.case_id != self.case_id
            or row.feature_names != self.feature_names
            or row.baseline_probability_hex != self.baseline_probability_hex
            or not row.is_active
            for row in actions
        ):
            raise ProtocolError("HARP v9 effective menu crossed roles or retained a no-op.")
        if len({row.action_id for row in actions}) != len(actions):
            raise ProtocolError("HARP v9 effective menu contains duplicate action ids.")
        if len({row.physical_output_hash for row in actions}) != len(actions):
            raise ProtocolError("HARP v9 effective menu contains duplicate physical outputs.")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(
            self,
            "menu_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_effective_menu_v9",
                    "outer_target_id": self.outer_target_id,
                    "query_center_id": self.query_center_id,
                    "case_id": self.case_id,
                    "feature_names": self.feature_names,
                    "baseline_probability_hex": self.baseline_probability_hex,
                    "action_hashes": tuple(row.action_hash for row in actions),
                    "dropped_noop_action_ids": self.dropped_noop_action_ids,
                    "duplicate_representatives": self.duplicate_representatives,
                    "filter_inputs": "LABEL_FREE_ONLY",
                    "virtual_B_added_after_menu_seal": True,
                }
            ),
        )

    @property
    def is_active(self) -> bool:
        return bool(self.actions)


def _representative_order(action: LabelFreeAction) -> tuple[str, str, str, str]:
    return (
        action.action_kind,
        action.direction.value,
        action.candidate_source_id or "",
        action.action_id,
    )


def build_effective_menu(actions: Sequence[LabelFreeAction]) -> EffectiveMenu:
    rows = tuple(actions)
    if not rows or any(not isinstance(row, LabelFreeAction) for row in rows):
        raise ProtocolError("HARP v9 menu construction requires label-free actions.")
    first = rows[0]
    if any(
        row.outer_target_id != first.outer_target_id
        or row.query_center_id != first.query_center_id
        or row.case_id != first.case_id
        or row.baseline_probability_hex != first.baseline_probability_hex
        or row.feature_names != first.feature_names
        for row in rows
    ):
        raise ProtocolError("HARP v9 physical menu crossed case/baseline/schema roles.")
    if len({row.action_id for row in rows}) != len(rows):
        raise ProtocolError("HARP v9 physical menu contains duplicate action ids.")
    dropped = tuple(sorted(row.action_id for row in rows if not row.is_active))
    by_output: dict[str, list[LabelFreeAction]] = {}
    for row in rows:
        if row.is_active:
            by_output.setdefault(row.physical_output_hash, []).append(row)
    retained: list[LabelFreeAction] = []
    aliases: list[tuple[str, str]] = []
    for output_hash in sorted(by_output):
        members = sorted(by_output[output_hash], key=_representative_order)
        representative = members[0]
        retained.append(representative)
        aliases.extend((member.action_id, representative.action_id) for member in members[1:])
    retained.sort(key=_representative_order)
    return EffectiveMenu(
        outer_target_id=first.outer_target_id,
        query_center_id=first.query_center_id,
        case_id=first.case_id,
        feature_names=first.feature_names,
        baseline_probability_hex=first.baseline_probability_hex,
        actions=tuple(retained),
        dropped_noop_action_ids=dropped,
        duplicate_representatives=tuple(sorted(aliases)),
    )


def group_effective_menus(actions: Sequence[LabelFreeAction]) -> tuple[EffectiveMenu, ...]:
    grouped: dict[tuple[str, str, str], list[LabelFreeAction]] = {}
    for action in actions:
        if not isinstance(action, LabelFreeAction):
            raise ProtocolError("HARP v9 menu grouping requires label-free actions.")
        grouped.setdefault(
            (action.outer_target_id, action.query_center_id, action.case_id), []
        ).append(action)
    return tuple(build_effective_menu(grouped[key]) for key in sorted(grouped))


__all__ = ("EffectiveMenu", "build_effective_menu", "group_effective_menus")
