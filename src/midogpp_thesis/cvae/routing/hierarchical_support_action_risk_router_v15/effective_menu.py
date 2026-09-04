"""Label-free no-op and duplicate filtering for HARP v15 menus."""

from __future__ import annotations

from typing import Sequence

from ...protocol import ProtocolError
from .contracts import (
    LabelFreeAction,
    LabelFreeCaseMenu,
    SurfaceRole,
    canonical_probability_hex,
)


def build_effective_menu(
    *,
    outer_target_id: str,
    case_id: str,
    surface_role: SurfaceRole,
    baseline_probability_hex: Sequence[str],
    raw_actions: Sequence[LabelFreeAction],
) -> LabelFreeCaseMenu:
    """Compile a deterministic menu before any support label is available.

    Structural no-ops are removed.  If multiple action identities produce the
    same exact float32 vector, the lexicographically first identity is kept.
    This tie-break uses no outcomes and is shared by support and target roles.
    """

    baseline = canonical_probability_hex(tuple(baseline_probability_hex))
    rows = tuple(sorted(raw_actions, key=lambda row: row.action_id))
    if any(
        not isinstance(row, LabelFreeAction)
        or row.outer_target_id != outer_target_id
        or row.case_id != case_id
        or row.surface_role is not surface_role
        or row.baseline_probability_hex != baseline
        for row in rows
    ):
        raise ProtocolError("HARP v15 raw actions crossed an effective-menu boundary.")
    retained: list[LabelFreeAction] = []
    outputs: set[tuple[str, ...]] = set()
    for row in rows:
        if not row.is_active or row.action_probability_hex in outputs:
            continue
        outputs.add(row.action_probability_hex)
        retained.append(row)
    return LabelFreeCaseMenu(
        outer_target_id=outer_target_id,
        case_id=case_id,
        surface_role=surface_role,
        baseline_probability_hex=baseline,
        actions=tuple(retained),
    )


def group_effective_menus(
    actions: Sequence[LabelFreeAction],
) -> tuple[LabelFreeCaseMenu, ...]:
    rows = tuple(actions)
    if not rows:
        return ()
    grouped: dict[
        tuple[str, str, SurfaceRole, tuple[str, ...]], list[LabelFreeAction]
    ] = {}
    for row in rows:
        key = (
            row.outer_target_id,
            row.case_id,
            row.surface_role,
            row.baseline_probability_hex,
        )
        grouped.setdefault(key, []).append(row)
    return tuple(
        build_effective_menu(
            outer_target_id=outer,
            case_id=case,
            surface_role=role,
            baseline_probability_hex=baseline,
            raw_actions=grouped[(outer, case, role, baseline)],
        )
        for outer, case, role, baseline in sorted(
            grouped,
            key=lambda key: (key[0], key[1], key[2].value, key[3]),
        )
    )


__all__ = ("build_effective_menu", "group_effective_menus")
