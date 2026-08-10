"""Deterministic B/U/A0/A1 action library.

The library describes row membership and fit-time sample weights only.  It does
not execute a classifier fit and it cannot select between A0 and A1.
"""

from __future__ import annotations

from functools import lru_cache

from .constants import (
    A1_OTHER_SAMPLE_WEIGHT,
    A1_SELECTED_SAMPLE_WEIGHT,
    BINARY_CLASSES,
    B_ACTION_ID,
    B_COUNT_PER_SOURCE_CLASS,
    GEOMETRY_IDS,
    MIDOGPP_CENTERS,
    OTHER_COUNT_PER_CLASS,
    SELECTED_COUNT_PER_CLASS,
    U_ACTION_ID,
    U_COUNT_PER_SOURCE_CLASS,
    candidate_sources,
    geometry_action_id,
)
from .contracts import ActionSpec


def _counts(sources: tuple[str, ...], selected: str | None, selected_count: int, other_count: int) -> dict[int, dict[str, int]]:
    per_class = {
        source: selected_count if source == selected else other_count for source in sources
    }
    return {label: dict(per_class) for label in BINARY_CLASSES}


@lru_cache(maxsize=len(MIDOGPP_CENTERS))
def actions_for_target(target: object) -> tuple[ActionSpec, ...]:
    """Return B, U, eight A0 actions and eight A1 actions in stable order."""

    sources = candidate_sources(target)
    target_center = str(target)
    actions: list[ActionSpec] = [
        ActionSpec(
            target_center=target_center,
            action_id=B_ACTION_ID,
            geometry_id=None,
            selected_source=None,
            counts_by_class=_counts(
                sources,
                selected=None,
                selected_count=B_COUNT_PER_SOURCE_CLASS,
                other_count=B_COUNT_PER_SOURCE_CLASS,
            ),
            sample_weight_by_source={source: 1.0 for source in sources},
            physical_fit_required=True,
        ),
        ActionSpec(
            target_center=target_center,
            action_id=U_ACTION_ID,
            geometry_id=None,
            selected_source=None,
            counts_by_class=_counts(
                sources,
                selected=None,
                selected_count=U_COUNT_PER_SOURCE_CLASS,
                other_count=U_COUNT_PER_SOURCE_CLASS,
            ),
            sample_weight_by_source={source: 1.0 for source in sources},
            physical_fit_required=True,
        ),
    ]
    for geometry in GEOMETRY_IDS:
        for source in sources:
            actions.append(
                ActionSpec(
                    target_center=target_center,
                    action_id=geometry_action_id(geometry, source),
                    geometry_id=geometry,
                    selected_source=source,
                    counts_by_class=_counts(
                        sources,
                        selected=source,
                        selected_count=SELECTED_COUNT_PER_CLASS,
                        other_count=OTHER_COUNT_PER_CLASS,
                    ),
                    sample_weight_by_source={
                        candidate: (
                            1.0
                            if geometry == "A0"
                            else A1_SELECTED_SAMPLE_WEIGHT
                            if candidate == source
                            else A1_OTHER_SAMPLE_WEIGHT
                        )
                        for candidate in sources
                    },
                    physical_fit_required=True,
                )
            )
    if len({action.action_id for action in actions}) != len(actions):
        raise AssertionError("Frozen action identifiers must be unique within a target.")
    return tuple(actions)


@lru_cache(maxsize=1)
def build_action_library() -> tuple[ActionSpec, ...]:
    """Materialize the complete nine-target library in canonical order."""

    return tuple(
        action
        for target in MIDOGPP_CENTERS
        for action in actions_for_target(target)
    )


__all__ = ("ActionSpec", "actions_for_target", "build_action_library")
