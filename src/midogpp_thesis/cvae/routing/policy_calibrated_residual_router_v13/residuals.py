"""Baseline/budget/allocation residual features for HARP v13.

These are utility-model features, not physical residual top-up operations.
The representation makes the hierarchy explicit:

``HXE - B = (U - B) + (HXE - U)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import LabelFreeAction
from .effective_menu import EffectiveMenu


@dataclass(frozen=True, slots=True)
class ResidualActionFeatures:
    action: LabelFreeAction
    names: tuple[str, ...]
    values: tuple[float, ...]
    budget_width: int
    allocation_width: int
    has_uniform_reference: bool

    def __post_init__(self) -> None:
        if (
            len(self.names) != len(self.values)
            or self.budget_width < 1
            or self.allocation_width < 1
            or self.budget_width + self.allocation_width + 5 != len(self.values)
        ):
            raise ProtocolError("HARP v13 residual feature block is malformed.")


def residual_feature_names(base_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(value) for value in base_names)
    if not names or len(set(names)) != len(names):
        raise ProtocolError("HARP v13 residualization requires a canonical base schema.")
    return (
        *(f"budget__{name}" for name in names),
        *(f"allocation__{name}" for name in names),
        "kind__U",
        "kind__HXE",
        "direction__D01",
        "direction__D10",
        "allocation__has_uniform_reference",
    )


def residualize_menu(menu: EffectiveMenu) -> tuple[ResidualActionFeatures, ...]:
    if not isinstance(menu, EffectiveMenu):
        raise ProtocolError("HARP v13 residualization requires an effective menu.")
    width = len(menu.feature_names)
    names = residual_feature_names(menu.feature_names)
    uniform: dict[object, LabelFreeAction] = {}
    for action in sorted(menu.actions, key=lambda row: row.action_id):
        if action.action_kind == "U":
            if action.direction in uniform:
                raise ProtocolError("HARP v13 menu has multiple uniform controls per direction.")
            uniform[action.direction] = action
    output: list[ResidualActionFeatures] = []
    for action in menu.actions:
        raw = np.asarray(action.feature_values, dtype=np.float64)
        if raw.shape != (width,) or not np.all(np.isfinite(raw)):
            raise ProtocolError("HARP v13 action feature vector is malformed.")
        reference = uniform.get(action.direction)
        if action.action_kind == "U":
            budget = raw
            allocation = np.zeros(width, dtype=np.float64)
            has_reference = False
        elif reference is None:
            # The missing reference is encoded explicitly.  The raw HXE-B
            # residual remains usable without pretending it is HXE-U.
            budget = np.zeros(width, dtype=np.float64)
            allocation = raw
            has_reference = False
        else:
            budget = np.asarray(reference.feature_values, dtype=np.float64)
            allocation = raw - budget
            has_reference = True
        indicators = (
            float(action.action_kind == "U"),
            float(action.action_kind == "HXE"),
            float(action.direction.value == "D01"),
            float(action.direction.value == "D10"),
            float(has_reference),
        )
        values = tuple(float(value) for value in (*budget, *allocation, *indicators))
        output.append(
            ResidualActionFeatures(
                action=action,
                names=names,
                values=values,
                budget_width=width,
                allocation_width=width,
                has_uniform_reference=has_reference,
            )
        )
    return tuple(output)


def assert_residual_identity(
    *, baseline: Sequence[float], uniform: Sequence[float], hxe: Sequence[float]
) -> None:
    b = np.asarray(tuple(baseline), dtype=np.float64)
    u = np.asarray(tuple(uniform), dtype=np.float64)
    h = np.asarray(tuple(hxe), dtype=np.float64)
    if b.shape != u.shape or b.shape != h.shape or not np.allclose(
        h - b, (u - b) + (h - u), rtol=0.0, atol=1e-12
    ):
        raise ProtocolError("HARP v13 residual hierarchy identity failed.")


__all__ = (
    "ResidualActionFeatures",
    "assert_residual_identity",
    "residual_feature_names",
    "residualize_menu",
)
