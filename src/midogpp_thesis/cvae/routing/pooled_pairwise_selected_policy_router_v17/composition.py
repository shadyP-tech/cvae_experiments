"""Byte-reconstructible B, exact-U, and branchwise soft top-K endpoints."""

from __future__ import annotations

import struct
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import (
    BASELINE_THRESHOLD,
    CompositeKind,
    Direction,
    LabelFreeAction,
    LabelFreeCaseMenu,
    SoftTopKComposite,
    decode_probability_hex,
    finite,
)


def soft_arm_id(k: int, mixing_lambda: float) -> str:
    return f"SOFT_K{int(k)}_L{float(mixing_lambda):.2f}"


def _float32(value: float) -> float:
    return float(struct.unpack("<f", struct.pack("<f", float(value)))[0])


def _hex32(value: float) -> str:
    return struct.pack("<f", float(value)).hex()


def _resolve_ranked(
    menu: LabelFreeCaseMenu,
    values: Sequence[LabelFreeAction | str],
    *,
    direction: Direction,
    k: int,
) -> tuple[LabelFreeAction, ...]:
    resolved: list[LabelFreeAction] = []
    seen: set[str] = set()
    for raw in values:
        action = menu.action_for(raw) if type(raw) is str else raw
        if (
            not isinstance(action, LabelFreeAction)
            or action.action_hash not in {row.action_hash for row in menu.actions}
            or action.direction is not direction
            or not action.is_active
            or action.arm_id in seen
        ):
            raise ProtocolError("HARP v17 top-K ranking contains an ineligible action.")
        resolved.append(action)
        seen.add(action.arm_id)
    if len(resolved) < int(k):
        raise ProtocolError(
            f"HARP v17 K={int(k)} arm is ineligible with fewer than K active actions."
        )
    return tuple(resolved[: int(k)])


def build_baseline_composite(menu: LabelFreeCaseMenu) -> SoftTopKComposite:
    if not isinstance(menu, LabelFreeCaseMenu):
        raise ProtocolError("HARP v17 baseline composition requires a sealed menu.")
    return SoftTopKComposite(
        surface_role=menu.surface_role,
        center_id=menu.center_id,
        case_id=menu.case_id,
        menu_hash=menu.menu_hash,
        kind=CompositeKind.B,
        arm_id="B",
        sample_ids=menu.sample_ids,
        baseline_probability_hex=menu.baseline_probability_hex,
        probability_hex=menu.baseline_probability_hex,
    )


def build_exact_u_composite(menu: LabelFreeCaseMenu) -> SoftTopKComposite:
    if not isinstance(menu, LabelFreeCaseMenu):
        raise ProtocolError("HARP v17 exact-U composition requires a sealed menu.")
    action = menu.full_action
    return SoftTopKComposite(
        surface_role=menu.surface_role,
        center_id=menu.center_id,
        case_id=menu.case_id,
        menu_hash=menu.menu_hash,
        kind=CompositeKind.U_FULL,
        arm_id=action.arm_id,
        sample_ids=menu.sample_ids,
        baseline_probability_hex=menu.baseline_probability_hex,
        # The registered U comparator is a physical arm, never a recomputed mean.
        probability_hex=action.action_probability_hex,
    )


def build_soft_topk_composite(
    menu: LabelFreeCaseMenu,
    *,
    d01_ranked_actions: Sequence[LabelFreeAction | str],
    d10_ranked_actions: Sequence[LabelFreeAction | str],
    k: int,
    mixing_lambda: float,
) -> SoftTopKComposite:
    """Build the branchwise endpoint with its exact two-round float32 contract.

    Component probabilities are decoded float32 cells.  Their equal-weight mean
    is accumulated in float64 in deterministic rank order and cast to float32.
    The shrinkage blend with B is then accumulated in float64 and cast once more
    to float32.  A row for which every selected component is byte-equal to B is
    copied from B without arithmetic.
    """

    if not isinstance(menu, LabelFreeCaseMenu) or type(k) is not int or k < 1:
        raise ProtocolError("HARP v17 soft top-K composition inputs are malformed.")
    lam = finite(mixing_lambda, name="mixing lambda")
    if not 0.0 < lam <= 1.0:
        raise ProtocolError("HARP v17 mixing lambda must lie in (0,1].")
    baseline = decode_probability_hex(menu.baseline_probability_hex)
    d01 = (
        _resolve_ranked(menu, d01_ranked_actions, direction=Direction.D01, k=k)
        if any(value < BASELINE_THRESHOLD for value in baseline)
        else ()
    )
    d10 = (
        _resolve_ranked(menu, d10_ranked_actions, direction=Direction.D10, k=k)
        if any(value >= BASELINE_THRESHOLD for value in baseline)
        else ()
    )
    decoded = {
        row.action_hash: decode_probability_hex(row.action_probability_hex)
        for row in (*d01, *d10)
    }
    probability_hex: list[str] = []
    weight = 1.0 / float(k)
    for ordinal, baseline_value in enumerate(baseline):
        selected = d01 if baseline_value < BASELINE_THRESHOLD else d10
        if all(
            row.action_probability_hex[ordinal] == menu.baseline_probability_hex[ordinal]
            for row in selected
        ):
            probability_hex.append(menu.baseline_probability_hex[ordinal])
            continue
        endpoint_accumulator = 0.0
        for row in selected:
            endpoint_accumulator += weight * float(decoded[row.action_hash][ordinal])
        selected_float32 = _float32(endpoint_accumulator)
        final = (1.0 - lam) * float(baseline_value) + lam * float(selected_float32)
        probability_hex.append(_hex32(final))
    return SoftTopKComposite(
        surface_role=menu.surface_role,
        center_id=menu.center_id,
        case_id=menu.case_id,
        menu_hash=menu.menu_hash,
        kind=CompositeKind.SOFT_TOPK,
        arm_id=soft_arm_id(k, lam),
        sample_ids=menu.sample_ids,
        baseline_probability_hex=menu.baseline_probability_hex,
        probability_hex=tuple(probability_hex),
        k=k,
        mixing_lambda=lam,
        d01_action_ids=tuple(row.arm_id for row in d01),
        d10_action_ids=tuple(row.arm_id for row in d10),
        donor_ids=tuple(row.donor_id for row in (*d01, *d10) if row.donor_id is not None),
    )


__all__ = (
    "build_baseline_composite",
    "build_exact_u_composite",
    "build_soft_topk_composite",
    "soft_arm_id",
)
