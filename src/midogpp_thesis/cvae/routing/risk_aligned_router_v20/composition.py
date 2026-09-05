"""Byte-reconstructible B, exact-U, and branchwise soft top-K endpoints."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import (
    BASELINE_THRESHOLD,
    CompositeKind,
    Direction,
    LabelFreeAction,
    LabelFreeCaseMenu,
    RouterFitConfig,
    SoftTopKComposite,
    decode_probability_hex,
    finite,
)


def soft_arm_id(k: int, mixing_lambda: float, kind: CompositeKind = CompositeKind.BOTH) -> str:
    return f"{kind.value}_K{int(k)}_L{float(mixing_lambda):.2f}"


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
    permitted_hashes = {row.action_hash for row in menu.actions}
    for raw in values:
        action = menu.action_for(raw) if type(raw) is str else raw
        if (
            not isinstance(action, LabelFreeAction)
            or action.action_hash not in permitted_hashes
            or action.direction is not direction
            or not action.is_active
            or action.arm_id in seen
        ):
            raise ProtocolError("HARP v20 top-K ranking contains an ineligible action.")
        resolved.append(action)
        seen.add(action.arm_id)
    if len(resolved) < int(k):
        raise ProtocolError(
            f"HARP v20 K={int(k)} arm is ineligible with fewer than K active actions."
        )
    return tuple(resolved[: int(k)])


def build_baseline_composite(menu: LabelFreeCaseMenu) -> SoftTopKComposite:
    if not isinstance(menu, LabelFreeCaseMenu):
        raise ProtocolError("HARP v20 baseline composition requires a sealed menu.")
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
        raise ProtocolError("HARP v20 exact-U composition requires a sealed menu.")
    action = menu.full_action
    return SoftTopKComposite(
        surface_role=menu.surface_role,
        center_id=menu.center_id,
        case_id=menu.case_id,
        menu_hash=menu.menu_hash,
        kind=CompositeKind.U_FULL,
        arm_id="U_FULL",
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
    kind: CompositeKind = CompositeKind.BOTH,
) -> SoftTopKComposite:
    """Build the branchwise endpoint with its exact two-round float32 contract.

    Component probabilities are decoded float32 cells.  Their equal-weight mean
    is accumulated in float64 in deterministic rank order and cast to float32.
    The shrinkage blend with B is then accumulated in float64 and cast once more
    to float32.  A row for which every selected component is byte-equal to B is
    copied from B without arithmetic.
    """

    if not isinstance(menu, LabelFreeCaseMenu) or type(k) is not int or k < 1:
        raise ProtocolError("HARP v20 soft top-K composition inputs are malformed.")
    lam = finite(mixing_lambda, name="mixing lambda")
    if kind not in (CompositeKind.D01_ONLY, CompositeKind.D10_ONLY, CompositeKind.BOTH):
        raise ProtocolError("HARP v20 directional composite kind is invalid.")
    if not 0.0 < lam <= 1.0:
        raise ProtocolError("HARP v20 mixing lambda must lie in (0,1].")
    baseline = decode_probability_hex(menu.baseline_probability_hex)
    d01 = (
        _resolve_ranked(menu, d01_ranked_actions, direction=Direction.D01, k=k)
        if kind in (CompositeKind.D01_ONLY, CompositeKind.BOTH) and any(value < BASELINE_THRESHOLD for value in baseline)
        else ()
    )
    d10 = (
        _resolve_ranked(menu, d10_ranked_actions, direction=Direction.D10, k=k)
        if kind in (CompositeKind.D10_ONLY, CompositeKind.BOTH) and any(value >= BASELINE_THRESHOLD for value in baseline)
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
        kind=kind,
        arm_id=soft_arm_id(k, lam, kind),
        sample_ids=menu.sample_ids,
        baseline_probability_hex=menu.baseline_probability_hex,
        probability_hex=tuple(probability_hex),
        k=k,
        mixing_lambda=lam,
        d01_action_ids=tuple(row.arm_id for row in d01),
        d10_action_ids=tuple(row.arm_id for row in d10),
        donor_ids=tuple(row.donor_id for row in (*d01, *d10) if row.donor_id is not None),
    )


@dataclass(frozen=True, slots=True)
class CandidateComposite:
    """One registered configuration and its label-free case-local disposition."""

    arm_id: str
    kind: CompositeKind
    composite: SoftTopKComposite | None
    k: int | None = None
    mixing_lambda: float | None = None
    ineligible_reason: str | None = None
    duplicate_of: str | None = None

    @property
    def eligible(self) -> bool:
        return self.composite is not None and self.ineligible_reason is None and self.duplicate_of is None

    def public_payload(self) -> dict[str, object]:
        return {
            "arm_id": self.arm_id, "kind": self.kind.value, "k": self.k,
            "mixing_lambda": self.mixing_lambda, "eligible": self.eligible,
            "ineligible_reason": self.ineligible_reason, "duplicate_of": self.duplicate_of,
            "composite_hash": None if self.composite is None else self.composite.composite_hash,
            "labels_consumed": False,
        }


def build_candidate_composites(menu: LabelFreeCaseMenu, prediction: object,
                               config: RouterFitConfig) -> tuple[CandidateComposite, ...]:
    """Freeze every configuration before truth; infeasibility never crosses cases.

    Candidates with identical bytes are aliases of the first candidate in the
    declared order. B precedes U, then one-branch actions precede BOTH.
    Malformed ranking identities raise; only insufficient active K is infeasible.
    """
    if getattr(prediction, "menu_hash", menu.menu_hash) != menu.menu_hash:
        raise ProtocolError("HARP v20 ranking prediction belongs to a different menu.")
    d01 = tuple(prediction.d01_ranked_action_ids)
    d10 = tuple(prediction.d10_ranked_action_ids)
    # Validate even unused rankings once, before classifying local availability.
    for direction, ids in ((Direction.D01, d01), (Direction.D10, d10)):
        if ids:
            _resolve_ranked(menu, ids, direction=direction, k=1)
    baseline = decode_probability_hex(menu.baseline_probability_hex)
    present = {Direction.D01: any(x < BASELINE_THRESHOLD for x in baseline),
               Direction.D10: any(x >= BASELINE_THRESHOLD for x in baseline)}
    rows: list[CandidateComposite] = []
    by_bytes: dict[tuple[str, ...], str] = {}

    def append(composite: SoftTopKComposite) -> None:
        alias = by_bytes.get(composite.probability_hex)
        if alias is None:
            by_bytes[composite.probability_hex] = composite.arm_id
        rows.append(CandidateComposite(composite.arm_id, composite.kind, composite,
                                       composite.k, composite.mixing_lambda,
                                       duplicate_of=alias))

    append(build_baseline_composite(menu))
    append(build_exact_u_composite(menu))
    for kind in (CompositeKind.D01_ONLY, CompositeKind.D10_ONLY, CompositeKind.BOTH):
        for k in config.k_values:
            for lam in config.lambda_values:
                reasons = []
                for direction, ids, enabled in (
                    (Direction.D01, d01, kind in (CompositeKind.D01_ONLY, CompositeKind.BOTH)),
                    (Direction.D10, d10, kind in (CompositeKind.D10_ONLY, CompositeKind.BOTH)),
                ):
                    if enabled and present[direction] and len(ids) < k:
                        reasons.append(f"INSUFFICIENT_{direction.value}_ACTIVE_DONORS")
                if reasons:
                    rows.append(CandidateComposite(soft_arm_id(k, lam, kind), kind, None,
                                                   k, lam, ";".join(reasons)))
                    continue
                append(build_soft_topk_composite(menu, d01_ranked_actions=d01,
                    d10_ranked_actions=d10, k=k, mixing_lambda=lam, kind=kind))
    return tuple(rows)


__all__ = ("CandidateComposite", "build_candidate_composites", "build_baseline_composite",
           "build_exact_u_composite", "build_soft_topk_composite", "soft_arm_id")
