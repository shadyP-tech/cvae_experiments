"""Canonical label-free H/J/d route planning for authorized P-DCAPS v4.

The planner is deliberately identifier-only.  It binds every target and
pseudo route to the whole-case exclusion plan that produced its posterior fit,
without accepting labels or filesystem state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.contracts import (
    RouteKey,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.inventory import (
    ExpectedRouteInventory,
    InventoryCase,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.physical_adapter import (
    PhysicalSurface,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.route_support import (
    OrderedPseudoPlan,
    WholeCasePlan,
)
from .identity import canonical_hash, require_sha256


@dataclass(frozen=True)
class RoutePlan:
    """One target or pseudo route with its exact exclusion witnesses."""

    route_key: RouteKey
    whole_case_plan: WholeCasePlan
    pseudo_plan: OrderedPseudoPlan | None
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        route = self.route_key
        whole = self.whole_case_plan
        pseudo = self.pseudo_plan
        if (
            whole.target_center != route.route_center
            or whole.case_id != route.held_case_id
            or (
                route.surface_role == "target"
                and (
                    pseudo is not None
                    or route.fit_scope_hash != whole.plan_hash
                    or route.excluded_scored_center is not None
                )
            )
            or (
                route.surface_role == "pseudo"
                and (
                    not isinstance(pseudo, OrderedPseudoPlan)
                    or pseudo.outer_center != route.outer_center
                    or pseudo.scored_center != route.route_center
                    or pseudo.held_case_id != route.held_case_id
                    or pseudo.source_plan_hash != whole.plan_hash
                    or route.fit_scope_hash != pseudo.plan_hash
                )
            )
        ):
            raise ProtocolError("P-DCAPS v4 H/J/d route plan drifted.")
        object.__setattr__(
            self,
            "plan_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v4_route_plan_v1",
                    "route_key": route.to_payload(),
                    "whole_case_plan_hash": whole.plan_hash,
                    "pseudo_plan_hash": None if pseudo is None else pseudo.plan_hash,
                    "outer_center_excluded_from_pseudo_donors": (
                        route.surface_role == "pseudo"
                    ),
                    "scored_center_excluded_from_candidate_sources": True,
                    "held_case_excluded_from_posterior_fit": True,
                    "labels_used": False,
                }
            ),
        )

    @property
    def endpoint_excluded_source_centers(self) -> tuple[str, ...]:
        return (
            ()
            if self.route_key.surface_role == "target"
            else (self.route_key.outer_center,)
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v4_route_plan_v1",
            "route_key": self.route_key.to_payload(),
            "whole_case_plan_hash": self.whole_case_plan.plan_hash,
            "pseudo_plan_hash": (
                None if self.pseudo_plan is None else self.pseudo_plan.plan_hash
            ),
            "endpoint_excluded_source_centers": list(
                self.endpoint_excluded_source_centers
            ),
            "labels_used": False,
            "plan_hash": self.plan_hash,
        }


@dataclass(frozen=True)
class RoutePlanInventory:
    """The exact outer-major route universe used by every control surface."""

    expected_inventory_hash: str
    plans: tuple[RoutePlan, ...]
    route_plan_inventory_hash: str = field(init=False)

    def __post_init__(self) -> None:
        inventory_hash = require_sha256(
            self.expected_inventory_hash, "v2 route-plan inventory"
        )
        plans = tuple(self.plans)
        if (
            not plans
            or len({row.route_key for row in plans}) != len(plans)
            or len({row.plan_hash for row in plans}) != len(plans)
        ):
            raise ProtocolError("P-DCAPS v4 route-plan inventory drifted.")
        object.__setattr__(self, "expected_inventory_hash", inventory_hash)
        object.__setattr__(self, "plans", plans)
        object.__setattr__(
            self,
            "route_plan_inventory_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v4_route_plan_inventory_v1",
                    "expected_inventory_hash": inventory_hash,
                    "route_plan_hashes": tuple(row.plan_hash for row in plans),
                    "route_count": len(plans),
                    "outer_major_order": True,
                    "labels_used": False,
                }
            ),
        )

    @property
    def route_keys(self) -> tuple[RouteKey, ...]:
        return tuple(row.route_key for row in self.plans)

    def plans_for_case(self, center: str, case_id: str) -> tuple[RoutePlan, ...]:
        rows = tuple(
            row
            for row in self.plans
            if row.route_key.route_center == str(center)
            and row.route_key.held_case_id == str(case_id)
        )
        if not rows:
            raise ProtocolError("P-DCAPS v4 case route plan is absent.")
        return rows

    def plans_for_outer(self, outer_center: str) -> tuple[RoutePlan, ...]:
        rows = tuple(
            row
            for row in self.plans
            if row.route_key.outer_center == str(outer_center)
        )
        if not rows:
            raise ProtocolError("P-DCAPS v4 outer route plan is absent.")
        return rows

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v4_route_plan_inventory_v1",
            "expected_inventory_hash": self.expected_inventory_hash,
            "plans": [row.to_payload() for row in self.plans],
            "route_count": len(self.plans),
            "outer_major_order": True,
            "labels_used": False,
            "route_plan_inventory_hash": self.route_plan_inventory_hash,
        }


def _whole_case_plan(
    case: InventoryCase,
    *,
    inventory: ExpectedRouteInventory,
    physical_surface: PhysicalSurface,
) -> WholeCasePlan:
    center_cases = tuple(
        row.case_id for row in inventory.cases if row.center == case.center
    )
    return WholeCasePlan(
        case.center,
        case.case_id,
        tuple(value for value in center_cases if value != case.case_id),
        case.sample_ids,
        physical_surface.physical_surface_hash,
        physical_surface.center(case.center).center_surface_hash,
    )


def _validate_physical_inventory(
    inventory: ExpectedRouteInventory,
    physical_surface: PhysicalSurface,
) -> None:
    for center in inventory.centers:
        surface = physical_surface.center(center)
        expected = tuple(
            (case.case_id, sample_id)
            for case in inventory.cases
            if case.center == center
            for sample_id in case.sample_ids
        )
        observed = tuple(zip(surface.case_ids, surface.sample_ids, strict=True))
        if set(observed) != set(expected) or len(observed) != len(expected):
            raise ProtocolError(
                "P-DCAPS v4 physical/inventory row topology drifted."
            )


def build_route_plan_inventory(
    inventory: ExpectedRouteInventory,
    physical_surface: PhysicalSurface,
) -> RoutePlanInventory:
    """Build one exact target-or-pseudo route for every outer H and case J/d."""

    if not isinstance(inventory, ExpectedRouteInventory) or not isinstance(
        physical_surface, PhysicalSurface
    ):
        raise ProtocolError("P-DCAPS v4 route planner requires typed inputs.")
    _validate_physical_inventory(inventory, physical_surface)
    whole_by_case = {
        case.key: _whole_case_plan(
            case, inventory=inventory, physical_surface=physical_surface
        )
        for case in inventory.cases
    }
    plans: list[RoutePlan] = []
    for outer in inventory.centers:
        for case in inventory.cases:
            whole = whole_by_case[case.key]
            if case.center == outer:
                route = RouteKey(
                    "target",
                    outer,
                    outer,
                    case.case_id,
                    outer,
                    None,
                    whole.plan_hash,
                )
                plans.append(RoutePlan(route, whole, None))
            else:
                pseudo = OrderedPseudoPlan(
                    outer,
                    case.center,
                    case.case_id,
                    outer,
                    case.center,
                    whole.plan_hash,
                )
                route = RouteKey(
                    "pseudo",
                    outer,
                    case.center,
                    case.case_id,
                    outer,
                    case.center,
                    pseudo.plan_hash,
                )
                plans.append(RoutePlan(route, whole, pseudo))
    output = RoutePlanInventory(inventory.inventory_hash, tuple(plans))
    if (
        len(output.plans) != inventory.total_route_count
        or sum(row.route_key.surface_role == "target" for row in output.plans)
        != inventory.target_route_count
        or sum(row.route_key.surface_role == "pseudo" for row in output.plans)
        != inventory.pseudo_route_count
        or tuple(
            row.route_key.outer_center
            for row in output.plans[:: inventory.case_count]
        )
        != inventory.centers
    ):
        raise ProtocolError("P-DCAPS v4 route-plan cardinality drifted.")
    return output


__all__ = (
    "RoutePlan",
    "RoutePlanInventory",
    "build_route_plan_inventory",
)
