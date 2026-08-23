"""Joint identity/cyclic action-surface seal for the P-DCAPS lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...protocol import ProtocolError
from .action_surface import (
    RouteActionDraftSurface,
    SealedActionSurface,
    seal_action_surface,
)
from .contracts import RouteKey
from .identity import canonical_hash, require_sha256
from .inventory import ExpectedRouteInventory
from .target_local_runtime import POSTERIOR_CONTROL_IDS


IDENTITY_CONTROL_ID = "IDENTITY"
CYCLIC_CONTROL_ID = "WITHIN_CASE_CYCLIC_SHIFT"


@dataclass(frozen=True)
class SealedActionSurfaceSet:
    """All predeclared posterior controls sealed before pseudo labels open."""

    surfaces: tuple[SealedActionSurface, ...]
    expected_inventory_hash: str
    route_inventory_seal_hashes: tuple[tuple[str, str], ...]
    surface_set_seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        surfaces = tuple(self.surfaces)
        inventory_hash = require_sha256(
            self.expected_inventory_hash, "action-surface-set inventory hash"
        )
        route_seals = tuple(
            (
                str(control_id),
                require_sha256(value, "validated route-inventory seal"),
            )
            for control_id, value in self.route_inventory_seal_hashes
        )
        controls = tuple(row.posterior_control_id for row in surfaces)
        expected_controls = tuple(
            control_id for control_id in POSTERIOR_CONTROL_IDS if control_id in controls
        )
        reference_routes = () if not surfaces else surfaces[0].routes
        if (
            not surfaces
            or controls != expected_controls
            or controls != (IDENTITY_CONTROL_ID, CYCLIC_CONTROL_ID)
            or tuple(control_id for control_id, _value in route_seals) != controls
            or len({row.action_surface_seal_hash for row in surfaces}) != len(surfaces)
            or len({row.physical_surface_hash for row in surfaces}) != 1
            or any(row.expected_inventory_hash != inventory_hash for row in surfaces)
            or any(
                tuple((route.route_key, route.sample_ids) for route in surface.routes)
                != tuple((route.route_key, route.sample_ids) for route in reference_routes)
                for surface in surfaces[1:]
            )
            or (
                len(surfaces) == 2
                and any(
                    identity_route.posterior_prediction_hash
                    == cyclic_route.posterior_prediction_hash
                    for identity_route, cyclic_route in zip(
                        surfaces[0].routes,
                        surfaces[1].routes,
                        strict=True,
                    )
                )
            )
        ):
            raise ProtocolError("P-DCAPS identity/cyclic surface set drifted.")
        object.__setattr__(self, "surfaces", surfaces)
        object.__setattr__(self, "expected_inventory_hash", inventory_hash)
        object.__setattr__(self, "route_inventory_seal_hashes", route_seals)
        object.__setattr__(
            self,
            "surface_set_seal_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_action_surface_set_v1",
                    "expected_inventory_hash": inventory_hash,
                    "physical_surface_hash": surfaces[0].physical_surface_hash,
                    "control_surface_seals": tuple(
                        (
                            row.posterior_control_id,
                            row.action_surface_seal_hash,
                        )
                        for row in surfaces
                    ),
                    "route_inventory_seal_hashes": route_seals,
                    "pseudo_labels_used": False,
                    "target_labels_used": False,
                }
            ),
        )

    @property
    def control_ids(self) -> tuple[str, ...]:
        return tuple(row.posterior_control_id for row in self.surfaces)

    @property
    def identity(self) -> SealedActionSurface:
        return self.surface(IDENTITY_CONTROL_ID)

    @property
    def cyclic(self) -> SealedActionSurface:
        return self.surface(CYCLIC_CONTROL_ID)

    def surface(self, control_id: str) -> SealedActionSurface:
        for row in self.surfaces:
            if row.posterior_control_id == str(control_id):
                return row
        raise ProtocolError("P-DCAPS requested posterior-control surface is absent.")

    def routes(self, route_key: RouteKey) -> tuple[tuple[str, object], ...]:
        return tuple(
            (surface.posterior_control_id, surface.route(route_key))
            for surface in self.surfaces
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_action_surface_set_v1",
            "expected_inventory_hash": self.expected_inventory_hash,
            "physical_surface_hash": self.identity.physical_surface_hash,
            "control_surface_seals": [
                [row.posterior_control_id, row.action_surface_seal_hash]
                for row in self.surfaces
            ],
            "route_inventory_seal_hashes": [
                list(row) for row in self.route_inventory_seal_hashes
            ],
            "pseudo_labels_used": False,
            "target_labels_used": False,
            "surface_set_seal_hash": self.surface_set_seal_hash,
        }


def seal_action_surface_set(
    identity_routes: Sequence[RouteActionDraftSurface],
    *,
    expected_inventory: ExpectedRouteInventory,
    cyclic_routes: Sequence[RouteActionDraftSurface] | None = None,
) -> SealedActionSurfaceSet:
    """Validate and jointly seal the identity and predeclared cyclic controls."""

    route_groups = [(IDENTITY_CONTROL_ID, tuple(identity_routes))]
    if cyclic_routes is not None:
        route_groups.append((CYCLIC_CONTROL_ID, tuple(cyclic_routes)))
    surfaces: list[SealedActionSurface] = []
    route_seals: list[tuple[str, str]] = []
    for expected_control, routes in route_groups:
        if not routes or {row.posterior_control_id for row in routes} != {
            expected_control
        }:
            raise ProtocolError("P-DCAPS posterior-control route identity drifted.")
        route_seal = expected_inventory.validate_draft_routes(routes)
        surface = seal_action_surface(
            routes,
            expected_outer_centers=expected_inventory.centers,
            expected_inventory_hash=expected_inventory.inventory_hash,
        )
        surfaces.append(surface)
        route_seals.append((expected_control, route_seal))
    return SealedActionSurfaceSet(
        tuple(surfaces),
        expected_inventory.inventory_hash,
        tuple(route_seals),
    )


__all__ = (
    "CYCLIC_CONTROL_ID",
    "IDENTITY_CONTROL_ID",
    "SealedActionSurfaceSet",
    "seal_action_surface_set",
)
