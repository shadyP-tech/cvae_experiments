"""Non-authorizing label firewall and route-identity public surface."""

from __future__ import annotations

from .protocol import ProtocolError
from .route_identity import (
    RouteCaseBinding,
    RouteIdentityInventory,
    RouteScopeWitness,
    SampleIdentity,
    build_route_identity_inventory,
    build_route_scope_witness,
)


class PlannedLabelFirewall:
    """Hard capability barrier for the non-authorized v1 identity."""

    execution_authorized = False
    consumed_test_reuse_authorized = False

    def open_donor_labels(self, *_args: object, **_kwargs: object) -> None:
        raise ProtocolError("SCALE-BP v1 label capabilities are not authorized.")

    def open_route_support_labels(self, *_args: object, **_kwargs: object) -> None:
        raise ProtocolError("SCALE-BP v1 label capabilities are not authorized.")

    def open_terminal_labels(self, *_args: object, **_kwargs: object) -> None:
        raise ProtocolError("SCALE-BP v1 label capabilities are not authorized.")


__all__ = (
    "PlannedLabelFirewall",
    "RouteCaseBinding",
    "RouteIdentityInventory",
    "RouteScopeWitness",
    "SampleIdentity",
    "build_route_identity_inventory",
    "build_route_scope_witness",
)
