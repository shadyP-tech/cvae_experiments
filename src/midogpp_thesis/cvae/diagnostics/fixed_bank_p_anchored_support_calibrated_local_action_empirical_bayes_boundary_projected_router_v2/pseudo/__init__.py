"""Strict pseudo-route scopes and closed-world seals."""

from .scope import PseudoRouteKey, PseudoRouteScope, build_pseudo_route_scopes
from .universe import (
    PseudoActionRecord,
    PseudoUniverseSeal,
    REQUIRED_COMPONENT_ROLES,
    build_pseudo_universe,
)

__all__ = tuple(name for name in globals() if not name.startswith("_"))
