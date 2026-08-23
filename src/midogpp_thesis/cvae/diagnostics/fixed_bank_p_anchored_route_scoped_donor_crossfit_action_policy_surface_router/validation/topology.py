"""H/J/d and surface-topology validation."""

from __future__ import annotations

from typing import Sequence

from ....protocol import ProtocolError
from ..contracts import RouteKey


def validate_route_exclusions(routes: Sequence[RouteKey]) -> dict[str, object]:
    rows = tuple(routes)
    if not rows or len({row.to_payload()["exclusion_hash"] for row in rows}) != len(rows):
        raise ProtocolError("P-DCAPS route exclusion inventory is empty or duplicate.")
    for row in rows:
        if row.excluded_outer_center != row.outer_center:
            raise ProtocolError("P-DCAPS route lost its outer-H exclusion.")
        if row.surface_role == "pseudo" and row.excluded_scored_center != row.route_center:
            raise ProtocolError("P-DCAPS route lost its scored-J exclusion.")
        if row.surface_role == "target" and row.excluded_scored_center is not None:
            raise ProtocolError("P-DCAPS target route invented a pseudo-center exclusion.")
    return {
        "status": "PASS",
        "route_count": len(rows),
        "target_route_count": sum(row.surface_role == "target" for row in rows),
        "pseudo_route_count": sum(row.surface_role == "pseudo" for row in rows),
    }


__all__ = ("validate_route_exclusions",)
