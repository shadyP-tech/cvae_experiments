"""Durable contract for global and center-local endpoint surface lineage."""

from __future__ import annotations

from collections.abc import Mapping

from ...protocol import ProtocolError
from .constants import CENTERS
from .hashing import require_sha256


ENDPOINT_SURFACE_LINEAGE_SCHEMA_VERSION = (
    "fixed_bank_cbpupr_endpoint_surface_lineage_v3"
)
ROUTE_ENDPOINT_STATES_SCHEMA_VERSION = (
    "fixed_bank_cbpupr_route_endpoint_states_v3"
)


def endpoint_surface_lineage_payload(
    endpoint_products: object,
) -> dict[str, object]:
    """Bind endpoint products to one global and nine distinct center hashes."""

    rows = tuple(endpoint_products)
    if tuple(str(getattr(row, "target_center", "")) for row in rows) != CENTERS:
        raise ProtocolError("CBPUPR endpoint product surface rectangle drifted.")
    physical_hashes = {
        require_sha256(
            getattr(row, "physical_surface_hash", None),
            "CBPUPR persisted endpoint physical_surface_hash",
        )
        for row in rows
    }
    if len(physical_hashes) != 1:
        raise ProtocolError("CBPUPR endpoint product global surface drifted.")
    physical_surface_hash = physical_hashes.pop()
    center_surface_hashes = {
        center: require_sha256(
            getattr(row, "center_surface_hash", None),
            "CBPUPR persisted endpoint center_surface_hash",
        )
        for center, row in zip(CENTERS, rows, strict=True)
    }
    _require_distinct_surface_roles(
        physical_surface_hash=physical_surface_hash,
        center_surface_hashes=center_surface_hashes,
    )
    return {
        "schema_version": ENDPOINT_SURFACE_LINEAGE_SCHEMA_VERSION,
        "physical_surface_hash": physical_surface_hash,
        "center_surface_hashes": center_surface_hashes,
    }


def expected_endpoint_surface_lineage(surface: object) -> dict[str, object]:
    """Build the exact persisted envelope expected from a physical surface."""

    try:
        physical_surface_hash = require_sha256(
            getattr(surface, "surface_hash"),
            "CBPUPR endpoint origin physical_surface_hash",
        )
        centers = getattr(surface, "centers")
        center_surface_hashes = {
            center: require_sha256(
                centers[center].surface_hash,
                "CBPUPR endpoint origin center_surface_hash",
            )
            for center in CENTERS
        }
    except (AttributeError, KeyError, TypeError) as exc:
        raise ProtocolError("CBPUPR endpoint origin surface rectangle drifted.") from exc
    _require_distinct_surface_roles(
        physical_surface_hash=physical_surface_hash,
        center_surface_hashes=center_surface_hashes,
    )
    return {
        "schema_version": ENDPOINT_SURFACE_LINEAGE_SCHEMA_VERSION,
        "physical_surface_hash": physical_surface_hash,
        "center_surface_hashes": center_surface_hashes,
    }


def _require_distinct_surface_roles(
    *,
    physical_surface_hash: str,
    center_surface_hashes: Mapping[str, str],
) -> None:
    if (
        tuple(center_surface_hashes) != CENTERS
        or len(set(center_surface_hashes.values())) != len(CENTERS)
        or physical_surface_hash in set(center_surface_hashes.values())
    ):
        raise ProtocolError("CBPUPR endpoint surface hash roles drifted.")


__all__ = (
    "ENDPOINT_SURFACE_LINEAGE_SCHEMA_VERSION",
    "ROUTE_ENDPOINT_STATES_SCHEMA_VERSION",
    "endpoint_surface_lineage_payload",
    "expected_endpoint_surface_lineage",
)
