"""Canonical whole-case/row inventory for P-DCAPS lifecycle sealing.

The inventory contains identifiers only.  It is built from the label-free
manifest projection ``(center, case_id, sample_id)`` and binds the exact row
order and manifest identities without carrying labels into the router.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from .identity import canonical_hash, require_sha256


CANONICAL_CASE_COUNT = 218
CANONICAL_ROW_COUNT = 9_928


@dataclass(frozen=True)
class InventoryCase:
    center: str
    case_id: str
    sample_ids: tuple[str, ...]
    case_inventory_hash: str = field(init=False)

    def __post_init__(self) -> None:
        center = str(self.center)
        case_id = str(self.case_id)
        samples = tuple(str(value) for value in self.sample_ids)
        if (
            center not in CENTERS
            or not case_id
            or not samples
            or len(samples) != len(set(samples))
            or any(not value for value in samples)
        ):
            raise ProtocolError("P-DCAPS inventory case topology drifted.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(
            self,
            "case_inventory_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_inventory_case_v1",
                    "center": center,
                    "case_id": case_id,
                    "sample_ids": samples,
                    "labels_used": False,
                }
            ),
        )

    @property
    def key(self) -> tuple[str, str]:
        return self.center, self.case_id

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_inventory_case_v1",
            "center": self.center,
            "case_id": self.case_id,
            "sample_ids": list(self.sample_ids),
            "row_count": len(self.sample_ids),
            "labels_used": False,
            "case_inventory_hash": self.case_inventory_hash,
        }


@dataclass(frozen=True)
class ExpectedRouteInventory:
    """Exact route universe required before any pseudo response can open."""

    inventory_role: str
    cases: tuple[InventoryCase, ...]
    manifest_sha256: str
    row_order_hash: str
    inventory_hash: str = field(init=False)

    def __post_init__(self) -> None:
        role = str(self.inventory_role)
        cases = tuple(
            sorted(
                tuple(self.cases),
                key=lambda row: (CENTERS.index(row.center), row.case_id),
            )
        )
        require_sha256(self.manifest_sha256, "inventory manifest hash")
        require_sha256(self.row_order_hash, "inventory row-order hash")
        case_keys = tuple(row.key for row in cases)
        samples = tuple(
            sample_id for row in cases for sample_id in row.sample_ids
        )
        observed_centers = tuple(
            center for center in CENTERS if center in {row.center for row in cases}
        )
        if (
            role not in {"canonical_midogpp_consumed_test_v1", "focused_fixture"}
            or not cases
            or len(case_keys) != len(set(case_keys))
            or len(samples) != len(set(samples))
            or not observed_centers
        ):
            raise ProtocolError("P-DCAPS expected route inventory drifted.")
        if role == "canonical_midogpp_consumed_test_v1" and (
            observed_centers != CENTERS
            or len(cases) != CANONICAL_CASE_COUNT
            or len(samples) != CANONICAL_ROW_COUNT
        ):
            raise ProtocolError("P-DCAPS canonical test inventory is incomplete.")
        object.__setattr__(self, "inventory_role", role)
        object.__setattr__(self, "cases", cases)
        object.__setattr__(
            self,
            "inventory_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_expected_route_inventory_v1",
                    "inventory_role": role,
                    "manifest_sha256": self.manifest_sha256,
                    "row_order_hash": self.row_order_hash,
                    "case_inventory_hashes": tuple(
                        row.case_inventory_hash for row in cases
                    ),
                    "centers": observed_centers,
                    "case_count": len(cases),
                    "row_count": len(samples),
                    "target_route_count": len(cases),
                    "pseudo_route_count": (len(observed_centers) - 1) * len(cases),
                    "labels_used": False,
                }
            ),
        )

    @property
    def centers(self) -> tuple[str, ...]:
        return tuple(
            center for center in CENTERS if center in {row.center for row in self.cases}
        )

    @property
    def case_count(self) -> int:
        return len(self.cases)

    @property
    def row_count(self) -> int:
        return sum(len(row.sample_ids) for row in self.cases)

    @property
    def target_route_count(self) -> int:
        return self.case_count

    @property
    def pseudo_route_count(self) -> int:
        return (len(self.centers) - 1) * self.case_count

    @property
    def total_route_count(self) -> int:
        return len(self.centers) * self.case_count

    @classmethod
    def from_label_free_keys(
        cls,
        keys: Sequence[tuple[str, str, str]],
        *,
        manifest_sha256: str,
        row_order_hash: str,
    ) -> "ExpectedRouteInventory":
        return cls._from_keys(
            keys,
            inventory_role="canonical_midogpp_consumed_test_v1",
            manifest_sha256=manifest_sha256,
            row_order_hash=row_order_hash,
        )

    @classmethod
    def focused_fixture(
        cls,
        keys: Sequence[tuple[str, str, str]],
    ) -> "ExpectedRouteInventory":
        canonical_keys = tuple(
            (str(center), str(case_id), str(sample_id))
            for center, case_id, sample_id in keys
        )
        return cls._from_keys(
            canonical_keys,
            inventory_role="focused_fixture",
            manifest_sha256=canonical_hash(
                {"fixture_manifest_keys": canonical_keys, "labels_used": False}
            ),
            row_order_hash=canonical_hash(
                {"fixture_row_order": canonical_keys, "labels_used": False}
            ),
        )

    @classmethod
    def _from_keys(
        cls,
        keys: Sequence[tuple[str, str, str]],
        *,
        inventory_role: str,
        manifest_sha256: str,
        row_order_hash: str,
    ) -> "ExpectedRouteInventory":
        canonical_keys = tuple(
            (str(center), str(case_id), str(sample_id))
            for center, case_id, sample_id in keys
        )
        if (
            not canonical_keys
            or len(canonical_keys) != len(set(canonical_keys))
            or len({sample for _center, _case, sample in canonical_keys})
            != len(canonical_keys)
        ):
            raise ProtocolError("P-DCAPS inventory row keys are empty or duplicated.")
        grouped: dict[tuple[str, str], list[str]] = {}
        for center, case_id, sample_id in canonical_keys:
            grouped.setdefault((center, case_id), []).append(sample_id)
        cases = tuple(
            InventoryCase(center, case_id, tuple(samples))
            for (center, case_id), samples in grouped.items()
        )
        return cls(
            inventory_role,
            cases,
            str(manifest_sha256),
            str(row_order_hash),
        )

    def validate_draft_routes(self, routes: Sequence[object]) -> str:
        """Require one exact target-or-pseudo route per case for every H."""

        rows = tuple(routes)
        observed: dict[tuple[str, str, str, str], object] = {}
        for row in rows:
            route = getattr(row, "route_key", None)
            if route is None:
                raise ProtocolError("P-DCAPS action route lacks its route key.")
            key = (
                str(route.surface_role),
                str(route.outer_center),
                str(route.route_center),
                str(route.held_case_id),
            )
            if key in observed:
                raise ProtocolError("P-DCAPS action route inventory is duplicated.")
            observed[key] = row
        expected: dict[tuple[str, str, str, str], InventoryCase] = {}
        for outer in self.centers:
            for case in self.cases:
                role = "target" if case.center == outer else "pseudo"
                expected[(role, outer, case.center, case.case_id)] = case
        if set(observed) != set(expected):
            raise ProtocolError("P-DCAPS complete route/case inventory drifted.")
        for key, case in expected.items():
            row = observed[key]
            if tuple(getattr(row, "sample_ids", ())) != case.sample_ids:
                raise ProtocolError("P-DCAPS complete route row inventory drifted.")
        payload = {
            "schema_version": "pdcaps_validated_action_route_inventory_v1",
            "inventory_hash": self.inventory_hash,
            "route_draft_hashes": tuple(
                str(getattr(observed[key], "route_draft_hash"))
                for key in sorted(expected)
            ),
            "outer_centers": self.centers,
            "target_route_count": self.target_route_count,
            "pseudo_route_count": self.pseudo_route_count,
            "total_route_count": self.total_route_count,
            "unique_row_count": self.row_count,
            "row_occurrence_count": len(self.centers) * self.row_count,
            "labels_used": False,
        }
        return canonical_hash(payload)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_expected_route_inventory_v1",
            "inventory_role": self.inventory_role,
            "manifest_sha256": self.manifest_sha256,
            "row_order_hash": self.row_order_hash,
            "centers": list(self.centers),
            "case_count": self.case_count,
            "row_count": self.row_count,
            "target_route_count": self.target_route_count,
            "pseudo_route_count": self.pseudo_route_count,
            "total_route_count": self.total_route_count,
            "labels_used": False,
            "inventory_hash": self.inventory_hash,
        }


__all__ = (
    "CANONICAL_CASE_COUNT",
    "CANONICAL_ROW_COUNT",
    "ExpectedRouteInventory",
    "InventoryCase",
)
