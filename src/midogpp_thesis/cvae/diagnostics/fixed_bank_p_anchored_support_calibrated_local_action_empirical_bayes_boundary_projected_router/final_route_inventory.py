"""Closed-world final-route task universe for SCALE-BP."""

from __future__ import annotations

from dataclasses import dataclass, field

from .case_inventory import DatasetCaseInventory
from .hashing import canonical_hash, require_sha256
from .identity import CENTERS, EXPECTED_CASE_COUNT, EXPECTED_CASE_COUNTS_BY_CENTER
from .protocol import ProtocolError


@dataclass(frozen=True, slots=True)
class FinalRouteInventoryReceipt:
    """Primitive seal of the exact 9-center, 218-case final-route universe."""

    dataset_case_inventory_hash: str
    cache_content_hash: str
    row_order_hash: str
    manifest_hash: str
    cases_by_center: tuple[tuple[str, tuple[str, ...]], ...]
    task_universe_hash: str = field(init=False)
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        inventory_hash = require_sha256(
            self.dataset_case_inventory_hash, "final-route case-inventory hash"
        )
        cache_hash = require_sha256(
            self.cache_content_hash, "final-route cache-content hash"
        )
        row_hash = require_sha256(
            self.row_order_hash, "final-route row-order hash"
        )
        manifest_hash = require_sha256(
            self.manifest_hash, "final-route manifest hash"
        )
        rows = tuple(
            (str(center), tuple(str(case_id) for case_id in cases))
            for center, cases in self.cases_by_center
        )
        expected_counts = dict(EXPECTED_CASE_COUNTS_BY_CENTER)
        flattened = tuple(case_id for _center, cases in rows for case_id in cases)
        if (
            tuple(center for center, _cases in rows) != CENTERS
            or any(
                cases != tuple(sorted(set(cases)))
                or len(cases) != expected_counts[center]
                for center, cases in rows
            )
            or len(flattened) != EXPECTED_CASE_COUNT
            or len(set(flattened)) != EXPECTED_CASE_COUNT
        ):
            raise ProtocolError("SCALE-BP final-route task universe drifted.")
        reconstructed = DatasetCaseInventory(
            cache_hash,
            row_hash,
            manifest_hash,
            rows,
        )
        if reconstructed.inventory_hash != inventory_hash:
            raise ProtocolError("SCALE-BP final-route case-inventory lineage drifted.")
        task_universe_hash = canonical_hash(
            {
                "schema_version": "scale_bp_final_route_task_universe_v1",
                "cases_by_center": rows,
                "center_count": len(CENTERS),
                "case_count": EXPECTED_CASE_COUNT,
                "one_complete_outer_center_per_task": True,
            }
        )
        payload = {
            "schema_version": "scale_bp_final_route_inventory_receipt_v1",
            "dataset_case_inventory_hash": inventory_hash,
            "cache_content_hash": cache_hash,
            "row_order_hash": row_hash,
            "manifest_hash": manifest_hash,
            "task_universe_hash": task_universe_hash,
            "cases_by_center": rows,
            "center_count": len(CENTERS),
            "case_count": EXPECTED_CASE_COUNT,
            "closed_world": True,
        }
        object.__setattr__(self, "dataset_case_inventory_hash", inventory_hash)
        object.__setattr__(self, "cache_content_hash", cache_hash)
        object.__setattr__(self, "row_order_hash", row_hash)
        object.__setattr__(self, "manifest_hash", manifest_hash)
        object.__setattr__(self, "cases_by_center", rows)
        object.__setattr__(self, "task_universe_hash", task_universe_hash)
        object.__setattr__(self, "receipt_hash", canonical_hash(payload))

    @classmethod
    def from_case_inventory(
        cls, inventory: DatasetCaseInventory
    ) -> "FinalRouteInventoryReceipt":
        if not isinstance(inventory, DatasetCaseInventory):
            raise ProtocolError("SCALE-BP final-route inventory input drifted.")
        return cls(
            inventory.inventory_hash,
            inventory.cache_content_hash,
            inventory.row_order_hash,
            inventory.manifest_hash,
            inventory.cases_by_center,
        )

    @property
    def case_count(self) -> int:
        return sum(len(cases) for _center, cases in self.cases_by_center)

    def cases(self, center: str) -> tuple[str, ...]:
        matches = tuple(
            cases for candidate, cases in self.cases_by_center if candidate == center
        )
        if len(matches) != 1:
            raise ProtocolError("SCALE-BP final-route center lookup drifted.")
        return matches[0]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_final_route_inventory_receipt_v1",
            "dataset_case_inventory_hash": self.dataset_case_inventory_hash,
            "cache_content_hash": self.cache_content_hash,
            "row_order_hash": self.row_order_hash,
            "manifest_hash": self.manifest_hash,
            "task_universe_hash": self.task_universe_hash,
            "cases_by_center": self.cases_by_center,
            "center_count": len(CENTERS),
            "case_count": self.case_count,
            "closed_world": True,
            "receipt_hash": self.receipt_hash,
        }


def build_final_route_inventory_receipt(
    inventory: DatasetCaseInventory,
) -> FinalRouteInventoryReceipt:
    return FinalRouteInventoryReceipt.from_case_inventory(inventory)


__all__ = (
    "FinalRouteInventoryReceipt",
    "build_final_route_inventory_receipt",
)
