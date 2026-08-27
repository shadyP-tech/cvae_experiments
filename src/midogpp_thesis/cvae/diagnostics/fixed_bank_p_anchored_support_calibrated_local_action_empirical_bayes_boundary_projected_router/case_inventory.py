"""Sealed canonical MIDOG++ case universe used by every SCALE-BP replay."""

from __future__ import annotations

from dataclasses import dataclass, field

from .hashing import canonical_hash, require_sha256
from .identity import CENTERS, EXPECTED_CASE_COUNT, EXPECTED_CASE_COUNTS_BY_CENTER
from .protocol import ProtocolError


@dataclass(frozen=True, slots=True)
class DatasetCaseInventory:
    cache_content_hash: str
    row_order_hash: str
    manifest_hash: str
    cases_by_center: tuple[tuple[str, tuple[str, ...]], ...]
    inventory_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cache_hash = require_sha256(self.cache_content_hash, "case-inventory cache hash")
        row_hash = require_sha256(self.row_order_hash, "case-inventory row-order hash")
        manifest_hash = require_sha256(self.manifest_hash, "case-inventory manifest hash")
        rows = tuple(
            (str(center), tuple(str(case) for case in cases))
            for center, cases in self.cases_by_center
        )
        expected_counts = dict(EXPECTED_CASE_COUNTS_BY_CENTER)
        all_cases = tuple(case for _center, cases in rows for case in cases)
        if (
            tuple(center for center, _cases in rows) != CENTERS
            or any(
                cases != tuple(sorted(set(cases)))
                or len(cases) != expected_counts[center]
                or any(not case for case in cases)
                for center, cases in rows
            )
            or len(all_cases) != EXPECTED_CASE_COUNT
            or len(set(all_cases)) != EXPECTED_CASE_COUNT
        ):
            raise ProtocolError("SCALE-BP canonical case inventory drifted.")
        payload = {
            "schema_version": "scale_bp_dataset_case_inventory_v1",
            "cache_content_hash": cache_hash,
            "row_order_hash": row_hash,
            "manifest_hash": manifest_hash,
            "cases_by_center": rows,
            "center_count": len(CENTERS),
            "case_count": EXPECTED_CASE_COUNT,
        }
        object.__setattr__(self, "cache_content_hash", cache_hash)
        object.__setattr__(self, "row_order_hash", row_hash)
        object.__setattr__(self, "manifest_hash", manifest_hash)
        object.__setattr__(self, "cases_by_center", rows)
        object.__setattr__(self, "inventory_hash", canonical_hash(payload))

    def cases(self, center: str) -> tuple[str, ...]:
        matches = tuple(cases for candidate, cases in self.cases_by_center if candidate == center)
        if len(matches) != 1:
            raise ProtocolError("SCALE-BP case-inventory center lookup drifted.")
        return matches[0]


__all__ = ("DatasetCaseInventory",)
