"""Primitive-only DTOs for SCALE-BP process boundaries.

The authorized successor will pass paths and offsets to read-only memmaps.
Estimator objects, mapping proxies, closures, file handles, and memmap instances
are deliberately absent so the spawn boundary is deterministic and pickle-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..hashing import canonical_hash, require_sha256
from ..identity import CENTERS, SUPPORT_FOLD_COUNT
from ..protocol import ProtocolError
from .memmap_contracts import MemmapReference
from .physical_bank import PhysicalBankReceipt


@dataclass(frozen=True, slots=True)
class OuterCenterTask:
    """One complete outer-H task; support folds remain sequential within it."""

    target_center: str
    case_ids: tuple[str, ...]
    physical_bank: PhysicalBankReceipt
    protocol_hash: str
    final_route_inventory_hash: str
    support_fold_count: int = SUPPORT_FOLD_COUNT
    task_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cases = tuple(str(value) for value in self.case_ids)
        if (
            self.target_center not in CENTERS
            or not cases
            or len(cases) != len(set(cases))
            or tuple(sorted(cases)) != cases
            or not isinstance(self.physical_bank, PhysicalBankReceipt)
            or self.support_fold_count != SUPPORT_FOLD_COUNT
        ):
            raise ProtocolError("SCALE-BP outer task topology drifted.")
        require_sha256(self.protocol_hash, "protocol hash")
        inventory_hash = require_sha256(
            self.final_route_inventory_hash, "final-route inventory hash"
        )
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "final_route_inventory_hash", inventory_hash)
        object.__setattr__(
            self,
            "task_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_outer_center_task_v3",
                    "target_center": self.target_center,
                    "case_ids": list(cases),
                    "physical_bank_receipt_hash": self.physical_bank.receipt_hash,
                    "protocol_hash": self.protocol_hash,
                    "final_route_inventory_hash": inventory_hash,
                    "support_fold_count": self.support_fold_count,
                    "task_unit": "one_complete_outer_H",
                    "support_folds_sequential_inside_worker": True,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class OuterCenterResult:
    """Small deterministic result returned across the spawn boundary."""

    target_center: str
    task_hash: str
    final_route_inventory_hash: str
    case_ids: tuple[str, ...]
    route_hashes: tuple[str, ...]
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cases = tuple(str(value) for value in self.case_ids)
        rows = tuple(str(value) for value in self.route_hashes)
        if (
            self.target_center not in CENTERS
            or not cases
            or cases != tuple(sorted(set(cases)))
            or not rows
            or len(rows) != len(cases)
            or len(rows) != len(set(rows))
        ):
            raise ProtocolError("SCALE-BP outer result topology drifted.")
        require_sha256(self.task_hash, "outer task hash")
        inventory_hash = require_sha256(
            self.final_route_inventory_hash, "outer result inventory hash"
        )
        for digest in rows:
            require_sha256(digest, "route hash")
        object.__setattr__(self, "final_route_inventory_hash", inventory_hash)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "route_hashes", rows)
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_outer_center_result_v2",
                    "target_center": self.target_center,
                    "task_hash": self.task_hash,
                    "final_route_inventory_hash": inventory_hash,
                    "case_ids": list(cases),
                    "route_hashes": list(rows),
                }
            ),
        )


__all__ = ("MemmapReference", "OuterCenterResult", "OuterCenterTask")
