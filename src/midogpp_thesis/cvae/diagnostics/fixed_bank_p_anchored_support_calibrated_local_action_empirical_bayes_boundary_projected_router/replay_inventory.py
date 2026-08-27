"""Closed-world expected universe for one outer-H pseudo admission replay."""

from __future__ import annotations

from dataclasses import dataclass, field

from .case_inventory import DatasetCaseInventory
from .controls import METHOD_IDS
from .hashing import canonical_hash, require_sha256
from .identity import ACTION_IDS, CENTERS
from .protocol import ProtocolError
from .replay_scope import PseudoReplayScope


@dataclass(frozen=True, slots=True)
class PseudoReplayInventoryReceipt:
    outer_center: str
    case_inventory: DatasetCaseInventory
    scope_bindings: tuple[tuple[str, str, str], ...]
    method_ids: tuple[str, ...] = METHOD_IDS
    action_ids: tuple[str, ...] = ACTION_IDS
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = str(self.outer_center)
        if not isinstance(self.case_inventory, DatasetCaseInventory):
            raise ProtocolError("SCALE-BP pseudo replay inventory drifted.")
        bindings = tuple(
            (str(center), str(case), str(scope_hash))
            for center, case, scope_hash in self.scope_bindings
        )
        for _center, _case, digest in bindings:
            require_sha256(digest, "pseudo replay scope-binding hash")
        expected_contexts = tuple(
            (center, case)
            for center in CENTERS
            if center != outer
            for case in self.case_inventory.cases(center)
        )
        if (
            outer not in CENTERS
            or tuple((center, case) for center, case, _digest in bindings)
            != expected_contexts
            or len({digest for _center, _case, digest in bindings}) != len(bindings)
            or self.method_ids != METHOD_IDS
            or self.action_ids != ACTION_IDS
        ):
            raise ProtocolError("SCALE-BP pseudo replay inventory drifted.")
        payload = {
            "schema_version": "scale_bp_pseudo_replay_inventory_receipt_v1",
            "outer_center": outer,
            "case_inventory_hash": self.case_inventory.inventory_hash,
            "cache_content_hash": self.case_inventory.cache_content_hash,
            "row_order_hash": self.case_inventory.row_order_hash,
            "manifest_hash": self.case_inventory.manifest_hash,
            "scope_bindings": bindings,
            "method_ids": self.method_ids,
            "action_ids": self.action_ids,
            "unfavorable_contexts_may_be_omitted": False,
        }
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(self, "scope_bindings", bindings)
        object.__setattr__(self, "receipt_hash", canonical_hash(payload))

    @property
    def expected_context_count(self) -> int:
        return len(self.scope_bindings)


def build_pseudo_replay_inventory(
    scopes: object,
    *,
    outer_center: str,
    case_inventory: DatasetCaseInventory,
) -> PseudoReplayInventoryReceipt:
    if not isinstance(case_inventory, DatasetCaseInventory):
        raise ProtocolError("SCALE-BP pseudo replay scope inventory drifted.")
    try:
        rows = tuple(scopes)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProtocolError(
            "SCALE-BP pseudo replay scope inventory drifted."
        ) from exc
    if any(not isinstance(row, PseudoReplayScope) for row in rows):
        raise ProtocolError("SCALE-BP pseudo replay scope inventory drifted.")
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                CENTERS.index(row.pseudo_center),
                row.held_case_id,
            ),
        )
    )
    if (
        any(row.outer_center != outer_center for row in ordered)
        or any(
            row.case_inventory.inventory_hash != case_inventory.inventory_hash
            for row in ordered
        )
    ):
        raise ProtocolError("SCALE-BP pseudo replay scope inventory drifted.")
    return PseudoReplayInventoryReceipt(
        outer_center=str(outer_center),
        case_inventory=case_inventory,
        scope_bindings=tuple(
            (row.pseudo_center, row.held_case_id, row.scope_hash) for row in ordered
        ),
    )


__all__ = ("PseudoReplayInventoryReceipt", "build_pseudo_replay_inventory")
