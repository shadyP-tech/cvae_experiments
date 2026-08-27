"""Frozen SCALE-BP method and negative-control identities."""

from __future__ import annotations

from dataclasses import dataclass

from .hashing import canonical_hash
from .identity import (
    CYCLIC_METHOD_ID,
    DONOR_ONLY_METHOD_ID,
    FULL_ENDPOINT_METHOD_ID,
    LEGACY_METHOD_ID,
    LOCAL_ONLY_METHOD_ID,
    METHOD_MENU,
    P_METHOD_ID,
    PRIMARY_METHOD_ID,
    SUPPORT_PERMUTATION_METHOD_ID,
)
from .protocol import ProtocolError


P_PROTECTED = P_METHOD_ID
SCALE_BP_PRIMARY = PRIMARY_METHOD_ID
DONOR_ONLY = DONOR_ONLY_METHOD_ID
LOCAL_ONLY = LOCAL_ONLY_METHOD_ID
LEGACY_SAME_RUN = LEGACY_METHOD_ID
SUPPORT_LABEL_PERMUTATION = SUPPORT_PERMUTATION_METHOD_ID
CYCLIC_ACTION_IDENTITY = CYCLIC_METHOD_ID
FULL_ENDPOINT_SENSITIVITY = FULL_ENDPOINT_METHOD_ID

METHOD_IDS = METHOD_MENU
NEGATIVE_CONTROL_IDS = (SUPPORT_LABEL_PERMUTATION, CYCLIC_ACTION_IDENTITY)
REQUIRED_CONTROL_METHOD_IDS = tuple(
    method_id for method_id in METHOD_IDS if method_id != SCALE_BP_PRIMARY
)


@dataclass(frozen=True, slots=True)
class ControlInventory:
    method_ids: tuple[str, ...] = METHOD_IDS
    primary_method_id: str = SCALE_BP_PRIMARY
    protected_fallback_id: str = P_PROTECTED
    control_hash: str = ""

    def __post_init__(self) -> None:
        if (
            tuple(self.method_ids) != METHOD_IDS
            or self.primary_method_id != SCALE_BP_PRIMARY
            or self.protected_fallback_id != P_PROTECTED
        ):
            raise ProtocolError("SCALE-BP control inventory drifted.")
        digest = canonical_hash(
            {
                "schema_version": "scale_bp_control_inventory_v1",
                "method_ids": list(METHOD_IDS),
                "required_control_method_ids": list(REQUIRED_CONTROL_METHOD_IDS),
                "primary_method_id": SCALE_BP_PRIMARY,
                "protected_fallback_id": P_PROTECTED,
                "negative_control_ids": list(NEGATIVE_CONTROL_IDS),
                "terminal_labels_may_select_control": False,
            }
        )
        if self.control_hash not in {"", digest}:
            raise ProtocolError("SCALE-BP control inventory hash drifted.")
        object.__setattr__(self, "control_hash", digest)


__all__ = (
    "CYCLIC_ACTION_IDENTITY",
    "ControlInventory",
    "DONOR_ONLY",
    "FULL_ENDPOINT_SENSITIVITY",
    "LEGACY_SAME_RUN",
    "LOCAL_ONLY",
    "METHOD_IDS",
    "NEGATIVE_CONTROL_IDS",
    "P_PROTECTED",
    "REQUIRED_CONTROL_METHOD_IDS",
    "SCALE_BP_PRIMARY",
    "SUPPORT_LABEL_PERMUTATION",
)
