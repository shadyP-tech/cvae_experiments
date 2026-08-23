"""Deterministic decision and exact-P fallback validation."""

from __future__ import annotations

from typing import Mapping, Sequence

from ....protocol import ProtocolError
from ..identity import METHOD_MENU, P_METHOD_ID


def validate_decision_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    values = tuple(dict(row) for row in rows)
    if not values:
        raise ProtocolError("P-DCAPS decision table is empty.")
    for row in values:
        if (
            row.get("method_id") not in METHOD_MENU
            or row.get("target_labels_used") is not False
            or row.get("selection_source") not in {
                "DONOR_CROSSFIT",
                "EXACT_P_FALLBACK",
                "FIXED_CONTROL",
            }
        ):
            raise ProtocolError("P-DCAPS decision row drifted.")
        if row.get("routed") is False and row.get("selected_method_id") != P_METHOD_ID:
            raise ProtocolError("P-DCAPS abstention is not exact P.")
    return {
        "status": "PASS",
        "decision_count": len(values),
        "fallback_count": sum(row.get("selected_method_id") == P_METHOD_ID for row in values),
    }


__all__ = ("validate_decision_rows",)
