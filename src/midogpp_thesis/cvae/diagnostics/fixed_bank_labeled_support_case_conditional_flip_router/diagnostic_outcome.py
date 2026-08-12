"""Predeclared terminal diagnostic recoverability gate."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError


def diagnostic_recoverability_outcome(
    contrast_rows: Sequence[Mapping[str, object]],
    *,
    evaluation: Mapping[str, object],
) -> Mapping[str, object]:
    """Apply the frozen five-contrast outer-center LCB rule exactly once."""

    contrast_ids = tuple(str(value) for value in evaluation.get("primary_contrasts", ()))
    rule = evaluation.get("diagnostic_recoverability_gate")
    if (
        not isinstance(rule, Mapping)
        or rule.get("gate_id")
        != "all_primary_contrast_outer_center_lcbs_positive_v1"
        or rule.get("lcb_field") != "one_sided_95_lcb"
        or float(rule.get("threshold", math.nan)) != 0.0
        or rule.get("comparison") != "strictly_greater_than"
        or int(rule.get("required_contrast_count", -1)) != 5
        or rule.get("pass_status") != "PASS"
        or rule.get("fail_status") != "FAIL"
        or rule.get("diagnostic_only") is not True
        or len(contrast_ids) != 5
        or len(set(contrast_ids)) != 5
    ):
        raise ProtocolError("Flip-router diagnostic recoverability rule drifted.")
    aggregates = {
        str(row.get("contrast_id")): row
        for row in contrast_rows
        if row.get("row_role") == "outer_center_aggregate"
    }
    if set(aggregates) != set(contrast_ids) or sum(
        row.get("row_role") == "outer_center_aggregate" for row in contrast_rows
    ) != len(contrast_ids):
        raise ProtocolError("Flip-router aggregate contrast topology drifted.")
    lcb_by_contrast: dict[str, float] = {}
    for contrast_id in contrast_ids:
        value = aggregates[contrast_id].get("one_sided_95_lcb")
        try:
            lcb = float(value)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Flip-router aggregate contrast LCB is absent.") from exc
        if not math.isfinite(lcb):
            raise ProtocolError("Flip-router aggregate contrast LCB is not finite.")
        lcb_by_contrast[contrast_id] = lcb
    passed = all(value > 0.0 for value in lcb_by_contrast.values())
    return MappingProxyType(
        {
            "schema_version": "fixed_bank_labeled_support_flip_diagnostic_recoverability_v1",
            "gate_id": str(rule["gate_id"]),
            "status": "PASS" if passed else "FAIL",
            "contrast_one_sided_95_lcb": lcb_by_contrast,
            "all_five_lcbs_strictly_greater_than_zero": passed,
            "threshold": 0.0,
            "comparison": "strictly_greater_than",
            "required_contrast_count": 5,
            "diagnostic_only": True,
            "routing_success_claimed": False,
            "routing_quality_claimed": False,
            "promotion_eligible": False,
        }
    )


__all__ = ("diagnostic_recoverability_outcome",)
