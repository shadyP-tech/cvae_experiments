"""Protocol guardrails for the extracted Virchow2 source-only pipeline."""

from __future__ import annotations

from typing import Mapping, Sequence


class ProtocolError(ValueError):
    """Raised when source-only selection or split safety is violated."""


ELIGIBILITY_DEPLOYABLE_DIAGNOSTIC = "deployable_diagnostic"
ELIGIBILITY_AUDIT_ONLY = "audit_only"

ROW_SOURCE_CANDIDATE = "source_inner_lodo_candidate"
ROW_SOURCE_TOP1 = "source_inner_lodo_selected_top1_virchow2"
ROW_SOURCE_DENSE = "source_inner_lodo_selected_dense_virchow2"


def bool_text(value: bool) -> str:
    return str(bool(value)).lower()


def assert_disjoint_ids(left: Sequence[str] | set[str], right: Sequence[str] | set[str]) -> None:
    overlap = sorted(set(left).intersection(set(right)))
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ProtocolError(f"Support/evaluation overlap detected: {preview}")


def validate_primary_rows(rows: Sequence[Mapping[str, object]]) -> None:
    """Reject rows that used target labels or target-center fitting for selection."""

    violations: list[str] = []
    for row in rows:
        if str(row.get("row_role")) not in {ROW_SOURCE_TOP1, ROW_SOURCE_DENSE}:
            continue
        if str(row.get("selection_used_target_labels")) != "false":
            violations.append(f"{row.get('row_id')}: target labels used for selection")
        if str(row.get("fit_used_target_center")) != "false":
            violations.append(f"{row.get('row_id')}: target center used for fitting")
        if str(row.get("selected_by_source_inner_lodo")) != "true":
            violations.append(f"{row.get('row_id')}: primary row was not source-selected")
    if violations:
        raise ProtocolError("; ".join(violations))
