from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


PRIMARY_METHOD = "support_nelbo_top2_geom"
METHOD_ROWS = (
    "real_feature_source_top1_reference",
    "cvae_source_top1_synthetic_reference",
    "support_nelbo_top1",
    "support_nelbo_top2_geom",
    "support_nelbo_top3_geom",
    "all4_geom",
    "metadata_top1",
    "metadata_top2_geom",
    "random_top1",
    "random_top2_geom",
    "downstream_oracle_diagnostic_only",
)
DEPLOYABLE_ROWS = {
    "cvae_source_top1_synthetic_reference",
    "support_nelbo_top1",
    "support_nelbo_top2_geom",
    "support_nelbo_top3_geom",
    "all4_geom",
    "metadata_top1",
    "metadata_top2_geom",
    "random_top1",
    "random_top2_geom",
}
ORACLE_ROW = "downstream_oracle_diagnostic_only"


class ProtocolError(ValueError):
    """Raised when a run would violate the locked thesis protocol."""


@dataclass(frozen=True)
class LeakageReport:
    status: str
    violations: tuple[str, ...]
    target_support_labels_for_selection: bool
    target_eval_labels_for_scoring_only: bool
    target_expert_excluded: bool
    oracle_rows_diagnostic_only: bool
    schema_version: str = "midogpp_cvae_leakage_report_v1"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "violations": list(self.violations),
            "target_support_labels_for_selection": self.target_support_labels_for_selection,
            "target_eval_labels_for_scoring_only": self.target_eval_labels_for_scoring_only,
            "target_expert_excluded": self.target_expert_excluded,
            "oracle_rows_diagnostic_only": self.oracle_rows_diagnostic_only,
        }


def split_budget(total_per_class: int, selected_experts: Sequence[str]) -> dict[str, int]:
    """Allocate a fixed total per-class budget in rank order.

    Remainders are assigned to earlier ranked experts. This intentionally
    differs from sorted-ID allocation because top-k rows are rank-sensitive.
    """

    if total_per_class <= 0:
        raise ProtocolError("total_per_class must be positive.")
    if not selected_experts:
        raise ProtocolError("selected_experts must be non-empty.")
    n = len(selected_experts)
    base = int(total_per_class) // n
    remainder = int(total_per_class) % n
    return {
        str(expert): base + (1 if idx < remainder else 0)
        for idx, expert in enumerate(str(v) for v in selected_experts)
    }


def assert_candidate_pool(
    *,
    heldout_center: str,
    candidate_experts: Sequence[str],
    expected_count: int = 4,
) -> tuple[str, ...]:
    candidates = tuple(str(v) for v in candidate_experts)
    if str(heldout_center) in candidates:
        raise ProtocolError("Held-out target expert appeared in candidate pool.")
    if len(candidates) != len(set(candidates)):
        raise ProtocolError("Candidate experts must be unique.")
    if len(candidates) != int(expected_count):
        raise ProtocolError(
            f"Expected {expected_count} source experts, got {len(candidates)}."
        )
    return candidates


def assert_support_eval_disjoint(
    support_sample_ids: Iterable[str],
    eval_sample_ids: Iterable[str],
) -> None:
    overlap = set(str(v) for v in support_sample_ids).intersection(str(v) for v in eval_sample_ids)
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise ProtocolError(f"Support/evaluation overlap detected: {preview}")


def assert_support_labels_unused(labels_used: bool) -> None:
    if bool(labels_used):
        raise ProtocolError("Target-support labels cannot be used for routing.")


def assert_oracle_diagnostic_only(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        method = str(row.get("method") or row.get("prior_method") or "")
        selection_source = str(row.get("selection_source", ""))
        if _is_oracle_row(method) and selection_source != "diagnostic_only":
            raise ProtocolError("Downstream oracle rows must be diagnostic-only.")
        if method in DEPLOYABLE_ROWS and "oracle" in selection_source:
            raise ProtocolError(f"Deployable row {method} used oracle selection source.")


def _is_oracle_row(method: str) -> bool:
    normalized = str(method)
    return normalized == ORACLE_ROW or "oracle" in normalized


def build_leakage_report(
    *,
    target_support_labels_for_selection: bool,
    target_eval_labels_for_scoring_only: bool,
    target_expert_excluded: bool,
    oracle_rows_diagnostic_only: bool,
    extra_violations: Sequence[str] = (),
) -> LeakageReport:
    violations = list(str(v) for v in extra_violations)
    if target_support_labels_for_selection:
        violations.append("target_support_labels_for_selection")
    if not target_eval_labels_for_scoring_only:
        violations.append("target_eval_labels_not_scoring_only")
    if not target_expert_excluded:
        violations.append("target_expert_not_excluded")
    if not oracle_rows_diagnostic_only:
        violations.append("oracle_rows_not_diagnostic_only")
    return LeakageReport(
        status="PASS" if not violations else "FAIL",
        violations=tuple(violations),
        target_support_labels_for_selection=bool(target_support_labels_for_selection),
        target_eval_labels_for_scoring_only=bool(target_eval_labels_for_scoring_only),
        target_expert_excluded=bool(target_expert_excluded),
        oracle_rows_diagnostic_only=bool(oracle_rows_diagnostic_only),
    )
