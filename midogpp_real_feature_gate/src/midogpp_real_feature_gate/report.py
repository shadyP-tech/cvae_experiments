"""Report builders for validated gate artifacts."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .contracts import DEFAULT_GATE_CRITERIA, GateCriteria, RowRole


def summarize_decision_labels(
    rows: Sequence[dict[str, object]],
    *,
    criteria: GateCriteria = DEFAULT_GATE_CRITERIA,
    artifact_completeness_pass: bool = True,
    leakage_provenance_pass: bool = True,
    negative_controls_pass: bool = True,
) -> tuple[str, ...]:
    """Return decision labels for a validated artifact bundle.

    This is intentionally conservative. It can allow only exploratory CVAE
    candidate-surface work, never claims about CVAE utility.
    """
    labels: list[str] = []
    if not artifact_completeness_pass:
        labels.append("NO_GO_ARTIFACT_INCOMPLETE")
    if not leakage_provenance_pass:
        labels.append("NO_GO_LEAKAGE_OR_PROVENANCE_FAILED")
    if not negative_controls_pass:
        labels.append("NO_GO_NEGATIVE_CONTROLS_FAILED")

    source_rows = [
        row
        for row in rows
        if str(row.get("row_role")) == RowRole.SOURCE_ONLY_TRANSFER and str(row.get("status")) == "valid"
    ]
    eligible_denominator = len(
        {
            str(row.get("heldout_center"))
            for row in rows
            if str(row.get("row_role")) == RowRole.SOURCE_ONLY_TRANSFER
        }
    )
    valid_fraction = float(len(source_rows) / eligible_denominator) if eligible_denominator else 0.0
    if valid_fraction < criteria.min_valid_eligible_fold_fraction:
        labels.append("NO_GO_INSUFFICIENT_VALID_ELIGIBLE_FOLDS")

    source_scores = [
        row
        for row in source_rows
        if _float(row.get("balanced_accuracy")) >= criteria.min_source_only_bacc
        or _float(row.get("auroc")) >= criteria.min_source_only_auroc
    ]
    if not source_scores:
        labels.append("NO_GO_SOURCE_ONLY_NEAR_CHANCE")

    worst_bacc = min((_float(row.get("balanced_accuracy")) for row in source_rows), default=math.nan)
    if not math.isnan(worst_bacc) and worst_bacc < criteria.worst_center_collapse_bacc:
        labels.append("WARN_WORST_CENTER_COLLAPSE")

    headroom_centers = _headroom_centers(rows, criteria)
    if len(headroom_centers) < criteria.min_headroom_centers:
        labels.append("NO_GO_NO_CLEAR_HEADROOM")

    if any(label.startswith("NO_GO") for label in labels):
        return tuple(labels)
    if any(label.startswith("WARN") for label in labels):
        labels.append("CONDITIONAL_GO_REVIEW_FAILURE_MODE")
    else:
        labels.append("GO_REAL_FEATURE_GATE_PASSED")
    labels.append("CLAIM_SCOPE_REAL_FEATURE_TRANSFER_ONLY")
    return tuple(labels)


def _headroom_centers(rows: Sequence[dict[str, object]], criteria: GateCriteria) -> set[str]:
    by_center: dict[str, dict[str, float]] = {}
    for row in rows:
        center = str(row.get("heldout_center", ""))
        role = str(row.get("row_role", ""))
        score = max(_float(row.get("balanced_accuracy")), _float(row.get("macro_f1")))
        if math.isnan(score):
            continue
        bucket = by_center.setdefault(center, {})
        if role == RowRole.SOURCE_ONLY_TRANSFER:
            bucket["source"] = max(bucket.get("source", -math.inf), score)
        elif role in {RowRole.POOLED_DIAGNOSTIC_CEILING, RowRole.SOURCE_ORACLE_DIAGNOSTIC}:
            bucket["diagnostic"] = max(bucket.get("diagnostic", -math.inf), score)
    return {
        center
        for center, values in by_center.items()
        if values.get("diagnostic", -math.inf) - values.get("source", math.inf) >= criteria.min_headroom_delta
    }


def _float(value: object) -> float:
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan
