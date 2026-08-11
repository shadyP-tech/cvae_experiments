"""Development-only simultaneous uncertainty diagnostics."""

from __future__ import annotations

from collections import defaultdict
from statistics import NormalDist
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import (
    CandidateContrastRow,
    DevelopmentSelectionDiagnostic,
    InferenceSelectionDiagnostic,
)
from .inference_contracts import (
    LabelFreeInferenceContext,
    assert_label_free_inference_context,
)
from .provenance import DevelopmentContext, assert_development_context


FAMILY_WISE_ALPHA = 0.05


def _selection_values(
    contrasts: Sequence[CandidateContrastRow],
    *,
    baseline_action_id: str,
    control_action_id: str,
    target_query_id: str,
) -> tuple[tuple[object, ...], ...]:
    rows = tuple(contrasts)
    if not rows or any(not isinstance(row, CandidateContrastRow) for row in rows):
        raise ProtocolError("Safe diagnostics require typed candidate contrasts.")
    if baseline_action_id == control_action_id:
        raise ProtocolError("Safe diagnostics require distinct B and U controls.")
    grouped: dict[tuple[str, str], list[CandidateContrastRow]] = defaultdict(list)
    for row in rows:
        if row.target_query_id != target_query_id:
            raise ProtocolError("Candidate contrasts drifted from the target context.")
        grouped[(row.family, row.case_id)].append(row)
    candidate_counts = {len(block) for block in grouped.values()}
    if len(candidate_counts) != 1 or next(iter(candidate_counts)) <= 0:
        raise ProtocolError("Every target case must cover the same candidate count.")
    candidate_count = next(iter(candidate_counts))
    z_value = NormalDist().inv_cdf(1.0 - FAMILY_WISE_ALPHA / (2.0 * candidate_count))
    output: list[tuple[object, ...]] = []
    for (family, case_id), block in sorted(grouped.items()):
        if len({row.candidate_action_id for row in block}) != len(block):
            raise ProtocolError("Safe diagnostics contain duplicate candidate actions.")
        margins = {
            row.candidate_action_id: min(
                row.predicted_preference_margin_vs_control
                - z_value * row.standard_error_vs_control,
                row.predicted_preference_margin_vs_baseline
                - z_value * row.standard_error_vs_baseline,
            )
            for row in block
        }
        raw = min(
            block,
            key=lambda row: (
                -min(
                    row.predicted_preference_margin_vs_control,
                    row.predicted_preference_margin_vs_baseline,
                ),
                row.candidate_action_id,
            ),
        )
        safe_candidates = tuple(
            row for row in block if margins[row.candidate_action_id] > 0.0
        )
        if safe_candidates:
            safe = min(
                safe_candidates,
                key=lambda row: (-margins[row.candidate_action_id], row.candidate_action_id),
            )
            safe_action = safe.candidate_action_id
            safe_margin = margins[safe_action]
            reason = "safe_candidate_strictly_positive_vs_b_and_u"
        else:
            safe_action = str(baseline_action_id)
            safe_margin = max(margins.values())
            reason = "simultaneous_lcb_nonpositive_vs_b_or_u"
        output.append(
            (
                family,
                case_id,
                raw.candidate_action_id,
                safe_action,
                z_value,
                safe_margin,
                reason,
            )
        )
    return tuple(output)


def build_safe_selection_diagnostics(
    contrasts: Sequence[CandidateContrastRow],
    *,
    baseline_action_id: str,
    control_action_id: str,
    context: DevelopmentContext,
) -> tuple[DevelopmentSelectionDiagnostic, ...]:
    """Describe raw/safe choices without emitting a policy or experiment row.

    A candidate is safe only when simultaneous one-sided lower bounds are
    strictly positive against both U and immutable B.  Otherwise B is retained.
    This output is structurally unable to authorize routing or promotion.
    """

    assert_development_context(context)
    output: list[DevelopmentSelectionDiagnostic] = []
    for family, case_id, raw_action, safe_action, z_value, safe_margin, reason in _selection_values(
        contrasts,
        baseline_action_id=baseline_action_id,
        control_action_id=control_action_id,
        target_query_id=context.outer_target_id,
    ):
        output.append(
            DevelopmentSelectionDiagnostic(
                family=family,
                target_query_id=context.outer_target_id,
                case_id=case_id,
                raw_action_id=raw_action,
                safe_action_id=safe_action,
                baseline_action_id=str(baseline_action_id),
                control_action_id=str(control_action_id),
                simultaneous_z_value=z_value,
                safe_margin=safe_margin,
                fallback_reason=reason,
            )
        )
    return tuple(output)


def build_label_free_inference_selection_diagnostics(
    contrasts: Sequence[CandidateContrastRow],
    *,
    context: LabelFreeInferenceContext,
) -> tuple[InferenceSelectionDiagnostic, ...]:
    """Build raw/safe consumed-target suggestions without a policy capability."""

    assert_label_free_inference_context(context)
    schema = context.action_schema
    return tuple(
        InferenceSelectionDiagnostic(
            family=family,
            target_query_id=context.outer_target_id,
            case_id=case_id,
            raw_action_id=raw_action,
            safe_action_id=safe_action,
            baseline_action_id=schema.baseline_action_id,
            control_action_id=schema.control_action_id,
            simultaneous_z_value=z_value,
            safe_margin=safe_margin,
            fallback_reason=reason,
        )
        for family, case_id, raw_action, safe_action, z_value, safe_margin, reason in _selection_values(
            contrasts,
            baseline_action_id=schema.baseline_action_id,
            control_action_id=schema.control_action_id,
            target_query_id=context.outer_target_id,
        )
    )


__all__ = (
    "FAMILY_WISE_ALPHA",
    "build_label_free_inference_selection_diagnostics",
    "build_safe_selection_diagnostics",
)
