"""Pre-support B/U/G/R/P and support-only static S_y decisions."""

from __future__ import annotations

import math
from typing import Sequence

from ...protocol import ProtocolError
from .constants import (
    B_ACTION_ID,
    GEOMETRY_IDS,
    MIDOGPP_CENTERS,
    U_ACTION_ID,
    candidate_sources,
    geometry_action_id,
)
from .contracts import ActionScoreRow, CaseConfusionCounts, MethodDecision


def build_pre_support_decisions(
    *,
    target_center: str,
    case_ids: Sequence[str],
    action_scores: Sequence[ActionScoreRow],
    minimum_gain: float = 0.0,
) -> tuple[MethodDecision, ...]:
    """Seal global B and parallel per-geometry U/G/R/P decisions.

    G/R/P fall back to the shared U control.  A0 and A1 are emitted as parallel rows;
    this function contains no cross-geometry comparison or selector.
    """

    target = str(target_center)
    if target not in MIDOGPP_CENTERS:
        raise ProtocolError("Pre-support decisions use an unknown target center.")
    cases = tuple(sorted(set(str(value) for value in case_ids)))
    if not cases or len(cases) != len(tuple(case_ids)):
        raise ProtocolError("Pre-support cases must be non-empty and unique.")
    threshold = float(minimum_gain)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ProtocolError("Pre-support minimum gain must be fixed and non-negative.")
    scores = tuple(action_scores)
    by_key = {row.row_key: row for row in scores}
    if len(by_key) != len(scores):
        raise ProtocolError("Pre-support action scores contain duplicate rows.")
    expected_keys = {
        (target, case_id, geometry, family, source)
        for case_id in cases
        for geometry in GEOMETRY_IDS
        for family in ("G", "R", "P")
        for source in candidate_sources(target)
    }
    if set(by_key) != expected_keys:
        raise ProtocolError("Pre-support action-score surface is incomplete or out of scope.")
    # G is deliberately target-global: each source score must be case invariant.
    for geometry in GEOMETRY_IDS:
        for source in candidate_sources(target):
            values = {
                by_key[(target, case_id, geometry, "G", source)].predicted_gain
                for case_id in cases
            }
            if len(values) != 1:
                raise ProtocolError("G must remain a static candidate prior within target/geometry.")

    output: list[MethodDecision] = []
    source_order = candidate_sources(target)
    for case_id in cases:
        output.append(
            MethodDecision(
                target_center=target,
                case_id=case_id,
                method_id="B",
                action_id=B_ACTION_ID,
                geometry_id=None,
                predicted_gain=0.0,
                decision_source="pre_support_global_baseline",
            )
        )
        for geometry in GEOMETRY_IDS:
            output.append(
                MethodDecision(
                    target_center=target,
                    case_id=case_id,
                    method_id="U",
                    action_id=U_ACTION_ID,
                    geometry_id=geometry,
                    predicted_gain=None,
                    decision_source="pre_support_uniform_control",
                )
            )
            for family in ("G", "R", "P"):
                best = max(
                    (
                        by_key[(target, case_id, geometry, family, source)]
                        for source in source_order
                    ),
                    key=lambda row: (
                        row.predicted_gain,
                        -source_order.index(row.selected_source),
                    ),
                )
                selected = (
                    geometry_action_id(geometry, best.selected_source)
                    if best.predicted_gain > threshold
                    else U_ACTION_ID
                )
                output.append(
                    MethodDecision(
                        target_center=target,
                        case_id=case_id,
                        method_id=family,
                        action_id=selected,
                        geometry_id=geometry,
                        predicted_gain=best.predicted_gain,
                        decision_source=(
                            "pre_support_fixed_alpha_ridge"
                            if selected != U_ACTION_ID
                            else "pre_support_shared_U_fallback"
                        ),
                    )
                )
    return tuple(output)


def _pooled_bacc(rows: Sequence[CaseConfusionCounts]) -> float:
    n_positive = sum(row.n_positive for row in rows)
    n_negative = sum(row.n_negative for row in rows)
    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("Support selection requires both classes after pooling.")
    return 0.5 * (
        sum(row.true_positive for row in rows) / n_positive
        + sum(row.true_negative for row in rows) / n_negative
    )


def build_support_static_decisions(
    *,
    target_center: str,
    geometry_id: str,
    support_counts: Sequence[CaseConfusionCounts],
    evaluation_case_ids: Sequence[str],
) -> tuple[MethodDecision, ...]:
    """Select one U-or-source action on pooled support, then freeze it for a fold."""

    target = str(target_center)
    if target not in MIDOGPP_CENTERS or geometry_id not in GEOMETRY_IDS:
        raise ProtocolError("Support selector has an invalid target/geometry context.")
    rows = tuple(support_counts)
    if not rows or any(row.target_center != target for row in rows):
        raise ProtocolError("Support counts must be non-empty and target-local.")
    support_cases = tuple(sorted({row.case_id for row in rows}))
    evaluation_cases = tuple(sorted(set(str(value) for value in evaluation_case_ids)))
    if not evaluation_cases or len(evaluation_cases) != len(tuple(evaluation_case_ids)):
        raise ProtocolError("Evaluation cases must be non-empty and unique.")
    overlap = set(support_cases).intersection(evaluation_cases)
    if overlap:
        raise ProtocolError("Support and evaluation cases must be whole-case disjoint.")
    action_order = (
        U_ACTION_ID,
        *(geometry_action_id(geometry_id, source) for source in candidate_sources(target)),
    )
    by_action_case = {(row.action_id, row.case_id): row for row in rows if row.action_id in action_order}
    expected = {(action, case_id) for action in action_order for case_id in support_cases}
    if set(by_action_case) != expected:
        raise ProtocolError("S_y support surface must cover U and every geometry action.")
    scores = {
        action: _pooled_bacc(tuple(by_action_case[(action, case_id)] for case_id in support_cases))
        for action in action_order
    }
    selected = max(action_order, key=lambda action: (scores[action], -action_order.index(action)))
    gain = scores[selected] - scores[U_ACTION_ID]
    return tuple(
        MethodDecision(
            target_center=target,
            case_id=case_id,
            method_id="S_y",
            action_id=selected,
            geometry_id=geometry_id,
            predicted_gain=gain,
            decision_source="support_only_static_pooled_exact_bacc",
            evaluation_labels_used=False,
        )
        for case_id in evaluation_cases
    )


__all__ = (
    "MethodDecision",
    "build_pre_support_decisions",
    "build_support_static_decisions",
)
