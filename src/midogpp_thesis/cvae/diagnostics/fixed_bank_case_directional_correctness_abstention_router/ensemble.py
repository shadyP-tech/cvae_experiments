"""Probability-level B/A1 composition for sealed pre-terminal decisions."""

from __future__ import annotations

from collections.abc import Sequence

from ...protocol import ProtocolError
from .constants import B_ACTION_ID, U_ACTION_ID, a1_action_id
from .probability_surfaces import (
    ExactNineProbabilityRow,
    ExactNineProbabilitySurface,
    ProbabilityIndex,
    hard_prediction,
)
from .products import CaseAbstentionDecision, MethodPrediction


def compose_case_predictions(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow],
    decision: CaseAbstentionDecision,
    *,
    method_id: str | None = None,
) -> tuple[MethodPrediction, ...]:
    method = decision.method_id if method_id is None else str(method_id)
    if method != decision.method_id:
        raise ProtocolError("Abstention-router composition method drifted.")
    index = ProbabilityIndex(surface_or_rows)
    baseline_rows = index.rows_for_case_action(
        decision.target_center, decision.case_id, B_ACTION_ID
    )
    if not baseline_rows:
        raise ProtocolError("Abstention-router composition lacks held-case B rows.")
    output: list[MethodPrediction] = []
    for baseline in baseline_rows:
        baseline_hard = baseline.hard_prediction
        branch = decision.zero_to_one if baseline_hard == 0 else decision.one_to_zero
        selected = branch.selected_source
        probability = baseline.probability_mean
        if selected is not None:
            try:
                probability = index[
                    (
                        decision.target_center,
                        decision.case_id,
                        baseline.sample_id,
                        a1_action_id(selected),
                    )
                ].probability_mean
            except KeyError as exc:
                raise ProtocolError(
                    "Abstention-router selected A1 probability is unavailable."
                ) from exc
        output.append(
            MethodPrediction(
                decision.target_center,
                decision.case_id,
                baseline.sample_id,
                method,
                probability,
                hard_prediction(probability),
                baseline_hard,
                selected,
            )
        )
    return tuple(output)


def compose_method_predictions(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow],
    decisions: Sequence[CaseAbstentionDecision],
    *,
    method_id: str | None = None,
) -> tuple[MethodPrediction, ...]:
    rows = tuple(decisions)
    if not rows or len({row.key for row in rows}) != len(rows):
        raise ProtocolError("Abstention-router case decisions are empty or duplicated.")
    method = rows[0].method_id if method_id is None else str(method_id)
    if any(row.method_id != method for row in rows):
        raise ProtocolError("Abstention-router method decision collection drifted.")
    output = tuple(
        prediction
        for decision in sorted(rows, key=lambda row: (row.target_center, row.case_id))
        for prediction in compose_case_predictions(
            surface_or_rows, decision, method_id=method
        )
    )
    if len({row.key for row in output}) != len(output):
        raise ProtocolError("Abstention-router composed predictions duplicated.")
    return output


def compose_fixed_action_predictions(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow],
    *,
    method_id: str,
) -> tuple[MethodPrediction, ...]:
    """Expose the frozen B and U controls through the same prediction DTO."""

    if method_id not in {B_ACTION_ID, U_ACTION_ID}:
        raise ProtocolError("Abstention-router fixed control must be B or U.")
    index = ProbabilityIndex(surface_or_rows)
    selected = sorted(
        (row for row in index.values() if row.action_id == method_id),
        key=lambda row: (row.target_center, row.case_id, row.sample_id),
    )
    if not selected:
        raise ProtocolError("Abstention-router fixed control surface is empty.")
    output: list[MethodPrediction] = []
    for row in selected:
        baseline = index[
            (row.target_center, row.case_id, row.sample_id, B_ACTION_ID)
        ]
        output.append(
            MethodPrediction(
                row.target_center,
                row.case_id,
                row.sample_id,
                method_id,
                row.probability_mean,
                row.hard_prediction,
                baseline.hard_prediction,
                None,
            )
        )
    return tuple(output)


__all__ = (
    "compose_case_predictions",
    "compose_fixed_action_predictions",
    "compose_method_predictions",
)
