"""Composition of replay-adapter decisions against sealed action geometry."""

from __future__ import annotations

import numpy as np

from ..action_geometry import canonical_probabilities, probability_hash
from ..composition import ComposedAction
from ..engine import CaseRouteRequest
from ..protocol import ProtocolError
from .methods import ReplayActionScore, ReplayDecision


def compose_replay_decision(
    request: CaseRouteRequest,
    scores: tuple[ReplayActionScore, ...],
    decision: ReplayDecision,
    *,
    mode: str,
) -> ComposedAction:
    """Apply a generic replay decision with byte-exact P off its masks."""

    if mode not in {"boundary", "full_endpoint"}:
        raise ProtocolError("SCALE-BP replay composition mode drifted.")
    baseline = canonical_probabilities(request.portfolio_probabilities)
    if (
        decision.case_id != request.case_id
        or decision.baseline_probability_hash != probability_hash(baseline)
        or tuple(sorted(row.score_hash for row in scores)) != decision.score_hashes
    ):
        raise ProtocolError("SCALE-BP replay composition lineage drifted.")
    by_input = {row.action_id: row for row in request.action_inputs}
    by_score = {row.action_id: row for row in scores}
    if len(by_input) != len(request.action_inputs) or len(by_score) != len(scores):
        raise ProtocolError("SCALE-BP replay action identity is duplicated.")
    composed = np.array(baseline, dtype=np.float32, copy=True, order="C")
    used: set[int] = set()
    for action_id in decision.selected_action_ids:
        action_input = by_input.get(action_id)
        score = by_score.get(action_id)
        if action_input is None or score is None or not score.opportunity:
            raise ProtocolError("SCALE-BP replay selected an unresolved action.")
        projection = action_input.endpoint_projection.projection
        crossing = set(projection.crossing_indices)
        if crossing != set(score.crossing_indices) or used.intersection(crossing):
            raise ProtocolError("SCALE-BP replay selected overlapping action masks.")
        values = (
            projection.projected_probabilities
            if mode == "boundary"
            else projection.full_endpoint_probabilities
        )
        indices = tuple(sorted(crossing))
        composed[list(indices)] = np.asarray(values, dtype=np.float32)[list(indices)]
        used.update(crossing)
    return ComposedAction(
        case_id=request.case_id,
        mode=mode,
        baseline_probabilities=tuple(float(value) for value in baseline),
        composed_probabilities=tuple(float(value) for value in composed),
        selected_action_ids=decision.selected_action_ids,
        crossing_indices=tuple(sorted(used)),
        decision_hash=decision.decision_hash,
    )


__all__ = ("compose_replay_decision",)
