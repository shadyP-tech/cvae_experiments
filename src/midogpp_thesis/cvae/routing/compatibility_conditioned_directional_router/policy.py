"""Baseline-anchored, direction-specific fail-closed selection."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import (
    ActionKind,
    BoundedActionEvidence,
    Direction,
    LearnabilityAdmission,
    RoutingDecision,
)


DEFAULT_OPPORTUNITY_THRESHOLD = 0.5
DEFAULT_TOP_K = 2
DEFAULT_MIXTURE_LAMBDA = 0.5
DEFAULT_TEMPERATURE = 0.25


def _fallback(
    *, outer_target_id: str, case_id: str, admission_hash: str, reason: str
) -> RoutingDecision:
    return RoutingDecision(
        outer_target_id=outer_target_id,
        case_id=case_id,
        enabled=False,
        selected_direction=None,
        selected_action_ids=(),
        selected_weights=(),
        mixture_lambda=0.0,
        reason=reason,
        admission_hash=admission_hash,
        evidence_hashes=(),
    )


def _softmax_weights(scores: Sequence[float], temperature: float) -> tuple[float, ...]:
    maximum = max(scores)
    exponentials = tuple(math.exp((value - maximum) / temperature) for value in scores)
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def select_baseline_anchored_route(
    evidence: Sequence[BoundedActionEvidence],
    *,
    admission: LearnabilityAdmission,
    outer_target_id: str,
    case_id: str,
    top_k: int = DEFAULT_TOP_K,
    mixture_lambda: float = DEFAULT_MIXTURE_LAMBDA,
    opportunity_threshold: float = DEFAULT_OPPORTUNITY_THRESHOLD,
    temperature: float = DEFAULT_TEMPERATURE,
) -> RoutingDecision:
    """Select safe actions without an unconditional HXE-vs-U veto.

    Every challenger first passes its own exact safe-vs-B bound.  A failed U is
    simply absent from the admitted set and cannot veto a safe HXE.  Eligible
    HXE candidates are grouped by direction and the best direction contributes
    at most ``top_k`` experts to a directional soft composition. Exact B is
    retained on the opposite branch and for every disabled decision.
    """

    if not isinstance(admission, LearnabilityAdmission):
        raise ProtocolError("Routing policy requires a typed source-only admission.")
    if (
        type(top_k) is not int
        or top_k < 1
        or not 0.0 < float(mixture_lambda) <= 1.0
        or not 0.0 <= float(opportunity_threshold) <= 1.0
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ProtocolError("Routing policy hyperparameters are invalid.")
    if not admission.passed:
        return _fallback(
            outer_target_id=outer_target_id,
            case_id=case_id,
            admission_hash=admission.admission_hash,
            reason="SOURCE_ONLY_LEARNABILITY_ADMISSION_FAILED",
        )
    rows = tuple(sorted(tuple(evidence), key=lambda row: row.prediction.feature.action_id))
    if any(not isinstance(row, BoundedActionEvidence) for row in rows):
        raise ProtocolError("Routing evidence must be typed and uncertainty-bound.")
    if rows:
        feature_keys = {
            (row.prediction.feature.outer_target_id, row.prediction.feature.case_id)
            for row in rows
        }
        if feature_keys != {(str(outer_target_id), str(case_id))} or len(
            {row.prediction.feature.action_id for row in rows}
        ) != len(rows):
            raise ProtocolError("Routing evidence crossed cases or duplicated actions.")
    eligible = tuple(
        row
        for row in rows
        if row.safe_vs_baseline
        and row.prediction.opportunity_probability >= float(opportunity_threshold)
    )
    if not eligible:
        return _fallback(
            outer_target_id=outer_target_id,
            case_id=case_id,
            admission_hash=admission.admission_hash,
            reason="NO_ACTION_SAFE_VS_B",
        )

    # U is a standalone physical menu option.  Expert portfolios never blend U
    # with HXE; both are instead compared by the frozen pairwise ranking score.
    proposals: list[tuple[float, str, Direction, tuple[BoundedActionEvidence, ...]]] = []
    for row in eligible:
        if row.prediction.feature.action_kind is ActionKind.U:
            proposals.append(
                (
                    row.prediction.ranking_score,
                    f"U::{row.prediction.feature.action_id}",
                    row.prediction.feature.direction,
                    (row,),
                )
            )
    expert_groups: dict[Direction, list[BoundedActionEvidence]] = defaultdict(list)
    for row in eligible:
        if row.prediction.feature.action_kind is ActionKind.HXE:
            expert_groups[row.prediction.feature.direction].append(row)
    for direction, group in sorted(expert_groups.items(), key=lambda item: item[0].value):
        ordered = tuple(
            sorted(
                group,
                key=lambda row: (
                    -row.prediction.ranking_score,
                    row.prediction.feature.action_id,
                ),
            )[:top_k]
        )
        weights = _softmax_weights(
            tuple(row.prediction.ranking_score for row in ordered), float(temperature)
        )
        score = sum(
            weight * row.prediction.ranking_score
            for weight, row in zip(weights, ordered, strict=True)
        )
        proposals.append((score, f"HXE::{direction.value}", direction, ordered))
    if not proposals:
        return _fallback(
            outer_target_id=outer_target_id,
            case_id=case_id,
            admission_hash=admission.admission_hash,
            reason="NO_DIRECTIONAL_ACTION_AVAILABLE",
        )
    _, proposal_id, direction, selected = min(
        proposals, key=lambda row: (-row[0], row[1])
    )
    weights = _softmax_weights(
        tuple(row.prediction.ranking_score for row in selected), float(temperature)
    )
    selected_actions = tuple(row.prediction.feature.action_id for row in selected)
    reason = (
        "UNIFORM_SAFE_VS_B"
        if selected[0].prediction.feature.action_kind is ActionKind.U
        else "DIRECTIONAL_TOPK_HXE_SAFE_VS_B"
    )
    return RoutingDecision(
        outer_target_id=str(outer_target_id),
        case_id=str(case_id),
        enabled=True,
        selected_direction=direction,
        selected_action_ids=selected_actions,
        selected_weights=weights,
        mixture_lambda=float(mixture_lambda),
        reason=f"{reason}:{proposal_id}",
        admission_hash=admission.admission_hash,
        evidence_hashes=tuple(row.evidence_hash for row in selected),
    )


__all__ = (
    "DEFAULT_MIXTURE_LAMBDA",
    "DEFAULT_OPPORTUNITY_THRESHOLD",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_K",
    "select_baseline_anchored_route",
)
