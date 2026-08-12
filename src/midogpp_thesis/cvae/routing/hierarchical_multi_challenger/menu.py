"""Fixed-B support ranking and deterministic top-three candidate menus."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from ...protocol import ProtocolError
from .contracts import CandidateMenu, SupportActionScore


TOP_K = 3
SUPPORT_PRIOR_CASES = 8.0


class ContributionTargetLike(Protocol):
    """Structural input needed for support-only pooled BACC ranking."""

    case_id: str
    action_id: str
    delta_tp: int
    delta_tn: int
    n_positive: int
    n_negative: int


def build_candidate_menu(
    action_targets: Mapping[str, Sequence[ContributionTargetLike]],
    *,
    top_k: int = TOP_K,
    support_prior_cases: float = SUPPORT_PRIOR_CASES,
) -> CandidateMenu:
    """Rank all eight candidates against fixed B, then freeze B plus top three."""

    if int(top_k) != TOP_K or float(support_prior_cases) != SUPPORT_PRIOR_CASES:
        raise ProtocolError("Multi-challenger menu hyperparameters are frozen.")
    scores: list[SupportActionScore] = []
    case_ids: tuple[str, ...] | None = None
    for action_id, supplied in action_targets.items():
        action = str(action_id)
        if action == "B":
            continue
        rows = tuple(supplied)
        identities = tuple(sorted(row.case_id for row in rows))
        if (
            not rows
            or any(row.action_id != action for row in rows)
            or len(set(identities)) != len(identities)
        ):
            raise ProtocolError("Support action target identity drifted.")
        if case_ids is None:
            case_ids = identities
        elif identities != case_ids:
            raise ProtocolError("Support candidates do not share the exact cases.")
        exact = _pooled_gain(rows)
        shrinkage = len(rows) / (len(rows) + SUPPORT_PRIOR_CASES)
        scores.append(
            SupportActionScore(action, exact, shrinkage * exact, len(rows))
        )
    if len(scores) < TOP_K:
        raise ProtocolError("Multi-challenger menu requires at least three actions.")
    ranked = tuple(
        sorted(scores, key=lambda row: (-row.shrunken_gain, row.action_id))
    )
    top = ranked[:TOP_K]
    anchor = top[0].action_id if top[0].shrunken_gain > 0.0 else "B"
    return CandidateMenu(
        action_ids=("B", *(row.action_id for row in top)),
        anchor_action_id=anchor,
        ranked_support_actions=ranked,
        top_k=TOP_K,
    )


def _pooled_gain(targets: Sequence[ContributionTargetLike]) -> float:
    n_positive = sum(int(row.n_positive) for row in targets)
    n_negative = sum(int(row.n_negative) for row in targets)
    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("Support ranking requires both classes across the pool.")
    return float(
        0.5 * sum(int(row.delta_tp) for row in targets) / n_positive
        + 0.5 * sum(int(row.delta_tn) for row in targets) / n_negative
    )


__all__ = ("SUPPORT_PRIOR_CASES", "TOP_K", "build_candidate_menu")
