"""Descriptive center-block feasibility reference for a frozen policy grid."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .constants import (
    LTT_FAMILYWISE_ALPHA,
    LTT_MAX_CENTER_HARM_RATE,
    PROPER_LOSS_TOLERANCE,
)


@dataclass(frozen=True)
class CenterBlockPolicyEvidence:
    policy_id: str
    center_bacc_gains: tuple[float, ...]
    center_log_loss_deltas: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not self.policy_id
            or len(self.center_bacc_gains) != len(self.center_log_loss_deltas)
            or len(self.center_bacc_gains) < 2
            or any(
                not math.isfinite(float(value))
                for value in (*self.center_bacc_gains, *self.center_log_loss_deltas)
            )
        ):
            raise ProtocolError("Center-block policy evidence drifted.")


def _binomial_upper_tail(successes: int, trials: int, probability: float) -> float:
    return float(
        sum(
            math.comb(trials, count)
            * probability**count
            * (1.0 - probability) ** (trials - count)
            for count in range(successes, trials + 1)
        )
    )


def learn_then_test_center_harm(
    evidence: Sequence[CenterBlockPolicyEvidence],
    *,
    familywise_alpha: float = LTT_FAMILYWISE_ALPHA,
    max_center_harm_rate: float = LTT_MAX_CENTER_HARM_RATE,
    proper_loss_tolerance: float = PROPER_LOSS_TOLERANCE,
) -> dict[str, object]:
    """Compute an optimistic independent-binomial feasibility reference.

    The eight center scores share cross-fitted training centers, so they are
    not independent Bernoulli trials.  The tail calculation is therefore a
    descriptive power reference only and is never allowed to authorize a
    route.  It does not create a fresh or target-center-conditional guarantee.
    """

    rows = tuple(evidence)
    if (
        not rows
        or len({row.policy_id for row in rows}) != len(rows)
        or not 0.0 < familywise_alpha < 1.0
        or not 0.0 < max_center_harm_rate < 1.0
    ):
        raise ProtocolError("Center-block feasibility family or reference target drifted.")
    null_safe_probability = 1.0 - max_center_harm_rate
    tests: list[dict[str, object]] = []
    for row in rows:
        safe = sum(
            gain > 0.0 and loss <= proper_loss_tolerance
            for gain, loss in zip(
                row.center_bacc_gains,
                row.center_log_loss_deltas,
                strict=True,
            )
        )
        trials = len(row.center_bacc_gains)
        p_value = _binomial_upper_tail(safe, trials, null_safe_probability)
        tests.append(
            {
                "policy_id": row.policy_id,
                "center_count": trials,
                "safe_center_count": safe,
                "optimistic_independent_binomial_tail_probability": p_value,
                "mean_bacc_gain": sum(row.center_bacc_gains) / trials,
                "worst_center_bacc_gain": min(row.center_bacc_gains),
                "mean_log_loss_delta": sum(row.center_log_loss_deltas) / trials,
            }
        )
    ordered = sorted(
        tests,
        key=lambda row: (
            float(row["optimistic_independent_binomial_tail_probability"]),
            str(row["policy_id"]),
        ),
    )
    still_rejecting = True
    for rank, row in enumerate(ordered, start=1):
        threshold = familywise_alpha / (len(ordered) - rank + 1)
        below_reference = bool(
            still_rejecting
            and float(row["optimistic_independent_binomial_tail_probability"])
            <= threshold
        )
        still_rejecting = still_rejecting and below_reference
        row["holm_reference_threshold"] = threshold
        row["reference_tail_below_threshold"] = below_reference
    by_policy = {str(row["policy_id"]): row for row in ordered}
    return {
        "schema_version": "fixed_bank_nested_regret_center_block_feasibility_v1",
        "familywise_alpha": familywise_alpha,
        "max_center_harm_rate": max_center_harm_rate,
        "proper_loss_tolerance": proper_loss_tolerance,
        "tests": [by_policy[row.policy_id] for row in rows],
        "authorized_policy_ids": [],
        "statistical_authorization_enabled": False,
        "fallback_when_none_authorized": "P_PROTECTED",
        "effective_units_are_centers_not_cases": True,
        "dependent_cross_fitted_center_blocks": True,
        "binomial_independence_assumption_claimed": False,
        "tail_probabilities_are_descriptive_power_references_only": True,
        "fresh_evidence": False,
        "target_center_conditional_guarantee_claimed": False,
        "formal_deployment_guarantee_claimed": False,
    }


__all__ = ("CenterBlockPolicyEvidence", "learn_then_test_center_harm")
