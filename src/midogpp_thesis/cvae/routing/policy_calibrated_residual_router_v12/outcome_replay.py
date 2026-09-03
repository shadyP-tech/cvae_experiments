"""Shared exact-B and selected-action replay semantics for HARP v12."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ...protocol import ProtocolError
from .contracts import CasePrediction
from .outcome_inventory import CaseKey, CaseOutcomeInventory


@dataclass(frozen=True, slots=True)
class PolicyDecisionOutcome:
    """Observed source-only outcome of one frozen policy decision."""

    selected_action_id: str
    bacc_gain: float
    brier_delta: float
    log_delta: float
    routed: bool

    def __post_init__(self) -> None:
        if (
            not self.selected_action_id
            or any(
                not math.isfinite(value)
                for value in (self.bacc_gain, self.brier_delta, self.log_delta)
            )
            or (not self.routed and self.selected_action_id != "B")
            or (
                not self.routed
                and (self.bacc_gain, self.brier_delta, self.log_delta)
                != (0.0, 0.0, 0.0)
            )
        ):
            raise ProtocolError("HARP v12 policy replay outcome is malformed.")

    def numeric(self) -> tuple[float, float, float, bool]:
        return (self.bacc_gain, self.brier_delta, self.log_delta, self.routed)


def selected_policy_action_id(
    prediction: CasePrediction,
    *,
    acceptance_threshold: float,
    rank_margin_threshold: float = 0.0,
    policy_enabled: bool = True,
) -> str:
    if not isinstance(prediction, CasePrediction):
        raise ProtocolError("HARP v12 policy replay requires a typed prediction.")
    if (
        not policy_enabled
        or prediction.top_action_id == "B"
        or prediction.acceptance_probability < float(acceptance_threshold)
        or prediction.rank_margin < float(rank_margin_threshold)
    ):
        return "B"
    return prediction.top_action_id


def replay_policy_decisions(
    inventory: CaseOutcomeInventory,
    *,
    acceptance_threshold: float,
    rank_margin_threshold: float,
    policy_enabled: bool = True,
) -> dict[CaseKey, PolicyDecisionOutcome]:
    """Replay one policy over all cases, including exact-B controls."""

    if not isinstance(inventory, CaseOutcomeInventory):
        raise ProtocolError("HARP v12 policy replay requires a case-outcome inventory.")
    output: dict[CaseKey, PolicyDecisionOutcome] = {}
    for context in inventory.contexts:
        selected = selected_policy_action_id(
            context.prediction,
            acceptance_threshold=acceptance_threshold,
            rank_margin_threshold=rank_margin_threshold,
            policy_enabled=policy_enabled,
        )
        if context.is_exact_b_control and selected != "B":
            raise ProtocolError("HARP v12 exact-B control was routed.")
        if selected == "B":
            outcome = PolicyDecisionOutcome("B", 0.0, 0.0, 0.0, False)
        else:
            observed = context.outcome_for(selected)
            if observed is None:
                raise ProtocolError("HARP v12 selected active action lacks an outcome.")
            outcome = PolicyDecisionOutcome(
                selected,
                observed.bacc_gain,
                observed.brier_delta,
                observed.log_delta,
                True,
            )
        if context.key in output:
            raise ProtocolError("HARP v12 policy replay contains duplicate cases.")
        output[context.key] = outcome
    return output


__all__ = (
    "PolicyDecisionOutcome",
    "replay_policy_decisions",
    "selected_policy_action_id",
)
