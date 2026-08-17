"""Predeclared policy menu and label-free route-selection controls."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

from ...protocol import ProtocolError
from .constants import (
    MIN_DELETE_DONOR_POSITIVE,
    MODEL_BASED_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    PROPER_LOSS_TOLERANCE,
    SENSITIVITY_DELETE_COUNTS,
    SENSITIVITY_DISPERSION_MULTIPLIERS,
    SUPPORT_DISPERSION_MULTIPLIER,
)
from .contracts import CandidateDescriptor, CenterBalancedRidgeModel, RouteDecision
from .hashing import canonical_hash
from .selection import select_model_based_route


@dataclass(frozen=True, order=True)
class RoutePolicySpec:
    policy_id: str
    support_dispersion_multiplier: float
    minimum_delete_donor_positive: int
    require_model: bool = True
    require_support_margin: bool = True
    require_proper_loss: bool = True
    policy_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if (
            not self.policy_id
            or not math.isfinite(float(self.support_dispersion_multiplier))
            or self.support_dispersion_multiplier < 0.0
            or isinstance(self.minimum_delete_donor_positive, bool)
            or not 0 <= self.minimum_delete_donor_positive <= 8
            or type(self.require_model) is not bool
            or type(self.require_support_margin) is not bool
            or type(self.require_proper_loss) is not bool
        ):
            raise ProtocolError("Route policy specification drifted.")
        object.__setattr__(self, "policy_hash", canonical_hash(self.to_payload(False)))

    def to_payload(self, include_hash: bool = True) -> dict[str, object]:
        payload = {
            "schema_version": "fixed_bank_nested_regret_route_policy_v1",
            "policy_id": self.policy_id,
            "support_dispersion_multiplier": self.support_dispersion_multiplier,
            "minimum_delete_donor_positive": self.minimum_delete_donor_positive,
            "require_model": self.require_model,
            "require_support_margin": self.require_support_margin,
            "require_proper_loss": self.require_proper_loss,
            "fallback": PORTFOLIO_METHOD_ID,
            "terminal_utility_used_to_define_policy": False,
        }
        return {**payload, "policy_hash": self.policy_hash} if include_hash else payload


PRIMARY_POLICY = RoutePolicySpec(
    MODEL_BASED_METHOD_ID,
    SUPPORT_DISPERSION_MULTIPLIER,
    MIN_DELETE_DONOR_POSITIVE,
)


def predeclared_policy_menu() -> tuple[RoutePolicySpec, ...]:
    rows = [
        PRIMARY_POLICY,
        RoutePolicySpec("CONTROL_P_ONLY", 0.0, 0, False, False, False),
        RoutePolicySpec(
            "CONTROL_SUPPORT_ONLY_DISPERSION_0_5",
            0.5,
            0,
            False,
            True,
            False,
        ),
        RoutePolicySpec("CONTROL_RIDGE_ONLY", 0.0, 0, True, False, False),
        RoutePolicySpec("CONTROL_PROPER_LOSS_OFF", 0.5, 7, True, True, False),
    ]
    rows.extend(
        RoutePolicySpec(f"SENSITIVITY_DISPERSION_{value:g}", value, 7)
        for value in SENSITIVITY_DISPERSION_MULTIPLIERS
        if value != SUPPORT_DISPERSION_MULTIPLIER
    )
    rows.extend(
        RoutePolicySpec(f"SENSITIVITY_DELETE_{count}_OF_8", 0.5, count)
        for count in SENSITIVITY_DELETE_COUNTS
        if count != MIN_DELETE_DONOR_POSITIVE
    )
    result = tuple(rows)
    if len({row.policy_id for row in result}) != len(result):
        raise ProtocolError("Predeclared policy menu contains duplicate identities.")
    return result


def select_route_for_policy(
    descriptor: CandidateDescriptor,
    policy: RoutePolicySpec,
    *,
    full_models: Mapping[str, CenterBalancedRidgeModel] | None,
    delete_donor_models: Mapping[
        str, Mapping[str, CenterBalancedRidgeModel]
    ] | None,
) -> RouteDecision:
    if policy.require_model:
        if full_models is None or delete_donor_models is None:
            raise ProtocolError("Model-based policy lacks its sealed model surface.")
        return select_model_based_route(
            descriptor,
            full_models=full_models,
            delete_donor_models=delete_donor_models,
            support_dispersion_multiplier=policy.support_dispersion_multiplier,
            minimum_delete_donor_positive=policy.minimum_delete_donor_positive,
            proper_loss_tolerance=PROPER_LOSS_TOLERANCE,
            policy_id=policy.policy_id,
            require_support_margin=policy.require_support_margin,
            require_proper_loss=policy.require_proper_loss,
        )
    selected = PORTFOLIO_METHOD_ID
    reason = "fallback_P_control_P_only"
    if policy.policy_id != "CONTROL_P_ONLY" and descriptor.is_candidate:
        support_pass = (not policy.require_support_margin) or descriptor.values[0] > (
            policy.support_dispersion_multiplier * descriptor.values[2]
        )
        if support_pass:
            selected = descriptor.alternative
            reason = "authorized_nested_support_only"
        else:
            reason = "fallback_P_insufficient_nested_support_margin"
    return RouteDecision(
        descriptor.target_center,
        descriptor.case_id,
        policy.policy_id,
        descriptor.alternative,
        selected,
        0.0,
        0.0,
        0,
        0,
        descriptor.values[0],
        descriptor.values[2],
        reason,
        descriptor.descriptor_hash,
        (),
    )


__all__ = (
    "PRIMARY_POLICY",
    "RoutePolicySpec",
    "predeclared_policy_menu",
    "select_route_for_policy",
)
