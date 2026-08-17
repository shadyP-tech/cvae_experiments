"""Fail-closed model-based route selection with protected-P fallback."""

from __future__ import annotations

from typing import Mapping

from ...protocol import ProtocolError
from .constants import (
    BACC_REGRET_TOLERANCE,
    MIN_DELETE_DONOR_POSITIVE,
    MODEL_BASED_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    PROPER_LOSS_TOLERANCE,
    SUPPORT_DISPERSION_MULTIPLIER,
)
from .contracts import (
    CandidateDescriptor,
    CenterBalancedRidgeModel,
    EndpointCasePrediction,
    RouteDecision,
)
from .donor_regret_model import predict_unseen_center


def select_model_based_route(
    descriptor: CandidateDescriptor,
    *,
    full_models: Mapping[str, CenterBalancedRidgeModel],
    delete_donor_models: Mapping[str, Mapping[str, CenterBalancedRidgeModel]],
    support_dispersion_multiplier: float = SUPPORT_DISPERSION_MULTIPLIER,
    minimum_delete_donor_positive: int = MIN_DELETE_DONOR_POSITIVE,
    proper_loss_tolerance: float = PROPER_LOSS_TOLERANCE,
    policy_id: str = MODEL_BASED_METHOD_ID,
    require_support_margin: bool = True,
    require_proper_loss: bool = True,
) -> RouteDecision:
    if not policy_id or type(require_support_margin) is not bool or type(require_proper_loss) is not bool:
        raise ProtocolError("Route policy identity or gate toggles drifted.")
    if tuple(full_models) != ("bacc_regret", "log_loss_delta"):
        raise ProtocolError("Route selection requires the full dual-response model surface.")
    training_centers = full_models["bacc_regret"].training_centers
    if (
        full_models["log_loss_delta"].training_centers != training_centers
        or tuple(delete_donor_models) != training_centers
        or not 0 <= minimum_delete_donor_positive <= len(training_centers)
        or any(
            tuple(models) != ("bacc_regret", "log_loss_delta")
            or models["bacc_regret"].training_centers
            != tuple(center for center in training_centers if center != deleted)
            or models["log_loss_delta"].training_centers
            != tuple(center for center in training_centers if center != deleted)
            for deleted, models in delete_donor_models.items()
        )
    ):
        raise ProtocolError("Route selection donor-deletion topology drifted.")
    model_hashes = tuple(
        [full_models[response].model_hash for response in full_models]
        + [
            delete_donor_models[donor][response].model_hash
            for donor in delete_donor_models
            for response in ("bacc_regret", "log_loss_delta")
        ]
    )
    if not descriptor.is_candidate:
        return RouteDecision(
            descriptor.target_center,
            descriptor.case_id,
            policy_id,
            descriptor.alternative,
            PORTFOLIO_METHOD_ID,
            0.0,
            0.0,
            0,
            0,
            descriptor.values[0],
            descriptor.values[2],
            "fallback_P_no_nominated_crossing_alternative",
            descriptor.descriptor_hash,
            model_hashes,
        )
    predicted_bacc = predict_unseen_center(
        full_models["bacc_regret"], descriptor.values
    )
    predicted_loss = predict_unseen_center(
        full_models["log_loss_delta"], descriptor.values
    )
    delete_bacc = tuple(
        predict_unseen_center(
            models["bacc_regret"], descriptor.values
        )
        for models in delete_donor_models.values()
    )
    delete_loss = tuple(
        predict_unseen_center(
            models["log_loss_delta"], descriptor.values
        )
        for models in delete_donor_models.values()
    )
    bacc_positive = sum(value > BACC_REGRET_TOLERANCE for value in delete_bacc)
    loss_safe = sum(value <= proper_loss_tolerance for value in delete_loss)
    support_safe = (not require_support_margin) or descriptor.values[0] > (
        support_dispersion_multiplier * descriptor.values[2]
    )
    conditions = (
        predicted_bacc > BACC_REGRET_TOLERANCE,
        bacc_positive >= minimum_delete_donor_positive,
        support_safe,
        (not require_proper_loss) or predicted_loss <= proper_loss_tolerance,
        (not require_proper_loss) or loss_safe >= minimum_delete_donor_positive,
    )
    if all(conditions):
        selected = descriptor.alternative
        reason = (
            "authorized_center_balanced_dual_response_delete_donor_consensus"
            if require_proper_loss
            else "authorized_center_balanced_bacc_only_delete_donor_consensus"
        )
    else:
        selected = PORTFOLIO_METHOD_ID
        failed = (
            "nonpositive_predicted_bacc"
            if not conditions[0]
            else "insufficient_delete_donor_bacc_consensus"
            if not conditions[1]
            else "insufficient_nested_support_margin"
            if not conditions[2]
            else "predicted_log_loss_point_estimate_worse_than_P"
            if not conditions[3]
            else "insufficient_delete_donor_log_loss_consensus"
        )
        reason = f"fallback_P_{failed}"
    return RouteDecision(
        descriptor.target_center,
        descriptor.case_id,
        policy_id,
        descriptor.alternative,
        selected,
        predicted_bacc,
        predicted_loss,
        bacc_positive,
        loss_safe,
        descriptor.values[0],
        descriptor.values[2],
        reason,
        descriptor.descriptor_hash,
        model_hashes,
    )


def selected_probabilities(
    decision: RouteDecision, endpoint_prediction: EndpointCasePrediction
) -> tuple[float, ...]:
    if (
        decision.target_center != endpoint_prediction.center
        or decision.case_id != endpoint_prediction.case_id
    ):
        raise ProtocolError("Route decision and endpoint prediction do not align.")
    return endpoint_prediction.probabilities[decision.selected_method]


__all__ = ("select_model_based_route", "selected_probabilities")
