"""Label-free held-case shift inflation and endpoint utility certificates."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ...protocol import ProtocolError
from .cell_residuals import posterior_point
from .constants import (
    MINIMAX_CONTROL_METHOD_ID,
    COMPOSED_POLICY_IDS,
    ZERO_SHIFT_CONTROL_METHOD_ID,
    RESIDUAL_SCALE_FLOOR,
    SHIFT_DESCRIPTOR_WEIGHT,
    SHIFT_DESCRIPTOR_Z_CAP,
    SHIFT_EFFECTIVE_COUNT_WEIGHT,
    SHIFT_FOLD_WEIGHT,
    SHIFT_FOLD_Z_CAP,
    SHIFT_KAPPA_CAP,
)
from .utility_contracts import (
    DirectionEnvelope,
    DonorUtilityRow,
    FeatureReference,
    PosteriorUtilityPrediction,
    ResidualScale,
    UtilityCertificate,
    UtilityDescriptor,
)


class EnvelopeLike(Protocol):
    control_id: str

    def scale_for(
        self, alternative: str, direction: str, response_id: str
    ) -> ResidualScale: ...

    def reference_for(
        self, alternative: str, direction: str
    ) -> FeatureReference: ...

    def envelope_for(self, direction: str) -> DirectionEnvelope: ...


def certify_utility(
    descriptor: UtilityDescriptor | DonorUtilityRow,
    prediction: PosteriorUtilityPrediction,
    calibration: EnvelopeLike,
    *,
    policy_id: str,
    calibration_hash: str,
) -> UtilityCertificate:
    """Construct one-sided BACC/loss bounds without held-case labels."""

    target_center = (
        descriptor.target_center
        if isinstance(descriptor, UtilityDescriptor)
        else descriptor.donor_center
    )
    descriptor_key = (
        target_center,
        descriptor.case_id,
        descriptor.alternative,
        descriptor.direction,
    )
    if (
        policy_id not in COMPOSED_POLICY_IDS
        or descriptor_key != prediction.key
        or descriptor.descriptor_hash != prediction.descriptor_hash
        or prediction.control_id != calibration.control_id
    ):
        raise ProtocolError("PSSCUR utility certificate binding drifted.")
    reference = calibration.reference_for(
        descriptor.alternative, descriptor.direction
    )
    descriptor_shift = _descriptor_shift(descriptor, reference)
    endpoint_scales = {
        response_id: calibration.scale_for(
            descriptor.alternative, descriptor.direction, response_id
        ).shrunk_scale
        for response_id in (
            "bacc_contribution_delta",
            "brier_contribution_delta",
            "log_loss_contribution_delta",
        )
    }
    fold_instability = _fold_instability(prediction, endpoint_scales)
    if policy_id == ZERO_SHIFT_CONTROL_METHOD_ID:
        kappa = 1.0
    else:
        kappa = min(
            SHIFT_KAPPA_CAP,
            1.0
            + SHIFT_DESCRIPTOR_WEIGHT * descriptor_shift
            + SHIFT_FOLD_WEIGHT * fold_instability
            + SHIFT_EFFECTIVE_COUNT_WEIGHT
            / np.sqrt(float(max(1, descriptor.crossing_count))),
        )
    envelope = calibration.envelope_for(descriptor.direction)
    radius = (
        envelope.maximum_radius
        if policy_id == MINIMAX_CONTROL_METHOD_ID
        else envelope.radius
    )
    point_bacc = posterior_point(prediction, "bacc_contribution_delta")
    point_brier = posterior_point(prediction, "brier_contribution_delta")
    point_log = posterior_point(prediction, "log_loss_contribution_delta")
    return UtilityCertificate(
        target_center,
        descriptor.case_id,
        descriptor.alternative,
        descriptor.direction,
        prediction.control_id,
        policy_id,
        descriptor.crossing_count,
        point_bacc,
        point_brier,
        point_log,
        descriptor_shift,
        fold_instability,
        float(kappa),
        radius,
        point_bacc
        - kappa
        * radius
        * endpoint_scales["bacc_contribution_delta"],
        point_brier
        + kappa
        * radius
        * endpoint_scales["brier_contribution_delta"],
        point_log
        + kappa
        * radius
        * endpoint_scales["log_loss_contribution_delta"],
        prediction.reliability_pass,
        descriptor.descriptor_hash,
        prediction.utility_hash,
        calibration_hash,
    )


def _descriptor_shift(
    descriptor: UtilityDescriptor | DonorUtilityRow, reference: FeatureReference
) -> float:
    values = np.asarray(descriptor.feature_values, dtype=np.float64)
    locations = np.asarray(reference.locations, dtype=np.float64)
    scales = np.maximum(
        np.asarray(reference.scales, dtype=np.float64), RESIDUAL_SCALE_FLOOR
    )
    z = np.minimum(np.abs((values - locations) / scales), SHIFT_DESCRIPTOR_Z_CAP)
    return float(np.sqrt(np.mean(z * z, dtype=np.float64)))


def _fold_instability(
    prediction: PosteriorUtilityPrediction,
    scales: dict[str, float],
) -> float:
    rows = (
        ("bacc_contribution_delta", prediction.fold_bacc_deltas),
        ("brier_contribution_delta", prediction.fold_brier_deltas),
        ("log_loss_contribution_delta", prediction.fold_log_loss_deltas),
    )
    values = [
        min(
            SHIFT_FOLD_Z_CAP,
            float(np.std(np.asarray(folds, dtype=np.float64), ddof=0))
            / max(scales[response_id], RESIDUAL_SCALE_FLOOR),
        )
        for response_id, folds in rows
    ]
    return max(values)


__all__ = ("EnvelopeLike", "certify_utility")
