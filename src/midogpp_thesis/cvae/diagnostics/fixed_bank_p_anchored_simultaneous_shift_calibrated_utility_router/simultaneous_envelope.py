"""Selection-aware direction-wise donor residual envelopes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .cell_residuals import (
    ResidualObservation,
    build_residual_observations,
    fit_feature_references,
    fit_residual_scales,
    scale_index,
)
from .constants import CENTERS, DIRECTION_IDS, DONOR_ENVELOPE_QUANTILE
from .hashing import canonical_hash
from .utility_contracts import (
    DirectionEnvelope,
    DonorUtilityRow,
    FeatureReference,
    PosteriorUtilityPrediction,
    ResidualScale,
)


@dataclass(frozen=True)
class FittedEnvelopeModel:
    """Internal preterminal model; donor labels never enter target scoring."""

    control_id: str
    donor_centers: tuple[str, ...]
    residual_scales: tuple[ResidualScale, ...]
    feature_references: tuple[FeatureReference, ...]
    direction_envelopes: tuple[DirectionEnvelope, ...]
    source_utility_hash: str
    source_response_hash: str
    model_hash: str

    def scale_for(self, alternative: str, direction: str, response_id: str) -> ResidualScale:
        return next(
            row
            for row in self.residual_scales
            if row.key == (alternative, direction, response_id)
        )

    def reference_for(self, alternative: str, direction: str) -> FeatureReference:
        return next(
            row for row in self.feature_references if row.key == (alternative, direction)
        )

    def envelope_for(self, direction: str) -> DirectionEnvelope:
        return next(row for row in self.direction_envelopes if row.direction == direction)


def fit_simultaneous_envelope(
    predictions: Sequence[PosteriorUtilityPrediction],
    donor_rows: Sequence[DonorUtilityRow],
    *,
    allowed_donors: Sequence[str],
) -> FittedEnvelopeModel:
    """Fit scales first, then calibrate maxima over cells/endpoints per donor."""

    donors = tuple(center for center in CENTERS if center in set(allowed_donors))
    controls = {
        row.control_id for row in predictions if row.target_center in set(donors)
    }
    if (
        len(donors) < 6
        or len(donors) != len(set(allowed_donors))
        or len(controls) != 1
    ):
        raise ProtocolError("PSSCUR simultaneous envelope scope drifted.")
    observations = build_residual_observations(
        predictions, donor_rows, allowed_donors=donors
    )
    scales = fit_residual_scales(observations)
    references = fit_feature_references(observations)
    indexed = scale_index(scales)
    envelopes = tuple(
        _fit_direction_envelope(
            direction, observations, indexed, donor_centers=donors
        )
        for direction in DIRECTION_IDS
    )
    utility_hash = canonical_hash(
        [
            row.to_payload()
            for row in sorted(predictions)
            if row.target_center in set(donors)
        ]
    )
    response_hash = canonical_hash(
        [
            row.to_payload()
            for row in sorted(donor_rows, key=lambda value: value.key)
            if row.donor_center in set(donors)
        ]
    )
    payload = {
        "schema_version": "fixed_bank_psscur_fitted_envelope_model_v1",
        "control_id": next(iter(controls)),
        "donor_centers": list(donors),
        "residual_scale_hashes": [row.scale_hash for row in scales],
        "feature_reference_hashes": [row.reference_hash for row in references],
        "direction_envelope_hashes": [row.envelope_hash for row in envelopes],
        "source_utility_hash": utility_hash,
        "source_response_hash": response_hash,
        "maximum_taken_before_quantile": True,
        "finite_sample_coverage_claimed": False,
    }
    return FittedEnvelopeModel(
        next(iter(controls)),
        donors,
        scales,
        references,
        envelopes,
        utility_hash,
        response_hash,
        canonical_hash(payload),
    )


def _fit_direction_envelope(
    direction: str,
    observations: Sequence[ResidualObservation],
    scales: Mapping[tuple[str, str, str], ResidualScale],
    *,
    donor_centers: Sequence[str],
) -> DirectionEnvelope:
    index = dict(scales)
    block_scores: list[tuple[str, float]] = []
    for donor in donor_centers:
        candidates = [0.0]
        for row in observations:
            if row.donor_center != donor or row.direction != direction:
                continue
            for response_id in (
                "bacc_contribution_delta",
                "brier_contribution_delta",
                "log_loss_contribution_delta",
            ):
                scale = index[(row.alternative, row.direction, response_id)].shrunk_scale
                candidates.append(max(0.0, row.error(response_id) / scale))
        block_scores.append((donor, max(candidates)))
    values = sorted(value for _donor, value in block_scores)
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ProtocolError("PSSCUR donor block maximum drifted.")
    quantile_index = min(
        len(values) - 1,
        int(math.ceil(DONOR_ENVELOPE_QUANTILE * (len(values) - 1))),
    )
    return DirectionEnvelope(
        direction,
        DONOR_ENVELOPE_QUANTILE,
        values[quantile_index],
        values[-1],
        tuple(block_scores),
    )


__all__ = ("FittedEnvelopeModel", "fit_simultaneous_envelope")
