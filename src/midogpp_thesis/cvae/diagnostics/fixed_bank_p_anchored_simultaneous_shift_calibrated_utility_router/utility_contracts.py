"""Typed contracts for complete signed-utility routing surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ...protocol import ProtocolError
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    BOUND_STRICT_TOLERANCE,
    CENTERS,
    COMPOSED_POLICY_IDS,
    DIRECTION_IDS,
    DONOR_ENVELOPE_QUANTILE,
    PORTFOLIO_METHOD_ID,
    RESIDUAL_SCALE_FLOOR,
    SIGN_PRESERVING_SHRINKAGE,
    SHIFT_KAPPA_CAP,
    SUPPORT_CROSSFIT_FOLD_COUNT,
    UTILITY_FEATURE_NAMES,
    UTILITY_RESPONSE_IDS,
)
from .hashing import canonical_hash, require_sha256


def _finite(values: tuple[float, ...], *, size: int, name: str) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if len(converted) != size or any(not math.isfinite(value) for value in converted):
        raise ProtocolError(f"PSSCUR {name} drifted.")
    return converted


@dataclass(frozen=True, order=True)
class UtilityDescriptor:
    """One label-free case x alternative x direction candidate."""

    target_center: str
    case_id: str
    alternative: str
    direction: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    crossing_sample_ids: tuple[str, ...]
    endpoint_prediction_hash: str
    descriptor_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        values = _finite(
            self.feature_values,
            size=len(UTILITY_FEATURE_NAMES),
            name="utility descriptor",
        )
        sample_ids = tuple(str(value) for value in self.crossing_sample_ids)
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or self.alternative not in ALTERNATIVE_METHOD_IDS
            or self.direction not in DIRECTION_IDS
            or self.feature_names != UTILITY_FEATURE_NAMES
            or len(sample_ids) != len(set(sample_ids))
        ):
            raise ProtocolError("PSSCUR utility descriptor identity drifted.")
        require_sha256(self.endpoint_prediction_hash, "endpoint_prediction_hash")
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "crossing_sample_ids", sample_ids)
        object.__setattr__(self, "descriptor_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.target_center, self.case_id, self.alternative, self.direction

    @property
    def crossing_count(self) -> int:
        return len(self.crossing_sample_ids)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_utility_descriptor_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "alternative": self.alternative,
            "direction": self.direction,
            "feature_names": list(self.feature_names),
            "feature_values": list(self.feature_values),
            "crossing_sample_ids": list(self.crossing_sample_ids),
            "crossing_count": self.crossing_count,
            "structural_zero": self.crossing_count == 0,
            "endpoint_prediction_hash": self.endpoint_prediction_hash,
            "labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "descriptor_hash": self.descriptor_hash}


@dataclass(frozen=True, order=True)
class DonorUtilityRow:
    """Observed signed contribution of one complete donor candidate."""

    outer_target_center: str
    donor_center: str
    case_id: str
    alternative: str
    direction: str
    feature_values: tuple[float, ...]
    crossing_count: int
    bacc_contribution_delta: float
    brier_contribution_delta: float
    log_loss_contribution_delta: float
    descriptor_hash: str

    def __post_init__(self) -> None:
        values = _finite(
            self.feature_values,
            size=len(UTILITY_FEATURE_NAMES),
            name="donor utility row",
        )
        responses = (
            float(self.bacc_contribution_delta),
            float(self.brier_contribution_delta),
            float(self.log_loss_contribution_delta),
        )
        if (
            self.outer_target_center not in CENTERS
            or self.donor_center not in CENTERS
            or self.outer_target_center == self.donor_center
            or not self.case_id
            or self.alternative not in ALTERNATIVE_METHOD_IDS
            or self.direction not in DIRECTION_IDS
            or type(self.crossing_count) is not int
            or self.crossing_count < 0
            or any(not math.isfinite(value) for value in responses)
            or (
                self.crossing_count == 0
                and any(abs(value) > 1.0e-15 for value in responses)
            )
        ):
            raise ProtocolError("PSSCUR donor utility response drifted.")
        require_sha256(self.descriptor_hash, "utility_descriptor_hash")
        object.__setattr__(self, "feature_values", values)

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.outer_target_center,
            self.donor_center,
            self.case_id,
            self.alternative,
            self.direction,
        )

    def response(self, response_id: str) -> float:
        if response_id not in UTILITY_RESPONSE_IDS:
            raise ProtocolError("PSSCUR requested an unknown utility response.")
        return float(getattr(self, response_id))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_donor_utility_row_v1",
            **self.__dict__,
            "feature_values": list(self.feature_values),
            "structural_zero": self.crossing_count == 0,
            "raw_label_persisted": False,
        }


@dataclass(frozen=True, order=True)
class PosteriorUtilityPrediction:
    """Analytic expected utility from a route-local posterior ensemble."""

    target_center: str
    case_id: str
    alternative: str
    direction: str
    control_id: str
    crossing_count: int
    fold_bacc_deltas: tuple[float, ...]
    fold_brier_deltas: tuple[float, ...]
    fold_log_loss_deltas: tuple[float, ...]
    robust_bacc_lower: float
    robust_brier_upper: float
    robust_log_loss_upper: float
    oof_auc: float
    oof_brier_skill: float
    reliability_pass: bool
    descriptor_hash: str
    ensemble_hash: str
    utility_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        fold_rows = (
            tuple(float(value) for value in self.fold_bacc_deltas),
            tuple(float(value) for value in self.fold_brier_deltas),
            tuple(float(value) for value in self.fold_log_loss_deltas),
        )
        summary = (
            float(self.robust_bacc_lower),
            float(self.robust_brier_upper),
            float(self.robust_log_loss_upper),
            float(self.oof_auc),
            float(self.oof_brier_skill),
        )
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or self.alternative not in ALTERNATIVE_METHOD_IDS
            or self.direction not in DIRECTION_IDS
            or self.control_id not in {"IDENTITY", "WITHIN_CASE_CYCLIC_SHIFT"}
            or type(self.crossing_count) is not int
            or self.crossing_count < 0
            or any(len(values) != SUPPORT_CROSSFIT_FOLD_COUNT for values in fold_rows)
            or any(not math.isfinite(value) for values in fold_rows for value in values)
            or any(not math.isfinite(value) for value in summary)
            or not 0.0 <= self.oof_auc <= 1.0
            or type(self.reliability_pass) is not bool
        ):
            raise ProtocolError("PSSCUR posterior utility prediction drifted.")
        for digest, name in (
            (self.descriptor_hash, "utility_descriptor_hash"),
            (self.ensemble_hash, "route_posterior_ensemble_hash"),
        ):
            require_sha256(digest, name)
        object.__setattr__(self, "fold_bacc_deltas", fold_rows[0])
        object.__setattr__(self, "fold_brier_deltas", fold_rows[1])
        object.__setattr__(self, "fold_log_loss_deltas", fold_rows[2])
        object.__setattr__(self, "utility_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.target_center, self.case_id, self.alternative, self.direction

    @property
    def proper_safe(self) -> bool:
        return self.robust_brier_upper <= 0.0 and self.robust_log_loss_upper <= 0.0

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_posterior_utility_prediction_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "alternative": self.alternative,
            "direction": self.direction,
            "control_id": self.control_id,
            "crossing_count": self.crossing_count,
            "fold_bacc_deltas": list(self.fold_bacc_deltas),
            "fold_brier_deltas": list(self.fold_brier_deltas),
            "fold_log_loss_deltas": list(self.fold_log_loss_deltas),
            "robust_bacc_lower": self.robust_bacc_lower,
            "robust_brier_upper": self.robust_brier_upper,
            "robust_log_loss_upper": self.robust_log_loss_upper,
            "oof_auc": self.oof_auc,
            "oof_brier_skill": self.oof_brier_skill,
            "reliability_pass": self.reliability_pass,
            "proper_safe": self.proper_safe,
            "descriptor_hash": self.descriptor_hash,
            "ensemble_hash": self.ensemble_hash,
            "terminal_labels_used": False,
            "confidence_bound_claimed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "utility_hash": self.utility_hash}


@dataclass(frozen=True, order=True)
class ResidualScale:
    """Partially pooled robust residual scale for one action cell/endpoint."""

    alternative: str
    direction: str
    response_id: str
    sample_count: int
    cell_scale: float
    pooled_scale: float
    shrunk_scale: float
    scale_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        numeric = tuple(
            float(value)
            for value in (self.cell_scale, self.pooled_scale, self.shrunk_scale)
        )
        if (
            self.alternative not in ALTERNATIVE_METHOD_IDS
            or self.direction not in DIRECTION_IDS
            or self.response_id not in UTILITY_RESPONSE_IDS
            or type(self.sample_count) is not int
            or self.sample_count <= 0
            or any(not math.isfinite(value) or value < RESIDUAL_SCALE_FLOOR for value in numeric)
        ):
            raise ProtocolError("PSSCUR residual scale drifted.")
        object.__setattr__(self, "cell_scale", numeric[0])
        object.__setattr__(self, "pooled_scale", numeric[1])
        object.__setattr__(self, "shrunk_scale", numeric[2])
        object.__setattr__(self, "scale_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str]:
        return self.alternative, self.direction, self.response_id

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_residual_scale_v1",
            "alternative": self.alternative,
            "direction": self.direction,
            "response_id": self.response_id,
            "sample_count": self.sample_count,
            "cell_scale": self.cell_scale,
            "pooled_scale": self.pooled_scale,
            "shrunk_scale": self.shrunk_scale,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "scale_hash": self.scale_hash}


@dataclass(frozen=True, order=True)
class FeatureReference:
    """Label-free robust donor reference for one action descriptor cell."""

    alternative: str
    direction: str
    sample_count: int
    locations: tuple[float, ...]
    scales: tuple[float, ...]
    reference_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        locations = _finite(
            self.locations, size=len(UTILITY_FEATURE_NAMES), name="feature locations"
        )
        scales = _finite(
            self.scales, size=len(UTILITY_FEATURE_NAMES), name="feature scales"
        )
        if (
            self.alternative not in ALTERNATIVE_METHOD_IDS
            or self.direction not in DIRECTION_IDS
            or type(self.sample_count) is not int
            or self.sample_count <= 0
            or any(value < RESIDUAL_SCALE_FLOOR for value in scales)
        ):
            raise ProtocolError("PSSCUR feature reference drifted.")
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "reference_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str]:
        return self.alternative, self.direction

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_feature_reference_v1",
            "alternative": self.alternative,
            "direction": self.direction,
            "sample_count": self.sample_count,
            "feature_names": list(UTILITY_FEATURE_NAMES),
            "locations": list(self.locations),
            "scales": list(self.scales),
            "labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "reference_hash": self.reference_hash}


@dataclass(frozen=True, order=True)
class DirectionEnvelope:
    """Selection-aware donor-block maximum residual envelope."""

    direction: str
    quantile: float
    radius: float
    maximum_radius: float
    donor_block_scores: tuple[tuple[str, float], ...]
    envelope_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        scores = tuple((str(center), float(value)) for center, value in self.donor_block_scores)
        numeric = (float(self.quantile), float(self.radius), float(self.maximum_radius))
        if (
            self.direction not in DIRECTION_IDS
            or self.quantile != DONOR_ENVELOPE_QUANTILE
            or len(scores) < 6
            or len({center for center, _value in scores}) != len(scores)
            or tuple(center for center, _value in scores)
            != tuple(center for center in CENTERS if center in {row[0] for row in scores})
            or any(center not in CENTERS or not math.isfinite(value) for center, value in scores)
            or any(not math.isfinite(value) or value < 0.0 for value in numeric)
            or self.radius > self.maximum_radius + BOUND_STRICT_TOLERANCE
        ):
            raise ProtocolError("PSSCUR direction envelope drifted.")
        object.__setattr__(self, "donor_block_scores", scores)
        object.__setattr__(self, "envelope_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_direction_envelope_v1",
            "direction": self.direction,
            "quantile": self.quantile,
            "radius": self.radius,
            "maximum_radius": self.maximum_radius,
            "donor_block_scores": [
                {"donor_center": center, "maximum_standardized_error": value}
                for center, value in self.donor_block_scores
            ],
            "maximum_taken_before_calibration": True,
            "finite_sample_coverage_claimed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "envelope_hash": self.envelope_hash}


@dataclass(frozen=True, order=True)
class InnerDonorReplay:
    """One donor-block replay using an envelope fitted on the other donors."""

    outer_target_center: str
    control_id: str
    held_donor_center: str
    training_donor_count: int
    selected_action_count: int
    bacc_delta: float
    brier_delta: float
    log_loss_delta: float
    envelope_model_hash: str
    replay_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        numeric = tuple(
            float(value)
            for value in (self.bacc_delta, self.brier_delta, self.log_loss_delta)
        )
        if (
            self.outer_target_center not in CENTERS
            or self.held_donor_center not in CENTERS
            or self.held_donor_center == self.outer_target_center
            or self.control_id not in {"IDENTITY", "WITHIN_CASE_CYCLIC_SHIFT"}
            or type(self.training_donor_count) is not int
            or self.training_donor_count != len(CENTERS) - 2
            or type(self.selected_action_count) is not int
            or self.selected_action_count < 0
            or any(not math.isfinite(value) for value in numeric)
        ):
            raise ProtocolError("PSSCUR inner donor replay drifted.")
        require_sha256(self.envelope_model_hash, "envelope_model_hash")
        object.__setattr__(self, "replay_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_inner_donor_replay_v1",
            "outer_target_center": self.outer_target_center,
            "control_id": self.control_id,
            "held_donor_center": self.held_donor_center,
            "training_donor_count": self.training_donor_count,
            "selected_action_count": self.selected_action_count,
            "bacc_delta": self.bacc_delta,
            "brier_delta": self.brier_delta,
            "log_loss_delta": self.log_loss_delta,
            "envelope_model_hash": self.envelope_model_hash,
            "outer_target_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "replay_hash": self.replay_hash}


@dataclass(frozen=True, order=True)
class EnvelopeCalibration:
    """Frozen simultaneous residual envelope for one outer center/control."""

    outer_target_center: str
    control_id: str
    authorized: bool
    residual_scales: tuple[ResidualScale, ...]
    feature_references: tuple[FeatureReference, ...]
    direction_envelopes: tuple[DirectionEnvelope, ...]
    selected_action_count: int
    donor_lower_tail_bacc_delta: float
    donor_upper_tail_brier_delta: float
    donor_upper_tail_log_loss_delta: float
    inner_replays: tuple[InnerDonorReplay, ...]
    source_utility_hash: str
    source_response_hash: str
    calibration_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        expected_donors = tuple(
            center for center in CENTERS if center != self.outer_target_center
        )
        scale_keys = tuple(row.key for row in self.residual_scales)
        reference_keys = tuple(row.key for row in self.feature_references)
        metrics = (
            float(self.donor_lower_tail_bacc_delta),
            float(self.donor_upper_tail_brier_delta),
            float(self.donor_upper_tail_log_loss_delta),
        )
        if (
            self.outer_target_center not in CENTERS
            or self.control_id not in {"IDENTITY", "WITHIN_CASE_CYCLIC_SHIFT"}
            or type(self.authorized) is not bool
            or type(self.selected_action_count) is not int
            or self.selected_action_count < 0
            or len(scale_keys) != 18
            or len(set(scale_keys)) != 18
            or scale_keys != tuple(sorted(scale_keys))
            or len(reference_keys) != 6
            or len(set(reference_keys)) != 6
            or reference_keys != tuple(sorted(reference_keys))
            or tuple(row.direction for row in self.direction_envelopes) != DIRECTION_IDS
            or any(
                tuple(center for center, _value in row.donor_block_scores)
                != expected_donors
                for row in self.direction_envelopes
            )
            or tuple(row.held_donor_center for row in self.inner_replays)
            != expected_donors
            or any(
                row.outer_target_center != self.outer_target_center
                or row.control_id != self.control_id
                for row in self.inner_replays
            )
            or any(not math.isfinite(value) for value in metrics)
        ):
            raise ProtocolError("PSSCUR envelope calibration drifted.")
        require_sha256(self.source_utility_hash, "envelope_source_utility_hash")
        require_sha256(self.source_response_hash, "envelope_source_response_hash")
        object.__setattr__(self, "calibration_hash", canonical_hash(self._unhashed()))

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

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_envelope_calibration_v1",
            "outer_target_center": self.outer_target_center,
            "control_id": self.control_id,
            "authorized": self.authorized,
            "residual_scales": [row.to_payload() for row in self.residual_scales],
            "feature_references": [row.to_payload() for row in self.feature_references],
            "direction_envelopes": [row.to_payload() for row in self.direction_envelopes],
            "selected_action_count": self.selected_action_count,
            "donor_lower_tail_bacc_delta": self.donor_lower_tail_bacc_delta,
            "donor_upper_tail_brier_delta": self.donor_upper_tail_brier_delta,
            "donor_upper_tail_log_loss_delta": self.donor_upper_tail_log_loss_delta,
            "inner_replays": [row.to_payload() for row in self.inner_replays],
            "source_utility_hash": self.source_utility_hash,
            "source_response_hash": self.source_response_hash,
            "inner_leave_one_donor_replay": True,
            "outer_target_labels_used": False,
            "finite_sample_coverage_claimed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "calibration_hash": self.calibration_hash}


@dataclass(frozen=True, order=True)
class UtilityCertificate:
    """Label-free bound certificate for one candidate under one frozen policy."""

    target_center: str
    case_id: str
    alternative: str
    direction: str
    control_id: str
    policy_id: str
    crossing_count: int
    point_bacc_delta: float
    point_brier_delta: float
    point_log_loss_delta: float
    descriptor_shift: float
    fold_instability: float
    shift_inflation: float
    envelope_radius: float
    lower_bacc_delta: float
    upper_brier_delta: float
    upper_log_loss_delta: float
    reliability_pass: bool
    descriptor_hash: str
    utility_hash: str
    calibration_hash: str
    certificate_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        numeric = tuple(
            float(value)
            for value in (
                self.point_bacc_delta,
                self.point_brier_delta,
                self.point_log_loss_delta,
                self.descriptor_shift,
                self.fold_instability,
                self.shift_inflation,
                self.envelope_radius,
                self.lower_bacc_delta,
                self.upper_brier_delta,
                self.upper_log_loss_delta,
            )
        )
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or self.alternative not in ALTERNATIVE_METHOD_IDS
            or self.direction not in DIRECTION_IDS
            or self.control_id not in {"IDENTITY", "WITHIN_CASE_CYCLIC_SHIFT"}
            or self.policy_id not in COMPOSED_POLICY_IDS
            or type(self.crossing_count) is not int
            or self.crossing_count < 0
            or type(self.reliability_pass) is not bool
            or any(not math.isfinite(value) for value in numeric)
            or self.descriptor_shift < 0.0
            or self.fold_instability < 0.0
            or not 1.0 <= self.shift_inflation <= SHIFT_KAPPA_CAP
            or self.envelope_radius < 0.0
        ):
            raise ProtocolError("PSSCUR utility certificate drifted.")
        for digest, name in (
            (self.descriptor_hash, "utility_descriptor_hash"),
            (self.utility_hash, "posterior_utility_hash"),
            (self.calibration_hash, "envelope_calibration_hash"),
        ):
            require_sha256(digest, name)
        object.__setattr__(self, "certificate_hash", canonical_hash(self._unhashed()))

    @property
    def admissible(self) -> bool:
        return (
            self.crossing_count > 0
            and self.reliability_pass
            and self.lower_bacc_delta > BOUND_STRICT_TOLERANCE
            and self.upper_brier_delta <= BOUND_STRICT_TOLERANCE
            and self.upper_log_loss_delta <= BOUND_STRICT_TOLERANCE
        )

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.target_center,
            self.case_id,
            self.alternative,
            self.direction,
            self.policy_id,
        )

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_utility_certificate_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "alternative": self.alternative,
            "direction": self.direction,
            "control_id": self.control_id,
            "policy_id": self.policy_id,
            "crossing_count": self.crossing_count,
            "point_bacc_delta": self.point_bacc_delta,
            "point_brier_delta": self.point_brier_delta,
            "point_log_loss_delta": self.point_log_loss_delta,
            "descriptor_shift": self.descriptor_shift,
            "fold_instability": self.fold_instability,
            "shift_inflation": self.shift_inflation,
            "envelope_radius": self.envelope_radius,
            "lower_bacc_delta": self.lower_bacc_delta,
            "upper_brier_delta": self.upper_brier_delta,
            "upper_log_loss_delta": self.upper_log_loss_delta,
            "reliability_pass": self.reliability_pass,
            "admissible": self.admissible,
            "descriptor_hash": self.descriptor_hash,
            "utility_hash": self.utility_hash,
            "calibration_hash": self.calibration_hash,
            "terminal_labels_used": False,
            "finite_sample_coverage_claimed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "certificate_hash": self.certificate_hash}


@dataclass(frozen=True, order=True)
class DirectionalDecision:
    target_center: str
    case_id: str
    policy_id: str
    direction: str
    selected_alternative: str
    selected_score: float
    candidate_prediction_hashes: tuple[str, ...]
    decision_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or self.policy_id not in COMPOSED_POLICY_IDS
            or self.direction not in DIRECTION_IDS
            or self.selected_alternative
            not in (*ALTERNATIVE_METHOD_IDS, PORTFOLIO_METHOD_ID)
            or not math.isfinite(float(self.selected_score))
        ):
            raise ProtocolError("PSSCUR directional decision drifted.")
        for digest in self.candidate_prediction_hashes:
            require_sha256(digest, "utility_prediction_hash")
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_directional_decision_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "policy_id": self.policy_id,
            "direction": self.direction,
            "selected_alternative": self.selected_alternative,
            "selected_score": self.selected_score,
            "candidate_prediction_hashes": list(self.candidate_prediction_hashes),
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_hash": self.decision_hash}


@dataclass(frozen=True, order=True)
class ComposedCasePrediction:
    target_center: str
    case_id: str
    policy_id: str
    sample_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    decisions: tuple[DirectionalDecision, ...]
    switched_sample_counts: tuple[tuple[str, int], ...]
    endpoint_prediction_hash: str
    prediction_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        samples = tuple(str(value) for value in self.sample_ids)
        probabilities = _finite(self.probabilities, size=len(samples), name="composition")
        counts = dict(self.switched_sample_counts)
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or self.policy_id not in COMPOSED_POLICY_IDS
            or not samples
            or len(samples) != len(set(samples))
            or any(not 0.0 <= value <= 1.0 for value in probabilities)
            or tuple(row.direction for row in self.decisions) != DIRECTION_IDS
            or any(
                row.target_center != self.target_center
                or row.case_id != self.case_id
                or row.policy_id != self.policy_id
                for row in self.decisions
            )
            or tuple(counts) != DIRECTION_IDS
            or any(type(value) is not int or value < 0 for value in counts.values())
        ):
            raise ProtocolError("PSSCUR composed prediction drifted.")
        require_sha256(self.endpoint_prediction_hash, "endpoint_prediction_hash")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "prediction_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_composed_case_prediction_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "policy_id": self.policy_id,
            "sample_ids": list(self.sample_ids),
            "probabilities": list(self.probabilities),
            "decisions": [row.to_payload() for row in self.decisions],
            "switched_sample_counts": dict(self.switched_sample_counts),
            "endpoint_prediction_hash": self.endpoint_prediction_hash,
            "sign_preserving_shrinkage": SIGN_PRESERVING_SHRINKAGE,
            "one_alternative_per_direction": True,
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prediction_hash": self.prediction_hash}


__all__ = (
    "ComposedCasePrediction",
    "DirectionEnvelope",
    "DirectionalDecision",
    "DonorUtilityRow",
    "EnvelopeCalibration",
    "FeatureReference",
    "InnerDonorReplay",
    "PosteriorUtilityPrediction",
    "ResidualScale",
    "UtilityCertificate",
    "UtilityDescriptor",
)
