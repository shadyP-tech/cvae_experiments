"""Typed contracts for complete signed-utility routing surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ...protocol import ProtocolError
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    CENTERS,
    COMPOSED_POLICY_IDS,
    DIRECTION_IDS,
    PORTFOLIO_METHOD_ID,
    SIGN_PRESERVING_SHRINKAGE,
    SUPPORT_CROSSFIT_FOLD_COUNT,
    UTILITY_FEATURE_NAMES,
    UTILITY_RESPONSE_IDS,
)
from .hashing import canonical_hash, require_sha256


def _finite(values: tuple[float, ...], *, size: int, name: str) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if len(converted) != size or any(not math.isfinite(value) for value in converted):
        raise ProtocolError(f"PUMR {name} drifted.")
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
            raise ProtocolError("PUMR utility descriptor identity drifted.")
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
            "schema_version": "fixed_bank_pumr_utility_descriptor_v1",
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
            raise ProtocolError("PUMR donor utility response drifted.")
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
            raise ProtocolError("PUMR requested an unknown utility response.")
        return float(getattr(self, response_id))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pumr_donor_utility_row_v1",
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
            raise ProtocolError("PUMR posterior utility prediction drifted.")
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
            "schema_version": "fixed_bank_pumr_posterior_utility_prediction_v1",
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
class InnerDonorReplay:
    """One leave-one-donor check of a margin chosen on the other donors."""

    outer_target_center: str
    control_id: str
    held_donor_center: str
    selected_margin: float
    selected_action_count: int
    bacc_delta: float
    brier_delta: float
    log_loss_delta: float
    replay_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        numeric = (
            float(self.selected_margin),
            float(self.bacc_delta),
            float(self.brier_delta),
            float(self.log_loss_delta),
        )
        if (
            self.outer_target_center not in CENTERS
            or self.held_donor_center not in CENTERS
            or self.held_donor_center == self.outer_target_center
            or self.control_id not in {"IDENTITY", "WITHIN_CASE_CYCLIC_SHIFT"}
            or type(self.selected_action_count) is not int
            or self.selected_action_count < 0
            or any(not math.isfinite(value) for value in numeric)
            or self.selected_margin < 0.0
        ):
            raise ProtocolError("PUMR inner donor replay drifted.")
        object.__setattr__(self, "replay_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pumr_inner_donor_replay_v1",
            "outer_target_center": self.outer_target_center,
            "control_id": self.control_id,
            "held_donor_center": self.held_donor_center,
            "selected_margin": self.selected_margin,
            "selected_action_count": self.selected_action_count,
            "bacc_delta": self.bacc_delta,
            "brier_delta": self.brier_delta,
            "log_loss_delta": self.log_loss_delta,
            "outer_target_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "replay_hash": self.replay_hash}


@dataclass(frozen=True, order=True)
class MarginCalibration:
    """Frozen donor-held scalar abstention margin for one outer center/control."""

    outer_target_center: str
    control_id: str
    selected_margin: float
    authorized: bool
    candidate_margins: tuple[float, ...]
    selected_action_count: int
    donor_bacc_delta: float
    donor_brier_delta: float
    donor_log_loss_delta: float
    inner_replays: tuple[InnerDonorReplay, ...]
    source_utility_hash: str
    source_response_hash: str
    calibration_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        candidates = tuple(float(value) for value in self.candidate_margins)
        metrics = (
            float(self.donor_bacc_delta),
            float(self.donor_brier_delta),
            float(self.donor_log_loss_delta),
        )
        expected_donors = tuple(
            center for center in CENTERS if center != self.outer_target_center
        )
        if (
            self.outer_target_center not in CENTERS
            or self.control_id not in {"IDENTITY", "WITHIN_CASE_CYCLIC_SHIFT"}
            or type(self.authorized) is not bool
            or type(self.selected_action_count) is not int
            or self.selected_action_count < 0
            or not candidates
            or candidates != tuple(sorted(set(candidates)))
            or self.selected_margin < 0.0
            or any(value < 0.0 or not math.isfinite(value) for value in candidates)
            or any(not math.isfinite(value) for value in metrics)
            or tuple(row.held_donor_center for row in self.inner_replays)
            != expected_donors
            or any(
                row.outer_target_center != self.outer_target_center
                or row.control_id != self.control_id
                for row in self.inner_replays
            )
        ):
            raise ProtocolError("PUMR margin calibration drifted.")
        require_sha256(self.source_utility_hash, "margin_source_utility_hash")
        require_sha256(self.source_response_hash, "margin_source_response_hash")
        object.__setattr__(self, "candidate_margins", candidates)
        object.__setattr__(self, "calibration_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pumr_margin_calibration_v1",
            "outer_target_center": self.outer_target_center,
            "control_id": self.control_id,
            "selected_margin": self.selected_margin,
            "authorized": self.authorized,
            "candidate_margins": list(self.candidate_margins),
            "selected_action_count": self.selected_action_count,
            "donor_bacc_delta": self.donor_bacc_delta,
            "donor_brier_delta": self.donor_brier_delta,
            "donor_log_loss_delta": self.donor_log_loss_delta,
            "inner_replays": [row.to_payload() for row in self.inner_replays],
            "source_utility_hash": self.source_utility_hash,
            "source_response_hash": self.source_response_hash,
            "inner_leave_one_donor_replay": True,
            "outer_target_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "calibration_hash": self.calibration_hash}


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
            raise ProtocolError("PUMR directional decision drifted.")
        for digest in self.candidate_prediction_hashes:
            require_sha256(digest, "utility_prediction_hash")
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pumr_directional_decision_v1",
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
            raise ProtocolError("PUMR composed prediction drifted.")
        require_sha256(self.endpoint_prediction_hash, "endpoint_prediction_hash")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "prediction_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pumr_composed_case_prediction_v1",
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
    "DirectionalDecision",
    "DonorUtilityRow",
    "InnerDonorReplay",
    "MarginCalibration",
    "PosteriorUtilityPrediction",
    "UtilityDescriptor",
)
