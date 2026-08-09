"""Per-target label-derived leave-one-center-out global prior G_H."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ...protocol import ProtocolError
from .core_contracts import CaseUtilitySurface
from .core_hashing import canonical_hash, finite, require_sha256
from .scientific_constants import BASELINE_ACTION_ID, MIDOGPP_CENTERS, candidate_actions


@dataclass(frozen=True)
class PriorConfig:
    prior_strength: float = 8.0
    variance_floor: float = 1.0e-6
    confidence_multiplier: float = 1.96
    minimum_gain: float = 0.0

    def __post_init__(self) -> None:
        for name in ("prior_strength", "variance_floor", "confidence_multiplier", "minimum_gain"):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if self.prior_strength <= 0.0 or self.variance_floor <= 0.0 or self.confidence_multiplier <= 0.0 or self.minimum_gain < 0.0:
            raise ProtocolError("LOCO global-prior hyperparameters violate the fixed domain.")

    def to_payload(self) -> dict[str, object]:
        return {
            "prior_strength": self.prior_strength,
            "variance_floor": self.variance_floor,
            "confidence_multiplier": self.confidence_multiplier,
            "minimum_gain": self.minimum_gain,
            "hyperparameters_fixed_prelabel": True,
            "tuning_grid_used": False,
        }


@dataclass(frozen=True, order=True)
class CandidatePriorEstimate:
    action_id: str
    other_center_count: int
    other_center_case_count: int
    shrunk_mean_gain_vs_b: float
    standard_error: float
    lower_confidence_bound: float
    estimate_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if self.action_id not in MIDOGPP_CENTERS:
            raise ProtocolError("Candidate prior must refer to a source action.")
        if (
            isinstance(self.other_center_count, bool)
            or self.other_center_count != 7
            or isinstance(self.other_center_case_count, bool)
            or self.other_center_case_count <= 0
        ):
            raise ProtocolError("Candidate prior requires other-center case evidence.")
        for name in ("shrunk_mean_gain_vs_b", "standard_error", "lower_confidence_bound"):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if self.standard_error < 0.0:
            raise ProtocolError("Candidate-prior standard error cannot be negative.")
        object.__setattr__(self, "estimate_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "other_center_count": self.other_center_count,
            "other_center_case_count": self.other_center_case_count,
            "shrunk_mean_gain_vs_b": self.shrunk_mean_gain_vs_b,
            "standard_error": self.standard_error,
            "lower_confidence_bound": self.lower_confidence_bound,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "estimate_hash": self.estimate_hash}


@dataclass(frozen=True)
class LocoGlobalPrior:
    target_center: str
    estimates: tuple[CandidatePriorEstimate, ...]
    global_action_id: str
    best_candidate_action_id: str
    source_exact_surface_hash: str
    probability_surface_hash: str
    config: PriorConfig
    prior_hash: str
    label_role: str = "label_derived_LOCO_global_prior"
    h_labels_used_in_g_h: bool = False
    g_h_shared_across_h: bool = False
    g_h_uses_other_consumed_test_centers: bool = True
    hyperparameters_fixed_prelabel: bool = True
    sealed_before_h_support_access: bool = True

    def __post_init__(self) -> None:
        estimates = tuple(self.estimates)
        if self.target_center not in MIDOGPP_CENTERS:
            raise ProtocolError("G_H has an unknown held-out target.")
        if tuple(value.action_id for value in estimates) != candidate_actions(self.target_center):
            raise ProtocolError("G_H must estimate exactly the eight non-target actions.")
        if self.best_candidate_action_id not in candidate_actions(self.target_center):
            raise ProtocolError("G_H best candidate is outside the fixed bank.")
        if self.global_action_id not in (BASELINE_ACTION_ID, *candidate_actions(self.target_center)):
            raise ProtocolError("G_H selected an invalid action.")
        require_sha256(self.source_exact_surface_hash, "source_exact_surface_hash")
        require_sha256(self.probability_surface_hash, "probability_surface_hash")
        require_sha256(self.prior_hash, "prior_hash")
        if (
            self.h_labels_used_in_g_h is not False
            or self.g_h_shared_across_h is not False
            or self.g_h_uses_other_consumed_test_centers is not True
            or self.hyperparameters_fixed_prelabel is not True
            or self.sealed_before_h_support_access is not True
        ):
            raise ProtocolError("G_H protocol flags drifted.")
        expected = canonical_hash(self._unhashed())
        if expected != self.prior_hash:
            raise ProtocolError("G_H prior hash drifted.")
        object.__setattr__(self, "estimates", estimates)

    def estimate(self, action_id: str) -> CandidatePriorEstimate | None:
        if action_id == BASELINE_ACTION_ID:
            return None
        for estimate in self.estimates:
            if estimate.action_id == action_id:
                return estimate
        raise KeyError(action_id)

    def mean_gain_vs_b(self, action_id: str) -> float:
        estimate = self.estimate(action_id)
        return 0.0 if estimate is None else estimate.shrunk_mean_gain_vs_b

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_label_aware_loco_global_prior_v1",
            "target_center": self.target_center,
            "estimates": [value.to_payload() for value in self.estimates],
            "global_action_id": self.global_action_id,
            "best_candidate_action_id": self.best_candidate_action_id,
            "source_exact_surface_hash": self.source_exact_surface_hash,
            "probability_surface_hash": self.probability_surface_hash,
            "config": self.config.to_payload(),
            "label_role": self.label_role,
            "G_H_uses_other_consumed_test_centers": True,
            "H_labels_used_in_G_H": False,
            "G_H_shared_across_H": False,
            "G_H_hyperparameters_fixed_prelabel": True,
            "G_H_sealed_before_H_support_access": True,
            "other_center_contribution_unit": "equal_weight_per_target_center",
            "target_expert_used": False,
            "expert_updates": False,
            "shared_model_updates": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prior_hash": self.prior_hash}


def fit_label_derived_loco_global_prior(
    target_center: str,
    other_center_utilities: CaseUtilitySurface,
    *,
    config: PriorConfig = PriorConfig(),
) -> LocoGlobalPrior:
    """Fit G_H from exact per-case effects of H' != H and seal it."""

    target = str(target_center)
    if target not in MIDOGPP_CENTERS:
        raise ProtocolError("Unknown held-out target for G_H.")
    if other_center_utilities.label_scope != f"label_derived_LOCO_global_prior::heldout_H={target}":
        raise ProtocolError("G_H received an incompatible label capability.")
    if any(row.target_center == target for row in other_center_utilities.rows):
        raise ProtocolError("H labels entered G_H.")
    expected_other_centers = set(MIDOGPP_CENTERS).difference((target,))
    if {row.target_center for row in other_center_utilities.rows} != expected_other_centers:
        raise ProtocolError("G_H must use all and only the other MIDOG++ centers.")
    estimates: list[CandidatePriorEstimate] = []
    for action in candidate_actions(target):
        gains_by_center = {
            center: tuple(
                row.exact_gain_vs_b
                for row in other_center_utilities.rows
                if row.target_center == center and row.action_id == action
            )
            for center in MIDOGPP_CENTERS
            if center not in (target, action)
        }
        if len(gains_by_center) != 7 or any(
            not gains for gains in gains_by_center.values()
        ):
            raise ProtocolError("G_H candidate lacks other-center exact utility.")
        center_means = tuple(
            sum(gains_by_center[center]) / len(gains_by_center[center])
            for center in MIDOGPP_CENTERS
            if center in gains_by_center
        )
        n = len(center_means)
        mean = sum(center_means) / n
        sample_variance = (
            sum((value - mean) ** 2 for value in center_means) / (n - 1)
            if n > 1
            else 0.0
        )
        shrunk = sum(center_means) / (n + config.prior_strength)
        standard_error = math.sqrt(
            max(sample_variance, config.variance_floor)
            / (n + config.prior_strength)
        )
        estimates.append(
            CandidatePriorEstimate(
                action_id=action,
                other_center_count=n,
                other_center_case_count=sum(
                    len(gains) for gains in gains_by_center.values()
                ),
                shrunk_mean_gain_vs_b=shrunk,
                standard_error=standard_error,
                lower_confidence_bound=shrunk - config.confidence_multiplier * standard_error,
            )
        )
    canonical = tuple(sorted(estimates, key=lambda value: candidate_actions(target).index(value.action_id)))
    best = max(canonical, key=lambda value: (value.shrunk_mean_gain_vs_b, -candidate_actions(target).index(value.action_id)))
    global_action = (
        best.action_id
        if best.lower_confidence_bound > config.minimum_gain
        else BASELINE_ACTION_ID
    )
    unhashed = {
        "schema_version": "fixed_bank_label_aware_loco_global_prior_v1",
        "target_center": target,
        "estimates": [value.to_payload() for value in canonical],
        "global_action_id": global_action,
        "best_candidate_action_id": best.action_id,
        "source_exact_surface_hash": other_center_utilities.exact_surface_hash,
        "probability_surface_hash": other_center_utilities.prerequisite_seal_hash,
        "config": config.to_payload(),
        "label_role": "label_derived_LOCO_global_prior",
        "G_H_uses_other_consumed_test_centers": True,
        "H_labels_used_in_G_H": False,
        "G_H_shared_across_H": False,
        "G_H_hyperparameters_fixed_prelabel": True,
        "G_H_sealed_before_H_support_access": True,
        "other_center_contribution_unit": "equal_weight_per_target_center",
        "target_expert_used": False,
        "expert_updates": False,
        "shared_model_updates": False,
    }
    return LocoGlobalPrior(
        target_center=target,
        estimates=canonical,
        global_action_id=global_action,
        best_candidate_action_id=best.action_id,
        source_exact_surface_hash=other_center_utilities.exact_surface_hash,
        probability_surface_hash=other_center_utilities.prerequisite_seal_hash,
        config=config,
        prior_hash=canonical_hash(unhashed),
    )


__all__ = (
    "CandidatePriorEstimate",
    "LocoGlobalPrior",
    "PriorConfig",
    "fit_label_derived_loco_global_prior",
)
