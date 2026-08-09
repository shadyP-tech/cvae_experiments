"""Pooled center-equal LOCO selection and challenger-vs-G prior seals."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ...protocol import ProtocolError
from .core_contracts import SufficientStatisticSurface
from .core_hashing import canonical_hash, finite, require_sha256
from .pooled_metrics import action_rows, paired_pooled_difference
from .scientific_constants import (
    BASELINE_ACTION_ID,
    DEFAULT_CONFIDENCE_MULTIPLIER,
    DEFAULT_MINIMUM_GAIN,
    DEFAULT_TIE_TOLERANCE,
    DEFAULT_VARIANCE_FLOOR,
    MIDOGPP_CENTERS,
    candidate_actions,
    legal_donor_centers,
    routing_challengers,
)


@dataclass(frozen=True)
class PriorConfig:
    variance_floor: float = DEFAULT_VARIANCE_FLOOR
    confidence_multiplier: float = DEFAULT_CONFIDENCE_MULTIPLIER
    minimum_gain: float = DEFAULT_MINIMUM_GAIN
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE

    def __post_init__(self) -> None:
        for name in (
            "variance_floor",
            "confidence_multiplier",
            "minimum_gain",
            "tie_tolerance",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if self.variance_floor <= 0.0 or self.confidence_multiplier <= 0.0:
            raise ProtocolError("LOCO prior variance/confidence constants are invalid.")
        if self.minimum_gain != 0.0:
            raise ProtocolError("The v2 G_H gate is frozen at a strict zero lower bound.")
        if self.tie_tolerance != DEFAULT_TIE_TOLERANCE:
            raise ProtocolError("The v2 G_H lexicographic tie tolerance drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            "variance_floor": self.variance_floor,
            "confidence_multiplier": self.confidence_multiplier,
            "minimum_gain": self.minimum_gain,
            "tie_tolerance": self.tie_tolerance,
            "hyperparameters_fixed_prelabel": True,
            "tuning_grid_used": False,
        }


@dataclass(frozen=True, order=True)
class CandidateGlobalEstimate:
    action_id: str
    donor_center_effects: tuple[tuple[str, float], ...]
    donor_center_case_count: int
    mean_gain_vs_b: float
    variance_of_mean: float
    standard_error: float
    lower_confidence_bound: float
    estimate_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        effects = tuple(self.donor_center_effects)
        if self.action_id not in MIDOGPP_CENTERS or len(effects) != 7:
            raise ProtocolError("Each G_H candidate estimate requires seven legal donors.")
        if len({center for center, _ in effects}) != 7 or self.action_id in {
            center for center, _ in effects
        }:
            raise ProtocolError("G_H candidate donor exclusions drifted.")
        if self.donor_center_case_count <= 0:
            raise ProtocolError("G_H candidate estimate lacks donor cases.")
        for _center, effect in effects:
            finite(effect, "donor_center_effect")
        for name in (
            "mean_gain_vs_b",
            "variance_of_mean",
            "standard_error",
            "lower_confidence_bound",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if self.variance_of_mean <= 0.0 or self.standard_error <= 0.0:
            raise ProtocolError("G_H candidate uncertainty must be positive.")
        if abs(self.standard_error**2 - self.variance_of_mean) > 1.0e-12:
            raise ProtocolError("G_H standard error differs from its variance.")
        object.__setattr__(self, "donor_center_effects", effects)
        object.__setattr__(self, "estimate_hash", canonical_hash(self._unhashed()))

    @property
    def other_center_count(self) -> int:
        return len(self.donor_center_effects)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_candidate_global_estimate_v2",
            "action_id": self.action_id,
            "donor_center_effects": [[center, value] for center, value in self.donor_center_effects],
            "donor_center_count": len(self.donor_center_effects),
            "donor_center_case_count": self.donor_center_case_count,
            "mean_gain_vs_b": self.mean_gain_vs_b,
            "variance_of_mean": self.variance_of_mean,
            "standard_error": self.standard_error,
            "lower_confidence_bound": self.lower_confidence_bound,
            "donor_center_weighting": "equal",
            "pooled_bacc_within_donor_center": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "estimate_hash": self.estimate_hash}


@dataclass(frozen=True, order=True)
class PairwisePriorEstimate:
    challenger_action_id: str
    reference_action_id: str
    donor_center_effects: tuple[tuple[str, float], ...]
    prior_mean: float
    prior_variance: float
    estimate_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        effects = tuple(self.donor_center_effects)
        expected_count = 7 if self.reference_action_id == BASELINE_ACTION_ID else 6
        if self.challenger_action_id == self.reference_action_id:
            raise ProtocolError("A source-valued G_H cannot be its own challenger.")
        if len(effects) != expected_count or len({center for center, _ in effects}) != expected_count:
            raise ProtocolError("Pairwise prior donor count/exclusions drifted.")
        if self.challenger_action_id in {center for center, _ in effects}:
            raise ProtocolError("Pairwise prior used its challenger as a donor center.")
        if self.reference_action_id != BASELINE_ACTION_ID and self.reference_action_id in {
            center for center, _ in effects
        }:
            raise ProtocolError("Pairwise prior used its source reference as a donor center.")
        for _center, effect in effects:
            finite(effect, "donor_center_effect")
        object.__setattr__(self, "prior_mean", finite(self.prior_mean, "prior_mean"))
        object.__setattr__(self, "prior_variance", finite(self.prior_variance, "prior_variance"))
        if self.prior_variance <= 0.0:
            raise ProtocolError("Pairwise prior variance must be positive.")
        object.__setattr__(self, "donor_center_effects", effects)
        object.__setattr__(self, "estimate_hash", canonical_hash(self._unhashed()))

    @property
    def donor_center_count(self) -> int:
        return len(self.donor_center_effects)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_pairwise_prior_estimate_v2",
            "challenger_action_id": self.challenger_action_id,
            "reference_action_id": self.reference_action_id,
            "donor_center_effects": [[center, value] for center, value in self.donor_center_effects],
            "donor_center_count": len(self.donor_center_effects),
            "prior_mean": self.prior_mean,
            "prior_variance": self.prior_variance,
            "prior_variance_formula": "max(sample_variance_over_J,variance_floor)",
            "donor_center_weighting": "equal",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "estimate_hash": self.estimate_hash}


@dataclass(frozen=True)
class PooledLocoPrior:
    target_center: str
    candidate_estimates: tuple[CandidateGlobalEstimate, ...]
    pairwise_estimates: tuple[PairwisePriorEstimate, ...]
    global_action_id: str
    best_candidate_action_id: str
    source_statistics_surface_hash: str
    probability_surface_hash: str
    config: PriorConfig
    prior_hash: str
    sealed_before_h_support_access: bool = True
    h_labels_used_in_g_h: bool = False

    def __post_init__(self) -> None:
        candidates = tuple(self.candidate_estimates)
        pairwise = tuple(self.pairwise_estimates)
        if self.target_center not in MIDOGPP_CENTERS:
            raise ProtocolError("G_H has an unknown held-out target.")
        expected_candidates = candidate_actions(self.target_center)
        if tuple(value.action_id for value in candidates) != expected_candidates:
            raise ProtocolError("G_H must estimate exactly eight non-target sources.")
        if self.global_action_id not in (BASELINE_ACTION_ID, *expected_candidates):
            raise ProtocolError("G_H selected an action outside the fixed bank.")
        if self.best_candidate_action_id not in expected_candidates:
            raise ProtocolError("G_H best candidate is invalid.")
        expected_challengers = routing_challengers(self.target_center, self.global_action_id)
        if tuple(value.challenger_action_id for value in pairwise) != expected_challengers:
            raise ProtocolError("Pairwise prior seal does not cover exactly the distinct challengers.")
        if any(value.reference_action_id != self.global_action_id for value in pairwise):
            raise ProtocolError("Pairwise prior references drifted from selected G_H.")
        for estimate in candidates:
            expected_donors = legal_donor_centers(
                self.target_center, estimate.action_id, BASELINE_ACTION_ID
            )
            if tuple(center for center, _value in estimate.donor_center_effects) != expected_donors:
                raise ProtocolError("G_H candidate donor-center exclusion/order drifted.")
            effects = tuple(value for _center, value in estimate.donor_center_effects)
            mean, variance = _mean_and_variance_of_mean(effects, self.config.variance_floor)
            standard_error = math.sqrt(variance)
            expected_lcb = mean - self.config.confidence_multiplier * standard_error
            if (
                abs(estimate.mean_gain_vs_b - mean) > 1.0e-12
                or abs(estimate.variance_of_mean - variance) > 1.0e-12
                or abs(estimate.standard_error - standard_error) > 1.0e-12
                or abs(estimate.lower_confidence_bound - expected_lcb) > 1.0e-12
            ):
                raise ProtocolError("G_H candidate mathematical identity drifted.")
        for estimate in pairwise:
            expected_donors = legal_donor_centers(
                self.target_center,
                estimate.challenger_action_id,
                self.global_action_id,
            )
            if tuple(center for center, _value in estimate.donor_center_effects) != expected_donors:
                raise ProtocolError("Pairwise-prior donor-center exclusion/order drifted.")
            effects = tuple(value for _center, value in estimate.donor_center_effects)
            mean, variance = _mean_and_variance_of_mean(effects, self.config.variance_floor)
            if (
                abs(estimate.prior_mean - mean) > 1.0e-12
                or abs(estimate.prior_variance - variance) > 1.0e-12
            ):
                raise ProtocolError("Pairwise prior mathematical identity drifted.")
        expected_best = _select_best_candidate(candidates, self.config.tie_tolerance)
        expected_global = (
            expected_best.action_id
            if expected_best.lower_confidence_bound > self.config.minimum_gain
            else BASELINE_ACTION_ID
        )
        if (
            self.best_candidate_action_id != expected_best.action_id
            or self.global_action_id != expected_global
        ):
            raise ProtocolError("G_H best-candidate/gate identity drifted.")
        for name in ("source_statistics_surface_hash", "probability_surface_hash", "prior_hash"):
            require_sha256(getattr(self, name), name)
        if self.sealed_before_h_support_access is not True or self.h_labels_used_in_g_h is not False:
            raise ProtocolError("G_H prior label/seal boundary drifted.")
        if canonical_hash(self._unhashed()) != self.prior_hash:
            raise ProtocolError("Pooled G_H prior hash drifted.")
        object.__setattr__(self, "candidate_estimates", candidates)
        object.__setattr__(self, "pairwise_estimates", pairwise)

    @property
    def estimates(self) -> tuple[CandidateGlobalEstimate, ...]:
        return self.candidate_estimates

    def estimate(self, action_id: str) -> CandidateGlobalEstimate | None:
        if action_id == BASELINE_ACTION_ID:
            return None
        for value in self.candidate_estimates:
            if value.action_id == action_id:
                return value
        raise KeyError(action_id)

    def pairwise_estimate(self, challenger_action_id: str) -> PairwisePriorEstimate:
        for value in self.pairwise_estimates:
            if value.challenger_action_id == challenger_action_id:
                return value
        raise KeyError(challenger_action_id)

    def mean_gain_vs_b(self, action_id: str) -> float:
        estimate = self.estimate(action_id)
        return 0.0 if estimate is None else estimate.mean_gain_vs_b

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_loco_prior_v2",
            "target_center": self.target_center,
            "candidate_estimates": [value.to_payload() for value in self.candidate_estimates],
            "pairwise_estimates": [value.to_payload() for value in self.pairwise_estimates],
            "global_action_id": self.global_action_id,
            "best_candidate_action_id": self.best_candidate_action_id,
            "source_statistics_surface_hash": self.source_statistics_surface_hash,
            "probability_surface_hash": self.probability_surface_hash,
            "config": self.config.to_payload(),
            "G_H_selection": "maximum_prior_mean_lexicographic_tie_positive_95pct_LCB_else_B",
            "G_H_donor_count_per_candidate": 7,
            "pairwise_donor_count": 7 if self.global_action_id == BASELINE_ACTION_ID else 6,
            "H_labels_used_in_G_H": False,
            "G_H_sealed_before_H_support_access": True,
            "target_expert_used": False,
            "expert_updates": False,
            "shared_model_updates": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prior_hash": self.prior_hash}


def fit_pooled_loco_prior(
    target_center: str,
    other_center_statistics: SufficientStatisticSurface,
    *,
    config: PriorConfig = PriorConfig(),
) -> PooledLocoPrior:
    target = str(target_center)
    if target not in MIDOGPP_CENTERS:
        raise ProtocolError("Unknown held-out target for pooled G_H.")
    if other_center_statistics.label_scope != f"label_derived_LOCO_pooled_prior::heldout_H={target}":
        raise ProtocolError("Pooled G_H received an incompatible label capability.")
    observed_centers = {row.target_center for row in other_center_statistics.rows}
    if target in observed_centers or observed_centers != set(MIDOGPP_CENTERS).difference((target,)):
        raise ProtocolError("Pooled G_H must use all and only H-pruned center labels.")
    candidates: list[CandidateGlobalEstimate] = []
    for action in candidate_actions(target):
        donors = legal_donor_centers(target, action, BASELINE_ACTION_ID)
        effects = tuple(
            (
                donor,
                paired_pooled_difference(
                    action_rows(other_center_statistics, target_center=donor, action_id=action),
                    action_rows(
                        other_center_statistics,
                        target_center=donor,
                        action_id=BASELINE_ACTION_ID,
                    ),
                ),
            )
            for donor in donors
        )
        mean, variance = _mean_and_variance_of_mean(
            tuple(value for _center, value in effects), config.variance_floor
        )
        standard_error = math.sqrt(variance)
        candidates.append(
            CandidateGlobalEstimate(
                action_id=action,
                donor_center_effects=effects,
                donor_center_case_count=sum(
                    len(
                        action_rows(
                            other_center_statistics, target_center=donor, action_id=action
                        )
                    )
                    for donor in donors
                ),
                mean_gain_vs_b=mean,
                variance_of_mean=variance,
                standard_error=standard_error,
                lower_confidence_bound=mean - config.confidence_multiplier * standard_error,
            )
        )
    canonical_candidates = tuple(candidates)
    order = candidate_actions(target)
    del order
    best = _select_best_candidate(canonical_candidates, config.tie_tolerance)
    global_action = (
        best.action_id
        if best.lower_confidence_bound > config.minimum_gain
        else BASELINE_ACTION_ID
    )
    pairwise: list[PairwisePriorEstimate] = []
    for challenger in routing_challengers(target, global_action):
        donors = legal_donor_centers(target, challenger, global_action)
        effects = tuple(
            (
                donor,
                paired_pooled_difference(
                    action_rows(
                        other_center_statistics, target_center=donor, action_id=challenger
                    ),
                    action_rows(
                        other_center_statistics,
                        target_center=donor,
                        action_id=global_action,
                    ),
                ),
            )
            for donor in donors
        )
        mean, variance = _mean_and_variance_of_mean(
            tuple(value for _center, value in effects), config.variance_floor
        )
        pairwise.append(
            PairwisePriorEstimate(
                challenger_action_id=challenger,
                reference_action_id=global_action,
                donor_center_effects=effects,
                prior_mean=mean,
                prior_variance=variance,
            )
        )
    canonical_pairwise = tuple(pairwise)
    values = {
        "target_center": target,
        "candidate_estimates": canonical_candidates,
        "pairwise_estimates": canonical_pairwise,
        "global_action_id": global_action,
        "best_candidate_action_id": best.action_id,
        "source_statistics_surface_hash": other_center_statistics.statistics_surface_hash,
        "probability_surface_hash": other_center_statistics.prerequisite_seal_hash,
        "config": config,
    }
    proto = _prior_payload(values)
    return PooledLocoPrior(**values, prior_hash=canonical_hash(proto))


# Runner-friendly explicit spelling.
fit_label_derived_loco_global_prior = fit_pooled_loco_prior


def _mean_and_variance_of_mean(
    effects: tuple[float, ...], variance_floor: float
) -> tuple[float, float]:
    if len(effects) < 2:
        raise ProtocolError("LOCO prior variance requires at least two donor centers.")
    mean = sum(effects) / len(effects)
    sample_variance = sum((value - mean) ** 2 for value in effects) / (len(effects) - 1)
    return mean, max(sample_variance / len(effects), variance_floor)


def _select_best_candidate(
    estimates: tuple[CandidateGlobalEstimate, ...], tie_tolerance: float
) -> CandidateGlobalEstimate:
    if not estimates:
        raise ProtocolError("G_H candidate selection cannot be empty.")
    maximum = max(value.mean_gain_vs_b for value in estimates)
    eligible = tuple(
        value
        for value in estimates
        if maximum - value.mean_gain_vs_b <= tie_tolerance
    )
    order = candidate_actions(
        next(center for center in MIDOGPP_CENTERS if center not in {v.action_id for v in estimates})
    )
    return min(eligible, key=lambda value: order.index(value.action_id))


def _prior_payload(values: dict[str, object]) -> dict[str, object]:
    target = str(values["target_center"])
    global_action = str(values["global_action_id"])
    return {
        "schema_version": "fixed_bank_pooled_bacc_loco_prior_v2",
        "target_center": target,
        "candidate_estimates": [value.to_payload() for value in values["candidate_estimates"]],
        "pairwise_estimates": [value.to_payload() for value in values["pairwise_estimates"]],
        "global_action_id": global_action,
        "best_candidate_action_id": values["best_candidate_action_id"],
        "source_statistics_surface_hash": values["source_statistics_surface_hash"],
        "probability_surface_hash": values["probability_surface_hash"],
        "config": values["config"].to_payload(),
        "G_H_selection": "maximum_prior_mean_lexicographic_tie_positive_95pct_LCB_else_B",
        "G_H_donor_count_per_candidate": 7,
        "pairwise_donor_count": 7 if global_action == BASELINE_ACTION_ID else 6,
        "H_labels_used_in_G_H": False,
        "G_H_sealed_before_H_support_access": True,
        "target_expert_used": False,
        "expert_updates": False,
        "shared_model_updates": False,
    }


__all__ = (
    "CandidateGlobalEstimate",
    "PairwisePriorEstimate",
    "PooledLocoPrior",
    "PriorConfig",
    "fit_label_derived_loco_global_prior",
    "fit_pooled_loco_prior",
)
