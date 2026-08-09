"""Fold-local exact-utility posterior centered on the sealed G_H prior."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ...protocol import ProtocolError
from .core_contracts import CaseUtilitySurface
from .core_hashing import canonical_hash, finite, require_sha256
from .global_prior import LocoGlobalPrior
from .partitions import CaseFold
from .scientific_constants import candidate_actions


@dataclass(frozen=True)
class PosteriorConfig:
    prior_strength: float = 8.0
    variance_floor: float = 1.0e-6
    confidence_multiplier: float = 1.96
    minimum_gain: float = 0.0

    def __post_init__(self) -> None:
        for name in ("prior_strength", "variance_floor", "confidence_multiplier", "minimum_gain"):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if self.prior_strength <= 0.0 or self.variance_floor <= 0.0 or self.confidence_multiplier <= 0.0 or self.minimum_gain < 0.0:
            raise ProtocolError("Fold-local posterior hyperparameters violate the fixed domain.")

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
class CandidatePosteriorEstimate:
    action_id: str
    support_case_count: int
    prior_mean_gain_vs_g: float
    posterior_mean_gain_vs_g: float
    standard_error: float
    lower_confidence_bound: float
    estimate_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if isinstance(self.support_case_count, bool) or self.support_case_count <= 0:
            raise ProtocolError("Posterior estimate requires support cases.")
        for name in (
            "prior_mean_gain_vs_g",
            "posterior_mean_gain_vs_g",
            "standard_error",
            "lower_confidence_bound",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if self.standard_error < 0.0:
            raise ProtocolError("Posterior standard error cannot be negative.")
        object.__setattr__(self, "estimate_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "support_case_count": self.support_case_count,
            "prior_mean_gain_vs_g": self.prior_mean_gain_vs_g,
            "posterior_mean_gain_vs_g": self.posterior_mean_gain_vs_g,
            "standard_error": self.standard_error,
            "lower_confidence_bound": self.lower_confidence_bound,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "estimate_hash": self.estimate_hash}


@dataclass(frozen=True)
class FoldPosterior:
    target_center: str
    fold_ordinal: int
    fold_hash: str
    global_prior_hash: str
    support_exact_surface_hash: str
    support_case_ids: tuple[str, ...]
    estimates: tuple[CandidatePosteriorEstimate, ...]
    config: PosteriorConfig
    posterior_hash: str
    exact_response_only: bool = True
    smooth_response_used: bool = False

    def __post_init__(self) -> None:
        for name in ("fold_hash", "global_prior_hash", "support_exact_surface_hash", "posterior_hash"):
            require_sha256(getattr(self, name), name)
        support = tuple(sorted(self.support_case_ids))
        estimates = tuple(self.estimates)
        if tuple(value.action_id for value in estimates) != candidate_actions(self.target_center):
            raise ProtocolError("Fold posterior must cover exactly eight non-target actions.")
        if not support or self.exact_response_only is not True or self.smooth_response_used is not False:
            raise ProtocolError("Fold posterior violated exact-only support use.")
        if canonical_hash(self._unhashed()) != self.posterior_hash:
            raise ProtocolError("Fold posterior hash drifted.")
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "estimates", estimates)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_label_aware_fold_posterior_v1",
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "fold_hash": self.fold_hash,
            "global_prior_hash": self.global_prior_hash,
            "support_exact_surface_hash": self.support_exact_surface_hash,
            "support_case_ids": list(self.support_case_ids),
            "estimates": [value.to_payload() for value in self.estimates],
            "config": self.config.to_payload(),
            "exact_response_only": True,
            "smooth_response_used": False,
            "evaluation_labels_used": False,
            "shared_model_updated": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "posterior_hash": self.posterior_hash}


def fit_fold_local_posterior(
    fold: CaseFold,
    support_utilities: CaseUtilitySurface,
    global_prior: LocoGlobalPrior,
    *,
    config: PosteriorConfig = PosteriorConfig(),
) -> FoldPosterior:
    """Shrink equal-case candidate-minus-G exact effects toward G_H."""

    if (
        global_prior.target_center != fold.target_center
        or not support_utilities.label_scope.startswith(
            f"fold_local_support::{fold.fold_id}"
        )
        or support_utilities.prerequisite_seal_hash != global_prior.prior_hash
        or set(support_utilities.allowed_case_keys)
        != {(fold.target_center, case_id) for case_id in fold.support_case_ids}
    ):
        raise ProtocolError("Fold-local posterior received a mismatched support capability.")
    if any(row.target_center != fold.target_center for row in support_utilities.rows):
        raise ProtocolError("Fold-local support crossed target centers.")
    utility = support_utilities.by_key()
    g_action = global_prior.global_action_id
    estimates: list[CandidatePosteriorEstimate] = []
    for action in candidate_actions(fold.target_center):
        gains = tuple(
            utility[(fold.target_center, case_id, action)].exact_bacc
            - utility[(fold.target_center, case_id, g_action)].exact_bacc
            for case_id in fold.support_case_ids
        )
        n = len(gains)
        local_mean = sum(gains) / n
        sample_variance = (
            sum((value - local_mean) ** 2 for value in gains) / (n - 1)
            if n > 1
            else 0.0
        )
        prior_mean = global_prior.mean_gain_vs_b(action) - global_prior.mean_gain_vs_b(g_action)
        posterior_mean = (
            config.prior_strength * prior_mean + sum(gains)
        ) / (config.prior_strength + n)
        standard_error = math.sqrt(
            max(sample_variance, config.variance_floor)
            / (config.prior_strength + n)
        )
        estimates.append(
            CandidatePosteriorEstimate(
                action_id=action,
                support_case_count=n,
                prior_mean_gain_vs_g=prior_mean,
                posterior_mean_gain_vs_g=posterior_mean,
                standard_error=standard_error,
                lower_confidence_bound=posterior_mean - config.confidence_multiplier * standard_error,
            )
        )
    canonical = tuple(
        sorted(estimates, key=lambda value: candidate_actions(fold.target_center).index(value.action_id))
    )
    unhashed = {
        "schema_version": "fixed_bank_label_aware_fold_posterior_v1",
        "target_center": fold.target_center,
        "fold_ordinal": fold.fold_ordinal,
        "fold_hash": fold.fold_hash,
        "global_prior_hash": global_prior.prior_hash,
        "support_exact_surface_hash": support_utilities.exact_surface_hash,
        "support_case_ids": list(fold.support_case_ids),
        "estimates": [value.to_payload() for value in canonical],
        "config": config.to_payload(),
        "exact_response_only": True,
        "smooth_response_used": False,
        "evaluation_labels_used": False,
        "shared_model_updated": False,
    }
    return FoldPosterior(
        target_center=fold.target_center,
        fold_ordinal=fold.fold_ordinal,
        fold_hash=fold.fold_hash,
        global_prior_hash=global_prior.prior_hash,
        support_exact_surface_hash=support_utilities.exact_surface_hash,
        support_case_ids=fold.support_case_ids,
        estimates=canonical,
        config=config,
        posterior_hash=canonical_hash(unhashed),
    )


__all__ = (
    "CandidatePosteriorEstimate",
    "FoldPosterior",
    "PosteriorConfig",
    "fit_fold_local_posterior",
)
