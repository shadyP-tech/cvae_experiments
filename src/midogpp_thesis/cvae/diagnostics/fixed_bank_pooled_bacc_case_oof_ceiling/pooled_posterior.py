"""Fixed normal-normal updates from pooled support effects and case clusters."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ...protocol import ProtocolError
from .case_partitions import CaseFold
from .core_contracts import SufficientStatisticSurface
from .core_hashing import canonical_hash, finite, require_sha256
from .pooled_metrics import action_rows, paired_whole_case_cluster_contrast
from .pooled_prior import PooledLocoPrior
from .scientific_constants import (
    DEFAULT_CONFIDENCE_MULTIPLIER,
    DEFAULT_MINIMUM_GAIN,
    DEFAULT_VARIANCE_FLOOR,
    routing_challengers,
)


@dataclass(frozen=True)
class PosteriorConfig:
    variance_floor: float = DEFAULT_VARIANCE_FLOOR
    confidence_multiplier: float = DEFAULT_CONFIDENCE_MULTIPLIER
    minimum_gain: float = DEFAULT_MINIMUM_GAIN

    def __post_init__(self) -> None:
        for name in ("variance_floor", "confidence_multiplier", "minimum_gain"):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if self.variance_floor <= 0.0 or self.confidence_multiplier <= 0.0:
            raise ProtocolError("Posterior variance/confidence constants are invalid.")
        if self.minimum_gain != 0.0:
            raise ProtocolError("The v2 posterior switch gate is frozen at strict zero.")

    def to_payload(self) -> dict[str, object]:
        return {
            "variance_floor": self.variance_floor,
            "confidence_multiplier": self.confidence_multiplier,
            "minimum_gain": self.minimum_gain,
            "hyperparameters_fixed_prelabel": True,
            "tuning_grid_used": False,
            "posterior_family": "normal_normal_fixed",
        }


@dataclass(frozen=True, order=True)
class CandidatePosteriorEstimate:
    action_id: str
    reference_action_id: str
    support_case_count: int
    support_positive_count: int
    support_negative_count: int
    prior_mean: float
    prior_variance: float
    support_pooled_difference: float
    support_cluster_variance: float
    posterior_mean: float
    posterior_variance: float
    posterior_standard_error: float
    lower_confidence_bound: float
    support_contrast_hash: str
    estimate_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if self.action_id == self.reference_action_id:
            raise ProtocolError("A source-valued G_H cannot be its own posterior challenger.")
        if self.support_case_count < 2 or self.support_positive_count <= 0 or self.support_negative_count <= 0:
            raise ProtocolError("Posterior estimate requires clustered support with both pooled classes.")
        require_sha256(self.support_contrast_hash, "support_contrast_hash")
        for name in (
            "prior_mean",
            "prior_variance",
            "support_pooled_difference",
            "support_cluster_variance",
            "posterior_mean",
            "posterior_variance",
            "posterior_standard_error",
            "lower_confidence_bound",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if (
            self.prior_variance <= 0.0
            or self.support_cluster_variance <= 0.0
            or self.posterior_variance <= 0.0
            or self.posterior_standard_error <= 0.0
        ):
            raise ProtocolError("Normal-normal variances must be positive.")
        if abs(self.posterior_standard_error**2 - self.posterior_variance) > 1.0e-12:
            raise ProtocolError("Posterior standard error differs from posterior variance.")
        object.__setattr__(self, "estimate_hash", canonical_hash(self._unhashed()))

    @property
    def posterior_mean_gain_vs_g(self) -> float:
        return self.posterior_mean

    @property
    def standard_error(self) -> float:
        return self.posterior_standard_error

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_candidate_posterior_v2",
            "action_id": self.action_id,
            "reference_action_id": self.reference_action_id,
            "support_case_count": self.support_case_count,
            "support_positive_count": self.support_positive_count,
            "support_negative_count": self.support_negative_count,
            "prior_mean": self.prior_mean,
            "prior_variance": self.prior_variance,
            "support_pooled_difference": self.support_pooled_difference,
            "support_cluster_variance": self.support_cluster_variance,
            "posterior_mean": self.posterior_mean,
            "posterior_variance": self.posterior_variance,
            "posterior_standard_error": self.posterior_standard_error,
            "lower_confidence_bound": self.lower_confidence_bound,
            "support_contrast_hash": self.support_contrast_hash,
            "normal_normal_update": True,
            "uncertainty_unit": "paired_whole_case_cluster",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "estimate_hash": self.estimate_hash}


@dataclass(frozen=True)
class PooledFoldPosterior:
    target_center: str
    fold_ordinal: int
    fold_hash: str
    global_prior_hash: str
    support_statistics_surface_hash: str
    support_case_ids: tuple[str, ...]
    global_action_id: str
    estimates: tuple[CandidatePosteriorEstimate, ...]
    config: PosteriorConfig
    posterior_hash: str
    evaluation_labels_used: bool = False

    def __post_init__(self) -> None:
        for name in (
            "fold_hash",
            "global_prior_hash",
            "support_statistics_surface_hash",
            "posterior_hash",
        ):
            require_sha256(getattr(self, name), name)
        support = tuple(sorted(self.support_case_ids))
        estimates = tuple(self.estimates)
        expected = routing_challengers(self.target_center, self.global_action_id)
        if tuple(value.action_id for value in estimates) != expected:
            raise ProtocolError("Fold posterior must cover exactly the distinct challengers.")
        if not support or self.evaluation_labels_used is not False:
            raise ProtocolError("Fold posterior violated support/evaluation isolation.")
        if any(value.reference_action_id != self.global_action_id for value in estimates):
            raise ProtocolError("Fold posterior reference drifted from G_H.")
        for value in estimates:
            expected_variance = 1.0 / (
                1.0 / value.prior_variance + 1.0 / value.support_cluster_variance
            )
            expected_mean = expected_variance * (
                value.prior_mean / value.prior_variance
                + value.support_pooled_difference / value.support_cluster_variance
            )
            expected_se = math.sqrt(expected_variance)
            expected_lcb = expected_mean - self.config.confidence_multiplier * expected_se
            if (
                value.support_cluster_variance < self.config.variance_floor
                or abs(value.posterior_variance - expected_variance) > 1.0e-12
                or abs(value.posterior_mean - expected_mean) > 1.0e-12
                or abs(value.posterior_standard_error - expected_se) > 1.0e-12
                or abs(value.lower_confidence_bound - expected_lcb) > 1.0e-12
            ):
                raise ProtocolError("Fold posterior mathematical identity drifted.")
        if canonical_hash(self._unhashed(support, estimates)) != self.posterior_hash:
            raise ProtocolError("Fold posterior hash drifted.")
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "estimates", estimates)

    def _unhashed(
        self,
        support: tuple[str, ...] | None = None,
        estimates: tuple[CandidatePosteriorEstimate, ...] | None = None,
    ) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_fold_posterior_v2",
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "fold_hash": self.fold_hash,
            "global_prior_hash": self.global_prior_hash,
            "support_statistics_surface_hash": self.support_statistics_surface_hash,
            "support_case_ids": list(self.support_case_ids if support is None else support),
            "global_action_id": self.global_action_id,
            "estimates": [
                value.to_payload() for value in (self.estimates if estimates is None else estimates)
            ],
            "config": self.config.to_payload(),
            "support_utility": "pooled_exact_bacc",
            "uncertainty_unit": "paired_whole_case_cluster",
            "per_case_bacc_used": False,
            "evaluation_labels_used": False,
            "shared_model_updated": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "posterior_hash": self.posterior_hash}


FoldPosterior = PooledFoldPosterior


def fit_pooled_fold_posterior(
    fold: CaseFold,
    support_statistics: SufficientStatisticSurface,
    global_prior: PooledLocoPrior,
    *,
    config: PosteriorConfig = PosteriorConfig(),
) -> PooledFoldPosterior:
    if (
        global_prior.target_center != fold.target_center
        or not support_statistics.label_scope.startswith(
            f"fold_local_support::{fold.fold_id}"
        )
        or support_statistics.prerequisite_seal_hash != global_prior.prior_hash
        or set(support_statistics.allowed_case_keys)
        != {(fold.target_center, case) for case in fold.support_case_ids}
    ):
        raise ProtocolError("Fold posterior received a mismatched support capability.")
    if config.variance_floor != global_prior.config.variance_floor:
        raise ProtocolError("Prior and support variance floors must be identical.")
    if config.confidence_multiplier != global_prior.config.confidence_multiplier:
        raise ProtocolError("Prior and posterior confidence multipliers must be identical.")
    estimates: list[CandidatePosteriorEstimate] = []
    for challenger in routing_challengers(fold.target_center, global_prior.global_action_id):
        pairwise_prior = global_prior.pairwise_estimate(challenger)
        contrast = paired_whole_case_cluster_contrast(
            action_rows(
                support_statistics,
                target_center=fold.target_center,
                action_id=challenger,
                case_ids=fold.support_case_ids,
            ),
            action_rows(
                support_statistics,
                target_center=fold.target_center,
                action_id=global_prior.global_action_id,
                case_ids=fold.support_case_ids,
            ),
            variance_floor=config.variance_floor,
        )
        posterior_variance = 1.0 / (
            1.0 / pairwise_prior.prior_variance + 1.0 / contrast.cluster_variance
        )
        posterior_mean = posterior_variance * (
            pairwise_prior.prior_mean / pairwise_prior.prior_variance
            + contrast.pooled_bacc_difference / contrast.cluster_variance
        )
        standard_error = math.sqrt(posterior_variance)
        estimates.append(
            CandidatePosteriorEstimate(
                action_id=challenger,
                reference_action_id=global_prior.global_action_id,
                support_case_count=contrast.case_count,
                support_positive_count=contrast.n_positive,
                support_negative_count=contrast.n_negative,
                prior_mean=pairwise_prior.prior_mean,
                prior_variance=pairwise_prior.prior_variance,
                support_pooled_difference=contrast.pooled_bacc_difference,
                support_cluster_variance=contrast.cluster_variance,
                posterior_mean=posterior_mean,
                posterior_variance=posterior_variance,
                posterior_standard_error=standard_error,
                lower_confidence_bound=(
                    posterior_mean - config.confidence_multiplier * standard_error
                ),
                support_contrast_hash=contrast.contrast_hash,
            )
        )
    canonical = tuple(estimates)
    values = {
        "target_center": fold.target_center,
        "fold_ordinal": fold.fold_ordinal,
        "fold_hash": fold.fold_hash,
        "global_prior_hash": global_prior.prior_hash,
        "support_statistics_surface_hash": support_statistics.statistics_surface_hash,
        "support_case_ids": fold.support_case_ids,
        "global_action_id": global_prior.global_action_id,
        "estimates": canonical,
        "config": config,
    }
    payload = {
        "schema_version": "fixed_bank_pooled_bacc_fold_posterior_v2",
        "target_center": fold.target_center,
        "fold_ordinal": fold.fold_ordinal,
        "fold_hash": fold.fold_hash,
        "global_prior_hash": global_prior.prior_hash,
        "support_statistics_surface_hash": support_statistics.statistics_surface_hash,
        "support_case_ids": list(fold.support_case_ids),
        "global_action_id": global_prior.global_action_id,
        "estimates": [value.to_payload() for value in canonical],
        "config": config.to_payload(),
        "support_utility": "pooled_exact_bacc",
        "uncertainty_unit": "paired_whole_case_cluster",
        "per_case_bacc_used": False,
        "evaluation_labels_used": False,
        "shared_model_updated": False,
    }
    return PooledFoldPosterior(**values, posterior_hash=canonical_hash(payload))


fit_fold_local_posterior = fit_pooled_fold_posterior


__all__ = (
    "CandidatePosteriorEstimate",
    "FoldPosterior",
    "PooledFoldPosterior",
    "PosteriorConfig",
    "fit_fold_local_posterior",
    "fit_pooled_fold_posterior",
)
