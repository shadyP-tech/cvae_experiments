"""Frozen numerical contracts for the label-free MMD/KMM mixture router.

These classes configure pure proxy mathematics only.  They deliberately carry
no artifact roots, validation manifests, target labels, or workspace bindings.
An experiment may freeze them only after an eligible unconsumed routing surface
has been authorized.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

from ...protocol import ProtocolError


CANONICAL_TRAINING_SEEDS = (17, 42, 101)
CANONICAL_GENERATION_SEEDS = (17, 42, 101)
CLASS_LABELS = (0, 1)
DEFAULT_MAX_SOURCE_WEIGHT = 0.25
DEFAULT_MIN_EFFECTIVE_SOURCES = 6.0


@dataclass(frozen=True)
class PriorControlConfig:
    """Source-only soft class-prior correction for unlabeled target support."""

    probability_clip: float
    temperature: float
    sensitivity_positive_priors: tuple[float, ...]
    reference_positive_prior: float = 0.5
    fit_role: str = "target_excluded_candidate_pool_generated_prior_model"

    def __post_init__(self) -> None:
        clip = float(self.probability_clip)
        temperature = float(self.temperature)
        reference = float(self.reference_positive_prior)
        priors = tuple(
            sorted(float(value) for value in self.sensitivity_positive_priors)
        )
        if (
            self.fit_role
            != "target_excluded_candidate_pool_generated_prior_model"
            or not 0.0 < clip < 0.5
            or not _positive_finite(temperature)
            or not 0.0 < reference < 1.0
            or len(priors) < 2
            or len(set(priors)) != len(priors)
            or any(not 0.0 < value < 1.0 for value in priors)
            or any(abs(value - reference) <= 1e-15 for value in priors)
            or not min(priors) < reference < max(priors)
        ):
            raise ProtocolError("MMD/KMM class-prior control is invalid.")
        object.__setattr__(self, "probability_clip", clip)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "reference_positive_prior", reference)
        object.__setattr__(self, "sensitivity_positive_priors", priors)

    @property
    def state_hash(self) -> str:
        return prior_control_state_hash(
            probability_clip=self.probability_clip,
            temperature=self.temperature,
            sensitivity_positive_priors=self.sensitivity_positive_priors,
            reference_positive_prior=self.reference_positive_prior,
            fit_role=self.fit_role,
        )


@dataclass(frozen=True)
class KMMOptimizationConfig:
    """Convex proxy objective and dense-simplex constraints."""

    regularization: float
    minimum_proxy_improvement: float
    max_source_weight: float = DEFAULT_MAX_SOURCE_WEIGHT
    minimum_effective_sources: float = DEFAULT_MIN_EFFECTIVE_SOURCES
    solver_tolerance: float = 1e-12
    optimality_tolerance: float = 1e-6
    max_iterations: int = 2000

    def __post_init__(self) -> None:
        if (
            not _positive_finite(self.regularization)
            or not _nonnegative_finite(self.minimum_proxy_improvement)
            or not math.isclose(
                float(self.max_source_weight),
                DEFAULT_MAX_SOURCE_WEIGHT,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(self.minimum_effective_sources),
                DEFAULT_MIN_EFFECTIVE_SOURCES,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not _positive_finite(self.solver_tolerance)
            or not _positive_finite(self.optimality_tolerance)
            or isinstance(self.max_iterations, bool)
            or int(self.max_iterations) <= 0
        ):
            raise ProtocolError("MMD/KMM optimizer configuration is invalid.")


@dataclass(frozen=True)
class KMMGateConfig:
    """Predeclared proxy-stability and duplicate-direction stop rules."""

    maximum_support_l1: float
    maximum_training_seed_l1: float
    maximum_generation_seed_l1: float
    maximum_prior_sensitivity_l1: float
    minimum_direction_cosine: float
    duplicate_direction_cosine: float
    duplicate_weight_l1: float

    def __post_init__(self) -> None:
        distances = (
            self.maximum_support_l1,
            self.maximum_training_seed_l1,
            self.maximum_generation_seed_l1,
            self.maximum_prior_sensitivity_l1,
            self.duplicate_weight_l1,
        )
        if (
            any(not _nonnegative_finite(value) for value in distances)
            or not -1.0 <= float(self.minimum_direction_cosine) <= 1.0
            or not -1.0 <= float(self.duplicate_direction_cosine) <= 1.0
        ):
            raise ProtocolError("MMD/KMM stability-gate configuration is invalid.")


def _positive_finite(value: object) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return parsed > 0.0 and math.isfinite(parsed)


def _nonnegative_finite(value: object) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return parsed >= 0.0 and math.isfinite(parsed)


def prior_control_state_hash(
    *,
    probability_clip: float,
    temperature: float,
    sensitivity_positive_priors: tuple[float, ...],
    reference_positive_prior: float,
    fit_role: str,
) -> str:
    values = (
        format(float(probability_clip), ".17g"),
        format(float(temperature), ".17g"),
        *(format(float(value), ".17g") for value in sensitivity_positive_priors),
        format(float(reference_positive_prior), ".17g"),
        str(fit_role),
    )
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = (
    "CANONICAL_GENERATION_SEEDS",
    "CANONICAL_TRAINING_SEEDS",
    "CLASS_LABELS",
    "DEFAULT_MAX_SOURCE_WEIGHT",
    "DEFAULT_MIN_EFFECTIVE_SOURCES",
    "KMMGateConfig",
    "KMMOptimizationConfig",
    "PriorControlConfig",
    "prior_control_state_hash",
)
