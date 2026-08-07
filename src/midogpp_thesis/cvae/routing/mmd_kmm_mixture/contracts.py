"""Fail-closed data contracts for label-free kernel-mean mixture routing."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .config import (
    CANONICAL_GENERATION_SEEDS,
    CANONICAL_TRAINING_SEEDS,
    DEFAULT_MAX_SOURCE_WEIGHT,
    DEFAULT_MIN_EFFECTIVE_SOURCES,
    prior_control_state_hash,
)


DATASET_FAMILY = "MIDOG++"
COMMON_FRAME_SEMANTICS = "common_inverse_virchow2"
PROXY_CLAIM_ROLE = "proxy_compatibility_only"
PROXY_FAMILIES = frozenset(
    {
        "class_prior_controlled_mmd_kmm",
        "class_conditional_contrast_mmd_kmm",
    }
)
SOURCE_EXPERT_TRAINING_ROLE = "source_only_frozen"
TARGET_SUPPORT_ROLE = "disjoint_unlabeled_target_support"
CROSSFIT_COHORT_SUPPORT_ROLE = (
    "cross_fitted_unlabeled_cohort_support_excluding_own_heldout_case"
)
SOURCE_GENERATION_ROLE = "frozen_prior_generation"
KERNEL_MAP_FIT_ROLE = "target_excluded_candidate_pool_generated_common_frame"
KERNEL_TRANSFORM_ROLE = "shared_frozen_source_pool_nystroem"
PRIOR_MODEL_FIT_ROLE = "target_excluded_candidate_pool_generated_prior_model"
PRIOR_FIT_POOL_ROLE = "target_excluded_equal_source_class_balanced_generated_pool"
ENERGY_REFERENCE_METHOD = (
    "uniform_anchored_residual_softmax_negative_calibrated_energy_"
    "automatic_max_weight_and_effective_source_constraints"
)


@dataclass(frozen=True)
class MMDKMMProtocol:
    """The only data regime in which this router core may be evaluated."""

    target_center: str
    candidate_sources: tuple[str, ...]
    support_case_ids: tuple[str, ...]
    evaluation_case_ids: tuple[str, ...]
    common_frame_hash: str
    training_seeds: tuple[int, ...] = CANONICAL_TRAINING_SEEDS
    generation_seeds: tuple[int, ...] = CANONICAL_GENERATION_SEEDS
    artifact_dataset_family: str = DATASET_FAMILY
    claim_dataset_family: str = DATASET_FAMILY
    common_frame_semantics: str = COMMON_FRAME_SEMANTICS
    source_expert_training_role: str = SOURCE_EXPERT_TRAINING_ROLE
    target_support_role: str = TARGET_SUPPORT_ROLE
    claim_role: str = PROXY_CLAIM_ROLE
    source_experts_frozen: bool = True
    target_expert_excluded: bool = True
    support_labels_used: bool = False
    evaluation_labels_available_to_router: bool = False
    evaluation_embeddings_available_to_router: bool = False
    cross_fitted_transductive_diagnostic: bool = False
    cohort_evaluation_embeddings_available_for_other_case_routes: bool = False
    heldout_evaluation_embeddings_available_to_own_route: bool = False
    previous_stage90_router_or_utility_inputs_used: bool = False

    def __post_init__(self) -> None:
        target = str(self.target_center)
        candidates = _canonical_ids(self.candidate_sources, "candidate source")
        support_cases = _canonical_ids(self.support_case_ids, "support case")
        evaluation_cases = _canonical_ids(self.evaluation_case_ids, "evaluation case")
        training_seeds = _canonical_seeds(self.training_seeds, "training")
        generation_seeds = _canonical_seeds(self.generation_seeds, "generation")
        if (
            not target
            or target in candidates
            or len(candidates) not in {7, 8}
            or len(support_cases) < 2
            or set(support_cases).intersection(evaluation_cases)
            or not str(self.common_frame_hash)
            or training_seeds != CANONICAL_TRAINING_SEEDS
            or generation_seeds != CANONICAL_GENERATION_SEEDS
            or self.artifact_dataset_family != DATASET_FAMILY
            or self.claim_dataset_family != DATASET_FAMILY
            or self.common_frame_semantics != COMMON_FRAME_SEMANTICS
            or self.source_expert_training_role != SOURCE_EXPERT_TRAINING_ROLE
            or self.target_support_role
            not in {TARGET_SUPPORT_ROLE, CROSSFIT_COHORT_SUPPORT_ROLE}
            or self.claim_role != PROXY_CLAIM_ROLE
            or self.source_experts_frozen is not True
            or self.target_expert_excluded is not True
            or self.support_labels_used is not False
            or self.evaluation_labels_available_to_router is not False
            or not self._embedding_access_is_fenced()
            or self.previous_stage90_router_or_utility_inputs_used is not False
        ):
            raise ProtocolError("MMD/KMM protocol violates the routing claim firewall.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "candidate_sources", candidates)
        object.__setattr__(self, "support_case_ids", support_cases)
        object.__setattr__(self, "evaluation_case_ids", evaluation_cases)
        object.__setattr__(self, "training_seeds", training_seeds)
        object.__setattr__(self, "generation_seeds", generation_seeds)

    def _embedding_access_is_fenced(self) -> bool:
        ordinary = (
            self.target_support_role == TARGET_SUPPORT_ROLE
            and self.evaluation_embeddings_available_to_router is False
            and self.cross_fitted_transductive_diagnostic is False
            and self.cohort_evaluation_embeddings_available_for_other_case_routes
            is False
            and self.heldout_evaluation_embeddings_available_to_own_route is False
        )
        cross_fitted = (
            self.target_support_role == CROSSFIT_COHORT_SUPPORT_ROLE
            and self.evaluation_embeddings_available_to_router is True
            and self.cross_fitted_transductive_diagnostic is True
            and self.cohort_evaluation_embeddings_available_for_other_case_routes
            is True
            and self.heldout_evaluation_embeddings_available_to_own_route is False
        )
        return ordinary or cross_fitted

    @property
    def support_partition_hash(self) -> str:
        values = (
            self.target_center,
            *self.support_case_ids,
            "evaluation_partition",
            *self.evaluation_case_ids,
        )
        if self.cross_fitted_transductive_diagnostic:
            values = (
                *values,
                self.target_support_role,
                "cohort_evaluation_embeddings_for_other_case_routes",
                "heldout_evaluation_embeddings_excluded_from_own_route",
            )
        return _text_sha256(*values)


@dataclass(frozen=True)
class EnergyDirectionReference:
    """Provenance-bound action from the prior label-free energy path."""

    target_center: str
    candidate_sources: tuple[str, ...]
    support_partition_hash: str
    common_frame_hash: str
    preprocessing_hash: str
    candidate_pool_fit_hash: str
    kernel_map_hash: str
    training_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    weights: Mapping[str, float]
    energy_calibration_hash: str
    action_id: str
    method_id: str = ENERGY_REFERENCE_METHOD
    target_expert_excluded: bool = True
    all_retained_seeds_aggregated: bool = True
    seed_selection_used: bool = False
    support_labels_used: bool = False
    target_labels_used: bool = False
    evaluation_embeddings_used: bool = False
    stage90_utility_used: bool = False

    def __post_init__(self) -> None:
        target = str(self.target_center)
        sources = _canonical_ids(self.candidate_sources, "energy-reference source")
        training = _canonical_seeds(self.training_seeds, "energy-reference training")
        generation = _canonical_seeds(
            self.generation_seeds, "energy-reference generation"
        )
        weights = _validated_weight_mapping(
            self.weights, sources, "energy-reference"
        )
        vector = np.asarray([weights[source] for source in sources])
        if (
            not target
            or target in sources
            or len(sources) not in {7, 8}
            or training != CANONICAL_TRAINING_SEEDS
            or generation != CANONICAL_GENERATION_SEEDS
            or not str(self.support_partition_hash)
            or not str(self.common_frame_hash)
            or not str(self.preprocessing_hash)
            or not str(self.candidate_pool_fit_hash)
            or not str(self.kernel_map_hash)
            or not str(self.energy_calibration_hash)
            or self.action_id not in {"rho_0.25", "rho_0.50"}
            or self.method_id != ENERGY_REFERENCE_METHOD
            or float(vector.max()) > DEFAULT_MAX_SOURCE_WEIGHT + 1e-8
            or 1.0 / float(np.dot(vector, vector))
            < DEFAULT_MIN_EFFECTIVE_SOURCES - 1e-7
            or self.target_expert_excluded is not True
            or self.all_retained_seeds_aggregated is not True
            or self.seed_selection_used is not False
            or self.support_labels_used is not False
            or self.target_labels_used is not False
            or self.evaluation_embeddings_used is not False
            or self.stage90_utility_used is not False
        ):
            raise ProtocolError("Energy-direction reference contract is invalid.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "training_seeds", training)
        object.__setattr__(self, "generation_seeds", generation)
        object.__setattr__(self, "weights", MappingProxyType(weights))


@dataclass(frozen=True)
class TransformedKernelFeatures:
    """Rows transformed by one target-excluded, source-pool-fitted map."""

    values: np.ndarray
    common_frame_hash: str
    preprocessing_hash: str
    candidate_pool_fit_hash: str
    kernel_map_hash: str
    map_fit_role: str = KERNEL_MAP_FIT_ROLE
    transform_role: str = KERNEL_TRANSFORM_ROLE
    target_rows_used_to_fit: bool = False
    evaluation_rows_used_to_fit: bool = False
    values_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        values = readonly_matrix(self.values, "transformed kernel features")
        if (
            not str(self.common_frame_hash)
            or not str(self.preprocessing_hash)
            or not str(self.candidate_pool_fit_hash)
            or not str(self.kernel_map_hash)
            or self.map_fit_role != KERNEL_MAP_FIT_ROLE
            or self.transform_role != KERNEL_TRANSFORM_ROLE
            or self.target_rows_used_to_fit is not False
            or self.evaluation_rows_used_to_fit is not False
        ):
            raise ProtocolError("Transformed kernel-feature provenance is invalid.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "values_sha256", _array_sha256(values))


@dataclass(frozen=True)
class SourceKernelReplica:
    """One generated class block after the shared source-only kernel map."""

    source_center: str
    training_seed: int
    generation_seed: int
    class_label: int
    kernel_features: TransformedKernelFeatures
    source_expert_training_role: str = SOURCE_EXPERT_TRAINING_ROLE
    generation_role: str = SOURCE_GENERATION_ROLE

    def __post_init__(self) -> None:
        features = self.kernel_features
        if (
            not isinstance(features, TransformedKernelFeatures)
            or not str(self.source_center)
            or isinstance(self.training_seed, bool)
            or isinstance(self.generation_seed, bool)
            or isinstance(self.class_label, bool)
            or int(self.class_label) not in {0, 1}
            or self.source_expert_training_role != SOURCE_EXPERT_TRAINING_ROLE
            or self.generation_role != SOURCE_GENERATION_ROLE
        ):
            raise ProtocolError("Source kernel replica contract is invalid.")
        object.__setattr__(self, "source_center", str(self.source_center))
        object.__setattr__(self, "training_seed", int(self.training_seed))
        object.__setattr__(self, "generation_seed", int(self.generation_seed))
        object.__setattr__(self, "class_label", int(self.class_label))
        object.__setattr__(self, "kernel_features", features)


@dataclass(frozen=True)
class TargetSupportKernelFeatures:
    """Unlabeled target support represented by source-only-fitted mechanisms."""

    target_center: str
    case_ids: tuple[str, ...]
    kernel_features: TransformedKernelFeatures
    prior_prediction: SourceOnlyPriorPrediction
    support_labels_used: bool = False
    evaluation_embeddings_used: bool = False
    cross_fitted_transductive_support: bool = False
    cohort_evaluation_embeddings_used: bool = False
    heldout_evaluation_embeddings_used: bool = False

    def __post_init__(self) -> None:
        features = self.kernel_features
        prediction = self.prior_prediction
        cases = tuple(str(value) for value in self.case_ids)
        if (
            not isinstance(features, TransformedKernelFeatures)
            or not isinstance(prediction, SourceOnlyPriorPrediction)
            or not str(self.target_center)
            or len(cases) != len(features.values)
            or not cases
            or any(not value for value in cases)
            or features.values.shape[1] <= 0
            or prediction.probabilities.shape != (len(features.values), 2)
            or prediction.target_center != str(self.target_center)
            or prediction.common_frame_hash != features.common_frame_hash
            or self.support_labels_used is not False
            or not self._embedding_use_is_fenced()
        ):
            raise ProtocolError("Target support kernel-feature contract is invalid.")
        object.__setattr__(self, "target_center", str(self.target_center))
        object.__setattr__(self, "case_ids", cases)

    def _embedding_use_is_fenced(self) -> bool:
        ordinary = (
            self.evaluation_embeddings_used is False
            and self.cross_fitted_transductive_support is False
            and self.cohort_evaluation_embeddings_used is False
            and self.heldout_evaluation_embeddings_used is False
        )
        cross_fitted = (
            self.evaluation_embeddings_used is True
            and self.cross_fitted_transductive_support is True
            and self.cohort_evaluation_embeddings_used is True
            and self.heldout_evaluation_embeddings_used is False
        )
        return ordinary or cross_fitted

    @property
    def soft_class_probabilities(self) -> np.ndarray:
        return self.prior_prediction.probabilities

    @property
    def common_frame_hash(self) -> str:
        return self.kernel_features.common_frame_hash

    @property
    def kernel_map_hash(self) -> str:
        return self.kernel_features.kernel_map_hash

    @property
    def prior_state_hash(self) -> str:
        return self.prior_prediction.prior_state_hash


@dataclass(frozen=True)
class KernelMeanProblem:
    """Finite-dimensional squared-MMD proxy objective."""

    protocol: MMDKMMProtocol
    candidate_sources: tuple[str, ...]
    source_kernel_means: np.ndarray
    target_kernel_mean: np.ndarray
    common_frame_hash: str
    kernel_map_hash: str
    preprocessing_hash: str
    candidate_pool_fit_hash: str
    kernel_transform_role: str
    prior_family_hash: str
    prior_control_hash: str
    prior_state_hash: str
    prior_sensitivity_positive_prior: float | None
    target_kernel_feature_sha256: str
    target_responsibility_sha256: str
    source_replica_count: int
    target_support_row_count: int
    proxy_family: str = "class_prior_controlled_mmd_kmm"
    claim_role: str = PROXY_CLAIM_ROLE

    def __post_init__(self) -> None:
        sources = _canonical_ids(self.candidate_sources, "kernel-mean source")
        source_means = readonly_matrix(self.source_kernel_means, "source kernel means")
        target_mean = readonly_vector(self.target_kernel_mean, "target kernel mean")
        if (
            sources != self.protocol.candidate_sources
            or source_means.shape != (len(sources), len(target_mean))
            or self.common_frame_hash != self.protocol.common_frame_hash
            or not str(self.kernel_map_hash)
            or not str(self.preprocessing_hash)
            or not str(self.candidate_pool_fit_hash)
            or self.kernel_transform_role != KERNEL_TRANSFORM_ROLE
            or not str(self.prior_family_hash)
            or not str(self.prior_control_hash)
            or not str(self.prior_state_hash)
            or len(str(self.target_kernel_feature_sha256)) != 64
            or len(str(self.target_responsibility_sha256)) != 64
            or (
                self.prior_sensitivity_positive_prior is not None
                and (
                    not math.isfinite(float(self.prior_sensitivity_positive_prior))
                    or not 0.0 < float(self.prior_sensitivity_positive_prior) < 1.0
                )
            )
            or isinstance(self.source_replica_count, bool)
            or int(self.source_replica_count) <= 0
            or isinstance(self.target_support_row_count, bool)
            or int(self.target_support_row_count) <= 0
            or self.proxy_family not in PROXY_FAMILIES
            or self.claim_role != PROXY_CLAIM_ROLE
        ):
            raise ProtocolError("Kernel-mean routing problem contract is invalid.")
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "source_kernel_means", source_means)
        object.__setattr__(self, "target_kernel_mean", target_mean)
        object.__setattr__(self, "source_replica_count", int(self.source_replica_count))
        object.__setattr__(
            self,
            "prior_sensitivity_positive_prior",
            None
            if self.prior_sensitivity_positive_prior is None
            else float(self.prior_sensitivity_positive_prior),
        )
        object.__setattr__(
            self, "target_support_row_count", int(self.target_support_row_count)
        )


@dataclass(frozen=True)
class SourceOnlyPriorPrediction:
    target_center: str
    candidate_sources: tuple[str, ...]
    common_frame_hash: str
    probabilities: np.ndarray
    prior_model_hash: str
    prior_fit_pool_hash: str
    temperature: float
    probability_clip: float
    sensitivity_positive_priors: tuple[float, ...]
    reference_positive_prior: float = 0.5
    sensitivity_positive_prior: float | None = None
    fit_role: str = PRIOR_MODEL_FIT_ROLE
    fit_pool_role: str = PRIOR_FIT_POOL_ROLE
    target_labels_used: bool = False
    target_center_excluded_from_fit: bool = True
    equal_source_class_balanced_fit_pool: bool = True
    responsibility_sha256: str = field(init=False)
    prior_control_hash: str = field(init=False)
    prior_family_hash: str = field(init=False)
    prior_state_hash: str = field(init=False)

    def __post_init__(self) -> None:
        probabilities = readonly_probabilities(self.probabilities)
        target = str(self.target_center)
        candidates = _canonical_ids(self.candidate_sources, "prior candidate source")
        sensitivity = self.sensitivity_positive_prior
        sensitivity_grid = tuple(
            sorted(float(value) for value in self.sensitivity_positive_priors)
        )
        if (
            not target
            or target in candidates
            or len(candidates) not in {7, 8}
            or not str(self.common_frame_hash)
            or not str(self.prior_model_hash)
            or not str(self.prior_fit_pool_hash)
            or not math.isfinite(float(self.temperature))
            or float(self.temperature) <= 0.0
            or not math.isfinite(float(self.probability_clip))
            or not 0.0 < float(self.probability_clip) < 0.5
            or not 0.0 < float(self.reference_positive_prior) < 1.0
            or len(sensitivity_grid) < 2
            or len(set(sensitivity_grid)) != len(sensitivity_grid)
            or any(not 0.0 < value < 1.0 for value in sensitivity_grid)
            or not min(sensitivity_grid)
            < float(self.reference_positive_prior)
            < max(sensitivity_grid)
            or (
                sensitivity is not None
                and (
                    not math.isfinite(float(sensitivity))
                    or not 0.0 < float(sensitivity) < 1.0
                )
            )
            or self.fit_role != PRIOR_MODEL_FIT_ROLE
            or self.fit_pool_role != PRIOR_FIT_POOL_ROLE
            or self.target_labels_used is not False
            or self.target_center_excluded_from_fit is not True
            or self.equal_source_class_balanced_fit_pool is not True
        ):
            raise ProtocolError("Source-only prior prediction contract is invalid.")
        responsibility_hash = _array_sha256(probabilities)
        control_hash = prior_control_state_hash(
            probability_clip=float(self.probability_clip),
            temperature=float(self.temperature),
            sensitivity_positive_priors=sensitivity_grid,
            reference_positive_prior=float(self.reference_positive_prior),
            fit_role=self.fit_role,
        )
        family_hash = _text_sha256(
            target,
            *candidates,
            str(self.common_frame_hash),
            str(self.prior_model_hash),
            str(self.prior_fit_pool_hash),
            control_hash,
            self.fit_role,
            self.fit_pool_role,
        )
        state_hash = _text_sha256(
            family_hash,
            "none" if sensitivity is None else format(float(sensitivity), ".17g"),
        )
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "candidate_sources", candidates)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "probability_clip", float(self.probability_clip))
        object.__setattr__(self, "sensitivity_positive_priors", sensitivity_grid)
        object.__setattr__(
            self, "reference_positive_prior", float(self.reference_positive_prior)
        )
        object.__setattr__(
            self,
            "sensitivity_positive_prior",
            None if sensitivity is None else float(sensitivity),
        )
        object.__setattr__(self, "responsibility_sha256", responsibility_hash)
        object.__setattr__(self, "prior_control_hash", control_hash)
        object.__setattr__(self, "prior_family_hash", family_hash)
        object.__setattr__(self, "prior_state_hash", state_hash)


@dataclass(frozen=True)
class FrozenNystroemFeatureMap:
    components: np.ndarray
    normalization: np.ndarray
    gamma: float
    common_frame_hash: str
    preprocessing_hash: str
    candidate_pool_fit_hash: str
    random_state: int
    fit_role: str = KERNEL_MAP_FIT_ROLE
    kernel_map_hash: str = field(init=False)

    def __post_init__(self) -> None:
        components = readonly_matrix(self.components, "Nyström components")
        normalization = readonly_matrix(self.normalization, "Nyström normalization")
        if (
            normalization.shape != (len(components), len(components))
            or not np.isfinite(float(self.gamma))
            or float(self.gamma) <= 0.0
            or not str(self.common_frame_hash)
            or not str(self.preprocessing_hash)
            or not str(self.candidate_pool_fit_hash)
            or isinstance(self.random_state, bool)
            or self.fit_role != KERNEL_MAP_FIT_ROLE
        ):
            raise ProtocolError("Frozen Nyström map contract is invalid.")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "gamma", float(self.gamma))
        object.__setattr__(self, "random_state", int(self.random_state))
        object.__setattr__(
            self,
            "kernel_map_hash",
            _text_sha256(
                _array_sha256(components),
                _array_sha256(normalization),
                format(float(self.gamma), ".17g"),
                str(self.common_frame_hash),
                str(self.preprocessing_hash),
                str(self.candidate_pool_fit_hash),
                str(int(self.random_state)),
                self.fit_role,
            ),
        )


@dataclass(frozen=True)
class KMMWeightSolution:
    candidate_sources: tuple[str, ...]
    uniform_weights: Mapping[str, float]
    weights: Mapping[str, float]
    delta: Mapping[str, float]
    proxy_objective: float
    uniform_proxy_objective: float
    proxy_improvement: float
    mmd_squared: float
    uniform_mmd_squared: float
    regularization_value: float
    effective_source_count: float
    maximum_source_weight: float
    used_uniform_fallback: bool
    fallback_reason: str | None
    solver_success: bool
    solver_message: str
    solver_iterations: int
    solver_method: str
    solver_version: str
    optimality_residual: float | None
    claim_role: str = PROXY_CLAIM_ROLE
    downstream_utility_claimed: bool = False

    def __post_init__(self) -> None:
        sources = _canonical_ids(self.candidate_sources, "solution source")
        uniform = _validated_weight_mapping(
            self.uniform_weights, sources, "uniform solution"
        )
        weights = _validated_weight_mapping(self.weights, sources, "KMM solution")
        delta = _validated_delta_mapping(self.delta, sources)
        uniform_vector = np.asarray([uniform[source] for source in sources])
        weight_vector = np.asarray([weights[source] for source in sources])
        delta_vector = np.asarray([delta[source] for source in sources])
        numeric = (
            self.proxy_objective,
            self.uniform_proxy_objective,
            self.proxy_improvement,
            self.mmd_squared,
            self.uniform_mmd_squared,
            self.regularization_value,
            self.effective_source_count,
            self.maximum_source_weight,
        )
        if (
            not np.isfinite(np.asarray(numeric, dtype=np.float64)).all()
            or any(float(value) < -1e-10 for value in numeric[:6])
            or float(self.effective_source_count) <= 0.0
            or float(self.maximum_source_weight) < 0.0
            or not np.allclose(
                uniform_vector,
                np.full(len(sources), 1.0 / float(len(sources))),
                rtol=0.0,
                atol=1e-12,
            )
            or float(self.maximum_source_weight)
            > DEFAULT_MAX_SOURCE_WEIGHT + 1e-8
            or float(self.effective_source_count)
            < DEFAULT_MIN_EFFECTIVE_SOURCES - 1e-7
            or not np.allclose(
                weight_vector - uniform_vector,
                delta_vector,
                rtol=0.0,
                atol=1e-10,
            )
            or not np.isclose(
                float(self.uniform_proxy_objective)
                - float(self.proxy_objective),
                float(self.proxy_improvement),
                rtol=0.0,
                atol=1e-8,
            )
            or not np.isclose(
                float(weight_vector.max()),
                float(self.maximum_source_weight),
                rtol=0.0,
                atol=1e-8,
            )
            or not np.isclose(
                1.0 / float(np.dot(weight_vector, weight_vector)),
                float(self.effective_source_count),
                rtol=0.0,
                atol=1e-8,
            )
            or self.claim_role != PROXY_CLAIM_ROLE
            or self.downstream_utility_claimed is not False
            or self.used_uniform_fallback is not (self.fallback_reason is not None)
            or isinstance(self.solver_iterations, bool)
            or int(self.solver_iterations) < 0
            or self.solver_method != "scipy_slsqp_continuous_convex_proxy"
            or not str(self.solver_version)
            or (
                self.optimality_residual is not None
                and (
                    not math.isfinite(float(self.optimality_residual))
                    or float(self.optimality_residual) < 0.0
                )
            )
            or (not self.used_uniform_fallback and self.optimality_residual is None)
        ):
            raise ProtocolError("MMD/KMM weight-solution contract is invalid.")
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "uniform_weights", MappingProxyType(uniform))
        object.__setattr__(self, "weights", MappingProxyType(weights))
        object.__setattr__(self, "delta", MappingProxyType(delta))
        object.__setattr__(self, "solver_iterations", int(self.solver_iterations))
        object.__setattr__(
            self,
            "optimality_residual",
            None
            if self.optimality_residual is None
            else float(self.optimality_residual),
        )


@dataclass(frozen=True)
class StabilityAudit:
    axis: str
    variant_ids: tuple[str, ...]
    maximum_l1_distance: float
    minimum_direction_cosine: float
    passed: bool
    failure_reason: str | None

    def __post_init__(self) -> None:
        variants = tuple(sorted(str(value) for value in self.variant_ids))
        if (
            not str(self.axis)
            or any(not value for value in variants)
            or len(set(variants)) != len(variants)
            or (self.passed and not variants)
            or np.isnan(float(self.maximum_l1_distance))
            or np.isnan(float(self.minimum_direction_cosine))
            or self.passed is not (self.failure_reason is None)
        ):
            raise ProtocolError("MMD/KMM stability-audit contract is invalid.")
        object.__setattr__(self, "axis", str(self.axis))
        object.__setattr__(self, "variant_ids", variants)


@dataclass(frozen=True)
class DirectionIdentityAudit:
    reference_role: str
    direction_cosine: float
    weight_l1_distance: float
    duplicate: bool

    def __post_init__(self) -> None:
        if (
            not str(self.reference_role)
            or not math.isfinite(float(self.direction_cosine))
            or not -1.0 - 1e-10 <= float(self.direction_cosine) <= 1.0 + 1e-10
            or not math.isfinite(float(self.weight_l1_distance))
            or float(self.weight_l1_distance) < 0.0
        ):
            raise ProtocolError("MMD/KMM direction-identity contract is invalid.")


@dataclass(frozen=True)
class KMMRouteDecision:
    candidate_sources: tuple[str, ...]
    base_solution: KMMWeightSolution
    final_weights: Mapping[str, float]
    used_uniform_fallback: bool
    fallback_reason: str | None
    stability_audits: tuple[StabilityAudit, ...]
    direction_identity: DirectionIdentityAudit
    claim_role: str = PROXY_CLAIM_ROLE
    downstream_utility_claimed: bool = False
    promotion_eligible: bool = False
    target_labels_used: bool = False
    previous_stage90_router_or_utility_inputs_used: bool = False

    def __post_init__(self) -> None:
        sources = _canonical_ids(self.candidate_sources, "route-decision source")
        weights = _validated_weight_mapping(
            self.final_weights, sources, "route-decision"
        )
        weight_vector = np.asarray([weights[source] for source in sources])
        if (
            sources != self.base_solution.candidate_sources
            or not self.stability_audits
            or self.claim_role != PROXY_CLAIM_ROLE
            or self.downstream_utility_claimed is not False
            or self.promotion_eligible is not False
            or self.target_labels_used is not False
            or self.previous_stage90_router_or_utility_inputs_used is not False
            or self.used_uniform_fallback is not (self.fallback_reason is not None)
            or float(weight_vector.max()) > DEFAULT_MAX_SOURCE_WEIGHT + 1e-8
            or 1.0 / float(np.dot(weight_vector, weight_vector))
            < DEFAULT_MIN_EFFECTIVE_SOURCES - 1e-7
        ):
            raise ProtocolError("MMD/KMM route-decision contract is invalid.")
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "final_weights", MappingProxyType(weights))


def readonly_matrix(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ProtocolError(f"{name} must be a nonempty finite matrix.")
    output = np.array(array, dtype=np.float64, copy=True)
    output.setflags(write=False)
    return output


def readonly_vector(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ProtocolError(f"{name} must be a nonempty finite vector.")
    output = np.array(array, dtype=np.float64, copy=True)
    output.setflags(write=False)
    return output


def readonly_probabilities(value: object) -> np.ndarray:
    array = readonly_matrix(value, "soft class probabilities")
    if (
        array.shape[1] != 2
        or np.any(array <= 0.0)
        or np.any(array >= 1.0)
        or not np.allclose(array.sum(axis=1), 1.0, rtol=0.0, atol=1e-10)
    ):
        raise ProtocolError("Soft class probabilities must be strict binary rows.")
    return array


def weight_mapping(sources: Sequence[str], values: np.ndarray) -> dict[str, float]:
    return {
        str(source): float(value)
        for source, value in zip(sources, values, strict=True)
    }


def _array_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _text_sha256(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validated_weight_mapping(
    values: Mapping[str, float], sources: tuple[str, ...], name: str
) -> dict[str, float]:
    normalized = {str(key): float(value) for key, value in values.items()}
    vector = np.asarray([normalized.get(source, np.nan) for source in sources])
    if (
        set(normalized) != set(sources)
        or not np.isfinite(vector).all()
        or np.any(vector < -1e-10)
        or not np.isclose(vector.sum(), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ProtocolError(f"MMD/KMM {name} weights violate the simplex.")
    return {source: float(normalized[source]) for source in sources}


def _validated_delta_mapping(
    values: Mapping[str, float], sources: tuple[str, ...]
) -> dict[str, float]:
    normalized = {str(key): float(value) for key, value in values.items()}
    vector = np.asarray([normalized.get(source, np.nan) for source in sources])
    if (
        set(normalized) != set(sources)
        or not np.isfinite(vector).all()
        or not np.isclose(vector.sum(), 0.0, rtol=0.0, atol=1e-8)
    ):
        raise ProtocolError("MMD/KMM weight deltas must be finite and sum to zero.")
    return {source: float(normalized[source]) for source in sources}


def _canonical_ids(values: Sequence[object], name: str) -> tuple[str, ...]:
    parsed = tuple(str(value) for value in values)
    if (
        not parsed
        or any(not value for value in parsed)
        or len(set(parsed)) != len(parsed)
    ):
        raise ProtocolError(f"MMD/KMM {name} identifiers must be unique and nonempty.")
    return tuple(sorted(parsed))


def _canonical_seeds(values: Sequence[object], name: str) -> tuple[int, ...]:
    if any(isinstance(value, bool) for value in values):
        raise ProtocolError(f"MMD/KMM {name} seeds must be integers, not booleans.")
    parsed_values: list[int] = []
    for value in values:
        parsed = int(value)
        if isinstance(value, (float, np.floating)) and float(value) != float(parsed):
            raise ProtocolError(f"MMD/KMM {name} seeds must be exact integers.")
        parsed_values.append(parsed)
    parsed = tuple(sorted(parsed_values))
    if not parsed or len(set(parsed)) != len(parsed):
        raise ProtocolError(f"MMD/KMM {name} seeds must be unique and nonempty.")
    return parsed


__all__ = (
    "COMMON_FRAME_SEMANTICS",
    "DATASET_FAMILY",
    "DirectionIdentityAudit",
    "ENERGY_REFERENCE_METHOD",
    "EnergyDirectionReference",
    "FrozenNystroemFeatureMap",
    "KMMRouteDecision",
    "KMMWeightSolution",
    "KernelMeanProblem",
    "MMDKMMProtocol",
    "PROXY_CLAIM_ROLE",
    "SOURCE_EXPERT_TRAINING_ROLE",
    "SOURCE_GENERATION_ROLE",
    "SourceKernelReplica",
    "SourceOnlyPriorPrediction",
    "StabilityAudit",
    "TARGET_SUPPORT_ROLE",
    "TargetSupportKernelFeatures",
    "TransformedKernelFeatures",
    "readonly_matrix",
    "readonly_probabilities",
    "readonly_vector",
    "weight_mapping",
)
