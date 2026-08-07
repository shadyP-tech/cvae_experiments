"""Class-, case-, and replica-equal kernel means for mixture routing."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .config import PriorControlConfig
from .contracts import (
    KernelMeanProblem,
    MMDKMMProtocol,
    SourceKernelReplica,
    TargetSupportKernelFeatures,
    readonly_probabilities,
)
from .prior import shift_source_only_prior_prediction


def case_equal_class_balanced_kernel_mean(
    kernel_features: object,
    soft_class_probabilities: object,
    case_ids: Sequence[object],
) -> np.ndarray:
    """Average soft class-conditional means equally over classes and cases."""

    features = np.asarray(kernel_features, dtype=np.float64)
    probabilities = readonly_probabilities(soft_class_probabilities)
    cases = np.asarray([str(value) for value in case_ids], dtype=object)
    if (
        features.ndim != 2
        or not features.size
        or not np.isfinite(features).all()
        or len(features) != len(probabilities)
        or len(features) != len(cases)
        or any(not value for value in cases.tolist())
    ):
        raise ProtocolError("Target support kernel-mean inputs do not align.")
    per_case: list[np.ndarray] = []
    for case_id in sorted(set(cases.tolist())):
        mask = cases == case_id
        class_means: list[np.ndarray] = []
        for label in (0, 1):
            responsibilities = probabilities[mask, label]
            mass = float(responsibilities.sum())
            if mass <= 0.0 or not np.isfinite(mass):
                raise ProtocolError("Target support case has zero soft class mass.")
            class_means.append(
                np.sum(features[mask] * responsibilities[:, None], axis=0) / mass
            )
        per_case.append(0.5 * (class_means[0] + class_means[1]))
    output = np.mean(np.asarray(per_case, dtype=np.float64), axis=0)
    if output.ndim != 1 or not np.isfinite(output).all():
        raise ProtocolError("Target support kernel mean is invalid.")
    output.setflags(write=False)
    return output


def build_kernel_mean_problem(
    protocol: MMDKMMProtocol,
    source_replicas: Sequence[SourceKernelReplica],
    target_support: TargetSupportKernelFeatures,
    *,
    training_seeds: Sequence[int] | None = None,
    generation_seeds: Sequence[int] | None = None,
    require_complete_target_support: bool = True,
) -> KernelMeanProblem:
    """Build one squared-MMD problem with equal class and replica mass."""

    train = _selected_seeds(training_seeds, protocol.training_seeds, "training")
    generation = _selected_seeds(
        generation_seeds, protocol.generation_seeds, "generation"
    )
    replicas = tuple(source_replicas)
    if not replicas:
        raise ProtocolError("MMD/KMM source replica library is empty.")
    if target_support.target_center != protocol.target_center:
        raise ProtocolError("Target support center does not match the routing target.")
    if target_support.common_frame_hash != protocol.common_frame_hash:
        raise ProtocolError("Target support is outside the frozen common frame.")
    if (
        target_support.prior_prediction.target_center != protocol.target_center
        or target_support.prior_prediction.candidate_sources
        != protocol.candidate_sources
        or target_support.prior_prediction.common_frame_hash
        != protocol.common_frame_hash
    ):
        raise ProtocolError("Target support prior crossed the frozen routing family.")
    if (
        target_support.evaluation_embeddings_used
        is not protocol.evaluation_embeddings_available_to_router
        or target_support.cross_fitted_transductive_support
        is not protocol.cross_fitted_transductive_diagnostic
        or target_support.cohort_evaluation_embeddings_used
        is not protocol.cohort_evaluation_embeddings_available_for_other_case_routes
        or target_support.heldout_evaluation_embeddings_used
        is not protocol.heldout_evaluation_embeddings_available_to_own_route
    ):
        raise ProtocolError(
            "Target support embedding use differs from the routing protocol."
        )
    target_cases = set(target_support.case_ids)
    declared_cases = set(protocol.support_case_ids)
    if (
        not target_cases.issubset(declared_cases)
        or (require_complete_target_support and target_cases != declared_cases)
        or target_cases.intersection(protocol.evaluation_case_ids)
    ):
        raise ProtocolError("Target support cases violate the frozen partition.")

    by_key: dict[tuple[str, int, int, int], SourceKernelReplica] = {}
    kernel_hash = target_support.kernel_map_hash
    target_feature_provenance = _feature_provenance(target_support.kernel_features)
    dimension = target_support.kernel_features.values.shape[1]
    for replica in replicas:
        if (
            replica.training_seed not in protocol.training_seeds
            or replica.generation_seed not in protocol.generation_seeds
        ):
            raise ProtocolError("MMD/KMM source replica used an undeclared seed.")
        if (
            replica.training_seed not in train
            or replica.generation_seed not in generation
        ):
            continue
        key = (
            replica.source_center,
            replica.training_seed,
            replica.generation_seed,
            replica.class_label,
        )
        if key in by_key:
            raise ProtocolError("MMD/KMM source replica key is duplicated.")
        if (
            replica.source_center not in protocol.candidate_sources
            or replica.source_center == protocol.target_center
            or _feature_provenance(replica.kernel_features)
            != target_feature_provenance
            or replica.kernel_features.values.shape[1] != dimension
        ):
            raise ProtocolError("MMD/KMM source replica crossed a protocol boundary.")
        by_key[key] = replica
    expected = {
        (source, training_seed, generation_seed, label)
        for source, training_seed, generation_seed, label in product(
            protocol.candidate_sources,
            train,
            generation,
            (0, 1),
        )
    }
    if set(by_key) != expected:
        raise ProtocolError(
            "MMD/KMM requires the complete candidate x training x generation "
            "x class grid."
        )
    replica_row_counts = {
        len(replica.kernel_features.values) for replica in by_key.values()
    }
    if len(replica_row_counts) != 1:
        raise ProtocolError(
            "MMD/KMM source replicas must use one equal per-class sample budget."
        )

    source_means: list[np.ndarray] = []
    for source in protocol.candidate_sources:
        replica_means = [
            np.asarray(
                by_key[(source, training_seed, generation_seed, label)]
                .kernel_features.values
            )
            .mean(axis=0)
            for training_seed, generation_seed, label in product(
                train, generation, (0, 1)
            )
        ]
        source_means.append(np.mean(np.asarray(replica_means), axis=0))
    source_matrix = np.asarray(source_means, dtype=np.float64)
    target_mean = case_equal_class_balanced_kernel_mean(
        target_support.kernel_features.values,
        target_support.soft_class_probabilities,
        target_support.case_ids,
    )
    return KernelMeanProblem(
        protocol=protocol,
        candidate_sources=protocol.candidate_sources,
        source_kernel_means=source_matrix,
        target_kernel_mean=target_mean,
        common_frame_hash=protocol.common_frame_hash,
        kernel_map_hash=kernel_hash,
        preprocessing_hash=target_support.kernel_features.preprocessing_hash,
        candidate_pool_fit_hash=(
            target_support.kernel_features.candidate_pool_fit_hash
        ),
        kernel_transform_role=target_support.kernel_features.transform_role,
        prior_family_hash=target_support.prior_prediction.prior_family_hash,
        prior_control_hash=target_support.prior_prediction.prior_control_hash,
        prior_state_hash=target_support.prior_state_hash,
        prior_sensitivity_positive_prior=(
            target_support.prior_prediction.sensitivity_positive_prior
        ),
        target_kernel_feature_sha256=(
            target_support.kernel_features.values_sha256
        ),
        target_responsibility_sha256=(
            target_support.prior_prediction.responsibility_sha256
        ),
        source_replica_count=len(expected),
        target_support_row_count=len(target_support.kernel_features.values),
    )


def build_support_case_problems(
    protocol: MMDKMMProtocol,
    source_replicas: Sequence[SourceKernelReplica],
    target_support: TargetSupportKernelFeatures,
) -> dict[str, KernelMeanProblem]:
    """Create one leave-one-support-case-out problem per declared case."""

    output: dict[str, KernelMeanProblem] = {}
    case_array = np.asarray(target_support.case_ids, dtype=object)
    for case_id in protocol.support_case_ids:
        held_out = case_array == case_id
        if not np.any(held_out) or not np.any(~held_out):
            raise ProtocolError("Declared support case has no target kernel rows.")
        mask = ~held_out
        subset_features = replace(
            target_support.kernel_features,
            values=target_support.kernel_features.values[mask],
        )
        subset_prediction = replace(
            target_support.prior_prediction,
            probabilities=target_support.soft_class_probabilities[mask],
        )
        case_support = TargetSupportKernelFeatures(
            target_center=target_support.target_center,
            case_ids=tuple(case_array[mask].tolist()),
            kernel_features=subset_features,
            prior_prediction=subset_prediction,
            support_labels_used=False,
            evaluation_embeddings_used=target_support.evaluation_embeddings_used,
            cross_fitted_transductive_support=(
                target_support.cross_fitted_transductive_support
            ),
            cohort_evaluation_embeddings_used=(
                target_support.cohort_evaluation_embeddings_used
            ),
            heldout_evaluation_embeddings_used=(
                target_support.heldout_evaluation_embeddings_used
            ),
        )
        output[case_id] = build_kernel_mean_problem(
            protocol,
            source_replicas,
            case_support,
            require_complete_target_support=False,
        )
    return output


def build_seed_axis_problems(
    protocol: MMDKMMProtocol,
    source_replicas: Sequence[SourceKernelReplica],
    target_support: TargetSupportKernelFeatures,
    *,
    axis: str,
) -> dict[str, KernelMeanProblem]:
    """Create all-retained single-axis source-replica stability problems."""

    if axis == "training_seed":
        return {
            str(seed): build_kernel_mean_problem(
                protocol,
                source_replicas,
                target_support,
                training_seeds=(seed,),
            )
            for seed in protocol.training_seeds
        }
    if axis == "generation_seed":
        return {
            str(seed): build_kernel_mean_problem(
                protocol,
                source_replicas,
                target_support,
                generation_seeds=(seed,),
            )
            for seed in protocol.generation_seeds
        }
    raise ProtocolError("MMD/KMM seed stability axis is invalid.")


def build_prior_sensitivity_problems(
    protocol: MMDKMMProtocol,
    source_replicas: Sequence[SourceKernelReplica],
    target_support: TargetSupportKernelFeatures,
    *,
    config: PriorControlConfig,
) -> dict[str, KernelMeanProblem]:
    """Build the exact predeclared no-label class-prior grid, including reference."""

    if (
        target_support.prior_prediction.sensitivity_positive_prior is not None
        or target_support.prior_prediction.fit_role != config.fit_role
        or target_support.prior_prediction.temperature != float(config.temperature)
        or target_support.prior_prediction.probability_clip
        != float(config.probability_clip)
        or target_support.prior_prediction.reference_positive_prior
        != float(config.reference_positive_prior)
    ):
        raise ProtocolError(
            "Target support does not use the frozen prior-control state."
        )
    reference_key = _prior_variant_id(config.reference_positive_prior)
    output: dict[str, KernelMeanProblem] = {
        reference_key: build_kernel_mean_problem(
            protocol,
            source_replicas,
            target_support,
        )
    }
    for prior in config.sensitivity_positive_priors:
        shifted_prediction = shift_source_only_prior_prediction(
            target_support.prior_prediction,
            positive_prior=prior,
            config=config,
        )
        key = _prior_variant_id(prior)
        shifted_support = TargetSupportKernelFeatures(
            target_center=target_support.target_center,
            case_ids=target_support.case_ids,
            kernel_features=target_support.kernel_features,
            prior_prediction=shifted_prediction,
            support_labels_used=False,
            evaluation_embeddings_used=target_support.evaluation_embeddings_used,
            cross_fitted_transductive_support=(
                target_support.cross_fitted_transductive_support
            ),
            cohort_evaluation_embeddings_used=(
                target_support.cohort_evaluation_embeddings_used
            ),
            heldout_evaluation_embeddings_used=(
                target_support.heldout_evaluation_embeddings_used
            ),
        )
        output[key] = build_kernel_mean_problem(
            protocol,
            source_replicas,
            shifted_support,
        )
    if len(output) != 1 + len(config.sensitivity_positive_priors):
        raise ProtocolError("MMD/KMM prior-sensitivity identifiers collided.")
    return output


def _prior_variant_id(value: float) -> str:
    return f"positive_prior_{float(value):.12g}"


def _feature_provenance(value: object) -> tuple[str, ...]:
    try:
        return (
            str(value.common_frame_hash),
            str(value.preprocessing_hash),
            str(value.candidate_pool_fit_hash),
            str(value.kernel_map_hash),
            str(value.map_fit_role),
            str(value.transform_role),
            str(value.target_rows_used_to_fit),
            str(value.evaluation_rows_used_to_fit),
        )
    except AttributeError as exc:
        raise ProtocolError("MMD/KMM kernel-feature provenance is incomplete.") from exc


def _selected_seeds(
    requested: Sequence[int] | None,
    allowed: tuple[int, ...],
    name: str,
) -> tuple[int, ...]:
    if requested is not None and any(isinstance(value, bool) for value in requested):
        raise ProtocolError(f"MMD/KMM {name} seed subset is invalid.")
    values = (
        allowed
        if requested is None
        else tuple(sorted(int(value) for value in requested))
    )
    if (
        not values
        or len(set(values)) != len(values)
        or not set(values).issubset(allowed)
    ):
        raise ProtocolError(f"MMD/KMM {name} seed subset is invalid.")
    return values


__all__ = (
    "build_kernel_mean_problem",
    "build_prior_sensitivity_problems",
    "build_seed_axis_problems",
    "build_support_case_problems",
    "case_equal_class_balanced_kernel_mean",
)
