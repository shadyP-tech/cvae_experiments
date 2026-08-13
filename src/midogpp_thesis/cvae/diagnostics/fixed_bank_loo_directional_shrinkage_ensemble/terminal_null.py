"""Vectorized execution of the sealed candidate-identity null plan."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    B_ACTION_ID,
    CENTERS,
    DIRECTION_IDS,
    a1_action_id,
    candidate_sources,
)
from .loo_plans import WholeCaseLooPlan
from .nulls import (
    CandidateIdentityNullPlan,
    descriptive_null_statistics,
    select_endpoint_values_vectorized,
)
from .probability_surfaces import ProbabilityIndex
from .products import CaseActionConfusion, DonorPrior
from .terminal_scoring import terminal_truth
from .terminal_sensitivity import canonical_case_action_counts


def _priors_by_target(
    rows: Sequence[DonorPrior],
) -> dict[str, tuple[DonorPrior, ...]]:
    result = {
        target: tuple(row for row in rows if row.heldout_center == target)
        for target in CENTERS
    }
    if any(len(values) != 16 for values in result.values()):
        raise ProtocolError("DCSE donor prior surface lacks 16 cells per target.")
    return result


def _null_prior_arrays(
    *,
    plans: Sequence[WholeCaseLooPlan],
    donor_priors: Sequence[DonorPrior],
) -> tuple[np.ndarray, np.ndarray]:
    priors = _priors_by_target(donor_priors)
    prior_array = np.empty((len(plans), 2, 8), dtype=np.float64)
    ranking_array = np.empty((len(plans), 2, 8), dtype=np.int8)
    for route_ordinal, plan in enumerate(plans):
        sources = candidate_sources(plan.target_center)
        prior_index = {
            (row.source, row.direction): row.exact
            for row in priors[plan.target_center]
        }
        for direction_ordinal, direction in enumerate(DIRECTION_IDS):
            exact_priors = tuple(
                prior_index[(source, direction)] for source in sources
            )
            prior_array[route_ordinal, direction_ordinal] = tuple(
                float(value) for value in exact_priors
            )
            ranking_array[route_ordinal, direction_ordinal] = tuple(
                sorted(
                    range(8),
                    key=lambda ordinal: (
                        -exact_priors[ordinal],
                        int(sources[ordinal]),
                    ),
                )
            )
    return prior_array, ranking_array


def _null_case_blocks(
    *,
    plans: Sequence[WholeCaseLooPlan],
    case_action_confusions: Sequence[CaseActionConfusion],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-(H,case) paired-direction sufficient-statistic blocks."""

    canonical = canonical_case_action_counts(case_action_confusions)
    route_index = {plan.key: ordinal for ordinal, plan in enumerate(plans)}
    if len(route_index) != len(plans):
        raise ProtocolError("DCSE null case-block route identities are duplicated.")
    favorable = np.empty((len(plans), 2, 8), dtype=np.int64)
    adverse = np.empty_like(favorable)
    n_positive = np.empty(len(plans), dtype=np.int64)
    n_negative = np.empty(len(plans), dtype=np.int64)
    for plan in plans:
        ordinal = route_index[plan.key]
        baseline = canonical[(plan.target_center, plan.case_id, B_ACTION_ID)]
        n_positive[ordinal] = baseline.n_positive
        n_negative[ordinal] = baseline.n_negative
        for source_ordinal, source in enumerate(
            candidate_sources(plan.target_center)
        ):
            row = canonical[
                (plan.target_center, plan.case_id, a1_action_id(source))
            ]
            if (row.n_positive, row.n_negative) != (
                baseline.n_positive,
                baseline.n_negative,
            ):
                raise ProtocolError(
                    "DCSE null case-block class denominators drifted."
                )
            favorable[ordinal, 0, source_ordinal] = row.flip_0to1_positive
            adverse[ordinal, 0, source_ordinal] = row.flip_0to1_negative
            favorable[ordinal, 1, source_ordinal] = row.flip_1to0_negative
            adverse[ordinal, 1, source_ordinal] = row.flip_1to0_positive
    if np.any(n_positive + n_negative <= 0):
        raise ProtocolError("DCSE null case-block surface contains an empty case.")
    return favorable, adverse, n_positive, n_negative


def _scrambled_loo_support_values(
    *,
    plans: Sequence[WholeCaseLooPlan],
    permutations: np.ndarray,
    favorable: np.ndarray,
    adverse: np.ndarray,
    n_positive: np.ndarray,
    n_negative: np.ndarray,
) -> np.ndarray:
    """Permute each case block jointly, then pool exact H-minus-c support."""

    shuffled = np.asarray(permutations, dtype=np.int64)
    if (
        shuffled.ndim != 3
        or shuffled.shape[1:] != (len(plans), 8)
        or favorable.shape != (len(plans), 2, 8)
        or adverse.shape != favorable.shape
        or n_positive.shape != (len(plans),)
        or n_negative.shape != (len(plans),)
    ):
        raise ProtocolError("DCSE null case-block arrays are not aligned.")
    replicate_count = shuffled.shape[0]
    permutation_index = np.broadcast_to(
        shuffled[:, :, None, :],
        (replicate_count, len(plans), 2, 8),
    )
    scrambled_favorable = np.take_along_axis(
        np.broadcast_to(
            favorable[None, :, :, :],
            (replicate_count, len(plans), 2, 8),
        ),
        permutation_index,
        axis=3,
    )
    scrambled_adverse = np.take_along_axis(
        np.broadcast_to(
            adverse[None, :, :, :],
            (replicate_count, len(plans), 2, 8),
        ),
        permutation_index,
        axis=3,
    )
    support = np.empty_like(scrambled_favorable, dtype=np.float64)
    for center in CENTERS:
        indices = np.asarray(
            [
                ordinal
                for ordinal, plan in enumerate(plans)
                if plan.target_center == center
            ],
            dtype=np.int64,
        )
        total_favorable = np.sum(
            scrambled_favorable[:, indices, :, :],
            axis=1,
            dtype=np.int64,
        )
        total_adverse = np.sum(
            scrambled_adverse[:, indices, :, :], axis=1, dtype=np.int64
        )
        pooled_favorable = (
            total_favorable[:, None, :, :]
            - scrambled_favorable[:, indices, :, :]
        )
        pooled_adverse = (
            total_adverse[:, None, :, :]
            - scrambled_adverse[:, indices, :, :]
        )
        positive = (
            np.sum(n_positive[indices], dtype=np.int64) - n_positive[indices]
        )
        negative = (
            np.sum(n_negative[indices], dtype=np.int64) - n_negative[indices]
        )
        if np.any(positive <= 0) or np.any(negative <= 0):
            raise ProtocolError(
                "DCSE null H-minus-c support lacks one label class."
            )
        support[:, indices, 0, :] = (
            pooled_favorable[:, :, 0, :] / (2.0 * positive[None, :, None])
            - pooled_adverse[:, :, 0, :] / (2.0 * negative[None, :, None])
        )
        support[:, indices, 1, :] = (
            pooled_favorable[:, :, 1, :] / (2.0 * negative[None, :, None])
            - pooled_adverse[:, :, 1, :] / (2.0 * positive[None, :, None])
        )
    return support


def evaluate_candidate_identity_null(
    *,
    plan: CandidateIdentityNullPlan,
    plans: Sequence[WholeCaseLooPlan],
    probability_surface: object,
    case_action_confusions: Sequence[CaseActionConfusion],
    donor_priors: Sequence[DonorPrior],
    terminal_labels: Sequence[object],
    observed_statistic: float,
    chunk_size: int = 32,
) -> tuple[dict[str, object], ...]:
    """Execute every sealed candidate-block scramble through all nine arms."""

    routes = tuple(plans)
    if tuple(route.key for route in routes) != plan.route_keys or chunk_size <= 0:
        raise ProtocolError("DCSE null plan/LOO route order drifted.")
    truth = terminal_truth(terminal_labels)
    probability = ProbabilityIndex(
        tuple(getattr(probability_surface, "rows", probability_surface))
    )
    priors, rankings = _null_prior_arrays(
        plans=routes, donor_priors=donor_priors
    )
    favorable, adverse, block_positive, block_negative = _null_case_blocks(
        plans=routes, case_action_confusions=case_action_confusions
    )
    permutations = plan.materialize()
    n_positive = np.zeros(len(CENTERS), dtype=np.int64)
    n_negative = np.zeros(len(CENTERS), dtype=np.int64)
    baseline_tp = np.zeros(len(CENTERS), dtype=np.int64)
    baseline_tn = np.zeros(len(CENTERS), dtype=np.int64)
    sample_labels: list[int] = []
    sample_centers: list[int] = []
    sample_routes: list[int] = []
    sample_branches: list[int] = []
    sample_physical: list[tuple[float, ...]] = []
    for route_ordinal, route in enumerate(routes):
        center_ordinal = CENTERS.index(route.target_center)
        keys = tuple(
            (route.target_center, route.case_id, sample)
            for sample in route.evaluation_sample_ids
        )
        if any(key not in truth for key in keys):
            raise ProtocolError("DCSE null route lacks terminal labels.")
        labels = np.asarray([truth[key] for key in keys], dtype=np.int8)
        baseline = np.asarray(
            [probability[(*key, B_ACTION_ID)].probability_mean for key in keys],
            dtype=np.float64,
        )
        branches = (baseline >= 0.5).astype(np.int8)
        candidates = np.asarray(
            [
                [
                    probability[
                        (*key, a1_action_id(source))
                    ].probability_mean
                    for source in candidate_sources(route.target_center)
                ]
                for key in keys
            ],
            dtype=np.float64,
        )
        positive = labels == 1
        negative = ~positive
        baseline_hard = baseline >= 0.5
        n_positive[center_ordinal] += int(
            np.sum(positive, dtype=np.int64)
        )
        n_negative[center_ordinal] += int(
            np.sum(negative, dtype=np.int64)
        )
        baseline_tp[center_ordinal] += int(
            np.sum(positive & baseline_hard, dtype=np.int64)
        )
        baseline_tn[center_ordinal] += int(
            np.sum(negative & (~baseline_hard), dtype=np.int64)
        )
        sample_labels.extend(int(value) for value in labels)
        sample_centers.extend([center_ordinal] * len(labels))
        sample_routes.extend([route_ordinal] * len(labels))
        sample_branches.extend(int(value) for value in branches)
        sample_physical.extend(
            tuple(float(value) for value in row)
            for row in np.column_stack((baseline, candidates))
        )
    if np.any(n_positive <= 0) or np.any(n_negative <= 0):
        raise ProtocolError(
            "DCSE null evaluation requires both classes in every center."
        )
    baseline_center = baseline_tp / (2.0 * n_positive) + baseline_tn / (
        2.0 * n_negative
    )
    label_array = np.asarray(sample_labels, dtype=np.int8)
    center_array = np.asarray(sample_centers, dtype=np.int8)
    route_array = np.asarray(sample_routes, dtype=np.int64)
    branch_array = np.asarray(sample_branches, dtype=np.int8)
    physical_array = np.asarray(sample_physical, dtype=np.float64)
    if (
        label_array.shape != (len(truth),)
        or physical_array.shape != (len(truth), 9)
        or route_array.shape != label_array.shape
        or branch_array.shape != label_array.shape
        or center_array.shape != label_array.shape
    ):
        raise ProtocolError(
            "DCSE null sample arrays are not the terminal universe."
        )
    sample_ordinal = np.arange(len(truth), dtype=np.int64)[None, :, None]
    replicate_statistics = np.empty(plan.replicates, dtype=np.float64)
    for start in range(0, plan.replicates, chunk_size):
        stop = min(plan.replicates, start + chunk_size)
        scrambled_support = _scrambled_loo_support_values(
            plans=routes,
            permutations=permutations[start:stop],
            favorable=favorable,
            adverse=adverse,
            n_positive=block_positive,
            n_negative=block_negative,
        )
        selected = select_endpoint_values_vectorized(
            scrambled_support, priors, rankings
        )
        choices = (
            selected[:, route_array, branch_array, :].astype(np.int64) + 1
        )
        arm_probability = physical_array[sample_ordinal, choices]
        predicted = np.mean(
            arm_probability, axis=2, dtype=np.float64
        ) >= 0.5
        tp = np.empty((stop - start, len(CENTERS)), dtype=np.int64)
        tn = np.empty_like(tp)
        for center_ordinal in range(len(CENTERS)):
            center_mask = center_array == center_ordinal
            tp[:, center_ordinal] = np.sum(
                predicted[:, center_mask]
                & (label_array[center_mask][None, :] == 1),
                axis=1,
                dtype=np.int64,
            )
            tn[:, center_ordinal] = np.sum(
                (~predicted[:, center_mask])
                & (label_array[center_mask][None, :] == 0),
                axis=1,
                dtype=np.int64,
            )
        null_center = tp / (2.0 * n_positive[None, :]) + tn / (
            2.0 * n_negative[None, :]
        )
        replicate_statistics[start:stop] = np.mean(
            null_center - baseline_center[None, :],
            axis=1,
            dtype=np.float64,
        )
    return descriptive_null_statistics(
        plan,
        observed_statistic=observed_statistic,
        replicate_statistics=replicate_statistics,
    )


__all__ = ("evaluate_candidate_identity_null",)
