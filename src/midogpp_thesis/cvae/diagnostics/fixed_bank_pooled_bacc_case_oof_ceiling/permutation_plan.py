"""Compact pre-evaluation seal of 10k pooled-BACC null routing actions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .case_partitions import CaseOOFPartition
from .core_contracts import SealedProbabilitySurface, SufficientStatisticSurface
from .core_hashing import canonical_hash, require_sha256
from .decisions import DecisionConfig
from .permutation_controls import _case_candidate_order, _case_shift_base
from .pooled_posterior import PosteriorConfig
from .pooled_prior import PooledLocoPrior
from .scientific_constants import (
    BASELINE_ACTION_ID,
    EXPECTED_FOLD_COUNT,
    MIDOGPP_CENTERS,
    action_ids,
    candidate_actions,
    routing_challengers,
)


PERMUTATION_DECISION_TIE_BREAK = "lexicographic_action_id_no_evaluation_utility_access"
NULL_DERANGEMENT_ALGORITHM = (
    "case_sha256_candidate_order_counter_splitmix64_nonzero_cyclic_shift_1_to_7_v1"
)


@dataclass(frozen=True)
class PermutationDecisionPlan:
    action_codes: np.ndarray
    permutation_seed: int
    permutation_count: int
    fold_keys: tuple[tuple[str, int], ...]
    partition_hash: str
    probability_surface_hash: str
    support_input_hash: str
    plan_hash: str
    sealed_before_evaluation_labels: bool = True
    evaluation_labels_used_to_generate_actions: bool = False

    def __post_init__(self) -> None:
        values = np.asarray(self.action_codes)
        expected_keys = tuple(
            (center, fold)
            for center in MIDOGPP_CENTERS
            for fold in range(EXPECTED_FOLD_COUNT)
        )
        if (
            values.dtype != np.uint8
            or values.shape != (self.permutation_count, len(expected_keys))
            or self.permutation_count <= 0
            or tuple(self.fold_keys) != expected_keys
            or self.sealed_before_evaluation_labels is not True
            or self.evaluation_labels_used_to_generate_actions is not False
        ):
            raise ProtocolError("Permutation decision plan geometry/protocol drifted.")
        if any(
            np.any(values[:, column] >= len(action_ids(center)))
            for column, (center, _fold) in enumerate(expected_keys)
        ):
            raise ProtocolError("Permutation action code escaped its target action domain.")
        for name in ("partition_hash", "probability_surface_hash", "support_input_hash", "plan_hash"):
            require_sha256(getattr(self, name), name)
        contiguous = np.ascontiguousarray(values, dtype=np.uint8)
        contiguous.setflags(write=False)
        object.__setattr__(self, "action_codes", contiguous)
        if _plan_hash(self._metadata(), contiguous) != self.plan_hash:
            raise ProtocolError("Permutation decision-plan hash drifted.")

    @property
    def action_codes_sha256(self) -> str:
        return hashlib.sha256(self.action_codes.tobytes(order="C")).hexdigest()

    @property
    def permutation_decision_seal_hash(self) -> str:
        return self.plan_hash

    @property
    def permutation_seal_hash(self) -> str:
        return self.plan_hash

    def _metadata(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_permutation_decision_plan_v2",
            "permutation_seed": self.permutation_seed,
            "permutation_count": self.permutation_count,
            "fold_keys": [list(value) for value in self.fold_keys],
            "partition_hash": self.partition_hash,
            "probability_surface_hash": self.probability_surface_hash,
            "support_input_hash": self.support_input_hash,
            "action_shape": list(self.action_codes.shape),
            "action_dtype": "uint8",
            "candidate_statistic_blocks_deranged_within_support_case": True,
            "candidate_multiset_preserved_per_case": True,
            "null_derangement_algorithm": NULL_DERANGEMENT_ALGORITHM,
            "permutation_derangement_family": NULL_DERANGEMENT_ALGORITHM,
            "permutation_candidate_order": "case_specific_sha256_of_seed_fold_id_case_id_action_then_action",
            "permutation_shift_generator": "independent_counter_splitmix64_per_fold_case_permutation_index",
            "permutation_shift_range_inclusive": [1, 7],
            "permutation_zero_shift_allowed": False,
            "uniform_over_all_derangements": False,
            "permutation_decision_tie_break": PERMUTATION_DECISION_TIE_BREAK,
            "evaluation_utility_used_for_permutation_tie_break": False,
            "baseline_action_permuted": False,
            "evaluation_case_donor_used": False,
            "pooled_bacc_posterior_recomputed": True,
            "sealed_before_evaluation_labels": True,
            "evaluation_labels_used_to_generate_actions": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._metadata(),
            "action_codes_sha256": self.action_codes_sha256,
            "permutation_decision_seal_hash": self.plan_hash,
            "permutation_seal_hash": self.plan_hash,
            "generated_before_evaluation_label_access": True,
            "plan_hash": self.plan_hash,
        }


def build_permutation_decision_plan(
    partition: CaseOOFPartition,
    probabilities: SealedProbabilitySurface,
    priors: Sequence[PooledLocoPrior],
    support_statistics: Mapping[tuple[str, int], SufficientStatisticSurface],
    *,
    posterior_config: PosteriorConfig = PosteriorConfig(),
    decision_config: DecisionConfig = DecisionConfig(),
    permutation_seed: int,
    permutation_count: int,
    chunk_size: int = 512,
) -> PermutationDecisionPlan:
    if isinstance(permutation_count, bool) or not isinstance(permutation_count, int) or permutation_count <= 0:
        raise ProtocolError("permutation_count must be a positive integer.")
    if isinstance(permutation_seed, bool) or not isinstance(permutation_seed, int):
        raise ProtocolError("permutation_seed must be an integer.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ProtocolError("chunk_size must be a positive integer.")
    prior_by_target = {prior.target_center: prior for prior in priors}
    if tuple(prior_by_target) != MIDOGPP_CENTERS:
        raise ProtocolError("Permutation plan requires nine canonical sealed G_H priors.")
    fold_keys = tuple((fold.target_center, fold.fold_ordinal) for fold in partition.folds)
    if set(support_statistics) != set(fold_keys):
        raise ProtocolError("Permutation plan requires all 45 support statistic surfaces.")
    support_payload = {
        "schema_version": "fixed_bank_pooled_bacc_null_support_inputs_v2",
        "partition_hash": partition.partition_hash,
        "probability_surface_hash": probabilities.surface_hash,
        "prior_hashes": [prior_by_target[center].prior_hash for center in MIDOGPP_CENTERS],
        "support_statistics_surface_hashes": [
            support_statistics[key].statistics_surface_hash for key in fold_keys
        ],
        "posterior_config": posterior_config.to_payload(),
        "decision_config": decision_config.to_payload(),
        "permutation_seed": permutation_seed,
        "permutation_count": permutation_count,
    }
    support_input_hash = canonical_hash(support_payload)
    codes = np.empty((permutation_count, len(fold_keys)), dtype=np.uint8)
    for column, fold in enumerate(partition.folds):
        codes[:, column] = _fold_null_action_codes(
            fold=fold,
            surface=support_statistics[(fold.target_center, fold.fold_ordinal)],
            prior=prior_by_target[fold.target_center],
            posterior_config=posterior_config,
            decision_config=decision_config,
            permutation_seed=permutation_seed,
            permutation_count=permutation_count,
            chunk_size=chunk_size,
        )
    metadata = {
        "schema_version": "fixed_bank_pooled_bacc_permutation_decision_plan_v2",
        "permutation_seed": permutation_seed,
        "permutation_count": permutation_count,
        "fold_keys": [list(value) for value in fold_keys],
        "partition_hash": partition.partition_hash,
        "probability_surface_hash": probabilities.surface_hash,
        "support_input_hash": support_input_hash,
        "action_shape": list(codes.shape),
        "action_dtype": "uint8",
        "candidate_statistic_blocks_deranged_within_support_case": True,
        "candidate_multiset_preserved_per_case": True,
        "null_derangement_algorithm": NULL_DERANGEMENT_ALGORITHM,
        "permutation_derangement_family": NULL_DERANGEMENT_ALGORITHM,
        "permutation_candidate_order": "case_specific_sha256_of_seed_fold_id_case_id_action_then_action",
        "permutation_shift_generator": "independent_counter_splitmix64_per_fold_case_permutation_index",
        "permutation_shift_range_inclusive": [1, 7],
        "permutation_zero_shift_allowed": False,
        "uniform_over_all_derangements": False,
        "permutation_decision_tie_break": PERMUTATION_DECISION_TIE_BREAK,
        "evaluation_utility_used_for_permutation_tie_break": False,
        "baseline_action_permuted": False,
        "evaluation_case_donor_used": False,
        "pooled_bacc_posterior_recomputed": True,
        "sealed_before_evaluation_labels": True,
        "evaluation_labels_used_to_generate_actions": False,
    }
    return PermutationDecisionPlan(
        action_codes=codes,
        permutation_seed=permutation_seed,
        permutation_count=permutation_count,
        fold_keys=fold_keys,
        partition_hash=partition.partition_hash,
        probability_surface_hash=probabilities.surface_hash,
        support_input_hash=support_input_hash,
        plan_hash=_plan_hash(metadata, codes),
    )


def _fold_null_action_codes(
    *,
    fold,
    surface: SufficientStatisticSurface,
    prior: PooledLocoPrior,
    posterior_config: PosteriorConfig,
    decision_config: DecisionConfig,
    permutation_seed: int,
    permutation_count: int,
    chunk_size: int,
) -> np.ndarray:
    cases = tuple(fold.support_case_ids)
    if (
        set(surface.allowed_case_keys)
        != {(fold.target_center, case) for case in cases}
        or surface.prerequisite_seal_hash != prior.prior_hash
        or prior.target_center != fold.target_center
        or posterior_config.variance_floor != prior.config.variance_floor
        or posterior_config.confidence_multiplier != prior.config.confidence_multiplier
    ):
        raise ProtocolError("Permutation fold input crossed its support/prior capability.")
    candidates = candidate_actions(fold.target_center)
    lookup = surface.by_key()
    matrix = np.asarray(
        [
            [
                [
                    lookup[(fold.target_center, case, action)].n_positive,
                    lookup[(fold.target_center, case, action)].true_positive,
                    lookup[(fold.target_center, case, action)].n_negative,
                    lookup[(fold.target_center, case, action)].true_negative,
                ]
                for action in candidates
            ]
            for case in cases
        ],
        dtype=np.int64,
    )
    baseline = np.asarray(
        [
            [
                lookup[(fold.target_center, case, BASELINE_ACTION_ID)].n_positive,
                lookup[(fold.target_center, case, BASELINE_ACTION_ID)].true_positive,
                lookup[(fold.target_center, case, BASELINE_ACTION_ID)].n_negative,
                lookup[(fold.target_center, case, BASELINE_ACTION_ID)].true_negative,
            ]
            for case in cases
        ],
        dtype=np.int64,
    )
    if np.any(matrix[:, :, (0, 2)] != baseline[:, None, (0, 2)]):
        raise ProtocolError("Candidate sufficient-statistic class totals drifted from B.")
    candidate_orders = np.empty((len(cases), len(candidates)), dtype=np.int8)
    for case_index, case_id in enumerate(cases):
        ordered = _case_candidate_order(
            candidates,
            permutation_seed=permutation_seed,
            fold_id=fold.fold_id,
            case_id=case_id,
        )
        candidate_orders[case_index] = np.asarray(
            [candidates.index(action) for action in ordered], dtype=np.int8
        )
    shifts = np.empty((permutation_count, len(cases)), dtype=np.int8)
    counters = np.arange(1, permutation_count + 1, dtype=np.uint64)
    for case_index, case in enumerate(cases):
        base = np.uint64(
            _case_shift_base(
                permutation_seed=permutation_seed,
                fold_id=fold.fold_id,
                case_id=case,
            )
        )
        values = base + counters * np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
        shifts[:, case_index] = (1 + values % np.uint64(7)).astype(np.int8)
    challengers = routing_challengers(fold.target_center, prior.global_action_id)
    challenger_indices = np.asarray([candidates.index(action) for action in challengers])
    prior_means = np.asarray(
        [prior.pairwise_estimate(action).prior_mean for action in challengers], dtype=np.float64
    )
    prior_variances = np.asarray(
        [prior.pairwise_estimate(action).prior_variance for action in challengers],
        dtype=np.float64,
    )
    global_code = action_ids(fold.target_center).index(prior.global_action_id)
    output = np.empty(permutation_count, dtype=np.uint8)
    for start in range(0, permutation_count, chunk_size):
        stop = min(start + chunk_size, permutation_count)
        chunk_shifts = shifts[start:stop]
        size = stop - start
        permuted = np.empty((size, len(cases), len(candidates), 4), dtype=np.int64)
        for case_index in range(len(cases)):
            order = candidate_orders[case_index].astype(np.int64)
            for shift in range(1, len(candidates)):
                selected = np.flatnonzero(chunk_shifts[:, case_index] == shift)
                if selected.size == 0:
                    continue
                donor = np.roll(order, -shift)
                permuted[
                    selected[:, None],
                    case_index,
                    order[None, :],
                    :,
                ] = matrix[case_index, donor, :][None, :, :]
        if prior.global_action_id == BASELINE_ACTION_ID:
            global_blocks = np.broadcast_to(
                baseline[None, :, :], (size, len(cases), 4)
            )
        else:
            global_blocks = permuted[:, :, candidates.index(prior.global_action_id), :]
        challenger_blocks = permuted[:, :, challenger_indices, :]
        # Shape: permutation x case x challenger.
        positive_difference = (
            challenger_blocks[:, :, :, 1] - global_blocks[:, :, None, 1]
        )
        negative_difference = (
            challenger_blocks[:, :, :, 3] - global_blocks[:, :, None, 3]
        )
        n_positive_case = challenger_blocks[:, :, :, 0]
        n_negative_case = challenger_blocks[:, :, :, 2]
        n_positive = n_positive_case.sum(axis=1)
        n_negative = n_negative_case.sum(axis=1)
        if np.any(n_positive <= 0) or np.any(n_negative <= 0):
            raise ProtocolError("Null support scope lacks a pooled binary class.")
        positive_mean = positive_difference.sum(axis=1) / n_positive
        negative_mean = negative_difference.sum(axis=1) / n_negative
        pooled_difference = 0.5 * (positive_mean + negative_mean)
        psi = 0.5 * (
            (
                positive_difference
                - n_positive_case * positive_mean[:, None, :]
            )
            / n_positive[:, None, :]
            + (
                negative_difference
                - n_negative_case * negative_mean[:, None, :]
            )
            / n_negative[:, None, :]
        )
        m = len(cases)
        if m < 2:
            raise ProtocolError("Null cluster variance requires at least two support cases.")
        support_variance = np.maximum(
            m / (m - 1) * np.square(psi).sum(axis=1),
            posterior_config.variance_floor,
        )
        posterior_variance = 1.0 / (
            1.0 / prior_variances[None, :] + 1.0 / support_variance
        )
        posterior_mean = posterior_variance * (
            prior_means[None, :] / prior_variances[None, :]
            + pooled_difference / support_variance
        )
        lower = posterior_mean - posterior_config.confidence_multiplier * np.sqrt(
            posterior_variance
        )
        maximum = lower.max(axis=1)
        eligible = maximum[:, None] - lower <= decision_config.tie_tolerance
        best = np.argmax(eligible, axis=1)
        best_lower = lower[np.arange(size), best]
        challenger_codes = np.asarray(
            [action_ids(fold.target_center).index(action) for action in challengers],
            dtype=np.uint8,
        )
        best_codes = challenger_codes[best]
        output[start:stop] = np.where(
            best_lower > decision_config.minimum_gain,
            best_codes,
            global_code,
        ).astype(np.uint8)
    return output


def _plan_hash(metadata: Mapping[str, object], actions: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")).encode()
    )
    digest.update(np.ascontiguousarray(actions, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


__all__ = (
    "NULL_DERANGEMENT_ALGORITHM",
    "PERMUTATION_DECISION_TIE_BREAK",
    "PermutationDecisionPlan",
    "build_permutation_decision_plan",
)
