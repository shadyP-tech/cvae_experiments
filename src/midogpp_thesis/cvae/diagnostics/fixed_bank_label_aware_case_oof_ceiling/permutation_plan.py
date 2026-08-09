"""Compact pre-evaluation seal of all blocked-null routing actions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .core_contracts import CaseUtilitySurface, SealedProbabilitySurface
from .core_hashing import require_sha256
from .decisions import DecisionConfig
from .global_prior import LocoGlobalPrior
from .partitions import CaseOOFPartition
from .posterior import PosteriorConfig
from .scientific_constants import BASELINE_ACTION_ID, MIDOGPP_CENTERS, action_ids, candidate_actions


PERMUTATION_DECISION_TIE_BREAK = (
    "lexicographic_action_id_no_evaluation_utility_access"
)


@dataclass(frozen=True)
class PermutationDecisionPlan:
    """A uint8 (permutation, 45-fold) null-action matrix sealed pre-eval."""

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
            (center, fold) for center in MIDOGPP_CENTERS for fold in range(5)
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
        if _plan_hash(self._metadata(), contiguous) != self.plan_hash:
            raise ProtocolError("Permutation decision-plan hash drifted.")
        object.__setattr__(self, "action_codes", contiguous)

    @property
    def action_codes_sha256(self) -> str:
        return hashlib.sha256(self.action_codes.tobytes(order="C")).hexdigest()

    @property
    def permutation_decision_seal_hash(self) -> str:
        """Compatibility name for the scientific plan hash."""

        return self.plan_hash

    def _metadata(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_label_aware_permutation_decision_plan_v1",
            "permutation_seed": self.permutation_seed,
            "permutation_count": self.permutation_count,
            "fold_keys": [list(value) for value in self.fold_keys],
            "partition_hash": self.partition_hash,
            "probability_surface_hash": self.probability_surface_hash,
            "support_input_hash": self.support_input_hash,
            "action_shape": list(self.action_codes.shape),
            "action_dtype": "uint8",
            "candidate_source_labels_deranged_within_support_case": True,
            "permutation_decision_tie_break": PERMUTATION_DECISION_TIE_BREAK,
            "evaluation_utility_used_for_permutation_tie_break": False,
            "baseline_action_permuted": False,
            "evaluation_case_donor_used": False,
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
    priors: Sequence[LocoGlobalPrior],
    support_utilities: Mapping[tuple[str, int], CaseUtilitySurface],
    *,
    posterior_config: PosteriorConfig = PosteriorConfig(),
    decision_config: DecisionConfig = DecisionConfig(),
    permutation_seed: int,
    permutation_count: int,
) -> PermutationDecisionPlan:
    """Generate 10k-ready null actions using support labels only, then seal."""

    if isinstance(permutation_count, bool) or not isinstance(permutation_count, int) or permutation_count <= 0:
        raise ProtocolError("permutation_count must be a positive integer.")
    if isinstance(permutation_seed, bool) or not isinstance(permutation_seed, int):
        raise ProtocolError("permutation_seed must be an integer.")
    prior_by_target = {prior.target_center: prior for prior in priors}
    if tuple(prior_by_target) != MIDOGPP_CENTERS:
        raise ProtocolError("Permutation plan requires the nine sealed G_H priors.")
    fold_keys = tuple((fold.target_center, fold.fold_ordinal) for fold in partition.folds)
    if set(support_utilities) != set(fold_keys):
        raise ProtocolError("Permutation plan requires all 45 support surfaces.")
    support_hash_payload = {
        "partition_hash": partition.partition_hash,
        "probability_surface_hash": probabilities.surface_hash,
        "priors": [prior_by_target[center].prior_hash for center in MIDOGPP_CENTERS],
        "support_exact_surface_hashes": [
            support_utilities[key].exact_surface_hash for key in fold_keys
        ],
        "posterior_config": posterior_config.to_payload(),
        "decision_config": decision_config.to_payload(),
        "permutation_seed": permutation_seed,
        "permutation_count": permutation_count,
    }
    support_input_hash = hashlib.sha256(
        json.dumps(support_hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    codes = np.empty((permutation_count, len(fold_keys)), dtype=np.uint8)
    for column, fold in enumerate(partition.folds):
        prior = prior_by_target[fold.target_center]
        surface = support_utilities[(fold.target_center, fold.fold_ordinal)]
        codes[:, column] = _fold_null_action_codes(
            fold=fold,
            surface=surface,
            prior=prior,
            posterior_config=posterior_config,
            decision_config=decision_config,
            permutation_seed=permutation_seed,
            permutation_count=permutation_count,
        )
    metadata = {
        "schema_version": "fixed_bank_label_aware_permutation_decision_plan_v1",
        "permutation_seed": permutation_seed,
        "permutation_count": permutation_count,
        "fold_keys": [list(value) for value in fold_keys],
        "partition_hash": partition.partition_hash,
        "probability_surface_hash": probabilities.surface_hash,
        "support_input_hash": support_input_hash,
        "action_shape": list(codes.shape),
        "action_dtype": "uint8",
        "candidate_source_labels_deranged_within_support_case": True,
        "permutation_decision_tie_break": PERMUTATION_DECISION_TIE_BREAK,
        "evaluation_utility_used_for_permutation_tie_break": False,
        "baseline_action_permuted": False,
        "evaluation_case_donor_used": False,
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
    surface: CaseUtilitySurface,
    prior: LocoGlobalPrior,
    posterior_config: PosteriorConfig,
    decision_config: DecisionConfig,
    permutation_seed: int,
    permutation_count: int,
) -> np.ndarray:
    expected_cases = tuple(fold.support_case_ids)
    if (
        set(surface.allowed_case_keys)
        != {(fold.target_center, case_id) for case_id in expected_cases}
        or surface.prerequisite_seal_hash != prior.prior_hash
    ):
        raise ProtocolError("Permutation fold input crossed its support capability.")
    candidates = candidate_actions(fold.target_center)
    lookup = surface.by_key()
    matrix = np.asarray(
        [
            [lookup[(fold.target_center, case_id, action)].exact_bacc for action in candidates]
            for case_id in expected_cases
        ],
        dtype=np.float64,
    )
    baseline = np.asarray(
        [lookup[(fold.target_center, case_id, BASELINE_ACTION_ID)].exact_bacc for case_id in expected_cases],
        dtype=np.float64,
    )
    shifts = np.empty((permutation_count, len(expected_cases)), dtype=np.int8)
    candidate_orders = np.empty((len(expected_cases), len(candidates)), dtype=np.int8)
    for case_index, case_id in enumerate(expected_cases):
        candidate_orders[case_index] = np.asarray(
            sorted(
                range(len(candidates)),
                key=lambda index: hashlib.sha256(
                    f"{permutation_seed}::{fold.fold_id}::{case_id}::{candidates[index]}".encode("utf-8")
                ).hexdigest(),
            ),
            dtype=np.int8,
        )
    fold_seed = int.from_bytes(
        hashlib.sha256(f"{permutation_seed}::{fold.fold_id}".encode("utf-8")).digest()[:8],
        "big",
    )
    rng = np.random.Generator(np.random.PCG64(fold_seed))
    shifts[:] = rng.integers(1, len(candidates), size=shifts.shape, dtype=np.int8)
    permuted = np.empty((permutation_count, len(expected_cases), len(candidates)), dtype=np.float64)
    for case_index in range(len(expected_cases)):
        order = candidate_orders[case_index].astype(np.int64)
        for shift in range(1, len(candidates)):
            recipient = order
            donor = np.roll(order, -shift)
            selected = shifts[:, case_index] == shift
            permuted[np.ix_(selected, [case_index], recipient)] = matrix[case_index, donor][None, None, :]
    global_action = prior.global_action_id
    if global_action == BASELINE_ACTION_ID:
        global_values = np.broadcast_to(baseline[None, :, None], (permutation_count, len(expected_cases), 1))
        global_code = 0
    else:
        global_index = candidates.index(global_action)
        global_values = permuted[:, :, global_index : global_index + 1]
        global_code = global_index + 1
    deltas = permuted - global_values
    n = len(expected_cases)
    local_mean = deltas.mean(axis=1)
    sample_variance = deltas.var(axis=1, ddof=1) if n > 1 else np.zeros_like(local_mean)
    prior_mean = np.asarray(
        [prior.mean_gain_vs_b(action) - prior.mean_gain_vs_b(global_action) for action in candidates],
        dtype=np.float64,
    )
    posterior_mean = (
        posterior_config.prior_strength * prior_mean[None, :] + n * local_mean
    ) / (posterior_config.prior_strength + n)
    standard_error = np.sqrt(
        np.maximum(sample_variance, posterior_config.variance_floor)
        / (posterior_config.prior_strength + n)
    )
    lower = posterior_mean - posterior_config.confidence_multiplier * standard_error
    # np.argmax is the predeclared lexicographic first-action tie break.
    best = np.argmax(lower, axis=1)
    best_lower = lower[np.arange(permutation_count), best]
    best_codes = best.astype(np.uint8) + 1
    switch = (best_codes != global_code) & (
        best_lower > decision_config.minimum_gain + decision_config.tie_tolerance
    )
    return np.where(switch, best_codes, global_code).astype(np.uint8)


def _plan_hash(metadata: Mapping[str, object], actions: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(np.ascontiguousarray(actions, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


__all__ = (
    "PERMUTATION_DECISION_TIE_BREAK",
    "PermutationDecisionPlan",
    "build_permutation_decision_plan",
)
