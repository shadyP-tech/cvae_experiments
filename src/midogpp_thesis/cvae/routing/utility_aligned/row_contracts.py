"""Immutable row and whole-case bootstrap contracts for utility routing."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256


INNER_ROLE = "source_inner_exact_tail"
TARGET_ROLE = "fresh_target_support"
INNER_CANDIDATE_COUNT = 7
TARGET_CANDIDATE_COUNT = 8
TRAIN_CANDIDATE_COUNT_AFTER_STRICT_EXCLUSION = 6
SEED_PAIR_COUNT = len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
MIN_TARGET_SUPPORT_CASES = 8
MIN_SUPPORT_BOOTSTRAP_REPLICATES = 32
DEFAULT_CASE_BOOTSTRAP_SEED = 90_703

EXACT_TAIL_UTILITY_SEMANTICS = (
    "bacc_of_exact_base_plus_single_source_additive_tail_minus_exact_base_bacc"
)


@dataclass(frozen=True)
class ExactTailUtilityRow:
    """One paired source-inner utility observation for the exact action.

    The row is legal only for a pseudoquery ``q`` nested inside an outer target
    ``H``.  The candidate expert ``e`` must be different from both.  Utilities
    are downstream scoring inputs opened only after the relevant predictions
    were sealed; they never enter the deployment feature API.
    """

    outer_target_id: str
    query_id: str
    candidate_source: str
    training_seed: int
    generation_seed: int
    candidate_source_count: int
    support_partition_hash: str
    evaluation_partition_hash: str
    prediction_seal_hash: str
    base_prediction_hash: str
    tail_prediction_hash: str
    base_bacc: float
    tail_bacc: float
    support_eval_disjoint: bool
    predictions_sealed_before_labels: bool
    source_expert_frozen: bool
    target_labels_used_for_routing: bool = False
    utility_semantics: str = EXACT_TAIL_UTILITY_SEMANTICS
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer, query, source = _identifiers(
            self.outer_target_id, self.query_id, self.candidate_source
        )
        if outer == query:
            raise ProtocolError("Source-inner utility requires q != H.")
        if source in {outer, query}:
            raise ProtocolError("Exact-tail utility candidate e must exclude H and q.")
        training_seed = _seed(self.training_seed, TRAINING_SEEDS, "training_seed")
        generation_seed = _seed(
            self.generation_seed, GENERATION_SEEDS, "generation_seed"
        )
        if (
            isinstance(self.candidate_source_count, bool)
            or not isinstance(self.candidate_source_count, Integral)
            or int(self.candidate_source_count) != INNER_CANDIDATE_COUNT
        ):
            raise ProtocolError("Source-inner exact-tail rows require seven candidates.")
        hashes = _hash_fields(
            support_partition_hash=self.support_partition_hash,
            evaluation_partition_hash=self.evaluation_partition_hash,
            prediction_seal_hash=self.prediction_seal_hash,
            base_prediction_hash=self.base_prediction_hash,
            tail_prediction_hash=self.tail_prediction_hash,
        )
        if hashes["support_partition_hash"] == hashes["evaluation_partition_hash"]:
            raise ProtocolError("Support and evaluation partition hashes must differ.")
        base = _bounded_utility(self.base_bacc, "base_bacc")
        tail = _bounded_utility(self.tail_bacc, "tail_bacc")
        if (
            self.support_eval_disjoint is not True
            or self.predictions_sealed_before_labels is not True
            or self.source_expert_frozen is not True
            or self.target_labels_used_for_routing is not False
            or self.utility_semantics != EXACT_TAIL_UTILITY_SEMANTICS
        ):
            raise ProtocolError("Exact-tail utility row violates the label boundary.")
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "query_id", query)
        object.__setattr__(self, "candidate_source", source)
        object.__setattr__(self, "training_seed", training_seed)
        object.__setattr__(self, "generation_seed", generation_seed)
        object.__setattr__(self, "candidate_source_count", INNER_CANDIDATE_COUNT)
        for name, value in hashes.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "base_bacc", base)
        object.__setattr__(self, "tail_bacc", tail)
        object.__setattr__(self, "row_hash", canonical_sha256(self.to_payload()))

    @property
    def replicate_id(self) -> str:
        return f"training_{self.training_seed}__generation_{self.generation_seed}"

    @property
    def utility_delta(self) -> float:
        return self.tail_bacc - self.base_bacc

    @property
    def row_key(self) -> tuple[str, str, str, int, int]:
        return (
            self.outer_target_id,
            self.query_id,
            self.candidate_source,
            self.training_seed,
            self.generation_seed,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_exact_tail_utility_row_v1",
            "outer_target_id": self.outer_target_id,
            "query_id": self.query_id,
            "candidate_source": self.candidate_source,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "replicate_id": self.replicate_id,
            "candidate_source_count": self.candidate_source_count,
            "support_partition_hash": self.support_partition_hash,
            "evaluation_partition_hash": self.evaluation_partition_hash,
            "prediction_seal_hash": self.prediction_seal_hash,
            "base_prediction_hash": self.base_prediction_hash,
            "tail_prediction_hash": self.tail_prediction_hash,
            "base_bacc": self.base_bacc,
            "tail_bacc": self.tail_bacc,
            "utility_delta": self.utility_delta,
            "support_eval_disjoint": self.support_eval_disjoint,
            "predictions_sealed_before_labels": self.predictions_sealed_before_labels,
            "source_expert_frozen": self.source_expert_frozen,
            "target_labels_used_for_routing": self.target_labels_used_for_routing,
            "utility_semantics": self.utility_semantics,
        }


@dataclass(frozen=True)
class CaseBootstrapReplicate:
    """One deterministic whole-case bootstrap index from a typed plan."""

    target_id: str
    replicate_index: int
    sampled_indices: tuple[int, ...]
    sampled_case_ids: tuple[str, ...]
    support_partition_hash: str
    replicate_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_case_bootstrap_replicate_v1",
            "target_id": self.target_id,
            "replicate_index": self.replicate_index,
            "sampled_indices": list(self.sampled_indices),
            "sampled_case_ids": list(self.sampled_case_ids),
            "support_partition_hash": self.support_partition_hash,
            "replicate_hash": self.replicate_hash,
        }


@dataclass(frozen=True)
class CaseBootstrapPlan:
    """Deterministic PCG64 whole-case resampling provenance.

    Callers provide only the original independent case IDs, seed, and count.
    Indices and replicate hashes are generated and sealed here, so arbitrary
    hash-only feature surfaces cannot masquerade as case bootstraps.
    """

    target_id: str
    support_case_ids: tuple[str, ...]
    bootstrap_seed: int = DEFAULT_CASE_BOOTSTRAP_SEED
    replicate_count: int = MIN_SUPPORT_BOOTSTRAP_REPLICATES
    support_partition_hash: str = field(init=False)
    replicates: tuple[CaseBootstrapReplicate, ...] = field(init=False)
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = _canonical_text(self.target_id, "target_id")
        case_ids = tuple(
            sorted(_canonical_text(value, "support_case_id") for value in self.support_case_ids)
        )
        if len(case_ids) < MIN_TARGET_SUPPORT_CASES or len(set(case_ids)) != len(case_ids):
            raise ProtocolError(
                "Case bootstrap requires at least eight unique independent support cases."
            )
        if (
            isinstance(self.bootstrap_seed, bool)
            or not isinstance(self.bootstrap_seed, Integral)
            or int(self.bootstrap_seed) < 0
        ):
            raise ProtocolError("Case bootstrap seed must be a nonnegative integer.")
        if (
            isinstance(self.replicate_count, bool)
            or not isinstance(self.replicate_count, Integral)
            or int(self.replicate_count) < MIN_SUPPORT_BOOTSTRAP_REPLICATES
        ):
            raise ProtocolError("Case bootstrap requires at least 32 replicates.")
        seed = int(self.bootstrap_seed)
        replicate_count = int(self.replicate_count)
        partition_hash = canonical_sha256(
            {
                "schema_version": "midogpp_utility_aligned_support_case_partition_v1",
                "target_id": target,
                "support_case_ids": list(case_ids),
            }
        )
        rng = np.random.Generator(np.random.PCG64(seed))
        replicates: list[CaseBootstrapReplicate] = []
        for replicate_index in range(replicate_count):
            indices = tuple(
                int(value)
                for value in rng.integers(0, len(case_ids), size=len(case_ids))
            )
            sampled = tuple(case_ids[index] for index in indices)
            sample_partition_hash = canonical_sha256(
                {
                    "schema_version": (
                        "midogpp_utility_aligned_bootstrap_case_partition_v1"
                    ),
                    "target_id": target,
                    "parent_support_partition_hash": partition_hash,
                    "bootstrap_seed": seed,
                    "replicate_index": replicate_index,
                    "sampled_indices": list(indices),
                    "sampled_case_ids": list(sampled),
                }
            )
            replicate_payload = {
                "schema_version": "midogpp_utility_aligned_case_bootstrap_replicate_v1",
                "target_id": target,
                "parent_support_partition_hash": partition_hash,
                "bootstrap_seed": seed,
                "replicate_index": replicate_index,
                "sampled_indices": list(indices),
                "sampled_case_ids": list(sampled),
                "support_partition_hash": sample_partition_hash,
            }
            replicates.append(
                CaseBootstrapReplicate(
                    target_id=target,
                    replicate_index=replicate_index,
                    sampled_indices=indices,
                    sampled_case_ids=sampled,
                    support_partition_hash=sample_partition_hash,
                    replicate_hash=canonical_sha256(replicate_payload),
                )
            )
        plan_payload = {
            "schema_version": "midogpp_utility_aligned_case_bootstrap_plan_v1",
            "target_id": target,
            "support_case_ids": list(case_ids),
            "support_partition_hash": partition_hash,
            "bootstrap_seed": seed,
            "replicate_count": replicate_count,
            "replicate_hashes": [item.replicate_hash for item in replicates],
            "resampling_unit": "independent_support_case",
            "rng": "numpy_pcg64",
        }
        object.__setattr__(self, "target_id", target)
        object.__setattr__(self, "support_case_ids", case_ids)
        object.__setattr__(self, "bootstrap_seed", seed)
        object.__setattr__(self, "replicate_count", replicate_count)
        object.__setattr__(self, "support_partition_hash", partition_hash)
        object.__setattr__(self, "replicates", tuple(replicates))
        object.__setattr__(self, "plan_hash", canonical_sha256(plan_payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_case_bootstrap_plan_v1",
            "target_id": self.target_id,
            "support_case_ids": list(self.support_case_ids),
            "support_partition_hash": self.support_partition_hash,
            "bootstrap_seed": self.bootstrap_seed,
            "replicate_count": self.replicate_count,
            "replicates": [item.to_payload() for item in self.replicates],
            "plan_hash": self.plan_hash,
            "resampling_unit": "independent_support_case",
            "rng": "numpy_pcg64",
        }


def build_case_bootstrap_plan(
    *,
    target_id: object,
    support_case_ids: tuple[object, ...],
    bootstrap_seed: int = DEFAULT_CASE_BOOTSTRAP_SEED,
    replicate_count: int = MIN_SUPPORT_BOOTSTRAP_REPLICATES,
) -> CaseBootstrapPlan:
    """Build the only accepted case-bootstrap provenance contract."""

    return CaseBootstrapPlan(
        target_id=str(target_id),
        support_case_ids=tuple(str(value) for value in support_case_ids),
        bootstrap_seed=bootstrap_seed,
        replicate_count=replicate_count,
    )


def _identifiers(*values: object) -> tuple[str, ...]:
    return tuple(_canonical_text(value, "domain/source identifier") for value in values)


def _canonical_text(value: object, name: str) -> str:
    try:
        text = str(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(f"{name} is invalid.") from exc
    if not text or text.strip() != text:
        raise ProtocolError(f"{name} must be nonempty and canonical.")
    return text


def _seed(value: object, allowed: tuple[int, ...], name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in allowed:
        raise ProtocolError(f"{name} is outside the frozen seed set.")
    return int(value)


def _hash_fields(**values: object) -> dict[str, str]:
    return {name: _canonical_text(value, name) for name, value in values.items()}


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"{name} must be finite.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(f"{name} must be finite.") from exc
    if not math.isfinite(number):
        raise ProtocolError(f"{name} must be finite.")
    return number


def _nonnegative(value: object, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0:
        raise ProtocolError(f"{name} must be nonnegative.")
    return number


def _bounded_utility(value: object, name: str) -> float:
    number = _finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise ProtocolError(f"{name} must lie in [0, 1].")
    return number


__all__ = (
    "DEFAULT_CASE_BOOTSTRAP_SEED",
    "EXACT_TAIL_UTILITY_SEMANTICS",
    "INNER_CANDIDATE_COUNT",
    "INNER_ROLE",
    "MIN_SUPPORT_BOOTSTRAP_REPLICATES",
    "MIN_TARGET_SUPPORT_CASES",
    "SEED_PAIR_COUNT",
    "TARGET_CANDIDATE_COUNT",
    "TARGET_ROLE",
    "TRAIN_CANDIDATE_COUNT_AFTER_STRICT_EXCLUSION",
    "CaseBootstrapPlan",
    "CaseBootstrapReplicate",
    "ExactTailUtilityRow",
    "build_case_bootstrap_plan",
)
