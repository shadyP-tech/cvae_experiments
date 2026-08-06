"""Label-blind case partitions and nested source-pool exclusions.

The helpers in this module deliberately accept identities only.  Labels are
not part of either API, so the support split cannot silently become
class-stratified or outcome-adaptive.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
from typing import Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError


@dataclass(frozen=True)
class CasePartitions:
    """One deterministic whole-case support/evaluation partition."""

    target_center: str
    namespace: str
    split_seed: int
    support_cases: tuple[str, ...]
    evaluation_cases: tuple[str, ...]
    support_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    support_sample_ids: tuple[str, ...]
    evaluation_sample_ids: tuple[str, ...]

    @property
    def partition_hash(self) -> str:
        return stable_hash(
            {
                "schema_version": "midogpp_dense_residual_case_partition_v1",
                "target_center": self.target_center,
                "namespace": self.namespace,
                "split_seed": self.split_seed,
                "support_cases": list(self.support_cases),
                "evaluation_cases": list(self.evaluation_cases),
                "support_sample_ids": list(self.support_sample_ids),
                "evaluation_sample_ids": list(self.evaluation_sample_ids),
                "label_blind": True,
                "whole_case": True,
            }
        )


def deterministic_case_partitions(
    sample_ids: Sequence[str],
    case_ids: Sequence[str],
    *,
    target_center: str,
    support_case_count: int = 2,
    namespace: str = "midogpp_dense_residual_support_v1",
    split_seed: int = 20260806,
) -> CasePartitions:
    """Select support cases by a stable identity hash, without labels.

    Rows remain in their input order inside each side of the partition.  The
    case choice depends only on ``namespace``, ``target_center``, and case ID.
    It therefore cannot retry or change in response to class balance or model
    outcomes.
    """

    samples = tuple(str(value) for value in sample_ids)
    cases_by_row = tuple(str(value) for value in case_ids)
    center = str(target_center)
    split_namespace = str(namespace)
    n_support = int(support_case_count)
    if isinstance(split_seed, bool) or not isinstance(split_seed, Integral):
        raise ProtocolError("Case partition split seed must be a non-boolean integer.")
    frozen_seed = int(split_seed)
    if len(samples) != len(cases_by_row) or not samples:
        raise ProtocolError("Case partition identities must be aligned and nonempty.")
    if len(set(samples)) != len(samples):
        raise ProtocolError("Case partition sample IDs must be unique.")
    if any(not value for value in samples) or any(not value for value in cases_by_row):
        raise ProtocolError("Case partition identities cannot be empty.")
    if not center or not split_namespace:
        raise ProtocolError("Case partition center and namespace cannot be empty.")

    unique_cases = tuple(sorted(set(cases_by_row)))
    if n_support <= 0 or len(unique_cases) <= n_support:
        raise ProtocolError(
            "Case partition requires positive support count and at least one "
            "evaluation case."
        )
    ordered_cases = tuple(
        sorted(
            unique_cases,
            key=lambda case_id: (
                hashlib.sha256(
                    f"{split_namespace}|{frozen_seed}|{center}|{case_id}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
                case_id,
            ),
        )
    )
    support_cases = tuple(sorted(ordered_cases[:n_support]))
    support_set = frozenset(support_cases)
    evaluation_cases = tuple(
        case_id for case_id in unique_cases if case_id not in support_set
    )
    support_indices = tuple(
        index for index, case_id in enumerate(cases_by_row) if case_id in support_set
    )
    evaluation_indices = tuple(
        index for index, case_id in enumerate(cases_by_row) if case_id not in support_set
    )
    support_samples = tuple(samples[index] for index in support_indices)
    evaluation_samples = tuple(samples[index] for index in evaluation_indices)
    if (
        not support_indices
        or not evaluation_indices
        or set(support_cases).intersection(evaluation_cases)
        or set(support_samples).intersection(evaluation_samples)
    ):
        raise ProtocolError("Case partition failed whole-case disjointness.")
    return CasePartitions(
        target_center=center,
        namespace=split_namespace,
        split_seed=frozen_seed,
        support_cases=support_cases,
        evaluation_cases=evaluation_cases,
        support_indices=support_indices,
        evaluation_indices=evaluation_indices,
        support_sample_ids=support_samples,
        evaluation_sample_ids=evaluation_samples,
    )


def assert_outer_query_source_exclusions(
    *,
    outer_target: str,
    query_center: str,
    candidate_sources: Sequence[str],
    all_centers: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Validate ``q != H`` and ``e not in {H, q}`` for nested development.

    If ``all_centers`` is provided, the candidate pool must be the complete
    complement of the outer target and pseudo-query.  Returning candidates in
    canonical order makes downstream hashes independent of caller ordering.
    """

    outer = str(outer_target)
    query = str(query_center)
    candidates = tuple(str(value) for value in candidate_sources)
    if not outer or not query or outer == query:
        raise ProtocolError("Nested router development requires query != outer target.")
    if not candidates or len(candidates) != len(set(candidates)):
        raise ProtocolError("Nested candidate sources must be nonempty and unique.")
    if outer in candidates:
        raise ProtocolError("Outer-target expert appeared in nested development.")
    if query in candidates:
        raise ProtocolError("Pseudo-query expert appeared in its candidate pool.")
    canonical = tuple(sorted(candidates))
    if all_centers is not None:
        centers = tuple(str(value) for value in all_centers)
        if len(centers) != len(set(centers)) or outer not in centers or query not in centers:
            raise ProtocolError("Nested development center universe is invalid.")
        expected = tuple(sorted(set(centers).difference((outer, query))))
        if canonical != expected:
            raise ProtocolError(
                "Nested candidate pool is not the complete outer/query-excluded set."
            )
    return canonical


__all__ = (
    "CasePartitions",
    "assert_outer_query_source_exclusions",
    "deterministic_case_partitions",
)
