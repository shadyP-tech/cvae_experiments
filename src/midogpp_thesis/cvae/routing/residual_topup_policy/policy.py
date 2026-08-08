"""Action-free construction of fixed G/S residual top-up proxy summaries."""

from __future__ import annotations

from numbers import Integral
from types import MappingProxyType
from typing import Iterable, Mapping

from ...protocol import ProtocolError
from .contracts import (
    GLOBAL_PSEUDOQUERY_ROLE,
    TARGET_SUPPORT_ROLE,
    FreshProxyScoreRow,
    TargetProxyPolicySummary,
    canonical_source_ids,
)
from .ranking import (
    build_leave_target_out_global_rank,
    build_target_support_rank,
)


def canonical_source_identity_permutation(
    candidate_sources: Iterable[object],
    *,
    permutation_index: int = 1,
) -> Mapping[str, str]:
    """Return a canonical non-identity cyclic source-label permutation.

    The caller predeclares ``permutation_index`` before any downstream outcome
    is opened.  Valid indices are the nonzero rotations of canonical source
    order, so every result is deterministic, bijective, and has no fixed point.
    """

    sources = canonical_source_ids(candidate_sources)
    if (
        isinstance(permutation_index, bool)
        or not isinstance(permutation_index, Integral)
        or not 1 <= int(permutation_index) < len(sources)
    ):
        raise ProtocolError(
            "Source-identity permutation index must select a nonzero canonical rotation."
        )
    shift = int(permutation_index)
    return MappingProxyType(
        {
            source: sources[(index + shift) % len(sources)]
            for index, source in enumerate(sources)
        }
    )


def build_target_proxy_policy(
    rows: Iterable[FreshProxyScoreRow],
    *,
    outer_target: str,
    candidate_sources: Iterable[object],
    permutation_index: int = 1,
) -> TargetProxyPolicySummary:
    """Build immutable G/S summaries for one held-out target H.

    This function deliberately stops before residual top-up action creation.
    """

    target = str(outer_target)
    if not target or target.strip() != target:
        raise ProtocolError("Target proxy-policy outer target H is invalid.")
    sources = canonical_source_ids(candidate_sources)
    if target in sources or len(sources) < 3:
        raise ProtocolError(
            "Target proxy-policy candidates must exclude H and contain at least three sources."
        )
    score_rows = tuple(rows)
    if not score_rows:
        raise ProtocolError("Target proxy-policy score grid is empty.")
    if any(not isinstance(row, FreshProxyScoreRow) for row in score_rows):
        raise ProtocolError("Target proxy-policy rows must use FreshProxyScoreRow.")
    if any(row.outer_target != target for row in score_rows):
        raise ProtocolError("Target proxy-policy score grid mixes outer targets.")
    case_owner: dict[str, tuple[str, str]] = {}
    for row in score_rows:
        owner = (row.query_role, row.query_center)
        previous = case_owner.setdefault(row.case_id, owner)
        if previous != owner:
            raise ProtocolError(
                "Proxy-policy case IDs must be disjoint across query surfaces."
            )
    global_rows = tuple(
        row for row in score_rows if row.query_role == GLOBAL_PSEUDOQUERY_ROLE
    )
    support_rows = tuple(
        row for row in score_rows if row.query_role == TARGET_SUPPORT_ROLE
    )
    if len(global_rows) + len(support_rows) != len(score_rows):
        raise ProtocolError("Target proxy-policy score grid contains an invalid role.")
    global_summary = build_leave_target_out_global_rank(
        global_rows,
        outer_target=target,
        candidate_sources=sources,
    )
    support_summary = build_target_support_rank(
        support_rows,
        outer_target=target,
        candidate_sources=sources,
    )
    permutation = canonical_source_identity_permutation(
        sources, permutation_index=permutation_index
    )
    return TargetProxyPolicySummary(
        outer_target=target,
        candidate_sources=sources,
        global_summary=global_summary,
        support_summary=support_summary,
        source_identity_permutation=permutation,
        permutation_index=int(permutation_index),
    )


def build_proxy_policies_by_target(
    rows: Iterable[FreshProxyScoreRow],
    *,
    source_centers: Iterable[object],
    permutation_index_by_target: Mapping[object, int] | None = None,
) -> Mapping[str, TargetProxyPolicySummary]:
    """Build a complete per-target policy map over a closed source universe."""

    centers = canonical_source_ids(source_centers)
    if len(centers) < 4:
        raise ProtocolError("All-target proxy policies require at least four centers.")
    score_rows = tuple(rows)
    if not score_rows or any(
        not isinstance(row, FreshProxyScoreRow) for row in score_rows
    ):
        raise ProtocolError("All-target proxy-policy rows are invalid or empty.")
    rows_by_target: dict[str, list[FreshProxyScoreRow]] = {
        center: [] for center in centers
    }
    for row in score_rows:
        if row.outer_target not in rows_by_target:
            raise ProtocolError("Proxy-policy row target is outside the source universe.")
        rows_by_target[row.outer_target].append(row)
    if any(not target_rows for target_rows in rows_by_target.values()):
        raise ProtocolError("All-target proxy-policy grid is incomplete.")

    if permutation_index_by_target is None:
        permutation_indices = {center: 1 for center in centers}
    else:
        if not isinstance(permutation_index_by_target, Mapping):
            raise ProtocolError("Permutation-index plan must be a mapping.")
        permutation_indices: dict[str, int] = {}
        for raw_target, raw_index in permutation_index_by_target.items():
            target = str(raw_target)
            if target in permutation_indices:
                raise ProtocolError("Permutation-index target keys are duplicated.")
            permutation_indices[target] = raw_index
        if set(permutation_indices) != set(centers):
            raise ProtocolError("Permutation-index target grid is incomplete.")

    summaries = {
        target: build_target_proxy_policy(
            rows_by_target[target],
            outer_target=target,
            candidate_sources=(center for center in centers if center != target),
            permutation_index=permutation_indices[target],
        )
        for target in centers
    }
    return MappingProxyType(summaries)


build_residual_topup_proxy_policy = build_target_proxy_policy
build_source_identity_permutation_control = canonical_source_identity_permutation


__all__ = (
    "build_proxy_policies_by_target",
    "build_residual_topup_proxy_policy",
    "build_source_identity_permutation_control",
    "build_target_proxy_policy",
    "canonical_source_identity_permutation",
)
