"""Pure case-ballot and proxy-rank aggregation for residual top-up policies."""

from __future__ import annotations

from collections import defaultdict
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from ...protocol import ProtocolError
from .contracts import (
    FIXED_TRAINING_SEEDS,
    GLOBAL_AGGREGATION_SEMANTICS,
    GLOBAL_POLICY_ID,
    GLOBAL_PSEUDOQUERY_ROLE,
    SUPPORT_AGGREGATION_SEMANTICS,
    SUPPORT_POLICY_ID,
    TARGET_SUPPORT_ROLE,
    CaseProxyBallot,
    FreshProxyScoreRow,
    ProxyRankSummary,
    canonical_source_ids,
)


def normalized_midranks(
    values: Mapping[object, float] | Sequence[float],
    *,
    source_ids: Sequence[object] | None = None,
    lower_is_better: bool = True,
) -> Mapping[str, float]:
    """Return true normalized midranks, preserving exact score ties.

    Rank one is best.  A tie occupying ranks two and three receives midrank
    2.5 for both sources.  The returned value is ``(midrank - 1) / (m - 1)``.
    """

    if type(lower_is_better) is not bool:
        raise ProtocolError("Proxy midrank direction must be explicit.")
    normalized: dict[str, float] = {}
    if isinstance(values, Mapping):
        try:
            for raw_source, raw_value in values.items():
                source = str(raw_source)
                if not source or source.strip() != source or source in normalized:
                    raise ProtocolError("Proxy midrank source keys are invalid.")
                if isinstance(raw_value, bool):
                    raise ProtocolError("Proxy midrank values must be finite.")
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ProtocolError("Proxy midrank values must be finite.")
                normalized[source] = value
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("Proxy midrank values must be finite.") from exc
        sources = (
            canonical_source_ids(normalized)
            if source_ids is None
            else canonical_source_ids(source_ids)
        )
        if set(normalized) != set(sources):
            raise ProtocolError("Proxy midrank source grid is incomplete.")
    else:
        if source_ids is None:
            raise ProtocolError(
                "Proxy midrank sequences require explicit source identifiers."
            )
        if isinstance(values, (str, bytes)):
            raise ProtocolError("Proxy midrank values must be a numeric sequence.")
        raw_sources = tuple(source_ids)
        raw_values = tuple(values)
        if len(raw_sources) != len(raw_values):
            raise ProtocolError("Proxy midrank values and sources are misaligned.")
        sources = canonical_source_ids(raw_sources)
        try:
            for raw_source, raw_value in zip(raw_sources, raw_values, strict=True):
                source = str(raw_source)
                if isinstance(raw_value, bool):
                    raise ProtocolError("Proxy midrank values must be finite.")
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ProtocolError("Proxy midrank values must be finite.")
                normalized[source] = value
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("Proxy midrank values must be finite.") from exc

    if len(sources) < 2:
        raise ProtocolError("A normalized proxy ballot requires at least two sources.")
    ordered = sorted(
        sources,
        key=lambda source: (
            normalized[source] if lower_is_better else -normalized[source],
            source,
        ),
    )
    result: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        stop = start + 1
        tied_value = normalized[ordered[start]]
        while stop < len(ordered) and normalized[ordered[stop]] == tied_value:
            stop += 1
        # ``start + 1`` is the first one-based rank and ``stop`` is the last.
        midrank = (float(start + 1) + float(stop)) / 2.0
        normalized_midrank = (midrank - 1.0) / float(len(ordered) - 1)
        for source in ordered[start:stop]:
            result[source] = normalized_midrank
        start = stop
    return MappingProxyType({source: result[source] for source in sources})


def average_replica_scores_before_ballot(
    rows: Iterable[FreshProxyScoreRow],
    *,
    outer_target: str,
    candidate_sources: Iterable[object],
    query_role: str | None = None,
) -> tuple[CaseProxyBallot, ...]:
    """Validate the score grid and construct fixed-replica case ballots."""

    target = _canonical_identifier(outer_target, name="outer target H")
    sources = canonical_source_ids(candidate_sources)
    if target in sources or len(sources) < 3:
        raise ProtocolError(
            "Proxy-policy candidate sources must exclude H and contain at least three sources."
        )
    if query_role not in {None, GLOBAL_PSEUDOQUERY_ROLE, TARGET_SUPPORT_ROLE}:
        raise ProtocolError("Proxy ballot query role is invalid.")
    score_rows = tuple(rows)
    if not score_rows:
        raise ProtocolError("Proxy ballot score grid is empty.")

    by_ballot: dict[
        tuple[str, str, str], dict[tuple[str, int], float]
    ] = defaultdict(dict)
    seen_cells: set[tuple[str, str, str, str, str, int]] = set()
    roles_seen: set[str] = set()
    queries_by_role: dict[str, set[str]] = defaultdict(set)
    cases_by_role_query: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in score_rows:
        if not isinstance(row, FreshProxyScoreRow):
            raise ProtocolError("Proxy ballot rows must use FreshProxyScoreRow.")
        if row.outer_target != target:
            raise ProtocolError("Proxy ballot rows mix outer targets.")
        if query_role is not None and row.query_role != query_role:
            raise ProtocolError("Proxy ballot rows mix query roles.")
        if row.candidate_source not in sources:
            raise ProtocolError("Proxy ballot contains an unknown candidate source.")
        if row.query_role == GLOBAL_PSEUDOQUERY_ROLE:
            if row.query_center not in sources:
                raise ProtocolError("Global pseudoquery q is outside the source universe.")
            expected_candidates = tuple(
                source for source in sources if source != row.query_center
            )
        else:
            if row.query_center != target:
                raise ProtocolError("Target-support query center drifted from H.")
            expected_candidates = sources
        if row.candidate_source not in expected_candidates:
            raise ProtocolError("H/q candidate leakage detected in proxy ballot.")

        cell = (
            row.outer_target,
            row.query_role,
            row.query_center,
            row.case_id,
            row.candidate_source,
            row.training_seed,
        )
        if cell in seen_cells:
            raise ProtocolError("Duplicate proxy score grid cell detected.")
        seen_cells.add(cell)
        ballot_key = (row.query_role, row.query_center, row.case_id)
        by_ballot[ballot_key][
            (row.candidate_source, row.training_seed)
        ] = row.proxy_energy
        roles_seen.add(row.query_role)
        queries_by_role[row.query_role].add(row.query_center)
        cases_by_role_query[(row.query_role, row.query_center)].add(row.case_id)

    expected_roles = {query_role} if query_role is not None else {
        GLOBAL_PSEUDOQUERY_ROLE,
        TARGET_SUPPORT_ROLE,
    }
    if roles_seen != expected_roles:
        raise ProtocolError("Proxy score grid is missing a required query role.")
    if GLOBAL_PSEUDOQUERY_ROLE in expected_roles:
        if queries_by_role[GLOBAL_PSEUDOQUERY_ROLE] != set(sources):
            raise ProtocolError("Global pseudoquery center grid is incomplete.")
        global_case_counts = {
            query: len(
                cases_by_role_query[(GLOBAL_PSEUDOQUERY_ROLE, query)]
            )
            for query in sources
        }
        if len(set(global_case_counts.values())) != 1:
            raise ProtocolError("Global pseudoquery case coverage must be equal.")
    if TARGET_SUPPORT_ROLE in expected_roles and queries_by_role[
        TARGET_SUPPORT_ROLE
    ] != {target}:
        raise ProtocolError("Target-support query grid is incomplete.")

    ballots: list[CaseProxyBallot] = []
    for role, query, case_id in sorted(by_ballot):
        expected_candidates = (
            tuple(source for source in sources if source != query)
            if role == GLOBAL_PSEUDOQUERY_ROLE
            else sources
        )
        expected_cells = {
            (source, seed)
            for source in expected_candidates
            for seed in FIXED_TRAINING_SEEDS
        }
        cells = by_ballot[(role, query, case_id)]
        if set(cells) != expected_cells:
            raise ProtocolError(
                "Proxy score grid has missing, extra, or seed-drifted cells."
            )
        means = {
            source: math.fsum(
                cells[(source, seed)] for seed in FIXED_TRAINING_SEEDS
            )
            / float(len(FIXED_TRAINING_SEEDS))
            for source in expected_candidates
        }
        ballots.append(
            CaseProxyBallot(
                outer_target=target,
                query_role=role,
                query_center=query,
                case_id=case_id,
                candidate_sources=expected_candidates,
                mean_proxy_energy_by_source=means,
                normalized_midrank_by_source=normalized_midranks(means),
            )
        )
    return tuple(ballots)


def build_leave_target_out_global_rank(
    rows: Iterable[FreshProxyScoreRow],
    *,
    outer_target: str,
    candidate_sources: Iterable[object],
) -> ProxyRankSummary:
    """Build G from pseudoqueries whose ballots exclude both H and q."""

    target = _canonical_identifier(outer_target, name="outer target H")
    sources = canonical_source_ids(candidate_sources)
    ballots = average_replica_scores_before_ballot(
        rows,
        outer_target=target,
        candidate_sources=sources,
        query_role=GLOBAL_PSEUDOQUERY_ROLE,
    )
    ranks_by_source: dict[str, list[float]] = {source: [] for source in sources}
    cases_by_query: dict[str, set[str]] = {source: set() for source in sources}
    for ballot in ballots:
        cases_by_query[ballot.query_center].add(ballot.case_id)
        for source, rank in ballot.normalized_midrank_by_source.items():
            ranks_by_source[source].append(rank)
    counts = {source: len(values) for source, values in ranks_by_source.items()}
    if not counts or len(set(counts.values())) != 1 or min(counts.values()) <= 0:
        raise ProtocolError("Global proxy-rank source ballot coverage drifted.")
    means = {
        source: math.fsum(ranks_by_source[source]) / float(counts[source])
        for source in sources
    }
    return ProxyRankSummary(
        outer_target=target,
        policy_id=GLOBAL_POLICY_ID,
        candidate_sources=sources,
        mean_normalized_midrank_by_source=means,
        priority_by_source={source: 1.0 - means[source] for source in sources},
        ballot_count_by_source=counts,
        query_centers=sources,
        case_count_by_query_center={
            query: len(cases_by_query[query]) for query in sources
        },
        ballots=ballots,
        aggregation_semantics=GLOBAL_AGGREGATION_SEMANTICS,
    )


def build_target_support_rank(
    rows: Iterable[FreshProxyScoreRow],
    *,
    outer_target: str,
    candidate_sources: Iterable[object],
) -> ProxyRankSummary:
    """Build S only from the unlabeled support cases of target H."""

    target = _canonical_identifier(outer_target, name="outer target H")
    sources = canonical_source_ids(candidate_sources)
    ballots = average_replica_scores_before_ballot(
        rows,
        outer_target=target,
        candidate_sources=sources,
        query_role=TARGET_SUPPORT_ROLE,
    )
    ranks_by_source: dict[str, list[float]] = {source: [] for source in sources}
    support_cases: set[str] = set()
    for ballot in ballots:
        support_cases.add(ballot.case_id)
        for source, rank in ballot.normalized_midrank_by_source.items():
            ranks_by_source[source].append(rank)
    counts = {source: len(values) for source, values in ranks_by_source.items()}
    if not counts or len(set(counts.values())) != 1 or min(counts.values()) <= 0:
        raise ProtocolError("Target-support proxy-rank source coverage drifted.")
    means = {
        source: math.fsum(ranks_by_source[source]) / float(counts[source])
        for source in sources
    }
    return ProxyRankSummary(
        outer_target=target,
        policy_id=SUPPORT_POLICY_ID,
        candidate_sources=sources,
        mean_normalized_midrank_by_source=means,
        priority_by_source={source: 1.0 - means[source] for source in sources},
        ballot_count_by_source=counts,
        query_centers=(target,),
        case_count_by_query_center={target: len(support_cases)},
        ballots=ballots,
        aggregation_semantics=SUPPORT_AGGREGATION_SEMANTICS,
    )


def _canonical_identifier(value: object, *, name: str) -> str:
    identifier = str(value)
    if not identifier or identifier.strip() != identifier:
        raise ProtocolError(f"Proxy-policy {name} is invalid.")
    return identifier


# Explicit aliases make the aggregation order visible at call sites.
average_replica_energies_before_case_ballots = average_replica_scores_before_ballot
build_global_rank_summary = build_leave_target_out_global_rank
build_support_rank_summary = build_target_support_rank
normalized_true_midranks = normalized_midranks


__all__ = (
    "average_replica_energies_before_case_ballots",
    "average_replica_scores_before_ballot",
    "build_global_rank_summary",
    "build_leave_target_out_global_rank",
    "build_support_rank_summary",
    "build_target_support_rank",
    "normalized_midranks",
    "normalized_true_midranks",
)
