"""Fixed-support replica aggregation and true normalized-midrank ballots."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Iterable, Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    GLOBAL_QUERY_ROLE,
    SUPPORT_QUERY_ROLE,
    TRAINING_SEEDS,
    CaseProxyBallot,
    ProxyRankSummary,
    ProxyScoreRow,
    TargetRankSurface,
    candidate_sources,
    global_candidate_sources,
)
from .partitions import CaseOOFSurface


def normalized_midranks(
    values: Mapping[object, float],
    *,
    lower_is_better: bool = True,
) -> Mapping[str, float]:
    """Return true normalized midranks without breaking score ties."""

    if type(lower_is_better) is not bool or not isinstance(values, Mapping):
        raise ProtocolError("Case-OOF midrank direction/input is invalid.")
    numeric: dict[str, float] = {}
    try:
        for raw_source, raw_value in values.items():
            source = str(raw_source)
            if not source or source.strip() != source or source in numeric:
                raise ProtocolError("Case-OOF midrank source keys are invalid.")
            if isinstance(raw_value, bool):
                raise ProtocolError("Case-OOF midrank values must be finite.")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ProtocolError("Case-OOF midrank values must be finite.")
            numeric[source] = value
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Case-OOF midrank values must be finite.") from exc
    sources = tuple(sorted(numeric))
    if len(sources) < 2:
        raise ProtocolError("Case-OOF ballot requires at least two sources.")
    ordered = sorted(
        sources,
        key=lambda source: (
            numeric[source] if lower_is_better else -numeric[source],
            source,
        ),
    )
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        stop = start + 1
        tied_value = numeric[ordered[start]]
        while stop < len(ordered) and numeric[ordered[stop]] == tied_value:
            stop += 1
        midrank = (float(start + 1) + float(stop)) / 2.0
        normalized = (midrank - 1.0) / float(len(ordered) - 1)
        for source in ordered[start:stop]:
            ranks[source] = normalized
        start = stop
    return MappingProxyType({source: ranks[source] for source in sources})


def build_global_rank_summary(
    rows: Iterable[ProxyScoreRow],
    *,
    outer_target: str,
    crossfit: CaseOOFSurface,
) -> ProxyRankSummary:
    """Build G only from fixed support cases q != H with H/q excluded."""

    target = str(outer_target)
    sources = candidate_sources(target)
    ballots: list[CaseProxyBallot] = []
    score_rows = tuple(rows)
    expected_cells: set[tuple[str, str, str, int]] = set()
    for query in sources:
        support_cases = tuple(
            sorted(
                {
                    str(row.case_id)
                    for row in crossfit.fixed_support_rows_by_center[query]
                }
            )
        )
        for case_id in support_cases:
            candidates = global_candidate_sources(target, query)
            ballots.append(
                _case_ballot(
                    score_rows,
                    outer_target=target,
                    query_role=GLOBAL_QUERY_ROLE,
                    query_center=query,
                    case_id=case_id,
                    candidates=candidates,
                )
            )
            expected_cells.update(
                (query, case_id, source, seed)
                for source in candidates
                for seed in TRAINING_SEEDS
            )
    observed_cells = {
        (
            row.query_center,
            row.case_id,
            row.candidate_source,
            row.training_seed,
        )
        for row in score_rows
    }
    if observed_cells != expected_cells or len(observed_cells) != len(score_rows):
        raise ProtocolError("Case-OOF G score grid is incomplete or duplicated.")
    return _aggregate_ballots(
        target,
        GLOBAL_QUERY_ROLE,
        sources,
        tuple(ballots),
    )


def build_support_rank_summary(
    rows: Iterable[ProxyScoreRow],
    *,
    outer_target: str,
    crossfit: CaseOOFSurface,
) -> ProxyRankSummary:
    """Build one S_H from exactly the two fixed support cases of H."""

    target = str(outer_target)
    sources = candidate_sources(target)
    support_cases = tuple(
        sorted(
            {
                str(row.case_id)
                for row in crossfit.fixed_support_rows_by_center[target]
            }
        )
    )
    score_rows = tuple(rows)
    ballots = tuple(
        _case_ballot(
            score_rows,
            outer_target=target,
            query_role=SUPPORT_QUERY_ROLE,
            query_center=target,
            case_id=case_id,
            candidates=sources,
        )
        for case_id in support_cases
    )
    expected_cells = {
        (target, case_id, source, seed)
        for case_id in support_cases
        for source in sources
        for seed in TRAINING_SEEDS
    }
    observed_cells = {
        (
            row.query_center,
            row.case_id,
            row.candidate_source,
            row.training_seed,
        )
        for row in score_rows
    }
    if observed_cells != expected_cells or len(observed_cells) != len(score_rows):
        raise ProtocolError("Case-OOF S score grid is incomplete or duplicated.")
    return _aggregate_ballots(
        target,
        SUPPORT_QUERY_ROLE,
        sources,
        ballots,
    )


def build_rank_surface(
    rows: Iterable[ProxyScoreRow],
    crossfit: CaseOOFSurface,
) -> Mapping[str, TargetRankSurface]:
    """Build complete fixed G/S summaries for all nine targets."""

    if not isinstance(crossfit, CaseOOFSurface):
        raise ProtocolError("Case-OOF rank surface requires a locked fold surface.")
    score_rows = tuple(rows)
    if not score_rows or any(
        not isinstance(row, ProxyScoreRow) for row in score_rows
    ):
        raise ProtocolError("Case-OOF proxy-score rows are invalid or empty.")
    rows_by_target_role: dict[tuple[str, str], list[ProxyScoreRow]] = {
        (target, role): []
        for target in CENTERS
        for role in (GLOBAL_QUERY_ROLE, SUPPORT_QUERY_ROLE)
    }
    for row in score_rows:
        key = (row.outer_target, row.query_role)
        if key not in rows_by_target_role:
            raise ProtocolError("Case-OOF proxy row target/role is unknown.")
        rows_by_target_role[key].append(row)
    surfaces: dict[str, TargetRankSurface] = {}
    for target in CENTERS:
        global_summary = build_global_rank_summary(
            rows_by_target_role[(target, GLOBAL_QUERY_ROLE)],
            outer_target=target,
            crossfit=crossfit,
        )
        support_summary = build_support_rank_summary(
            rows_by_target_role[(target, SUPPORT_QUERY_ROLE)],
            outer_target=target,
            crossfit=crossfit,
        )
        surfaces[target] = TargetRankSurface(
            outer_target=target,
            global_summary=global_summary,
            support_summary=support_summary,
        )
    return MappingProxyType(surfaces)


def _case_ballot(
    rows: tuple[ProxyScoreRow, ...],
    *,
    outer_target: str,
    query_role: str,
    query_center: str,
    case_id: str,
    candidates: tuple[str, ...],
) -> CaseProxyBallot:
    matching = tuple(
        row
        for row in rows
        if row.outer_target == outer_target
        and row.query_role == query_role
        and row.query_center == query_center
        and row.case_id == case_id
    )
    by_cell: dict[tuple[str, int], ProxyScoreRow] = {}
    for row in matching:
        key = (row.candidate_source, row.training_seed)
        if key in by_cell:
            raise ProtocolError("Case-OOF proxy ballot cell is duplicated.")
        by_cell[key] = row
    expected = {
        (source, seed) for source in candidates for seed in TRAINING_SEEDS
    }
    if set(by_cell) != expected:
        raise ProtocolError("Case-OOF proxy ballot grid drifted.")
    if len({row.row_count for row in matching}) != 1:
        raise ProtocolError("Case-OOF proxy ballot row count drifted by replica.")
    means = {
        source: math.fsum(
            by_cell[(source, seed)].proxy_energy for seed in TRAINING_SEEDS
        )
        / float(len(TRAINING_SEEDS))
        for source in candidates
    }
    midranks = normalized_midranks(means, lower_is_better=True)
    unhashed = {
        "schema_version": "midogpp_residual_topup_case_oof_proxy_ballot_v1",
        "outer_target": outer_target,
        "query_role": query_role,
        "query_center": query_center,
        "case_id": case_id,
        "candidate_sources": list(candidates),
        "training_seeds_averaged_before_ballot": list(TRAINING_SEEDS),
        "mean_proxy_energy_by_source": means,
        "normalized_midrank_by_source": dict(midranks),
        "lower_is_better": True,
        "labels_used": False,
        "evaluation_embeddings_used": False,
    }
    return CaseProxyBallot(
        outer_target=outer_target,
        query_role=query_role,
        query_center=query_center,
        case_id=case_id,
        candidate_sources=candidates,
        mean_proxy_energy_by_source=means,
        normalized_midrank_by_source=midranks,
        ballot_hash=stable_hash(unhashed),
    )


def _aggregate_ballots(
    outer_target: str,
    query_role: str,
    sources: tuple[str, ...],
    ballots: tuple[CaseProxyBallot, ...],
) -> ProxyRankSummary:
    ranks: dict[str, list[float]] = {source: [] for source in sources}
    for ballot in ballots:
        for source, value in ballot.normalized_midrank_by_source.items():
            ranks[source].append(float(value))
    counts = {source: len(values) for source, values in ranks.items()}
    if len(set(counts.values())) != 1 or min(counts.values(), default=0) <= 0:
        raise ProtocolError("Case-OOF proxy rank has unequal source coverage.")
    means = {
        source: math.fsum(ranks[source]) / float(counts[source])
        for source in sources
    }
    priorities = {source: 1.0 - means[source] for source in sources}
    unhashed = {
        "schema_version": "midogpp_residual_topup_case_oof_rank_summary_v1",
        "outer_target": outer_target,
        "query_role": query_role,
        "candidate_sources": list(sources),
        "mean_normalized_midrank_by_source": means,
        "priority_by_source": priorities,
        "priority_transform": "one_minus_mean_normalized_midrank",
        "ballot_count_by_source": counts,
        "ballot_hashes": [ballot.ballot_hash for ballot in ballots],
        "replicas_averaged_before_each_ballot": True,
        "true_midranks_preserved": True,
    }
    return ProxyRankSummary(
        outer_target=outer_target,
        query_role=query_role,
        candidate_sources=sources,
        mean_normalized_midrank_by_source=means,
        priority_by_source=priorities,
        ballot_count_by_source=counts,
        ballots=ballots,
        rank_hash=stable_hash(unhashed),
    )


__all__ = (
    "build_global_rank_summary",
    "build_rank_surface",
    "build_support_rank_summary",
    "normalized_midranks",
)
