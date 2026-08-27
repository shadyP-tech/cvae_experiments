"""Exact-tie-preserving SCEPTRE proxy ranking and raw routing."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    CandidateMenu,
    ExactBFallback,
    FamilyProxyScore,
    INVALID_EVIDENCE_REASON,
    NONFINITE_EVIDENCE_REASON,
    RankedWinnerSet,
    RawRoute,
    RouteDecision,
    UNSUPPORTED_EVIDENCE_REASON,
)


def normalized_true_midranks(
    values_by_source: Mapping[str, float],
    *,
    candidate_sources: Sequence[str] | None = None,
    lower_is_better: bool = True,
) -> Mapping[str, float]:
    """Return normalized true midranks while retaining every exact tie."""

    if type(lower_is_better) is not bool or not lower_is_better:
        raise ProtocolError("SCEPTRE proxy ranking must be explicitly lower-is-better.")
    sources = (
        tuple(str(source) for source in values_by_source)
        if candidate_sources is None
        else tuple(str(source) for source in candidate_sources)
    )
    if len(sources) < 2 or len(set(sources)) != len(sources):
        raise ProtocolError("SCEPTRE midrank candidate identifiers are invalid.")
    if set(values_by_source) != set(sources):
        raise ProtocolError("SCEPTRE midrank score grid is incomplete.")
    values: dict[str, float] = {}
    for source in sources:
        raw = values_by_source[source]
        if isinstance(raw, bool):
            raise ProtocolError("SCEPTRE midrank scores must be finite.")
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("SCEPTRE midrank scores must be finite.") from exc
        if not math.isfinite(value):
            raise ProtocolError("SCEPTRE midrank scores must be finite.")
        values[source] = value

    ordered = sorted(sources, key=lambda source: (values[source], source))
    result: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        stop = start + 1
        tied_value = values[ordered[start]]
        while stop < len(ordered) and values[ordered[stop]] == tied_value:
            stop += 1
        midrank = (float(start + 1) + float(stop)) / 2.0
        normalized = (midrank - 1.0) / float(len(ordered) - 1)
        for source in ordered[start:stop]:
            result[source] = normalized
        start = stop
    return MappingProxyType({source: result[source] for source in sources})


def rank_family_proxy_scores(
    candidate_menu: CandidateMenu,
    family_scores: Sequence[FamilyProxyScore] | Mapping[str, FamilyProxyScore],
) -> RankedWinnerSet:
    """Rank all eight source families using their three-replica means."""

    if isinstance(family_scores, Mapping):
        if set(family_scores) != set(candidate_menu.candidate_sources):
            raise ProtocolError("SCEPTRE family-score mapping is incomplete.")
        rows = tuple(family_scores[source] for source in candidate_menu.candidate_sources)
    else:
        rows = tuple(family_scores)
    if len(rows) != len(candidate_menu.candidate_sources) or any(
        not isinstance(row, FamilyProxyScore) for row in rows
    ):
        raise ProtocolError("SCEPTRE ranking requires one proxy score per source family.")
    by_source: dict[str, FamilyProxyScore] = {}
    for row in rows:
        if row.target_center != candidate_menu.target_center:
            raise ProtocolError("SCEPTRE ranking mixed target centers.")
        if row.source_center in by_source:
            raise ProtocolError("SCEPTRE ranking contains a duplicate source score.")
        by_source[row.source_center] = row
    if set(by_source) != set(candidate_menu.candidate_sources):
        raise ProtocolError("SCEPTRE ranking source-family coverage is incomplete.")

    values = {
        source: float(by_source[source].mean_proxy_energy)
        for source in candidate_menu.candidate_sources
    }
    midranks = normalized_true_midranks(
        values,
        candidate_sources=candidate_menu.candidate_sources,
        lower_is_better=True,
    )
    best = min(values.values())
    winners = tuple(
        source for source in candidate_menu.candidate_sources if values[source] == best
    )
    return RankedWinnerSet(
        target_center=candidate_menu.target_center,
        candidate_sources=candidate_menu.candidate_sources,
        mean_proxy_energy_by_source=values,
        normalized_midrank_by_source=midranks,
        score_hash_by_source={
            source: by_source[source].score_hash
            for source in candidate_menu.candidate_sources
        },
        winner_sources=winners,
        lower_is_better=True,
    )


def route_unique_winner_or_exact_b(
    candidate_menu: CandidateMenu,
    ranking: RankedWinnerSet,
) -> RouteDecision:
    """Issue a family route only for a unique winner; otherwise return exact B."""

    if (
        ranking.target_center != candidate_menu.target_center
        or ranking.candidate_sources != candidate_menu.candidate_sources
    ):
        raise ProtocolError("SCEPTRE ranking does not belong to the candidate menu.")
    if ranking.has_unique_winner:
        return RawRoute(
            target_center=candidate_menu.target_center,
            candidate_sources=candidate_menu.candidate_sources,
            selected_source_center=ranking.winner_sources[0],
            candidate_menu_hash=candidate_menu.menu_hash,
            ranking_hash=ranking.ranking_hash,
        )
    return ExactBFallback(
        target_center=candidate_menu.target_center,
        candidate_sources=candidate_menu.candidate_sources,
        winner_sources=ranking.winner_sources,
        candidate_menu_hash=candidate_menu.menu_hash,
        ranking_hash=ranking.ranking_hash,
    )


def route_raw_proxy_evidence_or_exact_b(
    candidate_menu: CandidateMenu,
    training_replica_scores_by_source: Mapping[str, Mapping[int, float]],
) -> RouteDecision:
    """Convert raw label-free evidence, falling back on every invalid path."""

    if not isinstance(candidate_menu, CandidateMenu):
        raise ProtocolError("SCEPTRE safe router requires a sealed candidate menu.")
    try:
        raw = dict(training_replica_scores_by_source)
    except (TypeError, ValueError):
        return _evidence_fallback(candidate_menu, INVALID_EVIDENCE_REASON)
    if set(raw) != set(candidate_menu.candidate_sources):
        return _evidence_fallback(candidate_menu, UNSUPPORTED_EVIDENCE_REASON)
    if _contains_nonfinite(raw):
        return _evidence_fallback(candidate_menu, NONFINITE_EVIDENCE_REASON)
    try:
        scores = tuple(
            FamilyProxyScore(
                target_center=candidate_menu.target_center,
                source_center=source,
                training_replica_scores=raw[source],
            )
            for source in candidate_menu.candidate_sources
        )
        ranking = rank_family_proxy_scores(candidate_menu, scores)
    except (ProtocolError, TypeError, ValueError, OverflowError):
        return _evidence_fallback(candidate_menu, INVALID_EVIDENCE_REASON)
    return route_unique_winner_or_exact_b(candidate_menu, ranking)


def _contains_nonfinite(raw: Mapping[str, object]) -> bool:
    for value in raw.values():
        if not isinstance(value, Mapping):
            continue
        for score in value.values():
            if isinstance(score, bool):
                continue
            try:
                if not math.isfinite(float(score)):
                    return True
            except (TypeError, ValueError, OverflowError):
                continue
    return False


def _evidence_fallback(
    candidate_menu: CandidateMenu,
    reason: str,
) -> ExactBFallback:
    evidence_receipt = stable_hash(
        {
            "schema_version": "midogpp_sceptre_inadmissible_evidence_v1",
            "candidate_menu_hash": candidate_menu.menu_hash,
            "reason": reason,
            "no_candidate_selected": True,
        }
    )
    return ExactBFallback(
        target_center=candidate_menu.target_center,
        candidate_sources=candidate_menu.candidate_sources,
        winner_sources=(),
        candidate_menu_hash=candidate_menu.menu_hash,
        ranking_hash=evidence_receipt,
        reason=reason,
    )


# Compact aliases for orchestration code.
rank_proxy_scores = rank_family_proxy_scores
select_raw_route = route_unique_winner_or_exact_b
select_raw_evidence_or_exact_b = route_raw_proxy_evidence_or_exact_b
true_midranks = normalized_true_midranks


__all__ = (
    "normalized_true_midranks",
    "rank_family_proxy_scores",
    "rank_proxy_scores",
    "route_unique_winner_or_exact_b",
    "route_raw_proxy_evidence_or_exact_b",
    "select_raw_evidence_or_exact_b",
    "select_raw_route",
    "true_midranks",
)
