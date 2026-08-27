"""Label-free proxy-score aggregation for SCEPTRE source families.

These scores are compatibility energies, not exact CVAE NELBO values and not
downstream utilities.  The module deliberately has no label-bearing inputs.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from ...expert_bank.uniform_b_v2_promotion.contracts import TRAINING_SEEDS
from ...protocol import ProtocolError
from .contracts import CandidateMenu, FamilyProxyScore


def aggregate_training_replica_scores(
    *,
    target_center: str,
    source_center: str,
    scores_by_training_seed: Mapping[int, float] | Iterable[tuple[int, float]],
) -> FamilyProxyScore:
    """Average exactly the 17/42/101 training replicas of one family."""

    try:
        rows = tuple(scores_by_training_seed.items())  # type: ignore[union-attr]
    except AttributeError:
        rows = tuple(scores_by_training_seed)
    if len(rows) != len(TRAINING_SEEDS):
        raise ProtocolError(
            "SCEPTRE proxy aggregation requires exactly three training replicas."
        )
    parsed: dict[int, float] = {}
    for raw_seed, raw_score in rows:
        if isinstance(raw_seed, bool):
            raise ProtocolError("SCEPTRE proxy training seed is invalid.")
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("SCEPTRE proxy training seed is invalid.") from exc
        if seed in parsed:
            raise ProtocolError("SCEPTRE proxy aggregation contains a duplicate seed.")
        parsed[seed] = raw_score
    return FamilyProxyScore(
        target_center=str(target_center),
        source_center=str(source_center),
        training_replica_scores=parsed,
        exact_nelbo=False,
        labels_consumed=False,
    )


def aggregate_family_proxy_score(
    candidate_menu: CandidateMenu,
    source_center: str,
    scores_by_training_seed: Mapping[int, float] | Iterable[tuple[int, float]],
) -> FamilyProxyScore:
    """Aggregate one admitted source family's label-free evidence."""

    source = str(source_center)
    if source not in candidate_menu.candidate_sources:
        raise ProtocolError("SCEPTRE proxy score refers to a source outside the menu.")
    return aggregate_training_replica_scores(
        target_center=candidate_menu.target_center,
        source_center=source,
        scores_by_training_seed=scores_by_training_seed,
    )


def aggregate_menu_proxy_scores(
    candidate_menu: CandidateMenu,
    scores_by_source_and_training_seed: Mapping[
        str, Mapping[int, float] | Iterable[tuple[int, float]]
    ],
) -> tuple[FamilyProxyScore, ...]:
    """Aggregate an exact candidate menu without source or seed selection."""

    if set(scores_by_source_and_training_seed) != set(candidate_menu.candidate_sources):
        raise ProtocolError("SCEPTRE proxy-score source grid is incomplete.")
    return tuple(
        aggregate_family_proxy_score(
            candidate_menu,
            source,
            scores_by_source_and_training_seed[source],
        )
        for source in candidate_menu.candidate_sources
    )


# The longer alias makes the aggregation order explicit at call sites.
average_training_replicas_before_source_ranking = aggregate_training_replica_scores


__all__ = (
    "aggregate_family_proxy_score",
    "aggregate_menu_proxy_scores",
    "aggregate_training_replica_scores",
    "average_training_replicas_before_source_ranking",
)
