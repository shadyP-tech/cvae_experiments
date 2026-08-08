"""Validation and pairwise views of sealed exact additive-tail utility."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product
import math
from typing import Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from .row_contracts import (
    INNER_CANDIDATE_COUNT,
    TARGET_CANDIDATE_COUNT,
    ExactTailUtilityRow,
)
from .surface_contracts import (
    ExactTailUtilitySurface,
    PairwisePreference,
)


def validate_exact_tail_utility_rows(
    rows: Sequence[ExactTailUtilityRow],
) -> ExactTailUtilitySurface:
    """Validate complete source-inner H/q/e/seed geometry and prediction seals.

    A valid MIDOG++ outer target contributes exactly ``8 * 7 * 3 * 3``
    observations.  Candidate utilities retain all paired seed cells; validation
    rejects missing cells, duplicated cells, target/query experts, and
    per-candidate drift in the paired base prediction.
    """

    if not rows or any(not isinstance(row, ExactTailUtilityRow) for row in rows):
        raise ProtocolError("Exact-tail utility requires typed, nonempty rows.")
    ordered = tuple(sorted(rows, key=lambda row: row.row_key))
    row_keys = tuple(row.row_key for row in ordered)
    if len(set(row_keys)) != len(row_keys):
        raise ProtocolError("Exact-tail utility rows contain duplicate cells.")
    by_outer: dict[str, list[ExactTailUtilityRow]] = defaultdict(list)
    for row in ordered:
        by_outer[row.outer_target_id].append(row)
    domain_universe: tuple[str, ...] | None = None
    for outer_target, outer_rows in sorted(by_outer.items()):
        query_ids = tuple(sorted({row.query_id for row in outer_rows}))
        if len(query_ids) != TARGET_CANDIDATE_COUNT or outer_target in query_ids:
            raise ProtocolError(
                "Each outer target requires exactly eight non-target pseudoqueries."
            )
        observed_universe = tuple(sorted((outer_target, *query_ids)))
        if domain_universe is None:
            domain_universe = observed_universe
        elif observed_universe != domain_universe:
            raise ProtocolError("Outer-target domain universes drifted.")
        expected_seed_pairs = set(product(TRAINING_SEEDS, GENERATION_SEEDS))
        for query in query_ids:
            query_rows = [row for row in outer_rows if row.query_id == query]
            expected_sources = tuple(source for source in query_ids if source != query)
            if len(expected_sources) != INNER_CANDIDATE_COUNT:
                raise ProtocolError("Source-inner candidate cardinality drifted.")
            if len({row.support_partition_hash for row in query_rows}) != 1:
                raise ProtocolError("Support partition drifted across one pseudoquery.")
            if len({row.evaluation_partition_hash for row in query_rows}) != 1:
                raise ProtocolError("Evaluation partition drifted across one pseudoquery.")
            for source in expected_sources:
                source_rows = [
                    row for row in query_rows if row.candidate_source == source
                ]
                observed_seed_pairs = {
                    (row.training_seed, row.generation_seed) for row in source_rows
                }
                if observed_seed_pairs != expected_seed_pairs:
                    raise ProtocolError(
                        "Every exact-tail candidate requires all nine paired seed cells."
                    )
            if {row.candidate_source for row in query_rows} != set(expected_sources):
                raise ProtocolError("Exact-tail candidate set is incomplete or illegal.")
            for training_seed, generation_seed in sorted(expected_seed_pairs):
                replicate_rows = [
                    row
                    for row in query_rows
                    if row.training_seed == training_seed
                    and row.generation_seed == generation_seed
                ]
                if len(replicate_rows) != INNER_CANDIDATE_COUNT:
                    raise ProtocolError("Exact-tail replicate geometry drifted.")
                if len({row.base_prediction_hash for row in replicate_rows}) != 1:
                    raise ProtocolError(
                        "Paired candidates must share one exact-base prediction."
                    )
                if len({row.base_bacc for row in replicate_rows}) != 1:
                    raise ProtocolError("Paired candidates must share exact-base BACC.")
    payload = {
        "schema_version": "midogpp_utility_aligned_exact_tail_surface_v1",
        "outer_target_ids": sorted(by_outer),
        "row_count": len(ordered),
        "row_hashes": [row.row_hash for row in ordered],
        "seed_pairs": [list(pair) for pair in product(TRAINING_SEEDS, GENERATION_SEEDS)],
        "candidate_counts": {
            "source_inner": INNER_CANDIDATE_COUNT,
            "fresh_target": TARGET_CANDIDATE_COUNT,
        },
        "target_labels_used_for_routing": False,
        "seed_selection_performed": False,
    }
    return ExactTailUtilitySurface(
        rows=ordered,
        outer_target_ids=tuple(sorted(by_outer)),
        row_keys=row_keys,
        surface_hash=canonical_sha256(payload),
    )


def build_pairwise_preferences(
    utility: ExactTailUtilitySurface | Sequence[ExactTailUtilityRow],
    *,
    tie_tolerance: float = 0.0,
) -> tuple[PairwisePreference, ...]:
    """Build cardinality-invariant paired preferences without seed selection."""

    surface = (
        utility
        if isinstance(utility, ExactTailUtilitySurface)
        else validate_exact_tail_utility_rows(utility)
    )
    try:
        tolerance = float(tie_tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Pairwise tie tolerance must be finite and nonnegative.") from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ProtocolError("Pairwise tie tolerance must be finite and nonnegative.")
    by_replicate: dict[tuple[str, str, int, int], list[ExactTailUtilityRow]] = (
        defaultdict(list)
    )
    for row in surface.rows:
        by_replicate[
            (
                row.outer_target_id,
                row.query_id,
                row.training_seed,
                row.generation_seed,
            )
        ].append(row)
    preferences: list[PairwisePreference] = []
    for group_key, group_rows in sorted(by_replicate.items()):
        by_source = {row.candidate_source: row for row in group_rows}
        if len(by_source) != INNER_CANDIDATE_COUNT:
            raise ProtocolError("Pairwise preference group is not a seven-source list.")
        outer, query, training_seed, generation_seed = group_key
        for left, right in combinations(sorted(by_source), 2):
            left_delta = by_source[left].utility_delta
            right_delta = by_source[right].utility_delta
            margin = left_delta - right_delta
            preferred = None
            if margin > tolerance:
                preferred = left
            elif margin < -tolerance:
                preferred = right
            payload = {
                "schema_version": "midogpp_utility_aligned_pairwise_preference_v1",
                "outer_target_id": outer,
                "query_id": query,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "left_source": left,
                "right_source": right,
                "left_utility_delta": left_delta,
                "right_utility_delta": right_delta,
                "preferred_source": preferred,
                "utility_margin": margin,
                "tie_tolerance": tolerance,
            }
            preferences.append(
                PairwisePreference(
                    outer_target_id=outer,
                    query_id=query,
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                    left_source=left,
                    right_source=right,
                    left_utility_delta=left_delta,
                    right_utility_delta=right_delta,
                    preferred_source=preferred,
                    utility_margin=margin,
                    preference_hash=canonical_sha256(payload),
                )
            )
    return tuple(preferences)


__all__ = ("build_pairwise_preferences", "validate_exact_tail_utility_rows")
