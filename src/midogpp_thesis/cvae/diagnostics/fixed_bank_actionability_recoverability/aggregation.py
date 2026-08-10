"""Pure exact-nine probability aggregation with complete action coverage."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Sequence

from ...protocol import ProtocolError
from .actions import actions_for_target
from .constants import SEED_PAIR_ORDINALS
from .contracts import (
    AggregatedProbabilityRow,
    ExactNineProbabilitySurface,
    SeedProbabilityRow,
)
from .hashing import canonical_hash


def aggregate_exact_nine_probabilities(
    seed_rows: Sequence[SeedProbabilityRow],
) -> ExactNineProbabilitySurface:
    """Average seed repetitions; never expose a seed as a selectable action."""

    rows = tuple(seed_rows)
    if not rows or any(not isinstance(row, SeedProbabilityRow) for row in rows):
        raise ProtocolError("Exact-nine aggregation requires typed, non-empty seed rows.")
    store_hashes = {row.probability_store_hash for row in rows}
    if len(store_hashes) != 1:
        raise ProtocolError("Seed probabilities must share one globally sealed store hash.")
    grouped: dict[tuple[str, str, str, str], list[SeedProbabilityRow]] = defaultdict(list)
    actions_by_sample: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[row.row_key].append(row)
        actions_by_sample[row.row_key[:3]].add(row.action_id)

    # Validate action coverage from the one-pass sample index.  Scanning every
    # grouped action key once per sample would be quadratic at the canonical
    # 9,928-sample / 178,704-action-cell scale.
    for (target, _case_id, _sample_id), observed in sorted(
        actions_by_sample.items()
    ):
        expected = {action.action_id for action in actions_for_target(target)}
        if observed != expected:
            raise ProtocolError(
                "Every sample must cover B, U and all A0/A1 source actions."
            )

    aggregated: list[AggregatedProbabilityRow] = []
    for key in sorted(grouped):
        values = tuple(sorted(grouped[key], key=lambda row: row.seed_pair_ordinal))
        ordinals = tuple(row.seed_pair_ordinal for row in values)
        if ordinals != SEED_PAIR_ORDINALS:
            raise ProtocolError("Every action/sample cell requires seed ordinals zero through eight exactly once.")
        probabilities = tuple(row.probability for row in values)
        mean = math.fsum(probabilities) / len(SEED_PAIR_ORDINALS)
        variance = math.fsum((value - mean) ** 2 for value in probabilities) / len(
            SEED_PAIR_ORDINALS
        )
        aggregated.append(
            AggregatedProbabilityRow(
                target_center=key[0],
                case_id=key[1],
                sample_id=key[2],
                action_id=key[3],
                probability_mean=mean,
                probability_sd=math.sqrt(max(0.0, variance)),
                seed_pair_count=len(SEED_PAIR_ORDINALS),
                seed_probability_hash=canonical_hash(
                    {
                        "target_center": key[0],
                        "case_id": key[1],
                        "sample_id": key[2],
                        "action_id": key[3],
                        "seed_pair_ordinals": list(SEED_PAIR_ORDINALS),
                        "probabilities": list(probabilities),
                    }
                ),
            )
        )
    canonical = tuple(sorted(aggregated, key=lambda row: row.row_key))
    store_hash = next(iter(store_hashes))
    surface_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_actionability_exact_nine_surface_v1",
            "probability_store_hash": store_hash,
            "rows": [row.to_payload() for row in canonical],
            "predictions_sealed_before_labels": True,
        }
    )
    return ExactNineProbabilitySurface(canonical, store_hash, surface_hash)


__all__ = (
    "AggregatedProbabilityRow",
    "ExactNineProbabilitySurface",
    "SeedProbabilityRow",
    "aggregate_exact_nine_probabilities",
)
