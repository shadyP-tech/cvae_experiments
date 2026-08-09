"""Exact-nine aggregation for globally sealed target-action probabilities."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from ...protocol import ProtocolError
from .core_contracts import (
    AggregatedProbabilityRow,
    SealedProbabilitySurface,
    SeedProbabilityRow,
    canonical_probability_rows,
)
from .core_hashing import canonical_hash
from .scientific_constants import EXPECTED_SEED_PAIR_COUNT, action_ids


def aggregate_exact_nine_probabilities(
    seed_rows: Sequence[SeedProbabilityRow],
) -> SealedProbabilitySurface:
    """Average exactly nine paired-seed probabilities for every action/row."""

    rows = tuple(seed_rows)
    if not rows:
        raise ProtocolError("Cannot aggregate an empty probability surface.")
    store_hashes = {row.probability_store_hash for row in rows}
    if len(store_hashes) != 1:
        raise ProtocolError("Exact-nine rows must share one global store seal.")
    grouped: dict[tuple[str, str, str, str], list[SeedProbabilityRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.target_center, row.case_id, row.sample_id, row.action_id)].append(row)
    identities = sorted({key[:3] for key in grouped})
    for target_center, case_id, sample_id in identities:
        observed_actions = {
            action
            for center, case, sample, action in grouped
            if (center, case, sample) == (target_center, case_id, sample_id)
        }
        expected_actions = set(action_ids(target_center))
        if observed_actions != expected_actions:
            raise ProtocolError(
                "Every target row must contain B and all eight non-target actions."
            )
    aggregated: list[AggregatedProbabilityRow] = []
    for key, values in grouped.items():
        ordered = tuple(sorted(values, key=lambda row: row.seed_pair_ordinal))
        ordinals = tuple(row.seed_pair_ordinal for row in ordered)
        if ordinals != tuple(range(EXPECTED_SEED_PAIR_COUNT)):
            raise ProtocolError("Probability aggregation requires exact seed ordinals 0..8.")
        probabilities = tuple(row.probability for row in ordered)
        mean = sum(probabilities) / EXPECTED_SEED_PAIR_COUNT
        variance = sum((value - mean) ** 2 for value in probabilities) / EXPECTED_SEED_PAIR_COUNT
        aggregated.append(
            AggregatedProbabilityRow(
                target_center=key[0],
                case_id=key[1],
                sample_id=key[2],
                action_id=key[3],
                probability_mean=mean,
                probability_sd=math.sqrt(max(0.0, variance)),
                seed_pair_count=EXPECTED_SEED_PAIR_COUNT,
                seed_probability_hash=canonical_hash(
                    {
                        "target_center": key[0],
                        "case_id": key[1],
                        "sample_id": key[2],
                        "action_id": key[3],
                        "seed_probabilities": list(probabilities),
                    }
                ),
            )
        )
    canonical = canonical_probability_rows(aggregated)
    store_hash = next(iter(store_hashes))
    surface_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_label_aware_probability_surface_v1",
            "probability_store_hash": store_hash,
            "rows": [row.to_payload() for row in canonical],
            "predictions_globally_sealed_before_labels": True,
            "labels_readable_during_materialization": False,
        }
    )
    return SealedProbabilitySurface(
        rows=canonical,
        probability_store_hash=store_hash,
        surface_hash=surface_hash,
    )


__all__ = ("aggregate_exact_nine_probabilities",)
