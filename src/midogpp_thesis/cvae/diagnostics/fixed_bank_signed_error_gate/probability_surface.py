"""Exact-nine aggregation for the signed diagnostic's sealed probability surface."""

from __future__ import annotations

import math
from typing import Sequence

from ...protocol import ProtocolError
from ..fixed_bank_hierarchical_residual_stacker.contracts import (
    SampleActionProbability,
)
from ..fixed_bank_hierarchical_residual_stacker.core_hashing import canonical_hash
from ..fixed_bank_hierarchical_residual_stacker.experiment_contracts import (
    SEED_PAIR_COUNT,
)
from .execution_adapter import RuntimeSeedProbabilityRow


def aggregate_exact_nine_probabilities(
    seed_rows: Sequence[RuntimeSeedProbabilityRow],
) -> tuple[tuple[SampleActionProbability, ...], str]:
    """Average all nine frozen seed pairs before any feature or threshold use."""

    grouped: dict[
        tuple[str, str, str, str], list[RuntimeSeedProbabilityRow]
    ] = {}
    store_hashes: set[str] = set()
    for row in seed_rows:
        grouped.setdefault(
            (row.target_center, row.case_id, row.sample_id, row.action_id), []
        ).append(row)
        store_hashes.add(row.probability_store_hash)
    if not grouped or len(store_hashes) != 1:
        raise ProtocolError("Signed-error seed probability surface is empty or unbound.")

    output: list[SampleActionProbability] = []
    for key, rows in sorted(grouped.items()):
        ordered = tuple(sorted(rows, key=lambda row: row.seed_pair_ordinal))
        if (
            len(ordered) != SEED_PAIR_COUNT
            or tuple(row.seed_pair_ordinal for row in ordered)
            != tuple(range(SEED_PAIR_COUNT))
        ):
            raise ProtocolError("Signed-error probabilities require every exact-nine seed cell.")
        output.append(
            SampleActionProbability(
                *key,
                probability=math.fsum(row.probability for row in ordered)
                / SEED_PAIR_COUNT,
            )
        )
    probabilities = tuple(output)
    surface_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_signed_error_probability_surface_v1",
            "probability_store_hash": next(iter(store_hashes)),
            "row_hashes": [row.row_hash for row in probabilities],
            "seed_pair_count": SEED_PAIR_COUNT,
            "averaged_before_feature_or_threshold_use": True,
            "labels_used": False,
            "target_expert_used": False,
        }
    )
    return probabilities, surface_hash


__all__ = ("aggregate_exact_nine_probabilities",)
