"""Post-seal scoring of the exact additive-tail utility surface."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...metrics import balanced_accuracy
from ...protocol import ProtocolError
from ...routing.utility_aligned import (
    ExactTailUtilityRow,
    ExactTailUtilitySurface,
    validate_exact_tail_utility_rows,
)
from .development_label_access import OpenedDevelopmentLabels
from .development_prediction_contracts import (
    EXPECTED_EXACT_TAIL_UTILITY_ROW_COUNT,
    INNER_BASE_ACTION_ID,
    expected_utility_keys,
    inner_tail_action_id,
)
from .development_seal import DevelopmentPredictionCapability


def score_exact_tail_development_utility(
    capability: DevelopmentPredictionCapability,
    labels: OpenedDevelopmentLabels,
    partitions: object,
) -> tuple[ExactTailUtilityRow, ...]:
    """Score 4,536 typed rows; labels never flow into prediction or routing."""

    seal = capability.seal
    if (
        labels.prediction_seal_hash != seal.prediction_seal_hash
        or labels.manifest_sha256 != seal.development_manifest_sha256
    ):
        raise ProtocolError("Stage-90 scoring labels bind another prediction seal.")
    evaluation_by_center = getattr(partitions, "evaluation_rows_by_center", None)
    if evaluation_by_center is None:
        raise ProtocolError("Stage-90 scoring partitions are absent.")
    cells = capability.store.cell_by_key
    output: list[ExactTailUtilityRow] = []
    for outer, query, source, training_seed, generation_seed in expected_utility_keys():
        truth = np.asarray(labels.labels_by_center[query], dtype=np.uint8)
        rows = tuple(evaluation_by_center[query])
        if len(truth) != len(rows) or set(int(value) for value in truth) != {0, 1}:
            raise ProtocolError("Stage-90 scoring truth geometry drifted.")
        base_key = (
            outer,
            query,
            INNER_BASE_ACTION_ID,
            training_seed,
            generation_seed,
        )
        tail_key = (
            outer,
            query,
            inner_tail_action_id(source),
            training_seed,
            generation_seed,
        )
        base = capability.store.prediction_for(base_key)
        tail = capability.store.prediction_for(tail_key)
        if base.shape != truth.shape or tail.shape != truth.shape:
            raise ProtocolError("Stage-90 prediction/label geometry drifted.")
        base_cell, tail_cell = cells[base_key], cells[tail_key]
        output.append(
            ExactTailUtilityRow(
                outer_target_id=outer,
                query_id=query,
                candidate_source=source,
                training_seed=training_seed,
                generation_seed=generation_seed,
                candidate_source_count=7,
                support_partition_hash=seal.support_partition_hash_by_center[query],
                evaluation_partition_hash=seal.evaluation_row_hash_by_center[query],
                prediction_seal_hash=seal.prediction_seal_hash,
                base_prediction_hash=str(base_cell["prediction_sha256"]),
                tail_prediction_hash=str(tail_cell["prediction_sha256"]),
                base_bacc=float(
                    balanced_accuracy(truth.tolist(), base.tolist())
                ),
                tail_bacc=float(
                    balanced_accuracy(truth.tolist(), tail.tolist())
                ),
                support_eval_disjoint=True,
                predictions_sealed_before_labels=True,
                source_expert_frozen=True,
                target_labels_used_for_routing=False,
            )
        )
    if len(output) != EXPECTED_EXACT_TAIL_UTILITY_ROW_COUNT:
        raise ProtocolError("Stage-90 exact-tail utility coverage drifted.")
    validate_exact_tail_utility_rows(output)
    return tuple(output)


def validate_scored_exact_tail_surface(
    rows: Sequence[ExactTailUtilityRow],
) -> ExactTailUtilitySurface:
    if len(rows) != EXPECTED_EXACT_TAIL_UTILITY_ROW_COUNT:
        raise ProtocolError("Stage-90 exact-tail scored row count drifted.")
    return validate_exact_tail_utility_rows(rows)


__all__ = (
    "score_exact_tail_development_utility",
    "validate_scored_exact_tail_surface",
)
