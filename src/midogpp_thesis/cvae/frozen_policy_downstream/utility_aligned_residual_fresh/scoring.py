"""Metric-only scoring after the complete utility-aligned prediction seal."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ...metrics import balanced_accuracy, macro_f1
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    DESCRIPTIVE_SEED_ENDPOINT,
    EXPECTED_ENSEMBLE_METRIC_COUNT,
    EXPECTED_LOGICAL_PREDICTION_COUNT,
    EXPECTED_SEED_CELL_COUNT,
    EnsembleMetric,
    PROBABILITY_THRESHOLD,
    ScoredEvaluation,
    SeedCellMetric,
    expected_action_ids,
)
from .ensemble_adapter import mean_exact_nine_probabilities
from .prediction_seal import (
    PredictionSealCapability,
    read_sealed_prediction_snapshot,
)


def score_sealed_predictions(
    capability: PredictionSealCapability,
    labels_by_row_id: Mapping[object, object],
) -> ScoredEvaluation:
    """Evaluate all-nine-seed probability ensembles without seed selection."""

    state = read_sealed_prediction_snapshot(capability)
    labels = _normalize_labels(labels_by_row_id, state.row_ids_by_target)
    seed_rows: list[SeedCellMetric] = []
    by_target_action: dict[tuple[str, str], list[object]] = {}
    for cell in state.predictions:
        truth = np.asarray(
            [labels[row_id] for row_id in cell.evaluation_row_ids], dtype=np.int64
        )
        predicted = (cell.probabilities >= PROBABILITY_THRESHOLD).astype(np.int64)
        seed_rows.append(
            SeedCellMetric(
                target_center=cell.target_center,
                training_seed=cell.training_seed,
                generation_seed=cell.generation_seed,
                action_id=cell.action_id,
                bacc=float(balanced_accuracy(truth.tolist(), predicted.tolist())),
                macro_f1=float(macro_f1(truth.tolist(), predicted.tolist())),
                evaluation_row_count=len(truth),
                prediction_seal_hash=state.seal_hash,
                endpoint_role=DESCRIPTIVE_SEED_ENDPOINT,
                descriptive_only=True,
            )
        )
        by_target_action.setdefault((cell.target_center, cell.action_id), []).append(
            cell
        )
    if len(seed_rows) != EXPECTED_LOGICAL_PREDICTION_COUNT:
        raise ProtocolError("Utility-aligned seed-cell metric coverage drifted.")

    ensemble_rows: list[EnsembleMetric] = []
    for target in CENTERS:
        expected_rows = state.row_ids_by_target[target]
        truth = np.asarray([labels[row_id] for row_id in expected_rows], dtype=np.int64)
        for action_id in expected_action_ids(target):
            cells = by_target_action.get((target, action_id), [])
            cells.sort(key=lambda cell: (cell.training_seed, cell.generation_seed))
            if (
                len(cells) != EXPECTED_SEED_CELL_COUNT
                or len(
                    {
                        (cell.training_seed, cell.generation_seed)
                        for cell in cells
                    }
                )
                != EXPECTED_SEED_CELL_COUNT
                or any(cell.evaluation_row_ids != expected_rows for cell in cells)
            ):
                raise ProtocolError("Utility-aligned ensemble seed coverage drifted.")
            mean_probability = mean_exact_nine_probabilities(
                tuple(np.asarray(cell.probabilities, dtype=np.float64) for cell in cells)
            )
            prediction = (mean_probability >= PROBABILITY_THRESHOLD).astype(np.int64)
            ensemble_rows.append(
                EnsembleMetric(
                    target_center=target,
                    action_id=action_id,
                    bacc=float(balanced_accuracy(truth.tolist(), prediction.tolist())),
                    macro_f1=float(macro_f1(truth.tolist(), prediction.tolist())),
                    evaluation_row_count=len(truth),
                    seed_cell_count=len(cells),
                    prediction_seal_hash=state.seal_hash,
                )
            )
    if len(ensemble_rows) != EXPECTED_ENSEMBLE_METRIC_COUNT:
        raise ProtocolError("Utility-aligned ensemble metric coverage drifted.")
    return ScoredEvaluation(
        seed_cell_metrics=tuple(seed_rows),
        ensemble_metrics=tuple(ensemble_rows),
        prediction_seal_hash=state.seal_hash,
        labels_used_for_scoring_only=True,
    )


score_predictions = score_sealed_predictions


def _normalize_labels(
    values: Mapping[object, object],
    rows_by_target: Mapping[str, tuple[str, ...]],
) -> dict[str, int]:
    if not isinstance(values, Mapping):
        raise ProtocolError("Utility-aligned labels must be row keyed.")
    raw = {str(key): value for key, value in values.items()}
    expected = {row for target in CENTERS for row in rows_by_target[target]}
    if set(raw) != expected:
        raise ProtocolError("Utility-aligned labels must exactly cover sealed rows.")
    labels: dict[str, int] = {}
    for row_id, raw_label in raw.items():
        if isinstance(raw_label, bool):
            raise ProtocolError("Utility-aligned labels must be binary integers.")
        try:
            label = int(raw_label)
            exact = float(raw_label) == float(label)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("Utility-aligned labels must be binary integers.") from exc
        if label not in {0, 1} or not exact:
            raise ProtocolError("Utility-aligned labels must be binary integers.")
        labels[row_id] = label
    for target in CENTERS:
        if {labels[row] for row in rows_by_target[target]} != {0, 1}:
            raise ProtocolError("Every utility-aligned target must contain both classes.")
    return labels


__all__ = ("score_predictions", "score_sealed_predictions")
