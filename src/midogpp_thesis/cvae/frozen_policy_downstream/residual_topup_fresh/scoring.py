"""Post-seal scoring with probability ensembles as the fixed primary endpoint."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ...metrics import balanced_accuracy, macro_f1
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    DESCRIPTIVE_SEED_ENDPOINT,
    EXPECTED_ENSEMBLE_METRIC_COUNT,
    EXPECTED_PLAN_CELL_COUNT,
    EXPECTED_SEED_CELL_COUNT,
    EnsembleMetric,
    PRIMARY_ENDPOINT,
    PROBABILITY_THRESHOLD,
    ScoredEvaluation,
    SeedCellMetric,
    expected_action_ids,
)
from .prediction_seal import (
    PredictionSealCapability,
    read_sealed_prediction_snapshot,
)


def score_sealed_predictions(
    capability: PredictionSealCapability,
    labels_by_row_id: Mapping[object, object],
) -> ScoredEvaluation:
    """Open labels only through a complete, integrity-checked prediction seal."""

    state = read_sealed_prediction_snapshot(capability)
    labels = _normalize_labels(labels_by_row_id, state.row_ids_by_target)

    seed_metrics: list[SeedCellMetric] = []
    by_target_action: dict[tuple[str, str], list[object]] = {}
    for cell in state.predictions:
        truth = np.asarray(
            [labels[row_id] for row_id in cell.evaluation_row_ids],
            dtype=np.int64,
        )
        predictions = (cell.probabilities >= PROBABILITY_THRESHOLD).astype(
            np.int64
        )
        seed_metrics.append(
            SeedCellMetric(
                target_center=cell.target_center,
                training_seed=cell.training_seed,
                generation_seed=cell.generation_seed,
                action_id=cell.action_id,
                bacc=float(
                    balanced_accuracy(truth.tolist(), predictions.tolist())
                ),
                macro_f1=float(macro_f1(truth.tolist(), predictions.tolist())),
                evaluation_row_count=len(truth),
                prediction_seal_hash=state.seal_hash,
                endpoint_role=DESCRIPTIVE_SEED_ENDPOINT,
                descriptive_only=True,
            )
        )
        by_target_action.setdefault(
            (cell.target_center, cell.action_id), []
        ).append(cell)
    if len(seed_metrics) != EXPECTED_PLAN_CELL_COUNT:
        raise ProtocolError("Fresh Stage-70 seed-cell metric coverage drifted.")

    ensemble_metrics: list[EnsembleMetric] = []
    for target in CENTERS:
        expected_rows = state.row_ids_by_target[target]
        truth = np.asarray([labels[row_id] for row_id in expected_rows], dtype=np.int64)
        for action_id in expected_action_ids(target):
            cells = by_target_action.get((target, action_id), [])
            cells.sort(
                key=lambda cell: (cell.training_seed, cell.generation_seed)
            )
            observed_seeds = {
                (cell.training_seed, cell.generation_seed) for cell in cells
            }
            if (
                len(cells) != EXPECTED_SEED_CELL_COUNT
                or len(observed_seeds) != EXPECTED_SEED_CELL_COUNT
                or any(cell.evaluation_row_ids != expected_rows for cell in cells)
            ):
                raise ProtocolError(
                    "Fresh Stage-70 probability-ensemble seed coverage drifted."
                )
            mean_probability = np.mean(
                np.stack(
                    [
                        np.asarray(cell.probabilities, dtype=np.float64)
                        for cell in cells
                    ],
                    axis=0,
                ),
                axis=0,
            )
            prediction = (mean_probability >= PROBABILITY_THRESHOLD).astype(
                np.int64
            )
            ensemble_metrics.append(
                EnsembleMetric(
                    target_center=target,
                    action_id=action_id,
                    bacc=float(
                        balanced_accuracy(truth.tolist(), prediction.tolist())
                    ),
                    macro_f1=float(
                        macro_f1(truth.tolist(), prediction.tolist())
                    ),
                    evaluation_row_count=len(truth),
                    seed_cell_count=len(cells),
                    prediction_seal_hash=state.seal_hash,
                    endpoint=PRIMARY_ENDPOINT,
                    primary_endpoint=True,
                )
            )
    if len(ensemble_metrics) != EXPECTED_ENSEMBLE_METRIC_COUNT:
        raise ProtocolError("Fresh Stage-70 ensemble metric coverage drifted.")
    return ScoredEvaluation(
        seed_cell_metrics=tuple(seed_metrics),
        ensemble_metrics=tuple(ensemble_metrics),
        prediction_seal_hash=state.seal_hash,
        primary_endpoint=PRIMARY_ENDPOINT,
        labels_used_for_scoring_only=True,
    )


score_predictions = score_sealed_predictions


def _normalize_labels(
    values: Mapping[object, object],
    rows_by_target: Mapping[str, tuple[str, ...]],
) -> dict[str, int]:
    if not isinstance(values, Mapping):
        raise ProtocolError("Fresh Stage-70 labels must be row keyed.")
    raw: dict[str, object] = {}
    normalized_keys = {str(key) for key in values}
    if normalized_keys == set(CENTERS) and all(
        isinstance(value, Mapping) for value in values.values()
    ):
        by_target = {str(key): value for key, value in values.items()}
        for target in CENTERS:
            target_values = by_target[target]
            assert isinstance(target_values, Mapping)
            observed = {str(key): value for key, value in target_values.items()}
            if set(observed) != set(rows_by_target[target]):
                raise ProtocolError(
                    "Fresh Stage-70 labels do not exactly cover target rows."
                )
            if set(raw).intersection(observed):
                raise ProtocolError("Fresh Stage-70 label rows duplicate across targets.")
            raw.update(observed)
    else:
        raw = {str(key): value for key, value in values.items()}

    expected = {
        row_id for target in CENTERS for row_id in rows_by_target[target]
    }
    if set(raw) != expected:
        raise ProtocolError(
            "Fresh Stage-70 labels must exactly cover the sealed rows, with no extras."
        )
    labels: dict[str, int] = {}
    for row_id, raw_label in raw.items():
        if isinstance(raw_label, bool):
            raise ProtocolError("Fresh Stage-70 labels must be binary integers.")
        try:
            label = int(raw_label)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Fresh Stage-70 labels must be binary integers.") from exc
        try:
            numerically_exact = float(raw_label) == float(label)
        except (TypeError, ValueError, OverflowError):
            numerically_exact = False
        if label not in {0, 1} or not numerically_exact:
            raise ProtocolError("Fresh Stage-70 labels must be binary integers.")
        labels[row_id] = label
    for target in CENTERS:
        target_labels = {labels[row_id] for row_id in rows_by_target[target]}
        if target_labels != {0, 1}:
            raise ProtocolError("Fresh Stage-70 target lacks both scoring classes.")
    return labels


__all__ = (
    "score_predictions",
    "score_sealed_predictions",
)
