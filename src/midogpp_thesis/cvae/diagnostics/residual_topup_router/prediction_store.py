"""Flat, hash-bound persistence for every residual top-up prediction cell."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .artifact_io import atomic_save_npz, atomic_write_csv_rows
from .contracts import EXPECTED_PREDICTION_CELL_COUNT, MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT


PREDICTION_ARRAY_MEMBER = "arrays/all_action_predictions.npz"
PREDICTION_INDEX_MEMBER = "tables/prediction_index.csv"

PREDICTION_INDEX_COLUMNS = (
    "schema_version",
    "config_contract_hash",
    "generation_lock_hash",
    "source_cache_lock_hash",
    "router_plan_lock_hash",
    "cell_ordinal",
    "phase",
    "outer_target",
    "query_center",
    "action_id",
    "arm_role",
    "budget_role",
    "training_seed",
    "generation_seed",
    "candidate_sources_json",
    "source_stream_ids_json",
    "expert_lock_hashes_json",
    "base_per_source",
    "base_total_per_class",
    "topup_total_per_class",
    "final_total_per_class",
    "topup_counts_json",
    "final_counts_by_class_json",
    "final_weights_by_class_json",
    "windows_by_class_json",
    "shuffle_seed_by_class_json",
    "action_hash",
    "allocation_hash",
    "window_hash",
    "composition_hash",
    "composition_output_sha256",
    "classifier_config_hash",
    "scaler_state_hash",
    "classifier_n_iter_json",
    "classifier_converged",
    "evaluation_row_ids_json",
    "evaluation_row_identity_hash",
    "prediction_offset_start",
    "prediction_offset_stop",
    "prediction_sha256",
    "probability_sha256",
    "plan_hash",
    "labels_available_to_fit_or_predict",
    "support_labels_used",
    "target_expert_excluded",
    "outer_and_query_experts_excluded",
    "seed_selection_performed",
    "fit_aliased_by_composition_hash",
    "selection_source",
    "claim_role",
)


@dataclass(frozen=True)
class PredictionStore:
    y_pred: np.ndarray
    prob_pos: np.ndarray
    index_rows: tuple[Mapping[str, object], ...]
    unique_classifier_fit_count: int

    def __post_init__(self) -> None:
        predictions = np.asarray(self.y_pred)
        probabilities = np.asarray(self.prob_pos)
        if (
            predictions.ndim != 1
            or probabilities.shape != predictions.shape
            or predictions.dtype != np.uint8
            or probabilities.dtype != np.float32
            or not np.isin(predictions, (0, 1)).all()
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
            or len(self.index_rows) != EXPECTED_PREDICTION_CELL_COUNT
            or not 1 <= int(self.unique_classifier_fit_count) <= MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT
        ):
            raise ProtocolError("Residual top-up prediction store is malformed.")
        cursor = 0
        for ordinal, row in enumerate(self.index_rows):
            start = _integer(row.get("prediction_offset_start"))
            stop = _integer(row.get("prediction_offset_stop"))
            if (
                set(row) != set(PREDICTION_INDEX_COLUMNS)
                or _integer(row.get("cell_ordinal")) != ordinal
                or start != cursor
                or stop <= start
                or _array_sha256(predictions[start:stop]) != row.get("prediction_sha256")
                or _array_sha256(probabilities[start:stop]) != row.get("probability_sha256")
            ):
                raise ProtocolError("Residual top-up prediction offsets or hashes drifted.")
            cursor = stop
        if cursor != len(predictions):
            raise ProtocolError("Residual top-up prediction coverage drifted.")
        object.__setattr__(self, "y_pred", predictions)
        object.__setattr__(self, "prob_pos", probabilities)
        object.__setattr__(self, "unique_classifier_fit_count", int(self.unique_classifier_fit_count))

    def slice_for(self, row: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
        start = _integer(row["prediction_offset_start"])
        stop = _integer(row["prediction_offset_stop"])
        return self.y_pred[start:stop], self.prob_pos[start:stop]


def assemble_prediction_store(
    ordered_cells: Sequence[Mapping[str, object]],
    *,
    unique_classifier_fit_count: int,
) -> PredictionStore:
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    cursor = 0
    for cell in ordered_cells:
        pred = np.asarray(cell["predictions"], dtype=np.uint8)
        prob = np.asarray(cell["probabilities"], dtype=np.float32)
        metadata = dict(cell["metadata"])
        if pred.ndim != 1 or prob.shape != pred.shape or not len(pred):
            raise ProtocolError("Residual top-up checkpoint cell is malformed.")
        stop = cursor + len(pred)
        row = {
            **metadata,
            "cell_ordinal": len(rows),
            "prediction_offset_start": cursor,
            "prediction_offset_stop": stop,
            "prediction_sha256": _array_sha256(pred),
            "probability_sha256": _array_sha256(prob),
        }
        if set(row) != set(PREDICTION_INDEX_COLUMNS):
            raise ProtocolError("Residual top-up prediction index schema drifted.")
        predictions.append(pred)
        probabilities.append(prob)
        rows.append(row)
        cursor = stop
    return PredictionStore(
        y_pred=np.concatenate(predictions).astype(np.uint8, copy=False),
        prob_pos=np.concatenate(probabilities).astype(np.float32, copy=False),
        index_rows=tuple(rows),
        unique_classifier_fit_count=unique_classifier_fit_count,
    )


def write_prediction_store(root: Path, store: PredictionStore) -> None:
    atomic_save_npz(
        root / PREDICTION_ARRAY_MEMBER,
        y_pred=store.y_pred,
        prob_pos=store.prob_pos,
        unique_classifier_fit_count=np.asarray(store.unique_classifier_fit_count, dtype=np.int64),
    )
    atomic_write_csv_rows(
        root / PREDICTION_INDEX_MEMBER,
        store.index_rows,
        columns=PREDICTION_INDEX_COLUMNS,
    )


def read_prediction_store(root: Path) -> PredictionStore:
    try:
        with np.load(root / PREDICTION_ARRAY_MEMBER, allow_pickle=False) as payload:
            if set(payload.files) != {"y_pred", "prob_pos", "unique_classifier_fit_count"}:
                raise ProtocolError("Residual top-up prediction NPZ keys drifted.")
            predictions = np.asarray(payload["y_pred"])
            probabilities = np.asarray(payload["prob_pos"])
            fit_count = int(np.asarray(payload["unique_classifier_fit_count"]).item())
    except (OSError, ValueError) as exc:
        raise ProtocolError("Residual top-up prediction arrays are unreadable.") from exc
    try:
        with (root / PREDICTION_INDEX_MEMBER).open(newline="", encoding="utf-8") as handle:
            rows = tuple(dict(row) for row in csv.DictReader(handle))
    except OSError as exc:
        raise ProtocolError("Residual top-up prediction index is unreadable.") from exc
    return PredictionStore(predictions, probabilities, rows, fit_count)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _integer(value: object) -> int:
    if isinstance(value, bool):
        raise ProtocolError("Residual top-up integer field is invalid.")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Residual top-up integer field is invalid.") from exc


__all__ = (
    "PREDICTION_ARRAY_MEMBER",
    "PREDICTION_INDEX_COLUMNS",
    "PREDICTION_INDEX_MEMBER",
    "PredictionStore",
    "assemble_prediction_store",
    "read_prediction_store",
    "write_prediction_store",
)
