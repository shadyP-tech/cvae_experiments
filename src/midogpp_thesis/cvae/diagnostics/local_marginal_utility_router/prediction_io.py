"""Compact label-free storage for the global development prediction pass."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError


DEVELOPMENT_ARRAY_MEMBER = "arrays/development_predictions.npz"

PREDICTION_INDEX_COLUMNS = (
    "schema_version",
    "phase",
    "cell_ordinal",
    "outer_target",
    "query_center",
    "action_id",
    "arm_role",
    "boosted_source",
    "training_seed",
    "generation_seed",
    "candidate_sources_json",
    "weights_json",
    "allocations_json",
    "epsilon",
    "effective_source_count",
    "shuffle_seed_by_class_json",
    "composition_hash",
    "classifier_config_hash",
    "scaler_state_hash",
    "classifier_classes_json",
    "classifier_n_iter_json",
    "classifier_converged",
    "evaluation_row_ids_json",
    "evaluation_row_identity_hash",
    "prediction_offset_start",
    "prediction_offset_stop",
    "prediction_sha256",
    "probability_sha256",
    "labels_available_to_fit_or_predict",
    "seed_selection_performed",
)


@dataclass(frozen=True)
class FlatPredictionStore:
    """Flat prediction arrays and exact metadata-bound slices."""

    y_pred: np.ndarray
    prob_pos: np.ndarray
    index_rows: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        predictions = np.asarray(self.y_pred)
        probabilities = np.asarray(self.prob_pos)
        if (
            predictions.ndim != 1
            or probabilities.shape != predictions.shape
            or predictions.dtype != np.uint8
            or probabilities.dtype != np.float32
            or not np.isin(predictions, [0, 1]).all()
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
        ):
            raise ProtocolError("Local-utility prediction arrays are malformed.")
        cursor = 0
        for ordinal, row in enumerate(self.index_rows):
            start = _integer(row.get("prediction_offset_start"), "prediction start")
            stop = _integer(row.get("prediction_offset_stop"), "prediction stop")
            if (
                _integer(row.get("cell_ordinal"), "cell ordinal") != ordinal
                or start != cursor
                or stop <= start
                or stop > len(predictions)
            ):
                raise ProtocolError(
                    "Local-utility prediction offsets are not contiguous."
                )
            if array_sha256(predictions[start:stop]) != row.get("prediction_sha256"):
                raise ProtocolError("Local-utility prediction slice hash drifted.")
            if array_sha256(probabilities[start:stop]) != row.get("probability_sha256"):
                raise ProtocolError("Local-utility probability slice hash drifted.")
            cursor = stop
        if cursor != len(predictions) or (not self.index_rows and len(predictions)):
            raise ProtocolError("Local-utility prediction index coverage drifted.")

    def slice_for(self, row: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
        start = _integer(row.get("prediction_offset_start"), "prediction start")
        stop = _integer(row.get("prediction_offset_stop"), "prediction stop")
        return self.y_pred[start:stop], self.prob_pos[start:stop]


class PredictionAccumulator:
    """Append-only prediction builder whose API cannot accept labels."""

    def __init__(self) -> None:
        self._predictions: list[np.ndarray] = []
        self._probabilities: list[np.ndarray] = []
        self._index: list[dict[str, object]] = []
        self._length = 0

    def append(
        self,
        *,
        predictions: Sequence[int] | np.ndarray,
        probabilities: Sequence[float] | np.ndarray,
        metadata: Mapping[str, object],
    ) -> dict[str, object]:
        y_pred = np.ascontiguousarray(predictions, dtype=np.uint8)
        prob_pos = np.ascontiguousarray(probabilities, dtype=np.float32)
        if (
            y_pred.ndim != 1
            or not len(y_pred)
            or prob_pos.shape != y_pred.shape
            or not np.isin(y_pred, [0, 1]).all()
            or not np.isfinite(prob_pos).all()
            or np.any(prob_pos < 0.0)
            or np.any(prob_pos > 1.0)
        ):
            raise ProtocolError("Local-utility classifier output is malformed.")
        forbidden = {str(key).lower() for key in metadata}.intersection(
            {"label", "labels", "y_true", "target", "class_label"}
        )
        if forbidden:
            raise ProtocolError(
                "Local-utility prediction metadata attempted to persist labels."
            )
        start = self._length
        stop = start + len(y_pred)
        row = {
            **dict(metadata),
            "cell_ordinal": len(self._index),
            "prediction_offset_start": start,
            "prediction_offset_stop": stop,
            "prediction_sha256": array_sha256(y_pred),
            "probability_sha256": array_sha256(prob_pos),
            "labels_available_to_fit_or_predict": False,
            "seed_selection_performed": False,
        }
        missing = set(PREDICTION_INDEX_COLUMNS).difference(row)
        extra = set(row).difference(PREDICTION_INDEX_COLUMNS)
        if missing or extra:
            raise ProtocolError(
                "Local-utility prediction-index schema drifted: "
                f"missing={sorted(missing)!r}, extra={sorted(extra)!r}."
            )
        if row.get("phase") != "development_utility_surface":
            raise ProtocolError("Local-utility prediction phase drifted.")
        self._predictions.append(y_pred)
        self._probabilities.append(prob_pos)
        self._index.append(row)
        self._length = stop
        return row

    def finish(self) -> FlatPredictionStore:
        if not self._index:
            raise ProtocolError("Local-utility prediction pass produced no cells.")
        return FlatPredictionStore(
            y_pred=np.concatenate(self._predictions).astype(np.uint8, copy=False),
            prob_pos=np.concatenate(self._probabilities).astype(np.float32, copy=False),
            index_rows=tuple(self._index),
        )


def write_prediction_store(path: str | Path, store: FlatPredictionStore) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        y_pred=np.asarray(store.y_pred, dtype=np.uint8),
        prob_pos=np.asarray(store.prob_pos, dtype=np.float32),
    )


def read_prediction_store(
    path: str | Path,
    index_rows: Sequence[Mapping[str, object]],
) -> FlatPredictionStore:
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            if set(payload.files) != {"y_pred", "prob_pos"}:
                raise ProtocolError("Local-utility NPZ member keys drifted.")
            y_pred = np.asarray(payload["y_pred"])
            prob_pos = np.asarray(payload["prob_pos"])
    except ProtocolError:
        raise
    except (OSError, ValueError) as exc:
        raise ProtocolError("Cannot read local-utility prediction arrays.") from exc
    return FlatPredictionStore(
        y_pred=y_pred,
        prob_pos=prob_pos,
        index_rows=tuple(index_rows),
    )


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: object, role: str) -> int:
    if isinstance(value, bool):
        raise ProtocolError(f"Local-utility {role} must be an integer.")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Local-utility {role} must be an integer.") from exc
    return parsed


__all__ = (
    "DEVELOPMENT_ARRAY_MEMBER",
    "PREDICTION_INDEX_COLUMNS",
    "FlatPredictionStore",
    "PredictionAccumulator",
    "array_sha256",
    "read_prediction_store",
    "sha256_file",
    "write_prediction_store",
)
