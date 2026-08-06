"""Compact prediction-array and index persistence for source-inner utility."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import EXPERIMENT_ID
from .prediction import PredictionPass, array_sha256


PREDICTION_ARRAY_MEMBER = "arrays/candidate_predictions.npz"
EVALUATION_ROW_MEMBER = "tables/evaluation_rows.csv"
FIT_TABLE_MEMBER = "tables/classifier_fits.csv"

EVALUATION_ROW_COLUMNS = (
    "schema_version",
    "row_ordinal",
    "manifest_row_index",
    "sample_id",
    "case_id",
    "center",
    "split",
    "cache_shard_path",
    "cache_row_index",
    "label_present",
)


def evaluation_row_table(predictions: PredictionPass) -> tuple[dict[str, object], ...]:
    rows = []
    for item in predictions.evaluation_rows:
        rows.append(
            {
                "schema_version": "midogpp_uniform_b_v2_utility_eval_row_v1",
                "row_ordinal": int(getattr(item, "row_ordinal")),
                "manifest_row_index": int(getattr(item, "manifest_row_index")),
                "sample_id": str(getattr(item, "sample_id")),
                "case_id": str(getattr(item, "case_id")),
                "center": str(getattr(item, "center")),
                "split": str(getattr(item, "split")),
                "cache_shard_path": str(getattr(item, "cache_shard_path")),
                "cache_row_index": int(getattr(item, "cache_row_index")),
                "label_present": False,
            }
        )
    return tuple(rows)


def write_prediction_arrays(path: str | Path, predictions: PredictionPass) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        y_pred=np.asarray(predictions.y_pred, dtype=np.uint8),
        prob_pos=np.asarray(predictions.prob_pos, dtype=np.float32),
    )


def read_prediction_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            if set(payload.files) != {"y_pred", "prob_pos"}:
                raise ProtocolError(
                    "Prediction array member keys drifted or expose undeclared data."
                )
            y_pred = np.asarray(payload["y_pred"])
            prob_pos = np.asarray(payload["prob_pos"])
    except ProtocolError:
        raise
    except (OSError, ValueError) as exc:
        raise ProtocolError("Cannot read compact candidate prediction arrays.") from exc
    if y_pred.dtype != np.uint8 or prob_pos.dtype != np.float32:
        raise ProtocolError("Candidate prediction array dtypes drifted.")
    return y_pred, prob_pos


def prediction_index_payload(
    predictions: PredictionPass,
    *,
    prediction_file_sha256: str,
) -> dict[str, object]:
    fit_ids = [str(row["fit_id"]) for row in predictions.fit_rows]
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_candidate_prediction_index_v1",
        "experiment_id": EXPERIMENT_ID,
        "array_member": PREDICTION_ARRAY_MEMBER,
        "format": "numpy_npz_compressed",
        "allowed_array_keys": ["y_pred", "prob_pos"],
        "labels_stored": False,
        "prediction_dtype": "uint8",
        "probability_dtype": "float32",
        "prediction_shape": list(predictions.y_pred.shape),
        "probability_shape": list(predictions.prob_pos.shape),
        "prediction_array_sha256": array_sha256(predictions.y_pred),
        "probability_array_sha256": array_sha256(predictions.prob_pos),
        "prediction_file_sha256": prediction_file_sha256,
        "fit_count": len(predictions.fit_rows),
        "eval_row_count": len(predictions.evaluation_rows),
        "fit_order_hash": stable_hash(fit_ids),
        "evaluation_order_hash": predictions.evaluation_order_hash,
        "fit_index_member": FIT_TABLE_MEMBER,
        "evaluation_row_index_member": EVALUATION_ROW_MEMBER,
        "predict_every_eval_row_once_per_classifier": True,
        "pseudo_target_slicing_performed_after_prediction": True,
        "eval_labels_available_to_fit_or_predict": False,
    }
    payload["prediction_index_hash"] = stable_hash(payload)
    return payload


__all__ = (
    "EVALUATION_ROW_COLUMNS",
    "EVALUATION_ROW_MEMBER",
    "FIT_TABLE_MEMBER",
    "PREDICTION_ARRAY_MEMBER",
    "evaluation_row_table",
    "prediction_index_payload",
    "read_prediction_arrays",
    "write_prediction_arrays",
)
