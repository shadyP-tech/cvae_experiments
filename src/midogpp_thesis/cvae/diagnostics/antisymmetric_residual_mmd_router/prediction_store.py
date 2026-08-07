"""Persistence and frozen-plan binding for cross-fit predictions."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np

from ...generation.generation import derived_composition_seed
from ...protocol import ProtocolError
from ._prediction_common import (
    atomic_save_npz,
    integer,
    is_hash,
    parse_json_value,
    sha256_array,
    truthy,
    require_mapping,
)
from .artifact_io import atomic_write_csv_rows
from .composition import arm_plan_payload, validate_plan
from .contracts import (
    ARM_ROLES,
    CENTERS,
    CONTROL_ARM,
    EXPECTED_PREDICTION_CELL_COUNT,
    EXPECTED_SEED_CELL_COUNT,
    GENERATION_SEEDS,
    MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT,
    ROUTED_ARM,
    TRAINING_SEEDS,
    candidate_sources,
    row_identity_hash,
)
from .partitions import CrossfitSurface

if TYPE_CHECKING:  # pragma: no cover
    from .config import AntisymmetricResidualMMDDiagnosticConfig


CROSSFIT_PREDICTION_ARRAY_MEMBER = "arrays/target_predictions.npz"
CROSSFIT_PREDICTION_INDEX_MEMBER = "tables/target_prediction_index.csv"

CROSSFIT_PREDICTION_INDEX_COLUMNS = (
    "schema_version",
    "config_contract_hash",
    "generation_lock_hash",
    "source_products_lock_hash",
    "router_plan_lock_hash",
    "cell_ordinal",
    "fold_ordinal",
    "fold_id",
    "fold_hash",
    "target_center",
    "heldout_case_id",
    "arm_role",
    "training_seed",
    "generation_seed",
    "candidate_sources_json",
    "weights_by_class_json",
    "allocations_by_class_json",
    "shuffle_seed_by_class_json",
    "composition_hash",
    "classifier_config_hash",
    "scaler_state_hash",
    "classifier_n_iter_json",
    "classifier_converged",
    "router_support_row_identity_hash",
    "evaluation_row_ids_json",
    "evaluation_row_identity_hash",
    "prediction_offset_start",
    "prediction_offset_stop",
    "prediction_sha256",
    "probability_sha256",
    "plan_hash",
    "heldout_case_excluded_from_route",
    "labels_available_to_fit_or_predict",
    "support_labels_used",
    "seed_selection_performed",
    "control_fit_aliased",
)


@dataclass(frozen=True)
class CrossfitPredictionStore:
    """Flat predictions plus a hash-bound row index for all 468 cells."""

    y_pred: np.ndarray
    prob_pos: np.ndarray
    index_rows: tuple[Mapping[str, object], ...]
    unique_classifier_fit_count: int

    def __post_init__(self) -> None:
        predictions = np.asarray(self.y_pred)
        probabilities = np.asarray(self.prob_pos)
        rows = tuple(self.index_rows)
        minimum_fit_count = len(CENTERS) * EXPECTED_SEED_CELL_COUNT
        if (
            predictions.ndim != 1
            or probabilities.shape != predictions.shape
            or predictions.dtype != np.uint8
            or probabilities.dtype != np.float32
            or not np.isin(predictions, (0, 1)).all()
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
            or len(rows) != EXPECTED_PREDICTION_CELL_COUNT
            or not minimum_fit_count
            <= int(self.unique_classifier_fit_count)
            <= MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT
        ):
            raise ProtocolError(
                "Antisymmetric cross-fit prediction store is malformed."
            )
        cursor = 0
        for ordinal, row in enumerate(rows):
            start = integer(row.get("prediction_offset_start"), "prediction offset")
            stop = integer(row.get("prediction_offset_stop"), "prediction offset")
            if (
                set(row) != set(CROSSFIT_PREDICTION_INDEX_COLUMNS)
                or integer(row.get("cell_ordinal"), "cell ordinal") != ordinal
                or start != cursor
                or stop <= start
                or sha256_array(predictions[start:stop])
                != row.get("prediction_sha256")
                or sha256_array(probabilities[start:stop])
                != row.get("probability_sha256")
            ):
                raise ProtocolError(
                    "Antisymmetric cross-fit prediction offsets or hashes drifted."
                )
            cursor = stop
        if cursor != len(predictions):
            raise ProtocolError(
                "Antisymmetric cross-fit prediction coverage drifted."
            )
        object.__setattr__(self, "y_pred", predictions)
        object.__setattr__(self, "prob_pos", probabilities)
        object.__setattr__(self, "index_rows", rows)
        object.__setattr__(
            self,
            "unique_classifier_fit_count",
            int(self.unique_classifier_fit_count),
        )

    def slice_for(
        self, row: Mapping[str, object]
    ) -> tuple[np.ndarray, np.ndarray]:
        start = integer(row["prediction_offset_start"], "prediction offset")
        stop = integer(row["prediction_offset_stop"], "prediction offset")
        return self.y_pred[start:stop], self.prob_pos[start:stop]


def assemble_crossfit_prediction_store(
    completed: Mapping[tuple[str, int, int], Mapping[str, object]],
    crossfit: CrossfitSurface,
) -> CrossfitPredictionStore:
    """Flatten validated target-task checkpoints in frozen cell order."""

    prediction_arrays: list[np.ndarray] = []
    probability_arrays: list[np.ndarray] = []
    index_rows: list[dict[str, object]] = []
    cursor = 0
    unique_fit_count = 0
    counted_task_keys: set[tuple[str, int, int]] = set()
    for fold in crossfit.folds:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                task_key = (fold.target_center, training_seed, generation_seed)
                cell = completed[task_key]
                if task_key not in counted_task_keys:
                    unique_fit_count += integer(
                        cell["unique_classifier_fit_count"],
                        "unique classifier fit count",
                    )
                    counted_task_keys.add(task_key)
                for arm in ARM_ROLES:
                    prefix = f"fold_{fold.fold_ordinal}_{arm}"
                    predictions = np.asarray(
                        cell[f"{prefix}_predictions"], dtype=np.uint8
                    )
                    probabilities = np.asarray(
                        cell[f"{prefix}_probabilities"], dtype=np.float32
                    )
                    metadata = dict(
                        require_mapping(cell, f"{prefix}_metadata")
                    )
                    stop = cursor + len(predictions)
                    row = {
                        **metadata,
                        "cell_ordinal": len(index_rows),
                        "prediction_offset_start": cursor,
                        "prediction_offset_stop": stop,
                        "prediction_sha256": sha256_array(predictions),
                        "probability_sha256": sha256_array(probabilities),
                    }
                    if set(row) != set(CROSSFIT_PREDICTION_INDEX_COLUMNS):
                        raise ProtocolError(
                            "Antisymmetric cross-fit prediction-index schema drifted."
                        )
                    prediction_arrays.append(predictions)
                    probability_arrays.append(probabilities)
                    index_rows.append(row)
                    cursor = stop

    return CrossfitPredictionStore(
        y_pred=np.concatenate(prediction_arrays).astype(np.uint8, copy=False),
        prob_pos=np.concatenate(probability_arrays).astype(
            np.float32, copy=False
        ),
        index_rows=tuple(index_rows),
        unique_classifier_fit_count=unique_fit_count,
    )


def read_crossfit_prediction_store(
    array_path: Path,
    index_path: Path,
) -> CrossfitPredictionStore:
    rows = tuple(_read_csv(index_path))
    try:
        with np.load(array_path, allow_pickle=False) as payload:
            if set(payload.files) != {
                "y_pred",
                "prob_pos",
                "unique_classifier_fit_count",
            }:
                raise ProtocolError("Antisymmetric prediction NPZ keys drifted.")
            predictions = np.asarray(payload["y_pred"])
            probabilities = np.asarray(payload["prob_pos"])
            fit_count = int(
                np.asarray(payload["unique_classifier_fit_count"]).item()
            )
    except (OSError, ValueError) as exc:
        raise ProtocolError("Antisymmetric prediction store is unreadable.") from exc
    return CrossfitPredictionStore(predictions, probabilities, rows, fit_count)


def write_crossfit_prediction_store(
    array_path: Path,
    index_path: Path,
    store: CrossfitPredictionStore,
) -> None:
    """Persist the prediction arrays and their closed-world CSV index."""

    atomic_save_npz(
        array_path,
        {
            "y_pred": store.y_pred,
            "prob_pos": store.prob_pos,
            "unique_classifier_fit_count": np.asarray(
                store.unique_classifier_fit_count, dtype=np.int64
            ),
        },
    )
    atomic_write_csv_rows(
        index_path,
        store.index_rows,
        columns=CROSSFIT_PREDICTION_INDEX_COLUMNS,
    )


def validate_crossfit_prediction_store_binding(
    store: CrossfitPredictionStore,
    *,
    config: "AntisymmetricResidualMMDDiagnosticConfig",
    generation_lock_hash: str,
    source_products_lock_hash: str,
    plans: object,
    crossfit: CrossfitSurface,
) -> None:
    plan_map, plan_lock_hash = plan_surface(plans)
    expected_keys = tuple(
        (fold.fold_ordinal, fold.fold_id, training_seed, generation_seed, arm)
        for fold in crossfit.folds
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
        for arm in ARM_ROLES
    )
    observed_keys: list[tuple[int, str, int, int, str]] = []
    for row in store.index_rows:
        fold_ordinal = integer(row.get("fold_ordinal"), "fold ordinal")
        if fold_ordinal < 0 or fold_ordinal >= len(crossfit.folds):
            raise ProtocolError(
                "Antisymmetric prediction references an unknown fold."
            )
        fold = crossfit.folds[fold_ordinal]
        plan = plan_map.get(fold.fold_id)
        validate_plan(plan, fold=fold)
        arm = str(row.get("arm_role"))
        if arm not in ARM_ROLES:
            raise ProtocolError("Antisymmetric prediction arm is invalid.")
        expected_weights, expected_allocations = arm_plan_payload(plan, arm)
        control_allocations = arm_plan_payload(plan, CONTROL_ARM)[1]
        expected_alias = (
            arm == ROUTED_ARM and expected_allocations == control_allocations
        )
        expected_shuffle_seeds = {
            str(class_label): derived_composition_seed(
                generation_lock_hash=generation_lock_hash,
                target_center=fold.target_center,
                training_seed=integer(
                    row.get("training_seed"), "training seed"
                ),
                generation_seed=integer(
                    row.get("generation_seed"), "generation seed"
                ),
                class_label=class_label,
            )
            for class_label in (0, 1)
        }
        observed_keys.append(
            (
                fold_ordinal,
                str(row.get("fold_id")),
                integer(row.get("training_seed"), "training seed"),
                integer(row.get("generation_seed"), "generation seed"),
                arm,
            )
        )
        if (
            row.get("config_contract_hash") != config.contract_hash
            or row.get("generation_lock_hash") != generation_lock_hash
            or row.get("source_products_lock_hash") != source_products_lock_hash
            or row.get("router_plan_lock_hash") != plan_lock_hash
            or row.get("fold_id") != fold.fold_id
            or row.get("fold_hash") != fold.fold_hash
            or row.get("target_center") != fold.target_center
            or row.get("heldout_case_id") != fold.heldout_case_id
            or parse_json_value(row.get("candidate_sources_json"))
            != list(candidate_sources(fold.target_center))
            or parse_json_value(row.get("weights_by_class_json"))
            != expected_weights
            or parse_json_value(row.get("allocations_by_class_json"))
            != expected_allocations
            or parse_json_value(row.get("shuffle_seed_by_class_json"))
            != expected_shuffle_seeds
            or row.get("router_support_row_identity_hash")
            != row_identity_hash(fold.router_support_rows)
            or parse_json_value(row.get("evaluation_row_ids_json"))
            != [item.sample_id for item in fold.heldout_rows]
            or row.get("evaluation_row_identity_hash")
            != row_identity_hash(fold.heldout_rows)
            or row.get("plan_hash") != plan["plan_hash"]
            or row.get("classifier_config_hash") != config.classifier.config_hash
            or not truthy(row.get("classifier_converged"))
            or not truthy(row.get("heldout_case_excluded_from_route"))
            or truthy(row.get("labels_available_to_fit_or_predict"))
            or truthy(row.get("support_labels_used"))
            or truthy(row.get("seed_selection_performed"))
            or truthy(row.get("control_fit_aliased")) != expected_alias
        ):
            raise ProtocolError(
                "Antisymmetric prediction store is not bound to its fold and plan."
            )
    if tuple(observed_keys) != expected_keys:
        raise ProtocolError(
            "Antisymmetric prediction cell order or coverage drifted."
        )


def plan_surface(
    plans: object,
) -> tuple[Mapping[str, Mapping[str, object]], str]:
    raw = getattr(plans, "plans_by_fold", None)
    lock_hash = str(getattr(plans, "lock_hash", ""))
    if not isinstance(raw, Mapping) or not is_hash(lock_hash):
        raise ProtocolError("Antisymmetric router-plan surface is malformed.")
    normalized: dict[str, Mapping[str, object]] = {}
    for fold_id, plan in raw.items():
        if not isinstance(plan, Mapping) or str(fold_id) in normalized:
            raise ProtocolError("Antisymmetric fold plan is malformed.")
        normalized[str(fold_id)] = plan
    return normalized, lock_hash


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise ProtocolError(
            f"Cannot read antisymmetric prediction table: {path}."
        ) from exc


__all__ = (
    "CROSSFIT_PREDICTION_ARRAY_MEMBER",
    "CROSSFIT_PREDICTION_INDEX_COLUMNS",
    "CROSSFIT_PREDICTION_INDEX_MEMBER",
    "CrossfitPredictionStore",
    "assemble_crossfit_prediction_store",
    "plan_surface",
    "read_crossfit_prediction_store",
    "validate_crossfit_prediction_store_binding",
    "write_crossfit_prediction_store",
)
