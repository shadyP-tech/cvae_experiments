"""Resumable matched target predictions for MMD/KMM and equal union."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np

from ....real_features.classifier_reference.classifiers import fit_logistic_classifier
from ...generation.generation import derived_composition_seed
from ...protocol import ProtocolError
from ...routing.dense_residual_soft_router import compose_prefix_blocks
from .config import MMDKMMRouterDiagnosticConfig
from .contracts import (
    CENTERS,
    EXPECTED_PREDICTION_CELL_COUNT,
    EXPECTED_SEED_CELL_COUNT,
    GENERATION_SEEDS,
    MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT,
    MAX_SOURCE_PREFIX_PER_CLASS,
    TOTAL_PER_CLASS,
    TRAINING_SEEDS,
    candidate_sources,
    row_identity_hash,
)
from .inputs import LabelFreeValidationFrame, PartitionSurface
from .planning import RouterPlans
from .source_products import SourceProducts


TARGET_PREDICTION_ARRAY_MEMBER = "arrays/target_predictions.npz"
TARGET_PREDICTION_INDEX_MEMBER = "tables/target_prediction_index.csv"
PREDICTION_INDEX_COLUMNS = (
    "schema_version",
    "config_contract_hash",
    "generation_lock_hash",
    "source_products_lock_hash",
    "router_plan_lock_hash",
    "cell_ordinal",
    "target_center",
    "arm_role",
    "training_seed",
    "generation_seed",
    "candidate_sources_json",
    "weights_json",
    "allocations_per_class_json",
    "shuffle_seed_by_class_json",
    "composition_hash",
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
    "support_rows_used_to_predict",
    "seed_selection_performed",
    "control_fit_aliased",
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
            or not np.isin(predictions, [0, 1]).all()
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
            or len(self.index_rows) != EXPECTED_PREDICTION_CELL_COUNT
            or not 0 < int(self.unique_classifier_fit_count) <= MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT
        ):
            raise ProtocolError("MMD/KMM prediction store is malformed.")
        cursor = 0
        for ordinal, row in enumerate(self.index_rows):
            start, stop = int(row["prediction_offset_start"]), int(row["prediction_offset_stop"])
            if (
                int(row["cell_ordinal"]) != ordinal
                or start != cursor
                or stop <= start
                or _sha256_array(predictions[start:stop]) != row["prediction_sha256"]
                or _sha256_array(probabilities[start:stop]) != row["probability_sha256"]
            ):
                raise ProtocolError("MMD/KMM prediction-store offsets or hashes drifted.")
            cursor = stop
        if cursor != len(predictions):
            raise ProtocolError("MMD/KMM prediction-store coverage drifted.")

    def slice_for(self, row: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
        start, stop = int(row["prediction_offset_start"]), int(row["prediction_offset_stop"])
        return self.y_pred[start:stop], self.prob_pos[start:stop]


def materialize_target_predictions(
    config: MMDKMMRouterDiagnosticConfig,
    generation_lock_hash: str,
    source_products: SourceProducts,
    plans: RouterPlans,
    frame: LabelFreeValidationFrame,
    partitions: PartitionSurface,
    *,
    source_products_lock_hash: str,
    root: Path,
) -> PredictionStore:
    final_array = root / TARGET_PREDICTION_ARRAY_MEMBER
    final_index = root / TARGET_PREDICTION_INDEX_MEMBER
    if final_array.is_file() and final_index.is_file():
        store = read_prediction_store(final_array, final_index)
        validate_prediction_store_binding(
            store,
            config=config,
            generation_lock_hash=generation_lock_hash,
            source_products_lock_hash=source_products_lock_hash,
            plans=plans,
            partitions=partitions,
        )
        shutil.rmtree(root / "checkpoints/predictions", ignore_errors=True)
        return store

    checkpoint_root = root / "checkpoints/predictions"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    evaluation_path = checkpoint_root / "evaluation_embeddings.npy"
    evaluation_index_path = checkpoint_root / "evaluation_index.json"
    evaluation_index = _write_evaluation_scratch(
        evaluation_path,
        evaluation_index_path,
        frame=frame,
        partitions=partitions,
    )
    tasks: list[dict[str, object]] = []
    for target in CENTERS:
        plan = plans.plans_by_target[target]
        target_eval = evaluation_index["targets"][target]
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                tasks.append(
                    {
                        "target_center": target,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "config_contract_hash": config.contract_hash,
                        "generation_lock_hash": generation_lock_hash,
                        "source_products_lock_hash": source_products_lock_hash,
                        "router_plan_lock_hash": plans.lock_hash,
                        "source_array_path": str(source_products.array_path),
                        "source_index_rows": [dict(row) for row in source_products.index_rows],
                        "evaluation_array_path": str(evaluation_path),
                        "evaluation_start": int(target_eval["start"]),
                        "evaluation_stop": int(target_eval["stop"]),
                        "evaluation_row_ids": tuple(target_eval["sample_ids"]),
                        "evaluation_row_identity_hash": str(target_eval["row_identity_hash"]),
                        "plan": dict(plan),
                        "classifier": config.classifier,
                        "threads_per_worker": int(config.runtime["classifier_threads_per_worker"]),
                        "checkpoint_path": str(
                            checkpoint_root
                            / f"target_{target}_train_{training_seed}_gen_{generation_seed}.npz"
                        ),
                    }
                )
    if len(tasks) != len(CENTERS) * EXPECTED_SEED_CELL_COUNT:
        raise ProtocolError("MMD/KMM classifier task scheduler drifted.")

    completed: dict[tuple[str, int, int], Mapping[str, object]] = {}
    pending: list[dict[str, object]] = []
    for task in tasks:
        checkpoint = Path(str(task["checkpoint_path"]))
        if not checkpoint.is_file():
            pending.append(task)
            continue
        completed[_task_key(task)] = _load_prediction_checkpoint(checkpoint, task=task)

    if pending:
        context = mp.get_context("spawn")
        worker_count = int(config.runtime["classifier_workers"])
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            future_to_task: dict[Future[dict[str, object]], dict[str, object]] = {
                executor.submit(_prediction_task, task): task for task in pending
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                result = future.result()
                _write_prediction_checkpoint(Path(str(task["checkpoint_path"])), result)
                completed[_task_key(task)] = _load_prediction_checkpoint(
                    Path(str(task["checkpoint_path"])), task=task
                )
                print(
                    f"[mmd-kmm] classifier cells {len(completed)}/{len(tasks)}",
                    flush=True,
                )
    if len(completed) != len(tasks):
        raise ProtocolError("MMD/KMM prediction checkpoint coverage is incomplete.")

    prediction_arrays: list[np.ndarray] = []
    probability_arrays: list[np.ndarray] = []
    index_rows: list[dict[str, object]] = []
    cursor = 0
    unique_fit_count = 0
    for target in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                cell = completed[(target, training_seed, generation_seed)]
                unique_fit_count += int(cell["unique_classifier_fit_count"])
                for arm in ("equal_union_control", "mmd_kmm"):
                    predictions = np.asarray(cell[f"{arm}_predictions"], dtype=np.uint8)
                    probabilities = np.asarray(cell[f"{arm}_probabilities"], dtype=np.float32)
                    metadata = dict(cell[f"{arm}_metadata"])
                    stop = cursor + len(predictions)
                    row = {
                        **metadata,
                        "cell_ordinal": len(index_rows),
                        "prediction_offset_start": cursor,
                        "prediction_offset_stop": stop,
                        "prediction_sha256": _sha256_array(predictions),
                        "probability_sha256": _sha256_array(probabilities),
                    }
                    if set(row) != set(PREDICTION_INDEX_COLUMNS):
                        raise ProtocolError("MMD/KMM prediction-index schema drifted.")
                    prediction_arrays.append(predictions)
                    probability_arrays.append(probabilities)
                    index_rows.append(row)
                    cursor = stop
    store = PredictionStore(
        y_pred=np.concatenate(prediction_arrays).astype(np.uint8, copy=False),
        prob_pos=np.concatenate(probability_arrays).astype(np.float32, copy=False),
        index_rows=tuple(index_rows),
        unique_classifier_fit_count=unique_fit_count,
    )
    _write_prediction_store(final_array, store)
    _atomic_write_csv_rows(
        final_index,
        store.index_rows,
        columns=PREDICTION_INDEX_COLUMNS,
    )
    validate_prediction_store_binding(
        store,
        config=config,
        generation_lock_hash=generation_lock_hash,
        source_products_lock_hash=source_products_lock_hash,
        plans=plans,
        partitions=partitions,
    )
    shutil.rmtree(checkpoint_root, ignore_errors=True)
    return store


def read_prediction_store(array_path: Path, index_path: Path) -> PredictionStore:
    rows = tuple(_read_csv(index_path))
    try:
        with np.load(array_path, allow_pickle=False) as payload:
            if set(payload.files) != {"y_pred", "prob_pos", "unique_classifier_fit_count"}:
                raise ProtocolError("MMD/KMM final prediction NPZ keys drifted.")
            y_pred = np.asarray(payload["y_pred"])
            prob_pos = np.asarray(payload["prob_pos"])
            fit_count = int(np.asarray(payload["unique_classifier_fit_count"]).item())
    except (OSError, ValueError) as exc:
        raise ProtocolError("MMD/KMM final prediction store is unreadable.") from exc
    return PredictionStore(y_pred, prob_pos, rows, fit_count)


def validate_prediction_store_binding(
    store: PredictionStore,
    *,
    config: MMDKMMRouterDiagnosticConfig,
    generation_lock_hash: str,
    source_products_lock_hash: str,
    plans: RouterPlans,
    partitions: PartitionSurface,
) -> None:
    expected_keys = tuple(
        (target, training_seed, generation_seed, arm)
        for target in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
        for arm in ("equal_union_control", "mmd_kmm")
    )
    observed_keys: list[tuple[str, int, int, str]] = []
    for row in store.index_rows:
        target = str(row["target_center"])
        arm = str(row["arm_role"])
        plan = plans.plans_by_target.get(target)
        expected_rows = partitions.evaluation_rows_by_center.get(target)
        if plan is None or expected_rows is None:
            raise ProtocolError("MMD/KMM prediction store contains an unknown target.")
        weights_key = "control_weights" if arm == "equal_union_control" else "final_weights"
        allocation_key = (
            "control_allocations_per_class"
            if arm == "equal_union_control"
            else "mmd_allocations_per_class"
        )
        observed_keys.append(
            (
                target,
                int(row["training_seed"]),
                int(row["generation_seed"]),
                arm,
            )
        )
        expected_ids = [item.sample_id for item in expected_rows]
        if (
            row.get("config_contract_hash") != config.contract_hash
            or row.get("generation_lock_hash") != generation_lock_hash
            or row.get("source_products_lock_hash") != source_products_lock_hash
            or row.get("router_plan_lock_hash") != plans.lock_hash
            or row.get("plan_hash") != plan["plan_hash"]
            or _json_value(row["candidate_sources_json"])
            != list(candidate_sources(target))
            or _json_value(row["weights_json"]) != plan[weights_key]
            or _json_value(row["allocations_per_class_json"])
            != plan[allocation_key]
            or _json_value(row["evaluation_row_ids_json"]) != expected_ids
            or row.get("evaluation_row_identity_hash")
            != row_identity_hash(expected_rows)
            or row.get("classifier_config_hash") != config.classifier.config_hash
            or _truthy(row["labels_available_to_fit_or_predict"])
            or _truthy(row["support_rows_used_to_predict"])
            or _truthy(row["seed_selection_performed"])
        ):
            raise ProtocolError(
                "MMD/KMM prediction store is not bound to the current inputs/plan."
            )
    if tuple(observed_keys) != expected_keys:
        raise ProtocolError("MMD/KMM prediction-store cell order/coverage drifted.")


def _prediction_task(task: Mapping[str, object]) -> dict[str, object]:
    target = str(task["target_center"])
    training_seed = int(task["training_seed"])
    generation_seed = int(task["generation_seed"])
    plan = task["plan"]
    if not isinstance(plan, Mapping) or plan.get("target_center") != target:
        raise ProtocolError("MMD/KMM prediction task plan drifted.")
    source_array = np.load(Path(str(task["source_array_path"])), mmap_mode="r")
    block_index = {
        (str(row["source_center"]), int(row["training_seed"]), int(row["generation_seed"])): int(row["block_ordinal"])
        for row in task["source_index_rows"]
    }
    blocks: dict[str, object] = {}
    labels = np.concatenate(
        (
            np.zeros(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
            np.ones(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
        )
    )
    for source in candidate_sources(target):
        ordinal = block_index[(source, training_seed, generation_seed)]
        blocks[source] = SimpleNamespace(
            embeddings=np.asarray(source_array[ordinal]),
            labels=labels,
            key=SimpleNamespace(source_center=source),
        )
    shuffle_seeds = {
        str(label): derived_composition_seed(
            generation_lock_hash=str(task["generation_lock_hash"]),
            target_center=target,
            training_seed=training_seed,
            generation_seed=generation_seed,
            class_label=label,
        )
        for label in (0, 1)
    }
    control = compose_prefix_blocks(
        blocks,
        plan["control_allocations_per_class"],
        shuffle_seed_by_class=shuffle_seeds,
        total_per_class=TOTAL_PER_CLASS,
    )
    routed = compose_prefix_blocks(
        blocks,
        plan["mmd_allocations_per_class"],
        shuffle_seed_by_class=shuffle_seeds,
        total_per_class=TOTAL_PER_CLASS,
    )
    evaluation = np.load(Path(str(task["evaluation_array_path"])), mmap_mode="r")[
        int(task["evaluation_start"]) : int(task["evaluation_stop"])
    ]
    control_fit = _fit(
        control.embeddings,
        control.labels,
        evaluation,
        classifier=task["classifier"],
        threads=int(task["threads_per_worker"]),
    )
    alias = control.composition_hash == routed.composition_hash
    routed_fit = control_fit if alias else _fit(
        routed.embeddings,
        routed.labels,
        evaluation,
        classifier=task["classifier"],
        threads=int(task["threads_per_worker"]),
    )
    base_metadata = {
        "schema_version": "midogpp_mmd_kmm_prediction_cell_v1",
        "config_contract_hash": task["config_contract_hash"],
        "generation_lock_hash": task["generation_lock_hash"],
        "source_products_lock_hash": task["source_products_lock_hash"],
        "router_plan_lock_hash": task["router_plan_lock_hash"],
        "target_center": target,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "candidate_sources_json": _compact(list(candidate_sources(target))),
        "shuffle_seed_by_class_json": _compact(shuffle_seeds),
        "evaluation_row_ids_json": _compact(list(task["evaluation_row_ids"])),
        "evaluation_row_identity_hash": task["evaluation_row_identity_hash"],
        "plan_hash": plan["plan_hash"],
        "labels_available_to_fit_or_predict": False,
        "support_rows_used_to_predict": False,
        "seed_selection_performed": False,
    }
    control_metadata = {
        **base_metadata,
        "arm_role": "equal_union_control",
        "weights_json": _compact(plan["control_weights"]),
        "allocations_per_class_json": _compact(plan["control_allocations_per_class"]),
        "composition_hash": control.composition_hash,
        **_fit_metadata(control_fit),
        "control_fit_aliased": False,
    }
    route_metadata = {
        **base_metadata,
        "arm_role": "mmd_kmm",
        "weights_json": _compact(plan["final_weights"]),
        "allocations_per_class_json": _compact(plan["mmd_allocations_per_class"]),
        "composition_hash": routed.composition_hash,
        **_fit_metadata(routed_fit),
        "control_fit_aliased": alias,
    }
    return {
        "schema_version": "midogpp_mmd_kmm_prediction_checkpoint_v1",
        "config_contract_hash": task["config_contract_hash"],
        "generation_lock_hash": task["generation_lock_hash"],
        "source_products_lock_hash": task["source_products_lock_hash"],
        "router_plan_lock_hash": task["router_plan_lock_hash"],
        "evaluation_row_identity_hash": task["evaluation_row_identity_hash"],
        "target_center": target,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "plan_hash": plan["plan_hash"],
        "equal_union_control_predictions": control_fit["predictions"],
        "equal_union_control_probabilities": control_fit["probabilities"],
        "equal_union_control_metadata": control_metadata,
        "mmd_kmm_predictions": routed_fit["predictions"],
        "mmd_kmm_probabilities": routed_fit["probabilities"],
        "mmd_kmm_metadata": route_metadata,
        "unique_classifier_fit_count": 1 if alias else 2,
    }


def _fit(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    evaluation_embeddings: np.ndarray,
    *,
    classifier: object,
    threads: int,
) -> dict[str, object]:
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("MMD/KMM classifier fitting requires threadpoolctl.") from exc
    with threadpool_limits(limits=int(threads)):
        fitted = fit_logistic_classifier(
            train_embeddings,
            train_labels,
            evaluation_embeddings,
            spec=classifier,
        )
    predictions = np.asarray(fitted.predictions, dtype=np.uint8)
    probabilities = np.asarray(fitted.probabilities, dtype=np.float64)
    if (
        tuple(int(value) for value in fitted.classes) != (0, 1)
        or predictions.shape != (len(evaluation_embeddings),)
        or probabilities.shape != (len(evaluation_embeddings), 2)
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7, rtol=0.0)
        or not fitted.converged
        or fitted.classifier_config_hash != classifier.config_hash
    ):
        raise ProtocolError("MMD/KMM downstream classifier fit drifted.")
    return {
        "predictions": predictions,
        "probabilities": probabilities[:, 1].astype(np.float32, copy=False),
        "classifier_config_hash": fitted.classifier_config_hash,
        "scaler_state_hash": fitted.scaler_state_hash,
        "n_iter": tuple(int(value) for value in fitted.n_iter),
        "converged": bool(fitted.converged),
    }


def _fit_metadata(fitted: Mapping[str, object]) -> dict[str, object]:
    return {
        "classifier_config_hash": fitted["classifier_config_hash"],
        "scaler_state_hash": fitted["scaler_state_hash"],
        "classifier_n_iter_json": _compact(list(fitted["n_iter"])),
        "classifier_converged": fitted["converged"],
    }


def _write_evaluation_scratch(
    array_path: Path,
    index_path: Path,
    *,
    frame: LabelFreeValidationFrame,
    partitions: PartitionSurface,
) -> Mapping[str, object]:
    rows = [row for target in CENTERS for row in partitions.evaluation_rows_by_center[target]]
    embeddings = frame.embeddings_for(rows)
    targets: dict[str, object] = {}
    cursor = 0
    for target in CENTERS:
        target_rows = partitions.evaluation_rows_by_center[target]
        stop = cursor + len(target_rows)
        targets[target] = {
            "start": cursor,
            "stop": stop,
            "sample_ids": [row.sample_id for row in target_rows],
            "row_identity_hash": row_identity_hash(target_rows),
        }
        cursor = stop
    payload = {
        "schema_version": "midogpp_mmd_kmm_evaluation_scratch_v1",
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "targets": targets,
        "labels_present": False,
    }
    _atomic_save_npy(array_path, embeddings)
    _atomic_json(index_path, payload)
    return payload


def _write_prediction_checkpoint(path: Path, result: Mapping[str, object]) -> None:
    arrays = {
        "equal_union_control_predictions": np.asarray(result["equal_union_control_predictions"], dtype=np.uint8),
        "equal_union_control_probabilities": np.asarray(result["equal_union_control_probabilities"], dtype=np.float32),
        "mmd_kmm_predictions": np.asarray(result["mmd_kmm_predictions"], dtype=np.uint8),
        "mmd_kmm_probabilities": np.asarray(result["mmd_kmm_probabilities"], dtype=np.float32),
    }
    metadata = {
        key: value
        for key, value in result.items()
        if key not in arrays
    }
    metadata["array_hashes"] = {key: _sha256_array(value) for key, value in arrays.items()}
    metadata["checkpoint_hash"] = stable_checkpoint_hash(metadata)
    _atomic_save_npz(path, {**arrays, "checkpoint_json": np.asarray(_compact(metadata))})


def _load_prediction_checkpoint(path: Path, *, task: Mapping[str, object]) -> Mapping[str, object]:
    try:
        with np.load(path, allow_pickle=False) as raw:
            metadata = json.loads(str(np.asarray(raw["checkpoint_json"]).item()))
            arrays = {
                key: np.asarray(raw[key])
                for key in (
                    "equal_union_control_predictions",
                    "equal_union_control_probabilities",
                    "mmd_kmm_predictions",
                    "mmd_kmm_probabilities",
                )
            }
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ProtocolError("MMD/KMM prediction checkpoint is unreadable.") from exc
    if not isinstance(metadata, Mapping):
        raise ProtocolError("MMD/KMM prediction checkpoint metadata is malformed.")
    unhashed = {key: value for key, value in metadata.items() if key != "checkpoint_hash"}
    hashes = metadata.get("array_hashes")
    if (
        metadata.get("checkpoint_hash") != stable_checkpoint_hash(unhashed)
        or metadata.get("config_contract_hash") != task["config_contract_hash"]
        or metadata.get("generation_lock_hash") != task["generation_lock_hash"]
        or metadata.get("source_products_lock_hash")
        != task["source_products_lock_hash"]
        or metadata.get("router_plan_lock_hash") != task["router_plan_lock_hash"]
        or metadata.get("evaluation_row_identity_hash")
        != task["evaluation_row_identity_hash"]
        or metadata.get("target_center") != task["target_center"]
        or int(metadata.get("training_seed", -1)) != int(task["training_seed"])
        or int(metadata.get("generation_seed", -1)) != int(task["generation_seed"])
        or metadata.get("plan_hash") != task["plan"]["plan_hash"]
        or not isinstance(hashes, Mapping)
        or any(hashes.get(key) != _sha256_array(value) for key, value in arrays.items())
    ):
        raise ProtocolError("MMD/KMM prediction checkpoint failed validation.")
    return {**metadata, **arrays}


def stable_checkpoint_hash(payload: Mapping[str, object]) -> str:
    from ....common.hashing import stable_hash

    return stable_hash(dict(payload))


def _write_prediction_store(path: Path, store: PredictionStore) -> None:
    _atomic_save_npz(
        path,
        {
            "y_pred": store.y_pred,
            "prob_pos": store.prob_pos,
            "unique_classifier_fit_count": np.asarray(store.unique_classifier_fit_count, dtype=np.int64),
        },
    )


def _task_key(task: Mapping[str, object]) -> tuple[str, int, int]:
    return str(task["target_center"]), int(task["training_seed"]), int(task["generation_seed"])


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise ProtocolError(f"Cannot read MMD/KMM prediction table: {path}.") from exc


def _atomic_write_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            for row in rows:
                if set(row) != set(columns):
                    raise ProtocolError(
                        "MMD/KMM atomic CSV row schema drifted."
                    )
                writer.writerow({column: row[column] for column in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_value(value: object) -> object:
    try:
        return json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ProtocolError("MMD/KMM prediction metadata JSON is malformed.") from exc


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = (
    "PREDICTION_INDEX_COLUMNS",
    "TARGET_PREDICTION_ARRAY_MEMBER",
    "TARGET_PREDICTION_INDEX_MEMBER",
    "PredictionStore",
    "materialize_target_predictions",
    "read_prediction_store",
    "validate_prediction_store_binding",
)
