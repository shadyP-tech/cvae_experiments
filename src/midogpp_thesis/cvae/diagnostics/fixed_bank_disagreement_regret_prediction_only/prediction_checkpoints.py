"""Checkpoint execution, validation, and durable product assembly."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_array, sha256_file
from .constants import (
    CLASSIFIER_COEFFICIENT_MEMBER,
    CLASSIFIER_INTERCEPT_MEMBER,
    CLASSIFIER_MEAN_MEMBER,
    CLASSIFIER_SCALE_MEMBER,
    EXPECTED_CLASSIFIER_FIT_COUNT,
    FEATURE_DIM,
    PHYSICAL_ACTION_COUNT_PER_TARGET,
)
from .hashing import canonical_hash
from .prediction_contracts import (
    ActionClassifierBank,
    ClassifierBankCell,
    PredictionCell,
    canonical_cell_keys,
    classifier_parameter_sha256,
)
from .prediction_workers import source_prediction_task, test_prediction_task


def execute_or_resume_source(
    tasks: Sequence[Mapping[str, object]], *, workers: int
) -> Mapping[str, Mapping[str, object]]:
    return _execute_or_resume(
        tasks,
        workers=workers,
        worker=source_prediction_task,
        loader=load_source_checkpoint,
        progress_label="source action fits",
    )


def execute_or_resume_test(
    tasks: Sequence[Mapping[str, object]], *, workers: int
) -> Mapping[str, Mapping[str, object]]:
    return _execute_or_resume(
        tasks,
        workers=workers,
        worker=test_prediction_task,
        loader=load_test_checkpoint,
        progress_label="test inference tasks",
    )


def load_source_checkpoint(
    task: Mapping[str, object],
) -> Mapping[str, object] | None:
    loaded = _load_checkpoint_files(task)
    if loaded is None:
        return None
    payload, arrays = loaded
    expected_members = (
        "source_probabilities",
        "scaler_mean",
        "scaler_scale",
        "coefficients",
        "intercepts",
    )
    if tuple(arrays) != expected_members:
        raise ProtocolError("Prediction-only source checkpoint members drifted.")
    probabilities = arrays["source_probabilities"]
    expected_parameter_shape = (PHYSICAL_ACTION_COUNT_PER_TARGET, FEATURE_DIM)
    actions = payload.get("actions")
    if (
        payload.get("schema_version")
        != "midogpp_prediction_only_source_fit_checkpoint_v1"
        or probabilities.ndim != 2
        or probabilities.shape[0] != PHYSICAL_ACTION_COUNT_PER_TARGET
        or probabilities.dtype != np.float32
        or arrays["scaler_mean"].shape != expected_parameter_shape
        or arrays["scaler_scale"].shape != expected_parameter_shape
        or arrays["coefficients"].shape != expected_parameter_shape
        or arrays["intercepts"].shape != (PHYSICAL_ACTION_COUNT_PER_TARGET,)
        or any(arrays[key].dtype != np.float64 for key in expected_members[1:])
        or not isinstance(actions, list)
        or len(actions) != PHYSICAL_ACTION_COUNT_PER_TARGET
        or payload.get("physical_fit_count") != PHYSICAL_ACTION_COUNT_PER_TARGET
        or payload.get("source_labels_available") is not False
        or payload.get("test_cache_admitted") is not False
    ):
        raise ProtocolError("Prediction-only source checkpoint validation failed.")
    expected_actions = task["actions"]
    for ordinal, (raw, expected) in enumerate(
        zip(actions, expected_actions, strict=True)
    ):
        if not isinstance(raw, Mapping) or not isinstance(expected, Mapping):
            raise ProtocolError("Prediction-only source checkpoint action is malformed.")
        parameter_hash = classifier_parameter_sha256(
            arrays["scaler_mean"][ordinal],
            arrays["scaler_scale"][ordinal],
            arrays["coefficients"][ordinal],
            float(arrays["intercepts"][ordinal]),
        )
        values = probabilities[ordinal]
        if (
            raw.get("action_id") != expected.get("action_id")
            or raw.get("action_hash") != expected.get("action_hash")
            or raw.get("parameter_sha256") != parameter_hash
            or raw.get("probability_sha256") != sha256_array(values)
            or raw.get("predictions_sha256")
            != sha256_array((values >= np.float32(0.5)).astype(np.uint8))
            or raw.get("converged") is not True
        ):
            raise ProtocolError("Prediction-only source checkpoint hashes drifted.")
    return payload


def load_test_checkpoint(
    task: Mapping[str, object],
) -> Mapping[str, object] | None:
    loaded = _load_checkpoint_files(task)
    if loaded is None:
        return None
    payload, arrays = loaded
    if tuple(arrays) != ("test_probabilities",):
        raise ProtocolError("Prediction-only test checkpoint members drifted.")
    values = arrays["test_probabilities"]
    actions = payload.get("actions")
    if (
        payload.get("schema_version")
        != "midogpp_prediction_only_test_inference_checkpoint_v1"
        or values.ndim != 2
        or values.shape[0] != PHYSICAL_ACTION_COUNT_PER_TARGET
        or values.dtype != np.float32
        or not isinstance(actions, list)
        or len(actions) != PHYSICAL_ACTION_COUNT_PER_TARGET
        or payload.get("classifier_fit_count") != 0
        or payload.get("labels_available") is not False
        or payload.get("target_scoring_permitted") is not False
    ):
        raise ProtocolError("Prediction-only test checkpoint validation failed.")
    for ordinal, raw in enumerate(actions):
        if not isinstance(raw, Mapping):
            raise ProtocolError("Prediction-only test action is malformed.")
        row = values[ordinal]
        if (
            raw.get("probability_sha256") != sha256_array(row)
            or raw.get("predictions_sha256")
            != sha256_array((row >= np.float32(0.5)).astype(np.uint8))
        ):
            raise ProtocolError("Prediction-only test checkpoint hashes drifted.")
    return payload


def assemble_source_products(
    root: Path,
    *,
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[str, Mapping[str, object]],
) -> tuple[tuple[ClassifierBankCell, ...], tuple[PredictionCell, ...]]:
    paths = (
        root / CLASSIFIER_MEAN_MEMBER,
        root / CLASSIFIER_SCALE_MEMBER,
        root / CLASSIFIER_COEFFICIENT_MEMBER,
        root / CLASSIFIER_INTERCEPT_MEMBER,
    )
    temporary = tuple(
        path.with_suffix(path.suffix + f".{os.getpid()}.tmp") for path in paths
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    matrices = (
        np.lib.format.open_memmap(
            temporary[0],
            mode="w+",
            dtype=np.float64,
            shape=(EXPECTED_CLASSIFIER_FIT_COUNT, FEATURE_DIM),
        ),
        np.lib.format.open_memmap(
            temporary[1],
            mode="w+",
            dtype=np.float64,
            shape=(EXPECTED_CLASSIFIER_FIT_COUNT, FEATURE_DIM),
        ),
        np.lib.format.open_memmap(
            temporary[2],
            mode="w+",
            dtype=np.float64,
            shape=(EXPECTED_CLASSIFIER_FIT_COUNT, FEATURE_DIM),
        ),
        np.lib.format.open_memmap(
            temporary[3],
            mode="w+",
            dtype=np.float64,
            shape=(EXPECTED_CLASSIFIER_FIT_COUNT,),
        ),
    )
    classifier_cells: list[ClassifierBankCell] = []
    prediction_cells: list[PredictionCell] = []
    cursor = 0
    try:
        for task in tasks:
            payload = completed[str(task["task_id"])]
            with np.load(
                Path(str(task["checkpoint_npz_path"])), allow_pickle=False
            ) as archive:
                probabilities = np.asarray(
                    archive["source_probabilities"], dtype=np.float32
                )
                means = np.asarray(archive["scaler_mean"], dtype=np.float64)
                scales = np.asarray(archive["scaler_scale"], dtype=np.float64)
                coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
                intercepts = np.asarray(archive["intercepts"], dtype=np.float64)
            for ordinal, raw in enumerate(payload["actions"]):
                matrices[0][cursor] = means[ordinal]
                matrices[1][cursor] = scales[ordinal]
                matrices[2][cursor] = coefficients[ordinal]
                matrices[3][cursor] = intercepts[ordinal]
                classifier_cells.append(
                    ClassifierBankCell(
                        cell_ordinal=cursor,
                        target_center=str(task["target_center"]),
                        action_id=str(raw["action_id"]),
                        action_hash=str(raw["action_hash"]),
                        training_seed=int(task["training_seed"]),
                        generation_seed=int(task["generation_seed"]),
                        composition_hash=str(raw["composition_hash"]),
                        scaler_state_hash=str(raw["scaler_state_hash"]),
                        parameter_sha256=str(raw["parameter_sha256"]),
                        fit_provenance_hash=str(raw["fit_provenance_hash"]),
                        classifier_config_hash=str(raw["classifier_config_hash"]),
                        n_iter=tuple(int(value) for value in raw["n_iter"]),
                        converged=bool(raw["converged"]),
                    )
                )
                values = probabilities[ordinal]
                prediction_cells.append(
                    PredictionCell(
                        frame_role="source",
                        target_center=str(task["target_center"]),
                        action_id=str(raw["action_id"]),
                        action_hash=str(raw["action_hash"]),
                        training_seed=int(task["training_seed"]),
                        generation_seed=int(task["generation_seed"]),
                        row_identity_hash=str(task["source_row_identity_hash"]),
                        probabilities=values,
                        probability_sha256=str(raw["probability_sha256"]),
                        predictions_sha256=str(raw["predictions_sha256"]),
                        classifier_parameter_sha256=str(raw["parameter_sha256"]),
                    )
                )
                cursor += 1
        for matrix in matrices:
            matrix.flush()
    finally:
        del matrices
    if cursor != EXPECTED_CLASSIFIER_FIT_COUNT:
        raise ProtocolError("Prediction-only source product coverage drifted.")
    for source, destination in zip(temporary, paths, strict=True):
        os.replace(source, destination)
    classifier_result = tuple(classifier_cells)
    prediction_result = tuple(prediction_cells)
    expected_keys = canonical_cell_keys()
    if (
        tuple(cell.key for cell in classifier_result) != expected_keys
        or tuple(cell.key for cell in prediction_result) != expected_keys
    ):
        raise ProtocolError("Prediction-only assembled source order drifted.")
    return classifier_result, prediction_result


def assemble_test_cells(
    *,
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[str, Mapping[str, object]],
    classifier_bank: ActionClassifierBank,
) -> tuple[PredictionCell, ...]:
    cells: list[PredictionCell] = []
    for task in tasks:
        payload = completed[str(task["task_id"])]
        with np.load(
            Path(str(task["checkpoint_npz_path"])), allow_pickle=False
        ) as archive:
            values = np.asarray(archive["test_probabilities"], dtype=np.float32)
        for ordinal, raw in enumerate(payload["actions"]):
            classifier_cell = classifier_bank.cells[
                int(raw["classifier_cell_ordinal"])
            ]
            if classifier_cell.parameter_sha256 != raw["classifier_parameter_sha256"]:
                raise ProtocolError("Prediction-only test classifier lineage drifted.")
            cells.append(
                PredictionCell(
                    frame_role="test",
                    target_center=str(task["target_center"]),
                    action_id=str(raw["action_id"]),
                    action_hash=str(raw["action_hash"]),
                    training_seed=int(task["training_seed"]),
                    generation_seed=int(task["generation_seed"]),
                    row_identity_hash=str(task["test_row_identity_hash"]),
                    probabilities=values[ordinal],
                    probability_sha256=str(raw["probability_sha256"]),
                    predictions_sha256=str(raw["predictions_sha256"]),
                    classifier_parameter_sha256=str(
                        raw["classifier_parameter_sha256"]
                    ),
                )
            )
    result = tuple(cells)
    if tuple(cell.key for cell in result) != canonical_cell_keys():
        raise ProtocolError("Prediction-only assembled test order drifted.")
    return result


def _execute_or_resume(
    tasks: Sequence[Mapping[str, object]],
    *,
    workers: int,
    worker: object,
    loader: object,
    progress_label: str,
) -> Mapping[str, Mapping[str, object]]:
    if workers != 4:
        raise ProtocolError("Prediction-only CPU phases require exactly four workers.")
    completed: dict[str, Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        loaded = loader(task)  # type: ignore[operator]
        if loaded is None:
            pending.append(task)
        else:
            completed[str(task["task_id"])] = loaded
    if pending:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = {executor.submit(worker, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                future.result()
                loaded = loader(task)  # type: ignore[operator]
                if loaded is None:
                    raise ProtocolError("Prediction-only worker checkpoint is absent.")
                completed[str(task["task_id"])] = loaded
                print(
                    f"[disagreement-regret] {progress_label} "
                    f"{len(completed)}/{len(tasks)}",
                    flush=True,
                )
    if len(completed) != len(tasks):
        raise ProtocolError("Prediction-only checkpoint coverage is incomplete.")
    return MappingProxyType(completed)


def _load_checkpoint_files(
    task: Mapping[str, object],
) -> tuple[Mapping[str, object], dict[str, np.ndarray]] | None:
    json_path = Path(str(task["checkpoint_json_path"]))
    npz_path = Path(str(task["checkpoint_npz_path"]))
    if not json_path.is_file() and not npz_path.is_file():
        return None
    if not json_path.is_file() or not npz_path.is_file():
        raise ProtocolError("Prediction-only checkpoint is partially present.")
    payload = read_json(json_path)
    unhashed = {
        key: value for key, value in payload.items() if key != "checkpoint_hash"
    }
    if (
        payload.get("checkpoint_hash") != canonical_hash(unhashed)
        or payload.get("task_hash") != task.get("task_hash")
        or payload.get("task_id") != task.get("task_id")
        or payload.get("array_sha256") != sha256_file(npz_path)
    ):
        raise ProtocolError("Prediction-only checkpoint envelope drifted.")
    try:
        archive = np.load(npz_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Prediction-only checkpoint archive is unreadable.") from exc
    with archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return payload, arrays


__all__ = (
    "assemble_source_products",
    "assemble_test_cells",
    "execute_or_resume_source",
    "execute_or_resume_test",
    "load_source_checkpoint",
    "load_test_checkpoint",
)
