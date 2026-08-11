"""Checkpoint assembly and durable strict source-OOF stores."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, atomic_npz, read_json, sha256_array, sha256_file
from .constants import CENTERS, FEATURE_DIM
from .development_actions import (
    DEVELOPMENT_ACTION_COUNT_PER_TASK,
    DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT,
    DEVELOPMENT_PHYSICAL_TASK_COUNT,
    development_action_library_payload,
)
from .development_prediction_contracts import (
    DEVELOPMENT_ACTION_LIBRARY_MEMBER,
    DEVELOPMENT_CLASSIFIER_COEFFICIENT_MEMBER,
    DEVELOPMENT_CLASSIFIER_INDEX_MEMBER,
    DEVELOPMENT_CLASSIFIER_INTERCEPT_MEMBER,
    DEVELOPMENT_CLASSIFIER_MEAN_MEMBER,
    DEVELOPMENT_CLASSIFIER_SCALE_MEMBER,
    DEVELOPMENT_CLASSIFIER_SEAL_MEMBER,
    DEVELOPMENT_CLASSIFIER_STATUS,
    DEVELOPMENT_PREDICTION_ARRAY_MEMBER,
    DEVELOPMENT_PREDICTION_INDEX_MEMBER,
    DEVELOPMENT_PREDICTION_SEAL_MEMBER,
    DEVELOPMENT_PREDICTION_STATUS,
    DevelopmentClassifierBank,
    DevelopmentClassifierCell,
    DevelopmentPredictionCell,
    DevelopmentPredictionStore,
    DevelopmentSourcePredictionSeal,
    canonical_logical_cell_keys,
    canonical_physical_cell_keys,
    development_prediction_store_hash,
)
from .development_prediction_workers import development_source_prediction_task
from .hashing import canonical_hash
from .prediction_contracts import classifier_parameter_sha256


def write_development_action_library(root: Path) -> Mapping[str, object]:
    payload = development_action_library_payload()
    path = root / DEVELOPMENT_ACTION_LIBRARY_MEMBER
    if path.is_file():
        if read_json(path) != payload:
            raise ProtocolError("Persisted strict source-OOF action library drifted.")
    else:
        atomic_json(path, payload)
    return payload


def execute_or_resume_development_source(
    tasks: Sequence[Mapping[str, object]], *, workers: int
) -> Mapping[str, Mapping[str, object]]:
    if workers != 4 or len(tasks) != DEVELOPMENT_PHYSICAL_TASK_COUNT:
        raise ProtocolError("Strict source-OOF execution topology drifted.")
    completed: dict[str, Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        loaded = load_development_source_checkpoint(task)
        if loaded is None:
            pending.append(task)
        else:
            completed[str(task["task_id"])] = loaded
    if pending:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(development_source_prediction_task, task): task
                for task in pending
            }
            for future in as_completed(futures):
                task = futures[future]
                future.result()
                loaded = load_development_source_checkpoint(task)
                if loaded is None:
                    raise ProtocolError("Strict source-OOF checkpoint is absent.")
                completed[str(task["task_id"])] = loaded
                print(
                    "[disagreement-regret] strict source-OOF physical fits "
                    f"{len(completed)}/{len(tasks)}",
                    flush=True,
                )
    if len(completed) != len(tasks):
        raise ProtocolError("Strict source-OOF checkpoint coverage is incomplete.")
    return MappingProxyType(completed)


def load_development_source_checkpoint(
    task: Mapping[str, object],
) -> Mapping[str, object] | None:
    json_path = Path(str(task["checkpoint_json_path"]))
    npz_path = Path(str(task["checkpoint_npz_path"]))
    if not json_path.is_file() and not npz_path.is_file():
        return None
    if not json_path.is_file() or not npz_path.is_file():
        raise ProtocolError("Strict source-OOF checkpoint is partially present.")
    payload = read_json(json_path)
    unhashed = {
        key: value for key, value in payload.items() if key != "checkpoint_hash"
    }
    if (
        payload.get("checkpoint_hash") != canonical_hash(unhashed)
        or payload.get("task_hash") != task.get("task_hash")
        or payload.get("task_id") != task.get("task_id")
        or payload.get("array_sha256") != sha256_file(npz_path)
        or payload.get("schema_version") != "midogpp_strict_source_oof_checkpoint_v1"
        or payload.get("physical_fit_count") != DEVELOPMENT_ACTION_COUNT_PER_TASK
        or payload.get("logical_prediction_count") != 2 * DEVELOPMENT_ACTION_COUNT_PER_TASK
        or payload.get("source_labels_available") is not False
        or payload.get("test_cache_admitted") is not False
    ):
        raise ProtocolError("Strict source-OOF checkpoint envelope drifted.")
    try:
        archive = np.load(npz_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Strict source-OOF checkpoint archive is unreadable.") from exc
    with archive:
        expected_members = (
            "source_probabilities_view_0",
            "source_probabilities_view_1",
            "scaler_mean",
            "scaler_scale",
            "coefficients",
            "intercepts",
        )
        if tuple(archive.files) != expected_members:
            raise ProtocolError("Strict source-OOF checkpoint members drifted.")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    views = task["evaluation_views"]
    actions = payload.get("actions")
    parameter_shape = (DEVELOPMENT_ACTION_COUNT_PER_TASK, FEATURE_DIM)
    if (
        arrays["source_probabilities_view_0"].shape
        != (DEVELOPMENT_ACTION_COUNT_PER_TASK, int(views[0]["row_count"]))
        or arrays["source_probabilities_view_1"].shape
        != (DEVELOPMENT_ACTION_COUNT_PER_TASK, int(views[1]["row_count"]))
        or any(arrays[name].shape != parameter_shape for name in ("scaler_mean", "scaler_scale", "coefficients"))
        or arrays["intercepts"].shape != (DEVELOPMENT_ACTION_COUNT_PER_TASK,)
        or any(arrays[name].dtype != np.float64 for name in ("scaler_mean", "scaler_scale", "coefficients", "intercepts"))
        or any(arrays[name].dtype != np.float32 for name in ("source_probabilities_view_0", "source_probabilities_view_1"))
        or not isinstance(actions, list)
        or len(actions) != DEVELOPMENT_ACTION_COUNT_PER_TASK
    ):
        raise ProtocolError("Strict source-OOF checkpoint geometry drifted.")
    for ordinal, (raw, expected_action) in enumerate(
        zip(actions, task["actions"], strict=True)
    ):
        if not isinstance(raw, Mapping) or not isinstance(expected_action, Mapping):
            raise ProtocolError("Strict source-OOF checkpoint action is malformed.")
        parameter_hash = classifier_parameter_sha256(
            arrays["scaler_mean"][ordinal],
            arrays["scaler_scale"][ordinal],
            arrays["coefficients"][ordinal],
            arrays["intercepts"][ordinal],
        )
        logical = raw.get("logical_predictions")
        if (
            raw.get("action_id") != expected_action.get("action_id")
            or raw.get("action_hash") != expected_action.get("action_hash")
            or raw.get("parameter_sha256") != parameter_hash
            or raw.get("converged") is not True
            or not isinstance(logical, list)
            or len(logical) != 2
        ):
            raise ProtocolError("Strict source-OOF checkpoint classifier drifted.")
        for view_ordinal, logical_row in enumerate(logical):
            values = arrays[f"source_probabilities_view_{view_ordinal}"][ordinal]
            view = views[view_ordinal]
            if (
                not isinstance(logical_row, Mapping)
                or logical_row.get("outer_target") != view["outer_target"]
                or logical_row.get("query_center") != view["query_center"]
                or logical_row.get("orientation_hash")
                != view["orientation_hashes"][ordinal]
                or logical_row.get("row_identity_hash") != view["row_identity_hash"]
                or logical_row.get("probability_sha256") != sha256_array(values)
                or logical_row.get("predictions_sha256")
                != sha256_array((values >= np.float32(0.5)).astype(np.uint8))
            ):
                raise ProtocolError("Strict source-OOF logical prediction drifted.")
    return payload


def assemble_development_source_products(
    root: Path,
    *,
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[str, Mapping[str, object]],
) -> tuple[tuple[DevelopmentClassifierCell, ...], tuple[DevelopmentPredictionCell, ...]]:
    paths = tuple(
        root / member
        for member in (
            DEVELOPMENT_CLASSIFIER_MEAN_MEMBER,
            DEVELOPMENT_CLASSIFIER_SCALE_MEMBER,
            DEVELOPMENT_CLASSIFIER_COEFFICIENT_MEMBER,
            DEVELOPMENT_CLASSIFIER_INTERCEPT_MEMBER,
        )
    )
    temporary = tuple(path.with_suffix(path.suffix + f".{os.getpid()}.tmp") for path in paths)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    matrices = (
        np.lib.format.open_memmap(temporary[0], mode="w+", dtype=np.float64, shape=(DEVELOPMENT_CLASSIFIER_FIT_COUNT, FEATURE_DIM)),
        np.lib.format.open_memmap(temporary[1], mode="w+", dtype=np.float64, shape=(DEVELOPMENT_CLASSIFIER_FIT_COUNT, FEATURE_DIM)),
        np.lib.format.open_memmap(temporary[2], mode="w+", dtype=np.float64, shape=(DEVELOPMENT_CLASSIFIER_FIT_COUNT, FEATURE_DIM)),
        np.lib.format.open_memmap(temporary[3], mode="w+", dtype=np.float64, shape=(DEVELOPMENT_CLASSIFIER_FIT_COUNT,)),
    )
    classifier_cells: list[DevelopmentClassifierCell] = []
    logical_by_key: dict[tuple[str, str, str, int, int], DevelopmentPredictionCell] = {}
    cursor = 0
    try:
        for task in tasks:
            payload = completed[str(task["task_id"])]
            with np.load(Path(str(task["checkpoint_npz_path"])), allow_pickle=False) as archive:
                probabilities = (
                    np.asarray(archive["source_probabilities_view_0"], dtype=np.float32),
                    np.asarray(archive["source_probabilities_view_1"], dtype=np.float32),
                )
                means = np.asarray(archive["scaler_mean"], dtype=np.float64)
                scales = np.asarray(archive["scaler_scale"], dtype=np.float64)
                coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
                intercepts = np.asarray(archive["intercepts"], dtype=np.float64)
            pair = tuple(str(value) for value in task["excluded_pair"])
            for ordinal, raw in enumerate(payload["actions"]):
                for matrix, rows in zip(matrices, (means, scales, coefficients, intercepts), strict=True):
                    matrix[cursor] = rows[ordinal]
                classifier_cells.append(
                    DevelopmentClassifierCell(
                        cell_ordinal=cursor,
                        excluded_pair=pair,  # type: ignore[arg-type]
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
                for view_ordinal, logical in enumerate(raw["logical_predictions"]):
                    values = probabilities[view_ordinal][ordinal]
                    cell = DevelopmentPredictionCell(
                        outer_target=str(logical["outer_target"]),
                        query_center=str(logical["query_center"]),
                        action_id=str(raw["action_id"]),
                        action_hash=str(raw["action_hash"]),
                        orientation_hash=str(logical["orientation_hash"]),
                        training_seed=int(task["training_seed"]),
                        generation_seed=int(task["generation_seed"]),
                        row_identity_hash=str(logical["row_identity_hash"]),
                        probabilities=values,
                        probability_sha256=str(logical["probability_sha256"]),
                        predictions_sha256=str(logical["predictions_sha256"]),
                        classifier_parameter_sha256=str(raw["parameter_sha256"]),
                    )
                    if cell.key in logical_by_key:
                        raise ProtocolError("Strict source-OOF logical cell duplicated.")
                    logical_by_key[cell.key] = cell
                cursor += 1
        for matrix in matrices:
            matrix.flush()
    finally:
        del matrices
    if cursor != DEVELOPMENT_CLASSIFIER_FIT_COUNT:
        raise ProtocolError("Strict source-OOF physical product coverage drifted.")
    for source, destination in zip(temporary, paths, strict=True):
        os.replace(source, destination)
    classifiers = tuple(classifier_cells)
    predictions = tuple(logical_by_key[key] for key in canonical_logical_cell_keys())
    if (
        tuple(cell.key for cell in classifiers) != canonical_physical_cell_keys()
        or len(predictions) != DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT
    ):
        raise ProtocolError("Strict source-OOF assembled order drifted.")
    return classifiers, predictions


def write_development_classifier_bank(
    root: Path,
    *,
    cells: Sequence[DevelopmentClassifierCell],
    config_contract_hash: str,
    source_stream_lock_hash: str,
    action_library_hash: str,
    source_cache_binding_hash: str,
) -> DevelopmentClassifierBank:
    ordered = tuple(cells)
    if tuple(cell.key for cell in ordered) != canonical_physical_cell_keys():
        raise ProtocolError("Strict source-OOF classifier cells are not canonical.")
    paths = tuple(
        root / member
        for member in (
            DEVELOPMENT_CLASSIFIER_MEAN_MEMBER,
            DEVELOPMENT_CLASSIFIER_SCALE_MEMBER,
            DEVELOPMENT_CLASSIFIER_COEFFICIENT_MEMBER,
            DEVELOPMENT_CLASSIFIER_INTERCEPT_MEMBER,
        )
    )
    if any(not path.is_file() for path in paths):
        raise ProtocolError("Strict source-OOF classifier arrays are incomplete.")
    hashes = tuple(sha256_file(path) for path in paths)
    bank_unhashed = {
        "schema_version": "midogpp_strict_source_oof_classifier_bank_v1",
        "config_contract_hash": config_contract_hash,
        "source_stream_lock_hash": source_stream_lock_hash,
        "action_library_hash": action_library_hash,
        "source_cache_binding_hash": source_cache_binding_hash,
        "physical_fit_count": len(ordered),
        "cells": [cell.to_payload() for cell in ordered],
        "parameter_file_sha256": dict(zip(("mean", "scale", "coefficient", "intercept"), hashes, strict=True)),
        "unordered_excluded_pair_fit_reuse": True,
        "source_labels_available_during_fit": False,
        "test_cache_admitted": False,
    }
    bank_hash = canonical_hash(bank_unhashed)
    index_path = root / DEVELOPMENT_CLASSIFIER_INDEX_MEMBER
    atomic_json(index_path, {**bank_unhashed, "classifier_bank_hash": bank_hash})
    seal_unhashed = {
        "schema_version": "midogpp_strict_source_oof_classifier_bank_seal_v1",
        "status": DEVELOPMENT_CLASSIFIER_STATUS,
        "config_contract_hash": config_contract_hash,
        "classifier_bank_hash": bank_hash,
        "classifier_bank_index_sha256": sha256_file(index_path),
        "source_stream_lock_hash": source_stream_lock_hash,
        "action_library_hash": action_library_hash,
        "source_cache_binding_hash": source_cache_binding_hash,
        "scaler_mean_file_sha256": hashes[0],
        "scaler_scale_file_sha256": hashes[1],
        "coefficient_file_sha256": hashes[2],
        "intercept_file_sha256": hashes[3],
        "physical_fit_count": DEVELOPMENT_CLASSIFIER_FIT_COUNT,
        "physical_task_count": DEVELOPMENT_PHYSICAL_TASK_COUNT,
        "physical_actions_per_task": DEVELOPMENT_ACTION_COUNT_PER_TASK,
        "unordered_excluded_pair_fit_reuse": True,
        "source_labels_available_during_fit": False,
        "test_cache_admitted": False,
    }
    seal_path = root / DEVELOPMENT_CLASSIFIER_SEAL_MEMBER
    atomic_json(seal_path, {**seal_unhashed, "development_classifier_bank_seal_hash": canonical_hash(seal_unhashed)})
    return load_development_classifier_bank(
        root,
        expected_config_hash=config_contract_hash,
        expected_source_stream_lock_hash=source_stream_lock_hash,
        expected_source_cache_binding_hash=source_cache_binding_hash,
    )


def load_development_classifier_bank(
    root: Path,
    *,
    expected_config_hash: str | None = None,
    expected_source_stream_lock_hash: str | None = None,
    expected_source_cache_binding_hash: str | None = None,
) -> DevelopmentClassifierBank:
    index_path = root / DEVELOPMENT_CLASSIFIER_INDEX_MEMBER
    seal_path = root / DEVELOPMENT_CLASSIFIER_SEAL_MEMBER
    index, seal = read_json(index_path), read_json(seal_path)
    index_unhashed = {key: value for key, value in index.items() if key != "classifier_bank_hash"}
    raw_cells = index.get("cells")
    if (
        index.get("classifier_bank_hash") != canonical_hash(index_unhashed)
        or seal.get("classifier_bank_index_sha256") != sha256_file(index_path)
        or seal.get("classifier_bank_hash") != index.get("classifier_bank_hash")
        or seal.get("source_stream_lock_hash")
        != index.get("source_stream_lock_hash")
        or seal.get("action_library_hash") != index.get("action_library_hash")
        or seal.get("source_cache_binding_hash")
        != index.get("source_cache_binding_hash")
        or seal.get("config_contract_hash") != index.get("config_contract_hash")
        or not isinstance(raw_cells, list)
        or len(raw_cells) != DEVELOPMENT_CLASSIFIER_FIT_COUNT
        or (expected_config_hash is not None and index.get("config_contract_hash") != expected_config_hash)
        or (
            expected_source_stream_lock_hash is not None
            and index.get("source_stream_lock_hash")
            != expected_source_stream_lock_hash
        )
        or (
            expected_source_cache_binding_hash is not None
            and index.get("source_cache_binding_hash") != expected_source_cache_binding_hash
        )
    ):
        raise ProtocolError("Strict source-OOF classifier index drifted.")
    try:
        cells = tuple(
            DevelopmentClassifierCell(
                cell_ordinal=int(row["cell_ordinal"]),
                excluded_pair=tuple(str(value) for value in row["excluded_pair"]),  # type: ignore[arg-type]
                action_id=str(row["action_id"]),
                action_hash=str(row["action_hash"]),
                training_seed=int(row["training_seed"]),
                generation_seed=int(row["generation_seed"]),
                composition_hash=str(row["composition_hash"]),
                scaler_state_hash=str(row["scaler_state_hash"]),
                parameter_sha256=str(row["parameter_sha256"]),
                fit_provenance_hash=str(row["fit_provenance_hash"]),
                classifier_config_hash=str(row["classifier_config_hash"]),
                n_iter=tuple(int(value) for value in row["n_iter"]),
                converged=bool(row["converged"]),
            )
            for row in raw_cells
            if isinstance(row, Mapping)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Strict source-OOF classifier cell is malformed.") from exc
    return DevelopmentClassifierBank(
        root=root,
        cells=cells,
        source_stream_lock_hash=str(index["source_stream_lock_hash"]),
        action_library_hash=str(index["action_library_hash"]),
        source_cache_binding_hash=str(index["source_cache_binding_hash"]),
        config_contract_hash=str(index["config_contract_hash"]),
        bank_hash=str(index["classifier_bank_hash"]),
        seal_payload=seal,
    )


def write_development_prediction_store(
    root: Path,
    *,
    cells: Sequence[DevelopmentPredictionCell],
    rows_by_query: Mapping[str, Sequence[str]],
    case_ids_by_query: Mapping[str, Sequence[str]],
    frame_cache_binding_hash: str,
    action_library_hash: str,
    classifier_bank_seal_hash: str,
    config_contract_hash: str,
) -> DevelopmentPredictionStore:
    ordered = tuple(cells)
    store_hash = development_prediction_store_hash(
        ordered,
        rows_by_query=rows_by_query,
        case_ids_by_query=case_ids_by_query,
        frame_cache_binding_hash=frame_cache_binding_hash,
        action_library_hash=action_library_hash,
        development_classifier_bank_seal_hash=classifier_bank_seal_hash,
    )
    array_path = root / DEVELOPMENT_PREDICTION_ARRAY_MEMBER
    atomic_npz(array_path, **{f"cell_{ordinal:05d}": cell.probabilities for ordinal, cell in enumerate(ordered)})
    unhashed = {
        "schema_version": "midogpp_strict_source_oof_prediction_index_v1",
        "config_contract_hash": config_contract_hash,
        "frame_cache_binding_hash": frame_cache_binding_hash,
        "action_library_hash": action_library_hash,
        "development_classifier_bank_seal_hash": classifier_bank_seal_hash,
        "prediction_store_hash": store_hash,
        "rows_by_query": {query: list(rows_by_query[query]) for query in CENTERS},
        "case_ids_by_query": {query: list(case_ids_by_query[query]) for query in CENTERS},
        "cells": [cell.index_payload(array_member=f"cell_{ordinal:05d}") for ordinal, cell in enumerate(ordered)],
        "logical_source_prediction_cell_count": len(ordered),
        "predictions_only_for_q_rows": True,
        "source_labels_consumed": False,
        "target_labels_available": False,
    }
    atomic_json(root / DEVELOPMENT_PREDICTION_INDEX_MEMBER, {**unhashed, "index_hash": canonical_hash(unhashed)})
    return load_development_prediction_store(
        root,
        expected_frame_cache_binding_hash=frame_cache_binding_hash,
        expected_classifier_bank_seal_hash=classifier_bank_seal_hash,
        expected_config_hash=config_contract_hash,
    )


def load_development_prediction_store(
    root: Path,
    *,
    expected_frame_cache_binding_hash: str | None = None,
    expected_classifier_bank_seal_hash: str | None = None,
    expected_config_hash: str | None = None,
) -> DevelopmentPredictionStore:
    array_path = root / DEVELOPMENT_PREDICTION_ARRAY_MEMBER
    index = read_json(root / DEVELOPMENT_PREDICTION_INDEX_MEMBER)
    unhashed = {key: value for key, value in index.items() if key != "index_hash"}
    raw_cells, raw_rows, raw_cases = index.get("cells"), index.get("rows_by_query"), index.get("case_ids_by_query")
    if (
        index.get("index_hash") != canonical_hash(unhashed)
        or index.get("logical_source_prediction_cell_count") != DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT
        or index.get("predictions_only_for_q_rows") is not True
        or index.get("source_labels_consumed") is not False
        or index.get("target_labels_available") is not False
        or not isinstance(raw_cells, list)
        or not isinstance(raw_rows, Mapping)
        or not isinstance(raw_cases, Mapping)
        or (expected_frame_cache_binding_hash is not None and index.get("frame_cache_binding_hash") != expected_frame_cache_binding_hash)
        or (expected_classifier_bank_seal_hash is not None and index.get("development_classifier_bank_seal_hash") != expected_classifier_bank_seal_hash)
        or (
            expected_config_hash is not None
            and index.get("config_contract_hash") != expected_config_hash
        )
    ):
        raise ProtocolError("Strict source-OOF prediction index drifted.")
    try:
        archive = np.load(array_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Strict source-OOF probability archive is unreadable.") from exc
    with archive:
        expected_members = tuple(f"cell_{ordinal:05d}" for ordinal in range(DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT))
        if tuple(archive.files) != expected_members:
            raise ProtocolError("Strict source-OOF probability members drifted.")
        cells = tuple(_prediction_cell(row, archive) for row in raw_cells if isinstance(row, Mapping))
    if len(cells) != len(raw_cells):
        raise ProtocolError("Strict source-OOF prediction cell is malformed.")
    try:
        rows = {query: tuple(str(value) for value in raw_rows[query]) for query in CENTERS}
        cases = {query: tuple(str(value) for value in raw_cases[query]) for query in CENTERS}
    except (KeyError, TypeError) as exc:
        raise ProtocolError("Strict source-OOF row identity maps drifted.") from exc
    return DevelopmentPredictionStore(
        cells=cells,
        rows_by_query=rows,
        case_ids_by_query=cases,
        frame_cache_binding_hash=str(index["frame_cache_binding_hash"]),
        action_library_hash=str(index["action_library_hash"]),
        development_classifier_bank_seal_hash=str(index["development_classifier_bank_seal_hash"]),
        store_hash=str(index["prediction_store_hash"]),
    )


def _prediction_cell(raw: Mapping[str, object], archive: object) -> DevelopmentPredictionCell:
    try:
        values = np.asarray(archive[str(raw["array_member"])], dtype=np.float32)  # type: ignore[index]
        return DevelopmentPredictionCell(
            outer_target=str(raw["outer_target"]),
            query_center=str(raw["query_center"]),
            action_id=str(raw["action_id"]),
            action_hash=str(raw["action_hash"]),
            orientation_hash=str(raw["orientation_hash"]),
            training_seed=int(raw["training_seed"]),
            generation_seed=int(raw["generation_seed"]),
            row_identity_hash=str(raw["row_identity_hash"]),
            probabilities=values,
            probability_sha256=str(raw["probability_sha256"]),
            predictions_sha256=str(raw["predictions_sha256"]),
            classifier_parameter_sha256=str(raw["classifier_parameter_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Strict source-OOF prediction cell is malformed.") from exc


def write_development_source_prediction_seal(
    root: Path,
    *,
    classifier_bank: DevelopmentClassifierBank,
    source_store: DevelopmentPredictionStore,
    config_contract_hash: str,
) -> DevelopmentSourcePredictionSeal:
    array_path = root / DEVELOPMENT_PREDICTION_ARRAY_MEMBER
    index_path = root / DEVELOPMENT_PREDICTION_INDEX_MEMBER
    seal_path = root / DEVELOPMENT_PREDICTION_SEAL_MEMBER
    unhashed = {
        "schema_version": "midogpp_strict_source_oof_prediction_seal_v1",
        "status": DEVELOPMENT_PREDICTION_STATUS,
        "config_contract_hash": config_contract_hash,
        "classifier_bank_seal_hash": classifier_bank.seal_hash,
        "source_prediction_store_hash": source_store.store_hash,
        "source_prediction_array_sha256": sha256_file(array_path),
        "source_prediction_index_sha256": sha256_file(index_path),
        "physical_fit_count": DEVELOPMENT_CLASSIFIER_FIT_COUNT,
        "logical_source_prediction_cell_count": DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT,
        "predictions_only_for_q_rows": True,
        "unordered_excluded_pair_fit_reuse": True,
        "query_excluded_from_every_composition": True,
        "outer_target_excluded_from_every_composition": True,
        "source_labels_opened": False,
        "test_cache_admitted": False,
        "target_labels_available": False,
    }
    atomic_json(seal_path, {**unhashed, "source_prediction_seal_hash": canonical_hash(unhashed)})
    return load_development_source_prediction_seal(
        root,
        expected_config_hash=config_contract_hash,
        expected_source_stream_lock_hash=classifier_bank.source_stream_lock_hash,
        expected_source_cache_binding_hash=source_store.frame_cache_binding_hash,
    )


def load_development_source_prediction_seal(
    root: Path,
    *,
    expected_config_hash: str | None = None,
    expected_source_stream_lock_hash: str | None = None,
    expected_source_cache_binding_hash: str | None = None,
) -> DevelopmentSourcePredictionSeal:
    bank = load_development_classifier_bank(
        root,
        expected_config_hash=expected_config_hash,
        expected_source_stream_lock_hash=expected_source_stream_lock_hash,
        expected_source_cache_binding_hash=expected_source_cache_binding_hash,
    )
    store = load_development_prediction_store(
        root,
        expected_frame_cache_binding_hash=expected_source_cache_binding_hash,
        expected_classifier_bank_seal_hash=bank.seal_hash,
        expected_config_hash=expected_config_hash,
    )
    seal_path = root / DEVELOPMENT_PREDICTION_SEAL_MEMBER
    seal = read_json(seal_path)
    if seal.get("source_prediction_index_sha256") != sha256_file(root / DEVELOPMENT_PREDICTION_INDEX_MEMBER):
        raise ProtocolError("Strict source-OOF prediction index lineage drifted.")
    return DevelopmentSourcePredictionSeal(
        classifier_bank=bank,
        source_store=store,
        seal_payload=seal,
        arrays_path=root / DEVELOPMENT_PREDICTION_ARRAY_MEMBER,
        index_path=root / DEVELOPMENT_PREDICTION_INDEX_MEMBER,
        seal_path=seal_path,
    )


__all__ = tuple(
    name
    for name in globals()
    if name.startswith(("assemble_", "execute_", "load_", "write_"))
)
