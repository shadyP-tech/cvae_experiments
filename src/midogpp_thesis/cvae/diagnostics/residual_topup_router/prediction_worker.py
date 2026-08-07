"""Spawn-safe classifier worker and hash-validated task checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from ...generation.generation import derived_composition_seed
from ...protocol import ProtocolError
from ...routing.residual_topup import (
    build_energy_directed_topup_action,
    build_uniform_topup_action,
    compose_equal_union_base_blocks,
    compose_residual_topup_blocks,
    inner_topup_geometry,
    target_topup_geometry,
)
from .artifact_io import atomic_save_npz, atomic_write_json, read_json, sha256_file
from .contracts import (
    BASE_ONLY_ACTION_ID,
    COMMON_FEATURE_DIM,
    ENERGY_TOPUP_ACTION_ID,
    EXPECTED_SOURCE_BLOCK_COUNT,
    MAX_SOURCE_PREFIX_PER_CLASS,
    UNIFORM_TOPUP_ACTION_ID,
)


PREDICTION_CHECKPOINT_DIRECTORY = "checkpoints/predictions"


def prediction_task(task: Mapping[str, object]) -> Mapping[str, object]:
    """Materialize all actions for one query/target and retained seed cell."""

    phase = str(task["phase"])
    outer = str(task["outer_target"])
    query = str(task["query_center"])
    training_seed = int(task["training_seed"])
    generation_seed = int(task["generation_seed"])
    candidates = tuple(str(value) for value in task["candidate_sources"])
    if phase not in {"development", "target"}:
        raise ProtocolError("Residual top-up prediction phase is invalid.")
    expected_source_count = 7 if phase == "development" else 8
    if (
        len(candidates) != expected_source_count
        or len(set(candidates)) != len(candidates)
        or query in candidates
        or (phase == "development" and outer in candidates)
    ):
        raise ProtocolError("Residual top-up worker candidate geometry drifted.")
    source_array = np.load(Path(str(task["source_array_path"])), mmap_mode="r")
    if source_array.shape != (
        EXPECTED_SOURCE_BLOCK_COUNT,
        2 * MAX_SOURCE_PREFIX_PER_CLASS,
        COMMON_FEATURE_DIM,
    ) or source_array.dtype != np.float32:
        raise ProtocolError("Residual top-up worker source cache drifted.")
    index_rows = tuple(task["source_index_rows"])
    index = {
        (str(row["source_center"]), int(row["training_seed"]), int(row["generation_seed"])): row
        for row in index_rows
    }
    blocks: dict[str, Mapping[str, object]] = {}
    stream_ids: dict[str, str] = {}
    expert_hashes: dict[str, str] = {}
    for source in candidates:
        try:
            row = index[(source, training_seed, generation_seed)]
        except KeyError as exc:
            raise ProtocolError("Residual top-up worker source block is absent.") from exc
        ordinal = int(row["block_ordinal"])
        if ordinal < 0 or ordinal >= len(source_array):
            raise ProtocolError("Residual top-up source block ordinal drifted.")
        blocks[source] = {
            "embeddings": source_array[ordinal],
            "labels": np.concatenate(
                (
                    np.zeros(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
                    np.ones(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
                )
            ),
        }
        stream_ids[source] = str(row["stream_id"])
        expert_hashes[source] = str(row["expert_lock_hash"])
    evaluation = np.load(Path(str(task["evaluation_array_path"])), mmap_mode="r")
    start, stop = int(task["evaluation_start"]), int(task["evaluation_stop"])
    if evaluation.ndim != 2 or evaluation.shape[1] != COMMON_FEATURE_DIM or not (0 <= start < stop <= len(evaluation)):
        raise ProtocolError("Residual top-up evaluation scratch drifted.")
    eval_matrix = np.ascontiguousarray(evaluation[start:stop], dtype=np.float32)
    shuffle_seeds = {
        label: derived_composition_seed(
            generation_lock_hash=str(task["generation_lock_hash"]),
            target_center=query,
            training_seed=training_seed,
            generation_seed=generation_seed,
            class_label=label,
        )
        for label in (0, 1)
    }
    classifier = _classifier(task["classifier"])
    plans = tuple(task["plans"])
    cells: list[dict[str, object]] = []
    fitted_by_composition: dict[str, Mapping[str, object]] = {}
    for raw_plan in plans:
        plan = dict(raw_plan)
        action_id = str(plan["action_id"])
        payload = _mapping(plan, "action_payload")
        geometry = (
            inner_topup_geometry(candidates)
            if phase == "development"
            else target_topup_geometry(candidates)
        )
        if action_id == BASE_ONLY_ACTION_ID:
            if phase != "target":
                raise ProtocolError("Residual top-up base-only arm is target-only.")
            composition = compose_equal_union_base_blocks(
                blocks, geometry, shuffle_seed_by_class=shuffle_seeds
            )
        elif action_id == UNIFORM_TOPUP_ACTION_ID:
            action = build_uniform_topup_action(geometry)
            if action.action_hash != payload.get("action_hash"):
                raise ProtocolError("Residual top-up uniform action hash drifted.")
            composition = compose_residual_topup_blocks(
                blocks, action, shuffle_seed_by_class=shuffle_seeds
            )
        elif action_id == ENERGY_TOPUP_ACTION_ID:
            energy = {
                str(source): float(value)
                for source, value in _mapping(payload, "calibrated_energy_by_source").items()
            }
            action = build_energy_directed_topup_action(energy, geometry=geometry)
            if action.action_hash != payload.get("action_hash"):
                raise ProtocolError("Residual top-up energy action hash drifted.")
            composition = compose_residual_topup_blocks(
                blocks, action, shuffle_seed_by_class=shuffle_seeds
            )
        else:
            raise ProtocolError("Residual top-up worker action is unknown.")
        aliased = composition.composition_hash in fitted_by_composition
        fitted = fitted_by_composition.get(composition.composition_hash)
        if fitted is None:
            fitted = _fit_classifier(
                classifier,
                composition.embeddings,
                composition.labels,
                eval_matrix,
                threads=int(task["threads_per_fit"]),
            )
            fitted_by_composition[composition.composition_hash] = fitted
        geometry_payload = _mapping(payload, "geometry")
        metadata = {
            "schema_version": "midogpp_residual_topup_prediction_cell_v1",
            "config_contract_hash": str(task["config_contract_hash"]),
            "generation_lock_hash": str(task["generation_lock_hash"]),
            "source_cache_lock_hash": str(task["source_cache_lock_hash"]),
            "router_plan_lock_hash": str(task["router_plan_lock_hash"]),
            "phase": phase,
            "outer_target": outer,
            "query_center": query,
            "action_id": action_id,
            "arm_role": str(plan["arm_role"]),
            "budget_role": str(plan["budget_role"]),
            "training_seed": training_seed,
            "generation_seed": generation_seed,
            "candidate_sources_json": _compact(list(candidates)),
            "source_stream_ids_json": _compact(stream_ids),
            "expert_lock_hashes_json": _compact(expert_hashes),
            "base_per_source": int(geometry_payload["base_per_source"]),
            "base_total_per_class": int(geometry_payload["base_total_per_class"]),
            "topup_total_per_class": int(geometry_payload["topup_total_per_class"]),
            "final_total_per_class": int(geometry_payload["final_total_per_class"]),
            "topup_counts_json": _compact(payload["topup_counts"]),
            "final_counts_by_class_json": _compact(payload["final_counts_by_class"]),
            "final_weights_by_class_json": _compact(payload["final_weights_by_class"]),
            "windows_by_class_json": _compact(payload["windows_by_class"]),
            "shuffle_seed_by_class_json": _compact({str(key): value for key, value in shuffle_seeds.items()}),
            "action_hash": str(payload["action_hash"]),
            "allocation_hash": str(payload["allocation_hash"]),
            "window_hash": str(payload["window_hash"]),
            "composition_hash": composition.composition_hash,
            "composition_output_sha256": composition.output_sha256,
            "classifier_config_hash": str(fitted["classifier_config_hash"]),
            "scaler_state_hash": str(fitted["scaler_state_hash"]),
            "classifier_n_iter_json": _compact(list(fitted["n_iter"])),
            "classifier_converged": bool(fitted["converged"]),
            "evaluation_row_ids_json": _compact(list(task["evaluation_row_ids"])),
            "evaluation_row_identity_hash": str(task["evaluation_row_identity_hash"]),
            "plan_hash": str(plan["plan_hash"]),
            "labels_available_to_fit_or_predict": False,
            "support_labels_used": False,
            "target_expert_excluded": True,
            "outer_and_query_experts_excluded": phase == "development",
            "seed_selection_performed": False,
            "fit_aliased_by_composition_hash": aliased,
            "selection_source": str(plan["selection_source"]),
            "claim_role": str(plan["claim_role"]),
        }
        cells.append(
            {
                "action_id": action_id,
                "predictions": fitted["predictions"],
                "probabilities": fitted["probabilities"],
                "metadata": metadata,
            }
        )
    result = {
        "task_hash": str(task["task_hash"]),
        "task_id": str(task["task_id"]),
        "unique_classifier_fit_count": len(fitted_by_composition),
        "cells": cells,
    }
    write_prediction_checkpoint(
        Path(str(task["checkpoint_json_path"])),
        Path(str(task["checkpoint_npz_path"])),
        result,
    )
    return {
        "task_id": str(task["task_id"]),
        "checkpoint_json_path": str(task["checkpoint_json_path"]),
    }


def write_prediction_checkpoint(
    json_path: Path,
    npz_path: Path,
    result: Mapping[str, object],
) -> None:
    cells = tuple(result["cells"])
    arrays: dict[str, object] = {}
    cell_rows: list[dict[str, object]] = []
    for index, raw in enumerate(cells):
        cell = dict(raw)
        predictions = np.asarray(cell["predictions"], dtype=np.uint8)
        probabilities = np.asarray(cell["probabilities"], dtype=np.float32)
        arrays[f"cell_{index}_predictions"] = predictions
        arrays[f"cell_{index}_probabilities"] = probabilities
        cell_rows.append(
            {
                "action_id": str(cell["action_id"]),
                "metadata": dict(cell["metadata"]),
                "prediction_sha256": _array_sha256(predictions),
                "probability_sha256": _array_sha256(probabilities),
                "row_count": len(predictions),
            }
        )
    atomic_save_npz(npz_path, **arrays)
    unhashed = {
        "schema_version": "midogpp_residual_topup_prediction_checkpoint_v1",
        "status": "COMPLETE",
        "task_hash": str(result["task_hash"]),
        "task_id": str(result["task_id"]),
        "unique_classifier_fit_count": int(result["unique_classifier_fit_count"]),
        "array_member": npz_path.name,
        "array_sha256": sha256_file(npz_path),
        "cells": cell_rows,
    }
    atomic_write_json(json_path, {**unhashed, "checkpoint_hash": stable_hash(unhashed)})


def load_prediction_checkpoint(
    json_path: Path,
    npz_path: Path,
    *,
    task: Mapping[str, object],
) -> Mapping[str, object]:
    payload = read_json(json_path)
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    cells = payload.get("cells")
    if (
        payload.get("checkpoint_hash") != stable_hash(unhashed)
        or payload.get("status") != "COMPLETE"
        or payload.get("task_hash") != task["task_hash"]
        or payload.get("task_id") != task["task_id"]
        or not npz_path.is_file()
        or payload.get("array_member") != npz_path.name
        or payload.get("array_sha256") != sha256_file(npz_path)
        or not isinstance(cells, list)
        or len(cells) != len(tuple(task["plans"]))
    ):
        raise ProtocolError("Residual top-up prediction checkpoint drifted.")
    output_cells: list[dict[str, object]] = []
    try:
        with np.load(npz_path, allow_pickle=False) as arrays:
            expected_keys = {
                f"cell_{index}_{kind}"
                for index in range(len(cells))
                for kind in ("predictions", "probabilities")
            }
            if set(arrays.files) != expected_keys:
                raise ProtocolError("Residual top-up checkpoint NPZ keys drifted.")
            for index, raw in enumerate(cells):
                if not isinstance(raw, Mapping):
                    raise ProtocolError("Residual top-up checkpoint cell is malformed.")
                predictions = np.asarray(arrays[f"cell_{index}_predictions"])
                probabilities = np.asarray(arrays[f"cell_{index}_probabilities"])
                if (
                    predictions.dtype != np.uint8
                    or probabilities.dtype != np.float32
                    or probabilities.shape != predictions.shape
                    or len(predictions) != int(raw["row_count"])
                    or _array_sha256(predictions) != raw["prediction_sha256"]
                    or _array_sha256(probabilities) != raw["probability_sha256"]
                    or str(raw["action_id"]) != str(tuple(task["plans"])[index]["action_id"])
                ):
                    raise ProtocolError("Residual top-up checkpoint arrays drifted.")
                output_cells.append(
                    {
                        "action_id": str(raw["action_id"]),
                        "predictions": predictions.copy(),
                        "probabilities": probabilities.copy(),
                        "metadata": dict(raw["metadata"]),
                    }
                )
    except (OSError, ValueError) as exc:
        raise ProtocolError("Residual top-up checkpoint arrays are unreadable.") from exc
    return {
        "task_id": str(payload["task_id"]),
        "unique_classifier_fit_count": int(payload["unique_classifier_fit_count"]),
        "cells": tuple(output_cells),
    }


def _fit_classifier(
    spec: ClassifierSpec,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    eval_embeddings: np.ndarray,
    *,
    threads: int,
) -> Mapping[str, object]:
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:
        raise RuntimeError("Residual top-up fitting requires threadpoolctl.") from exc
    if threads != 3:
        raise ProtocolError("Residual top-up worker requires three BLAS threads.")
    with threadpool_limits(limits=threads):
        fitted = fit_logistic_classifier(
            train_embeddings, train_labels, eval_embeddings, spec=spec
        )
    predictions = np.asarray(fitted.predictions)
    probabilities = np.asarray(fitted.probabilities, dtype=np.float64)
    if (
        tuple(int(value) for value in fitted.classes) != (0, 1)
        or predictions.shape != (len(eval_embeddings),)
        or probabilities.shape != (len(eval_embeddings), 2)
        or not np.isin(predictions, (0, 1)).all()
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
        or not fitted.converged
        or fitted.classifier_config_hash != spec.config_hash
    ):
        raise ProtocolError("Residual top-up classifier fit drifted.")
    return {
        "predictions": predictions.astype(np.uint8, copy=False),
        "probabilities": probabilities[:, 1].astype(np.float32, copy=False),
        "n_iter": tuple(int(value) for value in fitted.n_iter),
        "converged": True,
        "classifier_config_hash": fitted.classifier_config_hash,
        "scaler_state_hash": fitted.scaler_state_hash,
    }


def _classifier(payload: object) -> ClassifierSpec:
    if not isinstance(payload, Mapping):
        raise ProtocolError("Residual top-up classifier payload is malformed.")
    return ClassifierSpec(
        family=str(payload["family"]),
        C=float(payload["C"]),
        penalty=str(payload["penalty"]),
        solver=str(payload["solver"]),
        max_iter=int(payload["max_iter"]),
        class_weight=None if payload["class_weight"] is None else str(payload["class_weight"]),
        random_state=int(payload["random_state"]),
        l1_ratio=None if payload["l1_ratio"] is None else float(payload["l1_ratio"]),
        threshold_policy=str(payload["threshold_policy"]),
        scaler_fit=str(payload["scaler_fit"]),
    )


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ProtocolError(f"Residual top-up worker lacks mapping {key!r}.")
    return result


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = (
    "PREDICTION_CHECKPOINT_DIRECTORY",
    "load_prediction_checkpoint",
    "prediction_task",
    "write_prediction_checkpoint",
)
