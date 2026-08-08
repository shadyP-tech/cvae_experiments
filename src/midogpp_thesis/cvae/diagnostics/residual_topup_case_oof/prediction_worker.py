"""Spawn-safe classifier worker with hash-validated resumable checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...generation.generation import derived_composition_seed
from ...protocol import ProtocolError
from ...routing.residual_topup import (
    build_borda_directed_topup_action,
    build_single_source_tail_action,
    build_uniform_topup_action,
    compose_equal_union_base_blocks,
    compose_residual_topup_blocks,
    target_topup_geometry,
)
from .artifact_io import atomic_save_npz, atomic_write_json, read_json, sha256_file
from .contracts import (
    BASE_ACTION_ID,
    GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID,
    SUPPORT_ACTION_ID,
    UNIFORM_ACTION_ID,
    candidate_sources,
    expected_action_ids,
    tail_source,
)
from .prediction_store import array_sha256
from .source_cache_worker import MAX_SOURCE_PREFIX_PER_CLASS


PREDICTION_CHECKPOINT_DIRECTORY = "checkpoints/predictions"


def prediction_task(task: Mapping[str, object]) -> Mapping[str, object]:
    """Fit 13 fixed compositions once and slice predictions into case folds."""

    target = str(task["target_center"])
    training_seed = int(task["training_seed"])
    generation_seed = int(task["generation_seed"])
    candidates = tuple(str(value) for value in task["candidate_sources"])
    if candidates != candidate_sources(target):
        raise ProtocolError("Case-OOF prediction candidate geometry drifted.")
    if (
        task.get("labels_available") is not False
        or task.get("other_evaluation_embeddings_used_for_route") is not False
        or task.get("policy_selection_performed") is not False
        or task.get("fallback_performed") is not False
    ):
        raise ProtocolError("Case-OOF task escaped its label-free route boundary.")

    source_array = np.load(Path(str(task["source_array_path"])), mmap_mode="r")
    expected_blocks = 9 * 3 * 3
    if source_array.shape != (
        expected_blocks,
        2 * MAX_SOURCE_PREFIX_PER_CLASS,
        COMMON_OUTPUT_DIM,
    ) or source_array.dtype != np.float32:
        raise ProtocolError("Case-OOF worker source cache drifted.")
    source_index = {
        (
            str(row["source_center"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
        ): row
        for row in task["source_index_rows"]
    }
    blocks: dict[str, Mapping[str, object]] = {}
    stream_ids: dict[str, str] = {}
    expert_hashes: dict[str, str] = {}
    labels = np.concatenate(
        (
            np.zeros(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
            np.ones(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
        )
    )
    for source in candidates:
        try:
            row = source_index[(source, training_seed, generation_seed)]
        except KeyError as exc:
            raise ProtocolError("Case-OOF worker source block is absent.") from exc
        ordinal = int(row["block_ordinal"])
        if not 0 <= ordinal < len(source_array):
            raise ProtocolError("Case-OOF source-block ordinal drifted.")
        blocks[source] = {
            "embeddings": source_array[ordinal],
            "labels": labels,
        }
        stream_ids[source] = str(row["stream_id"])
        expert_hashes[source] = str(row["expert_lock_hash"])

    scratch = np.load(Path(str(task["evaluation_array_path"])), mmap_mode="r")
    if scratch.ndim != 2 or scratch.shape[1] != COMMON_OUTPUT_DIM:
        raise ProtocolError("Case-OOF evaluation scratch drifted.")
    folds = tuple(task["folds"])
    fold_arrays: list[np.ndarray] = []
    for fold in folds:
        start, stop = int(fold["start"]), int(fold["stop"])
        if not 0 <= start < stop <= len(scratch):
            raise ProtocolError("Case-OOF fold scratch offset drifted.")
        fold_arrays.append(
            np.ascontiguousarray(scratch[start:stop], dtype=np.float32)
        )
    evaluation = np.ascontiguousarray(np.concatenate(fold_arrays), dtype=np.float32)
    local_offsets: list[tuple[int, int]] = []
    cursor = 0
    for values in fold_arrays:
        local_offsets.append((cursor, cursor + len(values)))
        cursor += len(values)

    shuffle_seeds = {
        label: derived_composition_seed(
            generation_lock_hash=str(task["generation_lock_hash"]),
            target_center=target,
            training_seed=training_seed,
            generation_seed=generation_seed,
            class_label=label,
        )
        for label in (0, 1)
    }
    classifier = _classifier(task["classifier"])
    actions = tuple(task["actions"])
    if tuple(str(action["action_id"]) for action in actions) != expected_action_ids(
        target
    ):
        raise ProtocolError("Case-OOF worker action order drifted.")
    geometry = target_topup_geometry(candidates)
    fitted_by_composition: dict[str, Mapping[str, object]] = {}
    fitted_by_action: dict[str, Mapping[str, object]] = {}
    composition_by_action: dict[str, object] = {}
    aliased_by_action: dict[str, bool] = {}
    core_by_action: dict[str, object | None] = {}
    for payload in actions:
        action_id = str(payload["action_id"])
        core = _rebuild_core_action(payload, geometry=geometry)
        if core is None:
            composition = compose_equal_union_base_blocks(
                blocks, geometry, shuffle_seed_by_class=shuffle_seeds
            )
        else:
            if core.action_hash != str(payload["core_action_hash"]):
                raise ProtocolError("Case-OOF core action hash drifted.")
            composition = compose_residual_topup_blocks(
                blocks, core, shuffle_seed_by_class=shuffle_seeds
            )
        aliased = composition.composition_hash in fitted_by_composition
        fitted = fitted_by_composition.get(composition.composition_hash)
        if fitted is None:
            fitted = _fit_classifier(
                classifier,
                composition.embeddings,
                composition.labels,
                evaluation,
                threads=int(task["threads_per_fit"]),
            )
            fitted_by_composition[composition.composition_hash] = fitted
        fitted_by_action[action_id] = fitted
        composition_by_action[action_id] = composition
        aliased_by_action[action_id] = aliased
        core_by_action[action_id] = core

    cells: list[dict[str, object]] = []
    for fold, (local_start, local_stop) in zip(
        folds, local_offsets, strict=True
    ):
        for payload in actions:
            action_id = str(payload["action_id"])
            fitted = fitted_by_action[action_id]
            composition = composition_by_action[action_id]
            core = core_by_action[action_id]
            topup = {str(key): int(value) for key, value in _mapping(payload, "topup_counts_by_source").items()}
            final = _mapping(payload, "final_counts_by_class")
            if core is None:
                windows = _base_windows(candidates)
                allocation_hash = str(composition.allocation_hash)
                window_hash = str(composition.window_hash)
                core_hash = ""
            else:
                core_payload = core.to_payload()
                windows = core_payload["windows_by_class"]
                allocation_hash = str(core.allocation_hash)
                window_hash = str(core.window_hash)
                core_hash = str(core.action_hash)
                if (
                    topup != dict(core.topup_counts)
                    or final
                    != {
                        str(label): dict(core.final_counts_by_class[label])
                        for label in (0, 1)
                    }
                ):
                    raise ProtocolError("Case-OOF frozen/core action binding drifted.")
            metadata = {
                "schema_version": "midogpp_residual_topup_case_oof_prediction_cell_v1",
                "config_contract_hash": str(task["config_contract_hash"]),
                "generation_lock_hash": str(task["generation_lock_hash"]),
                "source_cache_lock_hash": str(task["source_cache_lock_hash"]),
                "crossfit_fold_lock_hash": str(task["crossfit_fold_lock_hash"]),
                "router_plan_lock_hash": str(task["router_plan_lock_hash"]),
                "fold_id": str(fold["fold_id"]),
                "fold_ordinal": int(fold["fold_ordinal"]),
                "target_center": target,
                "heldout_case_id": str(fold["heldout_case_id"]),
                "action_id": action_id,
                "action_role": str(payload["policy_id"]),
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "candidate_sources_json": _compact(list(candidates)),
                "source_stream_ids_json": _compact(stream_ids),
                "expert_lock_hashes_json": _compact(expert_hashes),
                "base_per_source": int(payload["base_per_source_per_class"]),
                "base_total_per_class": int(payload["base_per_source_per_class"]) * len(candidates),
                "topup_total_per_class": int(payload["topup_total_per_class"]),
                "final_total_per_class": int(payload["final_total_per_class"]),
                "topup_counts_json": _compact(topup),
                "final_counts_by_class_json": _compact(final),
                "windows_by_class_json": _compact(windows),
                "shuffle_seed_by_class_json": _compact({str(key): value for key, value in shuffle_seeds.items()}),
                "action_hash": str(payload["action_hash"]),
                "core_action_hash": core_hash,
                "allocation_hash": allocation_hash,
                "window_hash": window_hash,
                "composition_hash": str(composition.composition_hash),
                "composition_output_sha256": str(composition.output_sha256),
                "classifier_config_hash": str(fitted["classifier_config_hash"]),
                "scaler_state_hash": str(fitted["scaler_state_hash"]),
                "classifier_n_iter_json": _compact(list(fitted["n_iter"])),
                "classifier_converged": bool(fitted["converged"]),
                "evaluation_row_ids_json": _compact(list(fold["sample_ids"])),
                "evaluation_row_identity_hash": str(fold["row_identity_hash"]),
                "fold_hash": str(fold["fold_hash"]),
                "labels_available_to_fit_or_predict": False,
                "support_labels_used": False,
                "evaluation_embeddings_used_for_route": False,
                "other_evaluation_embeddings_used_for_route": False,
                "heldout_case_excluded_from_route": True,
                "target_expert_excluded": True,
                "global_excludes_target_and_query": action_id == GLOBAL_ACTION_ID,
                "seed_selection_performed": False,
                "policy_selection_performed": False,
                "fallback_performed": False,
                "fit_aliased_by_composition_hash": aliased_by_action[action_id],
                "claim_role": "terminal_consumed_validation_case_oof_diagnostic_only",
            }
            cells.append(
                {
                    "fold_id": str(fold["fold_id"]),
                    "action_id": action_id,
                    "predictions": np.asarray(fitted["predictions"])[
                        local_start:local_stop
                    ],
                    "probabilities": np.asarray(fitted["probabilities"])[
                        local_start:local_stop
                    ],
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
        predictions = np.asarray(raw["predictions"], dtype=np.uint8)
        probabilities = np.asarray(raw["probabilities"], dtype=np.float32)
        arrays[f"cell_{index}_predictions"] = predictions
        arrays[f"cell_{index}_probabilities"] = probabilities
        cell_rows.append(
            {
                "fold_id": str(raw["fold_id"]),
                "action_id": str(raw["action_id"]),
                "metadata": dict(raw["metadata"]),
                "prediction_sha256": array_sha256(predictions),
                "probability_sha256": array_sha256(probabilities),
                "row_count": len(predictions),
            }
        )
    atomic_save_npz(npz_path, **arrays)
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_case_oof_prediction_checkpoint_v1",
        "status": "COMPLETE",
        "task_hash": str(result["task_hash"]),
        "task_id": str(result["task_id"]),
        "unique_classifier_fit_count": int(result["unique_classifier_fit_count"]),
        "array_member": npz_path.name,
        "array_sha256": sha256_file(npz_path),
        "cells": cell_rows,
    }
    atomic_write_json(
        json_path, {**unhashed, "checkpoint_hash": stable_hash(unhashed)}
    )


def load_prediction_checkpoint(
    json_path: Path,
    npz_path: Path,
    *,
    task: Mapping[str, object],
) -> Mapping[str, object]:
    payload = read_json(json_path)
    unhashed = {
        key: value for key, value in payload.items() if key != "checkpoint_hash"
    }
    cells = payload.get("cells")
    expected_pairs = [
        (str(fold["fold_id"]), str(action["action_id"]))
        for fold in task["folds"]
        for action in task["actions"]
    ]
    if (
        payload.get("checkpoint_hash") != stable_hash(unhashed)
        or payload.get("status") != "COMPLETE"
        or payload.get("task_hash") != task["task_hash"]
        or payload.get("task_id") != task["task_id"]
        or not npz_path.is_file()
        or payload.get("array_member") != npz_path.name
        or payload.get("array_sha256") != sha256_file(npz_path)
        or not isinstance(cells, list)
        or len(cells) != len(expected_pairs)
    ):
        raise ProtocolError("Case-OOF prediction checkpoint drifted.")
    output: list[dict[str, object]] = []
    try:
        with np.load(npz_path, allow_pickle=False) as arrays:
            expected_keys = {
                f"cell_{index}_{kind}"
                for index in range(len(cells))
                for kind in ("predictions", "probabilities")
            }
            if set(arrays.files) != expected_keys:
                raise ProtocolError("Case-OOF checkpoint NPZ keys drifted.")
            for index, (raw, expected_pair) in enumerate(
                zip(cells, expected_pairs, strict=True)
            ):
                if not isinstance(raw, Mapping):
                    raise ProtocolError("Case-OOF checkpoint cell is malformed.")
                predictions = np.asarray(arrays[f"cell_{index}_predictions"])
                probabilities = np.asarray(arrays[f"cell_{index}_probabilities"])
                observed_pair = (str(raw["fold_id"]), str(raw["action_id"]))
                if (
                    observed_pair != expected_pair
                    or predictions.dtype != np.uint8
                    or probabilities.dtype != np.float32
                    or probabilities.shape != predictions.shape
                    or len(predictions) != int(raw["row_count"])
                    or array_sha256(predictions) != raw["prediction_sha256"]
                    or array_sha256(probabilities) != raw["probability_sha256"]
                ):
                    raise ProtocolError("Case-OOF checkpoint arrays drifted.")
                output.append(
                    {
                        "fold_id": observed_pair[0],
                        "action_id": observed_pair[1],
                        "predictions": predictions.copy(),
                        "probabilities": probabilities.copy(),
                        "metadata": dict(raw["metadata"]),
                    }
                )
    except (OSError, ValueError) as exc:
        raise ProtocolError("Case-OOF checkpoint arrays are unreadable.") from exc
    return {
        "task_id": str(payload["task_id"]),
        "unique_classifier_fit_count": int(payload["unique_classifier_fit_count"]),
        "cells": tuple(output),
    }


def _rebuild_core_action(payload: Mapping[str, object], *, geometry: object) -> object | None:
    action_id = str(payload["action_id"])
    if action_id == BASE_ACTION_ID:
        if payload.get("core_action_hash") is not None:
            raise ProtocolError("Case-OOF base action carries a core hash.")
        return None
    if action_id == UNIFORM_ACTION_ID:
        return build_uniform_topup_action(geometry)
    if action_id in {GLOBAL_ACTION_ID, SUPPORT_ACTION_ID, PERMUTATION_ACTION_ID}:
        ranks = {
            str(key): float(value)
            for key, value in _mapping(
                payload, "mean_normalized_midrank_by_source"
            ).items()
        }
        return build_borda_directed_topup_action(ranks, geometry=geometry)
    source = tail_source(action_id)
    if source is None or source != str(payload.get("selected_source")):
        raise ProtocolError("Case-OOF single-source action drifted.")
    return build_single_source_tail_action(source, geometry=geometry)


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
        raise RuntimeError("Case-OOF fitting requires threadpoolctl.") from exc
    if threads != 3:
        raise ProtocolError("Case-OOF worker requires three BLAS threads.")
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
        raise ProtocolError("Case-OOF classifier fit drifted.")
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
        raise ProtocolError("Case-OOF classifier payload is malformed.")
    return ClassifierSpec(
        family=str(payload["family"]),
        C=float(payload["C"]),
        penalty=str(payload["penalty"]),
        solver=str(payload["solver"]),
        max_iter=int(payload["max_iter"]),
        class_weight=(
            None
            if payload["class_weight"] is None
            else str(payload["class_weight"])
        ),
        random_state=int(payload["random_state"]),
        l1_ratio=(
            None if payload["l1_ratio"] is None else float(payload["l1_ratio"])
        ),
        threshold_policy=str(payload["threshold_policy"]),
        scaler_fit=str(payload["scaler_fit"]),
    )


def _base_windows(sources: tuple[str, ...]) -> dict[str, object]:
    return {
        str(label): {
            source: {
                "base": [0, 128],
                "topup": [128, 128],
                "base_count": 128,
                "topup_count": 0,
                "required_capacity": 128,
            }
            for source in sources
        }
        for label in (0, 1)
    }


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ProtocolError(f"Case-OOF worker lacks mapping {key!r}.")
    return result


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = (
    "PREDICTION_CHECKPOINT_DIRECTORY",
    "load_prediction_checkpoint",
    "prediction_task",
    "write_prediction_checkpoint",
)
