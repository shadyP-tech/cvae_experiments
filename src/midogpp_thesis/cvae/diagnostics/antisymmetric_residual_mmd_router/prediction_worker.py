"""Spawn-safe target-by-seed prediction worker and resumable checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping

import numpy as np

from ....common.hashing import stable_hash
from ...generation.generation import derived_composition_seed
from ...protocol import ProtocolError
from ..mmd_kmm_router.inputs import LabelFreeValidationFrame
from ..mmd_kmm_router.source_products import SourceProducts
from ._prediction_common import (
    atomic_save_npy,
    atomic_save_npz,
    compact_json,
    sha256_array,
    require_mapping,
)
from .artifact_io import atomic_write_json
from .composition import (
    ClassSpecificComposition,
    arm_plan_payload,
    compose_class_specific_prefix_blocks,
    fit_classifier,
    fit_metadata,
    validate_plan,
)
from .contracts import (
    ARM_ROLES,
    CENTERS,
    CONTROL_ARM,
    EXPECTED_SEED_CELL_COUNT,
    GENERATION_SEEDS,
    MAX_SOURCE_PREFIX_PER_CLASS,
    ROUTED_ARM,
    TRAINING_SEEDS,
    candidate_sources,
    row_identity_hash,
)
from .partitions import CrossfitSurface


PREDICTION_CHECKPOINT_DIRECTORY = "checkpoints/crossfit_predictions"

FitClassifier = Callable[..., dict[str, object]]
ComposeBlocks = Callable[..., ClassSpecificComposition]


def build_prediction_tasks(
    config: object,
    generation_lock_hash: str,
    source_products: SourceProducts,
    plan_map: Mapping[str, Mapping[str, object]],
    plan_lock_hash: str,
    crossfit: CrossfitSurface,
    *,
    source_products_lock_hash: str,
    scratch: Mapping[str, object],
    scratch_path: Path,
    checkpoint_root: Path,
) -> tuple[dict[str, object], ...]:
    """Build the fixed 81 target×seed tasks for spawn workers."""

    tasks: list[dict[str, object]] = []
    source_index_rows = [dict(row) for row in source_products.index_rows]
    for target in CENTERS:
        target_folds = crossfit.folds_by_target[target]
        fold_tasks: list[dict[str, object]] = []
        for fold in target_folds:
            plan = plan_map.get(fold.fold_id)
            validate_plan(plan, fold=fold)
            scratch_fold = require_mapping(
                require_mapping(scratch, "folds"), fold.fold_id
            )
            fold_tasks.append(
                {
                    "fold_ordinal": fold.fold_ordinal,
                    "fold_id": fold.fold_id,
                    "fold_hash": fold.fold_hash,
                    "heldout_case_id": fold.heldout_case_id,
                    "router_support_row_identity_hash": row_identity_hash(
                        fold.router_support_rows
                    ),
                    "evaluation_row_ids": tuple(
                        row.sample_id for row in fold.heldout_rows
                    ),
                    "evaluation_row_identity_hash": row_identity_hash(
                        fold.heldout_rows
                    ),
                    "evaluation_start": scratch_fold["start"],
                    "evaluation_stop": scratch_fold["stop"],
                    "plan": dict(plan),
                }
            )
        scratch_target = require_mapping(
            require_mapping(scratch, "targets"), target
        )
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                tasks.append(
                    {
                        "config_contract_hash": config.contract_hash,
                        "generation_lock_hash": generation_lock_hash,
                        "source_products_lock_hash": source_products_lock_hash,
                        "router_plan_lock_hash": plan_lock_hash,
                        "crossfit_surface_lock_hash": crossfit.lock_hash,
                        "heldout_scratch_hash": scratch[
                            "heldout_scratch_hash"
                        ],
                        "target_center": target,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "fold_tasks": tuple(fold_tasks),
                        "fold_plan_hash": stable_hash(
                            [
                                {
                                    "fold_id": fold_task["fold_id"],
                                    "fold_hash": fold_task["fold_hash"],
                                    "plan_hash": fold_task["plan"]["plan_hash"],
                                }
                                for fold_task in fold_tasks
                            ]
                        ),
                        "source_array_path": str(source_products.array_path),
                        "source_index_rows": source_index_rows,
                        "evaluation_array_path": str(scratch_path),
                        "target_evaluation_start": scratch_target["start"],
                        "target_evaluation_stop": scratch_target["stop"],
                        "target_evaluation_row_identity_hash": scratch_target[
                            "row_identity_hash"
                        ],
                        "classifier": config.classifier,
                        "threads_per_worker": int(
                            config.runtime["classifier_threads_per_worker"]
                        ),
                        "checkpoint_path": str(
                            checkpoint_root
                            / (
                                f"target_{target}_train_{training_seed}_"
                                f"gen_{generation_seed}.npz"
                            )
                        ),
                    }
                )
    expected_task_count = len(CENTERS) * EXPECTED_SEED_CELL_COUNT
    if len(tasks) != expected_task_count:
        raise ProtocolError("Antisymmetric cross-fit task scheduler drifted.")
    return tuple(tasks)


def prediction_task(
    task: Mapping[str, object],
    *,
    fit_classifier_fn: FitClassifier | None = None,
    compose_blocks_fn: ComposeBlocks | None = None,
) -> dict[str, object]:
    """Fit one target×training-seed×generation-seed checkpoint surface."""

    fit = fit_classifier if fit_classifier_fn is None else fit_classifier_fn
    compose = (
        compose_class_specific_prefix_blocks
        if compose_blocks_fn is None
        else compose_blocks_fn
    )
    target = str(task["target_center"])
    training_seed = int(task["training_seed"])
    generation_seed = int(task["generation_seed"])
    candidates = candidate_sources(target)
    fold_tasks_raw = task.get("fold_tasks")
    if not isinstance(fold_tasks_raw, (tuple, list)) or not fold_tasks_raw:
        raise ProtocolError("Antisymmetric target task has no cross-fit folds.")
    fold_tasks: tuple[Mapping[str, object], ...] = tuple(
        require_mapping({"fold": value}, "fold") for value in fold_tasks_raw
    )

    source_array = np.load(Path(str(task["source_array_path"])), mmap_mode="r")
    block_index = {
        (
            str(row["source_center"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
        ): int(row["block_ordinal"])
        for row in task["source_index_rows"]
    }
    labels = np.concatenate(
        (
            np.zeros(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
            np.ones(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
        )
    )
    blocks: dict[str, object] = {}
    for source in candidates:
        try:
            ordinal = block_index[(source, training_seed, generation_seed)]
        except KeyError as exc:
            raise ProtocolError(
                "Antisymmetric source-product block coverage drifted."
            ) from exc
        blocks[source] = SimpleNamespace(
            embeddings=np.asarray(source_array[ordinal]),
            labels=labels,
            key=SimpleNamespace(source_center=source),
        )

    shuffle_seeds = {
        str(class_label): derived_composition_seed(
            generation_lock_hash=str(task["generation_lock_hash"]),
            target_center=target,
            training_seed=training_seed,
            generation_seed=generation_seed,
            class_label=class_label,
        )
        for class_label in (0, 1)
    }
    first_plan = require_mapping(fold_tasks[0], "plan")
    control_weights, control_allocations = arm_plan_payload(
        first_plan, CONTROL_ARM
    )
    control = compose(
        blocks,
        control_allocations,
        shuffle_seed_by_class=shuffle_seeds,
    )
    target_evaluation = np.asarray(
        np.load(Path(str(task["evaluation_array_path"])), mmap_mode="r")[
            int(task["target_evaluation_start"]) : int(
                task["target_evaluation_stop"]
            )
        ],
        dtype=np.float32,
    )
    if not len(target_evaluation):
        raise ProtocolError("Antisymmetric target evaluation surface is empty.")
    control_fit = fit(
        control.embeddings,
        control.labels,
        target_evaluation,
        classifier=task["classifier"],
        threads=int(task["threads_per_worker"]),
    )
    routed_fit_by_hash: dict[str, Mapping[str, object]] = {
        control.composition_hash: control_fit
    }
    routed_composition_hash_by_fold: dict[str, str] = {}
    routed_payload_by_fold: dict[
        str, tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]
    ] = {}
    for fold_task in fold_tasks:
        plan = require_mapping(fold_task, "plan")
        if (
            plan.get("target_center") != target
            or plan.get("fold_id") != fold_task.get("fold_id")
            or tuple(str(value) for value in plan.get("candidate_sources", ()))
            != candidates
        ):
            raise ProtocolError("Antisymmetric target task plan drifted.")
        fold_control_weights, fold_control_allocations = arm_plan_payload(
            plan, CONTROL_ARM
        )
        if (
            fold_control_weights != control_weights
            or fold_control_allocations != control_allocations
        ):
            raise ProtocolError(
                "Antisymmetric equal-union control drifted across target folds."
            )
        routed_weights, routed_allocations = arm_plan_payload(plan, ROUTED_ARM)
        routed = compose(
            blocks,
            routed_allocations,
            shuffle_seed_by_class=shuffle_seeds,
        )
        fold_id = str(fold_task["fold_id"])
        routed_composition_hash_by_fold[fold_id] = routed.composition_hash
        routed_payload_by_fold[fold_id] = (routed_weights, routed_allocations)
        if routed.composition_hash not in routed_fit_by_hash:
            routed_fit_by_hash[routed.composition_hash] = fit(
                routed.embeddings,
                routed.labels,
                target_evaluation,
                classifier=task["classifier"],
                threads=int(task["threads_per_worker"]),
            )

    result: dict[str, object] = {
        "schema_version": "midogpp_antisymmetric_residual_mmd_prediction_checkpoint_v1",
        "config_contract_hash": task["config_contract_hash"],
        "generation_lock_hash": task["generation_lock_hash"],
        "source_products_lock_hash": task["source_products_lock_hash"],
        "router_plan_lock_hash": task["router_plan_lock_hash"],
        "crossfit_surface_lock_hash": task["crossfit_surface_lock_hash"],
        "heldout_scratch_hash": task["heldout_scratch_hash"],
        "target_center": target,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "fold_plan_hash": task["fold_plan_hash"],
        "target_evaluation_row_identity_hash": task[
            "target_evaluation_row_identity_hash"
        ],
        "unique_classifier_fit_count": len(routed_fit_by_hash),
    }
    target_start = int(task["target_evaluation_start"])
    for fold_task in fold_tasks:
        fold_id = str(fold_task["fold_id"])
        plan = require_mapping(fold_task, "plan")
        routed_composition_hash = routed_composition_hash_by_fold[fold_id]
        routed_fit = routed_fit_by_hash[routed_composition_hash]
        routed_weights, routed_allocations = routed_payload_by_fold[fold_id]
        relative_start = int(fold_task["evaluation_start"]) - target_start
        relative_stop = int(fold_task["evaluation_stop"]) - target_start
        if not 0 <= relative_start < relative_stop <= len(target_evaluation):
            raise ProtocolError(
                "Antisymmetric fold slice escaped its target surface."
            )
        base_metadata = {
            "schema_version": "midogpp_antisymmetric_residual_mmd_prediction_cell_v1",
            "config_contract_hash": task["config_contract_hash"],
            "generation_lock_hash": task["generation_lock_hash"],
            "source_products_lock_hash": task["source_products_lock_hash"],
            "router_plan_lock_hash": task["router_plan_lock_hash"],
            "fold_ordinal": int(fold_task["fold_ordinal"]),
            "fold_id": fold_id,
            "fold_hash": fold_task["fold_hash"],
            "target_center": target,
            "heldout_case_id": fold_task["heldout_case_id"],
            "training_seed": training_seed,
            "generation_seed": generation_seed,
            "candidate_sources_json": compact_json(list(candidates)),
            "shuffle_seed_by_class_json": compact_json(shuffle_seeds),
            "router_support_row_identity_hash": fold_task[
                "router_support_row_identity_hash"
            ],
            "evaluation_row_ids_json": compact_json(
                list(fold_task["evaluation_row_ids"])
            ),
            "evaluation_row_identity_hash": fold_task[
                "evaluation_row_identity_hash"
            ],
            "plan_hash": plan["plan_hash"],
            "heldout_case_excluded_from_route": True,
            "labels_available_to_fit_or_predict": False,
            "support_labels_used": False,
            "seed_selection_performed": False,
        }
        prefix = f"fold_{int(fold_task['fold_ordinal'])}"
        result[f"{prefix}_{CONTROL_ARM}_predictions"] = np.asarray(
            control_fit["predictions"]
        )[relative_start:relative_stop]
        result[f"{prefix}_{CONTROL_ARM}_probabilities"] = np.asarray(
            control_fit["probabilities"]
        )[relative_start:relative_stop]
        result[f"{prefix}_{CONTROL_ARM}_metadata"] = {
            **base_metadata,
            "arm_role": CONTROL_ARM,
            "weights_by_class_json": compact_json(control_weights),
            "allocations_by_class_json": compact_json(control_allocations),
            "composition_hash": control.composition_hash,
            **fit_metadata(control_fit),
            "control_fit_aliased": False,
        }
        result[f"{prefix}_{ROUTED_ARM}_predictions"] = np.asarray(
            routed_fit["predictions"]
        )[relative_start:relative_stop]
        result[f"{prefix}_{ROUTED_ARM}_probabilities"] = np.asarray(
            routed_fit["probabilities"]
        )[relative_start:relative_stop]
        result[f"{prefix}_{ROUTED_ARM}_metadata"] = {
            **base_metadata,
            "arm_role": ROUTED_ARM,
            "weights_by_class_json": compact_json(routed_weights),
            "allocations_by_class_json": compact_json(routed_allocations),
            "composition_hash": routed_composition_hash,
            **fit_metadata(routed_fit),
            "control_fit_aliased": routed_composition_hash
            == control.composition_hash,
        }
    return result


def write_heldout_scratch(
    array_path: Path,
    index_path: Path,
    *,
    frame: LabelFreeValidationFrame,
    crossfit: CrossfitSurface,
) -> Mapping[str, object]:
    """Persist the label-free heldout embedding surface shared by workers."""

    heldout_rows = [row for fold in crossfit.folds for row in fold.heldout_rows]
    if len({row.sample_id for row in heldout_rows}) != len(heldout_rows):
        raise ProtocolError("Antisymmetric heldout scratch rows duplicate.")
    embeddings = frame.embeddings_for(heldout_rows)
    folds: dict[str, object] = {}
    cursor = 0
    for fold in crossfit.folds:
        stop = cursor + len(fold.heldout_rows)
        folds[fold.fold_id] = {
            "fold_ordinal": fold.fold_ordinal,
            "start": cursor,
            "stop": stop,
            "heldout_case_id": fold.heldout_case_id,
            "row_ids": [row.sample_id for row in fold.heldout_rows],
            "row_identity_hash": row_identity_hash(fold.heldout_rows),
            "fold_hash": fold.fold_hash,
        }
        cursor = stop
    targets: dict[str, object] = {}
    for target in CENTERS:
        target_folds = crossfit.folds_by_target[target]
        first = require_mapping(folds, target_folds[0].fold_id)
        last = require_mapping(folds, target_folds[-1].fold_id)
        target_rows = [
            row for fold in target_folds for row in fold.heldout_rows
        ]
        targets[target] = {
            "start": first["start"],
            "stop": last["stop"],
            "row_ids": [row.sample_id for row in target_rows],
            "row_identity_hash": row_identity_hash(target_rows),
            "fold_ids": [fold.fold_id for fold in target_folds],
        }
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_antisymmetric_residual_mmd_heldout_scratch_v1",
        "crossfit_surface_lock_hash": crossfit.lock_hash,
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "array_sha256": sha256_array(embeddings),
        "folds": folds,
        "targets": targets,
        "labels_present": False,
    }
    payload = {**unhashed, "heldout_scratch_hash": stable_hash(unhashed)}
    atomic_save_npy(array_path, embeddings)
    atomic_write_json(index_path, payload)
    return payload


def write_prediction_checkpoint(
    path: Path, result: Mapping[str, object]
) -> None:
    array_keys = tuple(
        sorted(
            key
            for key in result
            if key.endswith("_predictions") or key.endswith("_probabilities")
        )
    )
    if not array_keys:
        raise ProtocolError("Antisymmetric checkpoint has no prediction arrays.")
    arrays = {
        key: np.asarray(
            result[key],
            dtype=np.uint8 if key.endswith("predictions") else np.float32,
        )
        for key in array_keys
    }
    metadata = {key: value for key, value in result.items() if key not in arrays}
    metadata["array_hashes"] = {
        key: sha256_array(value) for key, value in arrays.items()
    }
    metadata["checkpoint_hash"] = stable_hash(metadata)
    atomic_save_npz(
        path,
        {**arrays, "checkpoint_json": np.asarray(compact_json(metadata))},
    )


def load_prediction_checkpoint(
    path: Path,
    *,
    task: Mapping[str, object],
) -> Mapping[str, object]:
    fold_tasks_raw = task.get("fold_tasks")
    if not isinstance(fold_tasks_raw, (tuple, list)):
        raise ProtocolError("Antisymmetric checkpoint task folds are malformed.")
    array_keys = tuple(
        sorted(
            f"fold_{int(fold_task['fold_ordinal'])}_{arm}_{kind}"
            for fold_task in fold_tasks_raw
            for arm in ARM_ROLES
            for kind in ("predictions", "probabilities")
        )
    )
    try:
        with np.load(path, allow_pickle=False) as raw:
            if set(raw.files) != set(array_keys).union({"checkpoint_json"}):
                raise ProtocolError(
                    "Antisymmetric checkpoint NPZ members drifted."
                )
            metadata = json.loads(str(np.asarray(raw["checkpoint_json"]).item()))
            arrays = {key: np.asarray(raw[key]) for key in array_keys}
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            "Antisymmetric prediction checkpoint is unreadable."
        ) from exc
    if not isinstance(metadata, Mapping):
        raise ProtocolError("Antisymmetric checkpoint metadata is malformed.")
    unhashed = {
        key: value for key, value in metadata.items() if key != "checkpoint_hash"
    }
    hashes = metadata.get("array_hashes")
    if (
        metadata.get("checkpoint_hash") != stable_hash(unhashed)
        or metadata.get("config_contract_hash") != task["config_contract_hash"]
        or metadata.get("generation_lock_hash") != task["generation_lock_hash"]
        or metadata.get("source_products_lock_hash")
        != task["source_products_lock_hash"]
        or metadata.get("router_plan_lock_hash")
        != task["router_plan_lock_hash"]
        or metadata.get("crossfit_surface_lock_hash")
        != task["crossfit_surface_lock_hash"]
        or metadata.get("heldout_scratch_hash") != task["heldout_scratch_hash"]
        or metadata.get("target_center") != task["target_center"]
        or int(metadata.get("training_seed", -1)) != int(task["training_seed"])
        or int(metadata.get("generation_seed", -1))
        != int(task["generation_seed"])
        or metadata.get("fold_plan_hash") != task["fold_plan_hash"]
        or metadata.get("target_evaluation_row_identity_hash")
        != task["target_evaluation_row_identity_hash"]
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(array_keys)
        or any(
            hashes.get(key) != sha256_array(value)
            for key, value in arrays.items()
        )
    ):
        raise ProtocolError(
            "Antisymmetric prediction checkpoint failed validation."
        )
    return {**metadata, **arrays}


def task_key(task: Mapping[str, object]) -> tuple[str, int, int]:
    return (
        str(task["target_center"]),
        int(task["training_seed"]),
        int(task["generation_seed"]),
    )


__all__ = (
    "PREDICTION_CHECKPOINT_DIRECTORY",
    "build_prediction_tasks",
    "load_prediction_checkpoint",
    "prediction_task",
    "task_key",
    "write_heldout_scratch",
    "write_prediction_checkpoint",
)
