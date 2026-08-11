"""Deterministic source-fit and test-inference task planning."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.frozen_source_streams import FrozenSourceStreamCache
from .actions import actions_for_target
from .constants import (
    CENTERS,
    EXPECTED_CLASSIFIER_FIT_COUNT,
    EXPECTED_TASK_COUNT,
    GENERATION_SEEDS,
    PHYSICAL_ACTION_COUNT_PER_TARGET,
    PREDICTION_BATCH_ROWS,
    SOURCE_CHECKPOINT_DIRECTORY,
    TEST_CHECKPOINT_DIRECTORY,
    TRAINING_SEEDS,
    candidate_sources,
)
from .hashing import canonical_hash
from .prediction_contracts import ActionClassifierBank, ActionPredictionConfig


def build_source_tasks(
    config: ActionPredictionConfig,
    generated_sources: FrozenSourceStreamCache,
    *,
    scratch: Mapping[str, object],
    action_library_hash: str,
    root: Path,
) -> tuple[Mapping[str, object], ...]:
    records = [record.to_payload() for record in generated_sources.records]
    classifier = getattr(config, "classifier")
    classifier_payload = (
        classifier.to_payload() if hasattr(classifier, "to_payload") else dict(classifier)
    )
    task_root = root / SOURCE_CHECKPOINT_DIRECTORY / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)
    tasks: list[Mapping[str, object]] = []
    for target, training, generation in product(
        CENTERS, TRAINING_SEEDS, GENERATION_SEEDS
    ):
        task_id = f"source_target_{target}_train_{training}_generation_{generation}"
        actions = [action.to_payload() for action in actions_for_target(target)]
        unhashed = {
            "schema_version": "midogpp_prediction_only_source_fit_task_v1",
            "task_id": task_id,
            "config_contract_hash": config.contract_hash,
            "source_stream_lock_hash": generated_sources.lock_hash,
            "action_library_hash": action_library_hash,
            "target_center": target,
            "training_seed": training,
            "generation_seed": generation,
            "candidate_sources": list(candidate_sources(target)),
            "generated_array_path": str(generated_sources.source_array_path.resolve()),
            "generated_array_sha256": str(
                generated_sources.lock_payload["source_array_sha256"]
            ),
            "generated_index_rows": records,
            "generated_index_rows_hash": canonical_hash(records),
            "source_array_path": str(scratch["array_path"]),
            "source_array_file_sha256": str(scratch["array_file_sha256"]),
            "source_array_sha256": str(scratch["array_sha256"]),
            "source_array_shape": list(scratch["shape"]),
            "source_array_dtype": str(scratch["dtype"]),
            "source_cache_binding_hash": str(scratch["cache_binding_hash"]),
            "source_offsets": dict(scratch["offsets"]),
            "source_row_ids_by_center": dict(scratch["row_ids_by_center"]),
            "source_case_ids_by_center": dict(scratch["case_ids_by_center"]),
            "source_row_identity_hash": canonical_hash(
                [
                    value
                    for center in CENTERS
                    for value in scratch["row_ids_by_center"][center]
                ]
            ),
            "actions": actions,
            "classifier": classifier_payload,
            "threads_per_fit": int(config.runtime["threads_per_worker"]),
            "prediction_batch_rows": PREDICTION_BATCH_ROWS,
            "labels_available": False,
            "test_cache_admitted": False,
            "target_expert_available": False,
        }
        task_hash = canonical_hash(unhashed)
        tasks.append(
            {
                **unhashed,
                "task_hash": task_hash,
                "checkpoint_json_path": str(task_root / f"{task_id}.json"),
                "checkpoint_npz_path": str(task_root / f"{task_id}.npz"),
            }
        )
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise ProtocolError("Prediction-only source task coverage drifted.")
    return tuple(tasks)


def build_test_tasks(
    config: ActionPredictionConfig,
    classifier_bank: ActionClassifierBank,
    *,
    scratch: Mapping[str, object],
    root: Path,
    source_prediction_seal_hash: str,
    regret_model_bank_seal_hash: str,
) -> tuple[Mapping[str, object], ...]:
    offsets = scratch.get("offsets")
    if not isinstance(offsets, Mapping):
        raise ProtocolError("Prediction-only test scratch offsets are absent.")
    task_root = root / TEST_CHECKPOINT_DIRECTORY / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)
    mean_path, scale_path, coefficient_path, intercept_path = (
        classifier_bank.parameter_paths
    )
    tasks: list[Mapping[str, object]] = []
    cell_ordinal = 0
    for target, training, generation in product(
        CENTERS, TRAINING_SEEDS, GENERATION_SEEDS
    ):
        offset = offsets[target]
        if not isinstance(offset, Mapping):
            raise ProtocolError("Prediction-only test offset is malformed.")
        actions = [action.to_payload() for action in actions_for_target(target)]
        ordinals = list(
            range(cell_ordinal, cell_ordinal + PHYSICAL_ACTION_COUNT_PER_TARGET)
        )
        cell_ordinal += PHYSICAL_ACTION_COUNT_PER_TARGET
        task_id = f"test_target_{target}_train_{training}_generation_{generation}"
        unhashed = {
            "schema_version": "midogpp_prediction_only_test_inference_task_v1",
            "task_id": task_id,
            "config_contract_hash": config.contract_hash,
            "classifier_bank_seal_hash": classifier_bank.seal_hash,
            "source_prediction_seal_hash": source_prediction_seal_hash,
            "regret_model_bank_seal_hash": regret_model_bank_seal_hash,
            "target_center": target,
            "training_seed": training,
            "generation_seed": generation,
            "actions": actions,
            "classifier_cell_ordinals": ordinals,
            "scaler_mean_path": str(mean_path.resolve()),
            "scaler_scale_path": str(scale_path.resolve()),
            "coefficient_path": str(coefficient_path.resolve()),
            "intercept_path": str(intercept_path.resolve()),
            "test_array_path": str(scratch["array_path"]),
            "test_array_file_sha256": str(scratch["array_file_sha256"]),
            "test_start": int(offset["start"]),
            "test_stop": int(offset["stop"]),
            "test_row_identity_hash": str(offset["row_identity_hash"]),
            "test_slice_sha256": str(offset["embedding_slice_sha256"]),
            "prediction_batch_rows": PREDICTION_BATCH_ROWS,
            "labels_available": False,
            "classifier_refit_permitted": False,
            "target_scoring_permitted": False,
        }
        task_hash = canonical_hash(unhashed)
        tasks.append(
            {
                **unhashed,
                "task_hash": task_hash,
                "checkpoint_json_path": str(task_root / f"{task_id}.json"),
                "checkpoint_npz_path": str(task_root / f"{task_id}.npz"),
            }
        )
    if len(tasks) != EXPECTED_TASK_COUNT or cell_ordinal != EXPECTED_CLASSIFIER_FIT_COUNT:
        raise ProtocolError("Prediction-only test task coverage drifted.")
    return tuple(tasks)


def validate_source_task(task: Mapping[str, object]) -> None:
    target = str(task.get("target_center", ""))
    unhashed = {
        key: value
        for key, value in task.items()
        if key not in {"task_hash", "checkpoint_json_path", "checkpoint_npz_path"}
    }
    if (
        task.get("task_hash") != canonical_hash(unhashed)
        or task.get("labels_available") is not False
        or task.get("test_cache_admitted") is not False
        or task.get("target_expert_available") is not False
        or tuple(task.get("candidate_sources", ())) != candidate_sources(target)
        or task.get("actions")
        != [action.to_payload() for action in actions_for_target(target)]
    ):
        raise ProtocolError("Prediction-only source task escaped its boundary.")


def validate_test_task(task: Mapping[str, object]) -> None:
    target = str(task.get("target_center", ""))
    unhashed = {
        key: value
        for key, value in task.items()
        if key not in {"task_hash", "checkpoint_json_path", "checkpoint_npz_path"}
    }
    if (
        task.get("task_hash") != canonical_hash(unhashed)
        or task.get("labels_available") is not False
        or task.get("classifier_refit_permitted") is not False
        or task.get("target_scoring_permitted") is not False
        or task.get("actions")
        != [action.to_payload() for action in actions_for_target(target)]
        or len(task.get("classifier_cell_ordinals", ()))
        != PHYSICAL_ACTION_COUNT_PER_TARGET
    ):
        raise ProtocolError("Prediction-only test task escaped its boundary.")


__all__ = (
    "build_source_tasks",
    "build_test_tasks",
)
