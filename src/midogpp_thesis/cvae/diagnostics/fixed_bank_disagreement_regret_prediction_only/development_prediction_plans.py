"""Physical task planning for strict source-OOF H/q predictions."""

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.frozen_source_streams import FrozenSourceStreamCache
from .constants import CENTERS, GENERATION_SEEDS, PREDICTION_BATCH_ROWS, TRAINING_SEEDS
from .development_actions import (
    DEVELOPMENT_ACTION_COUNT_PER_TASK,
    DEVELOPMENT_PHYSICAL_TASK_COUNT,
    development_actions_for,
    development_candidate_sources,
)
from .development_prediction_contracts import (
    DEVELOPMENT_CHECKPOINT_DIRECTORY,
    DevelopmentPredictionConfig,
)
from .hashing import canonical_hash


def build_development_source_tasks(
    config: DevelopmentPredictionConfig,
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
    raw_offsets = scratch.get("offsets")
    if not isinstance(raw_offsets, Mapping):
        raise ProtocolError("Strict source-OOF scratch offsets are absent.")
    task_root = root / DEVELOPMENT_CHECKPOINT_DIRECTORY / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)
    tasks: list[Mapping[str, object]] = []
    for (left, right), training, generation in product(
        combinations(CENTERS, 2), TRAINING_SEEDS, GENERATION_SEEDS
    ):
        pair = (left, right)
        actions = development_actions_for(left, right)
        reverse_actions = development_actions_for(right, left)
        if any(
            first.action_hash != second.action_hash
            for first, second in zip(actions, reverse_actions, strict=True)
        ):
            raise ProtocolError("Strict source-OOF unordered-pair symmetry drifted.")
        evaluation_views: list[dict[str, object]] = []
        for target, query, oriented_actions in (
            (left, right, actions),
            (right, left, reverse_actions),
        ):
            raw_offset = raw_offsets.get(query)
            if not isinstance(raw_offset, Mapping):
                raise ProtocolError("Strict source-OOF query offset is malformed.")
            evaluation_views.append(
                {
                    "outer_target": target,
                    "query_center": query,
                    "start": int(raw_offset["start"]),
                    "stop": int(raw_offset["stop"]),
                    "row_count": int(raw_offset["row_count"]),
                    "row_identity_hash": str(raw_offset["row_identity_hash"]),
                    "embedding_slice_sha256": str(
                        raw_offset["embedding_slice_sha256"]
                    ),
                    "orientation_hashes": [
                        action.orientation_hash for action in oriented_actions
                    ],
                }
            )
        task_id = (
            f"source_oof_excluded_{left}_{right}_train_{training}_generation_{generation}"
        )
        unhashed = {
            "schema_version": "midogpp_strict_source_oof_physical_fit_task_v1",
            "task_id": task_id,
            "config_contract_hash": config.contract_hash,
            "source_stream_lock_hash": generated_sources.lock_hash,
            "action_library_hash": action_library_hash,
            "excluded_pair": list(pair),
            "training_seed": training,
            "generation_seed": generation,
            "candidate_sources": list(development_candidate_sources(left, right)),
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
            "evaluation_views": evaluation_views,
            "actions": [action.to_payload() for action in actions],
            "classifier": classifier_payload,
            "threads_per_fit": int(config.runtime["threads_per_worker"]),
            "prediction_batch_rows": PREDICTION_BATCH_ROWS,
            "physical_fit_count": DEVELOPMENT_ACTION_COUNT_PER_TASK,
            "logical_prediction_count": 2 * DEVELOPMENT_ACTION_COUNT_PER_TASK,
            "unordered_excluded_pair_fit_reuse": True,
            "query_excluded_from_every_composition": True,
            "outer_target_excluded_from_every_composition": True,
            "sample_weight_scope": "logistic_regression_fit_only",
            "scaler_fit_used_sample_weight": False,
            "source_labels_available": False,
            "test_cache_admitted": False,
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
    if len(tasks) != DEVELOPMENT_PHYSICAL_TASK_COUNT:
        raise ProtocolError("Strict source-OOF physical task coverage drifted.")
    return tuple(tasks)


def validate_development_source_task(task: Mapping[str, object]) -> None:
    unhashed = {
        key: value
        for key, value in task.items()
        if key not in {"task_hash", "checkpoint_json_path", "checkpoint_npz_path"}
    }
    raw_pair = task.get("excluded_pair")
    if not isinstance(raw_pair, list) or len(raw_pair) != 2:
        raise ProtocolError("Strict source-OOF excluded pair is malformed.")
    left, right = tuple(str(value) for value in raw_pair)
    views = task.get("evaluation_views")
    expected_actions = [
        action.to_payload() for action in development_actions_for(left, right)
    ]
    if (
        task.get("task_hash") != canonical_hash(unhashed)
        or (left, right) != tuple(sorted((left, right)))
        or left not in CENTERS
        or right not in CENTERS
        or left == right
        or tuple(task.get("candidate_sources", ()))
        != development_candidate_sources(left, right)
        or task.get("actions") != expected_actions
        or not isinstance(views, list)
        or len(views) != 2
        or task.get("physical_fit_count") != DEVELOPMENT_ACTION_COUNT_PER_TASK
        or task.get("logical_prediction_count") != 2 * DEVELOPMENT_ACTION_COUNT_PER_TASK
        or task.get("unordered_excluded_pair_fit_reuse") is not True
        or task.get("query_excluded_from_every_composition") is not True
        or task.get("outer_target_excluded_from_every_composition") is not True
        or task.get("sample_weight_scope") != "logistic_regression_fit_only"
        or task.get("scaler_fit_used_sample_weight") is not False
        or task.get("source_labels_available") is not False
        or task.get("test_cache_admitted") is not False
    ):
        raise ProtocolError("Strict source-OOF task escaped its boundary.")
    for view, (target, query) in zip(views, ((left, right), (right, left)), strict=True):
        if not isinstance(view, Mapping):
            raise ProtocolError("Strict source-OOF evaluation view is malformed.")
        oriented = development_actions_for(target, query)
        if (
            view.get("outer_target") != target
            or view.get("query_center") != query
            or int(view.get("stop", -1)) <= int(view.get("start", -1))
            or int(view.get("row_count", -1))
            != int(view.get("stop", -1)) - int(view.get("start", -1))
            or view.get("orientation_hashes")
            != [action.orientation_hash for action in oriented]
        ):
            raise ProtocolError("Strict source-OOF oriented prediction view drifted.")


__all__ = ("build_development_source_tasks", "validate_development_source_task")
