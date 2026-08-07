from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.diagnostics.residual_topup_router import predictions
from midogpp_thesis.cvae.diagnostics.residual_topup_router import (
    prediction_execution,
    prediction_planning,
    prediction_validation,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_router.contracts import (
    CENTERS,
    DEVELOPMENT_ACTION_IDS,
    EXPECTED_DEVELOPMENT_TASK_COUNT,
    EXPECTED_PREDICTION_CELL_COUNT,
    EXPECTED_TARGET_TASK_COUNT,
    TARGET_ACTION_IDS,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_router.prediction_worker import (
    load_prediction_checkpoint,
    write_prediction_checkpoint,
)
from midogpp_thesis.cvae.protocol import ProtocolError


class _Plans:
    lock_hash = "plan-lock"

    def plan(
        self, *, phase: str, outer_target: str, query_center: str, action_id: str
    ) -> dict[str, object]:
        return {
            "phase": phase,
            "outer_target": outer_target,
            "query_center": query_center,
            "action_id": action_id,
            "plan_hash": f"plan-{phase}-{outer_target}-{query_center}-{action_id}",
            "action_payload": {"action_hash": f"action-{action_id}"},
        }


def test_prediction_facade_preserves_public_api_across_cohesive_modules() -> None:
    """The coordinator stays a stable import boundary as internals move."""

    assert predictions.materialize_all_action_predictions is (
        prediction_execution.materialize_all_action_predictions
    )
    assert predictions.validate_prediction_store_binding is (
        prediction_validation.validate_prediction_store_binding
    )
    assert predictions.bind_task_plan_lock is prediction_planning.bind_task_plan_lock
    assert predictions._build_tasks is prediction_planning.build_prediction_tasks
    assert predictions._write_evaluation_scratch is (
        prediction_planning.write_evaluation_scratch
    )


def test_prediction_scheduler_builds_exact_workstation_task_surface(
    tmp_path: Path,
) -> None:
    scratch = {
        "evaluation_scratch_hash": "scratch-hash",
        "centers": {
            center: {
                "start": ordinal,
                "stop": ordinal + 1,
                "sample_ids": [f"sample-{center}"],
                "row_identity_hash": f"rows-{center}",
            }
            for ordinal, center in enumerate(CENTERS)
        },
    }
    config = SimpleNamespace(
        contract_hash="config-hash",
        classifier=SimpleNamespace(to_payload=lambda: {"family": "test"}),
    )
    source_cache = SimpleNamespace(
        array_path=tmp_path / "source.npy",
        index_rows=(),
    )
    tasks = predictions._build_tasks(
        config,
        "generation-lock",
        source_cache,
        _Plans(),
        object(),
        source_cache_lock_hash="source-lock",
        scratch=scratch,
        scratch_path=tmp_path / "evaluation.npy",
        checkpoint_root=tmp_path / "checkpoints",
    )

    assert len(tasks) == EXPECTED_DEVELOPMENT_TASK_COUNT + EXPECTED_TARGET_TASK_COUNT
    development = [task for task in tasks if task["phase"] == "development"]
    target = [task for task in tasks if task["phase"] == "target"]
    assert len(development) == EXPECTED_DEVELOPMENT_TASK_COUNT
    assert len(target) == EXPECTED_TARGET_TASK_COUNT
    assert all(len(task["plans"]) == len(DEVELOPMENT_ACTION_IDS) for task in development)
    assert all(len(task["plans"]) == len(TARGET_ACTION_IDS) for task in target)
    assert sum(len(task["plans"]) for task in tasks) == EXPECTED_PREDICTION_CELL_COUNT
    assert all(
        task["outer_target"] not in task["candidate_sources"]
        and task["query_center"] not in task["candidate_sources"]
        and len(task["candidate_sources"]) == 7
        for task in development
    )
    assert all(
        task["query_center"] not in task["candidate_sources"]
        and len(task["candidate_sources"]) == 8
        for task in target
    )
    assert all(task["threads_per_fit"] == 3 for task in tasks)
    assert all(
        "labels" not in task
        and "evaluation_labels" not in task
        and "support_labels" not in task
        for task in tasks
    )
    assert all(task["router_plan_lock_hash"] == "plan-lock" for task in tasks)
    for task in tasks:
        expected_hash = stable_hash(
            {
                key: value
                for key, value in task.items()
                if key
                not in {
                    "checkpoint_json_path",
                    "checkpoint_npz_path",
                    "task_hash",
                }
            }
        )
        assert task["task_hash"] == expected_hash


def test_prediction_checkpoint_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    json_path = tmp_path / "task.json"
    npz_path = tmp_path / "task.npz"
    task = {
        "task_hash": "task-hash",
        "task_id": "task-id",
        "plans": ({"action_id": "uniform_topup"}, {"action_id": "energy_directed_topup"}),
    }
    result = {
        "task_hash": "task-hash",
        "task_id": "task-id",
        "unique_classifier_fit_count": 2,
        "cells": (
            {
                "action_id": "uniform_topup",
                "predictions": np.asarray([0, 1], dtype=np.uint8),
                "probabilities": np.asarray([0.2, 0.8], dtype=np.float32),
                "metadata": {"role": "control"},
            },
            {
                "action_id": "energy_directed_topup",
                "predictions": np.asarray([1, 1], dtype=np.uint8),
                "probabilities": np.asarray([0.6, 0.9], dtype=np.float32),
                "metadata": {"role": "routed"},
            },
        ),
    }
    write_prediction_checkpoint(json_path, npz_path, result)
    loaded = load_prediction_checkpoint(json_path, npz_path, task=task)
    assert loaded["unique_classifier_fit_count"] == 2
    assert [cell["action_id"] for cell in loaded["cells"]] == [
        "uniform_topup",
        "energy_directed_topup",
    ]

    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]).copy() for key in payload.files}
    arrays["cell_0_predictions"][0] = 1
    np.savez_compressed(npz_path, **arrays)
    with pytest.raises(ProtocolError, match="checkpoint"):
        load_prediction_checkpoint(json_path, npz_path, task=task)
