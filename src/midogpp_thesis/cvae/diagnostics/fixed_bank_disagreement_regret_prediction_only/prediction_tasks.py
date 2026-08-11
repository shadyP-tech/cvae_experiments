"""Compatibility facade for prediction-only task execution.

The implementation is intentionally partitioned by lifecycle responsibility:
frame scratch materialization, deterministic task planning, worker math, and
checkpoint/product handling.  This module preserves the original public import
surface used by the runtime and existing diagnostics.
"""

from __future__ import annotations

from .frame_scratch import row_id, validate_frame_scratch, write_frame_scratch
from .prediction_checkpoints import (
    assemble_source_products,
    assemble_test_cells,
    execute_or_resume_source,
    execute_or_resume_test,
    load_source_checkpoint,
    load_test_checkpoint,
)
from .prediction_plans import build_source_tasks, build_test_tasks
from .prediction_workers import (
    classifier_from_payload,
    compose_action,
    fit_action_classifier,
    load_source_task_arrays,
    predict_probability_batched,
    source_prediction_task,
    test_prediction_task,
)


__all__ = (
    "assemble_source_products",
    "assemble_test_cells",
    "build_source_tasks",
    "build_test_tasks",
    "classifier_from_payload",
    "compose_action",
    "execute_or_resume_source",
    "execute_or_resume_test",
    "fit_action_classifier",
    "load_source_checkpoint",
    "load_source_task_arrays",
    "load_test_checkpoint",
    "predict_probability_batched",
    "row_id",
    "source_prediction_task",
    "test_prediction_task",
    "validate_frame_scratch",
    "write_frame_scratch",
)
