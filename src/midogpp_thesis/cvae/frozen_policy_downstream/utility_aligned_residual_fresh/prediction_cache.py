"""Compatibility facade for modular Stage-70 prediction execution."""

from .prediction_contracts import (
    EXPECTED_PREDICTION_TASK_COUNT,
    PREDICTION_CACHE_SCHEMA,
    PREDICTION_INDEX_COLUMNS,
    PREDICTION_TASK_SCHEMA,
    PredictionCache,
    PredictionTaskExecutor,
    PredictionTaskRecord,
    PredictionTaskSpec,
)
from .prediction_planning import build_prediction_tasks
from .prediction_store import (
    load_prediction_cache,
    materialize_prediction_cache,
    write_prediction_index,
)
from .prediction_worker import execute_prediction_task


__all__ = (
    "EXPECTED_PREDICTION_TASK_COUNT",
    "PREDICTION_CACHE_SCHEMA",
    "PREDICTION_INDEX_COLUMNS",
    "PREDICTION_TASK_SCHEMA",
    "PredictionCache",
    "PredictionTaskExecutor",
    "PredictionTaskRecord",
    "PredictionTaskSpec",
    "build_prediction_tasks",
    "execute_prediction_task",
    "load_prediction_cache",
    "materialize_prediction_cache",
    "write_prediction_index",
)
