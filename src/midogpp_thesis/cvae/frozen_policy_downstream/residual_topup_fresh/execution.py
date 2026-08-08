"""Stable public facade for fresh Stage-70 prediction execution.

Implementation is split by responsibility so the admission, task execution,
and checkpoint/cache protocols remain independently reviewable.  Imports from
this module retain the original public API.
"""

from .policy_loading import (
    ACTION_LIBRARY_SCHEMA,
    ACTION_SCHEMA,
    FrozenPolicySurface,
    load_frozen_policy_actions,
)
from .prediction_cache import (
    load_prediction_cache,
    materialize_prediction_cache,
    write_prediction_index,
)
from .prediction_contracts import (
    EXPECTED_PREDICTION_TASK_COUNT,
    PREDICTION_CACHE_SCHEMA,
    PREDICTION_INDEX_COLUMNS,
    PredictionCache,
    PredictionTaskExecutor,
    PredictionTaskRecord,
    PredictionTaskSpec,
)
from .prediction_tasks import execute_prediction_task


__all__ = (
    "ACTION_LIBRARY_SCHEMA",
    "ACTION_SCHEMA",
    "EXPECTED_PREDICTION_TASK_COUNT",
    "FrozenPolicySurface",
    "PREDICTION_CACHE_SCHEMA",
    "PREDICTION_INDEX_COLUMNS",
    "PredictionCache",
    "PredictionTaskExecutor",
    "PredictionTaskRecord",
    "PredictionTaskSpec",
    "execute_prediction_task",
    "load_frozen_policy_actions",
    "load_prediction_cache",
    "materialize_prediction_cache",
    "write_prediction_index",
)
