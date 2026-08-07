"""Public facade for sealed residual-top-up prediction materialization.

Planning/scratch construction, four-worker checkpoint scheduling, and store
binding validation are separate modules so this public surface remains the
single stable import point used by the runner and bundle validator.
"""

from .prediction_execution import materialize_all_action_predictions
from .prediction_planning import (
    bind_task_plan_lock,
    build_prediction_tasks as _build_tasks,
    write_evaluation_scratch as _write_evaluation_scratch,
)
from .prediction_validation import validate_prediction_store_binding


__all__ = (
    "bind_task_plan_lock",
    "materialize_all_action_predictions",
    "validate_prediction_store_binding",
)
