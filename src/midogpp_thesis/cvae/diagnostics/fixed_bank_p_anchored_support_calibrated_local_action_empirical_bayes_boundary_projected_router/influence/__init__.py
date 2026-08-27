"""Sample-level influence estimation for SCALE-BP."""

from .contracts import ActionDescriptor, ActionMetricVector, MetricStandardError
from .descriptors import ACTION_FEATURE_NAMES, build_action_descriptor, descriptor_matrix
from .metrics import (
    aggregate_sample_influences,
    expected_action_metrics,
    realized_action_metrics,
    sample_metric_influences,
)

__all__ = (
    "ACTION_FEATURE_NAMES",
    "ActionDescriptor",
    "ActionMetricVector",
    "MetricStandardError",
    "aggregate_sample_influences",
    "build_action_descriptor",
    "descriptor_matrix",
    "expected_action_metrics",
    "realized_action_metrics",
    "sample_metric_influences",
)
