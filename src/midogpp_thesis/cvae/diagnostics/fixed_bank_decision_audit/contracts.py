"""Stable facade for the modular fixed-bank scientific contracts."""

from . import constants as _constants
from .constants import *  # noqa: F403 - deliberate constant facade
from .metric_contracts import (
    AbstentionDecisionRow,
    AbstentionSummaryRow,
    FamilySummaryRow,
    FixedBankDecisionAuditResult,
    OuterMetricRow,
    QueryMetricRow,
)
from .model_contracts import (
    CrossfitFoldAudit,
    CrossfitPredictionRow,
    ExactCrossfitResult,
    FamilyDesign,
    FamilySpec,
    SmoothCrossfitResult,
    family_spec,
)
from .row_contracts import (
    FixedBankDataset,
    FixedBankFeatureRow,
    FixedBankResponseRow,
    feature_row_from_payload,
    response_row_from_payload,
)


__all__ = _constants.__all__ + (
    "AbstentionDecisionRow",
    "AbstentionSummaryRow",
    "CrossfitFoldAudit",
    "CrossfitPredictionRow",
    "ExactCrossfitResult",
    "FamilyDesign",
    "FamilySpec",
    "FamilySummaryRow",
    "FixedBankDataset",
    "FixedBankDecisionAuditResult",
    "FixedBankFeatureRow",
    "FixedBankResponseRow",
    "OuterMetricRow",
    "QueryMetricRow",
    "SmoothCrossfitResult",
    "family_spec",
    "feature_row_from_payload",
    "response_row_from_payload",
)
