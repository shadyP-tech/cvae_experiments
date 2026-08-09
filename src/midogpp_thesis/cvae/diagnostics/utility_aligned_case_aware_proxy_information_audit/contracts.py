"""Compatibility facade for the modular case-aware core contracts.

New code may import cohesive leaves directly.  Existing callers retain this
stable facade, which re-exports the *same class and constant objects* rather
than defining wrappers or duplicate identities.
"""

from . import constants as _constants
from .constants import *  # noqa: F403 - deliberate compatibility re-export
from .crossfit_contracts import (
    CaseAwareCrossfitResult,
    CrossfitFoldAudit,
    CrossfitPredictionRow,
    ProxyFamilyDesign,
    ProxyFamilySpec,
)
from .feature_contracts import (
    CaseAwareFeatureSurface,
    CaseAwareProxyFeatureRow,
    SupportCaseVectors,
)
from .metric_contracts import (
    CaseAwareProxyInformationAuditResult,
    FamilySummaryRow,
    OuterMetricRow,
    QueryMetricRow,
)
from .response_contracts import CaseAwareResponseRow, CaseAwareResponseSurface


__all__ = _constants.__all__ + (
    "CaseAwareCrossfitResult",
    "CaseAwareFeatureSurface",
    "CaseAwareProxyFeatureRow",
    "CaseAwareProxyInformationAuditResult",
    "CaseAwareResponseRow",
    "CaseAwareResponseSurface",
    "CrossfitFoldAudit",
    "CrossfitPredictionRow",
    "FamilySummaryRow",
    "OuterMetricRow",
    "ProxyFamilyDesign",
    "ProxyFamilySpec",
    "QueryMetricRow",
    "SupportCaseVectors",
)
