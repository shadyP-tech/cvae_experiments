"""Descriptive evaluation of frozen routing policies on consumed test rows."""

from .bootstrap import BootstrapSummary, paired_descriptive_bootstrap
from .composition import compose_policy_replicate
from .contracts import (
    CLAIM_SCOPE,
    CONTROL_ARM,
    METADATA_ARM,
    UTILITY_ARM,
    MaterializationAssignment,
    PolicyReplicate,
    PredictionCell,
    ScoringLabels,
    SyntheticComposition,
    TargetFrame,
)
from .contrasts import ArmSummary, PairedDelta, build_descriptive_contrasts
from .prediction import (
    FrozenPolicyPredictionPass,
    PersistedPredictionPass,
    run_label_free_prediction_pass,
)
from .scoring import (
    CaseConfusionRow,
    ScoredFrozenPolicies,
    TargetMetricRow,
    score_persisted_predictions,
)
from .runner import run_frozen_policy_downstream

__all__ = (
    "ArmSummary",
    "BootstrapSummary",
    "CLAIM_SCOPE",
    "CONTROL_ARM",
    "CaseConfusionRow",
    "FrozenPolicyPredictionPass",
    "METADATA_ARM",
    "MaterializationAssignment",
    "PairedDelta",
    "PersistedPredictionPass",
    "PolicyReplicate",
    "PredictionCell",
    "ScoredFrozenPolicies",
    "ScoringLabels",
    "SyntheticComposition",
    "TargetFrame",
    "TargetMetricRow",
    "UTILITY_ARM",
    "build_descriptive_contrasts",
    "compose_policy_replicate",
    "paired_descriptive_bootstrap",
    "run_label_free_prediction_pass",
    "run_frozen_policy_downstream",
    "score_persisted_predictions",
)
