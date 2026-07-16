"""Additive, non-consumable Stage-20 source-inner v2 study contracts."""

from .config import (
    FISHER_SHRINKAGE_STUDY_NAME,
    LEARNED_PRIOR_STUDY_NAME,
    LearnedConditionalPriorStudyConfig,
    SourceInnerStudyConfig,
    TaskFisherShrinkageStudyConfig,
    decision_contract_hash,
    decision_contract_payload,
    load_fisher_study_config,
    load_prior_study_config,
    load_source_inner_study_config,
    study_contract_hash,
    study_contract_payload,
)
from .contracts import (
    FISHER_ALPHAS,
    FISHER_SHRINKAGE_MODE,
    LEARNED_PRIOR_MODE,
    PRIOR_ARMS,
    SOURCE_INNER_STUDY_VERSION,
    FisherStudyDecisionV2,
    FisherStudyMetricV2,
    PriorStudyDecisionV2,
    PriorStudyMetricV2,
    StudyTrainingKey,
    StudyTrainingVariant,
)
from .fisher_decision import select_fisher_study_decision
from .prior_decision import select_prior_study_decision

__all__ = [
    "FISHER_ALPHAS",
    "FISHER_SHRINKAGE_MODE",
    "FISHER_SHRINKAGE_STUDY_NAME",
    "LEARNED_PRIOR_MODE",
    "LEARNED_PRIOR_STUDY_NAME",
    "PRIOR_ARMS",
    "SOURCE_INNER_STUDY_VERSION",
    "FisherStudyDecisionV2",
    "FisherStudyMetricV2",
    "LearnedConditionalPriorStudyConfig",
    "PriorStudyDecisionV2",
    "PriorStudyMetricV2",
    "SourceInnerStudyConfig",
    "StudyTrainingKey",
    "StudyTrainingVariant",
    "TaskFisherShrinkageStudyConfig",
    "decision_contract_hash",
    "decision_contract_payload",
    "load_fisher_study_config",
    "load_prior_study_config",
    "load_source_inner_study_config",
    "select_fisher_study_decision",
    "select_prior_study_decision",
    "study_contract_hash",
    "study_contract_payload",
]
