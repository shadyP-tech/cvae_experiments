"""Executable, isolated SCALE-BP v2 terminal consumed-test diagnostic."""

from .config import ScaleBPV2Config, load_config, load_scale_bp_v2_config
from .execution_admission import (
    ExecutionAdmissionReceipt,
    admit_single_use_execution,
)
from .identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    EXPECTED_CASE_COUNT,
    EXPECTED_CENTER_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPERIMENT_ID,
    FEATURE_DIM,
    GovernanceError,
    OUTPUT_ARTIFACT_ID,
)
from .label_capabilities import (
    DelegatedWorkerLabelJournal,
    LabelCapability,
    LabelCapabilityJournal,
    WorkerCapabilityAudit,
    WorkerLabelDelegation,
    WorkerSupportScope,
)
from .protocol import frozen_protocol_payload, validate_protocol_payload
from .source_fence import SourceFenceReceipt, validate_source_fence
from .workstation import (
    WorkstationPlan,
    canonical_workstation_plan,
    initialize_cpu_outer_worker,
    preflight_workstation,
)


__all__ = (
    "DIRECT_INPUT_ARTIFACT_IDS",
    "DelegatedWorkerLabelJournal",
    "EXPECTED_CASE_COUNT",
    "EXPECTED_CENTER_COUNT",
    "EXPECTED_TEST_ROW_COUNT",
    "EXPERIMENT_ID",
    "ExecutionAdmissionReceipt",
    "FEATURE_DIM",
    "GovernanceError",
    "LabelCapability",
    "LabelCapabilityJournal",
    "OUTPUT_ARTIFACT_ID",
    "ScaleBPV2Config",
    "SourceFenceReceipt",
    "WorkstationPlan",
    "WorkerCapabilityAudit",
    "WorkerLabelDelegation",
    "WorkerSupportScope",
    "admit_single_use_execution",
    "canonical_workstation_plan",
    "frozen_protocol_payload",
    "initialize_cpu_outer_worker",
    "load_config",
    "load_scale_bp_v2_config",
    "preflight_workstation",
    "validate_protocol_payload",
    "validate_source_fence",
)
