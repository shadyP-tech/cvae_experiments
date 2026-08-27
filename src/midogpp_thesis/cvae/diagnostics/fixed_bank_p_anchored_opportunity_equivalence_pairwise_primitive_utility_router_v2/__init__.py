"""Executable-successor contracts for OE-PPUR v2.

The checked-in state remains planned and non-authorized.  Importing this
package does not issue an amendment, claim a lease, open labels, or launch a
worker.
"""

from .config import (
    AUTHORIZATION_READY_STATE,
    PLANNED_STATE,
    ResolvedConfigBundle,
    RouterConfig,
    RouterV2Config,
    build_authorization_ready_config,
    build_planned_config,
    load_config,
    load_resolved_config,
)
from .execution_admission import (
    SixInputAdmissionReceipt,
    admit_execution,
    admit_six_input_execution,
    assert_execution_authorized,
)
from .identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
)
from .protocol import frozen_protocol_payload
from .runner import inspect_planned_router, run_oe_ppur_v2
from .workspace_inputs import WorkspaceInputBinding


__all__ = (
    "AUTHORIZATION_READY_STATE",
    "DIRECT_INPUT_ARTIFACT_IDS",
    "DIRECT_INPUT_ROLES",
    "EXPERIMENT_ID",
    "OUTPUT_ARTIFACT_ID",
    "PLANNED_STATE",
    "RouterConfig",
    "RouterV2Config",
    "ResolvedConfigBundle",
    "SixInputAdmissionReceipt",
    "WorkspaceInputBinding",
    "admit_execution",
    "admit_six_input_execution",
    "assert_execution_authorized",
    "build_authorization_ready_config",
    "build_planned_config",
    "frozen_protocol_payload",
    "load_config",
    "load_resolved_config",
    "inspect_planned_router",
    "run_oe_ppur_v2",
)
