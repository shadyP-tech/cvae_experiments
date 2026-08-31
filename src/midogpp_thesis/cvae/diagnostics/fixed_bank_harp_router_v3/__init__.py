"""Fenced, terminal-only HARP v3 consumed-test diagnostic."""

from .activation import (
    ACTIVATION_CONFIRMATION,
    HarpV3ActivationPlan,
    HarpV3ActivationReceipt,
    activate_harp_v3,
    inspect_harp_v3_activation_recovery,
    plan_harp_v3_activation,
    recover_harp_v3_activation,
)
from .config import HarpStage90V3Config, load_config
from .identity import (
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .preparation import HarpV3PreparedInputs, prepare_harp_consumed_test_inputs_v3


__all__ = (
    "EXECUTION_REVISION",
    "EXPERIMENT_ID",
    "ACTIVATION_CONFIRMATION",
    "HarpStage90V3Config",
    "HarpV3ActivationPlan",
    "HarpV3ActivationReceipt",
    "HarpV3PreparedInputs",
    "OUTPUT_ARTIFACT_ID",
    "PUBLICATION_STATUS",
    "TERMINAL_DECISION",
    "activate_harp_v3",
    "inspect_harp_v3_activation_recovery",
    "load_config",
    "plan_harp_v3_activation",
    "prepare_harp_consumed_test_inputs_v3",
    "recover_harp_v3_activation",
)
