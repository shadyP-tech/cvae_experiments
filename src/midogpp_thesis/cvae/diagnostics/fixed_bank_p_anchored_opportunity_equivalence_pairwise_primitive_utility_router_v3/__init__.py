"""Planned OE-PPUR v3 source-supervised executable-successor shell."""

from .config import RouterV3Config, build_planned_config
from .execution import (
    CanonicalScientificRouterService,
    PrimitiveWorkerResult,
    PrimitiveWorkerTask,
    SevenInputContractReceipt,
    build_planned_seven_input_contract,
)
from .runner import inspect_planned_router, run_oe_ppur_v3
from .source_seal import SourceSealReceipt, build_source_seal

__all__ = (
    "CanonicalScientificRouterService",
    "PrimitiveWorkerResult",
    "PrimitiveWorkerTask",
    "RouterV3Config",
    "SevenInputContractReceipt",
    "SourceSealReceipt",
    "build_planned_config",
    "build_planned_seven_input_contract",
    "build_source_seal",
    "inspect_planned_router",
    "run_oe_ppur_v3",
)
