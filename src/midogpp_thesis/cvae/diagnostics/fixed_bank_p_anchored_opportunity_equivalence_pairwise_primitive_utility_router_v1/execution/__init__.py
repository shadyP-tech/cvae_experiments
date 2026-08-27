"""Non-authorizing execution contracts for the planned OE-PPUR v1.

Pool launch functions intentionally remain internal to ``execution.pools``.
The v1 production path always rejects them; only source-sealed synthetic
pytest callbacks can exercise spawn transport. Importing this package cannot
mint a run capability or start a process.
"""

from .batch_receipts import ExecutionBatchResult
from .dtos import SealedCallbackDescriptorDTO, WorkerExecutionDTO
from .input_lineage import (
    FrozenGenerationLockReceipt,
    PlannedConfigProtocolReceipt,
    PreterminalInputLineage,
    PromotedBankValidationReceipt,
    build_frozen_generation_lock_receipt,
    build_planned_config_protocol_receipt,
    build_preterminal_input_lineage,
    validate_promoted_bank_input,
)
from .memmap import (
    CanonicalRowAlignmentReceipt,
    CanonicalRowIdentity,
    EXPECTED_EXECUTABLE_TEST_CACHE_CONTENT_SHA256,
    EXPECTED_EXECUTABLE_TEST_CACHE_ROW_ORDER_SHA256,
    ImmutableRowIndexReceipt,
    LoadedReadOnlyFloat32Memmap,
    MemmapValidationReceipt,
    build_canonical_row_alignment_receipt,
    load_read_only_float32_memmap,
    validate_canonical_row_alignment_receipt,
    validate_row_index_receipt,
)
from .preterminal import (
    LabelFreePreterminalInputs,
    PreterminalExecutionTelemetry,
    SealedLabelFreePreterminalResult,
    seal_label_free_preterminal_result,
)
from .surfaces import (
    CandidateProbabilitySurfaceReceipt,
    build_candidate_probability_surface_receipt,
    validate_candidate_probability_surface_receipt,
)
from .workstation import (
    NativeThreadpoolRecord,
    ThreadpoolEvidence,
    WorkstationPlan,
    build_workstation_plan,
    capture_threadpool_evidence,
)

__all__ = (
    "CanonicalRowAlignmentReceipt",
    "CanonicalRowIdentity",
    "CandidateProbabilitySurfaceReceipt",
    "EXPECTED_EXECUTABLE_TEST_CACHE_CONTENT_SHA256",
    "EXPECTED_EXECUTABLE_TEST_CACHE_ROW_ORDER_SHA256",
    "ExecutionBatchResult",
    "FrozenGenerationLockReceipt",
    "ImmutableRowIndexReceipt",
    "LabelFreePreterminalInputs",
    "LoadedReadOnlyFloat32Memmap",
    "MemmapValidationReceipt",
    "NativeThreadpoolRecord",
    "PlannedConfigProtocolReceipt",
    "PreterminalInputLineage",
    "PreterminalExecutionTelemetry",
    "PromotedBankValidationReceipt",
    "SealedCallbackDescriptorDTO",
    "SealedLabelFreePreterminalResult",
    "ThreadpoolEvidence",
    "WorkerExecutionDTO",
    "WorkstationPlan",
    "build_canonical_row_alignment_receipt",
    "build_candidate_probability_surface_receipt",
    "build_frozen_generation_lock_receipt",
    "build_planned_config_protocol_receipt",
    "build_preterminal_input_lineage",
    "build_workstation_plan",
    "capture_threadpool_evidence",
    "load_read_only_float32_memmap",
    "seal_label_free_preterminal_result",
    "validate_canonical_row_alignment_receipt",
    "validate_candidate_probability_surface_receipt",
    "validate_promoted_bank_input",
    "validate_row_index_receipt",
)
