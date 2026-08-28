"""Modular scientific core for OE-PPUR v3."""

from .admission import (
    ExactPFallbackReceipt,
    HeldLSourceOrderingCase,
    SourceOrderingAdmissionReceipt,
)
from .outer_orchestration import (
    OuterScienceResult,
    fit_outer_source_science,
    fit_outer_source_science_fail_closed,
    fit_outer_source_science_from_surface,
    fit_outer_source_science_from_surface_fail_closed,
)
from .target_decision import (
    OuterTargetDecisionInput,
    TargetCaseDecision,
    TargetDecisionLedger,
    TargetRowBinding,
    assemble_exact_218_case_decisions,
)

__all__ = (
    "ExactPFallbackReceipt",
    "HeldLSourceOrderingCase",
    "OuterScienceResult",
    "OuterTargetDecisionInput",
    "SourceOrderingAdmissionReceipt",
    "TargetCaseDecision",
    "TargetDecisionLedger",
    "TargetRowBinding",
    "assemble_exact_218_case_decisions",
    "fit_outer_source_science",
    "fit_outer_source_science_fail_closed",
    "fit_outer_source_science_from_surface",
    "fit_outer_source_science_from_surface_fail_closed",
)
