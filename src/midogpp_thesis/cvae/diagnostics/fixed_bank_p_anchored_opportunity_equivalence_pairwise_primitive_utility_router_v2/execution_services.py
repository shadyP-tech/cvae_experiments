"""Dependency boundary between runner mechanics and OE-PPUR science phases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .authorization_lease import AuthorizationLeaseClaim
from .execution.probability_matrix_receipts import (
    ParsedProbabilityMatrixScienceReceipt,
)
from .execution.decision_receipts import (
    TypedPreterminalDecisionLedgerReceipt,
)
from .execution_admission import SixInputAdmissionReceipt
from .phase_contracts import (
    OuterFoldExecutionReceipt,
    ProbabilityMaterializationReceipt,
    ServicePreflightReceipt,
)
from .source_seal import SourceContractReceipt
from .terminal_capability import AggregateOnlyTerminalScorer
from .workstation import WorkstationPreflightReceipt


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    artifact_root: Path
    scratch_root: Path
    run_identity_hash: str
    admission: SixInputAdmissionReceipt
    source: SourceContractReceipt
    workstation: WorkstationPreflightReceipt
    lease: AuthorizationLeaseClaim


@runtime_checkable
class RouterExecutionServices(Protocol):
    """Exact phase API; implementations must live inside the v2 source seal."""

    def preflight(
        self,
        admission: SixInputAdmissionReceipt,
        source: SourceContractReceipt,
    ) -> ServicePreflightReceipt: ...

    def materialize_probability_matrix(
        self,
        context: ExecutionContext,
    ) -> ProbabilityMaterializationReceipt: ...

    def run_outer_folds(
        self,
        context: ExecutionContext,
        matrix: ParsedProbabilityMatrixScienceReceipt,
    ) -> OuterFoldExecutionReceipt: ...

    def seal_preterminal_decisions(
        self,
        context: ExecutionContext,
        matrix: ParsedProbabilityMatrixScienceReceipt,
        outer: OuterFoldExecutionReceipt,
    ) -> TypedPreterminalDecisionLedgerReceipt: ...

    def build_terminal_scorer(
        self,
        context: ExecutionContext,
    ) -> AggregateOnlyTerminalScorer: ...


__all__ = ("ExecutionContext", "RouterExecutionServices")
