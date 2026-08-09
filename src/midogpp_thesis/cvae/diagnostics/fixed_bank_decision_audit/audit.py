"""Pure scientific composition for the fixed-bank decision audit."""

from __future__ import annotations

from ...protocol import ProtocolError
from .crossfit import crossfit_exact_families, crossfit_smooth_descriptive
from .decision import summarize_abstention_diagnostic
from .metric_contracts import FixedBankDecisionAuditResult
from .metrics import summarize_exact_crossfit
from .row_contracts import FixedBankDataset


def run_fixed_bank_decision_core(
    dataset: FixedBankDataset,
    *,
    include_smooth_descriptive: bool = True,
) -> FixedBankDecisionAuditResult:
    """Run exact-terminal decisions, then an isolated smooth description."""

    if not isinstance(dataset, FixedBankDataset):
        raise ProtocolError("Fixed-bank audit requires a typed sealed dataset.")
    if type(include_smooth_descriptive) is not bool:
        raise ProtocolError("Smooth-description switch must be boolean.")

    # Exact fits and decisions are complete before the smooth response is read.
    exact = crossfit_exact_families(dataset)
    query_rows, outer_rows, family_rows = summarize_exact_crossfit(exact)
    abstention_rows, abstention_summaries = summarize_abstention_diagnostic(
        exact,
        family_summaries=family_rows,
    )
    smooth = (
        crossfit_smooth_descriptive(dataset)
        if include_smooth_descriptive
        else None
    )
    primary = tuple(row for row in family_rows if row.publication_gate_eligible)
    if len(primary) != 1:
        raise ProtocolError("Fixed-bank audit lost its predeclared primary R arm.")
    return FixedBankDecisionAuditResult(
        exact_crossfit=exact,
        smooth_descriptive_crossfit=smooth,
        query_metrics=query_rows,
        outer_metrics=outer_rows,
        family_summaries=family_rows,
        abstention_decisions=abstention_rows,
        abstention_summaries=abstention_summaries,
        primary_exact_gate_passed=primary[0].exact_gate_passed,
    )


__all__ = ("run_fixed_bank_decision_core",)
