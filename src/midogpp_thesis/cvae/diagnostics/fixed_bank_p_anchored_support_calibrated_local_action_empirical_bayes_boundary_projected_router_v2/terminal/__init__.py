"""Public aggregate-only terminal scoring surface for SCALE-BP v2."""

from .contracts import (
    CenterMetrics,
    TerminalAggregate,
    TerminalComparison,
    TerminalMetrics,
)
from .scoring import (
    persist_terminal_aggregate,
    sealed_probability_hash,
    validate_persisted_terminal_aggregate,
)


__all__ = (
    "CenterMetrics",
    "TerminalAggregate",
    "TerminalComparison",
    "TerminalMetrics",
    "persist_terminal_aggregate",
    "sealed_probability_hash",
    "validate_persisted_terminal_aggregate",
)
