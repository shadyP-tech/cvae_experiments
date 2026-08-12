"""Public facade for the modular flip-router science phases.

The phase implementations are intentionally split by responsibility; this
module preserves the original import surface used by the runner and replay
validator.
"""

from .science_contracts import DecisionPhaseResult, DonorPhaseResult
from .science_decisions import build_fold_decision_phase
from .science_donor import fit_h_specific_donor_phase
from .science_terminal import evaluate_terminal_phase

__all__ = (
    "DecisionPhaseResult",
    "DonorPhaseResult",
    "build_fold_decision_phase",
    "evaluate_terminal_phase",
    "fit_h_specific_donor_phase",
)

