"""Terminal-only scoring for the authorized P-DCAPS v4 diagnostic."""

from .contracts import TerminalEvaluationResult
from .diagnostics import build_router_diagnostics, midrank_spearman
from .evaluation import evaluate_terminal
from .inference import exact_shared_center_max_sign_flip
from .scoring import score_composed_methods

__all__ = (
    "TerminalEvaluationResult",
    "build_router_diagnostics",
    "evaluate_terminal",
    "exact_shared_center_max_sign_flip",
    "midrank_spearman",
    "score_composed_methods",
)
