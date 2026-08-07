"""Terminal consumed-validation residual top-up router diagnostic."""

from .config import ResidualTopupDiagnosticConfig, load_residual_topup_config


def run_residual_topup_router_diagnostic(*args, **kwargs):
    """Import the heavy runner only when the experiment is launched."""

    from .runner import run_residual_topup_router_diagnostic as run

    return run(*args, **kwargs)

__all__ = (
    "ResidualTopupDiagnosticConfig",
    "load_residual_topup_config",
    "run_residual_topup_router_diagnostic",
)
