"""Public entry points for the Stage-90 residual-top-up case-OOF diagnostic.

Scientific contracts and execution helpers remain available from their owning
submodules; the package boundary intentionally exposes only configuration and
orchestration.
"""


def load_residual_topup_case_oof_config(*args, **kwargs):
    """Load the strict experiment config without eager runtime imports."""

    from .config import load_residual_topup_case_oof_config as load

    return load(*args, **kwargs)


def run_residual_topup_case_oof_diagnostic(*args, **kwargs):
    """Launch the diagnostic while keeping package import lightweight."""

    from .runner import run_residual_topup_case_oof_diagnostic as run

    return run(*args, **kwargs)


__all__ = (
    "load_residual_topup_case_oof_config",
    "run_residual_topup_case_oof_diagnostic",
)
