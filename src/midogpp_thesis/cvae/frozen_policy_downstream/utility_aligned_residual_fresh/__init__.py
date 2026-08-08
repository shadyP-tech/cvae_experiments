"""Fresh evaluation for the frozen utility-aligned residual-tail router.

The package facade is intentionally narrow.  Scientific contracts, execution
checkpoints and test injection seams live in their owning submodules and are
not part of the production integration API.
"""

from .bundle import validate_utility_aligned_residual_fresh_bundle
from .config import load_utility_aligned_residual_fresh_config
from .runner import run_utility_aligned_residual_fresh
from .workspace_binding import (
    validate_utility_aligned_residual_fresh_workspace_binding,
)


__all__ = (
    "load_utility_aligned_residual_fresh_config",
    "run_utility_aligned_residual_fresh",
    "validate_utility_aligned_residual_fresh_bundle",
    "validate_utility_aligned_residual_fresh_workspace_binding",
)
