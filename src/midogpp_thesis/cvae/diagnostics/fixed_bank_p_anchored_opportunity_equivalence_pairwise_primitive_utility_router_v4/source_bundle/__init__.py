"""Read-only v4 adapter for the immutable v3 source content lineage."""

# During import, contracts owns DTOs and parsing owns physical validation.
from .constants import *  # noqa: F401,F403
from .contracts import *  # noqa: F401,F403
from .parsing import *  # noqa: F401,F403

__all__ = tuple(name for name in globals() if not name.startswith("_"))
