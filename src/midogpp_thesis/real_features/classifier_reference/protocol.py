"""Protocol exception shared by the real-feature classifier surface."""

from __future__ import annotations


class ProtocolError(ValueError):
    """Raised before a real-feature run could cross a locked protocol boundary."""
