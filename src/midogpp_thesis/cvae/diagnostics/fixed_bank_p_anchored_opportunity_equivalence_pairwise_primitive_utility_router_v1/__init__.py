"""Planned OE-PPUR v1 diagnostic adapter.

The scientific kernels live in the stage-neutral routing package.  This
adapter owns only MIDOG++ identity, protocol, workstation, and fail-closed
execution boundaries.
"""

from .identity import CLI_SURFACE, EXPERIMENT_ID, OUTPUT_ARTIFACT_ID

__all__ = ("CLI_SURFACE", "EXPERIMENT_ID", "OUTPUT_ARTIFACT_ID")
