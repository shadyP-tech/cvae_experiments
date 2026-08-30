"""Compatibility façade for the complete-artifact index contract.

The dependency-light implementation lives under :mod:`.artifact`; existing
imports remain valid through this module.
"""

from .artifact.contracts import CompleteArtifactSealReceipt
from .artifact.schema import (
    build_complete_index_payload,
    issue_complete_artifact_seal,
    validate_complete_index_schema,
)


__all__ = ("CompleteArtifactSealReceipt",)
