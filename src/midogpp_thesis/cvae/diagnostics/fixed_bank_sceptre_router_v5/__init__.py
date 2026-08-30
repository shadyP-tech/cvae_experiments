"""SCEPTRE v5 candidate-set routing diagnostic.

The package is intentionally separate from every sealed SCEPTRE v1-v4
executable and artifact lineage.  It implements the transition

``label-free ranked set -> support-selected member -> same member or exact B``.

It is a post-hoc consumed-test sensitivity only.  Importing this package does
not authorize opening target labels or executing a workstation run.
"""

from .identity import (
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)

__all__ = (
    "EXPERIMENT_ID",
    "OUTPUT_ARTIFACT_ID",
    "PUBLICATION_STATUS",
    "TERMINAL_DECISION",
)
