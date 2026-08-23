"""Fail-before-write execution admission for the planned P-DCAPS router."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from .config import PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterConfig
from .identity import EXPERIMENT_ID


BLOCKED_MESSAGE = (
    "P-DCAPS execution is not authorized; implementation and planned "
    "registration do not authorize reuse of consumed-test labels."
)


def assert_execution_authorized(
    config: PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterConfig,
    *,
    artifact_root: str | Path | None = None,
    scratch_root: str | Path | None = None,
) -> None:
    """Reject the frozen planned identity before filesystem mutation.

    ``artifact_root`` and ``scratch_root`` are accepted only so a runner can
    make this its first call.  They are deliberately not resolved, inspected,
    created, or written here.
    """

    del artifact_root, scratch_root
    if config.experiment_id != EXPERIMENT_ID:
        raise ProtocolError("P-DCAPS execution identity drifted.")
    gates = (
        config.execution_authorized,
        config.protocol.get("execution_authorized"),
        config.runtime.get("execution_authorized"),
        config.claim_boundary.get("execution_authorized"),
    )
    if gates != (False, False, False, False):
        raise ProtocolError("P-DCAPS frozen non-authorizing registration drifted.")
    raise ProtocolError(BLOCKED_MESSAGE)


__all__ = ("BLOCKED_MESSAGE", "assert_execution_authorized")
