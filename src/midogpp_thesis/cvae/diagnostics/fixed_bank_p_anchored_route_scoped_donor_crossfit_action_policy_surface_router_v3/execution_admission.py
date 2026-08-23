"""Fail-before-mutation execution admission for planned P-DCAPS v3."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...protocol import ProtocolError
from .identity import EXPERIMENT_ID, V2_EXECUTION_STATUS
from .protocol import frozen_protocol_payload
from .source_seal import CombinedSourceSeal, validate_combined_source_seal


BLOCKED_MESSAGE = (
    "P-DCAPS v3 execution is not authorized; this mechanical repair is "
    "planned-only and requires a separate future run authorization."
)


def _field(value: object, name: str, *, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _experiment_id(config: object) -> object:
    direct = _field(config, "experiment_id")
    if direct is not None:
        return direct
    experiment = _field(config, "experiment", default={})
    return _field(experiment, "id")


def run_read_only_source_preflight() -> CombinedSourceSeal:
    """Seal inherited and repaired code before any run-state capability opens."""

    return validate_combined_source_seal()


def assert_execution_authorized(
    config: object,
    *,
    artifact_root: str | Path | None = None,
    scratch_root: str | Path | None = None,
) -> None:
    """Reject this identity before resolving, inspecting, or writing paths."""

    # This source-only read is deliberately the first operation.  It neither
    # resolves run paths nor inspects locks, artifacts, scratch state, or labels.
    run_read_only_source_preflight()

    # Paths are intentionally accepted but never resolved or touched.
    del artifact_root, scratch_root
    if _experiment_id(config) != EXPERIMENT_ID:
        raise ProtocolError("P-DCAPS v3 execution identity drifted.")
    protocol = _field(config, "protocol")
    runtime = _field(config, "runtime")
    claims = _field(config, "claim_boundary")
    if not all(isinstance(row, Mapping) for row in (protocol, runtime, claims)):
        raise ProtocolError("P-DCAPS v3 execution gate topology drifted.")
    if dict(protocol) != frozen_protocol_payload():
        raise ProtocolError("P-DCAPS v3 execution protocol drifted.")
    gates = (
        _field(config, "execution_authorized", default=False),
        protocol.get("execution_authorized"),
        runtime.get("execution_authorized"),
        claims.get("execution_authorized"),
    )
    if gates != (False, False, False, False):
        raise ProtocolError("P-DCAPS v3 non-authorizing gate drifted.")
    if (
        protocol.get("v2_execution_status") != V2_EXECUTION_STATUS
        or protocol.get("v2_authorization_exhausted") is not True
        or protocol.get("v2_retry_forbidden") is not True
        or protocol.get("promotion_allowed") is not False
    ):
        raise ProtocolError("P-DCAPS v3 predecessor failure fence drifted.")
    raise ProtocolError(BLOCKED_MESSAGE)


__all__ = (
    "BLOCKED_MESSAGE",
    "assert_execution_authorized",
    "run_read_only_source_preflight",
)
