"""Fail-before-mutation admission for planned SCALE-BP v1."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...protocol import ProtocolError
from .config_payloads import claim_boundary_payload, workstation_payload
from .identity import EXPERIMENT_ID
from .protocol import frozen_protocol_payload
from .source_fence import SourceFenceReceipt, validate_source_fence


BLOCKED_MESSAGE = (
    "SCALE-BP v1 execution is not authorized; this registered implementation "
    "is planned-only and requires a distinct future executable identity."
)


def run_read_only_source_preflight() -> SourceFenceReceipt:
    """Validate the source boundary before any run-path capability exists."""

    return validate_source_fence()


def assert_execution_authorized(
    config: object,
    *,
    artifact_root: str | Path | None = None,
    scratch_root: str | Path | None = None,
) -> None:
    """Reject v1 without resolving, inspecting, locking, or writing paths."""

    run_read_only_source_preflight()
    # Path values are deliberately never resolved or inspected.
    del artifact_root, scratch_root
    if _field(config, "experiment_id") != EXPERIMENT_ID:
        raise ProtocolError("SCALE-BP execution identity drifted.")
    protocol = _field(config, "protocol")
    runtime = _field(config, "runtime")
    claims = _field(config, "claim_boundary")
    if not all(isinstance(row, Mapping) for row in (protocol, runtime, claims)):
        raise ProtocolError("SCALE-BP execution gate topology drifted.")
    if dict(protocol) != frozen_protocol_payload():
        raise ProtocolError("SCALE-BP execution protocol drifted.")
    if dict(runtime) != workstation_payload():
        raise ProtocolError("SCALE-BP workstation admission contract drifted.")
    if dict(claims) != claim_boundary_payload():
        raise ProtocolError("SCALE-BP claim admission contract drifted.")
    gates = (
        _field(config, "execution_authorized", default=False),
        protocol.get("execution_authorized"),
        runtime.get("execution_authorized"),
        claims.get("execution_authorized"),
    )
    if gates != (False, False, False, False):
        raise ProtocolError("SCALE-BP non-authorizing gate drifted.")
    if (
        protocol.get("target_terminal_labels_may_open") is not False
        or protocol.get("may_feed_another_experiment") is not False
        or protocol.get("nelbo_compatibility_claimed") is not False
    ):
        raise ProtocolError("SCALE-BP terminal claim firewall drifted.")
    raise ProtocolError(BLOCKED_MESSAGE)


def _field(value: object, name: str, *, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = (
    "BLOCKED_MESSAGE",
    "assert_execution_authorized",
    "run_read_only_source_preflight",
)
