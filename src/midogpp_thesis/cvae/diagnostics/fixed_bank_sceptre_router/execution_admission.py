"""Mutation-free admission firewall for planned SCEPTRE v1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from midogpp_thesis.cvae.protocol import ProtocolError

from .experiment_contracts import claim_boundary_payload, direct_input_policy_payload
from .protocol import frozen_protocol_payload
from .source_fence import validate_science_source_receipt
from .workstation import workstation_payload


BLOCKED_MESSAGE = (
    "SCEPTRE v1 execution is not authorized: source-inner reuse is authorized "
    "only for adaptive descriptive development, while consumed-test cache, "
    "manifest, ledger, target-label access, output, and scratch capabilities "
    "remain absent. Use a separate future execution identity."
)


def assert_execution_authorized(
    config: object,
    *,
    artifact_root: object,
    scratch_root: object | None = None,
) -> None:
    """Validate immutable contracts, then reject without resolving run paths."""

    del artifact_root, scratch_root
    provenance = _field(config, "source_provenance", default={})
    if not isinstance(provenance, Mapping):
        raise ProtocolError("SCEPTRE source provenance is absent.")
    receipt = validate_science_source_receipt(
        provenance.get("scientific_source_tree_sha256")
    )
    if (
        provenance.get("scientific_source_receipt_hash") != receipt.receipt_hash
        or provenance.get("label_free_core_may_import_source_inner_utility") is not False
        or provenance.get("development_adapter_is_separate") is not True
    ):
        raise ProtocolError("SCEPTRE scientific source provenance drifted.")
    if _field(config, "protocol", default={}) != frozen_protocol_payload():
        raise ProtocolError("SCEPTRE protocol drifted before admission.")
    if _field(config, "runtime", default={}) != workstation_payload():
        raise ProtocolError("SCEPTRE workstation contract drifted before admission.")
    if _field(config, "claim_boundary", default={}) != claim_boundary_payload():
        raise ProtocolError("SCEPTRE claim boundary drifted before admission.")
    if _field(config, "inputs", default={}) != direct_input_policy_payload():
        raise ProtocolError("SCEPTRE direct-input policy drifted before admission.")
    gates = (
        _field(config, "execution_authorized", default=False),
        frozen_protocol_payload()["execution_authorized"],
        workstation_payload()["execution_authorized"],
        claim_boundary_payload()["execution_authorized"],
    )
    if gates != (False, False, False, False):
        raise ProtocolError("SCEPTRE non-authorizing gates drifted.")
    raise ProtocolError(BLOCKED_MESSAGE)


def _field(value: object, name: str, *, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = ("BLOCKED_MESSAGE", "assert_execution_authorized")
