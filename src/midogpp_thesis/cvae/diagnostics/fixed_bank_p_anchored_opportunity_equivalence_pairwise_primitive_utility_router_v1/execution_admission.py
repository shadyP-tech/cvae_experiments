"""Read-only source preflight and unconditional planned-run rejection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...protocol import ProtocolError
from .config import frozen_config_contract_payload
from .contracts import claim_boundary_payload, direct_input_policy_payload
from .execution.workstation import workstation_payload
from .hashing import canonical_hash
from .identity import EXPERIMENT_ID, INPUT_ARTIFACT_IDS, OUTPUT_ARTIFACT_ID
from .protocol import frozen_protocol_payload
from .source_fence import (
    SourceFenceReceipt,
    validate_source_fence,
    validate_source_fence_receipt,
)


BLOCKED_MESSAGE = (
    "OE-PPUR v1 execution is not authorized; the planned diagnostic has no "
    "test-cache, label, consumption-ledger, or single-use execution capability."
)


def run_read_only_source_preflight(
    expected: Mapping[str, object] | None = None,
) -> SourceFenceReceipt:
    """Recompute both source scopes and optionally match the frozen pins."""

    pins = {} if expected is None else dict(expected)
    receipt = validate_source_fence(
        expected_adapter_tree_sha256=pins.get("adapter_tree_sha256"),
        expected_core_tree_sha256=pins.get("core_tree_sha256"),
        expected_combined_source_seal_hash=pins.get(
            "combined_source_seal_hash"
        ),
    )
    validate_source_fence_receipt(receipt)
    if expected is not None:
        observed = {
            "schema_version": "oe_ppur_v1_source_provenance_config_v1",
            "adapter_scope_role": receipt.adapter.role,
            "adapter_member_count": receipt.adapter.member_count,
            "adapter_tree_sha256": receipt.adapter_tree_sha256,
            "adapter_receipt_hash": receipt.adapter.receipt_hash,
            "core_scope_role": receipt.core.role,
            "core_member_count": receipt.core.member_count,
            "core_tree_sha256": receipt.core_tree_sha256,
            "core_receipt_hash": receipt.core.receipt_hash,
            "source_scopes_are_disjoint": True,
            "combined_member_count": receipt.member_count,
            "combined_source_seal_hash": receipt.combined_source_seal_hash,
            "combined_source_receipt_hash": receipt.receipt_hash,
            "recompute_and_exact_match_on_load": True,
        }
        if pins != observed:
            raise ProtocolError("OE-PPUR persisted combined source provenance drifted.")
    return receipt


def assert_execution_authorized(
    config: object,
    *,
    artifact_root: object | None = None,
    scratch_root: object | None = None,
) -> None:
    """Validate source, then reject without coercing or touching run paths."""

    validate_planned_execution_contract(config)
    # These values may be poison objects.  Never call str/os.fspath/Path on them.
    del artifact_root, scratch_root
    raise ProtocolError(BLOCKED_MESSAGE)


def validate_planned_execution_contract(config: object) -> SourceFenceReceipt:
    """Validate every path-free v1 gate without claiming run authority.

    This is the safe inspection seam used by the runner blueprint.  It has no
    artifact/scratch arguments and therefore cannot accidentally acquire a
    filesystem capability.
    """

    source_provenance = _field(config, "source_provenance")
    if not isinstance(source_provenance, Mapping):
        raise ProtocolError("OE-PPUR source-provenance gate topology drifted.")
    source_receipt = run_read_only_source_preflight(source_provenance)
    if (
        source_receipt.adapter.role != "diagnostic_adapter"
        or source_receipt.core.role != "neutral_scientific_core"
        or source_receipt.adapter_member_count <= 0
        or source_receipt.core_member_count <= 0
        or source_receipt.adapter_tree_sha256
        == source_receipt.core_tree_sha256
    ):
        raise ProtocolError("OE-PPUR combined source provenance drifted.")
    if _field(config, "experiment_id") != EXPERIMENT_ID:
        raise ProtocolError("OE-PPUR execution identity drifted.")
    if _field(config, "output_artifact_id") != OUTPUT_ARTIFACT_ID:
        raise ProtocolError("OE-PPUR output identity drifted.")
    if tuple(_field(config, "input_artifact_ids", default=())) != INPUT_ARTIFACT_IDS:
        raise ProtocolError("OE-PPUR three-input planning identity drifted.")
    expected_hash = canonical_hash(frozen_config_contract_payload())
    if _field(config, "contract_hash") != expected_hash:
        raise ProtocolError("OE-PPUR config contract hash drifted.")
    protocol = _field(config, "protocol")
    runtime = _field(config, "runtime")
    claims = _field(config, "claim_boundary")
    inputs = _field(config, "inputs")
    if not all(
        isinstance(row, Mapping)
        for row in (protocol, runtime, claims, inputs, source_provenance)
    ):
        raise ProtocolError("OE-PPUR execution-gate topology drifted.")
    if dict(protocol) != frozen_protocol_payload():
        raise ProtocolError("OE-PPUR execution protocol drifted.")
    if dict(runtime) != workstation_payload():
        raise ProtocolError("OE-PPUR workstation contract drifted.")
    if dict(claims) != claim_boundary_payload():
        raise ProtocolError("OE-PPUR claim boundary drifted.")
    if dict(inputs) != direct_input_policy_payload():
        raise ProtocolError("OE-PPUR input policy drifted.")
    gates = (
        _field(config, "execution_authorized", default=False),
        protocol.get("execution_authorized"),
        runtime.get("execution_authorized"),
        claims.get("execution_authorized"),
    )
    if gates != (False, False, False, False):
        raise ProtocolError("OE-PPUR non-authorizing gates drifted.")
    if (
        protocol.get("may_feed_another_experiment") is not False
        or protocol.get("route_policy_proxy_is_cvae_compatibility") is not False
        or protocol.get("route_policy_proxy_is_nelbo_compatibility") is not False
    ):
        raise ProtocolError("OE-PPUR terminal claim firewall drifted.")
    return source_receipt


def _field(value: object, name: str, *, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = (
    "BLOCKED_MESSAGE",
    "assert_execution_authorized",
    "run_read_only_source_preflight",
    "validate_planned_execution_contract",
)
