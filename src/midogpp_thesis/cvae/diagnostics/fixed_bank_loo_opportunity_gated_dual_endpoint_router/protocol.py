"""Fail-closed science protocol for the dual-endpoint diagnostic."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .config_payloads import (
    canonical_action_library_payload,
    canonical_claim_boundary_payload,
    canonical_controls_payload,
    canonical_evaluation_payload,
    canonical_identification_endpoint_payload,
    canonical_portfolio_payload,
    canonical_protocol_payload,
    canonical_robust_endpoint_payload,
)
from .experiment_contracts import CLAIM_ROLE, PUBLICATION_STATUS, TERMINAL_DECISION


def frozen_science_protocol_payload() -> dict[str, object]:
    """Return every label, routing, probability, and claim rule in one seal."""

    return {
        "schema_version": "fixed_bank_ogde_frozen_science_protocol_v1",
        "protocol": canonical_protocol_payload(),
        "action_library": canonical_action_library_payload(),
        "identification_endpoint": canonical_identification_endpoint_payload(),
        "robust_endpoint": canonical_robust_endpoint_payload(),
        "portfolio": canonical_portfolio_payload(),
        "controls": canonical_controls_payload(),
        "evaluation": canonical_evaluation_payload(),
        "claim_boundary": canonical_claim_boundary_payload(),
    }


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FrozenScienceProtocol:
    payload: dict[str, object] = field(
        default_factory=frozen_science_protocol_payload
    )
    protocol_hash: str = ""

    def __post_init__(self) -> None:
        canonical = frozen_science_protocol_payload()
        if self.payload != canonical:
            raise ProtocolError("Dual-endpoint frozen science protocol drifted.")
        expected = _canonical_hash(canonical)
        if self.protocol_hash and self.protocol_hash != expected:
            raise ProtocolError("Dual-endpoint science protocol hash drifted.")
        object.__setattr__(self, "payload", canonical)
        object.__setattr__(self, "protocol_hash", expected)

    @property
    def contract_hash(self) -> str:
        return self.protocol_hash

    def to_payload(self) -> dict[str, object]:
        return {
            **self.payload,
            "claim_role": CLAIM_ROLE,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "may_authorize_routing": False,
            "may_authorize_downstream_utility": False,
            "may_authorize_action_selection": False,
            "may_authorize_policy_update": False,
            "may_authorize_promotion": False,
            "may_authorize_deployment": False,
            "may_feed_another_experiment": False,
            "protocol_hash": self.protocol_hash,
        }


# Naming aliases retained for the thin runner and independent tests.
OpportunityGatedDualEndpointProtocol = FrozenScienceProtocol


def build_frozen_science_protocol() -> FrozenScienceProtocol:
    return FrozenScienceProtocol()


def canonical_consumed_test_protocol() -> FrozenScienceProtocol:
    return build_frozen_science_protocol()


def assert_terminal_consumed_test_protocol(
    protocol: FrozenScienceProtocol,
) -> None:
    if protocol.to_payload() != build_frozen_science_protocol().to_payload():
        raise ProtocolError("Dual-endpoint protocol escaped its terminal boundary.")


__all__ = (
    "FrozenScienceProtocol",
    "OpportunityGatedDualEndpointProtocol",
    "assert_terminal_consumed_test_protocol",
    "build_frozen_science_protocol",
    "canonical_consumed_test_protocol",
    "frozen_science_protocol_payload",
)
