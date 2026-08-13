"""Fail-closed protocol boundary for the directional-shrinkage diagnostic."""

from __future__ import annotations

from dataclasses import dataclass

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .config_payloads import canonical_protocol_payload
from .experiment_contracts import CLAIM_ROLE, PUBLICATION_STATUS, TERMINAL_DECISION


@dataclass(frozen=True)
class LooDirectionalShrinkageEnsembleProtocol:
    """Immutable protocol manifest checked independently of YAML loading."""

    payload: dict[str, object]
    contract_hash: str

    def __post_init__(self) -> None:
        expected = canonical_protocol_payload()
        if self.payload != expected or self.contract_hash != stable_hash(expected):
            raise ProtocolError(
                "Directional-shrinkage consumed-test protocol contract drifted."
            )

    def to_payload(self) -> dict[str, object]:
        return {
            **self.payload,
            "claim_role": CLAIM_ROLE,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "consumed_test_data": True,
            "fresh_evidence": False,
            "may_authorize_routing": False,
            "may_authorize_downstream_utility": False,
            "may_authorize_action_selection": False,
            "may_authorize_policy_update": False,
            "may_authorize_promotion": False,
            "may_authorize_deployment": False,
            "may_feed_stage50": False,
            "may_feed_stage60": False,
            "may_feed_stage70": False,
            "may_feed_another_stage90": False,
            "may_feed_another_experiment": False,
            "contract_hash": self.contract_hash,
        }


# Short alias used by scientific modules and tests.
DirectionalShrinkageProtocol = LooDirectionalShrinkageEnsembleProtocol


def canonical_consumed_test_protocol() -> LooDirectionalShrinkageEnsembleProtocol:
    payload = canonical_protocol_payload()
    return LooDirectionalShrinkageEnsembleProtocol(payload, stable_hash(payload))


def assert_terminal_consumed_test_protocol(
    protocol: LooDirectionalShrinkageEnsembleProtocol,
) -> None:
    """Reject any widening of the consumed-test capability or claim surface."""

    expected = canonical_consumed_test_protocol()
    if protocol.to_payload() != expected.to_payload():
        raise ProtocolError(
            "Directional-shrinkage protocol escaped its terminal boundary."
        )


__all__ = (
    "DirectionalShrinkageProtocol",
    "LooDirectionalShrinkageEnsembleProtocol",
    "assert_terminal_consumed_test_protocol",
    "canonical_consumed_test_protocol",
)
