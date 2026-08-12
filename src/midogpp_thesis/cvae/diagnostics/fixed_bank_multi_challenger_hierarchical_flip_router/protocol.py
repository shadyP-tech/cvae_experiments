"""Fail-closed consumed-test protocol contract."""

from __future__ import annotations

from dataclasses import dataclass

from ...protocol import ProtocolError
from .config_payloads import canonical_protocol_payload
from .experiment_contracts import CLAIM_ROLE, PUBLICATION_STATUS
from .hashing import canonical_hash


@dataclass(frozen=True)
class MultiChallengerProtocol:
    payload: dict[str, object]
    contract_hash: str

    def __post_init__(self) -> None:
        expected = canonical_protocol_payload()
        if self.payload != expected or self.contract_hash != canonical_hash(expected):
            raise ProtocolError("Multi-challenger protocol contract drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            **self.payload,
            "claim_role": CLAIM_ROLE,
            "publication_status": PUBLICATION_STATUS,
            "consumed_test_data": True,
            "fresh_evidence": False,
            "may_authorize_routing": False,
            "may_authorize_promotion": False,
            "may_feed_another_experiment": False,
            "contract_hash": self.contract_hash,
        }


def canonical_consumed_test_protocol() -> MultiChallengerProtocol:
    payload = canonical_protocol_payload()
    return MultiChallengerProtocol(payload, canonical_hash(payload))


__all__ = ("MultiChallengerProtocol", "canonical_consumed_test_protocol")
