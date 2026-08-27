"""Neutral nonserializable lineage contracts for SCEPTRE phase orchestration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PhaseCapability:
    """Manager-issued lineage token; not itself raw-label authorization."""

    role: str
    target_center: str
    fold_ordinal: int
    partition_hash: str
    router_bundle_hash: str
    g_proposal_hash: str
    predecessor_decision_hash: str
    predecessor_seal_hash: str
    nonce_hash: str

    def __reduce__(self):  # pragma: no cover - exercised through pickle failure
        raise TypeError("SCEPTRE phase capabilities cannot cross process boundaries.")


@dataclass(frozen=True, slots=True)
class TerminalEvaluationCapability:
    """One-shot lineage token issued only after the complete policy seal."""

    partition_hash: str
    router_bundle_hash: str
    route_policy_hash: str
    policy_seal_hash: str
    durable_attestation_hash: str
    capability_hash: str

    def __reduce__(self):  # pragma: no cover - exercised through pickle failure
        raise TypeError("SCEPTRE terminal capabilities cannot be serialized.")


__all__ = ("PhaseCapability", "TerminalEvaluationCapability")
