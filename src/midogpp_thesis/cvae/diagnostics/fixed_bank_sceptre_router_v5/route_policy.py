"""Serializable 45-fold candidate-set route policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from ..fixed_bank_sceptre_router.hashing import (
    canonical_bytes,
    canonical_hash,
    require_sha256,
)
from ..fixed_bank_sceptre_router.outcome_surface import EXACT_B_CANDIDATE
from ..fixed_bank_sceptre_router.seals import EXPECTED_DECISION_KEYS
from .identity import POLICY_TRANSITION, PUBLICATION_STATUS, TERMINAL_DECISION


@dataclass(frozen=True, slots=True)
class FrozenRoutePolicy:
    partition_hash: str
    routing_context_hash: str
    proposal_set_seal_hash: str
    support_seal_hash: str
    policy_seal_hash: str
    route_rows: tuple[
        tuple[str, int, str, str, str | None, str, str], ...
    ]
    policy_artifact_hash: str = ""

    def __post_init__(self) -> None:
        rows = tuple(self.route_rows)
        if any(
            isinstance(row, (str, bytes, bytearray))
            or not isinstance(row, Sequence)
            or len(row) != 7
            for row in rows
        ):
            raise ProtocolError("SCEPTRE v5 route-policy row schema drifted.")
        if tuple((row[0], row[1]) for row in rows) != EXPECTED_DECISION_KEYS:
            raise ProtocolError("SCEPTRE v5 route policy lacks 45 folds.")
        normalized = []
        proposal_by_target: dict[str, str] = {}
        for (
            target,
            fold,
            proposal_hash,
            support_hash,
            support_candidate,
            route,
            confirmation_hash,
        ) in rows:
            target = str(target)
            legal = legal_routing_sources(target)
            proposal = require_sha256(proposal_hash, "route proposal set")
            support = require_sha256(support_hash, "route support decision")
            confirmation = require_sha256(
                confirmation_hash, "route confirmation decision"
            )
            if support_candidate is not None and support_candidate not in legal:
                raise ProtocolError("SCEPTRE v5 support route is illegal.")
            allowed = (
                {EXACT_B_CANDIDATE}
                if support_candidate is None
                else {support_candidate, EXACT_B_CANDIDATE}
            )
            if route not in allowed:
                raise ProtocolError("SCEPTRE v5 calibration changed the selected member.")
            prior = proposal_by_target.setdefault(target, proposal)
            if prior != proposal:
                raise ProtocolError("SCEPTRE v5 target proposal changed across folds.")
            normalized.append(
                (
                    target,
                    int(fold),
                    proposal,
                    support,
                    support_candidate,
                    str(route),
                    confirmation,
                )
            )
        for value, role in (
            (self.partition_hash, "policy partition"),
            (self.routing_context_hash, "policy routing context"),
            (self.proposal_set_seal_hash, "proposal-set seal"),
            (self.support_seal_hash, "support seal"),
            (self.policy_seal_hash, "policy seal"),
        ):
            require_sha256(value, role)
        object.__setattr__(self, "route_rows", tuple(normalized))
        expected = canonical_hash(self._payload_without_hash())
        if self.policy_artifact_hash and self.policy_artifact_hash != expected:
            raise ProtocolError("SCEPTRE v5 policy artifact hash drifted.")
        object.__setattr__(self, "policy_artifact_hash", expected)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_v5_frozen_route_policy_v1",
            "partition_hash": self.partition_hash,
            "routing_context_hash": self.routing_context_hash,
            "proposal_set_seal_hash": self.proposal_set_seal_hash,
            "support_seal_hash": self.support_seal_hash,
            "policy_seal_hash": self.policy_seal_hash,
            "policy_transition": POLICY_TRANSITION,
            "route_rows": [
                {
                    "target_center": target,
                    "fold_ordinal": fold,
                    "proposal_set_hash": proposal,
                    "support_decision_hash": support,
                    "support_selected_candidate": candidate,
                    "route": route,
                    "confirmation_decision_hash": confirmation,
                }
                for (
                    target,
                    fold,
                    proposal,
                    support,
                    candidate,
                    route,
                    confirmation,
                ) in self.route_rows
            ],
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
            "routing_success_claimed": False,
            "nelbo_compatibility_claimed": False,
            "may_feed_another_experiment": False,
            "raw_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._payload_without_hash(),
            "policy_artifact_hash": self.policy_artifact_hash,
        }

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_payload())

    def route_for(self, target_center: str, fold_ordinal: int) -> str:
        key = (str(target_center), int(fold_ordinal))
        try:
            return next(row[5] for row in self.route_rows if row[:2] == key)
        except StopIteration as exc:
            raise ProtocolError("SCEPTRE v5 route-policy key is absent.") from exc

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "FrozenRoutePolicy":
        if payload.get("schema_version") != "sceptre_v5_frozen_route_policy_v1":
            raise ProtocolError("SCEPTRE v5 policy schema drifted.")
        raw_rows = payload.get("route_rows")
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, str):
            raise ProtocolError("SCEPTRE v5 policy rows are malformed.")
        try:
            rows = tuple(
                (
                    str(row["target_center"]),
                    int(row["fold_ordinal"]),
                    str(row["proposal_set_hash"]),
                    str(row["support_decision_hash"]),
                    (
                        None
                        if row["support_selected_candidate"] is None
                        else str(row["support_selected_candidate"])
                    ),
                    str(row["route"]),
                    str(row["confirmation_decision_hash"]),
                )
                for row in raw_rows
            )
            return cls(
                partition_hash=str(payload["partition_hash"]),
                routing_context_hash=str(payload["routing_context_hash"]),
                proposal_set_seal_hash=str(payload["proposal_set_seal_hash"]),
                support_seal_hash=str(payload["support_seal_hash"]),
                policy_seal_hash=str(payload["policy_seal_hash"]),
                route_rows=rows,
                policy_artifact_hash=str(payload["policy_artifact_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("SCEPTRE v5 policy payload is malformed.") from exc

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "FrozenRoutePolicy":
        try:
            raw = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("Cannot parse SCEPTRE v5 policy bytes.") from exc
        if not isinstance(raw, Mapping) or canonical_bytes(raw) != payload:
            raise ProtocolError("SCEPTRE v5 policy bytes are not canonical.")
        return cls.from_payload(raw)


__all__ = ("FrozenRoutePolicy",)
