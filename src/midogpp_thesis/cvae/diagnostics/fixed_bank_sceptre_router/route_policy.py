"""Serializable final route-policy artifact for terminal SCEPTRE evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from .hashing import canonical_bytes, canonical_hash, require_sha256
from .identity import PUBLICATION_STATUS, TERMINAL_DECISION
from .outcome_surface import EXACT_B_CANDIDATE
from .seals import EXPECTED_DECISION_KEYS


@dataclass(frozen=True, slots=True)
class FrozenRoutePolicy:
    """Exact 45-fold route table derived from sealed calibration decisions."""

    partition_hash: str
    router_bundle_hash: str
    g_seal_hash: str
    selection_seal_hash: str
    policy_seal_hash: str
    route_rows: tuple[tuple[str, int, str, str | None, str, str], ...]
    publication_status: str = PUBLICATION_STATUS
    terminal_decision: str = TERMINAL_DECISION
    policy_artifact_hash: str = ""

    def __post_init__(self) -> None:
        rows = tuple(self.route_rows)
        if any(
            isinstance(row, (str, bytes, bytearray))
            or not isinstance(row, Sequence)
            or len(row) != 6
            for row in rows
        ):
            raise ProtocolError("SCEPTRE route-policy row schema drifted.")
        if any(
            not isinstance(target, str)
            or isinstance(fold, bool)
            or not isinstance(fold, int)
            for target, fold, _, _, _, _ in rows
        ):
            raise ProtocolError("SCEPTRE route-policy key type drifted.")
        if tuple(
            (target, fold) for target, fold, _, _, _, _ in rows
        ) != EXPECTED_DECISION_KEYS:
            raise ProtocolError("SCEPTRE route policy does not cover all 45 folds.")
        normalized: list[tuple[str, int, str, str | None, str, str]] = []
        proposal_by_target: dict[str, tuple[str, str | None]] = {}
        for target, fold, g_hash, g_candidate, route, decision_hash in rows:
            proposal_hash = require_sha256(g_hash, "route-policy G proposal")
            legal = legal_routing_sources(target)
            if g_candidate is not None and (
                not isinstance(g_candidate, str) or g_candidate not in legal
            ):
                raise ProtocolError("SCEPTRE route-policy G candidate is invalid.")
            prior = proposal_by_target.setdefault(target, (proposal_hash, g_candidate))
            if prior != (proposal_hash, g_candidate):
                raise ProtocolError(
                    "SCEPTRE route-policy target-global G proposal drifted."
                )
            allowed_routes = (
                {EXACT_B_CANDIDATE}
                if g_candidate is None
                else {g_candidate, EXACT_B_CANDIDATE}
            )
            if route not in allowed_routes:
                raise ProtocolError("SCEPTRE final route is outside G-or-exact-B.")
            normalized.append(
                (
                    target,
                    fold,
                    proposal_hash,
                    g_candidate,
                    route,
                    require_sha256(decision_hash, "route decision"),
                )
            )
        if (
            self.publication_status != PUBLICATION_STATUS
            or self.terminal_decision != TERMINAL_DECISION
        ):
            raise ProtocolError("SCEPTRE route-policy claim boundary drifted.")
        for digest, role in (
            (self.partition_hash, "route-policy partition"),
            (self.router_bundle_hash, "route-policy router bundle"),
            (self.g_seal_hash, "route-policy G seal"),
            (self.selection_seal_hash, "route-policy selection seal"),
            (self.policy_seal_hash, "route-policy decision seal"),
        ):
            require_sha256(digest, role)
        object.__setattr__(self, "route_rows", tuple(normalized))
        expected = canonical_hash(self._payload_without_hash())
        if self.policy_artifact_hash and require_sha256(
            self.policy_artifact_hash,
            "route-policy artifact",
        ) != expected:
            raise ProtocolError("SCEPTRE route-policy artifact hash drifted.")
        object.__setattr__(self, "policy_artifact_hash", expected)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_frozen_route_policy_v2",
            "partition_hash": self.partition_hash,
            "router_bundle_hash": self.router_bundle_hash,
            "g_seal_hash": self.g_seal_hash,
            "selection_seal_hash": self.selection_seal_hash,
            "policy_seal_hash": self.policy_seal_hash,
            "route_rows": [list(row) for row in self.route_rows],
            "route_inventory": "EXACT_45_TARGET_FOLDS",
            "only_legal_transitions": ["G_TO_SAME_EXPERT", "ANY_STAGE_TO_EXACT_B"],
            "publication_status": self.publication_status,
            "terminal_decision": self.terminal_decision,
            "fresh_evidence": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._payload_without_hash(),
            "policy_artifact_hash": self.policy_artifact_hash,
        }

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_payload())

    @classmethod
    def from_payload(cls, payload: object) -> "FrozenRoutePolicy":
        """Reconstruct an exact frozen route table without widening its schema."""

        if not isinstance(payload, Mapping):
            raise ProtocolError("SCEPTRE route-policy payload is not an object.")
        expected_keys = {
            "schema_version",
            "partition_hash",
            "router_bundle_hash",
            "g_seal_hash",
            "selection_seal_hash",
            "policy_seal_hash",
            "route_rows",
            "route_inventory",
            "only_legal_transitions",
            "publication_status",
            "terminal_decision",
            "fresh_evidence",
            "policy_artifact_hash",
        }
        if set(payload) != expected_keys:
            raise ProtocolError("SCEPTRE route-policy payload schema drifted.")
        if (
            payload.get("schema_version") != "sceptre_frozen_route_policy_v2"
            or payload.get("route_inventory") != "EXACT_45_TARGET_FOLDS"
            or payload.get("only_legal_transitions")
            != ["G_TO_SAME_EXPERT", "ANY_STAGE_TO_EXACT_B"]
            or payload.get("fresh_evidence") is not False
        ):
            raise ProtocolError("SCEPTRE route-policy payload semantics drifted.")
        raw_rows = payload.get("route_rows")
        if isinstance(raw_rows, (str, bytes, bytearray)) or not isinstance(
            raw_rows, Sequence
        ):
            raise ProtocolError("SCEPTRE route-policy rows are not a sequence.")
        try:
            rows = tuple(tuple(row) for row in raw_rows)
            policy = cls(
                partition_hash=str(payload["partition_hash"]),
                router_bundle_hash=str(payload["router_bundle_hash"]),
                g_seal_hash=str(payload["g_seal_hash"]),
                selection_seal_hash=str(payload["selection_seal_hash"]),
                policy_seal_hash=str(payload["policy_seal_hash"]),
                route_rows=rows,
                publication_status=str(payload["publication_status"]),
                terminal_decision=str(payload["terminal_decision"]),
                policy_artifact_hash=str(payload["policy_artifact_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("SCEPTRE route-policy payload is invalid.") from exc
        if policy.to_payload() != dict(payload):
            raise ProtocolError("SCEPTRE route-policy payload replay drifted.")
        return policy

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "FrozenRoutePolicy":
        """Load only the canonical byte representation emitted by this class."""

        if not isinstance(payload, bytes):
            raise ProtocolError("SCEPTRE route-policy serialization must be bytes.")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("SCEPTRE route-policy serialization is invalid.") from exc
        if canonical_bytes(decoded) != payload:
            raise ProtocolError("SCEPTRE route-policy serialization is not canonical.")
        return cls.from_payload(decoded)

    def route_for(self, target_center: str, fold_ordinal: int) -> str:
        matches = tuple(
            route
            for target, fold, _, _, route, _ in self.route_rows
            if (target, fold) == (str(target_center), int(fold_ordinal))
        )
        if len(matches) != 1:
            raise ProtocolError("SCEPTRE route-policy lookup is outside its grid.")
        return matches[0]

    def g_proposal_for(
        self, target_center: str, fold_ordinal: int
    ) -> tuple[str, str | None]:
        matches = tuple(
            (proposal_hash, candidate)
            for target, fold, proposal_hash, candidate, _, _ in self.route_rows
            if (target, fold) == (str(target_center), int(fold_ordinal))
        )
        if len(matches) != 1:
            raise ProtocolError("SCEPTRE route-policy G lookup is outside its grid.")
        return matches[0]


__all__ = ("FrozenRoutePolicy",)
