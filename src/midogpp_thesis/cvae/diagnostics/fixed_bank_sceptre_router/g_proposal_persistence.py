"""Canonical persistence for label-free SCEPTRE G proposals.

A proposal contains only the already-frozen adaptive decision and immutable
router lineage.  It neither owns the full router bundle nor imports any
label-consuming support, calibration, or uncertainty implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from .adaptive_model_freeze import (
    EXACT_UTILITY_TIE_REASON,
    PREDICTED_UTILITY_SEMANTICS,
    _FALLBACK_REASONS,
    _identifier,
)
from .hashing import canonical_hash, require_sha256
from .identity import PUBLICATION_STATUS, TERMINAL_DECISION
from .outcome_surface import EXACT_B_CANDIDATE
from .seals import FoldDecisionReceipt


@dataclass(frozen=True, slots=True)
class FrozenGProposal:
    """One label-free H proposal bound to the full router and phase partition."""

    target_center: str
    full_router_sha256: str
    frozen_model_sha256: str
    partition_hash: str
    generation_lock_payload_sha256: str
    candidate_menu_hash: str
    candidate_menu_payload_sha256: str
    exact_b_control_receipt_hash: str
    exact_b_control_payload_sha256: str
    decision_policy_sha256: str
    adaptive_decision_sha256: str
    evidence_sha256: str
    ranking_sha256: str | None
    winner_sources: tuple[str, ...]
    proposed_route: str
    fallback_to_exact_b: bool
    reason: str
    proposal_sha256: str = ""

    def __post_init__(self) -> None:
        target = _identifier(self.target_center, "G-proposal target")
        if target not in CENTERS:
            raise ProtocolError("SCEPTRE G-proposal target is unknown.")
        winners = tuple(self.winner_sources)
        if any(source not in legal_routing_sources(target) for source in winners):
            raise ProtocolError("SCEPTRE G-proposal winner inventory drifted.")
        expected_winner_order = tuple(
            source
            for source in legal_routing_sources(target)
            if source in set(winners)
        )
        if winners != expected_winner_order:
            raise ProtocolError("SCEPTRE G-proposal winner order drifted.")
        route = _identifier(self.proposed_route, "G-proposal route")
        reason = _identifier(self.reason, "G-proposal reason")
        ranking_hash = self.ranking_sha256
        if self.fallback_to_exact_b is True:
            if route != EXACT_B_CANDIDATE:
                raise ProtocolError("SCEPTRE G fallback does not route to exact B.")
            if reason not in _FALLBACK_REASONS:
                raise ProtocolError("SCEPTRE G fallback reason drifted.")
            if reason == EXACT_UTILITY_TIE_REASON:
                if len(winners) < 2 or ranking_hash is None:
                    raise ProtocolError("SCEPTRE G tie fallback lost its tie set.")
            elif winners or ranking_hash is not None:
                raise ProtocolError("SCEPTRE invalid G fallback invented a ranking.")
        elif (
            self.fallback_to_exact_b is not False
            or route not in legal_routing_sources(target)
            or winners != (route,)
            or reason != "UNIQUE_PREDICTED_UTILITY_ROUTE"
            or ranking_hash is None
        ):
            raise ProtocolError("SCEPTRE G proposal route semantics drifted.")
        for field_name, role in (
            ("full_router_sha256", "full prelabel router"),
            ("frozen_model_sha256", "frozen H model"),
            ("partition_hash", "G-proposal partition"),
            ("generation_lock_payload_sha256", "GenerationLock payload"),
            ("candidate_menu_payload_sha256", "candidate-menu payload"),
            ("exact_b_control_payload_sha256", "exact-B control payload"),
            ("decision_policy_sha256", "complete decision policy"),
            ("adaptive_decision_sha256", "adaptive decision"),
            ("evidence_sha256", "G-proposal evidence"),
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), role),
            )
        if ranking_hash is not None:
            ranking_hash = require_sha256(ranking_hash, "G-proposal ranking")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "candidate_menu_hash", _identifier(
            self.candidate_menu_hash,
            "G-proposal candidate-menu hash",
        ))
        object.__setattr__(self, "exact_b_control_receipt_hash", _identifier(
            self.exact_b_control_receipt_hash,
            "G-proposal exact-B control receipt",
        ))
        object.__setattr__(self, "ranking_sha256", ranking_hash)
        object.__setattr__(self, "winner_sources", winners)
        object.__setattr__(self, "proposed_route", route)
        object.__setattr__(self, "reason", reason)
        expected = canonical_hash(self._payload_without_hash())
        if self.proposal_sha256 and require_sha256(
            self.proposal_sha256,
            "G proposal",
        ) != expected:
            raise ProtocolError("SCEPTRE G-proposal SHA-256 drifted.")
        object.__setattr__(self, "proposal_sha256", expected)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_frozen_g_proposal_v1",
            "phase": "G_LABEL_FREE",
            "target_center": self.target_center,
            "full_router_sha256": self.full_router_sha256,
            "frozen_model_sha256": self.frozen_model_sha256,
            "partition_hash": self.partition_hash,
            "generation_lock_payload_sha256": (
                self.generation_lock_payload_sha256
            ),
            "candidate_menu_hash": self.candidate_menu_hash,
            "candidate_menu_payload_sha256": self.candidate_menu_payload_sha256,
            "exact_b_control_receipt_hash": self.exact_b_control_receipt_hash,
            "exact_b_control_payload_sha256": (
                self.exact_b_control_payload_sha256
            ),
            "decision_policy_sha256": self.decision_policy_sha256,
            "adaptive_decision_sha256": self.adaptive_decision_sha256,
            "evidence_sha256": self.evidence_sha256,
            "ranking_sha256": self.ranking_sha256,
            "winner_sources": list(self.winner_sources),
            "proposed_route": self.proposed_route,
            "fallback_to_exact_b": self.fallback_to_exact_b,
            "reason": self.reason,
            "score_semantics": PREDICTED_UTILITY_SEMANTICS,
            "higher_is_better": True,
            "labels_consumed": False,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._payload_without_hash(),
            "proposal_sha256": self.proposal_sha256,
        }

    @property
    def router_bundle_hash(self) -> str:
        return self.full_router_sha256

    @property
    def frozen_model_hash(self) -> str:
        return self.frozen_model_sha256

    @property
    def g_proposal_hash(self) -> str:
        return self.proposal_sha256

    @property
    def g_proposed_candidate(self) -> str | None:
        return None if self.fallback_to_exact_b else self.proposed_route

    def to_fold_receipt(self, fold_ordinal: int) -> FoldDecisionReceipt:
        return FoldDecisionReceipt(
            phase="G_LABEL_FREE",
            target_center=self.target_center,
            fold_ordinal=fold_ordinal,
            partition_hash=self.partition_hash,
            router_bundle_hash=self.full_router_sha256,
            g_proposal_hash=self.proposal_sha256,
        )



__all__ = ("FrozenGProposal",)

