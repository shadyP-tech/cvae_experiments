"""Label-free ranked proposal sets for SCEPTRE v4.

The historical source-inner surface contains expert-vs-expert utility only.
Consequently these scores are a ranking prior over ``C minus H`` and are never
represented as calibrated advantage over exact B.  Exact B is carried as an
explicit downstream fallback action with no invented source-inner score.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from ..fixed_bank_sceptre_router.adaptive_model_freeze import (
    FrozenAdaptiveUtilityModel,
)
from ..fixed_bank_sceptre_router.evidence_builder import EvidenceFeatureBundle
from ..fixed_bank_sceptre_router.hashing import canonical_hash, require_sha256
from .identity import EXACT_B_ACTION, PROPOSAL_SET_ROLE


@dataclass(frozen=True, slots=True)
class FrozenCandidateSetProposal:
    """One target-global, label-free full expert ranking plus explicit B."""

    target_center: str
    candidate_sources: tuple[str, ...]
    ranked_sources: tuple[str, ...]
    predicted_utility_by_source: tuple[tuple[str, float], ...]
    exact_tie_groups: tuple[tuple[str, ...], ...]
    frozen_model_sha256: str
    candidate_menu_hash: str
    candidate_menu_payload_sha256: str
    exact_b_control_receipt_hash: str
    exact_b_control_payload_sha256: str
    target_evidence_sha256: str
    target_evidence_receipt_sha256: str
    proposal_set_hash: str = ""

    def __post_init__(self) -> None:
        target = str(self.target_center)
        if target not in CENTERS:
            raise ProtocolError("SCEPTRE v4 proposal target is unknown.")
        candidates = tuple(map(str, self.candidate_sources))
        expected = legal_routing_sources(target)
        if candidates != expected:
            raise ProtocolError("SCEPTRE v4 proposal is not exact C minus H.")
        ranked = tuple(map(str, self.ranked_sources))
        if len(ranked) != len(candidates) or set(ranked) != set(candidates):
            raise ProtocolError("SCEPTRE v4 proposal ranking is incomplete.")
        score_rows = tuple(
            (str(source), float(value))
            for source, value in self.predicted_utility_by_source
        )
        if (
            tuple(source for source, _ in score_rows) != candidates
            or any(not math.isfinite(value) for _, value in score_rows)
        ):
            raise ProtocolError("SCEPTRE v4 proposal scores are invalid.")
        score_by_source = dict(score_rows)
        expected_ranked = tuple(
            sorted(candidates, key=lambda source: (-score_by_source[source], source))
        )
        if ranked != expected_ranked:
            raise ProtocolError("SCEPTRE v4 proposal order does not replay.")
        groups = _exact_tie_groups(ranked, score_by_source)
        if tuple(tuple(map(str, row)) for row in self.exact_tie_groups) != groups:
            raise ProtocolError("SCEPTRE v4 proposal tie groups drifted.")
        for value, role in (
            (self.frozen_model_sha256, "frozen model"),
            (self.candidate_menu_payload_sha256, "candidate-menu payload"),
            (self.exact_b_control_payload_sha256, "exact-B payload"),
            (self.target_evidence_sha256, "target evidence"),
            (self.target_evidence_receipt_sha256, "target evidence receipt"),
        ):
            require_sha256(value, role)
        if not self.candidate_menu_hash or not self.exact_b_control_receipt_hash:
            raise ProtocolError("SCEPTRE v4 proposal lost a control identity.")
        body = self._payload_without_hash(
            target=target,
            candidates=candidates,
            ranked=ranked,
            scores=score_rows,
            groups=groups,
        )
        expected_hash = canonical_hash(body)
        if self.proposal_set_hash and require_sha256(
            self.proposal_set_hash, "proposal set"
        ) != expected_hash:
            raise ProtocolError("SCEPTRE v4 proposal-set hash drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "candidate_sources", candidates)
        object.__setattr__(self, "ranked_sources", ranked)
        object.__setattr__(self, "predicted_utility_by_source", score_rows)
        object.__setattr__(self, "exact_tie_groups", groups)
        object.__setattr__(self, "proposal_set_hash", expected_hash)

    def _payload_without_hash(
        self,
        *,
        target: str | None = None,
        candidates: tuple[str, ...] | None = None,
        ranked: tuple[str, ...] | None = None,
        scores: tuple[tuple[str, float], ...] | None = None,
        groups: tuple[tuple[str, ...], ...] | None = None,
    ) -> dict[str, object]:
        return {
            "schema_version": "sceptre_v4_frozen_candidate_set_proposal_v1",
            "proposal_role": PROPOSAL_SET_ROLE,
            "target_center": self.target_center if target is None else target,
            "candidate_sources": list(
                self.candidate_sources if candidates is None else candidates
            ),
            "ranked_sources": list(
                self.ranked_sources if ranked is None else ranked
            ),
            "predicted_utility_by_source": [
                [source, value]
                for source, value in (
                    self.predicted_utility_by_source if scores is None else scores
                )
            ],
            "exact_tie_groups": [
                list(row)
                for row in (self.exact_tie_groups if groups is None else groups)
            ],
            "proposal_set_size": len(
                self.candidate_sources if candidates is None else candidates
            ),
            "proposal_set_is_full_C_minus_H": True,
            "top_k_selected_from_consumed_results": False,
            "exact_b_action": EXACT_B_ACTION,
            "exact_b_source_inner_score": None,
            "exact_b_advantage_model_available": False,
            "source_inner_score_semantics": (
                "EXPERT_RANKING_PRIOR_HIGHER_IS_BETTER_NOT_ADVANTAGE_OVER_B"
            ),
            "frozen_model_sha256": self.frozen_model_sha256,
            "candidate_menu_hash": self.candidate_menu_hash,
            "candidate_menu_payload_sha256": self.candidate_menu_payload_sha256,
            "exact_b_control_receipt_hash": self.exact_b_control_receipt_hash,
            "exact_b_control_payload_sha256": self.exact_b_control_payload_sha256,
            "target_evidence_sha256": self.target_evidence_sha256,
            "target_evidence_receipt_sha256": (
                self.target_evidence_receipt_sha256
            ),
            "target_labels_consumed": False,
            "fresh_evidence": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._payload_without_hash(),
            "proposal_set_hash": self.proposal_set_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "FrozenCandidateSetProposal":
        """Rehydrate and semantically replay a durable proposal-set DTO."""

        try:
            value = cls(
                target_center=str(payload["target_center"]),
                candidate_sources=tuple(map(str, payload["candidate_sources"])),
                ranked_sources=tuple(map(str, payload["ranked_sources"])),
                predicted_utility_by_source=tuple(
                    (str(row[0]), float(row[1]))
                    for row in payload["predicted_utility_by_source"]
                ),
                exact_tie_groups=tuple(
                    tuple(map(str, row)) for row in payload["exact_tie_groups"]
                ),
                frozen_model_sha256=str(payload["frozen_model_sha256"]),
                candidate_menu_hash=str(payload["candidate_menu_hash"]),
                candidate_menu_payload_sha256=str(
                    payload["candidate_menu_payload_sha256"]
                ),
                exact_b_control_receipt_hash=str(
                    payload["exact_b_control_receipt_hash"]
                ),
                exact_b_control_payload_sha256=str(
                    payload["exact_b_control_payload_sha256"]
                ),
                target_evidence_sha256=str(payload["target_evidence_sha256"]),
                target_evidence_receipt_sha256=str(
                    payload["target_evidence_receipt_sha256"]
                ),
                proposal_set_hash=str(payload["proposal_set_hash"]),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ProtocolError("SCEPTRE v4 proposal-set payload is malformed.") from exc
        if value.to_payload() != dict(payload):
            raise ProtocolError("SCEPTRE v4 proposal-set payload drifted.")
        return value

    @property
    def score_by_source(self) -> dict[str, float]:
        return dict(self.predicted_utility_by_source)


def build_candidate_set_proposal(
    frozen: FrozenAdaptiveUtilityModel,
    evidence: EvidenceFeatureBundle,
) -> FrozenCandidateSetProposal:
    """Score all eight experts and freeze their complete deterministic order."""

    if not isinstance(frozen, FrozenAdaptiveUtilityModel):
        raise ProtocolError("SCEPTRE v4 proposal requires a frozen adaptive model.")
    if not isinstance(evidence, EvidenceFeatureBundle):
        raise ProtocolError("SCEPTRE v4 proposal requires bound target evidence.")
    receipt = evidence.receipt
    expected_keys = tuple(
        (frozen.outer_target, source) for source in frozen.candidate_sources
    )
    if (
        receipt.role != "TARGET_PREDICTION"
        or receipt.target_center != frozen.outer_target
        or receipt.feature_names != frozen.feature_names
        or receipt.retained_keys != expected_keys
        or receipt.labels_consumed is not False
        or receipt.exact_nelbo is not False
    ):
        raise ProtocolError("SCEPTRE v4 target evidence lineage drifted.")
    rows = tuple(evidence.rows)
    model = frozen.reconstruct_model()
    scores = tuple(
        (source, model.predict(next(row for row in rows if row.candidate_center == source)))
        for source in frozen.candidate_sources
    )
    if any(not math.isfinite(value) for _, value in scores):
        raise ProtocolError("SCEPTRE v4 proposal score is non-finite.")
    score_by_source = dict(scores)
    ranked = tuple(
        sorted(
            frozen.candidate_sources,
            key=lambda source: (-score_by_source[source], source),
        )
    )
    evidence_hash = canonical_hash(
        {
            "schema_version": "sceptre_v4_target_ranking_evidence_v1",
            "target_center": frozen.outer_target,
            "frozen_model_sha256": frozen.model_sha256,
            "target_evidence_receipt_sha256": receipt.receipt_hash,
            "feature_names": list(frozen.feature_names),
            "rows": [
                {
                    "candidate_center": row.candidate_center,
                    "values": list(row.values),
                    "labels_consumed": False,
                }
                for row in rows
            ],
        }
    )
    return FrozenCandidateSetProposal(
        target_center=frozen.outer_target,
        candidate_sources=frozen.candidate_sources,
        ranked_sources=ranked,
        predicted_utility_by_source=scores,
        exact_tie_groups=_exact_tie_groups(ranked, score_by_source),
        frozen_model_sha256=frozen.model_sha256,
        candidate_menu_hash=frozen.candidate_menu_hash,
        candidate_menu_payload_sha256=frozen.candidate_menu_payload_sha256,
        exact_b_control_receipt_hash=frozen.exact_b_control_receipt_hash,
        exact_b_control_payload_sha256=frozen.exact_b_control_payload_sha256,
        target_evidence_sha256=evidence_hash,
        target_evidence_receipt_sha256=receipt.receipt_hash,
    )


def _exact_tie_groups(
    ranked: tuple[str, ...], scores: dict[str, float]
) -> tuple[tuple[str, ...], ...]:
    groups: list[list[str]] = []
    for source in ranked:
        if not groups or scores[groups[-1][0]] != scores[source]:
            groups.append([source])
        else:
            groups[-1].append(source)
    return tuple(tuple(row) for row in groups)


__all__ = ("FrozenCandidateSetProposal", "build_candidate_set_proposal")
