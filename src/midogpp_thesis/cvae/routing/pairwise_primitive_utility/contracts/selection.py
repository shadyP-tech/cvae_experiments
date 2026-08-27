"""Typed evidence and deterministic fail-closed selection contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .pairwise import ActionQuery
from .shared import P_ACTION_ID, ProtocolError, _text, canonical_sha256
from .uncertainty import CalibratedBound
from .utility import NormalizedUtility


def _bound_payload(bound: CalibratedBound) -> tuple[float, float, str, str]:
    return (bound.mean, bound.bound, bound.side, bound.component_hash)


@dataclass(frozen=True, slots=True)
class ActionSelectionEvidence:
    """Canonical typed analytic evidence for one active representative."""

    query: ActionQuery
    equivalent_action_ids: tuple[str, ...]
    utility: NormalizedUtility
    ranking_score: float
    bacc: CalibratedBound
    brier: CalibratedBound
    log: CalibratedBound
    pairwise_bounds: tuple[tuple[ActionQuery, CalibratedBound], ...]
    candidate_pool_receipt_hash: str
    pairwise_model_hash: str
    uncertainty_calibration_hash: str
    opportunity_case_receipt_hash: str
    bacc_ranking_policy_hash: str
    evidence_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.query, ActionQuery) or self.query.action_id == P_ACTION_ID:
            raise ProtocolError("Action selection evidence requires a typed active query.")
        equivalents = tuple(
            sorted({_text(value, role="equivalent action") for value in self.equivalent_action_ids})
        )
        pairwise = tuple(sorted(self.pairwise_bounds, key=lambda row: row[0].action_id))
        if (
            self.query.action_id not in equivalents
            or not isinstance(self.utility, NormalizedUtility)
            or len({query.action_id for query, _ in pairwise}) != len(pairwise)
            or any(
                not isinstance(query, ActionQuery)
                or query.action_id == self.query.action_id
                or not isinstance(bound, CalibratedBound)
                or bound.side != "lower"
                for query, bound in pairwise
            )
            or not math.isfinite(float(self.ranking_score))
            or not isinstance(self.bacc, CalibratedBound)
            or not isinstance(self.brier, CalibratedBound)
            or not isinstance(self.log, CalibratedBound)
            or self.bacc.side != "lower"
            or self.brier.side != "upper"
            or self.log.side != "upper"
        ):
            raise ProtocolError("Action selection evidence is invalid.")
        object.__setattr__(self, "equivalent_action_ids", equivalents)
        object.__setattr__(self, "pairwise_bounds", pairwise)
        object.__setattr__(self, "ranking_score", float(self.ranking_score))
        for name in (
            "candidate_pool_receipt_hash",
            "pairwise_model_hash",
            "uncertainty_calibration_hash",
            "opportunity_case_receipt_hash",
            "bacc_ranking_policy_hash",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), role=name))
        object.__setattr__(
            self,
            "evidence_hash",
            canonical_sha256(
                {
                    "schema": "analytic_action_selection_evidence_v2",
                    "query": (
                        self.query.action_id,
                        self.query.family,
                        self.query.direction,
                        self.query.feature_names,
                        self.query.feature_values,
                    ),
                    "equivalent_action_ids": equivalents,
                    "utility_response_hash": self.utility.response_hash,
                    "ranking_score": self.ranking_score,
                    "bacc": _bound_payload(self.bacc),
                    "brier": _bound_payload(self.brier),
                    "log": _bound_payload(self.log),
                    "pairwise": tuple(
                        (
                            query.action_id,
                            query.family,
                            query.direction,
                            query.feature_names,
                            query.feature_values,
                            _bound_payload(bound),
                        )
                        for query, bound in pairwise
                    ),
                    "candidate_pool_receipt_hash": self.candidate_pool_receipt_hash,
                    "pairwise_model_hash": self.pairwise_model_hash,
                    "uncertainty_calibration_hash": self.uncertainty_calibration_hash,
                    "opportunity_case_receipt_hash": self.opportunity_case_receipt_hash,
                    "bacc_ranking_policy_hash": self.bacc_ranking_policy_hash,
                }
            ),
        )

    @property
    def action_id(self) -> str:
        return self.query.action_id

    @property
    def family(self) -> str:
        return self.query.family

    @property
    def direction(self) -> str:
        return self.query.direction

    def pairwise_lower(self, comparator_id: object) -> CalibratedBound:
        key = str(comparator_id)
        for query, value in self.pairwise_bounds:
            if query.action_id == key:
                return value
        raise ProtocolError(f"Missing pairwise lower bound versus {key}.")

    def comparator_query(self, comparator_id: object) -> ActionQuery:
        key = str(comparator_id)
        for query, _ in self.pairwise_bounds:
            if query.action_id == key:
                return query
        raise ProtocolError(f"Missing pairwise comparator query: {key}.")


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """Deterministic decision; all failures return exact P."""

    selected_action_id: str
    raw_winner_action_id: str
    fallback_to_p: bool
    reason: str
    active_representative_count: int
    runner_up_action_id: str | None
    selected_equivalent_action_ids: tuple[str, ...]
    candidate_pool_receipt_hash: str
    pairwise_model_hash: str
    uncertainty_calibration_hash: str
    opportunity_case_receipt_hash: str
    bacc_ranking_policy_hash: str
    opportunity_active_representative_ids: tuple[str, ...]
    decision_hash: str = field(init=False)

    def __post_init__(self) -> None:
        selected = _text(self.selected_action_id, role="selected action")
        raw = _text(self.raw_winner_action_id, role="raw winner action")
        runner = (
            None
            if self.runner_up_action_id is None
            else _text(self.runner_up_action_id, role="runner-up action")
        )
        fallback = bool(self.fallback_to_p)
        equivalents = tuple(
            sorted(_text(value, role="selected equivalent action") for value in self.selected_equivalent_action_ids)
        )
        active_ids = tuple(
            sorted(_text(value, role="active opportunity representative") for value in self.opportunity_active_representative_ids)
        )
        count = int(self.active_representative_count)
        if (
            len(set(active_ids)) != len(active_ids)
            or count != len(active_ids)
            or fallback != (selected == P_ACTION_ID)
            or (count == 0 and (raw != P_ACTION_ID or runner is not None or not fallback))
            or (count > 0 and raw not in active_ids)
            or (count == 1 and runner is not None)
            or (count > 1 and (runner not in active_ids or runner == raw))
            or (not fallback and (selected != raw or selected not in active_ids))
            or (fallback and equivalents != (P_ACTION_ID,))
            or (not fallback and selected not in equivalents)
        ):
            raise ProtocolError("Selection decision is inconsistent with its opportunity inventory.")
        object.__setattr__(self, "selected_action_id", selected)
        object.__setattr__(self, "raw_winner_action_id", raw)
        object.__setattr__(self, "runner_up_action_id", runner)
        object.__setattr__(self, "fallback_to_p", fallback)
        object.__setattr__(self, "reason", _text(self.reason, role="selection reason"))
        object.__setattr__(self, "active_representative_count", count)
        object.__setattr__(self, "selected_equivalent_action_ids", equivalents)
        object.__setattr__(self, "opportunity_active_representative_ids", active_ids)
        for name in (
            "candidate_pool_receipt_hash",
            "pairwise_model_hash",
            "uncertainty_calibration_hash",
            "opportunity_case_receipt_hash",
            "bacc_ranking_policy_hash",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), role=name))
        object.__setattr__(
            self,
            "decision_hash",
            canonical_sha256(
                {
                    "schema": "pairwise_selection_decision_v3",
                    "selected_action_id": selected,
                    "raw_winner_action_id": raw,
                    "fallback_to_p": fallback,
                    "reason": self.reason,
                    "active_count": count,
                    "runner_up": runner,
                    "equivalent_actions": equivalents,
                    "candidate_pool_receipt_hash": self.candidate_pool_receipt_hash,
                    "pairwise_model_hash": self.pairwise_model_hash,
                    "uncertainty_calibration_hash": self.uncertainty_calibration_hash,
                    "opportunity_case_receipt_hash": self.opportunity_case_receipt_hash,
                    "bacc_ranking_policy_hash": self.bacc_ranking_policy_hash,
                    "opportunity_active_representative_ids": active_ids,
                }
            ),
        )


__all__ = ("ActionSelectionEvidence", "SelectionDecision")
