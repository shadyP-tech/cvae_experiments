"""Lineage-bound selection-fold source-family tournament against exact B."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from .hashing import canonical_hash, require_sha256
from .outcome_surface import EXACT_B_CANDIDATE, FamilyOutcome
from .partitions import FOLD_COUNT, ThreeRoleFold
from .policy_contracts import SUPPORT_MINIMUM_BACC_GAIN



@dataclass(frozen=True, slots=True)
class SupportTournamentDecision:
    target_center: str
    fold_ordinal: int
    fold_hash: str
    partition_hash: str
    selection_case_set_hash: str
    calibration_case_set_hash: str
    evaluation_case_set_hash: str
    candidate_menu_hash: str
    exact_b_control_receipt_hash: str
    candidate_menu_payload_sha256: str
    exact_b_control_payload_sha256: str
    router_bundle_hash: str
    decision_policy_sha256: str
    frozen_model_hash: str
    g_proposal_hash: str
    g_proposed_candidate: str | None
    candidate_centers: tuple[str, ...]
    candidate_bacc: tuple[tuple[str, float], ...]
    candidate_outcome_hashes: tuple[tuple[str, str], ...]
    exact_b_outcome_hash: str
    winner_set: tuple[str, ...]
    selected_candidate: str | None
    baseline_bacc: float
    selected_bacc_gain: float
    minimum_bacc_gain: float
    fallback_required: bool
    reason: str
    decision_hash: str = ""

    def __post_init__(self) -> None:
        target = str(self.target_center)
        candidates = tuple(self.candidate_centers)
        expected = legal_routing_sources(target) if target in CENTERS else ()
        if (
            target not in CENTERS
            or isinstance(self.fold_ordinal, bool)
            or self.fold_ordinal not in range(FOLD_COUNT)
            or candidates != expected
        ):
            raise ProtocolError("SCEPTRE support decision scope or inventory drifted.")
        fold_hash = require_sha256(self.fold_hash, "support fold")
        partition = require_sha256(self.partition_hash, "support partition")
        case_set = require_sha256(self.selection_case_set_hash, "selection case set")
        calibration_set = require_sha256(
            self.calibration_case_set_hash, "calibration case set"
        )
        evaluation_set = require_sha256(
            self.evaluation_case_set_hash, "evaluation case set"
        )
        if len({case_set, calibration_set, evaluation_set}) != 3:
            raise ProtocolError("SCEPTRE fold role case sets are not disjoint identities.")
        menu_hash = _identifier(self.candidate_menu_hash, "candidate menu")
        control_hash = _identifier(
            self.exact_b_control_receipt_hash, "exact-B control receipt"
        )
        menu_payload = require_sha256(
            self.candidate_menu_payload_sha256, "candidate-menu payload"
        )
        control_payload = require_sha256(
            self.exact_b_control_payload_sha256, "exact-B control payload"
        )
        router_bundle = require_sha256(
            self.router_bundle_hash, "support frozen router bundle"
        )
        decision_policy = require_sha256(
            self.decision_policy_sha256, "support frozen decision policy"
        )
        frozen_model = require_sha256(self.frozen_model_hash, "support frozen model")
        g_proposal_hash = require_sha256(self.g_proposal_hash, "support G proposal")
        proposed = self.g_proposed_candidate
        if proposed is not None and proposed not in candidates:
            raise ProtocolError("SCEPTRE G proposal is outside exact C minus H.")
        bacc_rows = tuple(self.candidate_bacc)
        outcome_rows = tuple(self.candidate_outcome_hashes)
        if tuple(candidate for candidate, _ in bacc_rows) != candidates:
            raise ProtocolError("SCEPTRE support BACC inventory drifted.")
        if tuple(candidate for candidate, _ in outcome_rows) != candidates:
            raise ProtocolError("SCEPTRE support outcome-hash inventory drifted.")
        if any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for _, value in bacc_rows
        ):
            raise ProtocolError("SCEPTRE support BACC value is invalid.")
        for _, value in outcome_rows:
            require_sha256(value, "support outcome")
        exact_b_hash = require_sha256(self.exact_b_outcome_hash, "exact-B outcome")
        baseline = float(self.baseline_bacc)
        minimum = _nonnegative_finite(self.minimum_bacc_gain, "minimum BACC gain")
        if not math.isfinite(baseline) or not 0.0 <= baseline <= 1.0:
            raise ProtocolError("SCEPTRE support baseline BACC is invalid.")
        scores = dict(bacc_rows)
        winners = () if proposed is None else (proposed,)
        gain = 0.0 if proposed is None else scores[proposed] - baseline
        if tuple(self.winner_set) != winners or self.selected_bacc_gain != gain:
            raise ProtocolError("SCEPTRE support G proposal or gain was not replayed.")
        if proposed is None:
            expected_selected = None
            expected_fallback = True
            expected_reason = "G_PRELABEL_FALLBACK_TO_B"
        elif gain <= minimum:
            expected_selected = None
            expected_fallback = True
            expected_reason = "G_PROPOSAL_INSUFFICIENT_SUPPORT_FALLBACK"
        else:
            expected_selected = proposed
            expected_fallback = False
            expected_reason = "G_PROPOSAL_SUPPORT_ACCEPT"
        if (
            self.selected_candidate != expected_selected
            or self.fallback_required is not expected_fallback
            or self.reason != expected_reason
        ):
            raise ProtocolError("SCEPTRE support decision semantics drifted.")
        body = {
            "schema_version": "sceptre_g_proposal_support_decision_v3",
            "target_center": target,
            "fold_ordinal": self.fold_ordinal,
            "fold_hash": fold_hash,
            "partition_hash": partition,
            "selection_case_set_hash": case_set,
            "calibration_case_set_hash": calibration_set,
            "evaluation_case_set_hash": evaluation_set,
            "candidate_menu_hash": menu_hash,
            "exact_b_control_receipt_hash": control_hash,
            "candidate_menu_payload_sha256": menu_payload,
            "exact_b_control_payload_sha256": control_payload,
            "router_bundle_hash": router_bundle,
            "decision_policy_sha256": decision_policy,
            "frozen_model_hash": frozen_model,
            "g_proposal_hash": g_proposal_hash,
            "g_proposed_candidate": proposed,
            "candidate_centers": list(candidates),
            "candidate_bacc": [list(row) for row in bacc_rows],
            "candidate_outcome_hashes": [list(row) for row in outcome_rows],
            "exact_b_outcome_hash": exact_b_hash,
            "support_evaluated_proposal_set": list(winners),
            "selected_candidate": expected_selected,
            "baseline_bacc": baseline,
            "selected_bacc_gain": gain,
            "minimum_bacc_gain": minimum,
            "fallback_required": expected_fallback,
            "fallback_policy": EXACT_B_CANDIDATE,
            "reason": expected_reason,
            "seed_cells_are_replications_not_candidates": True,
        }
        expected_hash = canonical_hash(body)
        if self.decision_hash and self.decision_hash != expected_hash:
            raise ProtocolError("SCEPTRE support decision hash drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "fold_hash", fold_hash)
        object.__setattr__(self, "partition_hash", partition)
        object.__setattr__(self, "selection_case_set_hash", case_set)
        object.__setattr__(self, "calibration_case_set_hash", calibration_set)
        object.__setattr__(self, "evaluation_case_set_hash", evaluation_set)
        object.__setattr__(self, "candidate_menu_hash", menu_hash)
        object.__setattr__(self, "exact_b_control_receipt_hash", control_hash)
        object.__setattr__(self, "candidate_menu_payload_sha256", menu_payload)
        object.__setattr__(self, "exact_b_control_payload_sha256", control_payload)
        object.__setattr__(self, "router_bundle_hash", router_bundle)
        object.__setattr__(self, "decision_policy_sha256", decision_policy)
        object.__setattr__(self, "frozen_model_hash", frozen_model)
        object.__setattr__(self, "g_proposal_hash", g_proposal_hash)
        object.__setattr__(self, "g_proposed_candidate", proposed)
        object.__setattr__(self, "candidate_centers", candidates)
        object.__setattr__(self, "candidate_bacc", bacc_rows)
        object.__setattr__(self, "candidate_outcome_hashes", outcome_rows)
        object.__setattr__(self, "exact_b_outcome_hash", exact_b_hash)
        object.__setattr__(self, "winner_set", winners)
        object.__setattr__(self, "selected_candidate", expected_selected)
        object.__setattr__(self, "baseline_bacc", baseline)
        object.__setattr__(self, "selected_bacc_gain", gain)
        object.__setattr__(self, "minimum_bacc_gain", minimum)
        object.__setattr__(self, "fallback_required", expected_fallback)
        object.__setattr__(self, "reason", expected_reason)
        object.__setattr__(self, "decision_hash", expected_hash)


def select_support_family(
    outcomes: Iterable[FamilyOutcome],
    *,
    target_center: str,
    fold: ThreeRoleFold,
    partition_hash: str,
    exact_b: FamilyOutcome,
    g_proposal: object,
    frozen_router: object,
    minimum_bacc_gain: float = SUPPORT_MINIMUM_BACC_GAIN,
) -> SupportTournamentDecision:
    from .router_bundle_freeze import FrozenGProposal, FrozenPrelabelRouter

    target = str(target_center)
    if target not in CENTERS:
        raise ProtocolError("SCEPTRE support tournament target is unknown.")
    if (
        not isinstance(fold, ThreeRoleFold)
        or fold.target_center != target
        or exact_b.fold_ordinal != fold.fold_ordinal
    ):
        raise ProtocolError("SCEPTRE support tournament fold lineage drifted.")
    partition = require_sha256(partition_hash, "support partition")
    minimum = _nonnegative_finite(minimum_bacc_gain, "minimum BACC gain")
    if minimum != SUPPORT_MINIMUM_BACC_GAIN:
        raise ProtocolError("SCEPTRE support threshold drifted from the frozen value.")
    rows = tuple(sorted(outcomes, key=lambda row: row.candidate_center))
    expected_candidates = legal_routing_sources(target)
    if (
        len({row.candidate_center for row in rows}) != len(rows)
        or tuple(row.candidate_center for row in rows) != expected_candidates
    ):
        raise ProtocolError("SCEPTRE support tournament is not exact C minus H.")
    if exact_b.candidate_center != EXACT_B_CANDIDATE:
        raise ProtocolError("SCEPTRE support baseline is not exact B.")
    if not isinstance(frozen_router, FrozenPrelabelRouter):
        raise ProtocolError("SCEPTRE support requires its full frozen router.")
    if not isinstance(g_proposal, FrozenGProposal):
        raise ProtocolError("SCEPTRE support requires a bundle-bound G proposal.")
    frozen_model = frozen_router.model_for_target(target)
    proposal_hash = g_proposal.g_proposal_hash
    if (
        frozen_router.partition_hash != partition
        or g_proposal.target_center != target
        or g_proposal.router_bundle_hash != frozen_router.router_bundle_hash
        or g_proposal.partition_hash != partition
        or g_proposal.frozen_model_hash != frozen_model.model_sha256
        or g_proposal.decision_policy_sha256
        != frozen_router.decision_policy_sha256
        or g_proposal.candidate_menu_hash != frozen_model.candidate_menu_hash
        or g_proposal.candidate_menu_payload_sha256
        != frozen_model.candidate_menu_payload_sha256
        or g_proposal.exact_b_control_receipt_hash
        != frozen_model.exact_b_control_receipt_hash
        or g_proposal.exact_b_control_payload_sha256
        != frozen_model.exact_b_control_payload_sha256
        or frozen_model.candidate_menu_hash != exact_b.candidate_menu_hash
        or frozen_model.exact_b_control_receipt_hash
        != exact_b.exact_b_control_receipt_hash
    ):
        raise ProtocolError("SCEPTRE support G/router/model/control lineage differs.")
    if minimum != frozen_router.support_minimum_bacc_gain:
        raise ProtocolError("SCEPTRE support threshold differs from frozen router.")
    proposed = g_proposal.g_proposed_candidate
    router_bundle = frozen_router.router_bundle_hash
    scope = exact_b.scope_key
    if (
        scope[0] != target
        or scope[2] != "SELECTION"
        or scope[3] != partition
        or scope[4] != fold.case_set_hash("SELECTION")
        or exact_b.exact_b_control_receipt_hash is None
        or any(row.scope_key != scope for row in rows)
    ):
        raise ProtocolError("SCEPTRE support candidate/B outcome lineage differs.")
    scores = tuple((row.candidate_center, row.confusion.bacc) for row in rows)
    winners = () if proposed is None else (proposed,)
    gain = 0.0 if proposed is None else dict(scores)[proposed] - exact_b.confusion.bacc
    if proposed is None:
        selected = None
        fallback, reason = True, "G_PRELABEL_FALLBACK_TO_B"
    elif gain <= minimum:
        selected = None
        fallback, reason = True, "G_PROPOSAL_INSUFFICIENT_SUPPORT_FALLBACK"
    else:
        selected = proposed
        fallback, reason = False, "G_PROPOSAL_SUPPORT_ACCEPT"
    return SupportTournamentDecision(
        target_center=target,
        fold_ordinal=exact_b.fold_ordinal,
        fold_hash=fold.fold_hash,
        partition_hash=exact_b.partition_hash,
        selection_case_set_hash=exact_b.case_set_hash,
        calibration_case_set_hash=fold.case_set_hash("CALIBRATION"),
        evaluation_case_set_hash=fold.case_set_hash("EVALUATION"),
        candidate_menu_hash=exact_b.candidate_menu_hash,
        exact_b_control_receipt_hash=exact_b.exact_b_control_receipt_hash,
        candidate_menu_payload_sha256=frozen_model.candidate_menu_payload_sha256,
        exact_b_control_payload_sha256=frozen_model.exact_b_control_payload_sha256,
        router_bundle_hash=router_bundle,
        decision_policy_sha256=frozen_router.decision_policy_sha256,
        frozen_model_hash=frozen_model.model_sha256,
        g_proposal_hash=proposal_hash,
        g_proposed_candidate=proposed,
        candidate_centers=tuple(row.candidate_center for row in rows),
        candidate_bacc=scores,
        candidate_outcome_hashes=tuple(
            (row.candidate_center, row.outcome_hash) for row in rows
        ),
        exact_b_outcome_hash=exact_b.outcome_hash,
        winner_set=winners,
        selected_candidate=selected,
        baseline_bacc=exact_b.confusion.bacc,
        selected_bacc_gain=gain,
        minimum_bacc_gain=minimum,
        fallback_required=fallback,
        reason=reason,
    )


def _nonnegative_finite(value: object, role: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"SCEPTRE {role} is invalid.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(f"SCEPTRE {role} is invalid.") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ProtocolError(f"SCEPTRE {role} must be finite and nonnegative.")
    return parsed


def _identifier(value: object, role: str) -> str:
    text = "" if value is None else str(value)
    if not text or text.strip() != text:
        raise ProtocolError(f"SCEPTRE {role} is invalid.")
    return text


__all__ = (
    "SUPPORT_MINIMUM_BACC_GAIN",
    "SupportTournamentDecision",
    "select_support_family",
)
