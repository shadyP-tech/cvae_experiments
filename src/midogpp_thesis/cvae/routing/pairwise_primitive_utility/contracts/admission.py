"""Pre-label sealing and source-only terminal admission contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .opportunity import OpportunityCaseReceipt
from .pairwise import BaccRankingPolicy, CandidatePoolReceipt, PairwiseRankerModel
from .selection import ActionSelectionEvidence, SelectionDecision
from .shared import P_ACTION_ID, ProtocolError, _text, canonical_sha256
from .uncertainty import UncertaintyCalibration


@dataclass(frozen=True, slots=True)
class AdmissionCandidate:
    """Source-OOF candidate joined to terminal utility only after sealing."""

    selection_evidence: ActionSelectionEvidence
    realized_bacc_gain: float
    realized_brier_delta: float
    realized_log_delta: float

    def __post_init__(self) -> None:
        numeric = (self.realized_bacc_gain, self.realized_brier_delta, self.realized_log_delta)
        if (
            not isinstance(self.selection_evidence, ActionSelectionEvidence)
            or not all(math.isfinite(float(value)) for value in numeric)
        ):
            raise ProtocolError("Admission candidate is invalid.")
        for name in ("realized_bacc_gain", "realized_brier_delta", "realized_log_delta"):
            object.__setattr__(self, name, float(getattr(self, name)))

    @property
    def action_id(self) -> str:
        return self.selection_evidence.action_id

    @property
    def surface_hash(self) -> str:
        return self.selection_evidence.utility.candidate_probability_hash

    @property
    def predicted_score(self) -> float:
        return self.selection_evidence.ranking_score


def _recompute_action_score(
    model: PairwiseRankerModel, evidence: ActionSelectionEvidence
) -> float:
    query = evidence.query
    if query.feature_names != model.feature_names:
        raise ProtocolError("Admission evidence feature schema drifted from its model.")
    schema = {action: (family, direction) for action, family, direction in model.action_schema}
    if schema.get(query.action_id) != (query.family, query.direction):
        raise ProtocolError("Admission evidence action schema drifted from its model.")
    coefficients = dict(zip(model.design_names, model.coefficients, strict=True))
    standardized = tuple(
        (value - mean) / scale
        for value, mean, scale in zip(
            query.feature_values, model.feature_mean, model.feature_scale, strict=True
        )
    )
    score = coefficients[f"action_intercept::{query.action_id}"]
    for feature, value in zip(model.feature_names, standardized, strict=True):
        score += value * coefficients[f"action_feature::{query.action_id}::{feature}"]
        score += value * coefficients[f"family_feature::{query.family}::{feature}"]
        score += value * coefficients[f"direction_feature::{query.direction}::{feature}"]
    if not math.isfinite(score):
        raise ProtocolError("Admission evidence recomputed a non-finite score.")
    return float(score)


@dataclass(frozen=True, slots=True)
class AdmissionDecisionReceipt:
    """Exact pre-label decision and prediction inventory."""

    center_id: str
    case_id: str
    selection_decision: SelectionDecision
    candidate_evidence: tuple[ActionSelectionEvidence, ...]
    candidate_pool: CandidatePoolReceipt
    pairwise_model: PairwiseRankerModel
    uncertainty_calibration: UncertaintyCalibration
    opportunity_receipt: OpportunityCaseReceipt
    ranking_policy: BaccRankingPolicy
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        center_id = _text(self.center_id, role="decision center")
        case_id = _text(self.case_id, role="decision case")
        evidence = tuple(sorted(self.candidate_evidence, key=lambda row: row.action_id))
        if (
            not isinstance(self.selection_decision, SelectionDecision)
            or not isinstance(self.candidate_pool, CandidatePoolReceipt)
            or not isinstance(self.pairwise_model, PairwiseRankerModel)
            or not isinstance(self.uncertainty_calibration, UncertaintyCalibration)
            or not isinstance(self.opportunity_receipt, OpportunityCaseReceipt)
            or not isinstance(self.ranking_policy, BaccRankingPolicy)
            or any(not isinstance(row, ActionSelectionEvidence) for row in evidence)
            or len({row.action_id for row in evidence}) != len(evidence)
            or (center_id, case_id)
            != (self.opportunity_receipt.center_id, self.opportunity_receipt.case_id)
            or self.selection_decision.pairwise_model_hash != self.pairwise_model.model_hash
            or self.selection_decision.candidate_pool_receipt_hash
            != self.candidate_pool.receipt_hash
            or self.selection_decision.uncertainty_calibration_hash
            != self.uncertainty_calibration.calibration_hash
            or self.selection_decision.opportunity_case_receipt_hash
            != self.opportunity_receipt.receipt_hash
            or self.selection_decision.bacc_ranking_policy_hash
            != self.ranking_policy.policy_hash
            or self.pairwise_model.candidate_action_ids
            != self.opportunity_receipt.candidate_action_ids
            or tuple(row.action_id for row in evidence)
            != self.selection_decision.opportunity_active_representative_ids
            or (
                self.selection_decision.selected_action_id != P_ACTION_ID
                and self.selection_decision.selected_action_id
                not in {row.action_id for row in evidence}
            )
        ):
            raise ProtocolError("Admission decision receipt is inconsistent with its candidate inventory.")
        for row in evidence:
            member = self.opportunity_receipt.opportunity.member(row.action_id)
            if (
                row.pairwise_model_hash != self.pairwise_model.model_hash
                or row.opportunity_case_receipt_hash != self.opportunity_receipt.receipt_hash
                or row.utility.action_id != row.action_id
                or row.utility.baseline_probability_hash
                != self.opportunity_receipt.opportunity.baseline_hash
                or row.utility.candidate_probability_hash != member.probability_hash
                or member.representative_action_id != row.action_id
                or (member.family, member.direction) != (row.family, row.direction)
                or not math.isclose(
                    row.ranking_score,
                    _recompute_action_score(self.pairwise_model, row),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            ):
                raise ProtocolError(
                    "Admission evidence drifted from its frozen model or opportunity surface."
                )
        from ..selection import select_fail_closed_action

        recomputed = select_fail_closed_action(
            evidence,
            candidate_pool=self.candidate_pool,
            pairwise_model=self.pairwise_model,
            uncertainty_calibration=self.uncertainty_calibration,
            opportunity_receipt=self.opportunity_receipt,
            ranking_policy=self.ranking_policy,
        )
        if recomputed != self.selection_decision:
            raise ProtocolError("Admission selection decision drifted from its sealed evidence.")
        object.__setattr__(self, "center_id", center_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "candidate_evidence", evidence)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_sha256(
                {
                    "schema": "prelabel_admission_decision_receipt_v3",
                    "center_id": self.center_id,
                    "case_id": self.case_id,
                    "selection_decision_hash": self.selection_decision.decision_hash,
                    "candidate_evidence_hashes": tuple(row.evidence_hash for row in evidence),
                    "candidate_prediction_inventory": self.candidate_inventory,
                    "candidate_pool_receipt_hash": self.selection_decision.candidate_pool_receipt_hash,
                    "pairwise_model_hash": self.selection_decision.pairwise_model_hash,
                    "uncertainty_calibration_hash": self.selection_decision.uncertainty_calibration_hash,
                    "opportunity_case_receipt_hash": self.opportunity_receipt.receipt_hash,
                    "bacc_ranking_policy_hash": self.ranking_policy.policy_hash,
                    "terminal_labels_opened": False,
                }
            ),
        )

    @property
    def candidate_inventory(self) -> tuple[tuple[str, str, float], ...]:
        return tuple(
            (row.action_id, row.utility.candidate_probability_hash, row.ranking_score)
            for row in self.candidate_evidence
        )


@dataclass(frozen=True, slots=True)
class AdmissionCase:
    """Unique-surface source-OOF case used for non-vacuous admission."""

    center_id: str
    case_id: str
    candidates: tuple[AdmissionCandidate, ...]
    decision_receipt: AdmissionDecisionReceipt

    def __post_init__(self) -> None:
        candidates = tuple(sorted(self.candidates, key=lambda row: row.action_id))
        if (
            len({row.action_id for row in candidates}) != len(candidates)
            or len({row.surface_hash for row in candidates}) != len(candidates)
            or not isinstance(self.decision_receipt, AdmissionDecisionReceipt)
            or self.decision_receipt.center_id != self.center_id
            or self.decision_receipt.case_id != self.case_id
            or tuple(row.selection_evidence for row in candidates)
            != self.decision_receipt.candidate_evidence
        ):
            raise ProtocolError("Admission requires the exact sealed candidate evidence.")
        object.__setattr__(self, "center_id", _text(self.center_id, role="admission center"))
        object.__setattr__(self, "case_id", _text(self.case_id, role="admission case"))
        object.__setattr__(self, "candidates", candidates)

    @property
    def selected_action_id(self) -> str:
        return self.decision_receipt.selection_decision.selected_action_id


@dataclass(frozen=True, slots=True)
class AdmissionThresholds:
    min_center_count: int = 6
    min_unique_active_cases: int = 24
    min_pairwise_comparisons: int = 72
    min_sign_accuracy: float = 0.55
    min_unique_surface_top1_accuracy: float = 0.40
    min_rank_lower_bound: float = 0.0
    min_safe_coverage: float = 0.05
    min_worst_center_sign_accuracy: float = 0.50
    max_harmful_selected: int = 0
    max_proper_loss_violations: int = 0

    def __post_init__(self) -> None:
        if (
            self.min_center_count < 6
            or self.min_unique_active_cases <= 0
            or self.min_pairwise_comparisons <= 0
            or not 0.5 < self.min_sign_accuracy <= 1.0
            or not 0.0 < self.min_unique_surface_top1_accuracy <= 1.0
            or self.min_rank_lower_bound < -1.0
            or not 0.0 < self.min_safe_coverage <= 1.0
            or not 0.0 <= self.min_worst_center_sign_accuracy <= 1.0
            or self.max_harmful_selected < 0
            or self.max_proper_loss_violations < 0
        ):
            raise ProtocolError("Admission thresholds are invalid or vacuous.")


DEFAULT_ADMISSION_THRESHOLDS = AdmissionThresholds()


@dataclass(frozen=True, slots=True)
class CenterAdmission:
    center_id: str
    passed: bool
    case_count: int
    selected_count: int
    sign_accuracy: float
    pairwise_tau_b: float
    unique_surface_top1_accuracy: float
    harmful_selected_count: int
    proper_loss_violation_count: int
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        metrics = (self.sign_accuracy, self.pairwise_tau_b, self.unique_surface_top1_accuracy)
        if not all(math.isfinite(float(value)) for value in metrics):
            raise ProtocolError("Center admission metrics are non-finite.")
        if bool(self.passed) == bool(self.reasons):
            raise ProtocolError("Passing centers have no reasons; failed centers require reasons.")
        object.__setattr__(self, "center_id", _text(self.center_id, role="center admission id"))
        object.__setattr__(self, "reasons", tuple(self.reasons))


@dataclass(frozen=True, slots=True)
class AdmissionReport:
    """Source-only admission report without raw labels or row probabilities."""

    passed: bool
    center_results: tuple[CenterAdmission, ...]
    admitted_center_ids: tuple[str, ...]
    sealed_to_p_center_ids: tuple[str, ...]
    case_count: int
    unique_active_case_count: int
    pairwise_comparison_count: int
    selected_count: int
    sign_accuracy: float
    pairwise_tau_b: float
    minimum_delete_center_tau_b: float
    unique_surface_top1_accuracy: float
    safe_coverage: float
    harmful_selected_count: int
    proper_loss_violation_count: int
    reasons: tuple[str, ...]
    report_hash: str = field(init=False)

    def __post_init__(self) -> None:
        centers = tuple(self.center_results)
        center_ids = tuple(row.center_id for row in centers)
        admitted = tuple(self.admitted_center_ids)
        sealed = tuple(self.sealed_to_p_center_ids)
        metrics = (
            self.sign_accuracy,
            self.pairwise_tau_b,
            self.minimum_delete_center_tau_b,
            self.unique_surface_top1_accuracy,
            self.safe_coverage,
        )
        if (
            not centers
            or tuple(sorted(center_ids)) != center_ids
            or set(admitted).intersection(sealed)
            or tuple(sorted((*admitted, *sealed))) != center_ids
            or bool(self.passed) != bool(admitted)
            or not all(math.isfinite(float(value)) for value in metrics)
        ):
            raise ProtocolError("Admission report contract is invalid.")
        object.__setattr__(self, "center_results", centers)
        object.__setattr__(self, "admitted_center_ids", admitted)
        object.__setattr__(self, "sealed_to_p_center_ids", sealed)
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(
            self,
            "report_hash",
            canonical_sha256(
                {
                    "schema": "pairwise_source_only_admission_report_v3",
                    "passed": self.passed,
                    "center_results": tuple(
                        (
                            row.center_id, row.passed, row.case_count, row.selected_count,
                            row.sign_accuracy, row.pairwise_tau_b,
                            row.unique_surface_top1_accuracy, row.harmful_selected_count,
                            row.proper_loss_violation_count, row.reasons,
                        )
                        for row in centers
                    ),
                    "admitted_centers": admitted,
                    "sealed_to_P_centers": sealed,
                    "counts": (
                        self.case_count, self.unique_active_case_count,
                        self.pairwise_comparison_count, self.selected_count,
                    ),
                    "metrics": metrics,
                    "harmful_selected": self.harmful_selected_count,
                    "proper_loss_violations": self.proper_loss_violation_count,
                    "reasons": self.reasons,
                    "raw_labels_persisted": False,
                }
            ),
        )


__all__ = (
    "AdmissionCandidate", "AdmissionCase", "AdmissionDecisionReceipt",
    "AdmissionReport", "AdmissionThresholds", "CenterAdmission",
    "DEFAULT_ADMISSION_THRESHOLDS",
)
