"""Outer source-ordering admission and exact-P fail-closed receipts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from ....protocol import ProtocolError
from ....routing.pairwise_primitive_utility import (
    AdmissionCase,
    AdmissionReport,
    P_ACTION_ID,
    canonical_sha256,
    evaluate_source_only_admission,
)
from ..identity import CENTERS


FALLBACK_REASON_CODES = (
    "protocol_lineage_failure",
    "source_model_unavailable",
    "source_ordering_admission_failed",
    "uncertainty_surface_incomplete",
)


@dataclass(frozen=True, slots=True)
class HeldLSourceOrderingCase:
    """One genuinely held-L source case, before any target labels are opened."""

    center_id: str
    case_id: str
    predicted_scores: tuple[tuple[str, float], ...]
    realized_bacc_gains: tuple[tuple[str, float], ...]
    active_representative_ids: tuple[str, ...]
    candidate_pool_receipt_hash: str
    held_l_scope_receipt_hash: str
    held_l_model_hash: str

    def __post_init__(self) -> None:
        predicted = tuple(sorted((str(action), float(value)) for action, value in self.predicted_scores))
        realized = tuple(sorted((str(action), float(value)) for action, value in self.realized_bacc_gains))
        active = tuple(sorted(str(action) for action in self.active_representative_ids))
        if (
            not self.center_id
            or not self.case_id
            or tuple(action for action, _ in predicted) != tuple(action for action, _ in realized)
            or not all(math.isfinite(value) for _action, value in (*predicted, *realized))
            or tuple(action for action, _ in predicted) != active
        ):
            raise ProtocolError("OE-PPUR v4 held-L ordering case drifted.")
        object.__setattr__(self, "predicted_scores", predicted)
        object.__setattr__(self, "realized_bacc_gains", realized)
        object.__setattr__(self, "active_representative_ids", active)
        for name in ("candidate_pool_receipt_hash", "held_l_scope_receipt_hash", "held_l_model_hash"):
            object.__setattr__(self, name, _sha256(getattr(self, name), role=name))


@dataclass(frozen=True, slots=True)
class SourceCenterOrderingResult:
    center_id: str
    passed: bool
    case_count: int
    active_case_count: int
    sign_accuracy: float
    pairwise_tau_b: float
    top1_accuracy: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.center_id not in CENTERS
            or self.case_count <= 0
            or self.active_case_count <= 0
            or not all(math.isfinite(float(value)) for value in (self.sign_accuracy, self.pairwise_tau_b, self.top1_accuracy))
            or bool(self.passed) == bool(self.reasons)
        ):
            raise ProtocolError("OE-PPUR v4 source-center ordering result drifted.")


@dataclass(frozen=True, slots=True)
class SourceOrderingDiagnosticReport:
    outer_target_center: str
    center_results: tuple[SourceCenterOrderingResult, ...]
    case_count: int
    sign_accuracy: float
    minimum_center_tau_b: float
    top1_accuracy: float
    uncertainty_calibration_hash: str
    source_case_inventory_hash: str
    passed: bool = field(init=False)
    report_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = str(self.outer_target_center)
        centers = tuple(self.center_results)
        expected = tuple(center for center in CENTERS if center != h)
        if (
            h not in CENTERS
            or tuple(row.center_id for row in centers) != expected
            or self.case_count != sum(row.case_count for row in centers)
            or not all(math.isfinite(float(value)) for value in (self.sign_accuracy, self.minimum_center_tau_b, self.top1_accuracy))
        ):
            raise ProtocolError("OE-PPUR v4 source ordering report topology drifted.")
        object.__setattr__(self, "uncertainty_calibration_hash", _sha256(self.uncertainty_calibration_hash, role="uncertainty calibration hash"))
        object.__setattr__(self, "source_case_inventory_hash", _sha256(self.source_case_inventory_hash, role="source case inventory hash"))
        passed = all(row.passed for row in centers) and self.sign_accuracy >= 0.55 and self.minimum_center_tau_b > 0.0 and self.top1_accuracy >= 0.40
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "report_hash", canonical_sha256({
            "schema": "oe_ppur_v4_genuine_held_L_source_ordering_report_v1",
            "H": h,
            "centers": tuple((row.center_id, row.passed, row.case_count, row.active_case_count, row.sign_accuracy, row.pairwise_tau_b, row.top1_accuracy, row.reasons) for row in centers),
            "metrics": (self.sign_accuracy, self.minimum_center_tau_b, self.top1_accuracy),
            "uncertainty_calibration_hash": self.uncertainty_calibration_hash,
            "source_case_inventory_hash": self.source_case_inventory_hash,
            "all_source_centers_required": True,
            "target_labels_used": False,
        }))


def _sha256(value: object, *, role: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ProtocolError(f"OE-PPUR v4 {role} is not a SHA-256 digest.")
    return result


@dataclass(frozen=True, slots=True)
class SourceOrderingAdmissionReceipt:
    outer_target_center: str
    report: AdmissionReport | SourceOrderingDiagnosticReport
    source_supervision_contract_hash: str
    admitted: bool = field(init=False)
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = str(self.outer_target_center)
        if h not in CENTERS or not isinstance(self.report, (AdmissionReport, SourceOrderingDiagnosticReport)):
            raise ProtocolError("OE-PPUR v4 source-ordering admission is untyped.")
        source_hash = _sha256(
            self.source_supervision_contract_hash,
            role="source-supervision contract hash",
        )
        expected_source_centers = tuple(center for center in CENTERS if center != h)
        reported_centers = tuple(row.center_id for row in self.report.center_results)
        neutral_complete = (
            isinstance(self.report, AdmissionReport)
            and self.report.admitted_center_ids == expected_source_centers
            and not self.report.sealed_to_p_center_ids
        )
        held_l_complete = isinstance(self.report, SourceOrderingDiagnosticReport)
        admitted = bool(
            self.report.passed
            and reported_centers == expected_source_centers
            and (neutral_complete or held_l_complete)
        )
        object.__setattr__(self, "outer_target_center", h)
        object.__setattr__(self, "source_supervision_contract_hash", source_hash)
        object.__setattr__(self, "admitted", admitted)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v4_source_ordering_admission_v1",
                    "H": h,
                    "source_supervision_contract_hash": source_hash,
                    "neutral_admission_report_hash": self.report.report_hash,
                    "admitted": admitted,
                    "required_source_centers": expected_source_centers,
                    "all_source_centers_required_for_unseen_H": True,
                    "sealed_to_P_source_centers": (
                        self.report.sealed_to_p_center_ids
                        if isinstance(self.report, AdmissionReport)
                        else tuple(row.center_id for row in self.report.center_results if not row.passed)
                    ),
                    "raw_source_labels_persisted": False,
                    "target_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ExactPFallbackReceipt:
    outer_target_center: str
    reason_code: str
    evidence_hash: str
    selected_action_id: str = P_ACTION_ID
    target_labels_used: bool = False
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = str(self.outer_target_center)
        if (
            h not in CENTERS
            or self.reason_code not in FALLBACK_REASON_CODES
            or self.selected_action_id != P_ACTION_ID
            or type(self.target_labels_used) is not bool
            or self.target_labels_used
        ):
            raise ProtocolError("OE-PPUR v4 exact-P fallback contract drifted.")
        evidence_hash = _sha256(self.evidence_hash, role="fallback evidence hash")
        object.__setattr__(self, "outer_target_center", h)
        object.__setattr__(self, "evidence_hash", evidence_hash)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v4_exact_P_fallback_v1",
                    "H": h,
                    "reason_code": self.reason_code,
                    "evidence_hash": evidence_hash,
                    "selected_action_id": P_ACTION_ID,
                    "target_labels_used": False,
                }
            ),
        )


def evaluate_source_ordering_admission(
    cases: Sequence[AdmissionCase],
    *,
    outer_target_center: object,
    source_supervision_contract_hash: object,
) -> SourceOrderingAdmissionReceipt:
    report = evaluate_source_only_admission(cases)
    return SourceOrderingAdmissionReceipt(
        outer_target_center=str(outer_target_center),
        report=report,
        source_supervision_contract_hash=str(source_supervision_contract_hash),
    )


def _tau_and_sign(rows: Sequence[HeldLSourceOrderingCase]) -> tuple[float, float, float]:
    concordant = discordant = predicted_ties = realized_ties = sign_ok = sign_total = top1 = active_cases = 0
    for case in rows:
        predicted = {"P_PROTECTED": 0.0, **dict(case.predicted_scores)}
        realized = {"P_PROTECTED": 0.0, **dict(case.realized_bacc_gains)}
        active = ("P_PROTECTED", *case.active_representative_ids)
        if len(active) > 1:
            active_cases += 1
            predicted_winner = min(active, key=lambda action: (-predicted[action], action))
            best = max(realized[action] for action in active)
            top1 += int(realized[predicted_winner] == best)
        for action in case.active_representative_ids:
            ps, rs = predicted[action], realized[action]
            if ps != 0.0 or rs != 0.0:
                sign_total += 1
                sign_ok += int(((ps > 0.0) - (ps < 0.0)) == ((rs > 0.0) - (rs < 0.0)))
        for left_index, left in enumerate(active):
            for right in active[left_index + 1:]:
                ps = (predicted[left] > predicted[right]) - (predicted[left] < predicted[right])
                rs = (realized[left] > realized[right]) - (realized[left] < realized[right])
                if ps == 0 and rs == 0:
                    continue
                if ps == 0:
                    predicted_ties += 1
                elif rs == 0:
                    realized_ties += 1
                elif ps == rs:
                    concordant += 1
                else:
                    discordant += 1
    denominator = math.sqrt((concordant + discordant + predicted_ties) * (concordant + discordant + realized_ties))
    tau = 0.0 if denominator == 0.0 else (concordant - discordant) / denominator
    return (sign_ok / sign_total if sign_total else 0.0, tau, top1 / active_cases if active_cases else 0.0)


def evaluate_genuine_held_l_source_ordering(
    cases: Sequence[HeldLSourceOrderingCase],
    *,
    outer_target_center: object,
    source_supervision_contract_hash: object,
    uncertainty_calibration_hash: object,
    source_case_inventory_hash: object,
) -> SourceOrderingAdmissionReceipt:
    h = str(outer_target_center)
    rows = tuple(sorted(tuple(cases), key=lambda row: (row.center_id, row.case_id)))
    expected_centers = tuple(center for center in CENTERS if center != h)
    if not rows or tuple(sorted({row.center_id for row in rows})) != expected_centers or len({(row.center_id, row.case_id) for row in rows}) != len(rows):
        raise ProtocolError("OE-PPUR v4 held-L source admission coverage drifted.")
    center_results = []
    for center in expected_centers:
        group = tuple(row for row in rows if row.center_id == center)
        sign, tau, top1 = _tau_and_sign(group)
        reasons = []
        active_count = sum(bool(row.active_representative_ids) for row in group)
        if active_count <= 0:
            reasons.append("center_zero_active_cases")
        if sign < 0.50:
            reasons.append("center_sign_accuracy_below_floor")
        if tau <= 0.0:
            reasons.append("center_pairwise_ordering_nonpositive")
        if top1 < 0.40:
            reasons.append("center_top1_below_floor")
        center_results.append(SourceCenterOrderingResult(center, not reasons, len(group), active_count, sign, tau, top1, tuple(reasons)))
    global_sign, _global_tau, global_top1 = _tau_and_sign(rows)
    report = SourceOrderingDiagnosticReport(
        outer_target_center=h,
        center_results=tuple(center_results),
        case_count=len(rows),
        sign_accuracy=global_sign,
        minimum_center_tau_b=min(row.pairwise_tau_b for row in center_results),
        top1_accuracy=global_top1,
        uncertainty_calibration_hash=str(uncertainty_calibration_hash),
        source_case_inventory_hash=str(source_case_inventory_hash),
    )
    return SourceOrderingAdmissionReceipt(
        outer_target_center=h,
        report=report,
        source_supervision_contract_hash=str(source_supervision_contract_hash),
    )


def exact_p_fail_closed_reason(
    *, outer_target_center: object, reason_code: object, evidence_hash: object
) -> ExactPFallbackReceipt:
    return ExactPFallbackReceipt(
        outer_target_center=str(outer_target_center),
        reason_code=str(reason_code),
        evidence_hash=str(evidence_hash),
    )


__all__ = (
    "ExactPFallbackReceipt",
    "FALLBACK_REASON_CODES",
    "SourceOrderingAdmissionReceipt",
    "HeldLSourceOrderingCase",
    "SourceCenterOrderingResult",
    "SourceOrderingDiagnosticReport",
    "evaluate_genuine_held_l_source_ordering",
    "evaluate_source_ordering_admission",
    "exact_p_fail_closed_reason",
)
