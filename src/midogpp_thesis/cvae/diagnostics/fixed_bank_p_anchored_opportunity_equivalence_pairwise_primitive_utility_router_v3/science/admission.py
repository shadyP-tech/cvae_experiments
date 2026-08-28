"""Outer source-ordering admission and exact-P fail-closed receipts."""

from __future__ import annotations

from dataclasses import dataclass, field
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


def _sha256(value: object, *, role: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ProtocolError(f"OE-PPUR v3 {role} is not a SHA-256 digest.")
    return result


@dataclass(frozen=True, slots=True)
class SourceOrderingAdmissionReceipt:
    outer_target_center: str
    report: AdmissionReport
    source_supervision_contract_hash: str
    admitted: bool = field(init=False)
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = str(self.outer_target_center)
        if h not in CENTERS or not isinstance(self.report, AdmissionReport):
            raise ProtocolError("OE-PPUR v3 source-ordering admission is untyped.")
        source_hash = _sha256(
            self.source_supervision_contract_hash,
            role="source-supervision contract hash",
        )
        admitted = bool(self.report.passed)
        object.__setattr__(self, "outer_target_center", h)
        object.__setattr__(self, "source_supervision_contract_hash", source_hash)
        object.__setattr__(self, "admitted", admitted)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_source_ordering_admission_v1",
                    "H": h,
                    "source_supervision_contract_hash": source_hash,
                    "neutral_admission_report_hash": self.report.report_hash,
                    "admitted": admitted,
                    "sealed_to_P_source_centers": self.report.sealed_to_p_center_ids,
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
            raise ProtocolError("OE-PPUR v3 exact-P fallback contract drifted.")
        evidence_hash = _sha256(self.evidence_hash, role="fallback evidence hash")
        object.__setattr__(self, "outer_target_center", h)
        object.__setattr__(self, "evidence_hash", evidence_hash)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_exact_P_fallback_v1",
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
    "evaluate_source_ordering_admission",
    "exact_p_fail_closed_reason",
)
