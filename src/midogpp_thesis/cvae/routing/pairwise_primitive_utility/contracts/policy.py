"""Stable facade for selection and admission/report contracts."""

from .admission import (
    AdmissionCandidate,
    AdmissionCase,
    AdmissionDecisionReceipt,
    AdmissionReport,
    AdmissionThresholds,
    CenterAdmission,
    DEFAULT_ADMISSION_THRESHOLDS,
)
from .selection import ActionSelectionEvidence, SelectionDecision


__all__ = (
    "ActionSelectionEvidence",
    "AdmissionCandidate",
    "AdmissionCase",
    "AdmissionDecisionReceipt",
    "AdmissionReport",
    "AdmissionThresholds",
    "CenterAdmission",
    "DEFAULT_ADMISSION_THRESHOLDS",
    "SelectionDecision",
)
