"""Pure top-level composition for the terminal case-aware audit."""

from __future__ import annotations

from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .case_features import build_case_aware_feature_surface
from .contracts import (
    DIAGNOSTIC_RESPONSE_NAMES,
    EXPERIMENT_ID,
    OUTER_INFERENCE_UNIT_COUNT,
    PRIMARY_RESPONSE_NAME,
    PUBLICATION_STATUS,
    CaseAwareFeatureSurface,
    CaseAwareProxyFeatureRow,
    CaseAwareProxyInformationAuditResult,
    CaseAwareResponseRow,
    CaseAwareResponseSurface,
)
from .crossfit import crossfit_proxy_families
from .metrics import summarize_crossfit
from .response_surfaces import build_response_surface


def run_case_aware_proxy_information_audit(
    feature_rows: CaseAwareFeatureSurface
    | Sequence[CaseAwareProxyFeatureRow | Mapping[str, object]],
    response_rows: CaseAwareResponseSurface
    | Sequence[CaseAwareResponseRow | Mapping[str, object]],
) -> CaseAwareProxyInformationAuditResult:
    """Build, cross-fit, summarize, and hash the pure scientific audit."""

    feature_surface = (
        feature_rows
        if isinstance(feature_rows, CaseAwareFeatureSurface)
        else build_case_aware_feature_surface(feature_rows)
    )
    response_surface = (
        response_rows
        if isinstance(response_rows, CaseAwareResponseSurface)
        else build_response_surface(feature_surface, response_rows)
    )
    crossfit = crossfit_proxy_families(feature_surface, response_surface)
    query_rows, outer_rows, summaries = summarize_crossfit(crossfit)
    informative = tuple(
        row.family_id
        for row in summaries
        if row.response_name == PRIMARY_RESPONSE_NAME and row.screening_passed
    )
    if any(
        row.response_name != PRIMARY_RESPONSE_NAME and row.screening_passed
        for row in summaries
    ):
        raise ProtocolError("A diagnostic smooth response entered the primary gate.")
    gate_passed = bool(informative)
    unhashed = {
        "schema_version": "midogpp_stage90_case_aware_proxy_audit_result_v1",
        "experiment_id": EXPERIMENT_ID,
        "crossfit_result_hash": crossfit.result_hash,
        "family_summary_row_hashes": [row.row_hash for row in summaries],
        "primary_proxy_information_gate_passed": gate_passed,
        "informative_family_ids": list(informative),
        "primary_response": PRIMARY_RESPONSE_NAME,
        "diagnostic_responses": list(DIAGNOSTIC_RESPONSE_NAMES),
        "outer_target_centers_are_inference_units": True,
        "outer_inference_unit_count": OUTER_INFERENCE_UNIT_COUNT,
        "consumed_test_data": True,
        "publication_status": PUBLICATION_STATUS,
        "terminal_diagnostic_only": True,
        "routing_quality_claimed": False,
        "policy_update_authorized": False,
        "stage60_feed_authorized": False,
        "stage70_feed_authorized": False,
        "target_actions_authorized": False,
        "promotion_eligible": False,
    }
    return CaseAwareProxyInformationAuditResult(
        crossfit=crossfit,
        query_metrics=query_rows,
        outer_metrics=outer_rows,
        family_summaries=summaries,
        primary_proxy_information_gate_passed=gate_passed,
        informative_family_ids=informative,
        result_hash=canonical_sha256(unhashed),
    )


run_proxy_information_audit = run_case_aware_proxy_information_audit


__all__ = (
    "run_case_aware_proxy_information_audit",
    "run_proxy_information_audit",
)
