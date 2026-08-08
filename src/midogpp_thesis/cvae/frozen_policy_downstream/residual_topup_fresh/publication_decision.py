"""Thesis-facing decision mapping for the two predeclared primary contrasts."""

from __future__ import annotations

from ...protocol import ProtocolError
from .contracts import FreshEvaluationReport


def build_publication_decision(
    report: FreshEvaluationReport,
) -> dict[str, object]:
    inferred = {row.contrast_id: row for row in report.contrast_inference}
    try:
        s_u = inferred["S-U"]
        s_g = inferred["S-G"]
    except KeyError as exc:
        raise ProtocolError(
            "Fresh Stage-70 publication contrasts are incomplete."
        ) from exc
    s_u_positive = s_u.mean_probability_ensemble_bacc_delta > 0.0
    s_g_positive = s_g.mean_probability_ensemble_bacc_delta > 0.0
    if s_u_positive and s_g_positive:
        decision = "TARGET_SPECIFIC_ROUTING_MECHANISM_SUPPORTED"
    elif s_u_positive:
        decision = "GLOBAL_SOURCE_PRIOR_ONLY_NOT_TARGET_SPECIFIC"
    else:
        decision = "TARGET_SPECIFIC_ROUTING_NOT_SUPPORTED"
    return {
        "schema_version": "midogpp_residual_topup_fresh_publication_decision_v1",
        "status": "COMPLETE",
        "decision": decision,
        "claim_scope": "synthetic_downstream_utility",
        "primary_endpoint": report.primary_endpoint,
        "inference_unit": "target_center",
        "effective_sample_size": 9,
        "S_minus_U_mean": s_u.mean_probability_ensemble_bacc_delta,
        "S_minus_G_mean": s_g.mean_probability_ensemble_bacc_delta,
        "S_minus_U_one_sided_95_lcb": s_u.one_sided_95_lcb,
        "S_minus_G_one_sided_95_lcb": s_g.one_sided_95_lcb,
        "success_requires_positive_S_minus_U_and_S_minus_G": True,
        "success_gate_passed": s_u_positive and s_g_positive,
        "negative_result_valid": True,
        "policy_update_emitted": False,
        "oracle_action_exported": False,
    }


__all__ = ("build_publication_decision",)
