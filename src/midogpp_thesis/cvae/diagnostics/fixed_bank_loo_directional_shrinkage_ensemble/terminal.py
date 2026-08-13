"""Thin terminal orchestrator for sealed DCSE science products."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from ...protocol import ProtocolError
from .constants import (
    B_ACTION_ID,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    METHOD_IDS,
    PRE_TERMINAL_METHOD_IDS,
    TERMINAL_DECISION,
    U_ACTION_ID,
)
from .ensemble import DESCRIPTIVE_METHOD_IDS
from .hashing import canonical_hash, json_native
from .loo_plans import WholeCaseLooPlan
from .nulls import CandidateIdentityNullPlan
from .products import (
    BinaryLabel,
    CaseActionConfusion,
    CaseArmDecision,
    DonorPrior,
    MethodPrediction,
)
from .terminal_null import evaluate_candidate_identity_null
from .terminal_scoring import (
    TERMINAL_REPORTED_METHOD_IDS,
    center_metrics,
    equal_center_contrast,
    method_metric_rows,
    method_role,
    score_terminal_predictions,
    terminal_oracle_predictions,
    terminal_truth,
)
from .terminal_sensitivity import (
    delete_one_center_rows,
    descriptive_inference,
    leave_one_arm_rows,
    support_gains_by_route,
)


def evaluate_terminal(
    *,
    probability_surface: object,
    plans: Sequence[WholeCaseLooPlan],
    donor_counts: Sequence[CaseActionConfusion],
    case_action_confusions: Sequence[CaseActionConfusion],
    donor_priors: Sequence[DonorPrior],
    arm_decisions: Sequence[CaseArmDecision],
    method_predictions: Sequence[MethodPrediction],
    descriptive_predictions: Sequence[MethodPrediction],
    terminal_labels: Sequence[BinaryLabel] | Sequence[object],
    config: object,
    null_plan: CandidateIdentityNullPlan,
    aggregate_plan_decision_seal_hash: str,
) -> dict[str, object]:
    """Evaluate sealed DCSE products after the terminal capability opens."""

    del donor_counts, config
    routes = tuple(plans)
    labels = tuple(terminal_labels)
    predictions = tuple(method_predictions)
    descriptive = tuple(descriptive_predictions)
    decisions = tuple(arm_decisions)
    priors = tuple(donor_priors)
    aggregate_hash = str(aggregate_plan_decision_seal_hash)
    if len(aggregate_hash) != 64 or any(
        char not in "0123456789abcdef" for char in aggregate_hash
    ):
        raise ProtocolError(
            "DCSE terminal evaluator lacks the persisted aggregate seal hash."
        )
    if (
        len(routes) != EXPECTED_TOTAL_CASE_COUNT
        or len({plan.key for plan in routes}) != EXPECTED_TOTAL_CASE_COUNT
        or len(terminal_truth(labels)) != EXPECTED_TEST_ROW_COUNT
        or tuple(plan.key for plan in routes) != null_plan.route_keys
    ):
        raise ProtocolError(
            "DCSE terminal evaluator requires the sealed 218-case universe."
        )
    observed_preterminal_methods = tuple(
        dict.fromkeys(row.method_id for row in predictions)
    )
    if observed_preterminal_methods != PRE_TERMINAL_METHOD_IDS:
        raise ProtocolError(
            "DCSE terminal labels opened before all six pre-terminal method "
            "predictions were sealed."
        )
    observed_descriptive_methods = tuple(
        dict.fromkeys(row.method_id for row in descriptive)
    )
    if observed_descriptive_methods != DESCRIPTIVE_METHOD_IDS:
        raise ProtocolError(
            "DCSE terminal labels opened before all five descriptive prediction "
            "controls were sealed."
        )

    oracle_predictions = terminal_oracle_predictions(
        probability_surface, labels
    )
    confusions = score_terminal_predictions(
        (*predictions, *oracle_predictions, *descriptive),
        labels,
        require_methods=TERMINAL_REPORTED_METHOD_IDS,
    )
    metric_rows = center_metrics(confusions, TERMINAL_REPORTED_METHOD_IDS)
    aggregate_metrics = method_metric_rows(
        metric_rows, TERMINAL_REPORTED_METHOD_IDS
    )
    contrasts = tuple(
        equal_center_contrast(
            metric_rows,
            method_id="DCSE_LOO",
            reference_id=reference,
        )
        for reference in (B_ACTION_ID, U_ACTION_ID, "G_directional_matched")
    )
    support_gains = support_gains_by_route(
        routes, case_action_confusions
    )
    delete_rows = delete_one_center_rows(
        probability_surface=probability_surface,
        plans=routes,
        support_gains=support_gains,
        donor_priors=priors,
        terminal_labels=labels,
        preterminal_confusions=confusions,
    )
    baseline_confusions = tuple(
        row for row in confusions if row.method_id == B_ACTION_ID
    )
    leave_rows = leave_one_arm_rows(
        probability_surface=probability_surface,
        arm_decisions=decisions,
        terminal_labels=labels,
        baseline_confusions=baseline_confusions,
    )
    primary = contrasts[0]
    null_rows = evaluate_candidate_identity_null(
        plan=null_plan,
        plans=routes,
        probability_surface=probability_surface,
        case_action_confusions=case_action_confusions,
        donor_priors=priors,
        terminal_labels=labels,
        observed_statistic=primary.estimate,
    )
    inference = descriptive_inference(
        primary=primary, delete_one_center=delete_rows
    )
    center_b_deltas = tuple(
        Fraction(numerator, denominator)
        for _center, numerator, denominator in primary.center_differences
    )
    delete_b = tuple(
        row for row in delete_rows if row["reference_id"] == B_ACTION_ID
    )
    delete_u = tuple(
        row for row in delete_rows if row["reference_id"] == U_ACTION_ID
    )
    rubric = {
        "full_DCSE_LOO_minus_B_strictly_positive": contrasts[0].exact > 0,
        "full_DCSE_LOO_minus_U_strictly_positive": contrasts[1].exact > 0,
        "both_primary_B_and_U_contrasts_positive_in_all_nine_whole_pipeline_center_deletions": all(
            Fraction(*row["exact_fraction"]) > 0
            for row in (*delete_b, *delete_u)
        ),
        "at_least_eight_of_nine_center_DCSE_LOO_minus_B_deltas_nonnegative": sum(
            value >= 0 for value in center_b_deltas
        )
        >= 8,
        "every_leave_one_arm_DCSE_LOO_minus_B_contrast_strictly_positive": all(
            Fraction(*row["exact_fraction"]) > 0 for row in leave_rows
        ),
    }
    unhashed_seal = {
        "schema_version": "fixed_bank_dcse_terminal_evaluation_seal_v1",
        "bindings": {
            "probability_surface_hash": str(
                getattr(probability_surface, "surface_hash")
            ),
            "ordered_loo_plan_hashes_hash": canonical_hash(
                [plan.plan_hash for plan in routes]
            ),
            "ordered_donor_prior_hashes_hash": canonical_hash(
                [row.prior_hash for row in priors]
            ),
            "ordered_arm_decision_hashes_hash": canonical_hash(
                [row.decision_hash for row in decisions]
            ),
            "ordered_preterminal_prediction_hashes_hash": canonical_hash(
                [row.probability_hash for row in predictions]
            ),
            "ordered_descriptive_prediction_hashes_hash": canonical_hash(
                [row.probability_hash for row in descriptive]
            ),
            "candidate_identity_null_plan_hash": null_plan.plan_hash,
            "aggregate_plan_decision_seal_hash": aggregate_hash,
        },
        "bound_product_counts": {
            "loo_plans": len(routes),
            "donor_priors": len(priors),
            "arm_decisions": len(decisions),
            "preterminal_predictions": len(predictions),
            "descriptive_predictions": len(descriptive),
        },
        "bound_preterminal_method_order": list(PRE_TERMINAL_METHOD_IDS),
        "bound_descriptive_method_order": list(DESCRIPTIVE_METHOD_IDS),
        "canonical_method_ids": list(METHOD_IDS),
        "reported_method_ids": list(TERMINAL_REPORTED_METHOD_IDS),
        "descriptive_control_method_ids": list(DESCRIPTIVE_METHOD_IDS),
        "primary_contrasts": [row.contrast_id for row in contrasts],
        "descriptive_success_rubric": rubric,
        "terminal_labels_used_only_for_scoring_oracles_and_descriptive_sensitivities": True,
        "terminal_labels_used_to_train_tune_rank_or_select_preterminal_methods": False,
        "matched_G_contrast_is_gate": False,
        "nominal_t_interval_is_gate": False,
        "jackknife_interval_is_gate": False,
        "null_summary_is_gate": False,
        "candidate_identity_null_exchangeability_claimed": False,
        "candidate_identity_null_p_value": None,
        "terminal_decision": TERMINAL_DECISION,
    }
    terminal_seal = {
        **unhashed_seal,
        "seal_hash": canonical_hash(unhashed_seal),
    }
    result = {
        "case_confusions": [
            {
                **row.to_payload(),
                "method_role": method_role(row.method_id),
                "success_gate_eligible": row.method_id
                in {"B", "U", "DCSE_LOO", "G_directional_matched"},
            }
            for row in confusions
        ],
        "method_metrics": list(aggregate_metrics),
        "center_metrics": [
            {
                **row.to_payload(),
                "method_role": method_role(row.method_id),
                "success_gate_eligible": row.method_id
                in {"B", "U", "DCSE_LOO", "G_directional_matched"},
            }
            for row in metric_rows
        ],
        "equal_center_contrasts": [row.to_payload() for row in contrasts],
        "delete_one_center": list(delete_rows),
        "leave_one_arm": list(leave_rows),
        "null_statistics": list(null_rows),
        "descriptive_inference": inference,
        "terminal_seal": terminal_seal,
    }
    converted = json_native(result)
    if not isinstance(converted, dict):  # pragma: no cover
        raise ProtocolError("DCSE terminal result is not JSON-native.")
    return converted


__all__ = (
    "TERMINAL_REPORTED_METHOD_IDS",
    "evaluate_candidate_identity_null",
    "evaluate_terminal",
    "score_terminal_predictions",
)
