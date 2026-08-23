from __future__ import annotations

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.identity import (
    METHOD_MENU,
    P_METHOD_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.terminal import (
    exact_shared_center_max_sign_flip,
    midrank_spearman,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.validation_terminal import (
    validate_terminal_row_inventory,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError


def test_v2_midrank_spearman_is_tie_aware_and_fail_closed() -> None:
    assert midrank_spearman((1.0, 1.0, 2.0), (3.0, 3.0, 1.0)) == pytest.approx(-1.0)
    assert midrank_spearman((1.0,), (1.0,)) is None
    assert midrank_spearman((1.0, 1.0), (2.0, 3.0)) is None


def test_v2_sign_flip_enumerates_all_shared_center_patterns() -> None:
    metrics = {
        method: {
            center: {
                "center_bacc": 0.5
                if method == P_METHOD_ID
                else 0.5 + (0.01 * (METHOD_MENU.index(method)))
            }
            for center in CENTERS
        }
        for method in METHOD_MENU
    }
    result = exact_shared_center_max_sign_flip(metrics)
    assert result["null_replicate_count"] == 512
    assert result["center_signs_shared_across_methods"] is True
    assert result["route_pipeline_refit_inside_null_replicate"] is False
    assert result["observed_selected_method_id"] == METHOD_MENU[-1]
    assert 0.0 < result["selection_aware_descriptive_randomization_p_value"] <= 1.0


def _canonical_terminal_rows():
    case_ids_by_center = {
        center: tuple(
            f"case-{center}-{ordinal:03d}"
            for ordinal in range(26 if index == 0 else 24)
        )
        for index, center in enumerate(CENTERS)
    }
    ordered_cases = tuple(
        (center, case_id)
        for center in CENTERS
        for case_id in case_ids_by_center[center]
    )
    sample_count_by_case = {
        key: 46 if ordinal < 118 else 45
        for ordinal, key in enumerate(ordered_cases)
    }
    method_rows = tuple(
        {
            "method_id": method,
            "equal_center_bacc": 1.0,
            "sample_pooled_bacc": 1.0,
            "global_brier": 0.0,
            "equal_center_brier": 0.0,
            "global_log_loss": 0.0,
            "equal_center_log_loss": 0.0,
            "mean_center_bacc_delta_vs_P": 0.0,
            "minimum_center_bacc_delta_vs_P": 0.0,
            "maximum_center_bacc_delta_vs_P": 0.0,
            "mean_center_brier_delta_vs_P": 0.0,
            "mean_center_log_loss_delta_vs_P": 0.0,
            "positive_center_count": 0,
            "negative_center_count": 0,
            "zero_center_count": len(CENTERS),
            "descriptive_t8_lower": 0.0,
            "descriptive_t8_upper": 0.0,
            "descriptive_interval_has_no_nominal_coverage_claim": True,
            "route_count": 0,
            "case_harm_count": 0,
            "case_harm_rate": 0.0,
            "formal_claim_authorized": False,
        }
        for method in METHOD_MENU
    )
    center_rows = tuple(
        {
            "method_id": method,
            "target_center": center,
            "reference_method": P_METHOD_ID,
            "case_count": len(case_ids_by_center[center]),
            "sample_count": sum(
                sample_count_by_case[(center, case_id)]
                for case_id in case_ids_by_center[center]
            ),
            "n_positive": 1,
            "n_negative": sum(
                sample_count_by_case[(center, case_id)]
                for case_id in case_ids_by_center[center]
            )
            - 1,
            "true_positive": 1,
            "true_negative": sum(
                sample_count_by_case[(center, case_id)]
                for case_id in case_ids_by_center[center]
            )
            - 1,
            "false_positive": 0,
            "false_negative": 0,
            "changed_case_count": 0,
            "center_bacc": 1.0,
            "center_brier": 0.0,
            "center_log_loss": 0.0,
            "threshold_switch_count": 0,
            "helpful_threshold_switch_count": 0,
            "harmful_threshold_switch_count": 0,
            "squared_error_sum": 0.0,
            "log_loss_sum": 0.0,
            "center_bacc_delta_vs_P": 0.0,
            "center_brier_delta_vs_P": 0.0,
            "center_log_loss_delta_vs_P": 0.0,
            "formal_claim_authorized": False,
        }
        for method in METHOD_MENU
        for center in CENTERS
    )
    case_rows = tuple(
        {
            "target_center": center,
            "method_id": method,
            "case_id": case_id,
            "sample_count": sample_count_by_case[(center, case_id)],
            "probability_changed_vs_P": False,
            "threshold_error_delta_vs_P": 0,
            "case_harmed_vs_P": False,
            "raw_labels_persisted": False,
            "formal_claim_authorized": False,
        }
        for center in CENTERS
        for method in METHOD_MENU
        for case_id in case_ids_by_center[center]
    )
    case_sample_counts_by_center = {
        center: {
            case_id: sample_count_by_case[(center, case_id)]
            for case_id in case_ids_by_center[center]
        }
        for center in CENTERS
    }
    return (
        method_rows,
        center_rows,
        case_rows,
        case_ids_by_center,
        case_sample_counts_by_center,
    )


def test_v2_terminal_inventory_requires_exact_canonical_rectangles() -> None:
    (
        method_rows,
        center_rows,
        case_rows,
        case_ids_by_center,
        case_sample_counts_by_center,
    ) = _canonical_terminal_rows()
    checks = validate_terminal_row_inventory(
        method_rows=method_rows,
        center_rows=center_rows,
        case_rows=case_rows,
        case_ids_by_center=case_ids_by_center,
        case_sample_counts_by_center=case_sample_counts_by_center,
    )
    assert checks == {
        "method_count": 6,
        "center_method_count": 54,
        "case_diagnostic_count": 1308,
        "canonical_case_count": 218,
        "canonical_row_count": 9928,
    }

    with pytest.raises(ProtocolError, match="case rectangle drifted"):
        validate_terminal_row_inventory(
            method_rows=method_rows,
            center_rows=center_rows,
            case_rows=case_rows[:-1],
            case_ids_by_center=case_ids_by_center,
            case_sample_counts_by_center=case_sample_counts_by_center,
        )

    promoted_methods = tuple(dict(row) for row in method_rows)
    promoted_methods[0]["formal_claim_authorized"] = True
    with pytest.raises(ProtocolError, match="method semantics drifted"):
        validate_terminal_row_inventory(
            method_rows=promoted_methods,
            center_rows=center_rows,
            case_rows=case_rows,
            case_ids_by_center=case_ids_by_center,
            case_sample_counts_by_center=case_sample_counts_by_center,
        )

    redistributed_cases = tuple(dict(row) for row in case_rows)
    redistributed_cases[0]["sample_count"] += 1
    redistributed_cases[-1]["sample_count"] -= 1
    with pytest.raises(ProtocolError, match="case semantics drifted"):
        validate_terminal_row_inventory(
            method_rows=method_rows,
            center_rows=center_rows,
            case_rows=redistributed_cases,
            case_ids_by_center=case_ids_by_center,
            case_sample_counts_by_center=case_sample_counts_by_center,
        )
