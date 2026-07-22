from __future__ import annotations

import pytest

from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.reporting import (
    build_decision_summary,
    render_decision_report,
)


@pytest.mark.parametrize(
    ("deltas", "expected"),
    (
        ((0.01,) * 9, "PASS_DIAGNOSTIC_ONLY"),
        (
            (-0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
            "WEAK_PASS_DIAGNOSTIC_ONLY",
        ),
        ((0.0,) * 9, "NEGATIVE_RESULT_DIAGNOSTIC_ONLY"),
    ),
)
def test_decision_gate_distinguishes_pass_weak_pass_and_negative(
    deltas: tuple[float, ...],
    expected: str,
) -> None:
    outer_results = []
    comparisons = []
    gamma_summaries = []
    for index, delta in enumerate(deltas):
        center = str(index)
        baseline = 0.5
        selected = baseline + delta
        outer_results.extend(
            (
                {
                    "heldout_center": center,
                    "evaluation_role": "selected",
                    "heldout_bacc": selected,
                },
                {
                    "heldout_center": center,
                    "evaluation_role": "gamma0",
                    "heldout_bacc": baseline,
                },
            )
        )
        comparisons.append(
            {
                "heldout_center": center,
                "delta_bacc": delta,
                "delta_macro_f1": delta,
            }
        )
        gamma_summaries.append(
            {
                "heldout_center": center,
                "gamma": 0.1,
                "selected": "true",
            }
        )

    summary = build_decision_summary(
        outer_results,
        comparisons,
        gamma_summaries,
        design_hash="design",
        table_bundle_hash="tables",
        protocol_hash="protocol",
    )

    assert summary["decision"] == expected
    assert summary["minimum_selected_bacc"] == pytest.approx(0.49 if min(deltas) < 0 else 0.5 + min(deltas))
    if expected == "WEAK_PASS_DIAGNOSTIC_ONLY":
        assert summary["positive_mean_delta_passed"] is True
        assert summary["nonworse_minimum_center_passed"] is False
    report = render_decision_report(summary)
    assert expected in report
    assert "minimum selected-center BACC" in report
    assert "minimum gamma-0-center BACC" in report


def test_nonnegative_center_gate_does_not_apply_numerical_epsilon() -> None:
    deltas = (0.01, 0.01, 0.01, 0.01, -5.0e-13, -0.002, -0.002, -0.002, -0.002)
    outer_results = []
    comparisons = []
    gamma_summaries = []
    for index, delta in enumerate(deltas):
        center = str(index)
        baseline = 0.4 if index < 4 else 0.8
        outer_results.extend(
            (
                {
                    "heldout_center": center,
                    "evaluation_role": "selected",
                    "heldout_bacc": baseline + delta,
                },
                {
                    "heldout_center": center,
                    "evaluation_role": "gamma0",
                    "heldout_bacc": baseline,
                },
            )
        )
        comparisons.append(
            {
                "heldout_center": center,
                "delta_bacc": delta,
                "delta_macro_f1": delta,
            }
        )
        gamma_summaries.append(
            {"heldout_center": center, "gamma": 0.1, "selected": "true"}
        )

    summary = build_decision_summary(
        outer_results,
        comparisons,
        gamma_summaries,
        design_hash="design",
        table_bundle_hash="tables",
        protocol_hash="protocol",
    )

    assert summary["mean_delta_bacc"] > 1.0e-12
    assert summary["nonworse_minimum_center_passed"] is True
    assert summary["nonnegative_center_delta_count"] == 4
    assert summary["nonnegative_center_count_passed"] is False
    assert summary["decision"] == "WEAK_PASS_DIAGNOSTIC_ONLY"
