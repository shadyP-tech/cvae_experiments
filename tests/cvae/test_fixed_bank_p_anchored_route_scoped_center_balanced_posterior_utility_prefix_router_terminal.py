from __future__ import annotations

import math

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.constants import (
    CENTERS,
    COMPOSED_POLICY_IDS,
    ENDPOINT_METHOD_IDS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_TOTAL_CASE_COUNT,
    PORTFOLIO_METHOD_ID,
    PRIMARY_METHOD_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.contracts import (
    BinaryLabel,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.evaluation import (
    evaluate_terminal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.terminal_metrics import (
    selection_aware_center_sign_flip,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _center_metrics_for_known_max_test() -> dict[str, dict[str, dict[str, float]]]:
    # Plausible BACC values whose four delta vectors are respectively
    # +0.25, +0.50, 0.00, and -0.50 in every center.
    values = dict(zip(COMPOSED_POLICY_IDS, (0.75, 1.0, 0.5, 0.0), strict=True))
    rows = {
        PORTFOLIO_METHOD_ID: {
            center: {"center_bacc": 0.5} for center in CENTERS
        }
    }
    rows.update(
        {
            method: {
                center: {"center_bacc": value} for center in CENTERS
            }
            for method, value in values.items()
        }
    )
    return rows


def _terminal_fixture():
    method_order = (*ENDPOINT_METHOD_IDS, *COMPOSED_POLICY_IDS)
    sample_ids: dict[str, dict[str, tuple[str, ...]]] = {}
    p_by_center: dict[str, dict[str, tuple[float, ...]]] = {}
    labels: list[BinaryLabel] = []
    for center in CENTERS:
        center_samples: dict[str, tuple[str, ...]] = {}
        center_p: dict[str, tuple[float, ...]] = {}
        for index in range(EXPECTED_CASE_COUNTS_BY_CENTER[center]):
            case_id = f"case-{center}-{index:03d}"
            sample_id = f"sample-{center}-{index:03d}"
            value = index % 2
            center_samples[case_id] = (sample_id,)
            center_p[case_id] = (0.75 if value else 0.25,)
            labels.append(
                BinaryLabel(
                    center,
                    case_id,
                    sample_id,
                    value,
                    "target_terminal_after_aggregate_seal",
                )
            )
        sample_ids[center] = center_samples
        p_by_center[center] = center_p

    probabilities = {
        method: {
            center: dict(p_by_center[center])
            for center in CENTERS
        }
        for method in method_order
    }
    first_center = CENTERS[0]
    first_case = next(iter(sample_ids[first_center]))
    probabilities[PRIMARY_METHOD_ID][first_center][first_case] = (0.30,)
    return probabilities, sample_ids, tuple(labels)


def test_exact_selection_aware_sign_flip_maximizes_over_four_methods() -> None:
    result = selection_aware_center_sign_flip(
        _center_metrics_for_known_max_test()
    )

    assert result["null_replicate_count"] == 512
    assert result["observed_selected_method_id"] == COMPOSED_POLICY_IDS[1]
    assert result["observed_max_statistic"] == pytest.approx(0.5)
    assert result["null_exceedance_count"] == 2
    assert result["selection_aware_descriptive_randomization_p_value"] == pytest.approx(
        2.0 / 512.0
    )
    assert result["method_identity_reselected_inside_each_null_replicate"] is True
    assert result["route_pipeline_refit_inside_null_replicate"] is False
    assert result["formal_claim_authorized"] is False


def test_terminal_accepts_endpoint_oracles_and_uses_218_case_denominator() -> None:
    probabilities, sample_ids, labels = _terminal_fixture()

    result = evaluate_terminal(
        probabilities=probabilities,
        sample_ids=sample_ids,
        labels=labels,
        aggregate_seal_hash="a" * 64,
        diagnostic_summary={"source": "synthetic"},
    )

    assert len(result.method_rows) == len(ENDPOINT_METHOD_IDS) + len(
        COMPOSED_POLICY_IDS
    )
    primary = next(
        row for row in result.method_rows if row["method_id"] == PRIMARY_METHOD_ID
    )
    assert primary["route_count"] == 1
    assert primary["case_route_coverage"] == pytest.approx(
        1.0 / EXPECTED_TOTAL_CASE_COUNT
    )
    assert "route_coverage" not in primary
    selection = result.diagnostic_summary["selection_aware_center_sign_flip"]
    assert selection["null_replicate_count"] == 512
    assert math.isclose(
        result.diagnostic_summary[
            "selection_aware_descriptive_randomization_p_value"
        ],
        1.0,
    )
    assert result.diagnostic_summary["formal_claim_authorized"] is False


def test_terminal_rejects_old_menu_without_b_i_r_oracle_surfaces() -> None:
    probabilities, sample_ids, labels = _terminal_fixture()
    old_internally_contradictory_menu = {
        method: probabilities[method]
        for method in (PORTFOLIO_METHOD_ID, *COMPOSED_POLICY_IDS)
    }

    with pytest.raises(ProtocolError, match="method menu drifted"):
        evaluate_terminal(
            probabilities=old_internally_contradictory_menu,
            sample_ids=sample_ids,
            labels=labels,
            aggregate_seal_hash="a" * 64,
            diagnostic_summary={},
        )
