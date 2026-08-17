from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.constants import (
    CENTERS,
    ENDPOINT_METHOD_IDS,
    EXPECTED_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_ORDERED_VOTER_COUNT,
    EXPECTED_OUTER_PLAN_COUNT,
    EXPECTED_UNORDERED_PAIR_COUNT,
    MODEL_BASED_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    REGRET_FEATURE_NAMES,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.contracts import (
    BinaryLabel,
    CandidateDescriptor,
    CenterProbabilitySurface,
    DonorRegretRow,
    EndpointCasePrediction,
    PhysicalProbabilitySurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.donor_regret_model import (
    fit_center_balanced_ridge,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.engine import (
    build_preterminal_result,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.evaluation import (
    evaluate_terminal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.nested_endpoint_regret import (
    build_donor_regret_row,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.workstation import (
    assert_canonical_workload,
    schedule_center_batches,
)


def _surface_and_labels() -> tuple[PhysicalProbabilitySurface, dict[tuple[str, str, str], int]]:
    center_surfaces = {}
    labels: dict[tuple[str, str, str], int] = {}
    for center_index, center in enumerate(CENTERS):
        sample_ids = tuple(
            f"sample-{center}-{case}-{row}" for case in range(3) for row in range(4)
        )
        case_ids = tuple(f"case-{center}-{case}" for case in range(3) for _row in range(4))
        truth = np.asarray([0, 1, 0, 1] * 3, dtype=np.int8)
        for sample, case, value in zip(sample_ids, case_ids, truth, strict=True):
            labels[(center, case, sample)] = int(value)
        arrays = {}
        for action_index, action in enumerate(physical_action_ids(center)):
            base = np.asarray(
                [0.30, 0.70, 0.45, 0.55] * 3, dtype=np.float32
            )
            shift = np.float32(
                ((action_index + center_index) % 5 - 2) * 0.025
            )
            values = np.clip(base + shift, 0.02, 0.98)
            arrays[action] = np.stack(
                [
                    np.clip(values + np.float32((seed - 4) * 0.002), 0.01, 0.99)
                    for seed in range(9)
                ]
            ).astype(np.float32)
        center_surfaces[center] = CenterProbabilitySurface(
            center,
            sample_ids,
            case_ids,
            MappingProxyType(arrays),
            "1" * 16,
        )
    return (
        PhysicalProbabilitySurface(
            MappingProxyType(center_surfaces),
            "1" * 16,
            strict_canonical_topology=False,
        ),
        labels,
    )


def _loader(labels: dict[tuple[str, str, str], int], poison_case: str | None = None):
    def load(allowed: frozenset[tuple[str, str, str]], _role: str):
        return tuple(
            SimpleNamespace(
                center=center,
                case_id=case,
                sample_id=sample,
                value=(1 - labels[key] if case == poison_case else labels[key]),
            )
            for key in sorted(allowed)
            for center, case, sample in (key,)
        )

    return load


def test_canonical_workload_and_lpt_schedule() -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.constants import (
        EXPECTED_CASE_COUNTS_BY_CENTER,
    )

    assert EXPECTED_OUTER_PLAN_COUNT == 218
    assert EXPECTED_UNORDERED_PAIR_COUNT == 2_660
    assert EXPECTED_ORDERED_VOTER_COUNT == 5_320
    assert EXPECTED_ENDPOINT_MODEL_FIT_COUNT == 46_048
    assert_canonical_workload(EXPECTED_CASE_COUNTS_BY_CENTER)
    batches = schedule_center_batches(EXPECTED_CASE_COUNTS_BY_CENTER)
    assert len(batches) == 4
    assert {row.center for batch in batches for row in batch} == set(CENTERS)


def test_center_balanced_ridge_is_invariant_to_within_center_duplication() -> None:
    feature_a = (1.0,) + (0.0,) * (len(REGRET_FEATURE_NAMES) - 1)
    feature_b = (2.0,) + (0.0,) * (len(REGRET_FEATURE_NAMES) - 1)

    def rows(duplicate: bool) -> tuple[DonorRegretRow, ...]:
        result = []
        for center, feature, response in (
            ("0", feature_a, 0.1),
            ("1", feature_b, -0.1),
        ):
            count = 2 if duplicate and center == "0" else 1
            for index in range(count):
                result.append(
                    DonorRegretRow(
                        center,
                        f"{center}-{index}",
                        "B",
                        feature,
                        response,
                        -response,
                        count,
                        f"{index + 1:064x}",
                    )
                )
        return tuple(result)

    original = fit_center_balanced_ridge(
        rows(False), response_name="bacc_regret", training_centers=("0", "1")
    )
    duplicated = fit_center_balanced_ridge(
        rows(True), response_name="bacc_regret", training_centers=("0", "1")
    )
    np.testing.assert_allclose(original.feature_mean, duplicated.feature_mean)
    np.testing.assert_allclose(original.feature_scale, duplicated.feature_scale)
    np.testing.assert_allclose(
        original.coefficients, duplicated.coefficients, atol=1.0e-15
    )


def test_no_candidate_row_has_explicit_candidate_indicator() -> None:
    descriptor = CandidateDescriptor(
        "0",
        "case",
        PORTFOLIO_METHOD_ID,
        REGRET_FEATURE_NAMES,
        (0.0,) * len(REGRET_FEATURE_NAMES),
        ("1" * 64,),
    )
    assert descriptor.is_candidate is False
    assert descriptor.values[-1] == 0.0


def test_donor_case_labels_enter_only_response_not_descriptor() -> None:
    values = [0.0] * len(REGRET_FEATURE_NAMES)
    values[3] = 1.0
    values[-1] = 1.0
    descriptor = CandidateDescriptor(
        "0",
        "case-d",
        "B",
        REGRET_FEATURE_NAMES,
        tuple(values),
        ("1" * 64,),
    )
    endpoint_probabilities = {
        method: ((0.6, 0.4) if method == "B" else (0.4, 0.6))
        for method in ENDPOINT_METHOD_IDS
    }
    prediction = EndpointCasePrediction(
        "0",
        "case-d",
        ("sample-0", "sample-1"),
        MappingProxyType(endpoint_probabilities),
        "2" * 64,
    )

    def row(values: tuple[int, int]) -> DonorRegretRow:
        labels = tuple(
            BinaryLabel("0", "case-d", sample, value, "regret_donor")
            for sample, value in zip(prediction.sample_ids, values, strict=True)
        )
        return build_donor_regret_row(
            descriptor,
            prediction,
            labels,
            center_case_count=1,
            center_n_positive=1,
            center_n_negative=1,
            center_sample_count=2,
        )

    baseline = row((1, 0))
    poisoned = row((0, 1))
    assert baseline.descriptor_hash == poisoned.descriptor_hash == descriptor.descriptor_hash
    assert baseline.feature_values == poisoned.feature_values == descriptor.values
    assert baseline.bacc_regret == -poisoned.bacc_regret
    assert baseline.log_loss_delta == pytest.approx(-poisoned.log_loss_delta)


def test_synthetic_full_engine_seals_before_terminal_and_poison_exclusions() -> None:
    surface, labels = _surface_and_labels()
    baseline = build_preterminal_result(
        surface, _loader(labels), use_processes=False
    )
    assert all(
        row.selected_method in {"B", "I_OPPORTUNITY_GATED", "R_NINE_ARM_ROBUST", PORTFOLIO_METHOD_ID}
        for row in baseline.decisions_by_policy[MODEL_BASED_METHOD_ID]
    )
    target_case = "case-0-0"
    voter_case = "case-0-1"
    poisoned = build_preterminal_result(
        surface, _loader(labels, poison_case=target_case), use_processes=False
    )
    baseline_descriptor = next(
        row for row in baseline.descriptors_by_center["0"] if row.case_id == target_case
    )
    poisoned_descriptor = next(
        row for row in poisoned.descriptors_by_center["0"] if row.case_id == target_case
    )
    assert baseline_descriptor.descriptor_hash == poisoned_descriptor.descriptor_hash
    baseline_decision = next(
        row
        for row in baseline.decisions_by_policy[MODEL_BASED_METHOD_ID]
        if (row.target_center, row.case_id) == ("0", target_case)
    )
    poisoned_decision = next(
        row
        for row in poisoned.decisions_by_policy[MODEL_BASED_METHOD_ID]
        if (row.target_center, row.case_id) == ("0", target_case)
    )
    assert baseline_decision.decision_hash == poisoned_decision.decision_hash

    poisoned_voter = build_preterminal_result(
        surface, _loader(labels, poison_case=voter_case), use_processes=False
    )
    poisoned_support_descriptor = next(
        row
        for row in poisoned_voter.descriptors_by_center["0"]
        if row.case_id == target_case
    )
    assert poisoned_support_descriptor.descriptor_hash != baseline_descriptor.descriptor_hash
    pair = tuple(sorted((target_case, voter_case)))
    baseline_pair_hash = next(
        digest
        for first, second, digest in baseline.endpoint_products[0].pair_state_hashes
        if (first, second) == pair
    )
    poisoned_pair_hash = next(
        digest
        for first, second, digest in poisoned_voter.endpoint_products[0].pair_state_hashes
        if (first, second) == pair
    )
    assert baseline_pair_hash == poisoned_pair_hash

    terminal = evaluate_terminal(baseline)
    assert terminal.capability_report["status"] == "PASS"
    assert terminal.diagnostic_summary["fresh_evidence"] is False
    assert terminal.selection_control[
        "policy_identity_reselected_inside_every_replicate"
    ] is True
    assert terminal.selection_control[
        "route_features_models_and_decisions_refit_inside_replicate"
    ] is False
