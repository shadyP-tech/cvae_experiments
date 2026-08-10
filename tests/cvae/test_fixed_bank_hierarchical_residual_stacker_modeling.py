from __future__ import annotations

from dataclasses import replace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.scientific_core import (
    DESIGN_TERMS,
    BinaryLabel,
    CaseFeatureRow,
    DonorResponseRow,
    Standardization,
    candidate_sources,
    compute_source_control,
    context_permute_training_features,
    fit_loco_hierarchical_model,
    interaction_design,
    strict_transfer_training_rows,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.scientific_constants import (
    MIDOGPP_CENTERS,
)


def _surfaces() -> tuple[tuple[CaseFeatureRow, ...], tuple[DonorResponseRow, ...]]:
    features: list[CaseFeatureRow] = []
    responses: list[DonorResponseRow] = []
    for donor_index, donor in enumerate(MIDOGPP_CENTERS):
        for case_index in range(2):
            case = f"case-{donor}-{case_index}"
            for source_index, source in enumerate(candidate_sources(donor)):
                base = 0.01 * (1 + donor_index) + 0.003 * source_index + 0.001 * case_index
                phi = (base, abs(base) + 0.01, 0.02 + 0.001 * case_index, 0.1 + 0.01 * source_index)
                features.append(CaseFeatureRow(donor, case, source, 4, phi))
                for side in (0, 1):
                    response = (1 if side else -1) * (0.02 + 0.2 * base)
                    responses.append(DonorResponseRow(donor, case, source, side, 2 + side, response))
    return tuple(sorted(features)), tuple(sorted(responses))


def test_interaction_design_has_no_source_identity_or_extra_coordinates() -> None:
    standardization = Standardization((0.0,) * 5, (1.0,) * 5)
    design = interaction_design((1.0, 2.0, 3.0, 0.5), 4.0, standardization)
    assert len(design) == len(DESIGN_TERMS) == 10
    assert DESIGN_TERMS == (
        "intercept",
        "residual_logit_mean",
        "residual_logit_abs_mean",
        "residual_logit_std",
        "hard_disagreement_rate",
        "global_source_control",
        "global_source_control_x_residual_logit_mean",
        "global_source_control_x_residual_logit_abs_mean",
        "global_source_control_x_residual_logit_std",
        "global_source_control_x_hard_disagreement_rate",
    )
    assert design == pytest.approx((1.0, 1.0, 2.0, 3.0, 0.5, 4.0, 4.0, 8.0, 12.0, 2.0))


def test_strict_training_filter_excludes_both_query_and_source_H_q_e_roles() -> None:
    _features, responses = _surfaces()
    rows = strict_transfer_training_rows(
        responses,
        target_center="0",
        heldout_query_center="2",
        heldout_source_id="1",
        class_side=0,
    )
    forbidden = {"0", "1", "2"}
    assert rows
    assert all(row.donor_center not in forbidden for row in rows)
    assert all(row.source_id not in forbidden for row in rows)


def test_context_P_permutation_never_uses_H_e_q_feature_origins() -> None:
    features, _responses = _surfaces()
    permuted = context_permute_training_features(
        features,
        target_center="0",
        heldout_source_id="1",
        excluded_query_center="2",
    )
    forbidden = {"0", "1", "2"}
    assert permuted
    assert all(row.target_center not in forbidden for row in permuted)
    assert all(row.source_id not in forbidden for row in permuted)
    assert all(row.feature_origin_source_id not in forbidden for row in permuted)


def test_source_control_binds_all_context_exclusions_and_ignores_poisoned_query() -> None:
    features, _responses = _surfaces()
    control = compute_source_control(
        features,
        target_center="0",
        source_id="3",
        excluded_query_center="2",
        additional_excluded_centers=("1",),
    )
    poisoned = tuple(
        replace(row, phi=(999.0, 999.0, 999.0, 1.0))
        if row.source_id == "3" and row.target_center in {"0", "1", "2", "3"}
        else row
        for row in features
    )
    after = compute_source_control(
        poisoned,
        target_center="0",
        source_id="3",
        excluded_query_center="2",
        additional_excluded_centers=("1",),
    )
    assert control.global_source_control == after.global_source_control
    assert control.donor_query_centers == after.donor_query_centers
    assert control.context_excluded_centers == ("1",)


@pytest.mark.parametrize("family", ("G", "R", "P"))
def test_final_candidate_model_is_invariant_to_H_e_role_feature_poison(family: str) -> None:
    features, responses = _surfaces()
    kwargs = {"source_control_features": features} if family == "P" else {}
    original = fit_loco_hierarchical_model(
        features,
        responses,
        target_center="0",
        model_family=family,
        **kwargs,
    )
    # For deployed e=1, neither query-center H/e nor source H/e features may
    # affect its final fit, standardization, or nested alpha selection.
    poisoned = tuple(
        replace(row, phi=(50.0, 50.0, 50.0, 1.0))
        if row.target_center in {"0", "1"} or row.source_id == "0"
        else row
        for row in features
    )
    poisoned_kwargs = {"source_control_features": poisoned} if family == "P" else {}
    refit = fit_loco_hierarchical_model(
        poisoned,
        responses,
        target_center="0",
        model_family=family,
        **poisoned_kwargs,
    )
    for side in (0, 1):
        left = original.candidate("1", side)
        right = refit.candidate("1", side)
        assert left.model_hash == right.model_hash
        assert left.ridge_alpha == right.ridge_alpha
        assert left.ridge_alpha == original.candidate("1", 1 - side).ridge_alpha


def test_P_candidate_model_does_not_consume_held_e_local_phi() -> None:
    features, responses = _surfaces()
    original = fit_loco_hierarchical_model(
        features,
        responses,
        target_center="0",
        source_control_features=features,
        model_family="P",
    )
    local_phi_poison = tuple(
        replace(row, phi=(80.0, 80.0, 80.0, 1.0)) if row.source_id == "1" else row
        for row in features
    )
    # The separately sealed source-control surface remains original: P changes
    # local phi alignment only and must preserve g/residual arrays.
    refit = fit_loco_hierarchical_model(
        local_phi_poison,
        responses,
        target_center="0",
        source_control_features=features,
        model_family="P",
    )
    assert tuple(original.candidate("1", side).model_hash for side in (0, 1)) == tuple(
        refit.candidate("1", side).model_hash for side in (0, 1)
    )


def test_G_design_is_case_independent_and_local_coefficients_are_zeroed() -> None:
    features, responses = _surfaces()
    model = fit_loco_hierarchical_model(
        features, responses, target_center="0", model_family="G"
    )
    candidate = model.candidate("1", 0)
    # Phi and interaction columns are structurally zero in every G fit; ridge
    # therefore leaves their coefficients exactly zero.
    local_indices = (1, 2, 3, 4, 6, 7, 8, 9)
    assert all(candidate.coefficients[index] == 0.0 for index in local_indices)
