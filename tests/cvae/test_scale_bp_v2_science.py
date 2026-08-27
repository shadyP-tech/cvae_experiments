from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.hashing import canonical_hash
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.capability_scoring import score_scoped_action_rectangle
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.identity import GovernanceError
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.physical.contracts import (
    ACTION_IDS,
    FeatureVector,
    MetricVector,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.physical.endpoints import (
    RouteEndpointPlan,
    reconstruct_case_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.physical.geometry import (
    HIGH_BOUNDARY_FLOAT32,
    LOW_BOUNDARY_FLOAT32,
    build_boundary_action,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.posterior.contracts import (
    DonorFitScope,
    DonorObservation,
    LocalResidualObservation,
    ScaleVector,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.posterior.donor import (
    DonorDeleteCenterFold,
    DonorPrediction,
    fit_donor_action_model,
    predict_donor_action,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.posterior.empirical_bayes import combine_empirical_bayes
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.posterior.local import (
    fit_route_local_residual,
    predict_local_residual,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.posterior.uncertainty import build_preargmax_bounds
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.routing.selection import select_action
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.scientific_contracts import canonical_scientific_contracts_payload
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.utility.actions import build_action_rectangle
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.worker.emission import decision_payload


def _digest(value: object) -> str:
    return canonical_hash({"value": value})


class _SyntheticPhysicalStore:
    def __init__(self) -> None:
        self.requested: list[str] = []
        self.values = {
            "B": np.asarray([0.4, 0.6]),
            "U": np.asarray([0.55, 0.45]),
            "A1::source=2": np.asarray([0.8, 0.2]),
            "A1::source=3": np.asarray([0.7, 0.3]),
            "A1::source=5": np.asarray([0.9, 0.1]),
            "A1::source=6": np.asarray([0.65, 0.35]),
            "A1::source=7": np.asarray([0.6, 0.4]),
            "A1::source=8": np.asarray([0.55, 0.45]),
            "A1::source=9": np.asarray([0.51, 0.49]),
        }

    def case_ids(self, target_center: object) -> tuple[str, ...]:
        assert str(target_center) == "0"
        return ("route-case",)

    def exact_nine_view(
        self, target_center: object, action_id: object, *, case_id: object
    ) -> SimpleNamespace:
        assert str(target_center) == "0"
        assert str(case_id) == "route-case"
        action = str(action_id)
        self.requested.append(action)
        values = self.values[action]
        seeds = np.tile(values, (9, 1))
        return SimpleNamespace(
            sample_ids=("row-0", "row-1"),
            seed_probabilities=seeds,
            mean_probability=np.mean(seeds, axis=0),
            view_hash=_digest(action),
        )


def _surface_and_rectangle():
    store = _SyntheticPhysicalStore()
    plan = RouteEndpointPlan(
        target_center="0",
        case_id="route-case",
        identification_sources={"zero_to_one": "2", "one_to_zero": "2"},
        robust_arm_sources={
            "zero_to_one": ("3",) * 9,
            "one_to_zero": ("3",) * 9,
        },
        support_scope_hash=_digest("H-minus-c"),
        source_excluded_centers=("0", "1"),
        support_excluded_case_ids=("route-case",),
        outer_held_case_id="route-case",
    )
    surface = reconstruct_case_surface(store, plan)
    return store, surface, build_action_rectangle(surface)


def test_challengers_exclude_forbidden_source_and_protected_p_is_independent() -> None:
    store, surface, _ = _surface_and_rectangle()

    assert "A1::source=1" not in store.requested
    assert surface.available_sources == ("2", "3", "5", "6", "7", "8", "9")
    np.testing.assert_allclose(
        surface.protected_component_probabilities["I_PROTECTED"], [0.8, 0.2]
    )
    np.testing.assert_allclose(
        surface.protected_component_probabilities["R_PROTECTED"], [0.7, 0.3]
    )
    np.testing.assert_allclose(surface.protected_p, [0.76, 0.24], rtol=0, atol=1e-15)
    np.testing.assert_allclose(surface.challenger("I", "zero_to_one"), [0.9, 0.49])
    np.testing.assert_allclose(surface.challenger("I", "one_to_zero"), [0.51, 0.1])


def test_boundary_projection_changes_only_crossing_coordinates() -> None:
    protected = np.asarray([0.2, 0.4, 0.6, 0.8])
    endpoint = np.asarray([0.1, 0.9, 0.1, 0.9])

    upward = build_boundary_action(
        protected, endpoint, family="B", direction="zero_to_one"
    )
    downward = build_boundary_action(
        protected, endpoint, family="B", direction="one_to_zero"
    )
    assert upward.crossing_indices == (1,)
    assert downward.crossing_indices == (2,)
    assert upward.projected[1] == HIGH_BOUNDARY_FLOAT32
    assert downward.projected[2] == LOW_BOUNDARY_FLOAT32
    np.testing.assert_array_equal(upward.projected[[0, 2, 3]], protected[[0, 2, 3]])
    np.testing.assert_array_equal(downward.projected[[0, 1, 3]], protected[[0, 1, 3]])


def test_donor_local_and_empirical_bayes_math_is_finite_and_route_local() -> None:
    training_centers = ("1", "2", "3", "5", "6", "7", "8", "9")
    scope = DonorFitScope(
        outer_center="0",
        prediction_center="0",
        held_case_id="route-case",
        training_case_ids_by_center={
            center: (f"donor-{center}",) for center in training_centers
        },
        source_excluded_centers=("0",),
        role="FINAL_H_C",
    )
    donor_rows = []
    for center_index, center in enumerate(training_centers):
        source_centers = tuple(
            value for value in training_centers if value != center
        )
        for action_index, action_id in enumerate(ACTION_IDS):
            donor_rows.append(
                DonorObservation(
                    query_center=center,
                    case_id=f"donor-{center}",
                    action_id=action_id,
                    descriptor=FeatureVector(("x",), (center_index + action_index / 10.0,)),
                    realized=MetricVector(
                        0.01 * action_index + 0.001 * center_index,
                        -0.02 * action_index,
                        -0.03 * action_index,
                    ),
                    source_centers=source_centers,
                    scope_hash=scope.scope_hash,
                )
            )
    delete_center_folds = []
    for deleted_center in training_centers:
        remaining = tuple(
            center for center in training_centers if center != deleted_center
        )
        fold_scope = DonorFitScope(
            outer_center="0",
            prediction_center="0",
            held_case_id="route-case",
            training_case_ids_by_center={
                center: (f"donor-{center}",) for center in remaining
            },
            source_excluded_centers=(
                "0",
                deleted_center,
            )
            if deleted_center != "0"
            else ("0",),
            role="FINAL_H_C",
        )
        training_rows = tuple(
            DonorObservation(
                query_center=row.query_center,
                case_id=row.case_id,
                action_id=row.action_id,
                descriptor=row.descriptor,
                realized=row.realized,
                source_centers=tuple(
                    center for center in remaining if center != row.query_center
                ),
                scope_hash=fold_scope.scope_hash,
            )
            for row in donor_rows
            if row.query_center != deleted_center
        )
        validation_rows = tuple(
            row for row in donor_rows if row.query_center == deleted_center
        )
        delete_center_folds.append(
            DonorDeleteCenterFold(
                deleted_center,
                scope,
                fold_scope,
                training_rows,
                validation_rows,
            )
        )
    donor_model = fit_donor_action_model(
        donor_rows,
        scope=scope,
        delete_center_folds=delete_center_folds,
        ridge_alpha=0.5,
    )
    query = FeatureVector(("x",), (0.75,))
    donor = predict_donor_action(
        donor_model, action_id="B::zero_to_one", descriptor=query
    )
    assert donor_model.training_row_count == len(training_centers) * len(ACTION_IDS)
    assert np.isfinite(donor.mean.as_array()).all()

    support_hash = _digest("local-H-minus-c")
    local_rows = []
    for case_index in range(4):
        for action_index, action_id in enumerate(ACTION_IDS):
            local_rows.append(
                LocalResidualObservation(
                    target_center="0",
                    route_case_id="route-case",
                    support_case_id=f"support-{case_index}",
                    action_id=action_id,
                    descriptor=FeatureVector(("x",), (float(case_index),)),
                    residual=MetricVector(
                        0.05 + 0.002 * action_index,
                        -0.03,
                        -0.04,
                    ),
                    donor_prediction_hash=_digest((case_index, action_id)),
                    support_scope_hash=support_hash,
                    endpoint_plan_hash=_digest(("endpoint", case_index, action_id)),
                    support_excluded_case_ids=(
                        "route-case",
                        f"support-{case_index}",
                    ),
                    outer_held_case_id="route-case",
                )
            )
    local_model = fit_route_local_residual(local_rows, ridge_alpha=0.5)
    local = predict_local_residual(
        local_model, action_id="B::zero_to_one", descriptor=query
    )
    estimate = combine_empirical_bayes(
        donor,
        local,
        target_center="0",
        case_id="route-case",
        structural_noop=False,
    )

    prior_variance = np.square(donor.heterogeneity.as_tuple())
    local_noise = (
        np.square(local.oof_rmse.as_tuple())
        + np.square(local.estimator_se.as_tuple())
    )
    expected_weights = np.divide(
        prior_variance,
        prior_variance + local_noise,
        out=np.zeros(3),
        where=(prior_variance + local_noise) > 0,
    )
    np.testing.assert_allclose(estimate.shrinkage_weights, expected_weights)
    np.testing.assert_allclose(
        estimate.mean.as_array(),
        donor.mean.as_array() + expected_weights * local.correction.as_array(),
    )
    assert local_model.route_case_id not in local_model.support_case_ids
    assert set(fold for _, fold in local_model.fold_assignments) == {0, 1, 2, 3}

    with pytest.raises(GovernanceError):
        DonorFitScope(
            outer_center="0",
            prediction_center="0",
            held_case_id="route-case",
            training_case_ids_by_center={"0": ("leaked",), "1": ("a",), "2": ("b",)},
            source_excluded_centers=("0",),
            role="FINAL_H_C",
        )


def _selection_estimates(rectangle, *, safe: bool):
    output = []
    for index, action_id in enumerate(ACTION_IDS):
        structural_noop = rectangle.cell(action_id).structural_noop
        if structural_noop:
            mean = MetricVector.zeros()
        elif safe:
            mean = MetricVector(0.2 if index == 0 else 0.1, -0.1, -0.1)
        else:
            mean = MetricVector(-0.1, 0.1, 0.1)
        donor = DonorPrediction(
            action_id,
            _digest(("descriptor", action_id)),
            mean,
            ScaleVector.zeros(),
            ScaleVector.zeros(),
            ScaleVector.zeros(),
            _digest(("model", action_id)),
            _digest(("scope", action_id)),
        )
        output.append(
            combine_empirical_bayes(
                donor,
                None,
                target_center=rectangle.target_center,
                case_id=rectangle.case_id,
                structural_noop=structural_noop,
            )
        )
    return tuple(output)


def test_direct_selection_uses_unique_safe_action_or_exact_p_fallback() -> None:
    _, surface, rectangle = _surface_and_rectangle()
    safe = _selection_estimates(rectangle, safe=True)
    selected = select_action(rectangle, safe, build_preargmax_bounds(safe))
    assert selected.selected_action_id == "B::zero_to_one"
    assert selected.is_exact_p is False

    unsafe = _selection_estimates(rectangle, safe=False)
    fallback = select_action(rectangle, unsafe, build_preargmax_bounds(unsafe))
    assert fallback.is_exact_p is True
    np.testing.assert_array_equal(fallback.emitted_probabilities, surface.protected_p)
    np.testing.assert_array_equal(
        fallback.full_endpoint_probabilities, surface.protected_p
    )
    audit = decision_payload(fallback, rectangle)
    assert audit["p_candidate"] == {
        "representation": "IMPLICIT_ZERO_UTILITY_EXACT_FALLBACK",
        "expected_utility_anchor": 0.0,
        "candidate_assessment_emitted": False,
        "wins_without_unique_robust_safe_positive_action": True,
    }
    assert audit["exact_p_fallback"] is True
    assert [row["action_id"] for row in audit["action_switch_audit"]] == list(
        ACTION_IDS
    )
    for cell, row in zip(rectangle.cells, audit["action_switch_audit"], strict=True):
        assert row["threshold_switch_count"] == len(cell.action.crossing_indices)
        assert row["harmful_switch_count"] is None
        assert row["harmful_switch_count_status"] == (
            "UNAVAILABLE_PRETERMINAL_TARGET_LABELS_CLOSED"
        )


def test_optional_blend_and_unavailable_evidence_are_frozen_explicitly() -> None:
    contracts = canonical_scientific_contracts_payload()
    geometry = contracts["action_geometry"]
    evidence = contracts["influence"]
    selection = contracts["selection"]

    assert geometry["calibrated_convex_blend_available"] is False
    assert geometry["calibrated_convex_blend_status"] == (
        "DEFERRED_NO_LEGAL_CALIBRATION_SURFACE"
    )
    assert selection["P_candidate_representation"] == (
        "IMPLICIT_ZERO_UTILITY_EXACT_FALLBACK"
    )
    assert selection["P_candidate_assessment_emitted"] is False
    assert evidence["threshold_switch_count_persisted_per_action"] is True
    assert evidence["harmful_switch_count_preterminal_available"] is False
    assert evidence["latent_embedding_distance_available"] is False
    assert evidence["effective_source_training_support_available"] is False
    assert evidence["source_calibration_status_available"] is False


def test_capability_scoring_cannot_accept_foreign_denominators() -> None:
    signature = inspect.signature(score_scoped_action_rectangle)
    assert "denominators" not in signature.parameters
