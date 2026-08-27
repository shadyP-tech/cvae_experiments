from __future__ import annotations

from dataclasses import FrozenInstanceError
import pickle

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.action_geometry import (
    HIGH_BOUNDARY,
    LOW_BOUNDARY,
    build_boundary_projection,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.composition import (
    compose_selection,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.config_payloads import (
    controls_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.controls import (
    METHOD_IDS,
    REQUIRED_CONTROL_METHOD_IDS,
    SCALE_BP_PRIMARY,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.empirical_bayes import (
    shrink_action_value,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.influence.contracts import (
    ActionMetricVector,
    MetricStandardError,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.influence.descriptors import (
    ACTION_FEATURE_NAMES,
    build_action_descriptor,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.influence.metrics import (
    expected_action_metrics,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.selection import (
    ActionCandidate,
    select_case_actions,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.uncertainty import (
    SelectionAwareRadius,
    build_action_envelope,
)


SHA = "a" * 64


def _radius(value: float = 0.0) -> SelectionAwareRadius:
    return SelectionAwareRadius(
        SHA,
        0.90,
        4,
        tuple((f"member-{index}", value, value, value) for index in range(4)),
        MetricStandardError(value, value, value),
    )


def _candidate(
    portfolio: np.ndarray,
    endpoint: np.ndarray,
    *,
    case_id: str,
    family: str,
    direction: str,
    metrics: ActionMetricVector,
    within_support: bool = True,
) -> ActionCandidate:
    projection = build_boundary_projection(
        portfolio, endpoint, family=family, direction=direction
    )
    estimate = shrink_action_value(
        action_id=projection.action_id,
        donor_metrics=metrics,
        local_residual=ActionMetricVector.zeros(),
        donor_standard_error=MetricStandardError.zeros(),
        local_standard_error=MetricStandardError.zeros(),
        between_center_variance=(0.0, 0.0, 0.0),
    )
    return ActionCandidate(
        case_id,
        projection,
        build_action_envelope(estimate, _radius()),
        within_support,
        True,
    )


def test_boundary_projection_uses_adjacent_float32_and_exact_p_off_mask() -> None:
    p = np.asarray([0.49, 0.51, 0.10], dtype=np.float32)
    endpoint = np.asarray([0.90, 0.20, 0.20], dtype=np.float32)
    up = build_boundary_projection(p, endpoint, family="B", direction="zero_to_one")
    down = build_boundary_projection(p, endpoint, family="B", direction="one_to_zero")

    assert up.crossing_indices == (0,)
    assert down.crossing_indices == (1,)
    assert np.float32(up.projected_probabilities[0]) == HIGH_BOUNDARY
    assert np.float32(down.projected_probabilities[1]) == LOW_BOUNDARY
    assert HIGH_BOUNDARY == np.nextafter(
        np.float32(0.5), np.float32(np.inf), dtype=np.float32
    )
    assert LOW_BOUNDARY == np.nextafter(
        np.float32(0.5), np.float32(-np.inf), dtype=np.float32
    )
    np.testing.assert_array_equal(up.projected_array()[1:], p[1:])
    assert up.full_endpoint_array()[0] == endpoint[0]
    np.testing.assert_array_equal(up.full_endpoint_array()[1:], p[1:])


def test_no_crossing_is_byte_identical_exact_p() -> None:
    p = np.asarray([0.1, 0.8], dtype=np.float32)
    endpoint = np.asarray([0.2, 0.9], dtype=np.float32)
    action = build_boundary_projection(p, endpoint, family="I", direction="zero_to_one")
    assert action.is_exact_p
    assert action.crossing_indices == ()
    np.testing.assert_array_equal(action.projected_array().view(np.uint32), p.view(np.uint32))
    np.testing.assert_array_equal(action.full_endpoint_array().view(np.uint32), p.view(np.uint32))
    assert action.projected_probability_hash == action.baseline_probability_hash


def test_boundary_and_full_endpoint_share_bacc_but_boundary_is_smaller_perturbation() -> None:
    p = np.asarray([0.49, 0.80], dtype=np.float32)
    endpoint = np.asarray([0.95, 0.90], dtype=np.float32)
    action = build_boundary_projection(p, endpoint, family="R", direction="zero_to_one")
    kwargs = {
        "support_positive_count": 10.0,
        "support_negative_count": 10.0,
        "support_row_count": 20,
    }
    eta = np.asarray([0.75, 0.25], dtype=np.float64)
    boundary = expected_action_metrics(p, action.projected_array(), eta, **kwargs)
    full = expected_action_metrics(p, action.full_endpoint_array(), eta, **kwargs)
    assert boundary.bacc_gain == pytest.approx(full.bacc_gain, abs=1.0e-15)
    assert abs(boundary.brier_loss_delta) < abs(full.brier_loss_delta)
    assert abs(boundary.log_loss_delta) < abs(full.log_loss_delta)


def test_descriptor_is_label_free_fixed_and_pickle_safe() -> None:
    p = np.asarray([0.49, 0.8], dtype=np.float32)
    endpoint = np.asarray([0.9, 0.8], dtype=np.float32)
    projection = build_boundary_projection(
        p, endpoint, family="B", direction="zero_to_one"
    )
    descriptor = build_action_descriptor(
        projection,
        case_id="case-1",
        posterior_eta=np.asarray([0.7, 0.2]),
        posterior_sd=np.asarray([0.1, 0.1]),
        seed_sd=np.asarray([0.05, 0.03]),
        positive_vote_fraction=np.asarray([2.0 / 3.0, 0.0]),
        support_positive_count=9.0,
        support_negative_count=11.0,
        support_row_count=20,
        bank_ess=4.5,
    )
    assert descriptor.feature_names == ACTION_FEATURE_NAMES
    assert len(descriptor.values) == len(ACTION_FEATURE_NAMES)
    assert "label" not in descriptor.__dataclass_fields__
    assert pickle.loads(pickle.dumps(descriptor)) == descriptor
    with pytest.raises(FrozenInstanceError):
        descriptor.case_id = "poison"  # type: ignore[misc]


def test_empirical_bayes_shrinks_sparse_local_residual() -> None:
    result = shrink_action_value(
        action_id="B::zero_to_one",
        donor_metrics=ActionMetricVector(0.1, -0.02, -0.03),
        local_residual=ActionMetricVector(0.3, 0.1, 0.2),
        donor_standard_error=MetricStandardError(0.01, 0.01, 0.01),
        local_standard_error=MetricStandardError(0.3, 0.3, 0.3),
        between_center_variance=(0.01, 0.01, 0.01),
    )
    expected_weight = 0.01 / (0.01 + 0.09)
    assert result.shrinkage_weight == pytest.approx((expected_weight,) * 3)
    assert result.posterior_metrics.bacc_gain == pytest.approx(
        0.1 + expected_weight * 0.3
    )


def test_direct_selection_composes_disjoint_directions_and_full_endpoint_control() -> None:
    case_id = "case-joint"
    p = np.asarray([0.49, 0.51, 0.2], dtype=np.float32)
    up = _candidate(
        p,
        np.asarray([0.9, 0.8, 0.2], dtype=np.float32),
        case_id=case_id,
        family="B",
        direction="zero_to_one",
        metrics=ActionMetricVector(0.03, -0.01, -0.01),
    )
    down = _candidate(
        p,
        np.asarray([0.4, 0.1, 0.2], dtype=np.float32),
        case_id=case_id,
        family="R",
        direction="one_to_zero",
        metrics=ActionMetricVector(0.02, -0.01, -0.01),
    )
    decision = select_case_actions(
        case_id=case_id,
        baseline_probability_hash=up.projection.baseline_probability_hash,
        candidates=(up, down),
    )
    assert decision.selected_action_ids == (
        "B::zero_to_one",
        "R::one_to_zero",
    )
    boundary = compose_selection(p, (up, down), decision)
    full = compose_selection(p, (up, down), decision, mode="full_endpoint")
    assert boundary.crossing_indices == (0, 1)
    assert np.float32(boundary.composed_probabilities[0]) == HIGH_BOUNDARY
    assert np.float32(boundary.composed_probabilities[1]) == LOW_BOUNDARY
    assert full.composed_probabilities[0] == pytest.approx(0.9)
    assert full.composed_probabilities[1] == pytest.approx(0.1)
    assert boundary.composed_probabilities[2] == p[2]
    assert full.composed_probabilities[2] == p[2]
    assert boundary.composed_probability_hash != full.composed_probability_hash


def test_p_wins_tie_and_failed_support_is_exact_p() -> None:
    case_id = "case-abstain"
    p = np.asarray([0.49, 0.8], dtype=np.float32)
    candidate = _candidate(
        p,
        np.asarray([0.9, 0.8], dtype=np.float32),
        case_id=case_id,
        family="I",
        direction="zero_to_one",
        metrics=ActionMetricVector(1.0e-12, -0.01, -0.01),
        within_support=False,
    )
    decision = select_case_actions(
        case_id=case_id,
        baseline_probability_hash=candidate.projection.baseline_probability_hash,
        candidates=(candidate,),
    )
    assert decision.is_exact_p
    composed = compose_selection(p, (candidate,), decision)
    assert composed.is_exact_p
    np.testing.assert_array_equal(
        composed.as_array().view(np.uint32), p.view(np.uint32)
    )


def test_controls_payload_distinguishes_full_menu_from_nonprimary_controls() -> None:
    payload = controls_payload()
    assert tuple(payload["required_methods"]) == METHOD_IDS
    assert tuple(payload["required_control_methods"]) == REQUIRED_CONTROL_METHOD_IDS
    assert SCALE_BP_PRIMARY not in REQUIRED_CONTROL_METHOD_IDS
    assert set(REQUIRED_CONTROL_METHOD_IDS) == set(METHOD_IDS) - {SCALE_BP_PRIMARY}
    assert "required" not in payload
