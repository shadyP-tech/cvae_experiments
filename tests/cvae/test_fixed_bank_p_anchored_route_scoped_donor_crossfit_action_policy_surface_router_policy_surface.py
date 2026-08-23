from __future__ import annotations

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.contracts import (
    FavorableUtility,
    RouteKey,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.identity import (
    ACTION_STRATA,
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.policy_surface import (
    CalibratedPrefixCell,
    PolicyAction,
    PolicyOOFResidual,
    apply_policy_envelope,
    attach_prefix_responses,
    build_policy_envelope,
    build_prefix_surface,
    equal_center_route_prefix_weights,
    fit_nested_policy_calibrations,
    observations_from_surfaces,
    select_policy_prefix,
)


def _hash(value: object) -> str:
    return canonical_hash(value)


def _action(
    *,
    outer: str,
    route: str,
    case: str,
    bacc: float,
    stratum: tuple[str, str],
    role: str = "pseudo",
) -> PolicyAction:
    key = RouteKey(
        role,
        outer,
        route,
        case,
        outer,
        None if role == "target" else route,
        _hash(("fit", outer, route, case)),
    )
    return PolicyAction(
        key,
        case,
        _hash(("action", outer, route, case, stratum)),
        stratum[0],
        stratum[1],
        FavorableUtility(bacc, bacc + 0.02, bacc + 0.03),
        _hash(("action-calibration", outer, route, case)),
    )


def _surface(outer: str, donor: str, offset: float):
    actions = (
        _action(
            outer=outer,
            route=donor,
            case=f"{donor}-low",
            bacc=0.10 + offset,
            stratum=ACTION_STRATA[0],
        ),
        _action(
            outer=outer,
            route=donor,
            case=f"{donor}-high",
            bacc=0.30 + offset,
            stratum=ACTION_STRATA[-1],
        ),
    )
    sealed = build_prefix_surface(
        actions,
        surface_role="pseudo",
        outer_center=outer,
        route_center=donor,
        action_surface_seal_hash=_hash(("seal", outer, donor)),
    )
    realized = {
        row.action_hash: FavorableUtility(
            row.predicted_utility.bacc_gain - 0.01 * (int(donor) % 3),
            row.predicted_utility.brier_gain - 0.005,
            row.predicted_utility.log_gain - 0.004,
        )
        for row in sealed.ranked_actions
    }
    return sealed, attach_prefix_responses(sealed, realized)


def test_complete_prefix_surface_is_sealed_before_responses() -> None:
    sealed, response = _surface("0", "1", 0.0)
    assert tuple(row.k for row in sealed.cells) == (0, 1, 2)
    assert sealed.cells[0].ordered_action_hashes == ()
    assert sealed.cells[0].predicted_utility == FavorableUtility.zeros()
    assert sealed.ranked_actions[0].case_id == "1-high"
    assert sealed.cells[1].ordered_action_hashes == (
        sealed.ranked_actions[0].action_hash,
    )
    assert sealed.cells[2].ordered_action_hashes == tuple(
        row.action_hash for row in sealed.ranked_actions
    )
    assert sealed.cells[1].normalized_depth == 0.5
    assert sealed.cells[2].normalized_depth == 1.0
    assert sealed.cells[1].max_positive_candidate_share == 1.0
    assert sum(sealed.cells[2].stratum_proportions) == pytest.approx(1.0)
    assert not sealed.responses_available
    assert all(row.realized_utility is None for row in sealed.cells)

    assert response.responses_available
    assert response.surface_hash == sealed.surface_hash
    assert all(row.response_hash is not None for row in response.cells)
    assert response.cells[0].realized_utility == FavorableUtility.zeros()
    with pytest.raises(ProtocolError, match="opened twice"):
        attach_prefix_responses(response, {})
    with pytest.raises(ProtocolError, match="incomplete or extra"):
        attach_prefix_responses(sealed, {})


def test_nested_h_j_exclusions_and_hierarchical_weights() -> None:
    pairs = tuple(
        _surface("0", donor, 0.01 * index)
        for index, donor in enumerate(("1", "2", "3", "5"))
    )
    response_surfaces = tuple(response for _sealed, response in pairs)
    observations = observations_from_surfaces(response_surfaces)
    weights = equal_center_route_prefix_weights(observations)
    assert weights.sum() == pytest.approx(1.0)
    for center in ("1", "2", "3", "5"):
        assert weights[
            np.asarray([row.center == center for row in observations])
        ].sum() == pytest.approx(0.25)

    nested = fit_nested_policy_calibrations(
        response_surfaces, outer_center="0"
    )
    assert nested.final_calibration.excluded_centers == ("0",)
    assert set(nested.final_calibration.supported_centers) == {"1", "2", "3", "5"}
    assert tuple(row.scored_center for row in nested.oof_calibrations) == (
        "1",
        "2",
        "3",
        "5",
    )
    for model in nested.oof_calibrations:
        assert set(model.excluded_centers) == {"0", model.scored_center}
        assert model.scored_center not in model.supported_centers
        assert "0" not in model.supported_centers
    assert len(nested.oof_residuals) == 4 * 3
    assert all(
        set(row.calibration_excluded_centers)
        == {row.outer_center, row.scored_center}
        for row in nested.oof_residuals
    )


def test_policy_envelope_is_center_balanced_and_applied_once() -> None:
    residuals = []
    for center, overprediction in (("1", 1.0), ("2", 3.0), ("3", 5.0)):
        residuals.append(
            PolicyOOFResidual(
                "0",
                center,
                _hash(("route", center)),
                _hash(("cell", center)),
                FavorableUtility(overprediction, overprediction, overprediction),
                FavorableUtility.zeros(),
                _hash(("model", center)),
                tuple(sorted(("0", center))),
            )
        )
    envelope = build_policy_envelope(residuals, outer_center="0")
    assert envelope.full_equal_center_mean == FavorableUtility(3.0, 3.0, 3.0)
    # Omitting the low-overprediction center yields the maximum mean: (3+5)/2.
    assert envelope.correction == FavorableUtility(4.0, 4.0, 4.0)
    corrected, count = apply_policy_envelope(
        FavorableUtility(6.0, 6.0, 6.0), envelope
    )
    assert corrected == FavorableUtility(2.0, 2.0, 2.0)
    assert count == 1
    with pytest.raises(ProtocolError, match="more than once"):
        apply_policy_envelope(corrected, envelope, correction_applied_count=count)
    payload = envelope.to_payload()
    assert payload["descriptive_lower_envelope_only"] is True
    assert payload["finite_sample_coverage_claimed"] is False


def test_selection_falls_back_to_p_and_resolves_ties_to_smaller_prefix() -> None:
    sealed, _response = _surface("0", "1", 0.0)
    calibration_hash = _hash("policy-calibration")
    envelope_hash = _hash("policy-envelope")

    def calibrated(k: int, value: FavorableUtility) -> CalibratedPrefixCell:
        if k == 0:
            return CalibratedPrefixCell(
                sealed.cells[k],
                FavorableUtility.zeros(),
                FavorableUtility.zeros(),
                FavorableUtility.zeros(),
                calibration_hash,
                envelope_hash,
                0,
            )
        return CalibratedPrefixCell(
            sealed.cells[k],
            value,
            FavorableUtility.zeros(),
            value,
            calibration_hash,
            envelope_hash,
            1,
        )

    unsafe = (
        calibrated(0, FavorableUtility.zeros()),
        calibrated(1, FavorableUtility(-0.1, 0.1, 0.1)),
        calibrated(2, FavorableUtility(0.3, -0.1, 0.1)),
    )
    fallback = select_policy_prefix(sealed, unsafe)
    assert fallback.selected_k == 0
    assert not fallback.authorized

    tied = (
        calibrated(0, FavorableUtility.zeros()),
        calibrated(1, FavorableUtility(0.2, 0.1, 0.1)),
        calibrated(2, FavorableUtility(0.2, 0.1, 0.1)),
    )
    selected = select_policy_prefix(sealed, tied)
    assert selected.selected_k == 1
    assert selected.authorized
