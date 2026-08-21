from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import inspect
from itertools import product
import multiprocessing as mp
import pickle

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.candidate_runtime import (
    CandidateRuntimeResult,
    build_case_candidates,
    directional_candidate_probabilities,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.canonical_probabilities import (
    CanonicalProbabilityVector,
    canonical_hash,
    exact_p_fallback,
    require_byte_exact_p,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.composition import (
    compose_center_probabilities,
    compose_exact_p,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.controls import (
    candidate_only_control,
    cyclically_shift_within_case,
    observed_maximum_bias,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.contracts import (
    EndpointCasePrediction,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.decision import (
    ABSTAIN_TO_P,
    ROUTE_PREFIX,
    make_route_decision,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.donor_replay_runtime import (
    realized_favorable_utility,
    replay_candidate,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.eligibility import (
    ActionCandidate,
    assess_action,
    select_best_eligible_action,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.policy_calibration import (
    PolicyReplay,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.policy_prefixes import (
    PrefixCandidate,
    enumerate_prefixes,
    select_prefix,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.policy_replay_runtime import (
    replay_pseudo_policy,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.persistence import (
    persist_dense_npz,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.posterior_contracts import (
    CasePosteriorPrediction,
    PseudoPosteriorReference,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.posterior_expected_utility import (
    FavorableUtility,
    PosteriorUtilityEstimate,
    compute_expected_utility,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.terminal_diagnostics import (
    favorable_terminal_contrast,
    score_terminal_metric,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.transport_geometry import (
    StructuralTransportGate,
    audit_numeric_transport,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.utility_calibration import (
    UtilityReplay,
    build_center_balanced_utility_calibration,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _spawn_echo(value: object) -> object:
    return value


def _candidate(
    case_id: str,
    utility: FavorableUtility,
    *,
    center: str = "H",
    alternative: str = "B",
    direction: str = "zero_to_one",
    probabilities: tuple[float, ...] = (0.8, 0.2),
    control: str = "IDENTITY",
) -> ActionCandidate:
    action_id = f"{alternative}::{direction}"
    estimate = PosteriorUtilityEstimate(
        center,
        case_id,
        action_id,
        direction,
        control,
        1,
        (utility,),
        utility,
        "a" * 64,
    )
    return ActionCandidate(
        center,
        case_id,
        alternative,
        direction,
        control,
        CanonicalProbabilityVector(probabilities),
        estimate,
    )


def _utility_replay(
    donor: str,
    residual: float,
    *,
    case: str,
    outer: str = "H",
) -> UtilityReplay:
    predicted = FavorableUtility(residual + 0.2, residual + 0.3, residual + 0.4)
    realized = FavorableUtility(0.2, 0.3, 0.4)
    return UtilityReplay(
        outer,
        donor,
        case,
        canonical_hash([outer, donor, case]),
        predicted,
        realized,
        tuple(sorted((outer, donor))),
        "IDENTITY",
    )


def _runtime_result(candidate: ActionCandidate) -> CandidateRuntimeResult:
    eligibility = assess_action(candidate)
    return CandidateRuntimeResult(
        candidate.center,
        candidate.center,
        candidate.case_id,
        candidate.control_id,
        6,
        5,
        (candidate,),
        (eligibility,),
        candidate if eligibility.eligible else None,
        1,
        "a" * 64,
        "b" * 64,
        (candidate.center,),
        "e" * 64,
    )


def test_exact_p_fallback_is_byte_exact_float32() -> None:
    p = np.asarray([0.1, 0.5, 0.90000004], dtype=np.float32)
    fallback = exact_p_fallback(p)
    assert fallback.dtype == np.float32
    assert fallback.view(np.uint32).tolist() == p.view(np.uint32).tolist()
    require_byte_exact_p(fallback, p)
    composition = compose_exact_p(p)
    assert composition.exact_p
    assert composition.probabilities.as_array().tobytes() == p.tobytes()


def test_endpoint_probability_contract_canonicalizes_reference_p_to_float32() -> None:
    raw = (0.123456789012345, 0.876543210987655)
    prediction = EndpointCasePrediction(
        "0",
        "case-1",
        ("sample-1", "sample-2"),
        {
            "B": raw,
            "I_OPPORTUNITY_GATED": raw,
            "R_NINE_ARM_ROBUST": raw,
            "P_PROTECTED": raw,
        },
        "a" * 64,
    )
    reference = np.asarray(prediction.probabilities["P_PROTECTED"], dtype=np.float32)
    fallback = compose_exact_p(prediction.probabilities["P_PROTECTED"])
    assert fallback.probabilities.as_array().view(np.uint32).tolist() == (
        reference.view(np.uint32).tolist()
    )


def test_posterior_eta_is_float32_before_scoring_and_persistence(tmp_path) -> None:
    raw = (0.123456789012345, 0.876543210987655)
    prediction = CasePosteriorPrediction(
        "0",
        "case-1",
        "IDENTITY",
        ("sample-1", "sample-2"),
        raw,
        "a" * 64,
        "b" * 64,
    )
    expected = np.asarray(raw, dtype=np.float32)
    in_memory = np.asarray(prediction.natural_probabilities, dtype=np.float32)
    assert in_memory.tobytes() == expected.tobytes()

    store = tmp_path / "posterior.npz"
    manifest = persist_dense_npz(
        store,
        {prediction.prediction_hash: prediction.natural_probabilities},
        role="test_target_posterior",
    )
    with np.load(store, allow_pickle=False) as persisted:
        on_disk = np.asarray(persisted[prediction.prediction_hash], dtype=np.float32)
    assert on_disk.tobytes() == in_memory.tobytes()
    assert manifest["arrays"][0]["dtype"] == "float32"


def test_pseudo_posterior_discloses_role_not_covariate_exclusion() -> None:
    reference = PseudoPosteriorReference(
        "0",
        "1",
        "case-1",
        "IDENTITY",
        "a" * 64,
    )
    payload = reference.to_payload()
    assert (
        payload[
            "outer_H_support_rows_or_labels_enter_J_minus_d_posterior_fit_or_normalization"
        ]
        is False
    )
    assert (
        payload["outer_H_frozen_label_free_expert_fingerprint_covariates_present"]
        is True
    )
    assert payload["posterior_is_outer_H_covariate_invariant"] is False
    assert payload["outer_H_specific_posterior_refit_performed"] is False
    assert "outer_H_enters_model_fit_or_normalization" not in payload


def test_posterior_expected_utility_matches_bruteforce_expectation() -> None:
    p = np.asarray([0.2, 0.8], dtype=np.float32)
    a = np.asarray([0.8, 0.2], dtype=np.float32)
    eta = np.asarray([0.75, 0.25], dtype=np.float64)
    expected = compute_expected_utility(
        p,
        a,
        eta,
        support_n_positive=2.0,
        support_n_negative=2.0,
        support_row_count=4,
    )
    brute = np.zeros(3, dtype=np.float64)
    for labels in product((0, 1), repeat=2):
        weight = np.prod(
            [eta[index] if label else 1.0 - eta[index] for index, label in enumerate(labels)]
        )
        realized = realized_favorable_utility(
            p,
            a,
            labels,
            center_n_positive=3,
            center_n_negative=3,
            center_row_count=6,
        )
        brute += weight * np.asarray(realized.as_tuple())
    assert expected.as_tuple() == pytest.approx(tuple(brute), abs=1.0e-12)
    assert all(value > 0.0 for value in expected.as_tuple())


def test_directional_actions_only_replace_the_requested_crossings() -> None:
    p = np.asarray([0.2, 0.8, 0.4, 0.7], dtype=np.float32)
    alternative = np.asarray([0.9, 0.1, 0.3, 0.9], dtype=np.float32)
    up, up_mask = directional_candidate_probabilities(p, alternative, "zero_to_one")
    down, down_mask = directional_candidate_probabilities(p, alternative, "one_to_zero")
    assert up_mask.tolist() == [True, False, False, False]
    assert down_mask.tolist() == [False, True, False, False]
    assert up.tolist() == pytest.approx([0.9, 0.8, 0.4, 0.7])
    assert down.tolist() == pytest.approx([0.2, 0.1, 0.4, 0.7])


def test_case_runtime_uses_one_posterior_fit_and_b_i_r_tie_order() -> None:
    p = np.asarray([0.2, 0.8, 0.2, 0.8], dtype=np.float32)
    alternative = np.asarray([0.8, 0.2, 0.8, 0.2], dtype=np.float32)
    result = build_case_candidates(
        center="H",
        case_id="c",
        portfolio_probabilities=p,
        alternative_probabilities={
            "B": alternative,
            "I_OPPORTUNITY_GATED": alternative,
            "R_NINE_ARM_ROBUST": alternative,
        },
        posterior_eta=np.asarray([0.9, 0.1, 0.9, 0.1]),
        support_n_positive=10.0,
        support_n_negative=10.0,
        support_row_count=20,
    )
    assert result.posterior_model_reference_count == 1
    assert result.to_payload()["posterior_fit_increment"] == 0
    assert result.to_payload()["posterior_refit"] is False
    assert result.descriptor_count == 6
    assert result.no_crossing_count == 0
    assert result.selected_candidate is not None
    assert result.selected_candidate.alternative_id == "B"
    payload = result.to_payload()
    assert payload["support_labels_used_indirectly"] is True
    assert payload["held_case_label_used"] is False
    assert pickle.loads(pickle.dumps(result)) == result


def test_candidate_runtime_crosses_spawn_boundary() -> None:
    p = np.asarray([0.2, 0.8], dtype=np.float32)
    alternative = np.asarray([0.8, 0.2], dtype=np.float32)
    result = build_case_candidates(
        center="H",
        case_id="c",
        portfolio_probabilities=p,
        alternative_probabilities={
            "B": alternative,
            "I_OPPORTUNITY_GATED": alternative,
            "R_NINE_ARM_ROBUST": alternative,
        },
        posterior_eta=np.asarray([0.9, 0.1]),
        support_n_positive=4.0,
        support_n_negative=4.0,
        support_row_count=8,
    )
    try:
        with ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context("spawn")
        ) as executor:
            spawned = executor.submit(_spawn_echo, result).result(timeout=30)
    except (OSError, PermissionError) as exc:
        pytest.skip(f"OS spawn boundary is unavailable: {exc}")
    assert spawned == result


def test_proper_unsafe_action_is_ineligible_and_never_selected() -> None:
    unsafe = _candidate("c1", FavorableUtility(0.2, -0.01, 0.1))
    safe = _candidate(
        "c1",
        FavorableUtility(0.1, 0.01, 0.01),
        alternative="I_OPPORTUNITY_GATED",
    )
    assert not assess_action(unsafe).eligible
    assert select_best_eligible_action((unsafe, safe)) == safe


def test_prefix_selection_uses_aggregate_proper_guards_and_smaller_k_tie() -> None:
    c1 = PrefixCandidate(
        _candidate("a", FavorableUtility(0.3, 0.1, 0.1)),
        FavorableUtility(0.3, 0.1, 0.1),
        "1" * 64,
    )
    c2 = PrefixCandidate(
        _candidate("b", FavorableUtility(0.2, -0.3, 0.1)),
        FavorableUtility(0.2, -0.3, 0.1),
        "1" * 64,
    )
    c3 = PrefixCandidate(
        _candidate("c", FavorableUtility(0.1, 0.3, 0.3)),
        FavorableUtility(0.1, 0.3, 0.3),
        "1" * 64,
    )
    selection = select_prefix((c3, c2, c1))
    assert tuple(row.candidate.case_id for row in selection.ranked_candidates) == (
        "a",
        "b",
        "c",
    )
    assert selection.evaluations[1].feasible
    assert not selection.evaluations[2].feasible
    assert selection.selected_k == 3
    assert selection.to_payload()["ranked_candidates"][0]["policy_hash"] == c1.policy_hash

    tie = PrefixCandidate(
        _candidate("z", FavorableUtility(0.0, 0.0, 0.0)),
        FavorableUtility(0.0, 0.0, 0.0),
        "2" * 64,
    )
    smaller = select_prefix((c1, tie))
    assert smaller.selected_k == 1


def test_center_balanced_bias_prevents_case_rich_center_domination() -> None:
    rows = [
        _utility_replay("D1", 1.0, case=f"rich-{index}")
        for index in range(50)
    ]
    rows.extend(
        _utility_replay(f"D{index}", 0.1, case=f"small-{index}")
        for index in range(2, 7)
    )
    calibration = build_center_balanced_utility_calibration(
        rows, outer_center="H"
    )
    assert calibration.bias.as_tuple() == pytest.approx((0.1, 0.1, 0.1))
    assert len(calibration.supported_donor_centers) == 6
    with pytest.raises(ProtocolError, match="fewer than six"):
        build_center_balanced_utility_calibration(
            tuple(row for row in rows if row.donor_center != "D6"),
            outer_center="H",
        )


def test_leave_j_calibration_rejects_any_excluded_donor() -> None:
    rows = tuple(
        _utility_replay(f"D{index}", 0.1, case=f"c{index}")
        for index in range(1, 8)
    )
    with pytest.raises(ProtocolError, match="excluded donor"):
        build_center_balanced_utility_calibration(
            rows,
            outer_center="H",
            calibration_excluded_centers=("H", "D7"),
        )
    calibration = build_center_balanced_utility_calibration(
        rows[:-1],
        outer_center="H",
        calibration_excluded_centers=("H", "D7"),
    )
    assert len(calibration.supported_donor_centers) == 6


def test_donor_replay_requires_exact_capability_scope() -> None:
    candidate = _candidate(
        "case-1",
        FavorableUtility(0.1, 0.1, 0.1),
        center="J",
        probabilities=(0.8, 0.2),
    )
    expected_scope = "PSEUDO_EVALUATION::H=H::J=J::excluded_d=case-1"
    result = replay_candidate(
        candidate,
        portfolio_probabilities=(0.2, 0.8),
        labels=(1, 0),
        outer_center="H",
        donor_center="J",
        center_n_positive=2,
        center_n_negative=2,
        center_row_count=4,
        label_scope=expected_scope,
        source_excluded_centers=("H", "J"),
        endpoint_lineage_hash="e" * 64,
    )
    assert result.label_scope == expected_scope
    with pytest.raises(ProtocolError, match="scope drifted"):
        replay_candidate(
            candidate,
            portfolio_probabilities=(0.2, 0.8),
            labels=(1, 0),
            outer_center="H",
            donor_center="J",
            center_n_positive=2,
            center_n_negative=2,
            center_row_count=4,
            label_scope="PSEUDO_EVALUATION::case-1",
            source_excluded_centers=("H", "J"),
            endpoint_lineage_hash="e" * 64,
        )


def test_pseudo_policy_requires_outer_h_removed_from_endpoint_sources() -> None:
    p = np.asarray([0.2, 0.8], dtype=np.float32)
    alternative = np.asarray([0.8, 0.2], dtype=np.float32)
    kwargs = dict(
        center="J",
        case_id="d",
        portfolio_probabilities=p,
        alternative_probabilities={
            "B": alternative,
            "I_OPPORTUNITY_GATED": alternative,
            "R_NINE_ARM_ROBUST": alternative,
        },
        posterior_eta=np.asarray([0.9, 0.1]),
        support_n_positive=4.0,
        support_n_negative=4.0,
        support_row_count=8,
        outer_center="H",
    )
    with pytest.raises(ProtocolError, match="exact outer-H/target-J"):
        build_case_candidates(**kwargs, source_excluded_centers=("J",))
    result = build_case_candidates(
        **kwargs,
        source_excluded_centers=("H", "J"),
        endpoint_lineage_hash="e" * 64,
    )
    assert result.selected_candidate is not None
    calibration = build_center_balanced_utility_calibration(
        tuple(
            _utility_replay(f"K{index}", 0.001, case=f"k{index}")
            for index in range(6)
        ),
        outer_center="H",
        calibration_excluded_centers=("H", "J"),
    )
    replay = replay_pseudo_policy(
        (result,),
        {
            result.selected_candidate.action_hash:
                result.selected_candidate.estimate.utility
        },
        outer_center="H",
        donor_center="J",
        leave_j_candidate_calibration=calibration,
    )
    assert replay.replay.donor_center == "J"
    assert set(replay.replay.calibration_excluded_centers) == {"H", "J"}


def test_zero_mad_transport_is_dropped_and_recorded_as_novelty() -> None:
    audit = audit_numeric_transport(
        target_center="H",
        target_vector=(2.0, 2.5, 1.0),
        reference_vectors_by_center={
            "D1": ((1.0, 1.0, 0.0), (1.0, 1.2, 0.0)),
            "D2": ((1.0, 2.0, 0.0),),
            "D3": ((1.0, 3.0, 0.0),),
        },
        feature_names=("constant", "continuous", "sparse"),
        sparse_feature_names=("sparse",),
    )
    assert audit.active_continuous_dimension_count == 1
    assert audit.zero_scale_dimensions == ("constant", "sparse")
    assert audit.zero_scale_novelty_dimensions == ("constant", "sparse")
    assert audit.l2_distance < 10.0
    assert audit.to_payload()["authorization_gate"] is False


def test_cyclic_control_is_deterministic_and_nonzero() -> None:
    assert cyclically_shift_within_case((1.0, 2.0, 3.0)) == (3.0, 1.0, 2.0)
    assert cyclically_shift_within_case((1.0, 2.0, 3.0)) == cyclically_shift_within_case(
        (1.0, 2.0, 3.0)
    )
    with pytest.raises(ProtocolError, match="derangement"):
        cyclically_shift_within_case((1.0, 2.0, 3.0), shift=3)


def test_composition_changes_only_selected_complete_cases() -> None:
    p = np.asarray([0.2, 0.8, 0.3, 0.7], dtype=np.float32)
    candidate = _candidate(
        "c1", FavorableUtility(0.2, 0.1, 0.1), probabilities=(0.8, 0.2)
    )
    result = compose_center_probabilities(p, ("c1", "c1", "c2", "c2"), (candidate,))
    assert result.probabilities.values == pytest.approx((0.8, 0.2, 0.3, 0.7))
    assert result.changed_probability_count == 2


def test_primary_decision_routes_only_when_structural_and_aggregate_gates_pass() -> None:
    rows = tuple(
        _utility_replay(f"D{index}", 0.01, case=f"c{index}")
        for index in range(6)
    )
    utility_calibration = build_center_balanced_utility_calibration(
        rows, outer_center="H"
    )
    candidate = _candidate(
        "c1", FavorableUtility(0.2, 0.1, 0.1), probabilities=(0.8, 0.2)
    )
    transport = StructuralTransportGate("H", True, True, True, True, True)
    decision = make_route_decision(
        center="H",
        portfolio_probabilities=(0.2, 0.8, 0.3, 0.7),
        sample_case_ids=("c1", "c1", "c2", "c2"),
        candidate_results=(_runtime_result(candidate),),
        utility_calibration=utility_calibration,
        structural_transport=transport,
    )
    assert decision.action == ROUTE_PREFIX
    assert decision.prefix_selection.selected_k == 1
    blocked = make_route_decision(
        center="H",
        portfolio_probabilities=(0.2, 0.8, 0.3, 0.7),
        sample_case_ids=("c1", "c1", "c2", "c2"),
        candidate_results=(_runtime_result(candidate),),
        utility_calibration=utility_calibration,
        structural_transport=StructuralTransportGate(
            "H", True, True, True, False, True
        ),
    )
    assert blocked.action == ABSTAIN_TO_P
    assert blocked.composition.exact_p


def test_controls_and_terminal_metrics_use_favorable_signs() -> None:
    candidate = _candidate("c", FavorableUtility(0.2, 0.1, 0.05))
    control = candidate_only_control((candidate,))
    assert control.authorized
    replay = _utility_replay("D", 0.3, case="c")
    assert observed_maximum_bias((replay,)).as_tuple() == pytest.approx(
        (0.3, 0.3, 0.3)
    )
    assert "policy_bias" not in inspect.signature(enumerate_prefixes).parameters
    assert "policy_bias" not in inspect.signature(select_prefix).parameters
    baseline = score_terminal_metric(
        center="H", method_id="P", labels=(1, 0), probabilities=(0.4, 0.6)
    )
    routed = score_terminal_metric(
        center="H", method_id="R", labels=(1, 0), probabilities=(0.8, 0.2)
    )
    contrast = favorable_terminal_contrast(baseline, routed)
    assert all(value > 0.0 for value in contrast.as_tuple())
