from __future__ import annotations

import struct

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.compatibility_conditioned_directional_router import (
    ActionKind,
    ActionPrediction,
    AdmissionThresholds,
    BoundedActionEvidence,
    CandidateFeatureVector,
    Direction,
    EndpointBounds,
    EndpointCalibration,
    EndpointCalibrationCell,
    EndpointEffects,
    LearnabilityAdmission,
    OOFEndpointRow,
    ReplicaEnergyInput,
    RoutingDecision,
    SourceActionObservation,
    SourceAdmissionCandidate,
    SourceAdmissionCase,
    SupportPartitionReceipt,
    TargetAction,
    action_key,
    apply_endpoint_bounds,
    build_compatibility_receipts,
    build_label_free_opportunity,
    build_source_candidate_pool,
    build_target_candidate_pool,
    calibrate_endpoint_uncertainty,
    compose_directional_probability_bytes,
    compose_route,
    crossfit_source_predictions,
    evaluate_source_only_admission,
    fit_hurdle_pairwise_model,
    probability_hash,
    select_baseline_anchored_route,
)


SHA = "a" * 64
SHA_B = "b" * 64
CENTERS = ("A", "B", "C", "D", "E", "H")


def _f32(*values: float) -> tuple[bytes, ...]:
    return tuple(struct.pack("<f", value) for value in values)


def test_uncertainty_action_keys_are_direction_specific_for_uniform() -> None:
    pool = build_target_candidate_pool(
        outer_target_id="H", all_center_ids=CENTERS, bank_lock_hash=SHA
    )
    d01 = _target_feature(
        pool,
        action_id="U:D01",
        kind=ActionKind.U,
        direction=Direction.D01,
        source=None,
        probs=_f32(0.6, 0.8),
    )
    d10 = _target_feature(
        pool,
        action_id="U:D10",
        kind=ActionKind.U,
        direction=Direction.D10,
        source=None,
        probs=_f32(0.1, 0.4),
    )
    assert action_key(d01) == "U:D01"
    assert action_key(d10) == "U:D10"


def _replica(source: str, seed: int, query: float) -> ReplicaEnergyInput:
    return ReplicaEnergyInput(
        candidate_source_id=source,
        training_seed=seed,
        query_case_equal_energy=query,
        own_source_location=1.0,
        own_source_scale=0.5,
        checkpoint_hash=SHA,
        source_frame_hash=SHA_B,
        sampler_hash="c" * 64,
    )


def _feature(
    pool,
    *,
    case: str,
    action: str,
    kind: ActionKind,
    direction: Direction,
    candidate: str | None,
    x: float,
) -> CandidateFeatureVector:
    return CandidateFeatureVector(
        outer_target_id=pool.outer_target_id,
        query_center_id=pool.query_center_id,
        case_id=case,
        action_id=action,
        action_kind=kind,
        direction=direction,
        candidate_source_id=candidate,
        feature_names=("compatibility_proxy", "flip_fraction"),
        feature_values=(x, abs(x) / 4.0),
        candidate_pool_hash=pool.pool_hash,
        probability_hash=SHA,
        compatibility_receipt_hash=SHA_B if kind is ActionKind.HXE else None,
    )


def test_candidate_pools_enforce_role_complete_outer_exclusion() -> None:
    source = build_source_candidate_pool(
        outer_target_id="H",
        pseudo_query_id="A",
        all_center_ids=CENTERS,
        bank_lock_hash=SHA,
    )
    target = build_target_candidate_pool(
        outer_target_id="H", all_center_ids=CENTERS, bank_lock_hash=SHA
    )
    assert source.candidate_center_ids == ("B", "C", "D", "E")
    assert target.candidate_center_ids == ("A", "B", "C", "D", "E")
    with pytest.raises(ProtocolError, match="exact C-minus-H-minus-q"):
        type(source)(
            outer_target_id="H",
            query_center_id="A",
            all_center_ids=CENTERS,
            candidate_center_ids=("A", "B", "C", "D", "E"),
            bank_lock_hash=SHA,
        )


def test_compatibility_is_all_seed_label_free_hash_bound_and_ranked() -> None:
    pool = build_target_candidate_pool(
        outer_target_id="H", all_center_ids=("A", "B", "H"), bank_lock_hash=SHA
    )
    partition = SupportPartitionReceipt(
        center_id="H",
        support_case_ids=("s1", "s2"),
        evaluation_case_ids=("e1", "e2"),
        support_manifest_hash=SHA,
        evaluation_manifest_hash=SHA_B,
    )
    rows = tuple(
        _replica(source, seed, 1.0 + offset)
        for source, offset in (("A", 0.1), ("B", 0.6))
        for seed in (17, 42, 101)
    )
    receipts = build_compatibility_receipts(
        candidate_pool=pool,
        support_partition=partition,
        replica_energies=rows,
    )
    by_source = {row.candidate_source_id: row for row in receipts}
    assert by_source["A"].rank == 1
    assert by_source["A"].rank_margin > 0.0
    assert by_source["A"].exact_nelbo is False
    assert by_source["A"].labels_consumed is False
    assert tuple(row.training_seed for row in by_source["A"].replica_scores) == (17, 42, 101)
    with pytest.raises(ProtocolError, match="exactly seeds"):
        build_compatibility_receipts(
            candidate_pool=pool,
            support_partition=partition,
            replica_energies=rows[:-1],
        )


def _source_surface() -> tuple[SourceActionObservation, ...]:
    rows: list[SourceActionObservation] = []
    source_queries = ("A", "B", "C", "D", "E")
    for query_index, query in enumerate(source_queries):
        pool = build_source_candidate_pool(
            outer_target_id="H",
            pseudo_query_id=query,
            all_center_ids=CENTERS,
            bank_lock_hash=SHA,
        )
        candidates = pool.candidate_center_ids
        for case_index in range(3):
            case = f"{query}-case-{case_index}"
            x_u = -0.8 + 0.6 * case_index + 0.04 * query_index
            u = _feature(
                pool,
                case=case,
                action="U",
                kind=ActionKind.U,
                direction=Direction.ALL,
                candidate=None,
                x=x_u,
            )
            rows.append(
                SourceActionObservation(
                    feature=u,
                    candidate_pool=pool,
                    effects=EndpointEffects(
                        bacc_gain=0.09 * x_u,
                        brier_delta=-0.03 * max(x_u, 0.0),
                        log_delta=-0.02 * max(x_u, 0.0),
                    ),
                )
            )
            # The candidate identity changes by query and case.  There is no
            # frozen global action inventory for the model to rely on.
            candidate = candidates[(query_index + case_index) % len(candidates)]
            x_e = -0.35 + 0.75 * case_index + 0.03 * query_index
            direction = Direction.D01 if case_index % 2 == 0 else Direction.D10
            expert = _feature(
                pool,
                case=case,
                action=f"HXE:{candidate}:{direction.value}",
                kind=ActionKind.HXE,
                direction=direction,
                candidate=candidate,
                x=x_e,
            )
            rows.append(
                SourceActionObservation(
                    feature=expert,
                    candidate_pool=pool,
                    effects=EndpointEffects(
                        bacc_gain=0.12 * x_e + 0.015,
                        brier_delta=-0.035 * max(x_e, 0.0),
                        log_delta=-0.025 * max(x_e, 0.0),
                    ),
                )
            )
    return tuple(rows)


def test_candidate_aware_nested_lodo_fit_and_crossfit_exclude_both_roles() -> None:
    rows = _source_surface()
    model = fit_hurdle_pairwise_model(rows, outer_target_id="H")
    assert model.outer_target_id == "H"
    assert model.training_case_count == 15
    assert {row.held_center_id for row in model.fold_losses} == {"A", "B", "C", "D", "E"}
    oof = crossfit_source_predictions(rows, model=model)
    assert len(oof) == len(rows)
    assert all(row.held_center_id not in row.fold_training_query_ids for row in oof)
    assert all(row.held_center_id not in row.fold_training_candidate_ids for row in oof)
    assert all(row.prediction.feature.query_center_id == row.held_center_id for row in oof)


def test_endpoint_uncertainty_is_exact_grouped_and_has_no_pooled_fallback() -> None:
    rows = tuple(
        OOFEndpointRow(
            query_center_id=center,
            case_id=f"{center}-1",
            action_key="HXE:D01",
            comparator_key="B",
            predicted=EndpointEffects(0.2, -0.1, -0.1),
            observed=EndpointEffects(0.15 + offset, -0.08 + offset, -0.07 + offset),
            fold_model_hash=SHA,
        )
        for center, offset in (("A", 0.00), ("B", 0.01), ("C", -0.01))
    )
    calibration = calibrate_endpoint_uncertainty(rows, quantile=0.9)
    bounds = apply_endpoint_bounds(
        EndpointEffects(0.2, -0.1, -0.1),
        action_key="HXE:D01",
        comparator_key="B",
        calibration=calibration,
    )
    assert bounds.bacc_lcb < 0.2
    assert bounds.brier_ucb > -0.1
    with pytest.raises(ProtocolError, match="pooled fallback is forbidden"):
        apply_endpoint_bounds(
            EndpointEffects(0.2, -0.1, -0.1),
            action_key="HXE:D10",
            comparator_key="B",
            calibration=calibration,
        )


def _passing_admission() -> LearnabilityAdmission:
    cases = tuple(
        SourceAdmissionCase(
            query_center_id=center,
            case_id=f"{center}-{case}",
            candidates=(
                SourceAdmissionCandidate(
                    action_id="HXE:D01",
                    predicted_score=0.2 + case / 100.0,
                    opportunity_probability=0.8,
                    safe_selected=True,
                    observed=EndpointEffects(0.2 + case / 100.0, -0.02, -0.03),
                ),
            ),
        )
        for center in ("A", "B", "C", "D")
        for case in range(3)
    )
    return evaluate_source_only_admission(cases)


def test_source_only_admission_is_nonvacuous() -> None:
    admission = _passing_admission()
    assert admission.passed
    no_selection = tuple(
        SourceAdmissionCase(
            query_center_id=center,
            case_id=f"{center}-{case}",
            candidates=(
                SourceAdmissionCandidate(
                    action_id="HXE:D01",
                    predicted_score=0.2,
                    opportunity_probability=0.8,
                    safe_selected=False,
                    observed=EndpointEffects(0.2, -0.02, -0.03),
                ),
            ),
        )
        for center in ("A", "B", "C", "D")
        for case in range(3)
    )
    failed = evaluate_source_only_admission(no_selection)
    assert not failed.passed
    assert "ZERO_SAFE_SOURCE_SELECTIONS" in failed.reasons


def _target_feature(
    pool, *, action_id: str, kind: ActionKind, direction: Direction, source: str | None, probs
):
    return CandidateFeatureVector(
        outer_target_id="H",
        query_center_id="H",
        case_id="case-1",
        action_id=action_id,
        action_kind=kind,
        direction=direction,
        candidate_source_id=source,
        feature_names=("compatibility_proxy", "flip_fraction"),
        feature_values=(0.1, 0.2),
        candidate_pool_hash=pool.pool_hash,
        probability_hash=probability_hash(probs),
        compatibility_receipt_hash=SHA if kind is ActionKind.HXE else None,
    )


def _bounded(feature, *, safe: bool, score: float) -> BoundedActionEvidence:
    prediction = ActionPrediction(
        feature=feature,
        opportunity_probability=0.9,
        ranking_score=score,
        predicted_effects=EndpointEffects(0.2, -0.1, -0.1),
        model_hash=SHA,
    )
    return BoundedActionEvidence(
        prediction=prediction,
        comparator_key="B",
        bounds=(
            EndpointBounds(0.05, -0.01, -0.01)
            if safe
            else EndpointBounds(-0.01, -0.01, -0.01)
        ),
        uncertainty_calibration_hash=SHA_B,
    )


def test_hxe_safe_vs_b_is_not_vetoed_by_nonadmitted_u() -> None:
    pool = build_target_candidate_pool(
        outer_target_id="H", all_center_ids=CENTERS, bank_lock_hash=SHA
    )
    u_probs = _f32(0.1, 0.8)
    e_probs = _f32(0.9, 0.2)
    u = _target_feature(
        pool,
        action_id="U",
        kind=ActionKind.U,
        direction=Direction.ALL,
        source=None,
        probs=u_probs,
    )
    expert = _target_feature(
        pool,
        action_id="HXE:A:D01",
        kind=ActionKind.HXE,
        direction=Direction.D01,
        source="A",
        probs=e_probs,
    )
    decision = select_baseline_anchored_route(
        (_bounded(u, safe=False, score=1.0), _bounded(expert, safe=True, score=0.4)),
        admission=_passing_admission(),
        outer_target_id="H",
        case_id="case-1",
    )
    assert decision.enabled
    assert decision.selected_action_ids == ("HXE:A:D01",)
    assert decision.selected_direction is Direction.D01


def test_directional_composition_preserves_opposite_branch_and_exact_b_off() -> None:
    baseline = _f32(0.2, 0.8, 0.1, 0.9)
    expert = _f32(0.9, 0.1, 0.7, 0.2)
    assert compose_directional_probability_bytes(
        baseline,
        (expert,),
        weights=(1.0,),
        mixture_lambda=0.0,
        direction=Direction.D01,
    ) == baseline
    output = compose_directional_probability_bytes(
        baseline,
        (expert,),
        weights=(1.0,),
        mixture_lambda=0.5,
        direction=Direction.D01,
    )
    assert output[1] == baseline[1]
    assert output[3] == baseline[3]
    assert output[0] != baseline[0]

    admission = _passing_admission()
    fallback = RoutingDecision(
        outer_target_id="H",
        case_id="case-1",
        enabled=False,
        selected_direction=None,
        selected_action_ids=(),
        selected_weights=(),
        mixture_lambda=0.0,
        reason="OFF",
        admission_hash=admission.admission_hash,
        evidence_hashes=(),
    )
    result = compose_route(
        decision=fallback,
        baseline_sample_ids=("s1", "s2", "s3", "s4"),
        baseline_probability_bytes=baseline,
        actions=(),
    )
    assert result.output_probability_bytes == baseline
    assert result.receipt.exact_baseline_fallback


def test_label_free_opportunity_removes_threshold_noops() -> None:
    pool = build_target_candidate_pool(
        outer_target_id="H", all_center_ids=("A", "B", "H"), bank_lock_hash=SHA
    )
    baseline = _f32(0.2, 0.8)
    noop_probs = _f32(0.3, 0.7)
    flip_probs = _f32(0.7, 0.2)
    noop = TargetAction(
        feature=_target_feature(
            pool,
            action_id="HXE:A:D01",
            kind=ActionKind.HXE,
            direction=Direction.D01,
            source="A",
            probs=noop_probs,
        ),
        candidate_pool=pool,
        sample_ids=("s1", "s2"),
        probability_bytes=noop_probs,
        prediction_seal_hash=SHA,
    )
    flip = TargetAction(
        feature=_target_feature(
            pool,
            action_id="HXE:B:D01",
            kind=ActionKind.HXE,
            direction=Direction.D01,
            source="B",
            probs=flip_probs,
        ),
        candidate_pool=pool,
        sample_ids=("s1", "s2"),
        probability_bytes=flip_probs,
        prediction_seal_hash=SHA,
    )
    opportunity = build_label_free_opportunity(
        baseline_probability_bytes=baseline, actions=(noop, flip)
    )
    assert opportunity.member("HXE:A:D01").structural_noop
    assert opportunity.active_representative_ids == ("HXE:B:D01",)
