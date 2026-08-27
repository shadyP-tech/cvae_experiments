from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import pickle

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.pairwise_primitive_utility.admission import (
    evaluate_source_only_admission,
    seal_admission_decision,
)
from midogpp_thesis.cvae.routing.pairwise_primitive_utility.clustered_uncertainty import (
    apply_calibrated_bound,
    calibrate_clustered_uncertainty,
)
from midogpp_thesis.cvae.routing.pairwise_primitive_utility.contracts import (
    ActionQuery,
    ActionSelectionEvidence,
    ActionSurface,
    ActionUtilityObservation,
    AdmissionCandidate,
    AdmissionCase,
    AdmissionDecisionReceipt,
    BaccRankingPolicy,
    CalibratedBound,
    CandidatePoolReceipt,
    ExpectedDenominators,
    OOFResidualObservation,
    OpportunityCaseReceipt,
    P_ACTION_ID,
    PrimitiveUtility,
    RowPosteriorObservation,
    RowPosteriorPrediction,
    SourceScopeReceipt,
)
from midogpp_thesis.cvae.routing.pairwise_primitive_utility.opportunity import (
    build_opportunity_case_receipt,
    build_opportunity_set,
)
from midogpp_thesis.cvae.routing.pairwise_primitive_utility.pairwise_ranker import (
    PAIRWISE_ALPHA_GRID,
    fit_pairwise_ranker,
    predict_action_score,
    predict_pairwise_contrast,
)
from midogpp_thesis.cvae.routing.pairwise_primitive_utility.primitive_utility import (
    build_expected_denominators,
    expected_additive_utility,
    normalize_expected_utility,
    sum_primitives,
)
from midogpp_thesis.cvae.routing.pairwise_primitive_utility.row_posterior import (
    assert_label_free_feature_names,
    crossfit_source_row_posterior,
    fit_final_source_row_posterior,
    fit_source_row_posterior,
)
from midogpp_thesis.cvae.routing.pairwise_primitive_utility.selection import (
    assemble_action_selection_evidence,
    select_fail_closed_action,
)


def _scope(*, k: str, j: str, ell: str, centers: tuple[str, ...]) -> SourceScopeReceipt:
    training = tuple(center for center in centers if center not in {j, k, ell})
    return SourceScopeReceipt(
        outer_target_center="H",
        query_center=j,
        hyperparameter_center=k,
        calibration_center=ell,
        heldout_case_center=j,
        heldout_case_id="target-d",
        training_center_ids=training,
        training_case_keys=tuple(
            (center, f"case-{center}-{case}")
            for center in training
            for case in range(3)
        ),
    )


def _posterior_predictions(values, *, model="posterior-model", scope="posterior-receipt"):
    return tuple(RowPosteriorPrediction(value, model, scope) for value in values)


def test_opportunity_excludes_no_crossing_and_collapses_crossing_equivalence() -> None:
    action_ids = ("NOOP", "A", "B", "C", "D")
    result = build_opportunity_set(
        (0.4, 0.6),
        (
            ActionSurface("NOOP", "N", "none", (0.3, 0.7)),
            ActionSurface("A", "F", "up", (0.7, 0.6)),
            ActionSurface("B", "F", "up", (0.7, 0.6)),
            ActionSurface("C", "G", "up", (0.8, 0.6)),
            ActionSurface("D", "F", "down", (0.4, 0.2)),
        ),
        candidate_action_ids=action_ids,
    )
    assert result.member("NOOP").structural_noop
    assert not result.member("NOOP").exact_p_probability
    assert result.active_representative_ids == ("A", "C", "D")
    assert result.equivalent_action_ids("A") == ("A", "B")
    assert result == build_opportunity_set(
        (0.4, 0.6),
        tuple(
            reversed(
                (
                    ActionSurface("NOOP", "N", "none", (0.3, 0.7)),
                    ActionSurface("A", "F", "up", (0.7, 0.6)),
                    ActionSurface("B", "F", "up", (0.7, 0.6)),
                    ActionSurface("C", "G", "up", (0.8, 0.6)),
                    ActionSurface("D", "F", "down", (0.4, 0.2)),
                )
            )
        ),
        candidate_action_ids=action_ids,
    )
    with pytest.raises(ProtocolError, match="candidate actions"):
        build_opportunity_set(
            (0.4, 0.6),
            (ActionSurface("A", "F", "up", (0.7, 0.6)),),
            candidate_action_ids=("A", "B"),
        )


def test_expected_primitives_match_direct_bernoulli_expectation_and_add() -> None:
    p = np.asarray([0.2, 0.7, 0.4])
    q = np.asarray([0.8, 0.6, 0.1])
    eta = np.asarray([0.25, 0.8, 0.4])
    predictions = _posterior_predictions(eta)
    primitive = expected_additive_utility(
        p,
        q,
        predictions,
        action_id="A",
        scope_id="scope",
        row_manifest_hash="all-rows",
    )
    direct_brier = 0.0
    direct_log = 0.0
    for baseline, candidate, probability in zip(p, q, eta, strict=True):
        for label, mass in ((0, 1.0 - probability), (1, probability)):
            direct_brier += mass * ((candidate - label) ** 2 - (baseline - label) ** 2)
            direct_log += mass * (
                -(label * np.log(candidate) + (1 - label) * np.log(1 - candidate))
                + label * np.log(baseline)
                + (1 - label) * np.log(1 - baseline)
            )
    assert primitive.delta_brier_sum == pytest.approx(direct_brier)
    assert primitive.delta_log_sum == pytest.approx(direct_log)
    pieces = tuple(
        expected_additive_utility(
            (p[index],),
            (q[index],),
            (predictions[index],),
            action_id="A",
            scope_id="scope",
            row_manifest_hash=f"row-{index}",
        )
        for index in range(3)
    )
    aggregate = sum_primitives(pieces)
    assert aggregate.delta_tp == pytest.approx(primitive.delta_tp)
    assert aggregate.delta_tn == pytest.approx(primitive.delta_tn)
    assert aggregate.delta_brier_sum == pytest.approx(primitive.delta_brier_sum)
    assert aggregate.delta_log_sum == pytest.approx(primitive.delta_log_sum)
    with pytest.raises(ProtocolError, match="distinct case row manifests"):
        _ = primitive + primitive
    with pytest.raises(ProtocolError, match="typed row-posterior"):
        expected_additive_utility(
            p, q, eta, action_id="A", scope_id="scope", row_manifest_hash="rows"
        )
    with pytest.raises(ProtocolError, match="mixed model or scope"):
        build_expected_denominators(
            (*predictions[:2], replace(predictions[2], model_hash="other")),
            scope_id="scope",
            row_manifest_hash="rows",
        )


def test_one_action_invariant_expected_denominator_scope() -> None:
    predictions = _posterior_predictions((0.2, 0.8, 0.6, 0.4))
    denominator = build_expected_denominators(
        predictions,
        scope_id="center-H",
        row_manifest_hash="rows-H",
    )
    left = normalize_expected_utility(
        expected_additive_utility(
            (0.4, 0.6, 0.3, 0.7),
            (0.7, 0.2, 0.8, 0.1),
            predictions,
            action_id="LEFT",
            scope_id="center-H",
            row_manifest_hash="rows-H",
        ),
        denominator,
    )
    right = normalize_expected_utility(
        expected_additive_utility(
            (0.4, 0.6, 0.3, 0.7),
            (0.6, 0.3, 0.9, 0.2),
            predictions,
            action_id="RIGHT",
            scope_id="center-H",
            row_manifest_hash="rows-H",
        ),
        denominator,
    )
    assert left.denominator_eta_hash == right.denominator_eta_hash
    assert left.denominator_scope_id == right.denominator_scope_id == "center-H"
    assert left.response_hash != right.response_hash
    with pytest.raises(ProtocolError, match="exact scope"):
        normalize_expected_utility(
            expected_additive_utility(
                (0.4, 0.6, 0.3, 0.7),
                (0.7, 0.2, 0.8, 0.1),
                predictions,
                action_id="LEFT",
                scope_id="other",
                row_manifest_hash="rows-H",
            ),
            denominator,
        )
    with pytest.raises(ProtocolError, match="exact scope"):
        normalize_expected_utility(
            expected_additive_utility(
                (0.4, 0.6, 0.3, 0.7),
                (0.7, 0.2, 0.8, 0.1),
                tuple(
                    replace(row, source_scope_receipt_hash="wrong-posterior")
                    for row in predictions
                ),
                action_id="LEFT",
                scope_id="center-H",
                row_manifest_hash="rows-H",
            ),
            denominator,
        )


def _posterior_rows() -> tuple[RowPosteriorObservation, ...]:
    rows = []
    for center_index, center in enumerate(tuple("123456")):
        for case in range(3):
            for row_index in range(2):
                probability = 0.15 + 0.1 * case + 0.05 * row_index + 0.01 * center_index
                rows.append(
                    RowPosteriorObservation(
                        center,
                        f"case-{center}-{case}",
                        f"row-{center}-{case}-{row_index}",
                        ("protected_probability", "absolute_margin"),
                        (probability, abs(probability - 0.5)),
                        int(probability + 0.35 * ((case + row_index) % 2) > 0.45),
                    )
                )
    return tuple(rows)


def test_row_posterior_feature_firewall_and_hjkl_receipt() -> None:
    assert assert_label_free_feature_names(("protected_probability", "ensemble_disagreement"))
    with pytest.raises(ProtocolError, match="Forbidden"):
        assert_label_free_feature_names(("target_center",))
    with pytest.raises(ProtocolError, match="distinct"):
        SourceScopeReceipt(
            "H", "J", "K", "K", "J", "d", ("1", "2"), (("1", "c1"), ("2", "c2"))
        )
    with pytest.raises(ProtocolError, match="exact H/J/K/L/d"):
        SourceScopeReceipt(
            "H", "J", "K", "L", "OTHER", "d", ("1", "2"), (("1", "c1"), ("2", "c2"))
        )


def test_scope_case_keys_do_not_collapse_same_named_cases_across_centers() -> None:
    rows = tuple(
        RowPosteriorObservation(
            center,
            case,
            f"row-{center}-{ordinal}",
            ("protected_probability",),
            (0.2 + 0.15 * ordinal,),
            ordinal % 2,
        )
        for ordinal, (center, case) in enumerate(
            (("1", "shared"), ("1", "other"), ("2", "shared"), ("2", "other"))
        )
    )
    scope = SourceScopeReceipt(
        "H",
        "J",
        "K",
        "L",
        "J",
        "shared",
        ("1", "2"),
        (("1", "shared"), ("1", "other"), ("2", "shared"), ("2", "other")),
    )
    model = fit_source_row_posterior(rows, scope=scope)
    assert model.training_case_count == 4


def test_row_posterior_crossfit_is_label_free_deterministic_and_final_refits_c_minus_h() -> None:
    rows = _posterior_rows()
    first = crossfit_source_row_posterior(rows, outer_target_center="H")
    second = crossfit_source_row_posterior(tuple(reversed(rows)), outer_target_center="H")
    assert first == second
    assert len(first) == len(rows)
    assert not hasattr(first[0], "outcome")
    assert len({row.model_hash for row in first}) < len(first)  # one model reused per held case
    final = fit_final_source_row_posterior(
        rows,
        outer_target_center="H",
        fixed_capacity_receipt_hash="fixed-capacity",
    )
    assert final.training_center_count == 6
    assert not hasattr(final, "training_outcomes")


def _pairwise_rows() -> tuple[ActionUtilityObservation, ...]:
    rows = []
    for center_index, center in enumerate(tuple("12345678")):
        for case in range(3):
            margin = -0.6 + 0.3 * case + 0.02 * center_index
            for action_id, family, direction, shift in (
                ("A_UP", "A", "up", 0.35),
                ("B_DOWN", "B", "down", -0.15),
            ):
                rows.append(
                    ActionUtilityObservation(
                        center,
                        f"case-{center}-{case}",
                        action_id,
                        family,
                        direction,
                        ("crossing_fraction", "absolute_margin"),
                        (0.2 + 0.1 * case, abs(margin)),
                        _normalized_response(
                            center=center,
                            case=case,
                            action_id=action_id,
                            bacc=shift + (0.25 if action_id == "A_UP" else -0.1) * margin,
                        ),
                        "source-surface",
                        _candidate_pool().receipt_hash,
                        _opportunity_receipt(center, case).receipt_hash,
                    )
                )
    return tuple(rows)


def _candidate_pool() -> CandidatePoolReceipt:
    centers = ("H", *tuple("12345678"))
    return CandidatePoolReceipt(
        outer_target_center="H",
        all_center_ids=centers,
        candidate_center_ids=tuple("12345678"),
        expert_inventory=tuple((f"expert-{center}", center) for center in tuple("12345678")),
        bank_lock_hash="fixed-bank-lock",
        source_surface_receipt_hash="source-surface",
    )


def _normalized_response(*, center: str, case: int, action_id: str, bacc: float):
    scope_id = f"response::{center}::{case}"
    manifest = f"rows::{center}::{case}"
    opportunity = _opportunity_receipt(center, case).opportunity
    member = opportunity.member(action_id)
    denominator = ExpectedDenominators(
        scope_id,
        1.0,
        1.0,
        2,
        "eta-hash",
        manifest,
        "posterior-model",
        "posterior-receipt",
    )
    primitive = PrimitiveUtility(
        2.0 * bacc,
        0.0,
        -0.04,
        -0.02,
        2,
        action_id,
        opportunity.baseline_hash,
        member.probability_hash,
        scope_id,
        manifest,
        "posterior-model",
        "posterior-receipt",
    )
    return normalize_expected_utility(primitive, denominator)


def _pairwise_scopes() -> tuple[SourceScopeReceipt, ...]:
    centers = tuple("12345678")
    return tuple(_scope(k=k, j=centers[(i + 1) % 8], ell=centers[(i + 2) % 8], centers=centers) for i, k in enumerate(centers))


def _opportunity_receipt(center: str, case: int) -> OpportunityCaseReceipt:
    opportunity = build_opportunity_set(
        (.4, .6),
        (
            ActionSurface("A_UP", "A", "up", (.8, .6)),
            ActionSurface("B_DOWN", "B", "down", (.4, .2)),
        ),
        candidate_action_ids=("A_UP", "B_DOWN"),
    )
    return build_opportunity_case_receipt(center_id=center, case_id=f"case-{center}-{case}", opportunity=opportunity)


def _opportunity_receipts() -> tuple[OpportunityCaseReceipt, ...]:
    return tuple(_opportunity_receipt(center, case) for center in tuple("12345678") for case in range(3))


def _fit_pairwise(rows=None, scopes=None):
    return fit_pairwise_ranker(rows or _pairwise_rows(), delete_center_scopes=scopes or _pairwise_scopes(), candidate_pool=_candidate_pool(), opportunity_receipts=_opportunity_receipts(), ranking_policy=BaccRankingPolicy())


def test_pairwise_ranker_is_action_specific_nested_worst_center_and_antisymmetric() -> None:
    model = _fit_pairwise()
    assert model.selected_alpha in PAIRWISE_ALPHA_GRID
    assert model.candidate_pool_receipt_hash == _candidate_pool().receipt_hash
    assert len(model.delete_center_losses) == len(PAIRWISE_ALPHA_GRID) * 8
    assert any(name.startswith("action_feature::A_UP") for name in model.design_names)
    assert any(name.startswith("family_feature::A") for name in model.design_names)
    assert any(name.startswith("direction_feature::up") for name in model.design_names)
    left = ActionQuery(
        "A_UP", "A", "up", model.feature_names, (0.4, 0.2)
    )
    right = ActionQuery(
        "B_DOWN", "B", "down", model.feature_names, (0.4, 0.2)
    )
    forward = predict_pairwise_contrast(model, left, right)
    reverse = predict_pairwise_contrast(model, right, left)
    assert forward.mean_contrast == -reverse.mean_contrast
    assert predict_action_score(model, ActionQuery.p_anchor(model.feature_names)) == 0.0
    assert forward.mean_contrast > 0.0
    first = _pairwise_scopes()[0]
    real_d = f"case-{first.query_center}-0"
    real_d_scope = SourceScopeReceipt(
        first.outer_target_center,
        first.query_center,
        first.hyperparameter_center,
        first.calibration_center,
        first.query_center,
        real_d,
        first.training_center_ids,
        first.training_case_keys,
    )
    refit = fit_pairwise_ranker(
        _pairwise_rows(),
        delete_center_scopes=(real_d_scope, *_pairwise_scopes()[1:]),
        candidate_pool=_candidate_pool(),
        opportunity_receipts=_opportunity_receipts(),
        ranking_policy=BaccRankingPolicy(),
    )
    assert refit.training_case_count == model.training_case_count


def test_candidate_pool_and_pairwise_response_lineage_poison_fail_closed() -> None:
    pool = _candidate_pool()
    rows = _pairwise_rows()
    scopes = _pairwise_scopes()
    with pytest.raises(ProtocolError, match="exact C-minus-H"):
        CandidatePoolReceipt(
            "H",
            pool.all_center_ids,
            tuple("1234567"),
            tuple((f"expert-{center}", center) for center in tuple("1234567")),
            "fixed-bank-lock",
            "source-surface",
        )
    with pytest.raises(ProtocolError, match="outer-target H"):
        _fit_pairwise(
            (replace(rows[0], center_id="H"), *rows[1:]),
            scopes,
        )
    with pytest.raises(ProtocolError, match="absent source center|exact C-minus-H"):
        _fit_pairwise(
            tuple(row for row in rows if row.center_id != "8"),
            scopes,
        )
    with pytest.raises(ProtocolError, match="lineage"):
        _fit_pairwise(
            (
                replace(
                    rows[0],
                    response=_normalized_response(
                        center="1", case=99, action_id=rows[0].action_id, bacc=0.2
                    ),
                ),
                *rows[1:],
            ),
            scopes,
        )
    with pytest.raises(ProtocolError, match="cross-surface"):
        _fit_pairwise((replace(rows[0], response=rows[1].response), *rows[1:]), scopes)
    with pytest.raises(ProtocolError, match="lineage"):
        _fit_pairwise(
            (replace(rows[0], source_scope_receipt_hash="wrong-source"), *rows[1:]),
            scopes,
        )
    with pytest.raises(ProtocolError, match="lineage"):
        _fit_pairwise(
            (replace(rows[0], candidate_pool_receipt_hash="wrong-pool"), *rows[1:]),
            scopes,
        )


def _calibration_scopes() -> tuple[SourceScopeReceipt, ...]:
    centers = tuple("12345678")
    return (
        _scope(k="1", j="2", ell="3", centers=centers),
        _scope(k="2", j="3", ell="4", centers=centers),
        _scope(k="3", j="4", ell="5", centers=centers),
        _scope(k="4", j="5", ell="6", centers=centers),
    )


def test_uncertainty_is_center_oof_action_pair_specific_and_one_sided() -> None:
    scopes = _calibration_scopes()
    rows = []
    for scope_index, scope in enumerate(scopes):
        for case in range(2):
            for metric, predicted, observed in (
                ("bacc", 0.10, 0.08 - 0.005 * scope_index),
                ("brier", -0.04, -0.03 + 0.002 * case),
                ("log", -0.03, -0.02 + 0.002 * case),
                ("pairwise", 0.12, 0.10 - 0.005 * case),
            ):
                rows.append(
                    OOFResidualObservation(
                        scope.calibration_center,
                        f"L-{scope.calibration_center}-{case}",
                        scope.calibration_center,
                        "A_UP",
                        P_ACTION_ID,
                        metric,
                        predicted,
                        observed,
                        scope.receipt_hash,
                    )
                )
    calibration = calibrate_clustered_uncertainty(rows, calibration_scopes=scopes)
    assert calibration.outer_target_center == "H"
    lower = apply_calibrated_bound(
        calibration,
        action_id="A_UP",
        comparator_id=P_ACTION_ID,
        metric="bacc",
        mean=0.1,
    )
    upper = apply_calibrated_bound(
        calibration,
        action_id="A_UP",
        comparator_id=P_ACTION_ID,
        metric="brier",
        mean=-0.04,
    )
    assert lower.bound < lower.mean
    assert upper.bound > upper.mean
    with pytest.raises(ProtocolError, match="Missing"):
        apply_calibrated_bound(
            calibration,
            action_id="B_DOWN",
            comparator_id=P_ACTION_ID,
            metric="bacc",
            mean=0.1,
        )
    with pytest.raises(ProtocolError, match="exact H/J/K/L/d"):
        replace(scopes[0], heldout_case_center=scopes[0].calibration_center)
    mixed_h = replace(scopes[0], outer_target_center="OTHER-H")
    with pytest.raises(ProtocolError, match="mixed outer targets"):
        calibrate_clustered_uncertainty(rows, calibration_scopes=(mixed_h, *scopes[1:]))
    with pytest.raises(ProtocolError, match="four L scopes"):
        calibrate_clustered_uncertainty(
            tuple(row for row in rows if row.center_id != scopes[-1].calibration_center),
            calibration_scopes=scopes[:3],
        )


def _selection_calibration():
    scopes = _calibration_scopes()
    rows = []
    for scope in scopes:
        for action in ("A_UP", "B_DOWN"):
            for metric in ("bacc", "brier", "log"):
                predicted = 0.1 if metric == "bacc" else -0.03
                rows.append(OOFResidualObservation(scope.calibration_center, f"selection-{scope.calibration_center}-{action}-{metric}", scope.calibration_center, action, P_ACTION_ID, metric, predicted, predicted - 0.005 if metric == "bacc" else predicted + 0.005, scope.receipt_hash))
            for comparator in (P_ACTION_ID, "B_DOWN" if action == "A_UP" else "A_UP"):
                rows.append(OOFResidualObservation(scope.calibration_center, f"selection-{scope.calibration_center}-{action}-{comparator}", scope.calibration_center, action, comparator, "pairwise", 0.1, 0.095, scope.receipt_hash))
    return calibrate_clustered_uncertainty(rows, calibration_scopes=scopes)


def _selection_opportunity(
    active=("A_UP", "B_DOWN"), *, center_id="H", case_id="selection-case"
):
    active_ids = set(active)
    surfaces = (
        ActionSurface(
            "A_UP", "A", "up", (.8, .6) if "A_UP" in active_ids else (.3, .7)
        ),
        ActionSurface(
            "B_DOWN", "B", "down", (.4, .2) if "B_DOWN" in active_ids else (.3, .7)
        ),
    )
    opportunity = build_opportunity_set(
        (.4, .6), surfaces, candidate_action_ids=("A_UP", "B_DOWN")
    )
    return build_opportunity_case_receipt(
        center_id=center_id, case_id=case_id, opportunity=opportunity
    )


def _selection_utility(action: str, opportunity, *, bacc: float = 0.08):
    denominator = ExpectedDenominators(
        "selection-scope",
        1.0,
        1.0,
        2,
        "selection-eta",
        "selection-rows",
        "selection-posterior-model",
        "selection-posterior-scope",
    )
    member = opportunity.opportunity.member(action)
    return normalize_expected_utility(
        PrimitiveUtility(
            2.0 * bacc,
            0.0,
            -0.06,
            -0.04,
            2,
            action,
            opportunity.opportunity.baseline_hash,
            member.probability_hash,
            "selection-scope",
            "selection-rows",
            "selection-posterior-model",
            "selection-posterior-scope",
        ),
        denominator,
    )


def _selection_rows(*, calibration, model, opportunity, bacc: float = 0.08):
    queries = {
        action: ActionQuery(
            action,
            opportunity.opportunity.member(action).family,
            opportunity.opportunity.member(action).direction,
            model.feature_names,
            (0.4, 0.2),
        )
        for action in opportunity.active_representative_ids
    }
    p_query = ActionQuery.p_anchor(model.feature_names)
    return tuple(
        assemble_action_selection_evidence(
            query=query,
            equivalent_action_ids=opportunity.opportunity.equivalent_action_ids(action),
            utility=_selection_utility(action, opportunity, bacc=bacc),
            comparator_queries=(
                p_query,
                *(other for key, other in queries.items() if key != action),
            ),
            candidate_pool=_candidate_pool(),
            pairwise_model=model,
            uncertainty_calibration=calibration,
            opportunity_receipt=opportunity,
            ranking_policy=BaccRankingPolicy(),
        )
        for action, query in sorted(queries.items())
    )


def test_selection_is_exact_p_fail_closed_and_requires_runner_margin() -> None:
    pool = _candidate_pool()
    model = _fit_pairwise()
    calibration = _selection_calibration()
    empty = _selection_opportunity(())
    assert select_fail_closed_action(
        (),
        candidate_pool=pool,
        pairwise_model=model, uncertainty_calibration=calibration,
        opportunity_receipt=empty, ranking_policy=BaccRankingPolicy(),
    ).selected_action_id == P_ACTION_ID
    opportunity = _selection_opportunity()
    unsafe_model = replace(model, coefficients=tuple(0.0 for _ in model.coefficients))
    unsafe = select_fail_closed_action(
        _selection_rows(calibration=calibration, model=unsafe_model, opportunity=opportunity),
        candidate_pool=pool, pairwise_model=unsafe_model, uncertainty_calibration=calibration, opportunity_receipt=opportunity, ranking_policy=BaccRankingPolicy(),
    )
    assert unsafe.fallback_to_p
    safe = select_fail_closed_action(
        _selection_rows(calibration=calibration, model=model, opportunity=opportunity),
        candidate_pool=pool, pairwise_model=model, uncertainty_calibration=calibration, opportunity_receipt=opportunity, ranking_policy=BaccRankingPolicy(),
    )
    assert safe.selected_action_id == "A_UP"
    evidence = _selection_rows(calibration=calibration, model=model, opportunity=opportunity)
    with pytest.raises(ProtocolError, match="typed model|candidate-pool"):
        select_fail_closed_action(
            (replace(evidence[0], candidate_pool_receipt_hash="wrong"), evidence[1]),
            candidate_pool=pool, pairwise_model=model, uncertainty_calibration=calibration, opportunity_receipt=opportunity, ranking_policy=BaccRankingPolicy(),
        )
    with pytest.raises(ProtocolError, match="calibration arithmetic"):
        select_fail_closed_action(
            (replace(evidence[0], bacc=replace(evidence[0].bacc, bound=evidence[0].bacc.mean)), evidence[1]),
            candidate_pool=pool, pairwise_model=model, uncertainty_calibration=calibration, opportunity_receipt=opportunity, ranking_policy=BaccRankingPolicy(),
        )
    with pytest.raises(ProtocolError, match="typed model"):
        select_fail_closed_action(
            (replace(evidence[0], ranking_score=evidence[0].ranking_score + 10.0), evidence[1]),
            candidate_pool=pool, pairwise_model=model, uncertainty_calibration=calibration, opportunity_receipt=opportunity, ranking_policy=BaccRankingPolicy(),
        )
    with pytest.raises(ProtocolError, match="typed model"):
        select_fail_closed_action(
            (replace(evidence[0], utility=evidence[1].utility), evidence[1]),
            candidate_pool=pool, pairwise_model=model, uncertainty_calibration=calibration, opportunity_receipt=opportunity, ranking_policy=BaccRankingPolicy(),
        )


@lru_cache(maxsize=1)
def _selection_assets():
    return _fit_pairwise(), _selection_calibration(), _candidate_pool(), BaccRankingPolicy()


def _decision_bundle(selected: bool, *, active, center_id, case_id):
    model, calibration, pool, policy = _selection_assets()
    opportunity = _selection_opportunity(active, center_id=center_id, case_id=case_id)
    effective_model = model
    evidence = _selection_rows(
        calibration=calibration,
        model=effective_model,
        opportunity=opportunity,
        bacc=0.08 if selected else -0.08,
    ) if active else ()
    decision = select_fail_closed_action(
        evidence,
        candidate_pool=pool,
        pairwise_model=effective_model,
        uncertainty_calibration=calibration,
        opportunity_receipt=opportunity,
        ranking_policy=policy,
    )
    return decision, evidence, pool, effective_model, calibration, opportunity, policy


def _admission_cases(*, selected: bool = True) -> tuple[AdmissionCase, ...]:
    result = []
    for center in ("1", "2", "3", "4", "5", "6"):
        for case in range(4):
            case_id = f"case-{center}-{case}"
            bundle = _decision_bundle(
                selected,
                active=("A_UP", "B_DOWN"),
                center_id=center,
                case_id=case_id,
            )
            decision, evidence, pool, model, calibration, opportunity, policy = bundle
            by_action = {row.action_id: row for row in evidence}
            candidates = (
                AdmissionCandidate(
                    by_action["A_UP"], 0.18, -0.03, -0.02
                ),
                AdmissionCandidate(
                    by_action["B_DOWN"], -0.08, 0.01, 0.01
                ),
            )
            receipt = seal_admission_decision(
                center_id=center,
                case_id=case_id,
                decision=decision,
                candidate_evidence=evidence,
                candidate_pool=pool,
                pairwise_model=model,
                uncertainty_calibration=calibration,
                opportunity_receipt=opportunity,
                ranking_policy=policy,
            )
            result.append(AdmissionCase(center, case_id, candidates, receipt))
    return tuple(result)


def test_admission_is_nonvacuous_and_seals_failed_surface() -> None:
    zero_action_cases = []
    for center in ("1", "2", "3", "4", "5", "6"):
        case_id = f"zero-action-{center}"
        decision, evidence, pool, model, calibration, opportunity, policy = _decision_bundle(
            False, active=(), center_id=center, case_id=case_id
        )
        zero_action_cases.append(
            AdmissionCase(
                center,
                case_id,
                (),
                seal_admission_decision(
                    center_id=center,
                    case_id=case_id,
                    decision=decision,
                    candidate_evidence=evidence,
                    candidate_pool=pool,
                    pairwise_model=model,
                    uncertainty_calibration=calibration,
                    opportunity_receipt=opportunity,
                    ranking_policy=policy,
                ),
            )
        )
    passed = evaluate_source_only_admission((*_admission_cases(), *zero_action_cases))
    assert passed.passed
    assert passed.selected_count == 24
    assert passed.case_count == 30
    assert passed.unique_active_case_count == 24
    assert passed.safe_coverage == pytest.approx(24 / 30)
    assert passed.minimum_delete_center_tau_b > 0.0
    failed = evaluate_source_only_admission(_admission_cases(selected=False))
    assert not failed.passed
    assert "zero_safe_selections" in failed.reasons
    assert failed.admitted_center_ids == ()
    assert failed.sealed_to_p_center_ids == ("1", "2", "3", "4", "5", "6")
    case = _admission_cases()[0]
    assert not hasattr(case.candidates[0], "selected")
    with pytest.raises(ProtocolError, match="sealed candidate evidence"):
        replace(case, candidates=case.candidates[:1])
    with pytest.raises(ProtocolError, match="sealed candidate evidence"):
        replace(
            case,
            candidates=(
                replace(
                    case.candidates[0],
                    selection_evidence=replace(
                        case.candidates[0].selection_evidence,
                        ranking_score=case.candidates[0].predicted_score + 1.0,
                    ),
                ),
                *case.candidates[1:],
            ),
        )
    with pytest.raises(ProtocolError, match="sealed candidate evidence"):
        replace(
            case,
            candidates=(
                replace(
                    case.candidates[0],
                    selection_evidence=replace(
                        case.candidates[0].selection_evidence,
                        utility=replace(
                            case.candidates[0].selection_evidence.utility,
                            candidate_probability_hash="substituted-surface",
                        ),
                    ),
                ),
                *case.candidates[1:],
            ),
        )
    with pytest.raises(ProtocolError, match="sealed candidate evidence"):
        replace(
            case,
            candidates=(
                replace(
                    case.candidates[0],
                    selection_evidence=case.candidates[1].selection_evidence,
                ),
                *case.candidates[1:],
            ),
        )

    receipt = case.decision_receipt
    fabricated_fallback = replace(
        receipt.selection_decision,
        selected_action_id=P_ACTION_ID,
        fallback_to_p=True,
        reason="fabricated_fallback",
        selected_equivalent_action_ids=(P_ACTION_ID,),
    )
    with pytest.raises(ProtocolError, match="selection decision drifted"):
        replace(receipt, selection_decision=fabricated_fallback)


def test_admission_rates_are_case_then_equal_center_not_micro_pooled() -> None:
    cases = []
    for center in ("1", "2", "3", "4", "5", "6"):
        count = 100 if center == "1" else 4
        correct = center == "1"
        for ordinal in range(count):
            case_id = f"imbalanced-{center}-{ordinal}"
            decision, evidence, pool, model, calibration, opportunity, policy = _decision_bundle(
                correct, active=("A_UP",), center_id=center, case_id=case_id
            )
            candidate = AdmissionCandidate(
                evidence[0],
                0.1 if correct else -0.1,
                -0.01,
                -0.01,
            )
            receipt = seal_admission_decision(
                center_id=center,
                case_id=case_id,
                decision=decision,
                candidate_evidence=evidence,
                candidate_pool=pool,
                pairwise_model=model,
                uncertainty_calibration=calibration,
                opportunity_receipt=opportunity,
                ranking_policy=policy,
            )
            cases.append(AdmissionCase(center, case_id, (candidate,), receipt))
    report = evaluate_source_only_admission(cases)
    assert report.sign_accuracy == pytest.approx(1.0 / 6.0)
    assert report.pairwise_tau_b == pytest.approx(-4.0 / 6.0)
    assert report.safe_coverage == pytest.approx(1.0 / 6.0)
    assert not report.passed


def test_process_contracts_are_pickle_safe_and_prediction_records_have_no_labels() -> None:
    opportunity = build_opportunity_set(
        (0.4, 0.6),
        (ActionSurface("A", "F", "up", (0.7, 0.6)),),
        candidate_action_ids=("A",),
    )
    assert pickle.loads(pickle.dumps(opportunity)) == opportunity
    prediction = crossfit_source_row_posterior(
        _posterior_rows(), outer_target_center="H"
    )[0]
    assert pickle.loads(pickle.dumps(prediction)) == prediction
    assert not hasattr(prediction, "outcome")
    assert not hasattr(prediction, "label")
