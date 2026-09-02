from __future__ import annotations

from dataclasses import replace

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
import midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v9.calibration as calibration_v9
import midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v9.model as model_v9
from midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v9 import (
    AdmissionConfig,
    Direction,
    LabelFreeAction,
    OuterAdmission,
    PairwiseFitConfig,
    PairwiseRanker,
    PairwiseResidualRouterModel,
    PolicyCalibration,
    PolicyReplay,
    PolicyRiskConfig,
    ScaleOnlyTransform,
    SelectedActionAcceptor,
    SourceActionOutcome,
    assert_residual_identity,
    build_effective_menu,
    calibrate_selected_policy,
    fit_source_lodo,
    float32_probability_hex,
    predict_case,
    probability_hex_to_bytes,
    residual_feature_names,
    residualize_menu,
    select_policy_action,
)
from midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v9.acceptor import (
    LinearHead,
    SELECTION_FEATURE_NAMES,
    Standardizer,
)


OUTER = "H"
SOURCES = ("A", "B", "C", "D", "E")
FEATURE_NAMES = ("budget_signal", "allocation_signal")
BASELINE = float32_probability_hex((0.2, 0.7, 0.3, 0.6))
D01_PROBABILITY = float32_probability_hex((0.8, 0.7, 0.3, 0.6))
D10_PROBABILITY = float32_probability_hex((0.2, 0.3, 0.3, 0.6))
HXE_PROBABILITY = float32_probability_hex((0.65, 0.7, 0.3, 0.6))


def _action(
    *,
    query: str,
    case: str,
    action_id: str,
    direction: Direction,
    features: tuple[float, float],
    probability: tuple[str, ...],
    kind: str = "U",
    candidate: str | None = None,
) -> LabelFreeAction:
    return LabelFreeAction(
        outer_target_id=OUTER,
        query_center_id=query,
        case_id=case,
        action_id=action_id,
        action_kind=kind,
        direction=direction,
        candidate_source_id=candidate,
        feature_names=FEATURE_NAMES,
        feature_values=features,
        baseline_probability_hex=BASELINE,
        action_probability_hex=probability,
    )


def _surface() -> tuple[SourceActionOutcome, ...]:
    rows: list[SourceActionOutcome] = []
    for center_index, center in enumerate(SOURCES):
        candidates = tuple(value for value in SOURCES if value != center)
        for ordinal in range(3):
            case = f"case-{ordinal}"
            preference = 1.0 if ordinal % 2 == 0 else -1.0
            activity = 1.0 + 0.1 * ordinal
            d01_gain = 0.13 if preference > 0.0 else 0.04
            d10_gain = 0.12 if preference < 0.0 else 0.03
            candidate = candidates[(center_index + ordinal) % len(candidates)]
            actions = (
                (
                    _action(
                        query=center,
                        case=case,
                        action_id="u-d01",
                        direction=Direction.D01,
                        features=(activity, preference),
                        probability=D01_PROBABILITY,
                    ),
                    d01_gain,
                ),
                (
                    _action(
                        query=center,
                        case=case,
                        action_id="u-d10",
                        direction=Direction.D10,
                        features=(activity, -preference),
                        probability=D10_PROBABILITY,
                    ),
                    d10_gain,
                ),
                (
                    _action(
                        query=center,
                        case=case,
                        action_id=f"hxe-{candidate}",
                        direction=Direction.D01,
                        features=(activity + 0.25, preference + 0.2),
                        probability=HXE_PROBABILITY,
                        kind="HXE",
                        candidate=candidate,
                    ),
                    0.08 + 0.005 * ordinal,
                ),
            )
            rows.extend(
                SourceActionOutcome(
                    action=action,
                    bacc_gain=gain,
                    brier_delta=-gain / 2.0,
                    log_delta=-gain / 3.0,
                )
                for action, gain in actions
            )
    return tuple(rows)


def _linear_head(intercept: float) -> LinearHead:
    return LinearHead(
        intercept=intercept,
        coefficients=tuple(0.0 for _ in SELECTION_FEATURE_NAMES),
    )


def _manual_model(
    *,
    physical_tie_score: float,
    acceptance_logit: float = 30.0,
    harm_logit: float = 0.0,
    predicted_brier_delta: float = 0.0,
    predicted_log_delta: float = 0.0,
) -> PairwiseResidualRouterModel:
    names = residual_feature_names(FEATURE_NAMES)
    coefficients = [0.0] * len(names)
    coefficients[names.index("kind__U")] = physical_tie_score
    transform = ScaleOnlyTransform(
        names=names,
        scale=tuple(1.0 for _ in names),
    )
    ranker = PairwiseRanker(
        outer_target_id=OUTER,
        training_center_ids=SOURCES,
        training_candidate_ids=(),
        excluded_center_ids=(OUTER,),
        transform=transform,
        coefficients=tuple(coefficients),
        budget_width=len(FEATURE_NAMES),
        allocation_width=len(FEATURE_NAMES),
        pairwise_alpha=1.0,
        residual_alpha=1.0,
        pairwise_tie_tolerance=1e-12,
        training_pair_count=1,
    )
    acceptor = SelectedActionAcceptor(
        outer_target_id=OUTER,
        training_center_ids=SOURCES,
        excluded_center_ids=(OUTER,),
        standardizer=Standardizer(
            names=SELECTION_FEATURE_NAMES,
            mean=tuple(0.0 for _ in SELECTION_FEATURE_NAMES),
            scale=tuple(1.0 for _ in SELECTION_FEATURE_NAMES),
        ),
        beneficial_head=_linear_head(acceptance_logit),
        harm_head=_linear_head(harm_logit),
        gain_head=_linear_head(physical_tie_score),
        brier_head=_linear_head(predicted_brier_delta),
        log_head=_linear_head(predicted_log_delta),
        ridge_alpha=1.0,
        max_brier_delta=0.002,
        max_log_delta=0.005,
        training_record_hashes=("a" * 64,),
    )
    return PairwiseResidualRouterModel(
        outer_target_id=OUTER,
        training_center_ids=SOURCES,
        training_candidate_ids=(),
        excluded_center_ids=(OUTER,),
        ranker=ranker,
        acceptor=acceptor,
        fit_config=PairwiseFitConfig(),
    )


def _target_menu(case: str = "target"):
    return build_effective_menu(
        (
            _action(
                query=OUTER,
                case=case,
                action_id="z-action",
                direction=Direction.D01,
                features=(2.0, 1.0),
                probability=D01_PROBABILITY,
            ),
            _action(
                query=OUTER,
                case=case,
                action_id="a-action",
                direction=Direction.D10,
                features=(3.0, -1.0),
                probability=D10_PROBABILITY,
            ),
        )
    )


def _admission(*, admitted: bool) -> OuterAdmission:
    reasons = () if admitted else ("TEST_OUTER_ADMISSION_FAILURE",)
    return OuterAdmission(
        outer_target_id=OUTER,
        admitted=admitted,
        learned_top1_accuracy=1.0 if admitted else 0.0,
        always_b_top1_accuracy=0.0,
        pooled_top1_excess=1.0 if admitted else 0.0,
        min_delete_center_top1_excess=1.0 if admitted else 0.0,
        opportunity_top1_accuracy=1.0 if admitted else 0.0,
        opportunity_case_count=10,
        case_count=10,
        reasons=reasons,
        config=AdmissionConfig(),
    )


def _calibration() -> PolicyCalibration:
    replay = PolicyReplay(
        acceptance_threshold=0.0,
        rank_margin_threshold=0.0,
        routed_cases=1,
        case_count=1,
        coverage=1.0,
        case_equal_bacc_gain=0.1,
        min_delete_center_bacc_gain=0.1,
        case_equal_brier_delta=0.0,
        case_equal_log_delta=0.0,
        routed_harm_rate=0.0,
        safe=True,
    )
    return PolicyCalibration(
        outer_target_id=OUTER,
        calibrated=True,
        acceptance_threshold=0.0,
        rank_margin_threshold=0.0,
        selected_replay=replay,
        nested_replay=replay,
        heldout_thresholds=((SOURCES[0], 0.0, 0.0),),
        frontier=(replay,),
        config=PolicyRiskConfig(acceptance_thresholds=(0.0,)),
    )


def test_residual_hierarchy_preserves_budget_plus_allocation_identity() -> None:
    uniform = _action(
        query="A",
        case="case",
        action_id="uniform",
        direction=Direction.D01,
        features=(2.0, 3.0),
        probability=D01_PROBABILITY,
    )
    hxe = _action(
        query="A",
        case="case",
        action_id="hxe-B",
        direction=Direction.D01,
        features=(5.0, 7.0),
        probability=HXE_PROBABILITY,
        kind="HXE",
        candidate="B",
    )
    rows = {row.action.action_id: row for row in residualize_menu(build_effective_menu((hxe, uniform)))}

    assert rows["uniform"].values[:2] == (2.0, 3.0)
    assert rows["uniform"].values[2:4] == (0.0, 0.0)
    assert rows["hxe-B"].values[:2] == (2.0, 3.0)
    assert rows["hxe-B"].values[2:4] == (3.0, 4.0)
    assert rows["hxe-B"].has_uniform_reference
    assert_residual_identity(
        baseline=(0.0, 0.0),
        uniform=(2.0, 3.0),
        hxe=(5.0, 7.0),
    )
    with pytest.raises(ProtocolError, match="residual hierarchy identity"):
        assert_residual_identity(
            baseline=(0.0,),
            uniform=(1.0, 2.0),
            hxe=(3.0, 4.0),
        )


def test_virtual_b_wins_nonpositive_scores_and_action_id_breaks_physical_ties() -> None:
    menu = _target_menu()
    virtual_b = predict_case(_manual_model(physical_tie_score=0.0), menu)
    assert virtual_b.raw_top_action_id == "B"
    assert virtual_b.top_action_id == "B"
    assert virtual_b.acceptance_probability == 0.0
    assert all(score.pairwise_score == 0.0 for score in virtual_b.action_scores)

    tied_actions = predict_case(_manual_model(physical_tie_score=1.0), menu)
    assert tied_actions.raw_top_action_id == "a-action"
    assert tied_actions.ranked_action_ids == ("a-action", "z-action")
    assert tied_actions.rank_margin == 0.0


def test_failed_outer_admission_falls_back_to_exact_baseline_bytes() -> None:
    menu = _target_menu("fallback")
    prediction = predict_case(_manual_model(physical_tie_score=1.0), menu)
    decision = select_policy_action(
        menu,
        prediction,
        _admission(admitted=False),
        _calibration(),
    )

    assert decision.selected_action_id == "B"
    assert decision.exact_b_fallback
    assert decision.reason == "EXACT_B_OUTER_RANK_ADMISSION_FAILED"
    assert decision.probability_hex == BASELINE
    assert probability_hex_to_bytes(decision.probability_hex) == probability_hex_to_bytes(
        BASELINE
    )


def test_policy_ranks_first_without_a_per_action_certificate_gate() -> None:
    model = _manual_model(
        physical_tie_score=1.0,
        acceptance_logit=30.0,
        harm_logit=30.0,
        predicted_brier_delta=10.0,
        predicted_log_delta=10.0,
    )
    menu = _target_menu("rank-first")
    prediction = predict_case(model, menu)
    top = prediction.score_for(prediction.raw_top_action_id)
    assert top is not None
    assert top.predicted_harm_probability > 0.99
    assert top.predicted_brier_delta == pytest.approx(10.0)
    assert top.predicted_log_delta == pytest.approx(10.0)
    assert model.public_payload()["rank_all_before_acceptance"] is True
    assert model.public_payload()["per_action_certificate_gate"] is False

    decision = select_policy_action(
        menu,
        prediction,
        _admission(admitted=True),
        _calibration(),
    )
    assert decision.selected_action_id == "a-action"
    assert not decision.exact_b_fallback
    assert decision.reason == "ROUTED_POLICY_CALIBRATED_EXACT_TOP1"


@pytest.mark.parametrize("candidate", (OUTER, "A"))
def test_hxe_action_contract_rejects_outer_or_query_candidate(candidate: str) -> None:
    with pytest.raises(ProtocolError, match="H/q exclusion"):
        _action(
            query="A",
            case="leak",
            action_id="hxe",
            direction=Direction.D01,
            features=(1.0, 1.0),
            probability=HXE_PROBABILITY,
            kind="HXE",
            candidate=candidate,
        )


def test_nested_lodo_excludes_h_q_and_candidate_and_crossfits_acceptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[tuple[str, ...], tuple[object, ...]]] = []
    original = model_v9.fit_selected_action_acceptor

    def capture_acceptor(observations, **kwargs):
        captured.append(
            (
                tuple(kwargs["excluded_center_ids"]),
                tuple(observations),
            )
        )
        return original(observations, **kwargs)

    monkeypatch.setattr(model_v9, "fit_selected_action_acceptor", capture_acceptor)
    surface = _surface()
    result = fit_source_lodo(
        surface,
        config_grid=(
            PairwiseFitConfig(
                pairwise_alpha=0.1,
                residual_alpha=0.1,
                acceptor_alpha=0.1,
            ),
        ),
    )

    assert OUTER in result.final_model.excluded_center_ids
    assert OUTER not in result.final_model.training_center_ids
    assert OUTER not in result.final_model.training_candidate_ids
    assert all(
        row.query_center_id in row.excluded_center_ids
        and row.query_center_id not in row.training_center_ids
        and row.query_center_id not in row.training_candidate_ids
        for row in result.oof_predictions
    )
    assert all(
        fold.heldout_center_id in row.excluded_center_ids
        and row.query_center_id in row.excluded_center_ids
        and fold.heldout_center_id not in row.training_candidate_ids
        and row.query_center_id not in row.training_candidate_ids
        for fold in result.nested_policy_folds
        for row in (*fold.predictions, *fold.heldout_predictions)
    )

    final_records = next(rows for excluded, rows in captured if excluded == (OUTER,))
    assert final_records
    assert all(
        row.outer_target_id in row.selection_excluded_center_ids
        and row.query_center_id in row.selection_excluded_center_ids
        for row in final_records
    )
    assert result.final_model.acceptor.training_record_hashes == tuple(
        sorted(row.record_hash for row in final_records)
    )


def test_nested_policy_threshold_training_never_sees_heldout_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = _surface()
    result = fit_source_lodo(
        surface,
        config_grid=(PairwiseFitConfig(pairwise_alpha=0.1, residual_alpha=0.1),),
    )
    frontier_query_sets: list[frozenset[str]] = []
    original_frontier = calibration_v9._frontier

    def capture_frontier(predictions, outcome_map, config):
        frontier_query_sets.append(
            frozenset(row.query_center_id for row in predictions)
        )
        return original_frontier(predictions, outcome_map, config)

    monkeypatch.setattr(calibration_v9, "_frontier", capture_frontier)
    calibration = calibrate_selected_policy(
        result.oof_predictions,
        surface,
        nested_policy_folds=result.nested_policy_folds,
        config=PolicyRiskConfig(
            acceptance_thresholds=(0.0, 0.5),
            min_case_equal_bacc_gain=-1.0,
            min_delete_center_bacc_gain=-1.0,
            max_routed_harm_rate=1.0,
            max_case_equal_brier_delta=1.0,
            max_case_equal_log_delta=1.0,
            min_coverage=0.0,
            min_routed_cases=1,
        ),
    )

    assert frontier_query_sets[0] == frozenset(SOURCES)
    assert len(frontier_query_sets) == len(SOURCES) + 1
    for (heldout, _threshold, _margin), query_set in zip(
        calibration.heldout_thresholds,
        frontier_query_sets[1:],
        strict=True,
    ):
        assert query_set == frozenset(set(SOURCES) - {heldout})

    fold = result.nested_policy_folds[0]
    leaked = replace(
        fold.predictions[0],
        excluded_center_ids=(OUTER, fold.predictions[0].query_center_id),
    )
    with pytest.raises(ProtocolError, match="nested policy fold leaked"):
        replace(fold, predictions=(leaked, *fold.predictions[1:]))
