from __future__ import annotations

from dataclasses import replace

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
import midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v9.model as model_v9
from midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v9 import (
    Direction,
    LabelFreeAction,
    PairwiseFitConfig,
    SourceActionOutcome,
    assemble_source_lodo_result,
    build_effective_menu,
    fit_prelabel_pseudo_target_fold,
    float32_probability_hex,
)


OUTER = "H"
CENTERS = ("A", "B", "C", "D", "E")
FEATURE_NAMES = ("budget", "allocation")
BASELINE = float32_probability_hex((0.2, 0.7, 0.3, 0.6))
D01 = float32_probability_hex((0.8, 0.7, 0.3, 0.6))
D10 = float32_probability_hex((0.2, 0.3, 0.3, 0.6))
HXE = float32_probability_hex((0.65, 0.7, 0.3, 0.6))
CONFIG = PairwiseFitConfig(
    pairwise_alpha=0.1,
    residual_alpha=0.1,
    acceptor_alpha=0.1,
)


def _action(
    *,
    query: str,
    case_id: str,
    action_id: str,
    direction: Direction,
    values: tuple[float, float],
    probability: tuple[str, ...],
    candidate: str | None = None,
) -> LabelFreeAction:
    return LabelFreeAction(
        outer_target_id=OUTER,
        query_center_id=query,
        case_id=case_id,
        action_id=action_id,
        action_kind="HXE" if candidate is not None else "U",
        direction=direction,
        candidate_source_id=candidate,
        feature_names=FEATURE_NAMES,
        feature_values=values,
        baseline_probability_hex=BASELINE,
        action_probability_hex=probability,
    )


def _surface() -> tuple[SourceActionOutcome, ...]:
    rows: list[SourceActionOutcome] = []
    for center_index, center in enumerate(CENTERS):
        candidates = tuple(value for value in CENTERS if value != center)
        for ordinal in range(2):
            case_id = f"{center}-{ordinal}"
            sign = 1.0 if ordinal == 0 else -1.0
            candidate = candidates[(center_index + ordinal) % len(candidates)]
            actions = (
                (
                    _action(
                        query=center,
                        case_id=case_id,
                        action_id="u-d01",
                        direction=Direction.D01,
                        values=(1.0 + ordinal, sign),
                        probability=D01,
                    ),
                    0.12 if sign > 0.0 else 0.03,
                ),
                (
                    _action(
                        query=center,
                        case_id=case_id,
                        action_id="u-d10",
                        direction=Direction.D10,
                        values=(1.0 + ordinal, -sign),
                        probability=D10,
                    ),
                    0.11 if sign < 0.0 else 0.02,
                ),
                (
                    _action(
                        query=center,
                        case_id=case_id,
                        action_id=f"hxe-{candidate}",
                        direction=Direction.D01,
                        values=(1.25 + ordinal, sign + 0.2),
                        probability=HXE,
                        candidate=candidate,
                    ),
                    0.07,
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


def _menus(
    surface: tuple[SourceActionOutcome, ...], center: str
) -> tuple[object, ...]:
    case_ids = sorted(
        {row.action.case_id for row in surface if row.action.query_center_id == center}
    )
    return tuple(
        build_effective_menu(
            tuple(
                row.action
                for row in surface
                if row.action.query_center_id == center
                and row.action.case_id == case_id
            )
        )
        for case_id in case_ids
    )


def _prelabel_fold(
    surface: tuple[SourceActionOutcome, ...], heldout: str
):
    allowed = tuple(
        row for row in surface if row.action.query_center_id != heldout
    )
    allowed_menus = tuple(
        menu
        for center in CENTERS
        if center != heldout
        for menu in _menus(surface, center)
    )
    return fit_prelabel_pseudo_target_fold(
        allowed,
        heldout_center_id=heldout,
        heldout_menus=_menus(surface, heldout),
        fixed_excluded_center_ids=(OUTER, heldout),
        effective_menus=allowed_menus,
        config=CONFIG,
    )


def test_prelabel_q_fold_is_invariant_to_q_endpoint_poison_and_rejects_q_outcomes() -> None:
    surface = _surface()
    heldout = CENTERS[0]
    poisoned = tuple(
        replace(
            row,
            bacc_gain=-100.0 - index,
            brier_delta=50.0 + index,
            log_delta=60.0 + index,
        )
        if row.action.query_center_id == heldout
        else row
        for index, row in enumerate(surface)
    )

    clean_fold = _prelabel_fold(surface, heldout)
    poisoned_fold = _prelabel_fold(poisoned, heldout)

    assert clean_fold.fold_hash == poisoned_fold.fold_hash
    assert tuple(
        row.prediction_hash for row in clean_fold.heldout_predictions
    ) == tuple(row.prediction_hash for row in poisoned_fold.heldout_predictions)
    assert all(
        {OUTER, heldout}.issubset(row.excluded_center_ids)
        and heldout not in row.training_center_ids
        and heldout not in row.training_candidate_ids
        for row in clean_fold.heldout_predictions
    )
    assert all(
        {OUTER, heldout, row.query_center_id}.issubset(row.excluded_center_ids)
        and heldout not in row.training_center_ids
        and heldout not in row.training_candidate_ids
        and row.query_center_id not in row.training_center_ids
        and row.query_center_id not in row.training_candidate_ids
        for row in clean_fold.predictions
    )

    with pytest.raises(ProtocolError, match="heldout-q outcome"):
        fit_prelabel_pseudo_target_fold(
            poisoned,
            heldout_center_id=heldout,
            heldout_menus=_menus(poisoned, heldout),
            fixed_excluded_center_ids=(OUTER, heldout),
            config=CONFIG,
        )


def test_final_assembly_uses_presealed_q_predictions_without_recomputing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = _surface()
    folds = tuple(_prelabel_fold(surface, center) for center in CENTERS)
    sealed_hashes = tuple(
        prediction.prediction_hash
        for fold in folds
        for prediction in fold.heldout_predictions
    )

    def forbidden_prediction(*_args, **_kwargs):
        raise AssertionError("assembly recomputed a presealed q prediction")

    monkeypatch.setattr(model_v9, "predict_case", forbidden_prediction)
    result = assemble_source_lodo_result(
        surface,
        presealed_folds=folds,
        config_grid=(CONFIG,),
    )

    assert tuple(row.prediction_hash for row in result.oof_predictions) == sealed_hashes
    assert result.heldout_model_hashes == tuple(
        (fold.heldout_center_id, fold.heldout_model_hash) for fold in folds
    )
    assert result.final_model.excluded_center_ids == (OUTER,)

