from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import struct

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_action_model.contracts import (
    DIRECTIONS,
    LAMBDA_GRID,
    HarpActionScore,
    HarpSupportCell,
    HarpTargetAction,
)
from midogpp_thesis.cvae.routing.harp_action_model.fitting import (
    HarpActionModelBank,
    HarpLodoFoldAudit,
    HarpOutcomeModel,
    score_harp_actions,
)
from midogpp_thesis.cvae.routing.harp_action_model.ridge import HarpRidgeModel


FEATURES = (
    "margin_gap",
    "seed_dispersion",
    "entropy_gap",
    "source_distance",
    "baseline_margin",
    "expert_margin",
)
MODEL_FEATURES = (*FEATURES, "action_lambda")
CANDIDATES = ("A", "B", "C")
DONORS = ("d0", "d1", "d2")
OUTCOMES = ("gain", "brier", "log_loss")


def _ridge(*, donor: str | None, offset: float) -> HarpRidgeModel:
    dimension = 1 + len(MODEL_FEATURES) + len(CANDIDATES)
    diagonal = np.asarray(
        [0.7 + offset + 0.03 * index for index in range(dimension)],
        dtype=np.float64,
    )
    return HarpRidgeModel(
        feature_names=MODEL_FEATURES,
        candidate_levels=CANDIDATES,
        feature_mean=np.asarray(
            (0.17, 0.031, -0.12, 0.51, 0.08, -0.27, 0.61),
            dtype=np.float64,
        ),
        feature_scale=np.asarray(
            (0.43, 0.019, 0.37, 0.28, 0.11, 0.62, 0.29),
            dtype=np.float64,
        ),
        coefficients=np.asarray(
            [(-0.13 + offset) + 0.017 * index for index in range(dimension)],
            dtype=np.float64,
        ),
        normal_inverse=np.diag(diagonal),
        alpha=0.1 + offset,
        training_query_ids=("q0", "q1", "q2"),
        training_source_ids=CANDIDATES,
        training_case_ids=("case-0", "case-1"),
        excluded_donor_ids=() if donor is None else (donor,),
    )


def _outcome(outcome: str, direction: str, offset: float) -> HarpOutcomeModel:
    return HarpOutcomeModel(
        outcome=outcome,
        direction=direction,
        full_model=_ridge(donor=None, offset=offset),
        delete_donor_models=tuple(
            (donor, _ridge(donor=donor, offset=offset + 0.011 * (index + 1)))
            for index, donor in enumerate(DONORS)
        ),
        nested_lodo_audit=(
            HarpLodoFoldAudit("heldout", ("q0", "q1"), CANDIDATES, 0.1, 0.02),
        ),
    )


@pytest.fixture(scope="module")
def bank() -> HarpActionModelBank:
    models = tuple(
        _outcome(
            outcome,
            direction,
            0.07 * outcome_index + 0.19 * direction_index,
        )
        for direction_index, direction in enumerate(("ALL_MARGINS", "D01"))
        for outcome_index, outcome in enumerate(OUTCOMES)
    )
    support = tuple(
        HarpSupportCell(
            candidate,
            lambda_value,
            direction,
            4 + direction_index,
            12 + direction_index,
            (0, 1),
        )
        for candidate in CANDIDATES
        for lambda_value in LAMBDA_GRID
        for direction_index, direction in enumerate(DIRECTIONS)
    )
    return HarpActionModelBank(
        outer_target_id="H",
        feature_names=FEATURES,
        prediction_seal_hashes=("a" * 64,),
        response_receipt_hashes=("b" * 64,),
        models=models,
        support_cells=support,
    )


def _actions(repetitions: int = 2) -> tuple[HarpTargetAction, ...]:
    actions: list[HarpTargetAction] = []
    row = 0
    for repetition in range(repetitions):
        for direction in ("D10", "D01", "ALL_MARGINS"):
            for candidate_index, candidate in enumerate(CANDIDATES):
                for lambda_value in LAMBDA_GRID:
                    feature = (
                        -0.37 + 0.013 * row + 0.002 * candidate_index,
                        0.005 + 0.0007 * ((row + candidate_index) % 19),
                        0.21 - 0.003 * row,
                        0.44 + 0.0011 * (row % 23),
                        -0.08 + 0.0023 * (row % 17),
                        0.63 - 0.0017 * row,
                    )
                    actions.append(
                        HarpTargetAction(
                            outer_target_id="H",
                            target_query_id="H",
                            candidate_source_id=candidate,
                            case_id=f"case-{repetition}",
                            sample_id=f"sample-{row}",
                            lambda_value=lambda_value,
                            direction=direction,
                            feature_names=FEATURES,
                            feature_values=feature,
                            baseline_probability_bytes=struct.pack(
                                "<d", 0.2 + 0.001 * row
                            ),
                            expert_probability=0.3 + 0.002 * row,
                            ensemble_size=9,
                            ensemble_receipt_hash="c" * 64,
                            prediction_seal_hash="d" * 64,
                        )
                    )
                    row += 1
    # Interleave all direction/model groups so equivalence also protects the
    # caller-visible sequence rather than merely each grouped subsequence.
    return tuple(actions[index] for index in range(0, len(actions), 2)) + tuple(
        actions[index] for index in range(1, len(actions), 2)
    )


def _legacy_score_harp_actions(
    bank: HarpActionModelBank,
    actions: tuple[HarpTargetAction, ...],
) -> tuple[HarpActionScore, ...]:
    """Frozen reference for the pre-batching singleton implementation."""

    output: list[HarpActionScore] = []
    for action in actions:
        if (
            not isinstance(action, HarpTargetAction)
            or action.outer_target_id != bank.outer_target_id
            or action.feature_names != bank.feature_names
        ):
            raise ProtocolError("HARP target action drifted from its model bank.")
        direction = (
            action.direction
            if any(model.direction == action.direction for model in bank.models)
            else "ALL_MARGINS"
        )
        values: dict[
            str, tuple[tuple[float, ...], tuple[float, ...], tuple[str, ...]]
        ] = {}
        model_names = bank.model("gain", direction).full_model.feature_names
        values_row = (
            action.feature_values
            if "action_lambda" in action.feature_names
            else (*action.feature_values, action.lambda_value)
        )
        if len(values_row) != len(model_names):
            raise ProtocolError("HARP target action model feature geometry drifted.")
        matrix = np.asarray([values_row], dtype=np.float64)
        for outcome in OUTCOMES:
            model = bank.model(outcome, direction)
            predictions: list[float] = []
            leverages: list[float] = []
            donors: list[str] = []
            for donor, deleted in model.delete_donor_models:
                prediction, leverage = deleted.predict(
                    matrix, (action.candidate_source_id,)
                )
                predictions.append(float(prediction[0]))
                leverages.append(float(leverage[0]))
                donors.append(donor)
            values[outcome] = (
                tuple(predictions),
                tuple(leverages),
                tuple(donors),
            )
        donor_ids = values["gain"][2]
        if values["brier"][2] != donor_ids or values["log_loss"][2] != donor_ids:
            raise ProtocolError("HARP delete-donor outcome banks drifted.")
        output.append(
            HarpActionScore(
                action,
                values["gain"][0],
                values["brier"][0],
                values["log_loss"][0],
                tuple(
                    max(a, b, c)
                    for a, b, c in zip(
                        values["gain"][1],
                        values["brier"][1],
                        values["log_loss"][1],
                        strict=True,
                    )
                ),
                bank.support(action, direction),
                donor_ids,
            )
        )
    return tuple(output)


def _float64_bytes(values: tuple[float, ...]) -> bytes:
    return b"".join(struct.pack("<d", value) for value in values)


def test_batched_scoring_is_bit_exact_to_frozen_singleton_reference(
    bank: HarpActionModelBank,
) -> None:
    actions = _actions()
    expected = _legacy_score_harp_actions(bank, actions)
    actual = score_harp_actions(bank, actions)

    assert len(actual) == len(expected)
    for reference, batched, action in zip(expected, actual, actions, strict=True):
        assert batched.action is action
        assert batched.action == reference.action
        assert batched.support == reference.support
        assert batched.delete_donors == reference.delete_donors == DONORS
        for field in (
            "gain_predictions",
            "brier_predictions",
            "log_loss_predictions",
            "leverages",
        ):
            assert _float64_bytes(getattr(batched, field)) == _float64_bytes(
                getattr(reference, field)
            )


def test_predict_call_count_scales_with_model_groups_not_action_rows(
    bank: HarpActionModelBank,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = HarpRidgeModel.predict_singleton_equivalent_batch
    batch_sizes: list[int] = []

    def counted_predict(
        self: HarpRidgeModel,
        features: np.ndarray,
        candidates: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_sizes.append(len(candidates))
        return original(self, features, candidates)

    monkeypatch.setattr(
        HarpRidgeModel, "predict_singleton_equivalent_batch", counted_predict
    )

    action_menu = _actions(1)
    small = (
        next(action for action in action_menu if action.direction == "D01"),
        next(action for action in action_menu if action.direction == "ALL_MARGINS"),
    )
    score_harp_actions(bank, small)
    small_calls = len(batch_sizes)
    assert small_calls == len(OUTCOMES) * len(DONORS) * 2
    assert set(batch_sizes) == {1}

    batch_sizes.clear()
    large = _actions(5)
    score_harp_actions(bank, large)
    assert len(batch_sizes) == small_calls
    assert min(batch_sizes) > 1


def test_batched_scoring_retains_bank_and_action_hard_stops(
    bank: HarpActionModelBank,
) -> None:
    with pytest.raises(ProtocolError, match="typed frozen model bank"):
        score_harp_actions(object(), _actions(1))  # type: ignore[arg-type]
    with pytest.raises(ProtocolError, match="target action drifted"):
        score_harp_actions(bank, (object(),))  # type: ignore[arg-type]

    action = _actions(1)[0]
    foreign = replace(action, outer_target_id="X", target_query_id="X")
    with pytest.raises(ProtocolError, match="target action drifted"):
        score_harp_actions(bank, (foreign,))

    schema_drift = replace(
        action,
        feature_names=("seed_dispersion",),
        feature_values=(action.feature_values[1],),
    )
    with pytest.raises(ProtocolError, match="target action drifted"):
        score_harp_actions(bank, (schema_drift,))

    malformed = deepcopy(bank)
    gain_core = malformed.model("gain", "ALL_MARGINS").full_model
    object.__setattr__(gain_core, "feature_names", (*gain_core.feature_names, "drift"))
    with pytest.raises(ProtocolError, match="model feature geometry drifted"):
        score_harp_actions(malformed, (action,))
