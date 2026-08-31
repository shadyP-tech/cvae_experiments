"""Strict outer-H and nested center-LODO fitting for HARP."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import DIRECTIONS, HarpActionScore, HarpSupportCell, HarpTargetAction, HarpTrainingObservation
from .ridge import HarpRidgeModel, fit_partial_pool_ridge


DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0)


@dataclass(frozen=True)
class HarpLodoFoldAudit:
    heldout_donor_id: str
    training_query_ids: tuple[str, ...]
    training_source_ids: tuple[str, ...]
    selected_alpha: float
    validation_mse: float

    def __post_init__(self) -> None:
        if (
            type(self.heldout_donor_id) is not str or not self.heldout_donor_id
            or self.training_query_ids != tuple(sorted(set(self.training_query_ids)))
            or self.training_source_ids != tuple(sorted(set(self.training_source_ids)))
            or self.heldout_donor_id in self.training_query_ids
            or self.heldout_donor_id in self.training_source_ids
            or not math.isfinite(float(self.selected_alpha)) or self.selected_alpha <= 0
            or not math.isfinite(float(self.validation_mse)) or self.validation_mse < 0
        ):
            raise ProtocolError("HARP nested-LODO audit is malformed.")


@dataclass(frozen=True)
class HarpOutcomeModel:
    outcome: str
    direction: str
    full_model: HarpRidgeModel
    delete_donor_models: tuple[tuple[str, HarpRidgeModel], ...]
    nested_lodo_audit: tuple[HarpLodoFoldAudit, ...]

    def __post_init__(self) -> None:
        donors = tuple(donor for donor, _model in self.delete_donor_models)
        if (
            self.outcome not in ("gain", "brier", "log_loss")
            or self.direction not in DIRECTIONS
            or not isinstance(self.full_model, HarpRidgeModel)
            or self.full_model.excluded_donor_ids
            or not donors
            or donors != tuple(sorted(set(donors)))
            or any(not isinstance(model, HarpRidgeModel) or model.excluded_donor_ids != (donor,) for donor, model in self.delete_donor_models)
            or not self.nested_lodo_audit
            or any(not isinstance(row, HarpLodoFoldAudit) for row in self.nested_lodo_audit)
        ):
            raise ProtocolError("HARP outcome model state is malformed.")


@dataclass(frozen=True)
class HarpActionModelBank:
    outer_target_id: str
    feature_names: tuple[str, ...]
    prediction_seal_hashes: tuple[str, ...]
    response_receipt_hashes: tuple[str, ...]
    models: tuple[HarpOutcomeModel, ...]
    support_cells: tuple[HarpSupportCell, ...]

    def __post_init__(self) -> None:
        if (
            type(self.outer_target_id) is not str or not self.outer_target_id
            or not self.feature_names
            or len(set(self.feature_names)) != len(self.feature_names)
            or not self.prediction_seal_hashes
            or not self.response_receipt_hashes
            or any(len(value) != 64 for value in (*self.prediction_seal_hashes, *self.response_receipt_hashes))
            or not self.models
            or any(not isinstance(model, HarpOutcomeModel) or model.full_model.feature_names != _expected_model_names(self.feature_names) for model in self.models)
            or len({(model.outcome, model.direction) for model in self.models}) != len(self.models)
            or not self.support_cells
            or any(not isinstance(cell, HarpSupportCell) for cell in self.support_cells)
        ):
            raise ProtocolError("HARP action model bank is malformed.")

    def model(self, outcome: str, direction: str) -> HarpOutcomeModel:
        lookup = {(row.outcome, row.direction): row for row in self.models}
        if (outcome, direction) in lookup:
            return lookup[(outcome, direction)]
        try:
            return lookup[(outcome, "ALL_MARGINS")]
        except KeyError as exc:
            raise ProtocolError("HARP model bank lacks its ALL_MARGINS core.") from exc

    def support(self, action: HarpTargetAction, direction: str) -> HarpSupportCell:
        lookup = {(row.candidate_source_id, row.lambda_value, row.direction): row for row in self.support_cells}
        return lookup.get(
            (action.candidate_source_id, action.lambda_value, direction),
            lookup.get((action.candidate_source_id, action.lambda_value, "ALL_MARGINS"), HarpSupportCell(action.candidate_source_id, action.lambda_value, "ALL_MARGINS", 0, 0, ())),
        )


def _expected_model_names(feature_names: tuple[str, ...]) -> tuple[str, ...]:
    return feature_names if "action_lambda" in feature_names else (*feature_names, "action_lambda")


def _matrix(rows: Sequence[HarpTrainingObservation]) -> np.ndarray:
    return np.asarray([
        row.feature_values if "action_lambda" in row.feature_names else (*row.feature_values, row.lambda_value)
        for row in rows
    ], dtype=np.float64)


def _model_feature_names(row: HarpTrainingObservation) -> tuple[str, ...]:
    return row.feature_names if "action_lambda" in row.feature_names else (*row.feature_names, "action_lambda")


def _response(rows: Sequence[HarpTrainingObservation], outcome: str) -> np.ndarray:
    name = {"gain": "weighted_correctness_surrogate", "brier": "brier_delta", "log_loss": "log_loss_delta"}[outcome]
    return np.asarray([getattr(row, name) for row in rows], dtype=np.float64)


def _select_alpha(rows: Sequence[HarpTrainingObservation], outcome: str, alphas: tuple[float, ...]) -> tuple[float, tuple[HarpLodoFoldAudit, ...]]:
    donors = tuple(sorted({row.pseudo_query_id for row in rows}))
    if len(donors) < 3:
        raise ProtocolError("Nested HARP LODO requires at least three donor queries.")
    losses: dict[float, list[float]] = {alpha: [] for alpha in alphas}
    per_alpha_audits: dict[float, list[HarpLodoFoldAudit]] = {alpha: [] for alpha in alphas}
    for heldout in donors:
        train = tuple(row for row in rows if heldout not in (row.pseudo_query_id, row.candidate_source_id))
        test = tuple(row for row in rows if row.pseudo_query_id == heldout and row.candidate_source_id != heldout)
        if not train or not test:
            raise ProtocolError("Strict HARP LODO produced an empty train or validation fold.")
        for alpha in alphas:
            model = fit_partial_pool_ridge(
                _matrix(train), _response(train, outcome),
                [row.pseudo_query_id for row in train], [row.case_id for row in train], [row.candidate_source_id for row in train],
                feature_names=_model_feature_names(train[0]), alpha=alpha,
                excluded_donor_ids=(heldout,),
            )
            predicted, _ = model.predict(_matrix(test), [row.candidate_source_id for row in test])
            squared = (predicted - _response(test, outcome)) ** 2
            mse = float(np.mean([
                float(np.mean(squared[[index for index, row in enumerate(test) if row.case_id == case]]))
                for case in sorted({row.case_id for row in test})
            ]))
            losses[alpha].append(mse)
            per_alpha_audits[alpha].append(HarpLodoFoldAudit(heldout, model.training_query_ids, model.training_source_ids, alpha, mse))
    selected = min(alphas, key=lambda alpha: (float(np.mean(losses[alpha])), alpha))
    return selected, tuple(per_alpha_audits[selected])


def _fit_outcome(rows: tuple[HarpTrainingObservation, ...], *, outcome: str, direction: str, alphas: tuple[float, ...]) -> HarpOutcomeModel:
    selected, audit = _select_alpha(rows, outcome, alphas)
    names = _model_feature_names(rows[0])
    full = fit_partial_pool_ridge(_matrix(rows), _response(rows, outcome), [row.pseudo_query_id for row in rows], [row.case_id for row in rows], [row.candidate_source_id for row in rows], feature_names=names, alpha=selected)
    delete_models: list[tuple[str, HarpRidgeModel]] = []
    for donor in sorted({row.pseudo_query_id for row in rows} | {row.candidate_source_id for row in rows}):
        retained = tuple(row for row in rows if donor not in (row.pseudo_query_id, row.candidate_source_id))
        if len({row.pseudo_query_id for row in retained}) < 3:
            raise ProtocolError("Delete-donor HARP fitting left too few inner LODO centers.")
        donor_alpha, _ = _select_alpha(retained, outcome, alphas)
        model = fit_partial_pool_ridge(_matrix(retained), _response(retained, outcome), [row.pseudo_query_id for row in retained], [row.case_id for row in retained], [row.candidate_source_id for row in retained], feature_names=names, alpha=donor_alpha, excluded_donor_ids=(donor,))
        delete_models.append((donor, model))
    return HarpOutcomeModel(outcome, direction, full, tuple(delete_models), audit)


def _support_cells(rows: tuple[HarpTrainingObservation, ...]) -> tuple[HarpSupportCell, ...]:
    cells: list[HarpSupportCell] = []
    candidates = sorted({row.candidate_source_id for row in rows})
    lambdas = sorted({row.lambda_value for row in rows})
    for candidate in candidates:
        for lambda_value in lambdas:
            for direction in DIRECTIONS:
                selected = tuple(row for row in rows if row.candidate_source_id == candidate and row.lambda_value == lambda_value and (direction == "ALL_MARGINS" or row.direction == direction))
                cells.append(HarpSupportCell(candidate, lambda_value, direction, len({row.pseudo_query_id for row in selected}), len({(row.pseudo_query_id, row.case_id) for row in selected}), tuple(sorted({row.truth_class for row in selected}))))
    return tuple(cells)


def _deduplicate_identical_samples(rows: tuple[HarpTrainingObservation, ...]) -> tuple[HarpTrainingObservation, ...]:
    """Remove only exact sample clones inside one q/e/case/action cell."""

    seen: set[tuple[object, ...]] = set()
    output: list[HarpTrainingObservation] = []
    for row in rows:
        signature = (
            row.outer_target_id, row.pseudo_query_id, row.candidate_source_id,
            row.case_id, row.lambda_value, row.direction, row.feature_names,
            row.feature_values, row.weighted_correctness_surrogate, row.brier_delta,
            row.log_loss_delta, row.truth_class, row.prediction_seal_hash,
            row.response_receipt_hash, row.case_aggregation_receipt_hash,
        )
        if signature not in seen:
            seen.add(signature)
            output.append(row)
    return tuple(output)


def fit_harp_action_model_bank(observations: Sequence[HarpTrainingObservation], *, outer_target_id: str, alphas: Sequence[float] = DEFAULT_ALPHAS) -> HarpActionModelBank:
    rows = tuple(observations)
    if not rows or any(not isinstance(row, HarpTrainingObservation) for row in rows):
        raise ProtocolError("HARP fitting requires typed source-inner observations.")
    if any(row.outer_target_id != outer_target_id for row in rows):
        raise ProtocolError("HARP rows drifted across outer targets.")
    names = rows[0].feature_names
    if any(row.feature_names != names for row in rows):
        raise ProtocolError("HARP feature schema drifted across source-inner rows.")
    if len({row.row_key for row in rows}) != len(rows):
        raise ProtocolError("HARP sample-level training observations contain duplicate identities.")
    rows = _deduplicate_identical_samples(rows)
    candidates = tuple(sorted(set(float(value) for value in alphas)))
    if not candidates or any(value <= 0 or not np.isfinite(value) for value in candidates):
        raise ProtocolError("HARP alpha grid must be finite and positive.")
    models: list[HarpOutcomeModel] = []
    for direction in DIRECTIONS:
        selected = rows if direction == "ALL_MARGINS" else tuple(row for row in rows if row.direction == direction)
        if len({row.pseudo_query_id for row in selected}) < 4:
            continue
        for outcome in ("gain", "brier", "log_loss"):
            models.append(_fit_outcome(tuple(selected), outcome=outcome, direction=direction, alphas=candidates))
    if not all(any(model.outcome == outcome and model.direction == "ALL_MARGINS" for model in models) for outcome in ("gain", "brier", "log_loss")):
        raise ProtocolError("HARP must fit all three ALL_MARGINS outcomes.")
    return HarpActionModelBank(str(outer_target_id), names, tuple(sorted({row.prediction_seal_hash for row in rows})), tuple(sorted({row.response_receipt_hash for row in rows})), tuple(models), _support_cells(rows))


def score_harp_actions(bank: HarpActionModelBank, actions: Sequence[HarpTargetAction]) -> tuple[HarpActionScore, ...]:
    if not isinstance(bank, HarpActionModelBank):
        raise ProtocolError("HARP scoring requires a typed frozen model bank.")

    # Build immutable-bank lookups once.  The former singleton scoring loop
    # rebuilt both of these dictionaries for every target action through
    # ``bank.model`` and ``bank.support``.
    model_lookup = {(row.outcome, row.direction): row for row in bank.models}
    available_directions = {row.direction for row in bank.models}
    support_lookup = {
        (row.candidate_source_id, row.lambda_value, row.direction): row
        for row in bank.support_cells
    }

    def outcome_model(outcome: str, direction: str) -> HarpOutcomeModel:
        model = model_lookup.get((outcome, direction))
        if model is not None:
            return model
        try:
            return model_lookup[(outcome, "ALL_MARGINS")]
        except KeyError as exc:
            raise ProtocolError("HARP model bank lacks its ALL_MARGINS core.") from exc

    # Preserve the caller's exact action order while grouping only the numeric
    # prediction work.  Each delete-donor ridge model can then score the whole
    # direction group in one call instead of receiving one singleton matrix per
    # action.
    grouped: dict[str, list[tuple[int, HarpTargetAction, tuple[float, ...]]]] = {}
    action_rows = tuple(actions)
    for index, action in enumerate(action_rows):
        if not isinstance(action, HarpTargetAction) or action.outer_target_id != bank.outer_target_id or action.feature_names != bank.feature_names:
            raise ProtocolError("HARP target action drifted from its model bank.")
        direction = action.direction if action.direction in available_directions else "ALL_MARGINS"
        model_names = outcome_model("gain", direction).full_model.feature_names
        values_row = action.feature_values if "action_lambda" in action.feature_names else (*action.feature_values, action.lambda_value)
        if len(values_row) != len(model_names):
            raise ProtocolError("HARP target action model feature geometry drifted.")
        grouped.setdefault(direction, []).append((index, action, values_row))

    ordered: list[HarpActionScore | None] = [None] * len(action_rows)
    for direction, group in grouped.items():
        matrix = np.asarray([values_row for _index, _action, values_row in group], dtype=np.float64)
        candidates = tuple(action.candidate_source_id for _index, action, _values_row in group)
        values: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = {}
        for outcome in ("gain", "brier", "log_loss"):
            model = outcome_model(outcome, direction)
            donor_ids = tuple(donor for donor, _deleted in model.delete_donor_models)
            predictions = np.empty((len(group), len(donor_ids)), dtype=np.float64)
            leverages = np.empty_like(predictions)
            for donor_index, (_donor, deleted) in enumerate(model.delete_donor_models):
                prediction, leverage = deleted.predict_singleton_equivalent_batch(
                    matrix, candidates
                )
                predictions[:, donor_index] = prediction
                leverages[:, donor_index] = leverage
            values[outcome] = (predictions, leverages, donor_ids)

        donor_ids = values["gain"][2]
        if values["brier"][2] != donor_ids or values["log_loss"][2] != donor_ids:
            raise ProtocolError("HARP delete-donor outcome banks drifted.")

        for group_index, (output_index, action, _values_row) in enumerate(group):
            support = support_lookup.get(
                (action.candidate_source_id, action.lambda_value, direction)
            )
            if support is None:
                support = support_lookup.get(
                    (action.candidate_source_id, action.lambda_value, "ALL_MARGINS")
                )
            if support is None:
                support = HarpSupportCell(
                    action.candidate_source_id,
                    action.lambda_value,
                    "ALL_MARGINS",
                    0,
                    0,
                    (),
                )
            ordered[output_index] = HarpActionScore(
                action,
                tuple(float(value) for value in values["gain"][0][group_index]),
                tuple(float(value) for value in values["brier"][0][group_index]),
                tuple(float(value) for value in values["log_loss"][0][group_index]),
                tuple(
                    max(float(a), float(b), float(c))
                    for a, b, c in zip(
                        values["gain"][1][group_index],
                        values["brier"][1][group_index],
                        values["log_loss"][1][group_index],
                        strict=True,
                    )
                ),
                support,
                donor_ids,
            )
    if any(row is None for row in ordered):
        raise ProtocolError("HARP batched scoring lost an ordered target action.")
    return tuple(row for row in ordered if row is not None)


__all__ = ("DEFAULT_ALPHAS", "HarpActionModelBank", "HarpLodoFoldAudit", "HarpOutcomeModel", "fit_harp_action_model_bank", "score_harp_actions")
