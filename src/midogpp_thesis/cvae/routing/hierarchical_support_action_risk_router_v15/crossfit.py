"""Strict leave-one-case-out support cross-fitting for HARP v15."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .contracts import (
    CasePrediction,
    EndpointPrediction,
    LabelFreeCaseMenu,
    RouterFitConfig,
    SupportActionOutcome,
    SupportCaseClassProfile,
    SurfaceRole,
    canonical_text,
)
from .hashing import canonical_hash, require_sha256
from .hierarchical import fit_support_endpoint_model
from .outcome_normalization import (
    fit_support_fold_normalizer,
    validate_support_case_profiles,
)


@dataclass(frozen=True, slots=True)
class SupportOOFRecord:
    prediction: EndpointPrediction
    outcome: SupportActionOutcome
    fold_hash: str
    record_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.prediction, EndpointPrediction)
            or not isinstance(self.outcome, SupportActionOutcome)
            or not self.prediction.out_of_fold
            or self.prediction.action.action_hash != self.outcome.action.action_hash
            or self.prediction.menu_hash != self.outcome.menu_hash
            or self.prediction.action.surface_role
            is not SurfaceRole.TARGET_TRAIN_SUPPORT
            or self.outcome.normalization_hash is None
        ):
            raise ProtocolError("HARP v15 OOF record crossed its held-out support fold.")
        fold_hash = require_sha256(self.fold_hash, name="OOF fold hash")
        object.__setattr__(self, "fold_hash", fold_hash)
        object.__setattr__(
            self,
            "record_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_oof_record_v15",
                    "prediction_hash": self.prediction.prediction_hash,
                    "outcome_hash": self.outcome.outcome_hash,
                    "fold_hash": fold_hash,
                    "heldout_case_excluded_from_fit": True,
                    "evaluation_labels_consumed": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class SupportOOFCasePrediction:
    """Label-free heldout-case prediction seal, including exact-B controls."""

    outer_target_id: str
    case_id: str
    menu_hash: str
    prediction: CasePrediction
    fold_hash: str
    model_hash: str
    normalizer_hash: str
    training_case_ids: tuple[str, ...]
    null_model: bool
    seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = canonical_text(self.outer_target_id, name="OOF case outer target H")
        case = canonical_text(self.case_id, name="OOF heldout case id")
        menu_hash = require_sha256(self.menu_hash, name="OOF case menu hash")
        fold_hash = require_sha256(self.fold_hash, name="OOF case fold hash")
        model_hash = require_sha256(self.model_hash, name="OOF case model hash")
        normalizer_hash = require_sha256(
            self.normalizer_hash, name="OOF case normalizer hash"
        )
        training = tuple(
            sorted(
                canonical_text(value, name="OOF training case id")
                for value in self.training_case_ids
            )
        )
        rows = self.prediction.action_predictions
        if (
            not isinstance(self.prediction, CasePrediction)
            or self.prediction.menu_hash != menu_hash
            or not training
            or len(training) != len(set(training))
            or case in training
            or type(self.null_model) is not bool
            or (self.null_model and rows)
            or any(
                row.action.outer_target_id != outer
                or row.action.case_id != case
                or not row.out_of_fold
                or row.training_case_ids != training
                or row.model_hash != model_hash
                for row in rows
            )
        ):
            raise ProtocolError("HARP v15 OOF case prediction seal is malformed.")
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "menu_hash", menu_hash)
        object.__setattr__(self, "fold_hash", fold_hash)
        object.__setattr__(self, "model_hash", model_hash)
        object.__setattr__(self, "normalizer_hash", normalizer_hash)
        object.__setattr__(self, "training_case_ids", training)
        object.__setattr__(
            self,
            "seal_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_oof_case_prediction_seal_v15",
                    "outer_target_id": outer,
                    "case_id": case,
                    "menu_hash": menu_hash,
                    "prediction_hash": self.prediction.prediction_hash,
                    "fold_hash": fold_hash,
                    "model_hash": model_hash,
                    "normalizer_hash": normalizer_hash,
                    "training_case_ids": training,
                    "null_model": self.null_model,
                    "heldout_outcome_joined": False,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "hierarchical_support_oof_case_prediction_seal_v15",
            "outer_target_id": self.outer_target_id,
            "case_id": self.case_id,
            "menu_hash": self.menu_hash,
            "prediction": self.prediction.public_payload(),
            "fold_hash": self.fold_hash,
            "model_hash": self.model_hash,
            "normalizer_hash": self.normalizer_hash,
            "training_case_ids": list(self.training_case_ids),
            "null_model": self.null_model,
            "seal_hash": self.seal_hash,
            "heldout_outcome_joined": False,
            "evaluation_labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class SupportCrossfitResult:
    outer_target_id: str
    case_ids: tuple[str, ...]
    case_predictions: tuple[SupportOOFCasePrediction, ...]
    records: tuple[SupportOOFRecord, ...]
    heldout_model_hashes: tuple[tuple[str, str], ...]
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cases = tuple(sorted(self.case_ids))
        records = tuple(
            sorted(
                self.records,
                key=lambda row: (
                    row.prediction.action.case_id,
                    row.prediction.action.action_id,
                ),
            )
        )
        model_hashes = tuple(sorted(self.heldout_model_hashes))
        case_predictions = tuple(
            sorted(self.case_predictions, key=lambda row: row.case_id)
        )
        prediction_by_case = {row.case_id: row for row in case_predictions}
        if (
            not cases
            or len(cases) != len(set(cases))
            or tuple(row.case_id for row in case_predictions) != cases
            or {case for case, _ in model_hashes} != set(cases)
            or len(model_hashes) != len(cases)
            or any(row.outer_target_id != self.outer_target_id for row in case_predictions)
            or any(
                row.prediction.action.outer_target_id != self.outer_target_id
                or row.prediction.action.case_id in row.prediction.training_case_ids
                or row.fold_hash
                != prediction_by_case[row.prediction.action.case_id].fold_hash
                or row.outcome.normalization_hash
                != prediction_by_case[row.prediction.action.case_id].normalizer_hash
                or row.prediction.prediction_hash
                not in {
                    prediction.prediction_hash
                    for prediction in prediction_by_case[
                        row.prediction.action.case_id
                    ].prediction.action_predictions
                }
                for row in records
            )
        ):
            raise ProtocolError("HARP v15 support cross-fit inventory is incomplete.")
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "case_predictions", case_predictions)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "heldout_model_hashes", model_hashes)
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_case_crossfit_v15",
                    "outer_target_id": self.outer_target_id,
                    "case_ids": cases,
                    "case_prediction_seal_hashes": tuple(
                        row.seal_hash for row in case_predictions
                    ),
                    "record_hashes": tuple(row.record_hash for row in records),
                    "heldout_model_hashes": model_hashes,
                    "normalization_refit_per_fold": True,
                    "case_labels_excluded_per_fold": True,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    def records_for_case(self, case_id: str) -> tuple[SupportOOFRecord, ...]:
        return tuple(row for row in self.records if row.prediction.action.case_id == case_id)

    def prediction_for_case(self, case_id: str) -> SupportOOFCasePrediction:
        for row in self.case_predictions:
            if row.case_id == case_id:
                return row
        raise ProtocolError("HARP v15 cross-fit lacks a heldout-case prediction seal.")

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "hierarchical_support_case_crossfit_v15",
            "outer_target_id": self.outer_target_id,
            "case_ids": list(self.case_ids),
            "case_predictions": [row.public_payload() for row in self.case_predictions],
            "record_hashes": [row.record_hash for row in self.records],
            "heldout_model_hashes": [
                {"case_id": case, "model_hash": model_hash}
                for case, model_hash in self.heldout_model_hashes
            ],
            "result_hash": self.result_hash,
            "normalization_refit_per_fold": True,
            "case_labels_excluded_per_fold": True,
            "evaluation_labels_consumed": False,
        }


def validate_support_inventory(
    menus: Sequence[LabelFreeCaseMenu],
    outcomes: Sequence[SupportActionOutcome],
    *,
    minimum_support_cases: int,
) -> tuple[tuple[LabelFreeCaseMenu, ...], tuple[SupportActionOutcome, ...]]:
    menu_rows = tuple(sorted(menus, key=lambda row: row.case_id))
    outcome_rows = tuple(
        sorted(outcomes, key=lambda row: (row.action.case_id, row.action.action_id))
    )
    if (
        len(menu_rows) < int(minimum_support_cases)
        or len({row.case_id for row in menu_rows}) != len(menu_rows)
        or any(row.surface_role is not SurfaceRole.TARGET_TRAIN_SUPPORT for row in menu_rows)
    ):
        raise ProtocolError("HARP v15 support menus are missing, duplicated, or undersized.")
    outer_ids = {row.outer_target_id for row in menu_rows}
    if len(outer_ids) != 1:
        raise ProtocolError("HARP v15 support menus crossed outer targets.")
    expected: dict[str, tuple[str, str, str]] = {}
    for menu in menu_rows:
        for action in menu.actions:
            expected[action.action_hash] = (menu.case_id, action.action_id, menu.menu_hash)
    if len(expected) != sum(len(row.actions) for row in menu_rows):
        raise ProtocolError("HARP v15 support action hashes are not unique.")
    observed: dict[str, tuple[str, str, str]] = {}
    for row in outcome_rows:
        observed[row.action.action_hash] = (
            row.action.case_id,
            row.action.action_id,
            row.menu_hash,
        )
    if (
        len(observed) != len(outcome_rows)
        or observed != expected
        or any(row.action.outer_target_id not in outer_ids for row in outcome_rows)
        or any(not row.has_class_local_components for row in outcome_rows)
        or any(row.normalization_case_count is not None for row in outcome_rows)
    ):
        raise ProtocolError(
            "HARP v15 support outcomes do not exactly cover the sealed primitive action menus."
        )
    return menu_rows, outcome_rows


def leave_one_case_out_crossfit(
    menus: Sequence[LabelFreeCaseMenu],
    outcomes: Sequence[SupportActionOutcome],
    *,
    config: RouterFitConfig,
    minimum_support_cases: int | None = None,
    case_profiles: Sequence[SupportCaseClassProfile] = (),
    candidate_source_ids: Sequence[str] | None = None,
) -> SupportCrossfitResult:
    minimum = (
        config.minimum_support_cases
        if minimum_support_cases is None
        else int(minimum_support_cases)
    )
    if minimum < 3:
        raise ProtocolError("HARP v15 cross-fit requires at least three support cases.")
    menu_rows, outcome_rows = validate_support_inventory(
        menus,
        outcomes,
        minimum_support_cases=minimum,
    )
    outcomes_by_action: Mapping[str, SupportActionOutcome] = {
        row.action.action_hash: row for row in outcome_rows
    }
    profile_rows = validate_support_case_profiles(
        menu_rows,
        case_profiles,
        require_complete=True,
    )
    observed_candidates = {
        action.candidate_source_id
        for menu in menu_rows
        for action in menu.actions
        if action.candidate_source_id is not None
    }
    candidates = tuple(
        sorted(
            canonical_text(value, name="crossfit candidate source")
            for value in (
                observed_candidates
                if candidate_source_ids is None
                else tuple(candidate_source_ids)
            )
        )
    )
    if (
        len(candidates) != len(set(candidates))
        or not observed_candidates.issubset(candidates)
        or menu_rows[0].outer_target_id in candidates
    ):
        raise ProtocolError("HARP v15 crossfit candidate universe is malformed.")
    records: list[SupportOOFRecord] = []
    case_predictions: list[SupportOOFCasePrediction] = []
    heldout_models: list[tuple[str, str]] = []
    for heldout in menu_rows:
        training_raw = tuple(
            row for row in outcome_rows if row.action.case_id != heldout.case_id
        )
        training_case_ids = tuple(
            row.case_id for row in menu_rows if row.case_id != heldout.case_id
        )
        normalizer = fit_support_fold_normalizer(profile_rows, training_case_ids)
        training = tuple(normalizer.normalize(row) for row in training_raw)
        model = fit_support_endpoint_model(
            training,
            config=config,
            candidate_source_ids=candidates,
            training_case_ids=training_case_ids,
            outer_target_id=heldout.outer_target_id,
        )
        prediction = model.predict_menu(heldout, out_of_fold=True)
        fold_hash = canonical_hash(
            {
                "schema_version": "hierarchical_support_case_fold_v15",
                "outer_target_id": heldout.outer_target_id,
                "heldout_case_id": heldout.case_id,
                "heldout_menu_hash": heldout.menu_hash,
                "model_hash": model.model_hash,
                "training_case_ids": model.training_case_ids,
                "candidate_source_ids": candidates,
                "null_model": model.is_null,
                "bacc_normalization_fit_case_ids": training_case_ids,
                "bacc_normalization_hash": normalizer.normalizer_hash,
                "heldout_label_used": False,
                "heldout_features_used_for_normalization": False,
            }
        )
        heldout_models.append((heldout.case_id, model.model_hash))
        case_prediction = SupportOOFCasePrediction(
            outer_target_id=heldout.outer_target_id,
            case_id=heldout.case_id,
            menu_hash=heldout.menu_hash,
            prediction=prediction,
            fold_hash=fold_hash,
            model_hash=model.model_hash,
            normalizer_hash=normalizer.normalizer_hash,
            training_case_ids=model.training_case_ids,
            null_model=model.is_null,
        )
        # Only after the complete heldout prediction exists do label-derived
        # primitive outcomes enter the OOF scoring record.
        case_predictions.append(case_prediction)
        heldout_outcomes = tuple(
            normalizer.normalize(row)
            for row in tuple(
                outcomes_by_action[action.action_hash] for action in heldout.actions
            )
        )
        heldout_by_action = {
            row.action.action_hash: row for row in heldout_outcomes
        }
        records.extend(
            SupportOOFRecord(
                prediction=row,
                outcome=heldout_by_action[row.action.action_hash],
                fold_hash=fold_hash,
            )
            for row in prediction.action_predictions
        )
    return SupportCrossfitResult(
        outer_target_id=menu_rows[0].outer_target_id,
        case_ids=tuple(row.case_id for row in menu_rows),
        case_predictions=tuple(case_predictions),
        records=tuple(records),
        heldout_model_hashes=tuple(heldout_models),
    )


__all__ = (
    "SupportCrossfitResult",
    "SupportOOFCasePrediction",
    "SupportOOFRecord",
    "leave_one_case_out_crossfit",
    "validate_support_inventory",
)
