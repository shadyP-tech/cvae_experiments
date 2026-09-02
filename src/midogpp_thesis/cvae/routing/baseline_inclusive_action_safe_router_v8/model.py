"""Baseline-inclusive action heads with strict outer/held-center LODO.

The CPU implementation is deliberately vectorized and does not create worker
pools.  Workstation-level process and BLAS limits remain the runner's concern.
Every target-facing model is trained only on source-development outcomes and
is paired with residual envelopes built from predictions that held out both a
query center and every appearance of that center as a candidate expert.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .certification import (
    ResidualCalibration,
    ResidualObservation,
    calibrate_center_group_residuals,
    certify_action,
)
from .contracts import (
    ActionCertificate,
    ActionEstimate,
    CasePrediction,
    LabelFreeAction,
    SourceActionOutcome,
    action_group,
)
from .effective_menu import EffectiveMenu, build_effective_menu
from .hashing import canonical_hash


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Predeclared action-model and safety-certificate configuration.

    Candidate actions are safety-filtered by calibrated harm and proper-loss
    bounds.  Their signed BACC estimates are used only to rank that safe set;
    positive utility is required from the nested whole-policy replay rather
    than from an absolute per-case sign test.
    """

    harm_alpha: float = 1.0
    endpoint_alpha: float = 1.0
    max_irls_iterations: int = 32
    residual_quantile: float = 0.8
    max_harm_probability: float = 0.25
    max_action_brier_delta: float = 0.002
    max_action_log_delta: float = 0.005
    max_harm_brier_risk: float = 0.25
    max_harm_log_loss_risk: float = 0.7
    min_calibration_centers: int = 2
    min_calibration_rows_per_group: int = 2

    def __post_init__(self) -> None:
        numeric = (
            self.harm_alpha,
            self.endpoint_alpha,
            self.residual_quantile,
            self.max_harm_probability,
            self.max_action_brier_delta,
            self.max_action_log_delta,
            self.max_harm_brier_risk,
            self.max_harm_log_loss_risk,
        )
        if (
            any(not math.isfinite(value) for value in numeric)
            or self.harm_alpha <= 0.0
            or self.endpoint_alpha <= 0.0
            or int(self.max_irls_iterations) < 1
            or not 0.0 < self.residual_quantile <= 1.0
            or not 0.0 <= self.max_harm_probability <= 1.0
            or self.max_harm_brier_risk < 0.0
            or self.max_harm_log_loss_risk < 0.0
            or int(self.min_calibration_centers) < 1
            or int(self.min_calibration_rows_per_group) < 1
        ):
            raise ProtocolError("HARP v8 fit configuration is malformed.")

@dataclass(frozen=True, slots=True)
class Standardizer:
    names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not self.names
            or len(self.names) != len(self.mean)
            or len(self.names) != len(self.scale)
            or len(set(self.names)) != len(self.names)
            or any(not math.isfinite(value) for value in (*self.mean, *self.scale))
            or any(value <= 0.0 for value in self.scale)
        ):
            raise ProtocolError("HARP v8 feature standardizer is malformed.")


@dataclass(frozen=True, slots=True)
class LinearHead:
    intercept: float
    coefficients: tuple[float, ...]
    available: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.intercept) or any(
            not math.isfinite(value) for value in self.coefficients
        ):
            raise ProtocolError("HARP v8 linear head contains non-finite values.")


@dataclass(frozen=True, slots=True)
class ActionHeads:
    action_group: str
    gain_head: LinearHead
    harm_head: LinearHead
    brier_head: LinearHead
    log_head: LinearHead
    training_row_count: int

    def __post_init__(self) -> None:
        widths = {
            len(self.gain_head.coefficients),
            len(self.harm_head.coefficients),
            len(self.brier_head.coefficients),
            len(self.log_head.coefficients),
        }
        if len(widths) != 1 or self.training_row_count < 1:
            raise ProtocolError("HARP v8 action-head block is malformed.")


@dataclass(frozen=True, slots=True)
class BaselineInclusiveRouterModel:
    outer_target_id: str
    training_center_ids: tuple[str, ...]
    training_candidate_ids: tuple[str, ...]
    excluded_center_ids: tuple[str, ...]
    action_standardizer: Standardizer
    action_heads: tuple[ActionHeads, ...]
    residual_calibration: ResidualCalibration
    fit_config: FitConfig
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        training = tuple(sorted(self.training_center_ids))
        candidates = tuple(sorted(self.training_candidate_ids))
        excluded = tuple(sorted(self.excluded_center_ids))
        heads = tuple(sorted(self.action_heads, key=lambda row: row.action_group))
        if (
            not training
            or len(set(training)) != len(training)
            or len(set(candidates)) != len(candidates)
            or len(set(excluded)) != len(excluded)
            or self.outer_target_id not in excluded
            or set(training) & set(excluded)
            or set(candidates) & set(excluded)
            or len({row.action_group for row in heads}) != len(heads)
            or any(
                len(row.gain_head.coefficients) != len(self.action_standardizer.names)
                for row in heads
            )
            or self.residual_calibration.outer_target_id != self.outer_target_id
        ):
            raise ProtocolError("HARP v8 model roles, heads, or calibration are malformed.")
        object.__setattr__(self, "training_center_ids", training)
        object.__setattr__(self, "training_candidate_ids", candidates)
        object.__setattr__(self, "excluded_center_ids", excluded)
        object.__setattr__(self, "action_heads", heads)
        object.__setattr__(
            self,
            "model_hash",
            canonical_hash(
                {
                    "schema_version": "baseline_inclusive_action_safe_router_model_v8",
                    "outer_target_id": self.outer_target_id,
                    "training_center_ids": training,
                    "training_candidate_ids": candidates,
                    "excluded_center_ids": excluded,
                    "action_standardizer": self.action_standardizer,
                    "action_heads": heads,
                    "residual_calibration_hash": self.residual_calibration.calibration_hash,
                    "fit_config": self.fit_config,
                    "baseline_B_explicit": True,
                    "target_evaluation_labels_used": False,
                }
            ),
        )

    def heads_for(self, group: str) -> ActionHeads | None:
        return next((row for row in self.action_heads if row.action_group == group), None)


@dataclass(frozen=True, slots=True)
class ConfigTuningScore:
    config: FitConfig
    gain_mse: float
    harm_brier: float
    harm_log_loss: float
    brier_delta_mse: float
    log_delta_mse: float
    combined_loss: float
    evaluated_action_count: int
    score_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            self.gain_mse,
            self.harm_brier,
            self.harm_log_loss,
            self.brier_delta_mse,
            self.log_delta_mse,
            self.combined_loss,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values) or self.evaluated_action_count < 1:
            raise ProtocolError("HARP v8 tuning score is malformed.")
        object.__setattr__(
            self,
            "score_hash",
            canonical_hash(
                {
                    "schema_version": "baseline_inclusive_config_tuning_score_v8",
                    "config": self.config,
                    "gain_mse": self.gain_mse,
                    "harm_brier": self.harm_brier,
                    "harm_log_loss": self.harm_log_loss,
                    "brier_delta_mse": self.brier_delta_mse,
                    "log_delta_mse": self.log_delta_mse,
                    "combined_loss": self.combined_loss,
                    "evaluated_action_count": self.evaluated_action_count,
                }
            ),
        )

@dataclass(frozen=True, slots=True)
class ConfigSelection:
    fold_id: str
    training_center_ids: tuple[str, ...]
    selected_config: FitConfig
    scores: tuple[ConfigTuningScore, ...]
    selection_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.fold_id or not self.training_center_ids or not self.scores:
            raise ProtocolError("HARP v8 config selection is incomplete.")
        if self.selected_config not in {row.config for row in self.scores}:
            raise ProtocolError("HARP v8 selected config is absent from its score surface.")
        object.__setattr__(
            self,
            "selection_hash",
            canonical_hash(
                {
                    "schema_version": "baseline_inclusive_config_selection_v8",
                    "fold_id": self.fold_id,
                    "training_center_ids": self.training_center_ids,
                    "selected_config": self.selected_config,
                    "score_hashes": tuple(row.score_hash for row in self.scores),
                    "held_source_excluded": True,
                    "target_evaluation_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class NestedPolicyFold:
    """Training OOF policy surface plus one fully held-center replay surface."""

    heldout_center_id: str
    training_center_ids: tuple[str, ...]
    selected_config: FitConfig
    predictions: tuple[CasePrediction, ...]
    heldout_predictions: tuple[CasePrediction, ...]
    fold_hash: str = field(init=False)

    def __post_init__(self) -> None:
        training = tuple(sorted(self.training_center_ids))
        if (
            self.heldout_center_id in training
            or not training
            or not self.predictions
            or not self.heldout_predictions
            or {row.query_center_id for row in self.predictions} != set(training)
            or {row.query_center_id for row in self.heldout_predictions} != {self.heldout_center_id}
            or any(
                row.query_center_id in row.training_center_ids
                or row.query_center_id in row.training_candidate_ids
                or self.heldout_center_id not in row.excluded_center_ids
                for row in (*self.predictions, *self.heldout_predictions)
            )
        ):
            raise ProtocolError("HARP v8 nested held-center policy fold leaked or is incomplete.")
        object.__setattr__(self, "training_center_ids", training)
        object.__setattr__(
            self,
            "fold_hash",
            canonical_hash(
                {
                    "schema_version": "baseline_inclusive_nested_policy_fold_v8",
                    "heldout_center_id": self.heldout_center_id,
                    "training_center_ids": training,
                    "selected_config": self.selected_config,
                    "training_prediction_hashes": tuple(row.prediction_hash for row in self.predictions),
                    "heldout_prediction_hashes": tuple(row.prediction_hash for row in self.heldout_predictions),
                    "heldout_query_and_candidate_excluded_from_all_fits": True,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceLODOResult:
    outer_target_id: str
    final_model: BaselineInclusiveRouterModel
    oof_predictions: tuple[CasePrediction, ...]
    heldout_model_hashes: tuple[tuple[str, str], ...]
    config_selections: tuple[ConfigSelection, ...]
    nested_policy_folds: tuple[NestedPolicyFold, ...]
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.final_model.outer_target_id != self.outer_target_id:
            raise ProtocolError("HARP v8 final model crossed its outer target.")
        centers = tuple(sorted({row.query_center_id for row in self.oof_predictions}))
        if (
            not centers
            or any(
                row.outer_target_id != self.outer_target_id
                or row.query_center_id in row.training_center_ids
                or row.query_center_id in row.training_candidate_ids
                or row.query_center_id not in row.excluded_center_ids
                for row in self.oof_predictions
            )
            or len({(row.query_center_id, row.case_id) for row in self.oof_predictions})
            != len(self.oof_predictions)
            or {row.heldout_center_id for row in self.nested_policy_folds} != set(centers)
            or {center for center, _ in self.heldout_model_hashes} != set(centers)
        ):
            raise ProtocolError("HARP v8 source LODO inventory leaked or is incomplete.")
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "baseline_inclusive_source_lodo_v8",
                    "outer_target_id": self.outer_target_id,
                    "final_model_hash": self.final_model.model_hash,
                    "oof_prediction_hashes": tuple(row.prediction_hash for row in self.oof_predictions),
                    "heldout_model_hashes": self.heldout_model_hashes,
                    "config_selection_hashes": tuple(row.selection_hash for row in self.config_selections),
                    "nested_policy_fold_hashes": tuple(row.fold_hash for row in self.nested_policy_folds),
                    "strict_outer_and_source_query_candidate_lodo": True,
                }
            ),
        )

    def numeric_oof_payload(self) -> dict[str, object]:
        def prediction_row(row: CasePrediction) -> dict[str, object]:
            return {
                "query_center_id": row.query_center_id,
                "case_id": row.case_id,
                "safe_action_ids": list(row.safe_action_ids),
                "raw_top_action_id": row.raw_top_action_id,
                "top_action_id": row.top_action_id,
                "rank_margin": row.rank_margin,
                "training_center_ids": list(row.training_center_ids),
                "training_candidate_ids": list(row.training_candidate_ids),
                "excluded_center_ids": list(row.excluded_center_ids),
                "model_hash": row.model_hash,
                "menu_hash": row.menu_hash,
                "prediction_hash": row.prediction_hash,
                "action_certificates": [
                    {
                        "action_id": certificate.action_id,
                        "action_hash": certificate.action_hash,
                        "direction": certificate.direction.value,
                        "action_group": certificate.estimate.action_group,
                        "model_available": certificate.estimate.model_available,
                        "predicted_bacc_gain": certificate.estimate.predicted_bacc_gain,
                        "predicted_harm_probability": certificate.estimate.predicted_harm_probability,
                        "predicted_brier_delta": certificate.estimate.predicted_brier_delta,
                        "predicted_log_delta": certificate.estimate.predicted_log_delta,
                        "gain_lcb": certificate.gain_lcb,
                        "harm_probability_ucb": certificate.harm_probability_ucb,
                        "brier_delta_ucb": certificate.brier_delta_ucb,
                        "log_delta_ucb": certificate.log_delta_ucb,
                        "harm_brier_risk": certificate.harm_brier_risk,
                        "harm_log_loss_risk": certificate.harm_log_loss_risk,
                        "calibration_cell_hash": certificate.calibration_cell_hash,
                        "safe": certificate.safe,
                        "failed_gates": list(certificate.failed_gates),
                        "certificate_hash": certificate.certificate_hash,
                    }
                    for certificate in row.action_certificates
                ],
            }

        return {
            "schema_version": "baseline_inclusive_numeric_oof_v8",
            "outer_target_id": self.outer_target_id,
            "result_hash": self.result_hash,
            "config_selections": [
                {
                    "fold_id": selection.fold_id,
                    "training_center_ids": list(selection.training_center_ids),
                    "selected_config": selection.selected_config.__dict__
                    if hasattr(selection.selected_config, "__dict__")
                    else {
                        name: getattr(selection.selected_config, name)
                        for name in selection.selected_config.__dataclass_fields__
                    },
                    "selection_hash": selection.selection_hash,
                    "scores": [
                        {
                            "gain_mse": score.gain_mse,
                            "harm_brier": score.harm_brier,
                            "harm_log_loss": score.harm_log_loss,
                            "brier_delta_mse": score.brier_delta_mse,
                            "log_delta_mse": score.log_delta_mse,
                            "combined_loss": score.combined_loss,
                            "evaluated_action_count": score.evaluated_action_count,
                            "score_hash": score.score_hash,
                        }
                        for score in selection.scores
                    ],
                }
                for selection in self.config_selections
            ],
            "nested_policy_folds": [
                {
                    "heldout_center_id": fold.heldout_center_id,
                    "training_center_ids": list(fold.training_center_ids),
                    "fold_hash": fold.fold_hash,
                    "training_rows": [prediction_row(row) for row in fold.predictions],
                    "heldout_rows": [prediction_row(row) for row in fold.heldout_predictions],
                }
                for fold in self.nested_policy_folds
            ],
            "rows": [prediction_row(row) for row in self.oof_predictions],
        }


@dataclass(frozen=True, slots=True)
class _SourceCase:
    menu: EffectiveMenu
    outcomes: tuple[SourceActionOutcome, ...]


@dataclass(frozen=True, slots=True)
class _BaseModel:
    outer_target_id: str
    training_center_ids: tuple[str, ...]
    training_candidate_ids: tuple[str, ...]
    excluded_center_ids: tuple[str, ...]
    action_standardizer: Standardizer
    action_heads: tuple[ActionHeads, ...]
    fit_config: FitConfig
    base_hash: str


@dataclass(frozen=True, slots=True)
class _RawCasePrediction:
    outer_target_id: str
    query_center_id: str
    case_id: str
    estimates: tuple[ActionEstimate, ...]
    base_hash: str
    training_center_ids: tuple[str, ...]
    training_candidate_ids: tuple[str, ...]
    excluded_center_ids: tuple[str, ...]
    menu_hash: str


def _source_cases(
    observations: Sequence[SourceActionOutcome],
    effective_menus: Sequence[EffectiveMenu] | None,
    *,
    min_centers: int,
) -> tuple[_SourceCase, ...]:
    rows = tuple(observations)
    if any(not isinstance(row, SourceActionOutcome) for row in rows):
        raise ProtocolError("HARP v8 fitting requires source-development outcomes.")
    if effective_menus is None:
        if not rows:
            raise ProtocolError("HARP v8 source surface is empty.")
        grouped: dict[tuple[str, str], list[SourceActionOutcome]] = defaultdict(list)
        for row in rows:
            grouped[(row.action.query_center_id, row.action.case_id)].append(row)
        case_rows: list[_SourceCase] = []
        for key in sorted(grouped):
            members = grouped[key]
            if len({row.action.action_id for row in members}) != len(members):
                raise ProtocolError("HARP v8 source case contains duplicate action ids.")
            menu = build_effective_menu(tuple(row.action for row in members))
            by_id = {row.action.action_id: row for row in members}
            for duplicate, representative in menu.duplicate_representatives:
                left, right = by_id[duplicate], by_id[representative]
                if (left.bacc_gain, left.brier_delta, left.log_delta) != (
                    right.bacc_gain,
                    right.brier_delta,
                    right.log_delta,
                ):
                    raise ProtocolError(
                        "HARP v8 duplicate physical actions have inconsistent outcomes."
                    )
            case_rows.append(
                _SourceCase(
                    menu=menu,
                    outcomes=tuple(by_id[action.action_id] for action in menu.actions),
                )
            )
        cases = tuple(case_rows)
    else:
        menus = tuple(effective_menus)
        if not menus:
            raise ProtocolError("HARP v8 effective-menu surface is empty.")
        by_key = {(menu.query_center_id, menu.case_id): menu for menu in menus}
        if len(by_key) != len(menus):
            raise ProtocolError("HARP v8 effective-menu inventory contains duplicate cases.")
        outcomes: dict[tuple[str, str], list[SourceActionOutcome]] = defaultdict(list)
        for row in rows:
            key = (row.action.query_center_id, row.action.case_id)
            menu = by_key.get(key)
            if menu is None or not any(
                action.action_id == row.action.action_id
                and action.action_hash == row.action.action_hash
                for action in menu.actions
            ):
                raise ProtocolError("HARP v8 outcome is absent from its sealed effective menu.")
            outcomes[key].append(row)
        cases_list: list[_SourceCase] = []
        for key in sorted(by_key):
            menu = by_key[key]
            members = tuple(sorted(outcomes.get(key, ()), key=lambda row: row.action.action_id))
            if {row.action.action_id for row in members} != {row.action_id for row in menu.actions}:
                raise ProtocolError("HARP v8 source outcome/effective-menu inventory is incomplete.")
            cases_list.append(_SourceCase(menu=menu, outcomes=members))
        cases = tuple(cases_list)
    outers = {case.menu.outer_target_id for case in cases}
    schemas = {case.menu.feature_names for case in cases}
    centers = {case.menu.query_center_id for case in cases}
    if (
        len(outers) != 1
        or len(schemas) != 1
        or len(centers) < min_centers
        or next(iter(outers)) in centers
    ):
        raise ProtocolError("HARP v8 source surface crossed outer/schema/center roles.")
    return cases


def _filter_surface(
    observations: tuple[SourceActionOutcome, ...],
    effective_menus: tuple[EffectiveMenu, ...] | None,
    *,
    excluded_center_ids: frozenset[str],
) -> tuple[tuple[SourceActionOutcome, ...], tuple[EffectiveMenu, ...] | None]:
    rows = tuple(
        row
        for row in observations
        if row.action.query_center_id not in excluded_center_ids
        and row.action.candidate_source_id not in excluded_center_ids
    )
    if effective_menus is None:
        return rows, None
    menus: list[EffectiveMenu] = []
    for menu in effective_menus:
        if menu.query_center_id in excluded_center_ids:
            continue
        actions = tuple(
            row for row in menu.actions if row.candidate_source_id not in excluded_center_ids
        )
        aliases = tuple(
            pair for pair in menu.duplicate_representatives if pair[1] in {row.action_id for row in actions}
        )
        menus.append(
            EffectiveMenu(
                outer_target_id=menu.outer_target_id,
                query_center_id=menu.query_center_id,
                case_id=menu.case_id,
                feature_names=menu.feature_names,
                baseline_probability_hex=menu.baseline_probability_hex,
                actions=actions,
                dropped_noop_action_ids=menu.dropped_noop_action_ids,
                duplicate_representatives=aliases,
            )
        )
    allowed_hashes = {action.action_hash for menu in menus for action in menu.actions}
    return tuple(row for row in rows if row.action.action_hash in allowed_hashes), tuple(menus)


def _standardize_fit(
    names: tuple[str, ...], matrix: np.ndarray, weights: np.ndarray
) -> tuple[Standardizer, np.ndarray]:
    normalized = weights / np.sum(weights, dtype=np.float64)
    mean = np.sum(normalized[:, None] * matrix, axis=0, dtype=np.float64)
    variance = np.sum(normalized[:, None] * (matrix - mean) ** 2, axis=0, dtype=np.float64)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale <= math.sqrt(np.finfo(np.float64).eps)] = 1.0
    standardizer = Standardizer(
        names=names,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
    )
    return standardizer, (matrix - mean) / scale


def _apply_standardizer(values: np.ndarray, standardizer: Standardizer) -> np.ndarray:
    return (values - np.asarray(standardizer.mean)) / np.asarray(standardizer.scale)


def _ridge_head(matrix: np.ndarray, response: np.ndarray, weights: np.ndarray, alpha: float) -> LinearHead:
    design = np.column_stack((np.ones(matrix.shape[0]), matrix))
    normalized = weights * (len(weights) / np.sum(weights, dtype=np.float64))
    normal = design.T @ (normalized[:, None] * design)
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    rhs = design.T @ (normalized * response)
    coefficients = np.linalg.lstsq(normal + penalty, rhs, rcond=None)[0]
    return LinearHead(float(coefficients[0]), tuple(float(value) for value in coefficients[1:]), True)


def _logistic_head(
    matrix: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    max_iterations: int,
) -> LinearHead:
    normalized = weights * (len(weights) / np.sum(weights, dtype=np.float64))
    if np.all(response == response[0]):
        positive = float(np.sum(normalized * response, dtype=np.float64))
        probability = (positive + 0.5) / (float(np.sum(normalized)) + 1.0)
        return LinearHead(
            math.log(probability / (1.0 - probability)),
            tuple(0.0 for _ in range(matrix.shape[1])),
            True,
        )
    design = np.column_stack((np.ones(matrix.shape[0]), matrix))
    coefficients = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    for _ in range(max_iterations):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        curvature = np.maximum(probability * (1.0 - probability), 1e-6)
        adjusted = linear + (response - probability) / curvature
        combined = normalized * curvature
        updated = np.linalg.lstsq(
            design.T @ (combined[:, None] * design) + penalty,
            design.T @ (combined * adjusted),
            rcond=None,
        )[0]
        if float(np.max(np.abs(updated - coefficients))) <= 1e-10:
            coefficients = updated
            break
        coefficients = updated
    return LinearHead(float(coefficients[0]), tuple(float(value) for value in coefficients[1:]), True)


def _fit_base(
    observations: tuple[SourceActionOutcome, ...],
    effective_menus: tuple[EffectiveMenu, ...] | None,
    *,
    config: FitConfig,
    excluded_center_ids: tuple[str, ...],
) -> _BaseModel:
    cases = _source_cases(observations, effective_menus, min_centers=1)
    outer = cases[0].menu.outer_target_id
    if outer not in excluded_center_ids:
        raise ProtocolError("HARP v8 base fit did not exclude outer H.")
    action_rows = [(case, outcome) for case in cases for outcome in case.outcomes]
    if not action_rows:
        raise ProtocolError("HARP v8 base fit has no physical source actions.")
    names = cases[0].menu.feature_names
    matrix = np.asarray([row.action.feature_values for _, row in action_rows], dtype=np.float64)
    cases_per_center: dict[str, int] = defaultdict(int)
    actions_per_case: dict[tuple[str, str], int] = defaultdict(int)
    for case in cases:
        cases_per_center[case.menu.query_center_id] += 1
        actions_per_case[(case.menu.query_center_id, case.menu.case_id)] = max(len(case.outcomes), 1)
    weights = np.asarray(
        [
            1.0
            / (
                len(cases_per_center)
                * cases_per_center[case.menu.query_center_id]
                * actions_per_case[(case.menu.query_center_id, case.menu.case_id)]
            )
            for case, _ in action_rows
        ],
        dtype=np.float64,
    )
    standardizer, standardized = _standardize_fit(names, matrix, weights)
    heads: list[ActionHeads] = []
    groups = [action_group(row.action) for _, row in action_rows]
    for group in sorted(set(groups)):
        indices = np.asarray([value == group for value in groups], dtype=bool)
        group_rows = [row for index, (_, row) in enumerate(action_rows) if indices[index]]
        x = standardized[indices]
        w = weights[indices]
        gain = np.asarray([row.bacc_gain for row in group_rows])
        harm = np.asarray([row.bacc_gain < 0.0 for row in group_rows], dtype=np.float64)
        brier = np.asarray([row.brier_delta for row in group_rows])
        log_delta = np.asarray([row.log_delta for row in group_rows])
        heads.append(
            ActionHeads(
                action_group=group,
                gain_head=_ridge_head(x, gain, w, config.endpoint_alpha),
                harm_head=_logistic_head(
                    x,
                    harm,
                    w,
                    alpha=config.harm_alpha,
                    max_iterations=config.max_irls_iterations,
                ),
                brier_head=_ridge_head(x, brier, w, config.endpoint_alpha),
                log_head=_ridge_head(x, log_delta, w, config.endpoint_alpha),
                training_row_count=len(group_rows),
            )
        )
    training = tuple(sorted({case.menu.query_center_id for case in cases}))
    candidates = tuple(
        sorted(
            {
                row.action.candidate_source_id
                for _, row in action_rows
                if row.action.candidate_source_id is not None
            }
        )
    )
    excluded = tuple(sorted(excluded_center_ids))
    if set(training) & set(excluded) or set(candidates) & set(excluded):
        raise ProtocolError("HARP v8 held query/candidate entered a base fit.")
    base_hash = canonical_hash(
        {
            "schema_version": "baseline_inclusive_base_model_v8",
            "outer_target_id": outer,
            "training_center_ids": training,
            "training_candidate_ids": candidates,
            "excluded_center_ids": excluded,
            "action_standardizer": standardizer,
            "action_heads": heads,
            "fit_config": config,
        }
    )
    return _BaseModel(outer, training, candidates, excluded, standardizer, tuple(heads), config, base_hash)


def _linear(head: LinearHead, vector: np.ndarray) -> float:
    return float(head.intercept + np.dot(np.asarray(head.coefficients), vector))


def _predict_raw(model: _BaseModel, menu: EffectiveMenu) -> _RawCasePrediction:
    if (
        menu.outer_target_id != model.outer_target_id
        or menu.query_center_id not in model.excluded_center_ids
        or menu.query_center_id in model.training_center_ids
        or menu.query_center_id in model.training_candidate_ids
        or menu.feature_names != model.action_standardizer.names
    ):
        raise ProtocolError("HARP v8 raw prediction crossed model/query/schema roles.")
    by_group = {row.action_group: row for row in model.action_heads}
    estimates: list[ActionEstimate] = []
    for action in menu.actions:
        group = action_group(action)
        heads = by_group.get(group)
        if heads is None:
            estimates.append(
                ActionEstimate(
                    action.action_id,
                    action.action_hash,
                    group,
                    action.direction,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    False,
                )
            )
            continue
        vector = _apply_standardizer(np.asarray(action.feature_values), model.action_standardizer)
        harm_linear = float(np.clip(_linear(heads.harm_head, vector), -30.0, 30.0))
        estimates.append(
            ActionEstimate(
                action.action_id,
                action.action_hash,
                group,
                action.direction,
                _linear(heads.gain_head, vector),
                1.0 / (1.0 + math.exp(-harm_linear)),
                _linear(heads.brier_head, vector),
                _linear(heads.log_head, vector),
                True,
            )
        )
    return _RawCasePrediction(
        model.outer_target_id,
        menu.query_center_id,
        menu.case_id,
        tuple(estimates),
        model.base_hash,
        model.training_center_ids,
        model.training_candidate_ids,
        model.excluded_center_ids,
        menu.menu_hash,
    )


def _crossfit_raw(
    observations: tuple[SourceActionOutcome, ...],
    effective_menus: tuple[EffectiveMenu, ...] | None,
    *,
    config: FitConfig,
    fixed_excluded: tuple[str, ...],
) -> tuple[_RawCasePrediction, ...]:
    cases = _source_cases(observations, effective_menus, min_centers=2)
    centers = tuple(sorted({case.menu.query_center_id for case in cases}))
    output: list[_RawCasePrediction] = []
    for heldout in centers:
        excluded = frozenset((*fixed_excluded, heldout))
        rows, menus = _filter_surface(
            observations,
            effective_menus,
            excluded_center_ids=excluded,
        )
        base = _fit_base(rows, menus, config=config, excluded_center_ids=tuple(sorted(excluded)))
        output.extend(
            _predict_raw(base, case.menu)
            for case in cases
            if case.menu.query_center_id == heldout
        )
    return tuple(sorted(output, key=lambda row: (row.query_center_id, row.case_id)))


def _residual_rows(
    predictions: Sequence[_RawCasePrediction], cases: Sequence[_SourceCase]
) -> tuple[ResidualObservation, ...]:
    by_key = {(row.query_center_id, row.case_id): row for row in predictions}
    if set(by_key) != {(case.menu.query_center_id, case.menu.case_id) for case in cases}:
        raise ProtocolError("HARP v8 residual OOF predictions do not cover their cases.")
    output: list[ResidualObservation] = []
    for case in cases:
        prediction = by_key[(case.menu.query_center_id, case.menu.case_id)]
        by_action = {row.action_id: row for row in prediction.estimates}
        for outcome in case.outcomes:
            estimate = by_action.get(outcome.action.action_id)
            if estimate is None:
                raise ProtocolError("HARP v8 residual OOF prediction lacks an action.")
            output.append(ResidualObservation(case.menu.query_center_id, estimate, outcome))
    return tuple(output)


def _fit_certified_model(
    observations: tuple[SourceActionOutcome, ...],
    effective_menus: tuple[EffectiveMenu, ...] | None,
    *,
    config: FitConfig,
    fixed_excluded: tuple[str, ...],
    raw_oof: tuple[_RawCasePrediction, ...] | None = None,
) -> BaselineInclusiveRouterModel:
    base = _fit_base(
        observations,
        effective_menus,
        config=config,
        excluded_center_ids=fixed_excluded,
    )
    if raw_oof is None:
        raw_oof = _crossfit_raw(
            observations,
            effective_menus,
            config=config,
            fixed_excluded=fixed_excluded,
        )
    cases = _source_cases(observations, effective_menus, min_centers=2)
    residual = calibrate_center_group_residuals(
        _residual_rows(raw_oof, cases),
        outer_target_id=base.outer_target_id,
        residual_quantile=config.residual_quantile,
        min_calibration_centers=config.min_calibration_centers,
        min_calibration_rows_per_group=config.min_calibration_rows_per_group,
    )
    return BaselineInclusiveRouterModel(
        outer_target_id=base.outer_target_id,
        training_center_ids=base.training_center_ids,
        training_candidate_ids=base.training_candidate_ids,
        excluded_center_ids=base.excluded_center_ids,
        action_standardizer=base.action_standardizer,
        action_heads=base.action_heads,
        residual_calibration=residual,
        fit_config=config,
    )


def fit_baseline_inclusive_router(
    observations: Sequence[SourceActionOutcome],
    *,
    config: FitConfig = FitConfig(),
    effective_menus: Sequence[EffectiveMenu] | None = None,
) -> BaselineInclusiveRouterModel:
    """Fit one target-facing source-only model with source-OOF certificates."""

    rows = tuple(observations)
    menus = None if effective_menus is None else tuple(effective_menus)
    cases = _source_cases(rows, menus, min_centers=max(2, config.min_calibration_centers))
    outer = cases[0].menu.outer_target_id
    return _fit_certified_model(rows, menus, config=config, fixed_excluded=(outer,))


def predict_case(model: BaselineInclusiveRouterModel, menu: EffectiveMenu) -> CasePrediction:
    """Predict and certify a case without accepting any outcome argument."""

    base = _BaseModel(
        model.outer_target_id,
        model.training_center_ids,
        model.training_candidate_ids,
        model.excluded_center_ids,
        model.action_standardizer,
        model.action_heads,
        model.fit_config,
        canonical_hash(
            {
                "schema_version": "baseline_inclusive_base_projection_v8",
                "model_hash": model.model_hash,
            }
        ),
    )
    raw = _predict_raw(base, menu)
    certificates = tuple(
        certify_action(
            estimate,
            model.residual_calibration.for_group(estimate.action_group),
            max_harm_probability=model.fit_config.max_harm_probability,
            max_brier_delta=model.fit_config.max_action_brier_delta,
            max_log_delta=model.fit_config.max_action_log_delta,
            max_harm_brier_risk=model.fit_config.max_harm_brier_risk,
            max_harm_log_loss_risk=model.fit_config.max_harm_log_loss_risk,
        )
        for estimate in raw.estimates
    )
    return CasePrediction(
        outer_target_id=menu.outer_target_id,
        query_center_id=menu.query_center_id,
        case_id=menu.case_id,
        action_certificates=certificates,
        model_hash=model.model_hash,
        training_center_ids=model.training_center_ids,
        training_candidate_ids=model.training_candidate_ids,
        excluded_center_ids=model.excluded_center_ids,
        menu_hash=menu.menu_hash,
    )


def _crossfit_certified(
    observations: tuple[SourceActionOutcome, ...],
    effective_menus: tuple[EffectiveMenu, ...] | None,
    *,
    config: FitConfig,
    fixed_excluded: tuple[str, ...],
) -> tuple[CasePrediction, ...]:
    cases = _source_cases(observations, effective_menus, min_centers=3)
    centers = tuple(sorted({case.menu.query_center_id for case in cases}))
    output: list[CasePrediction] = []
    for heldout in centers:
        excluded = frozenset((*fixed_excluded, heldout))
        rows, menus = _filter_surface(observations, effective_menus, excluded_center_ids=excluded)
        model = _fit_certified_model(
            rows,
            menus,
            config=config,
            fixed_excluded=tuple(sorted(excluded)),
        )
        output.extend(
            predict_case(model, case.menu)
            for case in cases
            if case.menu.query_center_id == heldout
        )
    return tuple(sorted(output, key=lambda row: (row.query_center_id, row.case_id)))


def _score_config(
    observations: tuple[SourceActionOutcome, ...],
    effective_menus: tuple[EffectiveMenu, ...] | None,
    config: FitConfig,
    fixed_excluded: tuple[str, ...],
    predictions: tuple[_RawCasePrediction, ...] | None = None,
) -> ConfigTuningScore:
    cases = _source_cases(observations, effective_menus, min_centers=3)
    if predictions is None:
        predictions = _crossfit_raw(
            observations,
            effective_menus,
            config=config,
            fixed_excluded=fixed_excluded,
        )
    residuals = _residual_rows(predictions, cases)
    endpoint_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    epsilon = 1e-12
    for row in residuals:
        center = row.query_center_id
        estimate = row.estimate
        outcome = row.outcome
        harm = float(outcome.bacc_gain < 0.0)
        p = min(max(estimate.predicted_harm_probability, epsilon), 1.0 - epsilon)
        endpoint_values[center]["gain"].append((estimate.predicted_bacc_gain - outcome.bacc_gain) ** 2)
        endpoint_values[center]["harm_brier"].append((p - harm) ** 2)
        endpoint_values[center]["harm_log"].append(-(harm * math.log(p) + (1.0 - harm) * math.log(1.0 - p)))
        endpoint_values[center]["brier"].append((estimate.predicted_brier_delta - outcome.brier_delta) ** 2)
        endpoint_values[center]["log"].append((estimate.predicted_log_delta - outcome.log_delta) ** 2)
    centers = tuple(sorted(endpoint_values))

    def center_equal(name: str) -> float:
        means = [
            sum(endpoint_values[center][name]) / len(endpoint_values[center][name])
            for center in centers
            if endpoint_values[center][name]
        ]
        return sum(means) / len(means)

    gain = center_equal("gain")
    harm_brier = center_equal("harm_brier")
    harm_log = center_equal("harm_log")
    brier = center_equal("brier")
    log_delta = center_equal("log")
    return ConfigTuningScore(
        config,
        gain,
        harm_brier,
        harm_log,
        brier,
        log_delta,
        gain + harm_brier + harm_log + brier + log_delta,
        len(residuals),
    )


def _select_config(
    observations: tuple[SourceActionOutcome, ...],
    effective_menus: tuple[EffectiveMenu, ...] | None,
    grid: tuple[FitConfig, ...],
    *,
    fold_id: str,
    fixed_excluded: tuple[str, ...],
) -> tuple[ConfigSelection, tuple[_RawCasePrediction, ...]]:
    cases = _source_cases(observations, effective_menus, min_centers=3)
    prediction_cache: dict[str, tuple[_RawCasePrediction, ...]] = {}
    scores_list: list[ConfigTuningScore] = []
    for config in grid:
        predictions = _crossfit_raw(
            observations,
            effective_menus,
            config=config,
            fixed_excluded=fixed_excluded,
        )
        prediction_cache[canonical_hash(config)] = predictions
        scores_list.append(
            _score_config(
                observations,
                effective_menus,
                config,
                fixed_excluded,
                predictions,
            )
        )
    scores = tuple(scores_list)
    selected = min(
        scores,
        key=lambda row: (
            row.combined_loss,
            row.harm_brier,
            row.harm_log_loss,
            row.gain_mse,
            canonical_hash(row.config),
        ),
    )
    selection = ConfigSelection(
        fold_id,
        tuple(sorted({case.menu.query_center_id for case in cases})),
        selected.config,
        scores,
    )
    return selection, prediction_cache[canonical_hash(selected.config)]


def fit_source_lodo(
    observations: Sequence[SourceActionOutcome],
    *,
    config: FitConfig = FitConfig(),
    effective_menus: Sequence[EffectiveMenu] | None = None,
    config_grid: Sequence[FitConfig] | None = None,
) -> SourceLODOResult:
    """Nested source-center LODO with query-and-candidate exclusion."""

    rows = tuple(observations)
    menus = None if effective_menus is None else tuple(effective_menus)
    cases = _source_cases(rows, menus, min_centers=4)
    outer = cases[0].menu.outer_target_id
    centers = tuple(sorted({case.menu.query_center_id for case in cases}))
    grid = tuple(config_grid) if config_grid is not None else (config,)
    if not grid or any(not isinstance(value, FitConfig) for value in grid) or len({canonical_hash(value) for value in grid}) != len(grid):
        raise ProtocolError("HARP v8 nested config grid is empty or ambiguous.")
    oof: list[CasePrediction] = []
    hashes: list[tuple[str, str]] = []
    selections: list[ConfigSelection] = []
    nested: list[NestedPolicyFold] = []
    for heldout in centers:
        excluded = frozenset((outer, heldout))
        training_rows, training_menus = _filter_surface(rows, menus, excluded_center_ids=excluded)
        fixed_excluded = tuple(sorted(excluded))
        selection, selected_raw_oof = _select_config(
            training_rows,
            training_menus,
            grid,
            fold_id=f"HELD_SOURCE::{heldout}",
            fixed_excluded=fixed_excluded,
        )
        model = _fit_certified_model(
            training_rows,
            training_menus,
            config=selection.selected_config,
            fixed_excluded=fixed_excluded,
            raw_oof=selected_raw_oof,
        )
        heldout_predictions = tuple(
            predict_case(model, case.menu)
            for case in cases
            if case.menu.query_center_id == heldout
        )
        training_predictions = _crossfit_certified(
            training_rows,
            training_menus,
            config=selection.selected_config,
            fixed_excluded=fixed_excluded,
        )
        oof.extend(heldout_predictions)
        hashes.append((heldout, model.model_hash))
        selections.append(selection)
        nested.append(
            NestedPolicyFold(
                heldout,
                tuple(center for center in centers if center != heldout),
                selection.selected_config,
                training_predictions,
                heldout_predictions,
            )
        )
    final_selection, final_raw_oof = _select_config(
        rows,
        menus,
        grid,
        fold_id="FINAL_SOURCE_MODEL",
        fixed_excluded=(outer,),
    )
    final_model = _fit_certified_model(
        rows,
        menus,
        config=final_selection.selected_config,
        fixed_excluded=(outer,),
        raw_oof=final_raw_oof,
    )
    selections.append(final_selection)
    return SourceLODOResult(
        outer,
        final_model,
        tuple(sorted(oof, key=lambda row: (row.query_center_id, row.case_id))),
        tuple(hashes),
        tuple(selections),
        tuple(nested),
    )


def predict_target_actions(
    model: BaselineInclusiveRouterModel, actions: Sequence[LabelFreeAction]
) -> tuple[EffectiveMenu, CasePrediction]:
    menu = build_effective_menu(actions)
    if menu.query_center_id != menu.outer_target_id:
        raise ProtocolError("HARP v8 target action prediction requires q == outer H.")
    return menu, predict_case(model, menu)


__all__ = (
    "ActionHeads",
    "BaselineInclusiveRouterModel",
    "ConfigSelection",
    "ConfigTuningScore",
    "FitConfig",
    "LinearHead",
    "NestedPolicyFold",
    "SourceLODOResult",
    "Standardizer",
    "fit_baseline_inclusive_router",
    "fit_source_lodo",
    "predict_case",
    "predict_target_actions",
)
