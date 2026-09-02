"""Case-opportunity and conditional directional rank models.

The implementation is intentionally a small vectorized CPU workload.  It
performs no nested multiprocessing, leaves BLAS thread control to the runner,
and reuses each design matrix inside a fold.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    ActionScore,
    CasePrediction,
    Direction,
    LabelFreeAction,
    SourceActionOutcome,
)
from .effective_menu import EffectiveMenu, build_effective_menu
from .hashing import canonical_hash


@dataclass(frozen=True, slots=True)
class FitConfig:
    opportunity_alpha: float = 1.0
    rank_alpha: float = 1.0
    min_opportunity_gain: float = 0.0
    max_irls_iterations: int = 32

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.opportunity_alpha)
            or self.opportunity_alpha <= 0.0
            or not math.isfinite(self.rank_alpha)
            or self.rank_alpha <= 0.0
            or not math.isfinite(self.min_opportunity_gain)
            or int(self.max_irls_iterations) < 1
        ):
            raise ProtocolError("Source-active fit configuration is malformed.")


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
            raise ProtocolError("Source-active feature standardizer is malformed.")


@dataclass(frozen=True, slots=True)
class LinearHead:
    intercept: float
    coefficients: tuple[float, ...]
    available: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.intercept) or any(
            not math.isfinite(value) for value in self.coefficients
        ):
            raise ProtocolError("Source-active linear head contains non-finite values.")


@dataclass(frozen=True, slots=True)
class SourceActiveRouterModel:
    outer_target_id: str
    training_center_ids: tuple[str, ...]
    action_standardizer: Standardizer
    case_standardizer: Standardizer
    opportunity_head: LinearHead
    d01_rank_head: LinearHead
    d10_rank_head: LinearHead
    fit_config: FitConfig
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        training = tuple(sorted(self.training_center_ids))
        if (
            not training
            or len(training) != len(set(training))
            or self.outer_target_id in training
            or len(self.action_standardizer.names) != len(self.d01_rank_head.coefficients)
            or len(self.action_standardizer.names) != len(self.d10_rank_head.coefficients)
            or len(self.case_standardizer.names) != len(self.opportunity_head.coefficients)
        ):
            raise ProtocolError("Source-active router model roles or dimensions are malformed.")
        object.__setattr__(self, "training_center_ids", training)
        object.__setattr__(
            self,
            "model_hash",
            canonical_hash(
                {
                    "schema_version": "source_active_router_model_v7",
                    "outer_target_id": self.outer_target_id,
                    "training_center_ids": training,
                    "action_standardizer": self.action_standardizer,
                    "case_standardizer": self.case_standardizer,
                    "opportunity_head": self.opportunity_head,
                    "d01_rank_head": self.d01_rank_head,
                    "d10_rank_head": self.d10_rank_head,
                    "fit_config": self.fit_config,
                    "target_evaluation_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ConfigTuningScore:
    config: FitConfig
    opportunity_brier: float
    opportunity_log_loss: float
    conditional_tie_rank_loss: float
    combined_loss: float
    evaluated_case_count: int
    score_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            any(
                not math.isfinite(value) or value < 0.0
                for value in (
                    self.opportunity_brier,
                    self.opportunity_log_loss,
                    self.conditional_tie_rank_loss,
                    self.combined_loss,
                )
            )
            or self.evaluated_case_count < 1
        ):
            raise ProtocolError("Source-active tuning score is malformed.")
        object.__setattr__(
            self,
            "score_hash",
            canonical_hash(
                {
                    "schema_version": "source_active_config_tuning_score_v7",
                    "config": self.config,
                    "opportunity_brier": self.opportunity_brier,
                    "opportunity_log_loss": self.opportunity_log_loss,
                    "conditional_tie_rank_loss": self.conditional_tie_rank_loss,
                    "combined_loss": self.combined_loss,
                    "evaluated_case_count": self.evaluated_case_count,
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
            raise ProtocolError("Source-active config selection is incomplete.")
        if self.selected_config not in {score.config for score in self.scores}:
            raise ProtocolError("Selected config is absent from its tuning surface.")
        object.__setattr__(
            self,
            "selection_hash",
            canonical_hash(
                {
                    "schema_version": "source_active_config_selection_v7",
                    "fold_id": self.fold_id,
                    "training_center_ids": self.training_center_ids,
                    "selected_config": self.selected_config,
                    "score_hashes": tuple(score.score_hash for score in self.scores),
                    "target_evaluation_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class NestedPolicyFold:
    """Inner-OOF predictions built without the outer held source center."""

    heldout_center_id: str
    training_center_ids: tuple[str, ...]
    selected_config: FitConfig
    predictions: tuple[CasePrediction, ...]
    fold_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.heldout_center_id in self.training_center_ids
            or not self.training_center_ids
            or not self.predictions
        ):
            raise ProtocolError("Nested policy fold roles are malformed.")
        expected_queries = set(self.training_center_ids)
        observed_queries = {row.query_center_id for row in self.predictions}
        if observed_queries != expected_queries or any(
            row.query_center_id in row.training_center_ids
            or self.heldout_center_id in row.training_center_ids
            for row in self.predictions
        ):
            raise ProtocolError("Nested policy predictions crossed held-source roles.")
        object.__setattr__(
            self,
            "fold_hash",
            canonical_hash(
                {
                    "schema_version": "source_active_nested_policy_fold_v7",
                    "heldout_center_id": self.heldout_center_id,
                    "training_center_ids": self.training_center_ids,
                    "selected_config": self.selected_config,
                    "prediction_hashes": tuple(
                        row.prediction_hash for row in self.predictions
                    ),
                    "heldout_center_excluded_from_all_fits": True,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceLODOResult:
    outer_target_id: str
    final_model: SourceActiveRouterModel
    oof_predictions: tuple[CasePrediction, ...]
    heldout_model_hashes: tuple[tuple[str, str], ...]
    config_selections: tuple[ConfigSelection, ...]
    nested_policy_folds: tuple[NestedPolicyFold, ...]
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.final_model.outer_target_id != self.outer_target_id:
            raise ProtocolError("Final source model crossed outer target.")
        seen: set[tuple[str, str]] = set()
        for row in self.oof_predictions:
            key = (row.query_center_id, row.case_id)
            if key in seen or row.outer_target_id != self.outer_target_id:
                raise ProtocolError("OOF prediction inventory is malformed.")
            if row.query_center_id in row.training_center_ids:
                raise ProtocolError("Source-center LODO prediction trained on its heldout center.")
            seen.add(key)
        centers = tuple(sorted({row.query_center_id for row in self.oof_predictions}))
        expected_folds = {f"HELD_SOURCE::{center}" for center in centers}
        held_selections = tuple(
            row for row in self.config_selections if row.fold_id != "FINAL_SOURCE_MODEL"
        )
        final_selections = tuple(
            row for row in self.config_selections if row.fold_id == "FINAL_SOURCE_MODEL"
        )
        if (
            {row.fold_id for row in held_selections} != expected_folds
            or len(final_selections) != 1
            or final_selections[0].selected_config != self.final_model.fit_config
            or {row.heldout_center_id for row in self.nested_policy_folds} != set(centers)
        ):
            raise ProtocolError("Source LODO tuning or nested-policy inventory is incomplete.")
        for selection in held_selections:
            heldout = selection.fold_id.split("::", 1)[1]
            if heldout in selection.training_center_ids:
                raise ProtocolError("Held source entered its config-selection training surface.")
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "source_active_source_lodo_v7",
                    "outer_target_id": self.outer_target_id,
                    "final_model_hash": self.final_model.model_hash,
                    "oof_prediction_hashes": tuple(
                        row.prediction_hash for row in self.oof_predictions
                    ),
                    "heldout_model_hashes": self.heldout_model_hashes,
                    "config_selection_hashes": tuple(
                        row.selection_hash for row in self.config_selections
                    ),
                    "nested_policy_fold_hashes": tuple(
                        row.fold_hash for row in self.nested_policy_folds
                    ),
                    "outer_H_excluded": True,
                    "source_center_lodo": True,
                }
            ),
        )

    def numeric_oof_payload(self) -> dict[str, object]:
        """Return JSON-native numeric OOF values suitable for durable storage."""

        return {
            "schema_version": "source_active_numeric_oof_v7",
            "outer_target_id": self.outer_target_id,
            "result_hash": self.result_hash,
            "config_selections": [
                {
                    "fold_id": selection.fold_id,
                    "training_center_ids": list(selection.training_center_ids),
                    "selected_config": {
                        "opportunity_alpha": selection.selected_config.opportunity_alpha,
                        "rank_alpha": selection.selected_config.rank_alpha,
                        "min_opportunity_gain": selection.selected_config.min_opportunity_gain,
                        "max_irls_iterations": selection.selected_config.max_irls_iterations,
                    },
                    "selection_hash": selection.selection_hash,
                    "scores": [
                        {
                            "config": {
                                "opportunity_alpha": score.config.opportunity_alpha,
                                "rank_alpha": score.config.rank_alpha,
                                "min_opportunity_gain": score.config.min_opportunity_gain,
                                "max_irls_iterations": score.config.max_irls_iterations,
                            },
                            "opportunity_brier": score.opportunity_brier,
                            "opportunity_log_loss": score.opportunity_log_loss,
                            "conditional_tie_rank_loss": score.conditional_tie_rank_loss,
                            "combined_loss": score.combined_loss,
                            "evaluated_case_count": score.evaluated_case_count,
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
                    "rows": [
                        {
                            "query_center_id": row.query_center_id,
                            "case_id": row.case_id,
                            "opportunity_probability": row.opportunity_probability,
                            "rank_margin": row.rank_margin,
                            "top_action_id": row.top_action_id,
                            "action_scores": [
                                {
                                    "action_id": score.action_id,
                                    "direction": score.direction.value,
                                    "score": score.score,
                                }
                                for score in row.action_scores
                            ],
                            "model_hash": row.model_hash,
                            "menu_hash": row.menu_hash,
                            "prediction_hash": row.prediction_hash,
                        }
                        for row in fold.predictions
                    ],
                }
                for fold in self.nested_policy_folds
            ],
            "rows": [
                {
                    "query_center_id": row.query_center_id,
                    "case_id": row.case_id,
                    "opportunity_probability": row.opportunity_probability,
                    "rank_margin": row.rank_margin,
                    "top_action_id": row.top_action_id,
                    "action_scores": [
                        {
                            "action_id": score.action_id,
                            "direction": score.direction.value,
                            "score": score.score,
                        }
                        for score in row.action_scores
                    ],
                    "model_hash": row.model_hash,
                    "menu_hash": row.menu_hash,
                    "prediction_hash": row.prediction_hash,
                }
                for row in self.oof_predictions
            ],
        }


@dataclass(frozen=True, slots=True)
class _SourceCase:
    menu: EffectiveMenu
    outcomes: tuple[SourceActionOutcome, ...]

    @property
    def opportunity_gain(self) -> float:
        return max((row.bacc_gain for row in self.outcomes), default=0.0)


def _source_cases(
    observations: Sequence[SourceActionOutcome],
    effective_menus: Sequence[EffectiveMenu] | None = None,
) -> tuple[_SourceCase, ...]:
    rows = tuple(observations)
    if any(not isinstance(row, SourceActionOutcome) for row in rows):
        raise ProtocolError("Source-active fitting requires source-development outcomes.")
    if effective_menus is not None:
        menus = tuple(effective_menus)
        if not menus or any(not isinstance(menu, EffectiveMenu) for menu in menus):
            raise ProtocolError("Source-active fitting requires typed effective menus.")
        keys = tuple((menu.query_center_id, menu.case_id) for menu in menus)
        if len(set(keys)) != len(keys):
            raise ProtocolError("Effective-menu inventory contains duplicate source cases.")
        outer_ids = {menu.outer_target_id for menu in menus}
        schemas = {menu.feature_names for menu in menus}
        if len(outer_ids) != 1 or len(schemas) != 1:
            raise ProtocolError("Effective-menu inventory crossed outer target or feature schema.")
        outer = next(iter(outer_ids))
        if any(menu.query_center_id == outer for menu in menus):
            raise ProtocolError("Outer target entered source effective-menu inventory.")
        by_key = {(menu.query_center_id, menu.case_id): menu for menu in menus}
        outcomes_by_key: dict[tuple[str, str], list[SourceActionOutcome]] = defaultdict(list)
        for row in rows:
            key = (row.action.query_center_id, row.action.case_id)
            menu = by_key.get(key)
            if (
                menu is None
                or row.action.outer_target_id != outer
                or not any(
                    action.action_id == row.action.action_id
                    and action.action_hash == row.action.action_hash
                    for action in menu.actions
                )
            ):
                raise ProtocolError("Source outcome is absent from its sealed effective menu.")
            outcomes_by_key[key].append(row)
        cases: list[_SourceCase] = []
        for key in sorted(by_key):
            menu = by_key[key]
            outcomes = tuple(
                sorted(outcomes_by_key.get(key, ()), key=lambda row: row.action.action_id)
            )
            if {row.action.action_id for row in outcomes} != {
                action.action_id for action in menu.actions
            }:
                raise ProtocolError("Sealed effective menu and source outcomes are not complete.")
            cases.append(_SourceCase(menu=menu, outcomes=outcomes))
        if len({case.menu.query_center_id for case in cases}) < 2:
            raise ProtocolError("Source-active fitting requires at least two source centers.")
        return tuple(cases)
    if not rows:
        raise ProtocolError("Source-active fitting requires source-development outcomes.")
    outer_ids = {row.action.outer_target_id for row in rows}
    schemas = {row.action.feature_names for row in rows}
    if len(outer_ids) != 1 or len(schemas) != 1:
        raise ProtocolError("Source-active fitting crossed outer target or feature schema.")
    outer = next(iter(outer_ids))
    if any(row.action.query_center_id == outer for row in rows):
        raise ProtocolError("Outer target entered source-development fitting.")
    grouped: dict[tuple[str, str], list[SourceActionOutcome]] = defaultdict(list)
    for row in rows:
        grouped[(row.action.query_center_id, row.action.case_id)].append(row)
    cases: list[_SourceCase] = []
    for key in sorted(grouped):
        members = grouped[key]
        if len({row.action.action_id for row in members}) != len(members):
            raise ProtocolError("Source case contains duplicate action outcomes.")
        menu = build_effective_menu(tuple(row.action for row in members))
        by_id = {row.action.action_id: row for row in members}
        # Byte-identical actions must have byte-identical realized endpoints.
        for duplicate, representative in menu.duplicate_representatives:
            left, right = by_id[duplicate], by_id[representative]
            if (left.bacc_gain, left.brier_delta, left.log_delta) != (
                right.bacc_gain,
                right.brier_delta,
                right.log_delta,
            ):
                raise ProtocolError("Duplicate physical actions have inconsistent outcomes.")
        retained = tuple(by_id[action.action_id] for action in menu.actions)
        cases.append(_SourceCase(menu=menu, outcomes=retained))
    if len({case.menu.query_center_id for case in cases}) < 2:
        raise ProtocolError("Source-active fitting requires at least two source centers.")
    return tuple(cases)


def _case_feature_names(action_names: tuple[str, ...]) -> tuple[str, ...]:
    return (
        *(f"mean::{name}" for name in action_names),
        *(f"max::{name}" for name in action_names),
        *(f"min::{name}" for name in action_names),
        "active_count",
        "d01_fraction",
        "d10_fraction",
    )


def _case_vector(menu: EffectiveMenu) -> np.ndarray:
    width = len(menu.feature_names)
    if menu.actions:
        matrix = np.asarray([row.feature_values for row in menu.actions], dtype=np.float64)
        mean = np.mean(matrix, axis=0, dtype=np.float64)
        maximum = np.max(matrix, axis=0)
        minimum = np.min(matrix, axis=0)
        count = float(len(menu.actions))
        d01 = sum(row.direction is Direction.D01 for row in menu.actions) / count
        d10 = 1.0 - d01
    else:
        mean = maximum = minimum = np.zeros(width, dtype=np.float64)
        count = d01 = d10 = 0.0
    return np.concatenate((mean, maximum, minimum, np.asarray((count, d01, d10))))


def _balanced_case_weights(cases: Sequence[_SourceCase]) -> np.ndarray:
    per_center: dict[str, int] = defaultdict(int)
    for case in cases:
        per_center[case.menu.query_center_id] += 1
    center_count = len(per_center)
    return np.asarray(
        [1.0 / (center_count * per_center[case.menu.query_center_id]) for case in cases],
        dtype=np.float64,
    )


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


def _ridge_head(
    matrix: np.ndarray, response: np.ndarray, weights: np.ndarray, *, alpha: float
) -> LinearHead:
    design = np.column_stack((np.ones(matrix.shape[0], dtype=np.float64), matrix))
    normal = design.T @ (weights[:, None] * design)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    rhs = design.T @ (weights * response)
    coefficients = np.linalg.lstsq(normal + penalty, rhs, rcond=None)[0]
    return LinearHead(
        intercept=float(coefficients[0]),
        coefficients=tuple(float(value) for value in coefficients[1:]),
        available=True,
    )


def _logistic_head(
    matrix: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    max_iterations: int,
) -> LinearHead:
    if np.all(response == response[0]):
        # Jeffreys-style finite smoothing for a degenerate development fold.
        probability = (float(np.sum(response)) + 0.5) / (len(response) + 1.0)
        return LinearHead(
            intercept=math.log(probability / (1.0 - probability)),
            coefficients=tuple(0.0 for _ in range(matrix.shape[1])),
            available=True,
        )
    design = np.column_stack((np.ones(matrix.shape[0], dtype=np.float64), matrix))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    for _ in range(max_iterations):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        curvature = np.maximum(probability * (1.0 - probability), 1e-6)
        adjusted = linear + (response - probability) / curvature
        combined = weights * curvature
        normal = design.T @ (combined[:, None] * design) + penalty
        rhs = design.T @ (combined * adjusted)
        updated = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        if float(np.max(np.abs(updated - coefficients))) <= 1e-10:
            coefficients = updated
            break
        coefficients = updated
    return LinearHead(
        intercept=float(coefficients[0]),
        coefficients=tuple(float(value) for value in coefficients[1:]),
        available=True,
    )


def fit_source_active_router(
    observations: Sequence[SourceActionOutcome],
    *,
    config: FitConfig = FitConfig(),
    effective_menus: Sequence[EffectiveMenu] | None = None,
) -> SourceActiveRouterModel:
    """Fit on source cases only; target-scope outcomes are rejected by contract."""

    cases = _source_cases(observations, effective_menus)
    outer = cases[0].menu.outer_target_id
    centers = tuple(sorted({case.menu.query_center_id for case in cases}))
    action_names = cases[0].menu.feature_names
    case_names = _case_feature_names(action_names)

    case_matrix = np.asarray([_case_vector(case.menu) for case in cases], dtype=np.float64)
    case_weights = _balanced_case_weights(cases)
    case_standardizer, standardized_cases = _standardize_fit(
        case_names, case_matrix, case_weights
    )
    opportunity = np.asarray(
        [case.opportunity_gain > config.min_opportunity_gain for case in cases],
        dtype=np.float64,
    )
    opportunity_head = _logistic_head(
        standardized_cases,
        opportunity,
        case_weights,
        alpha=config.opportunity_alpha,
        max_iterations=config.max_irls_iterations,
    )

    positive_cases = [case for case in cases if case.opportunity_gain > config.min_opportunity_gain]
    action_rows: list[tuple[_SourceCase, SourceActionOutcome]] = [
        (case, outcome) for case in positive_cases for outcome in case.outcomes
    ]
    heads: dict[Direction, LinearHead] = {}
    if not action_rows:
        # A nested training fold may legitimately contain no positive
        # opportunity.  Preserve the opportunity model and emit an explicit
        # unavailable conditional ranker; downstream policy then uses exact B
        # for this fold instead of aborting the whole experiment.
        action_standardizer = Standardizer(
            names=action_names,
            mean=tuple(0.0 for _ in action_names),
            scale=tuple(1.0 for _ in action_names),
        )
        for direction in Direction:
            heads[direction] = LinearHead(
                intercept=0.0,
                coefficients=tuple(0.0 for _ in action_names),
                available=False,
            )
    else:
        action_matrix = np.asarray(
            [row.action.feature_values for _, row in action_rows], dtype=np.float64
        )
        actions_per_case = {
            (case.menu.query_center_id, case.menu.case_id): len(case.outcomes)
            for case in positive_cases
        }
        cases_per_center: dict[str, int] = defaultdict(int)
        for case in positive_cases:
            cases_per_center[case.menu.query_center_id] += 1
        center_count = len(cases_per_center)
        action_weights = np.asarray(
            [
                1.0
                / (
                    center_count
                    * cases_per_center[case.menu.query_center_id]
                    * actions_per_case[(case.menu.query_center_id, case.menu.case_id)]
                )
                for case, _ in action_rows
            ],
            dtype=np.float64,
        )
        action_standardizer, standardized_actions = _standardize_fit(
            action_names, action_matrix, action_weights
        )
        responses = np.asarray(
            [row.bacc_gain for _, row in action_rows], dtype=np.float64
        )
        for direction in Direction:
            indices = np.asarray(
                [row.action.direction is direction for _, row in action_rows],
                dtype=bool,
            )
            if not np.any(indices):
                heads[direction] = LinearHead(
                    intercept=0.0,
                    coefficients=tuple(0.0 for _ in action_names),
                    available=False,
                )
            else:
                heads[direction] = _ridge_head(
                    standardized_actions[indices],
                    responses[indices],
                    action_weights[indices],
                    alpha=config.rank_alpha,
                )
    return SourceActiveRouterModel(
        outer_target_id=outer,
        training_center_ids=centers,
        action_standardizer=action_standardizer,
        case_standardizer=case_standardizer,
        opportunity_head=opportunity_head,
        d01_rank_head=heads[Direction.D01],
        d10_rank_head=heads[Direction.D10],
        fit_config=config,
    )


def _linear(head: LinearHead, vector: np.ndarray) -> float:
    return float(head.intercept + np.dot(np.asarray(head.coefficients), vector))


def predict_case(model: SourceActiveRouterModel, menu: EffectiveMenu) -> CasePrediction:
    """Predict opportunity and conditional ranks without accepting outcomes."""

    if model.outer_target_id != menu.outer_target_id:
        raise ProtocolError("Target menu crossed the model's outer target.")
    if menu.query_center_id in model.training_center_ids:
        raise ProtocolError("Prediction query center was used to fit this LODO model.")
    if menu.feature_names != model.action_standardizer.names:
        raise ProtocolError("Target menu feature schema drifted from the source model.")
    case_vector = _apply_standardizer(_case_vector(menu), model.case_standardizer)
    opportunity_linear = float(np.clip(_linear(model.opportunity_head, case_vector), -30.0, 30.0))
    opportunity = 1.0 / (1.0 + math.exp(-opportunity_linear))
    scores: list[ActionScore] = []
    for action in menu.actions:
        vector = _apply_standardizer(
            np.asarray(action.feature_values, dtype=np.float64), model.action_standardizer
        )
        head = (
            model.d01_rank_head
            if action.direction is Direction.D01
            else model.d10_rank_head
        )
        if head.available:
            scores.append(
                ActionScore(
                    action_id=action.action_id,
                    action_hash=action.action_hash,
                    direction=action.direction,
                    score=_linear(head, vector),
                )
            )
    return CasePrediction(
        outer_target_id=menu.outer_target_id,
        query_center_id=menu.query_center_id,
        case_id=menu.case_id,
        opportunity_probability=opportunity if menu.actions else 0.0,
        action_scores=tuple(scores),
        model_hash=model.model_hash,
        training_center_ids=model.training_center_ids,
        menu_hash=menu.menu_hash,
    )


def _nested_config_score(
    observations: tuple[SourceActionOutcome, ...],
    effective_menus: tuple[EffectiveMenu, ...] | None,
    config: FitConfig,
    predictions: tuple[CasePrediction, ...],
) -> ConfigTuningScore:
    """Evaluate one predeclared config by an inner source-center LODO."""

    cases = _source_cases(observations, effective_menus)
    centers = tuple(sorted({case.menu.query_center_id for case in cases}))
    if len(centers) < 3:
        raise ProtocolError("Nested config selection requires at least three source centers.")
    brier_by_center: dict[str, list[float]] = defaultdict(list)
    log_by_center: dict[str, list[float]] = defaultdict(list)
    rank_by_center: dict[str, list[float]] = defaultdict(list)
    by_key = {(row.query_center_id, row.case_id): row for row in predictions}
    expected = {(case.menu.query_center_id, case.menu.case_id) for case in cases}
    if set(by_key) != expected:
        raise ProtocolError("Nested config predictions do not cover their source inventory.")
    for heldout in centers:
        for case in cases:
            if case.menu.query_center_id != heldout:
                continue
            prediction = by_key[(heldout, case.menu.case_id)]
            if heldout in prediction.training_center_ids:
                raise ProtocolError("Inner config-selection fold leaked its heldout center.")
            actual = float(case.opportunity_gain > config.min_opportunity_gain)
            probability = min(
                max(prediction.opportunity_probability, 1e-12), 1.0 - 1e-12
            )
            brier_by_center[heldout].append((probability - actual) ** 2)
            log_by_center[heldout].append(
                -(
                    actual * math.log(probability)
                    + (1.0 - actual) * math.log(1.0 - probability)
                )
            )
            if actual:
                gains = {row.action.action_id: row.bacc_gain for row in case.outcomes}
                best = max(gains.values())
                selected = prediction.top_action_id
                correct = bool(
                    selected is not None
                    and math.isclose(
                        gains[selected], best, rel_tol=0.0, abs_tol=1e-12
                    )
                )
                rank_by_center[heldout].append(0.0 if correct else 1.0)

    def center_equal(mapping: dict[str, list[float]]) -> float:
        center_means = [
            sum(mapping[center]) / len(mapping[center])
            for center in centers
            if mapping.get(center)
        ]
        return sum(center_means) / len(center_means) if center_means else 1.0

    brier = center_equal(brier_by_center)
    log_loss = center_equal(log_by_center)
    rank_loss = center_equal(rank_by_center)
    return ConfigTuningScore(
        config=config,
        opportunity_brier=brier,
        opportunity_log_loss=log_loss,
        conditional_tie_rank_loss=rank_loss,
        combined_loss=brier + log_loss + rank_loss,
        evaluated_case_count=len(cases),
    )


def _select_config(
    observations: tuple[SourceActionOutcome, ...],
    effective_menus: tuple[EffectiveMenu, ...] | None,
    config_grid: tuple[FitConfig, ...],
    *,
    fold_id: str,
) -> tuple[ConfigSelection, tuple[CasePrediction, ...]]:
    cases = _source_cases(observations, effective_menus)
    centers = tuple(sorted({case.menu.query_center_id for case in cases}))
    prediction_cache: dict[str, tuple[CasePrediction, ...]] = {}
    scores_list: list[ConfigTuningScore] = []
    for config in config_grid:
        key = canonical_hash(config)
        predictions = _crossfit_with_config(observations, effective_menus, config)
        prediction_cache[key] = predictions
        scores_list.append(
            _nested_config_score(
                observations, effective_menus, config, predictions
            )
        )
    scores = tuple(scores_list)
    selected = min(
        scores,
        key=lambda row: (
            row.combined_loss,
            row.opportunity_brier,
            row.opportunity_log_loss,
            row.conditional_tie_rank_loss,
            row.config.opportunity_alpha,
            row.config.rank_alpha,
            row.config.min_opportunity_gain,
            row.config.max_irls_iterations,
        ),
    )
    selection = ConfigSelection(
        fold_id=fold_id,
        training_center_ids=centers,
        selected_config=selected.config,
        scores=scores,
    )
    return selection, prediction_cache[canonical_hash(selected.config)]


def _crossfit_with_config(
    observations: tuple[SourceActionOutcome, ...],
    effective_menus: tuple[EffectiveMenu, ...] | None,
    config: FitConfig,
) -> tuple[CasePrediction, ...]:
    cases = _source_cases(observations, effective_menus)
    centers = tuple(sorted({case.menu.query_center_id for case in cases}))
    predictions: list[CasePrediction] = []
    for heldout in centers:
        training_rows = tuple(
            row for row in observations if row.action.query_center_id != heldout
        )
        training_menus = (
            None
            if effective_menus is None
            else tuple(
                menu for menu in effective_menus if menu.query_center_id != heldout
            )
        )
        model = fit_source_active_router(
            training_rows, config=config, effective_menus=training_menus
        )
        predictions.extend(
            predict_case(model, case.menu)
            for case in cases
            if case.menu.query_center_id == heldout
        )
    return tuple(
        sorted(predictions, key=lambda row: (row.query_center_id, row.case_id))
    )


def fit_source_lodo(
    observations: Sequence[SourceActionOutcome],
    *,
    config: FitConfig = FitConfig(),
    effective_menus: Sequence[EffectiveMenu] | None = None,
    config_grid: Sequence[FitConfig] | None = None,
) -> SourceLODOResult:
    """Cross-fit every source center, then fit one target-facing source model."""

    rows = tuple(observations)
    cases = _source_cases(rows, effective_menus)
    outer = cases[0].menu.outer_target_id
    centers = tuple(sorted({case.menu.query_center_id for case in cases}))
    if len(centers) < 4:
        raise ProtocolError("Nested source-center LODO requires at least four source centers.")
    grid = tuple(config_grid) if config_grid is not None else (config,)
    if (
        not grid
        or any(not isinstance(value, FitConfig) for value in grid)
        or len({canonical_hash(value) for value in grid}) != len(grid)
    ):
        raise ProtocolError("Source-active nested config grid is empty or ambiguous.")
    menu_inventory = None if effective_menus is None else tuple(effective_menus)
    oof: list[CasePrediction] = []
    fold_hashes: list[tuple[str, str]] = []
    selections: list[ConfigSelection] = []
    nested_policy_folds: list[NestedPolicyFold] = []
    for heldout in centers:
        training = tuple(row for row in rows if row.action.query_center_id != heldout)
        training_menus = (
            None
            if menu_inventory is None
            else tuple(menu for menu in menu_inventory if menu.query_center_id != heldout)
        )
        selection, nested_predictions = _select_config(
            training,
            training_menus,
            grid,
            fold_id=f"HELD_SOURCE::{heldout}",
        )
        model = fit_source_active_router(
            training,
            config=selection.selected_config,
            effective_menus=training_menus,
        )
        if heldout in model.training_center_ids or outer in model.training_center_ids:
            raise ProtocolError("Source-center LODO exclusion failed.")
        for case in cases:
            if case.menu.query_center_id == heldout:
                oof.append(predict_case(model, case.menu))
        fold_hashes.append((heldout, model.model_hash))
        selections.append(selection)
        nested_policy_folds.append(
            NestedPolicyFold(
                heldout_center_id=heldout,
                training_center_ids=tuple(
                    center for center in centers if center != heldout
                ),
                selected_config=selection.selected_config,
                predictions=nested_predictions,
            )
        )
    final_selection, _final_inner_predictions = _select_config(
        rows, menu_inventory, grid, fold_id="FINAL_SOURCE_MODEL"
    )
    final_model = fit_source_active_router(
        rows,
        config=final_selection.selected_config,
        effective_menus=menu_inventory,
    )
    selections.append(final_selection)
    return SourceLODOResult(
        outer_target_id=outer,
        final_model=final_model,
        oof_predictions=tuple(
            sorted(oof, key=lambda row: (row.query_center_id, row.case_id))
        ),
        heldout_model_hashes=tuple(fold_hashes),
        config_selections=tuple(selections),
        nested_policy_folds=tuple(nested_policy_folds),
    )


def predict_target_actions(
    model: SourceActiveRouterModel, actions: Sequence[LabelFreeAction]
) -> tuple[EffectiveMenu, CasePrediction]:
    """Runtime adapter: filter/dedup a target case, then score it label-free."""

    menu = build_effective_menu(actions)
    if menu.query_center_id != menu.outer_target_id:
        raise ProtocolError("Target action prediction requires q == outer H.")
    return menu, predict_case(model, menu)


__all__ = (
    "ConfigSelection",
    "ConfigTuningScore",
    "FitConfig",
    "LinearHead",
    "NestedPolicyFold",
    "SourceActiveRouterModel",
    "SourceLODOResult",
    "Standardizer",
    "fit_source_active_router",
    "fit_source_lodo",
    "predict_case",
    "predict_target_actions",
)
