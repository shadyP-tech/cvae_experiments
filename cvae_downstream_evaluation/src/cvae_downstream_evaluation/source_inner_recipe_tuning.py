"""Source-inner LODO selection for real-feature recipe specs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .downstream import balanced_accuracy, macro_f1
from .protocol import ProtocolError
from .real_feature_recipes import (
    RecipeSpec,
    assert_fixed_recipe_random_state,
    fit_recipe,
    recipe_grid_hash,
)
from .schemas.midogpp_real_feature_recipe import (
    SOURCE_INNER_RECIPE_TUNING_COLUMNS,
    SOURCE_INNER_RECIPE_TUNING_SCHEMA_VERSION,
)
from .source_inner_classifier_tuning import SourceInnerClassifierFold


RecipeScoreFn = Callable[[RecipeSpec, SourceInnerClassifierFold], "RecipeFoldScore"]


@dataclass(frozen=True)
class RecipeFoldScore:
    bacc: float
    macro_f1: float
    converged: bool = True
    n_iter: tuple[int, ...] = ()
    status: str = "ok"
    error_message: str = ""
    validation_labels: tuple[int, ...] = ()
    predictions: tuple[int, ...] = ()

    def metric(self, name: str) -> float:
        if name == "bacc":
            return float(self.bacc)
        if name == "macro_f1":
            return float(self.macro_f1)
        raise ProtocolError(f"Unsupported recipe selection metric: {name!r}")


@dataclass(frozen=True)
class RecipeTuningCandidateRow:
    experiment_seed: int
    classifier_seed: int
    outer_target_center: str
    selector_centers: tuple[str, ...]
    inner_lodo_centers: tuple[str, ...]
    recipe_spec: RecipeSpec
    grid_hash: str
    source_inner_lodo_center_bacc_vector: Mapping[str, float]
    source_inner_lodo_center_macro_f1_vector: Mapping[str, float]
    selection_metric: str
    aggregate_score: float
    min_score: float
    se_score: float
    within_one_se_best: bool
    selected_by_source_inner_lodo: bool
    tie_breaker: tuple[object, ...]
    convergence_by_center: Mapping[str, bool]
    n_iter_by_center: Mapping[str, tuple[int, ...]]
    status_by_center: Mapping[str, str]
    row_role: str = "selection_candidate"

    @property
    def selected_recipe_config_hash(self) -> str:
        return self.recipe_spec.config_hash

    def to_artifact_row(self) -> dict[str, object]:
        return {
            "schema_version": SOURCE_INNER_RECIPE_TUNING_SCHEMA_VERSION,
            "experiment_seed": int(self.experiment_seed),
            "classifier_seed": int(self.classifier_seed),
            "outer_target_center": self.outer_target_center,
            "selector_centers": json.dumps(list(self.selector_centers)),
            "inner_lodo_centers": json.dumps(list(self.inner_lodo_centers)),
            "recipe_grid_hash": self.grid_hash,
            "recipe_spec": json.dumps(self.recipe_spec.to_payload(), sort_keys=True),
            "selected_recipe_config_hash": self.selected_recipe_config_hash,
            "recipe_family": self.recipe_spec.family,
            "preprocessing_id": self.recipe_spec.preprocessing_id,
            "source_inner_lodo_center_bacc_vector": json.dumps(
                dict(self.source_inner_lodo_center_bacc_vector), sort_keys=True
            ),
            "source_inner_lodo_center_macro_f1_vector": json.dumps(
                dict(self.source_inner_lodo_center_macro_f1_vector), sort_keys=True
            ),
            "selection_metric": self.selection_metric,
            "selection_source": "source_inner_lodo",
            "selected_by_source_inner_lodo": str(bool(self.selected_by_source_inner_lodo)).lower(),
            "aggregate_selection_score": self.aggregate_score,
            "min_selection_score": self.min_score,
            "se_selection_score": self.se_score,
            "within_one_se_best": str(bool(self.within_one_se_best)).lower(),
            "tie_breaker": json.dumps(list(self.tie_breaker), sort_keys=True),
            "convergence_by_center": json.dumps(dict(self.convergence_by_center), sort_keys=True),
            "n_iter_by_center": json.dumps(
                {key: list(value) for key, value in self.n_iter_by_center.items()},
                sort_keys=True,
            ),
            "status_by_center": json.dumps(dict(self.status_by_center), sort_keys=True),
            "row_role": self.row_role,
            "protocol_hash": "",
            "feature_cache_path": "",
            "feature_cache_hash": "",
            "manifest_path": "",
            "manifest_hash": "",
            "source_inner_fold_audit": "{}",
            "fit_scope_policy": "source_inner_lodo_fit_train_centers_only",
            "preprocessing_fit_scope": "source_inner_train_centers_only",
            "model_fit_scope": "source_inner_train_centers_only",
            "decision_rule": self.recipe_spec.model.decision_rule,
            "selection_used_target_labels": "false",
            "fit_used_target_center": "false",
            "target_eval_labels_used_for_scoring": "false",
            "generated_embeddings_used": "false",
            "cvae_checkpoint_used": "false",
            "source_summary_manifest_used": "false",
            "is_router": "false",
            "claim_scope": "real_feature_transfer_only",
            "probabilities_calibrated": "false",
        }


@dataclass(frozen=True)
class SourceInnerRecipeSelectionResult:
    experiment_seed: int
    classifier_seed: int
    outer_target_center: str
    selected_recipe: RecipeSpec
    selected_config_hash: str
    grid_hash: str
    selection_metric: str
    selector_centers: tuple[str, ...]
    inner_lodo_centers: tuple[str, ...]
    rows: tuple[RecipeTuningCandidateRow, ...]

    def to_artifact_rows(self) -> list[dict[str, object]]:
        return [row.to_artifact_row() for row in self.rows]


def select_recipe_source_inner_lodo(
    *,
    outer_target_center: str,
    folds: Sequence[SourceInnerClassifierFold],
    candidate_recipes: Sequence[RecipeSpec],
    experiment_seed: int = 0,
    classifier_seed: int = 23,
    selection_metric: str = "bacc",
    score_fn: RecipeScoreFn | None = None,
    row_role: str = "selection_candidate",
) -> SourceInnerRecipeSelectionResult:
    """Select a full recipe using source-inner folds only."""

    _validate_selection_inputs(
        outer_target_center=outer_target_center,
        folds=folds,
        candidate_recipes=candidate_recipes,
        selection_metric=selection_metric,
    )
    scorer = score_fn or _fit_and_score_fold
    grid_hash = recipe_grid_hash(candidate_recipes)
    inner_centers = tuple(fold.pseudo_target_center for fold in folds)
    selector_centers = tuple(sorted({center for fold in folds for center in (*fold.train_centers, fold.pseudo_target_center)}))
    rows: list[RecipeTuningCandidateRow] = []
    for recipe in candidate_recipes:
        fold_scores = {fold.pseudo_target_center: scorer(recipe, fold) for fold in folds}
        metric_scores = {
            center: score.metric(selection_metric)
            for center, score in fold_scores.items()
            if score.status == "ok" and score.converged
        }
        if len(metric_scores) == len(folds):
            values = list(metric_scores.values())
            aggregate = sum(values) / float(len(values))
            min_score = min(values)
            se_score = _standard_error(values)
        else:
            aggregate = math.nan
            min_score = math.nan
            se_score = math.nan
        rows.append(
            RecipeTuningCandidateRow(
                experiment_seed=int(experiment_seed),
                classifier_seed=int(classifier_seed),
                outer_target_center=str(outer_target_center),
                selector_centers=selector_centers,
                inner_lodo_centers=inner_centers,
                recipe_spec=recipe,
                grid_hash=grid_hash,
                source_inner_lodo_center_bacc_vector={center: fold_scores[center].bacc for center in inner_centers},
                source_inner_lodo_center_macro_f1_vector={
                    center: fold_scores[center].macro_f1 for center in inner_centers
                },
                selection_metric=selection_metric,
                aggregate_score=aggregate,
                min_score=min_score,
                se_score=se_score,
                within_one_se_best=False,
                selected_by_source_inner_lodo=False,
                tie_breaker=recipe.tie_break_key(),
                convergence_by_center={center: fold_scores[center].converged for center in inner_centers},
                n_iter_by_center={center: fold_scores[center].n_iter for center in inner_centers},
                status_by_center={center: fold_scores[center].status for center in inner_centers},
                row_role=row_role,
            )
        )
    eligible = [row for row in rows if not math.isnan(float(row.aggregate_score))]
    if not eligible:
        raise ProtocolError("No recipe produced valid source-inner LODO scores.")
    best_mean = max(float(row.aggregate_score) for row in eligible)
    best_rows = [row for row in eligible if float(row.aggregate_score) == best_mean]
    best_se = min(float(row.se_score) for row in best_rows)
    one_se_floor = best_mean - best_se
    within_one_se = [row for row in eligible if float(row.aggregate_score) >= one_se_floor]
    selected = sorted(within_one_se, key=lambda row: (-float(row.min_score), row.recipe_spec.tie_break_key()))[0]
    final_rows = tuple(
        _replace_flags(
            row,
            selected=row.recipe_spec.config_hash == selected.recipe_spec.config_hash,
            within_one_se=row in within_one_se,
        )
        for row in rows
    )
    return SourceInnerRecipeSelectionResult(
        experiment_seed=int(experiment_seed),
        classifier_seed=int(classifier_seed),
        outer_target_center=str(outer_target_center),
        selected_recipe=selected.recipe_spec,
        selected_config_hash=selected.recipe_spec.config_hash,
        grid_hash=grid_hash,
        selection_metric=selection_metric,
        selector_centers=selector_centers,
        inner_lodo_centers=inner_centers,
        rows=final_rows,
    )


def assert_source_inner_recipe_artifacts(rows: Sequence[Mapping[str, object]]) -> None:
    required = set(SOURCE_INNER_RECIPE_TUNING_COLUMNS)
    for row in rows:
        missing = sorted(required.difference(row))
        if missing:
            raise ProtocolError(f"Source-inner recipe artifact row missing fields: {missing}")
        if str(row["schema_version"]) != SOURCE_INNER_RECIPE_TUNING_SCHEMA_VERSION:
            raise ProtocolError(f"Unexpected source-inner recipe schema_version: {row['schema_version']!r}")
        outer = str(row["outer_target_center"])
        selector_centers = _json_string_tuple(row["selector_centers"], "selector_centers")
        inner_centers = _json_string_tuple(row["inner_lodo_centers"], "inner_lodo_centers")
        if outer in selector_centers or outer in inner_centers:
            raise ProtocolError("Held-out target center appears in source-inner recipe selection artifacts.")
        if str(row["selection_source"]) != "source_inner_lodo":
            raise ProtocolError("Recipe selection_source must be source_inner_lodo.")
        for flag in (
            "selection_used_target_labels",
            "fit_used_target_center",
            "target_eval_labels_used_for_scoring",
            "generated_embeddings_used",
            "cvae_checkpoint_used",
            "source_summary_manifest_used",
            "is_router",
            "probabilities_calibrated",
        ):
            if str(row[flag]).lower() != "false":
                raise ProtocolError(f"{flag} must be false in source-inner recipe artifacts.")
        if str(row["claim_scope"]) != "real_feature_transfer_only":
            raise ProtocolError("Recipe claim_scope must be real_feature_transfer_only.")


def _validate_selection_inputs(
    *,
    outer_target_center: str,
    folds: Sequence[SourceInnerClassifierFold],
    candidate_recipes: Sequence[RecipeSpec],
    selection_metric: str,
) -> None:
    if selection_metric not in {"bacc", "macro_f1"}:
        raise ProtocolError(f"Unsupported recipe selection metric: {selection_metric!r}")
    if not folds:
        raise ProtocolError("At least one source-inner LODO fold is required.")
    if not candidate_recipes:
        raise ProtocolError("At least one recipe is required.")
    assert_fixed_recipe_random_state(candidate_recipes)
    seen_centers: set[str] = set()
    for fold in folds:
        if fold.pseudo_target_center == outer_target_center:
            raise ProtocolError("Outer target center cannot be a source-inner pseudo-target.")
        if outer_target_center in set(fold.train_centers):
            raise ProtocolError("Outer target center cannot appear in source-inner training centers.")
        if fold.pseudo_target_center in set(fold.train_centers):
            raise ProtocolError("Source-inner pseudo-target center cannot appear in its training centers.")
        if fold.pseudo_target_center in seen_centers:
            raise ProtocolError(f"Duplicate source-inner pseudo-target center: {fold.pseudo_target_center}")
        seen_centers.add(fold.pseudo_target_center)


def _fit_and_score_fold(recipe: RecipeSpec, fold: SourceInnerClassifierFold) -> RecipeFoldScore:
    try:
        fitted = fit_recipe(
            fold.train_embeddings,
            fold.train_labels,
            fold.validation_embeddings,
            recipe=recipe,
        )
        return RecipeFoldScore(
            bacc=balanced_accuracy(fold.validation_labels, fitted.predictions),
            macro_f1=macro_f1(fold.validation_labels, fitted.predictions),
            converged=fitted.converged,
            n_iter=fitted.n_iter,
            status=fitted.status,
            error_message=fitted.error_message,
            validation_labels=tuple(int(value) for value in fold.validation_labels),
            predictions=tuple(int(value) for value in fitted.predictions),
        )
    except Exception as exc:  # pragma: no cover - artifact row preserves workstation failures.
        return RecipeFoldScore(
            bacc=math.nan,
            macro_f1=math.nan,
            converged=False,
            status="error",
            error_message=str(exc),
        )


def _replace_flags(row: RecipeTuningCandidateRow, *, selected: bool, within_one_se: bool) -> RecipeTuningCandidateRow:
    return RecipeTuningCandidateRow(
        experiment_seed=row.experiment_seed,
        classifier_seed=row.classifier_seed,
        outer_target_center=row.outer_target_center,
        selector_centers=row.selector_centers,
        inner_lodo_centers=row.inner_lodo_centers,
        recipe_spec=row.recipe_spec,
        grid_hash=row.grid_hash,
        source_inner_lodo_center_bacc_vector=row.source_inner_lodo_center_bacc_vector,
        source_inner_lodo_center_macro_f1_vector=row.source_inner_lodo_center_macro_f1_vector,
        selection_metric=row.selection_metric,
        aggregate_score=row.aggregate_score,
        min_score=row.min_score,
        se_score=row.se_score,
        within_one_se_best=within_one_se,
        selected_by_source_inner_lodo=selected,
        tie_breaker=row.tie_breaker,
        convergence_by_center=row.convergence_by_center,
        n_iter_by_center=row.n_iter_by_center,
        status_by_center=row.status_by_center,
        row_role=row.row_role,
    )


def _standard_error(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    mean_value = sum(float(value) for value in values) / float(len(values))
    variance = sum((float(value) - mean_value) ** 2 for value in values) / float(len(values))
    return math.sqrt(variance) / math.sqrt(float(len(values)))


def _json_string_tuple(raw: object, field: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed JSON field {field}: {raw!r}") from exc
    if not isinstance(parsed, list):
        raise ProtocolError(f"Expected {field} to be a JSON list.")
    return tuple(str(value) for value in parsed)
