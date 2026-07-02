"""Source-inner LODO selection for downstream classifier hyperparameters."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .classifiers import (
    ClassifierSpec,
    assert_fixed_random_state_policy,
    classifier_grid_hash,
    fit_logistic_classifier,
)
from .downstream import balanced_accuracy, macro_f1
from .protocol import ProtocolError
from .schemas.classifier_tuning import (
    SOURCE_INNER_CLASSIFIER_TUNING_COLUMNS,
    SOURCE_INNER_CLASSIFIER_TUNING_SCHEMA_VERSION,
)


ScoreFn = Callable[[ClassifierSpec, "SourceInnerClassifierFold"], "ClassifierFoldScore"]


@dataclass(frozen=True)
class SourceInnerClassifierFold:
    """One source-inner LODO fold for classifier-spec selection."""

    pseudo_target_center: str
    train_centers: tuple[str, ...]
    train_embeddings: Sequence[Sequence[float]]
    train_labels: Sequence[int]
    validation_embeddings: Sequence[Sequence[float]]
    validation_labels: Sequence[int]


@dataclass(frozen=True)
class ClassifierFoldScore:
    bacc: float
    macro_f1: float
    converged: bool = True
    n_iter: tuple[int, ...] = ()
    status: str = "ok"
    error_message: str = ""

    def metric(self, name: str) -> float:
        if name == "bacc":
            return float(self.bacc)
        if name == "macro_f1":
            return float(self.macro_f1)
        raise ProtocolError(f"Unsupported classifier selection metric: {name!r}")


@dataclass(frozen=True)
class ClassifierTuningCandidateRow:
    experiment_seed: int
    classifier_seed: int
    outer_target_center: str
    selector_centers: tuple[str, ...]
    inner_lodo_centers: tuple[str, ...]
    classifier_spec: ClassifierSpec
    grid_hash: str
    source_inner_lodo_center_scores: Mapping[str, float]
    source_inner_lodo_center_bacc_vector: Mapping[str, float]
    source_inner_lodo_center_macro_f1_vector: Mapping[str, float]
    selection_metric: str
    aggregate_score: float
    selected_by_source_inner_lodo: bool
    tie_breaker: tuple[object, ...]
    convergence_by_center: Mapping[str, bool]
    n_iter_by_center: Mapping[str, tuple[int, ...]]
    status_by_center: Mapping[str, str]

    @property
    def selected_classifier_config_hash(self) -> str:
        return self.classifier_spec.config_hash

    def to_artifact_row(self) -> dict[str, object]:
        return {
            "schema_version": SOURCE_INNER_CLASSIFIER_TUNING_SCHEMA_VERSION,
            "experiment_seed": int(self.experiment_seed),
            "classifier_seed": int(self.classifier_seed),
            "outer_target_center": self.outer_target_center,
            "selector_centers": json.dumps(list(self.selector_centers)),
            "inner_lodo_centers": json.dumps(list(self.inner_lodo_centers)),
            "classifier_grid_hash": self.grid_hash,
            "selected_classifier_spec": json.dumps(self.classifier_spec.to_payload(), sort_keys=True),
            "selected_classifier_config_hash": self.selected_classifier_config_hash,
            "source_inner_lodo_center_bacc_vector": json.dumps(
                dict(self.source_inner_lodo_center_bacc_vector), sort_keys=True
            ),
            "source_inner_lodo_center_macro_f1_vector": json.dumps(
                dict(self.source_inner_lodo_center_macro_f1_vector), sort_keys=True
            ),
            "selection_metric": self.selection_metric,
            "selection_source": "source_inner_lodo",
            "selected_by_source_inner_lodo": str(self.selected_by_source_inner_lodo).lower(),
            "aggregate_selection_score": self.aggregate_score,
            "tie_breaker": json.dumps(list(self.tie_breaker), sort_keys=True),
            "convergence_by_center": json.dumps(dict(self.convergence_by_center), sort_keys=True),
            "n_iter_by_center": json.dumps(
                {key: list(value) for key, value in self.n_iter_by_center.items()},
                sort_keys=True,
            ),
            "status_by_center": json.dumps(dict(self.status_by_center), sort_keys=True),
            "selection_used_target_labels": "false",
            "fit_used_target_center": "false",
            "target_eval_labels_used_for_scoring": "false",
        }


@dataclass(frozen=True)
class SourceInnerClassifierSelectionResult:
    experiment_seed: int
    classifier_seed: int
    outer_target_center: str
    selected_spec: ClassifierSpec
    selected_config_hash: str
    grid_hash: str
    selection_metric: str
    selector_centers: tuple[str, ...]
    inner_lodo_centers: tuple[str, ...]
    rows: tuple[ClassifierTuningCandidateRow, ...]

    def to_artifact_rows(self) -> list[dict[str, object]]:
        return [row.to_artifact_row() for row in self.rows]


def select_classifier_spec_source_inner_lodo(
    *,
    outer_target_center: str,
    folds: Sequence[SourceInnerClassifierFold],
    candidate_specs: Sequence[ClassifierSpec],
    experiment_seed: int = 0,
    classifier_seed: int = 0,
    selection_metric: str = "bacc",
    score_fn: ScoreFn | None = None,
    reject_non_converged: bool = True,
) -> SourceInnerClassifierSelectionResult:
    """Select a classifier spec using source-inner LODO folds only.

    The held-out real target center must be absent from every fold. The returned
    rows are selection artifacts only; held-out target metrics must be written by
    the final downstream evaluation path after the selected recipe is frozen.
    """

    _validate_selection_inputs(
        outer_target_center=outer_target_center,
        folds=folds,
        candidate_specs=candidate_specs,
        selection_metric=selection_metric,
    )
    scorer = score_fn or _fit_and_score_fold
    grid_hash = classifier_grid_hash(candidate_specs)
    inner_centers = tuple(fold.pseudo_target_center for fold in folds)
    selector_centers = tuple(sorted({center for fold in folds for center in (*fold.train_centers, fold.pseudo_target_center)}))
    rows: list[ClassifierTuningCandidateRow] = []
    for spec in candidate_specs:
        fold_scores: dict[str, ClassifierFoldScore] = {}
        for fold in folds:
            fold_scores[fold.pseudo_target_center] = scorer(spec, fold)
        metric_scores = {
            center: score.metric(selection_metric)
            for center, score in fold_scores.items()
            if score.status == "ok" and (score.converged or not reject_non_converged)
        }
        if len(metric_scores) != len(folds):
            aggregate = math.nan
        else:
            aggregate = sum(metric_scores.values()) / float(len(metric_scores))
        rows.append(
            ClassifierTuningCandidateRow(
                outer_target_center=outer_target_center,
                experiment_seed=int(experiment_seed),
                classifier_seed=int(classifier_seed),
                selector_centers=selector_centers,
                inner_lodo_centers=inner_centers,
                classifier_spec=spec,
                grid_hash=grid_hash,
                source_inner_lodo_center_scores=metric_scores,
                source_inner_lodo_center_bacc_vector={
                    center: fold_scores[center].bacc for center in inner_centers
                },
                source_inner_lodo_center_macro_f1_vector={
                    center: fold_scores[center].macro_f1 for center in inner_centers
                },
                selection_metric=selection_metric,
                aggregate_score=aggregate,
                selected_by_source_inner_lodo=False,
                tie_breaker=spec.tie_break_key(),
                convergence_by_center={center: fold_scores[center].converged for center in inner_centers},
                n_iter_by_center={center: fold_scores[center].n_iter for center in inner_centers},
                status_by_center={center: fold_scores[center].status for center in inner_centers},
            )
        )
    eligible = [row for row in rows if not math.isnan(float(row.aggregate_score))]
    if not eligible:
        raise ProtocolError("No classifier spec produced valid source-inner LODO scores.")
    selected = max(eligible, key=lambda row: (float(row.aggregate_score), _reverse_tie_breaker(row.tie_breaker)))
    selected_rows = tuple(
        _replace_selected(row, row.classifier_spec.config_hash == selected.classifier_spec.config_hash)
        for row in rows
    )
    return SourceInnerClassifierSelectionResult(
        experiment_seed=int(experiment_seed),
        classifier_seed=int(classifier_seed),
        outer_target_center=outer_target_center,
        selected_spec=selected.classifier_spec,
        selected_config_hash=selected.classifier_spec.config_hash,
        grid_hash=grid_hash,
        selection_metric=selection_metric,
        selector_centers=selector_centers,
        inner_lodo_centers=inner_centers,
        rows=selected_rows,
    )


def assert_source_inner_classifier_artifacts(rows: Sequence[Mapping[str, object]]) -> None:
    """Validate persisted source-inner classifier-selection rows."""

    required = set(SOURCE_INNER_CLASSIFIER_TUNING_COLUMNS)
    for row in rows:
        missing = sorted(required.difference(row))
        if missing:
            raise ProtocolError(f"Source-inner classifier artifact row missing fields: {missing}")
        if str(row["schema_version"]) != SOURCE_INNER_CLASSIFIER_TUNING_SCHEMA_VERSION:
            raise ProtocolError(
                f"Unexpected source-inner classifier schema_version: {row['schema_version']!r}"
            )
        outer = str(row["outer_target_center"])
        selector_centers = _json_string_tuple(row["selector_centers"], "selector_centers")
        inner_centers = _json_string_tuple(row["inner_lodo_centers"], "inner_lodo_centers")
        if outer in selector_centers or outer in inner_centers:
            raise ProtocolError("Held-out target center appears in source-inner classifier selection artifacts.")
        if str(row["selection_source"]) != "source_inner_lodo":
            raise ProtocolError("Classifier selection_source must be source_inner_lodo.")
        if str(row["selection_used_target_labels"]).lower() != "false":
            raise ProtocolError("Classifier selection must not use target labels.")
        if str(row["fit_used_target_center"]).lower() != "false":
            raise ProtocolError("Classifier selection fit must not use the target center.")
        if str(row["target_eval_labels_used_for_scoring"]).lower() != "false":
            raise ProtocolError("Source-inner classifier rows must not score target evaluation labels.")


def _validate_selection_inputs(
    *,
    outer_target_center: str,
    folds: Sequence[SourceInnerClassifierFold],
    candidate_specs: Sequence[ClassifierSpec],
    selection_metric: str,
) -> None:
    if selection_metric not in {"bacc", "macro_f1"}:
        raise ProtocolError(f"Unsupported classifier selection metric: {selection_metric!r}")
    if not folds:
        raise ProtocolError("At least one source-inner LODO fold is required.")
    if not candidate_specs:
        raise ProtocolError("At least one classifier spec is required.")
    assert_fixed_random_state_policy(candidate_specs)
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


def _fit_and_score_fold(spec: ClassifierSpec, fold: SourceInnerClassifierFold) -> ClassifierFoldScore:
    try:
        fitted = fit_logistic_classifier(
            fold.train_embeddings,
            fold.train_labels,
            fold.validation_embeddings,
            spec=spec,
        )
        predictions = fitted.predictions.tolist()
        return ClassifierFoldScore(
            bacc=balanced_accuracy(fold.validation_labels, predictions),
            macro_f1=macro_f1(fold.validation_labels, predictions),
            converged=fitted.converged,
            n_iter=fitted.n_iter,
            status="ok",
        )
    except Exception as exc:  # pragma: no cover - artifact row preserves workstation failures.
        return ClassifierFoldScore(
            bacc=math.nan,
            macro_f1=math.nan,
            converged=False,
            status="error",
            error_message=str(exc),
        )


def _replace_selected(
    row: ClassifierTuningCandidateRow,
    selected: bool,
) -> ClassifierTuningCandidateRow:
    return ClassifierTuningCandidateRow(
        outer_target_center=row.outer_target_center,
        experiment_seed=row.experiment_seed,
        classifier_seed=row.classifier_seed,
        selector_centers=row.selector_centers,
        inner_lodo_centers=row.inner_lodo_centers,
        classifier_spec=row.classifier_spec,
        grid_hash=row.grid_hash,
        source_inner_lodo_center_scores=row.source_inner_lodo_center_scores,
        source_inner_lodo_center_bacc_vector=row.source_inner_lodo_center_bacc_vector,
        source_inner_lodo_center_macro_f1_vector=row.source_inner_lodo_center_macro_f1_vector,
        selection_metric=row.selection_metric,
        aggregate_score=row.aggregate_score,
        selected_by_source_inner_lodo=selected,
        tie_breaker=row.tie_breaker,
        convergence_by_center=row.convergence_by_center,
        n_iter_by_center=row.n_iter_by_center,
        status_by_center=row.status_by_center,
    )


def _reverse_tie_breaker(key: tuple[object, ...]) -> tuple[object, ...]:
    reversed_items: list[object] = []
    for item in key:
        if isinstance(item, (int, float)):
            reversed_items.append(-float(item))
        else:
            reversed_items.append(_reverse_lex(str(item)))
    return tuple(reversed_items)


def _reverse_lex(value: str) -> str:
    return "".join(chr(255 - ord(ch)) for ch in value)


def _json_string_tuple(raw: object, field: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed JSON field {field}: {raw!r}") from exc
    if not isinstance(parsed, list):
        raise ProtocolError(f"Expected {field} to be a JSON list.")
    return tuple(str(value) for value in parsed)
