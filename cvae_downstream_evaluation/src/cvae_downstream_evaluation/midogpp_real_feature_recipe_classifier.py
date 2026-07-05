"""MIDOG++ real-feature source-inner recipe classifier reference runner."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .artifacts import stable_hash
from .downstream import balanced_accuracy, macro_f1
from .midogpp_real_feature_classifier import (
    RealFeatureFrame,
    RealFeatureRow,
    build_source_inner_folds,
    load_midogpp_real_feature_frame,
    _embeddings_for_indices,
    _indices_for_centers,
    _labels_for_indices,
    _outer_and_inner_leakage_rows,
    _validate_heldout_centers,
)
from .protocol import ProtocolError
from .real_feature_recipes import OUTER_MODEL_FIT_SCOPE, OUTER_PREPROCESSING_FIT_SCOPE, RecipeSpec, fit_recipe, recipe_grid_hash
from .recipe_grid import build_v3_logistic_baseline_grid, build_v3_recipe_grid
from .schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS, MIDOGPP_EXCLUDED_CENTERS
from .schemas.midogpp_real_feature_recipe import (
    RECIPE_PREDICTION_COLUMNS,
    RECIPE_PREDICTIONS_SCHEMA_VERSION,
    RECIPE_REFERENCE_SCHEMA_VERSION,
    RECIPE_RESULT_COLUMNS,
    RECIPE_RESULTS_SCHEMA_VERSION,
    SOURCE_INNER_RECIPE_TUNING_COLUMNS,
    assert_midogpp_real_feature_recipe_artifacts,
)
from .source_inner_classifier_tuning import SourceInnerClassifierFold
from .source_inner_recipe_tuning import (
    RecipeScoreFn,
    SourceInnerRecipeSelectionResult,
    assert_source_inner_recipe_artifacts,
    select_recipe_source_inner_lodo,
)


RecipeScoreFnFactory = Callable[[str], RecipeScoreFn | None]


@dataclass(frozen=True)
class RealFeatureRecipeRunPaths:
    source_inner_tuning: Path
    results: Path
    predictions: Path
    protocol_manifest: Path
    leakage_report: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "source_inner_tuning": self.source_inner_tuning,
            "results": self.results,
            "predictions": self.predictions,
            "protocol_manifest": self.protocol_manifest,
            "leakage_report": self.leakage_report,
        }


def run_midogpp_real_feature_recipe_tuning(
    *,
    manifest_path: Path,
    feature_cache_path: Path,
    out_dir: Path,
    candidate_recipes: Sequence[RecipeSpec] | None = None,
    logistic_baseline_recipes: Sequence[RecipeSpec] | None = None,
    heldout_centers: Sequence[str] = MIDOGPP_ELIGIBLE_CENTERS,
    experiment_seed: int = 42,
    classifier_seed: int = 23,
    expected_feature_dim: int = 2560,
    selection_metric: str = "bacc",
    score_fn_factory: RecipeScoreFnFactory | None = None,
) -> RealFeatureRecipeRunPaths:
    """Run the approved source-inner recipe sweep directly on real features."""

    frame = load_midogpp_real_feature_frame(
        manifest_path=Path(manifest_path),
        feature_cache_path=Path(feature_cache_path),
        expected_feature_dim=int(expected_feature_dim),
    )
    heldouts = _validate_heldout_centers(heldout_centers, frame=frame)
    recipes = tuple(candidate_recipes or build_v3_recipe_grid(classifier_seed=int(classifier_seed)))
    baseline_recipes = tuple(logistic_baseline_recipes or build_v3_logistic_baseline_grid(classifier_seed=int(classifier_seed)))
    if not any(recipe.family == "logistic" for recipe in baseline_recipes):
        raise ProtocolError("Recipe run requires a same-cache logistic baseline grid.")
    grid_hash = recipe_grid_hash(recipes)
    baseline_grid_hash = recipe_grid_hash(baseline_recipes)
    protocol_hash_payload = {
        "schema_version": RECIPE_REFERENCE_SCHEMA_VERSION,
        "artifact_identity": "real_feature_source_only_classifier_reference",
        "method": "real_feature_source_only_recipe_reference",
        "experiment_seed": int(experiment_seed),
        "classifier_seed": int(classifier_seed),
        "recipe_grid_hash": grid_hash,
        "logistic_baseline_grid_hash": baseline_grid_hash,
        "recipe_grid": [recipe.to_payload() for recipe in recipes],
        "logistic_baseline_grid": [recipe.to_payload() for recipe in baseline_recipes],
        "feature_cache_hash": frame.feature_cache_hash,
        "manifest_hash": frame.manifest_hash,
        "heldout_centers": list(heldouts),
        "eligible_centers": list(MIDOGPP_ELIGIBLE_CENTERS),
        "excluded_centers": list(MIDOGPP_EXCLUDED_CENTERS),
        "selection_metric": selection_metric,
        "selection_rule_id": "source_inner_mean_bacc_one_se_min_bacc_simpler_family_v3",
        "decision_rule": "predict",
        "probabilities_calibrated": False,
    }
    protocol_hash = stable_hash(protocol_hash_payload)
    tuning_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    fit_scope_rows: list[dict[str, object]] = []
    for heldout in heldouts:
        folds, fold_audit = build_source_inner_folds(frame, outer_target_center=heldout)
        score_fn = score_fn_factory(heldout) if score_fn_factory else None
        selection = select_recipe_source_inner_lodo(
            outer_target_center=heldout,
            folds=folds,
            candidate_recipes=recipes,
            experiment_seed=int(experiment_seed),
            classifier_seed=int(classifier_seed),
            selection_metric=selection_metric,
            score_fn=score_fn,
            row_role="selection_candidate",
        )
        baseline_selection = select_recipe_source_inner_lodo(
            outer_target_center=heldout,
            folds=folds,
            candidate_recipes=baseline_recipes,
            experiment_seed=int(experiment_seed),
            classifier_seed=int(classifier_seed),
            selection_metric=selection_metric,
            score_fn=score_fn,
            row_role="locked_logistic_baseline_candidate",
        )
        selection_rows = [
            _augment_tuning_row(row, frame=frame, protocol_hash=protocol_hash, fold_audit=fold_audit)
            for row in selection.to_artifact_rows()
        ]
        baseline_rows = [
            _augment_tuning_row(row, frame=frame, protocol_hash=protocol_hash, fold_audit=fold_audit)
            for row in baseline_selection.to_artifact_rows()
        ]
        tuning_rows.extend(selection_rows)
        tuning_rows.extend(baseline_rows)
        selected_eval = fit_and_score_heldout_recipe(
            frame,
            heldout_center=heldout,
            recipe=selection.selected_recipe,
            method="source_inner_recipe_selected_predict",
            row_role="selected_recipe",
            experiment_seed=int(experiment_seed),
            classifier_seed=int(classifier_seed),
            grid_hash=grid_hash,
            source_inner_selection=selection,
        )
        result_rows.append(selected_eval["result"])
        prediction_rows.extend(selected_eval["predictions"])
        baseline_eval = fit_and_score_heldout_recipe(
            frame,
            heldout_center=heldout,
            recipe=baseline_selection.selected_recipe,
            method="locked_logistic_baseline_predict",
            row_role="locked_logistic_baseline",
            experiment_seed=int(experiment_seed),
            classifier_seed=int(classifier_seed),
            grid_hash=baseline_grid_hash,
            source_inner_selection=baseline_selection,
        )
        result_rows.append(baseline_eval["result"])
        prediction_rows.extend(baseline_eval["predictions"])
        leakage_rows.extend(_outer_and_inner_leakage_rows(frame, heldout_center=heldout, fold_audit=fold_audit))
        fit_scope_rows.extend(_fit_scope_rows(frame, heldout_center=heldout, fold_audit=fold_audit))
    paths = _write_real_feature_recipe_artifacts(
        out_dir=Path(out_dir),
        frame=frame,
        tuning_rows=tuning_rows,
        result_rows=result_rows,
        prediction_rows=prediction_rows,
        leakage_rows=leakage_rows,
        fit_scope_rows=fit_scope_rows,
        protocol_hash=protocol_hash,
        protocol_hash_payload=protocol_hash_payload,
    )
    assert_midogpp_real_feature_recipe_artifacts(Path(out_dir))
    return paths


def fit_and_score_heldout_recipe(
    frame: RealFeatureFrame,
    *,
    heldout_center: str,
    recipe: RecipeSpec,
    method: str,
    row_role: str,
    experiment_seed: int,
    classifier_seed: int,
    grid_hash: str,
    source_inner_selection: SourceInnerRecipeSelectionResult,
) -> dict[str, object]:
    """Refit a frozen recipe on non-heldout source centers and score H once."""

    heldout = str(heldout_center)
    train_centers = tuple(center for center in frame.eligible_centers if center != heldout)
    train_idx = _indices_for_centers(frame, train_centers)
    eval_idx = _indices_for_centers(frame, (heldout,))
    train_labels = _labels_for_indices(frame, train_idx)
    eval_labels = _labels_for_indices(frame, eval_idx)
    fitted = fit_recipe(
        _embeddings_for_indices(frame, train_idx),
        train_labels,
        _embeddings_for_indices(frame, eval_idx),
        recipe=recipe,
    )
    result_row = _result_row(
        method=method,
        row_role=row_role,
        experiment_seed=experiment_seed,
        classifier_seed=classifier_seed,
        heldout_center=heldout,
        train_centers=train_centers,
        n_train=len(train_idx),
        n_eval=len(eval_idx),
        grid_hash=grid_hash,
        recipe=recipe,
        source_inner_summary=_source_inner_summary(source_inner_selection),
        heldout_bacc=balanced_accuracy(eval_labels, fitted.predictions),
        heldout_macro_f1=macro_f1(eval_labels, fitted.predictions),
        converged=fitted.converged,
        n_iter=fitted.n_iter,
        status=fitted.status,
        error_message=fitted.error_message,
        feature_cache_hash=frame.feature_cache_hash,
        manifest_hash=frame.manifest_hash,
    )
    prediction_rows = []
    for local_idx, row_index in enumerate(eval_idx):
        sample = frame.rows[row_index]
        prediction_rows.append(
            _prediction_row(
                method=method,
                row_role=row_role,
                experiment_seed=experiment_seed,
                classifier_seed=classifier_seed,
                heldout_center=heldout,
                sample=sample,
                y_pred=fitted.predictions[local_idx],
                score_pos=fitted.score_pos[local_idx],
                decision_score=fitted.decision_score[local_idx],
                score_kind=fitted.score_kind,
                recipe=recipe,
                feature_cache_hash=frame.feature_cache_hash,
                manifest_hash=frame.manifest_hash,
            )
        )
    return {"result": result_row, "predictions": prediction_rows}


def validate_real_feature_recipe_outputs(root: Path) -> None:
    assert_midogpp_real_feature_recipe_artifacts(Path(root))


def _write_real_feature_recipe_artifacts(
    *,
    out_dir: Path,
    frame: RealFeatureFrame,
    tuning_rows: list[dict[str, object]],
    result_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    leakage_rows: list[dict[str, object]],
    fit_scope_rows: list[dict[str, object]],
    protocol_hash: str,
    protocol_hash_payload: Mapping[str, object],
) -> RealFeatureRecipeRunPaths:
    tables_dir = out_dir / "tables"
    manifests_dir = out_dir / "manifests"
    reports_dir = out_dir / "reports"
    paths = RealFeatureRecipeRunPaths(
        source_inner_tuning=tables_dir / "source_inner_recipe_tuning.csv",
        results=tables_dir / "recipe_tuned_source_results.csv",
        predictions=tables_dir / "recipe_tuned_predictions.csv",
        protocol_manifest=manifests_dir / "protocol_manifest.json",
        leakage_report=reports_dir / "leakage_provenance_report.json",
    )
    assert_source_inner_recipe_artifacts(tuning_rows)
    _write_csv(paths.source_inner_tuning, tuning_rows, columns=SOURCE_INNER_RECIPE_TUNING_COLUMNS)
    _write_csv(paths.results, result_rows, columns=RECIPE_RESULT_COLUMNS)
    _write_csv(paths.predictions, prediction_rows, columns=RECIPE_PREDICTION_COLUMNS)
    protocol_manifest = {
        **dict(protocol_hash_payload),
        "protocol_hash": protocol_hash,
        "feature_cache_path": str(frame.feature_cache_path),
        "manifest_path": str(frame.manifest_path),
        "expected_feature_dim": int(frame.expected_feature_dim),
        "target_eval_labels_used_for_scoring_only": True,
        "selection_used_target_labels": False,
        "fit_used_target_center": False,
        "generated_embeddings_used": False,
        "cvae_checkpoint_used": False,
        "source_summary_manifest_used": False,
        "is_router": False,
        "claim_scope": "real_feature_transfer_only",
        "selections_from_target_metrics": "forbidden",
        "nonselected_candidate_summary_role": "diagnostic_only_if_written",
        "adoption_gate": {
            "mean_bacc_gt": 0.740312,
            "worst_center_bacc_gte": 0.679245,
            "center_wins_gte": 7,
            "ceiling_0_8_requires_seed_stability": True,
        },
    }
    _write_json(paths.protocol_manifest, protocol_manifest)
    leakage_report = {
        "schema_version": RECIPE_REFERENCE_SCHEMA_VERSION,
        "status": "PASS",
        "fold_unit": "heldout_center",
        "target_center_excluded_from_tuning": True,
        "target_center_excluded_from_final_fit": True,
        "target_labels_used_for_scoring_only": True,
        "target_metric_used_for_selection": False,
        "selection_used_target_labels": False,
        "fit_used_target_center": False,
        "generated_embeddings_used": False,
        "cvae_checkpoint_used": False,
        "source_summary_manifest_used": False,
        "is_router": False,
        "claim_scope": "real_feature_transfer_only",
        "probabilities_calibrated": False,
        "overlap_rows": leakage_rows,
        "fit_scope_rows": fit_scope_rows,
    }
    _write_json(paths.leakage_report, leakage_report)
    return paths


def _augment_tuning_row(
    row: Mapping[str, object],
    *,
    frame: RealFeatureFrame,
    protocol_hash: str,
    fold_audit: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        **dict(row),
        "protocol_hash": protocol_hash,
        "feature_cache_path": str(frame.feature_cache_path),
        "feature_cache_hash": frame.feature_cache_hash,
        "manifest_path": str(frame.manifest_path),
        "manifest_hash": frame.manifest_hash,
        "source_inner_fold_audit": json.dumps(fold_audit, sort_keys=True),
    }


def _result_row(
    *,
    method: str,
    row_role: str,
    experiment_seed: int,
    classifier_seed: int,
    heldout_center: str,
    train_centers: Sequence[str],
    n_train: int,
    n_eval: int,
    grid_hash: str,
    recipe: RecipeSpec,
    source_inner_summary: Mapping[str, object],
    heldout_bacc: float,
    heldout_macro_f1: float,
    converged: bool,
    n_iter: Sequence[int],
    status: str,
    error_message: str,
    feature_cache_hash: str,
    manifest_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": RECIPE_RESULTS_SCHEMA_VERSION,
        "method": method,
        "row_role": row_role,
        "experiment_seed": int(experiment_seed),
        "classifier_seed": int(classifier_seed),
        "heldout_center": heldout_center,
        "train_centers": json.dumps(list(train_centers)),
        "n_train": int(n_train),
        "n_eval": int(n_eval),
        "recipe_grid_hash": grid_hash,
        "selected_recipe_config_hash": recipe.config_hash,
        "selected_recipe_spec": json.dumps(recipe.to_payload(), sort_keys=True),
        "recipe_family": recipe.family,
        "preprocessing_id": recipe.preprocessing_id,
        "selection_source": "source_inner_lodo",
        "source_inner_mean_bacc": source_inner_summary.get("mean_bacc", ""),
        "source_inner_min_bacc": source_inner_summary.get("min_bacc", ""),
        "source_inner_se_bacc": source_inner_summary.get("se_bacc", ""),
        "source_inner_n_centers": source_inner_summary.get("n_centers", 0),
        "heldout_bacc": float(heldout_bacc),
        "heldout_macro_f1": float(heldout_macro_f1),
        "converged": str(bool(converged)).lower(),
        "n_iter": json.dumps(list(int(value) for value in n_iter)),
        "status": status,
        "error_message": error_message,
        "preprocessing_fit_scope": OUTER_PREPROCESSING_FIT_SCOPE,
        "model_fit_scope": OUTER_MODEL_FIT_SCOPE,
        "decision_rule": recipe.model.decision_rule,
        "feature_cache_hash": feature_cache_hash,
        "manifest_hash": manifest_hash,
        "target_eval_labels_used_for_scoring_only": "true",
        "selection_used_target_labels": "false",
        "fit_used_target_center": "false",
        "generated_embeddings_used": "false",
        "cvae_checkpoint_used": "false",
        "source_summary_manifest_used": "false",
        "is_router": "false",
        "claim_scope": "real_feature_transfer_only",
        "probabilities_calibrated": "false",
    }


def _prediction_row(
    *,
    method: str,
    row_role: str,
    experiment_seed: int,
    classifier_seed: int,
    heldout_center: str,
    sample: RealFeatureRow,
    y_pred: int,
    score_pos: float | None,
    decision_score: float | None,
    score_kind: str,
    recipe: RecipeSpec,
    feature_cache_hash: str,
    manifest_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": RECIPE_PREDICTIONS_SCHEMA_VERSION,
        "method": method,
        "row_role": row_role,
        "experiment_seed": int(experiment_seed),
        "classifier_seed": int(classifier_seed),
        "heldout_center": heldout_center,
        "sample_id": sample.sample_id,
        "case_id": sample.case_id,
        "center": sample.center,
        "y_true": int(sample.label),
        "y_pred": int(y_pred),
        "score_pos": "" if score_pos is None else float(score_pos),
        "decision_score": "" if decision_score is None else float(decision_score),
        "score_kind": score_kind,
        "selected_recipe_config_hash": recipe.config_hash,
        "recipe_family": recipe.family,
        "preprocessing_id": recipe.preprocessing_id,
        "feature_cache_hash": feature_cache_hash,
        "manifest_hash": manifest_hash,
        "target_eval_labels_used_for_scoring_only": "true",
        "selection_used_target_labels": "false",
        "fit_used_target_center": "false",
        "probabilities_calibrated": "false",
    }


def _source_inner_summary(selection: SourceInnerRecipeSelectionResult) -> dict[str, object]:
    selected = [row for row in selection.rows if row.selected_by_source_inner_lodo]
    if not selected:
        return {"mean_bacc": "", "min_bacc": "", "se_bacc": "", "n_centers": 0}
    vector = selected[0].source_inner_lodo_center_bacc_vector
    values = [float(value) for value in vector.values()]
    if not values:
        return {"mean_bacc": "", "min_bacc": "", "se_bacc": "", "n_centers": 0}
    mean_value = sum(values) / float(len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values))
    return {
        "mean_bacc": mean_value,
        "min_bacc": min(values),
        "se_bacc": math.sqrt(variance) / math.sqrt(float(len(values))),
        "n_centers": len(values),
    }


def _fit_scope_rows(
    frame: RealFeatureFrame,
    *,
    heldout_center: str,
    fold_audit: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "fold_role": "outer_final_fit",
            "heldout_center": str(heldout_center),
            "pseudo_target_center": "",
            "fit_state_components": ["scaler", "pca", "nystroem", "model", "class_weight_or_sample_weight"],
            "target_center_excluded_from_fit": True,
            "pseudo_target_excluded_from_inner_fit": True,
            "fit_centers": [center for center in frame.eligible_centers if center != str(heldout_center)],
        }
    ]
    for pseudo_target, audit in fold_audit.items():
        rows.append(
            {
                "fold_role": "source_inner_lodo",
                "heldout_center": str(heldout_center),
                "pseudo_target_center": str(pseudo_target),
                "fit_state_components": ["scaler", "pca", "nystroem", "model", "class_weight_or_sample_weight"],
                "target_center_excluded_from_fit": True,
                "pseudo_target_excluded_from_inner_fit": True,
                "fit_centers": list(audit["train_centers"]),  # type: ignore[index]
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], *, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
