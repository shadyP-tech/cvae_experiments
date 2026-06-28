"""End-to-end learned downstream utility artifact pipeline."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..downstream import read_candidate_downstream_matrix
from ..features.feature_table_builder import (
    build_allowed_feature_table,
    build_allowed_feature_table_from_artifacts,
    read_csv_rows,
    write_allowed_feature_table,
)
from ..protocol import ProtocolError
from ..reports.leakage_report import build_leakage_report, write_leakage_report
from ..reports.rank_metrics import build_learned_utility_alignment_rows, learned_utility_alignment_columns
from ..reports.tables import write_rows
from ..utility_matrix import assert_diagnostic_matrix_path
from .diagnostics import estimator_diagnostics
from .estimators import load_estimator, predict_rows, save_estimator
from .select_candidates import build_baseline_selection_rows, build_top1_selection_rows, write_selection_rows
from .train_source_inner import train_linear_utility_estimator


@dataclass(frozen=True)
class LearnedUtilityPipelineInputs:
    candidates: Path
    source_inner_training: Path
    diagnostic_matrix: Path
    out_dir: Path
    feature_columns: tuple[str, ...]
    support_features: Path | None = None
    source_inner_features: Path | None = None
    metadata_features: Path | None = None
    label: str = "source_inner_heldout_bacc"
    ridge_lambda: float = 1e-6
    method: str = "learned_downstream_utility_top1"
    generation_frozen: bool = True
    classifier_frozen: bool = True


@dataclass(frozen=True)
class LearnedUtilityPipelineOutputs:
    allowed_features: Path
    predicted_features: Path
    estimator_model: Path
    estimator_diagnostics: Path
    selections: Path
    baseline_selections: Path
    alignment: Path
    baseline_alignment: Path
    leakage_report: Path
    manifest: Path


def run_learned_utility_pipeline(inputs: LearnedUtilityPipelineInputs) -> LearnedUtilityPipelineOutputs:
    """Run the protocol-safe learned utility artifact path."""

    assert_diagnostic_matrix_path(inputs.diagnostic_matrix)
    out_dir = Path(inputs.out_dir)
    outputs = LearnedUtilityPipelineOutputs(
        allowed_features=out_dir / "features" / "allowed_pre_eval_features.csv",
        predicted_features=out_dir / "features" / "allowed_pre_eval_features_with_predictions.csv",
        estimator_model=out_dir / "models" / "source_inner_utility_estimator.json",
        estimator_diagnostics=out_dir / "reports" / "source_inner_estimator_diagnostics.json",
        selections=out_dir / "selections" / "adoption_eligible_predictions.csv",
        baseline_selections=out_dir / "selections" / "baseline_results.csv",
        alignment=out_dir / "reports" / "learned_utility_alignment.csv",
        baseline_alignment=out_dir / "reports" / "baseline_alignment.csv",
        leakage_report=out_dir / "reports" / "leakage_report.json",
        manifest=out_dir / "reports" / "learned_utility_pipeline_manifest.json",
    )

    candidate_rows = read_csv_rows(inputs.candidates)
    allowed_rows = build_allowed_feature_table_from_artifacts(
        candidate_rows=candidate_rows,
        support_feature_rows=_optional_rows(inputs.support_features),
        source_inner_rows=_optional_rows(inputs.source_inner_features),
        metadata_rows=_optional_rows(inputs.metadata_features),
    )
    write_allowed_feature_table(outputs.allowed_features, allowed_rows)

    source_inner_rows = _read_csv(inputs.source_inner_training)
    estimator = train_linear_utility_estimator(
        source_inner_rows,
        feature_columns=inputs.feature_columns,
        label=inputs.label,
        ridge_lambda=inputs.ridge_lambda,
    )
    save_estimator(outputs.estimator_model, estimator)
    training_predictions = predict_rows(estimator, source_inner_rows)
    diagnostics = estimator_diagnostics(training_predictions, label_column=inputs.label)
    _write_json(outputs.estimator_diagnostics, diagnostics)

    reloaded_estimator = load_estimator(outputs.estimator_model)
    predicted_rows = predict_rows(reloaded_estimator, build_allowed_feature_table(allowed_rows))
    write_allowed_feature_table(outputs.predicted_features, predicted_rows)

    selection_rows = build_top1_selection_rows(predicted_rows, method=inputs.method)
    write_selection_rows(outputs.selections, selection_rows)
    baseline_selection_rows = build_baseline_selection_rows(predicted_rows)
    write_selection_rows(outputs.baseline_selections, baseline_selection_rows)

    downstream_rows = read_candidate_downstream_matrix(inputs.diagnostic_matrix)
    alignment_rows = build_learned_utility_alignment_rows(
        selection_rows=selection_rows,
        downstream_rows=downstream_rows,
    )
    write_rows(outputs.alignment, learned_utility_alignment_columns(), alignment_rows)
    baseline_alignment_rows = build_learned_utility_alignment_rows(
        selection_rows=baseline_selection_rows,
        downstream_rows=downstream_rows,
    )
    write_rows(outputs.baseline_alignment, learned_utility_alignment_columns(), baseline_alignment_rows)

    leakage = build_leakage_report(
        candidate_rows=candidate_rows,
        feature_rows=predicted_rows,
        selection_rows=selection_rows,
        frozen_generation=inputs.generation_frozen,
        frozen_classifier=inputs.classifier_frozen,
    )
    write_leakage_report(outputs.leakage_report, leakage)

    _write_json(
        outputs.manifest,
        {
            "schema_version": "learned_utility_pipeline_manifest_v1",
            "candidates": str(inputs.candidates),
            "source_inner_training": str(inputs.source_inner_training),
            "diagnostic_matrix": str(inputs.diagnostic_matrix),
            "feature_columns": list(inputs.feature_columns),
            "label": inputs.label,
            "ridge_lambda": inputs.ridge_lambda,
            "method": inputs.method,
            "outputs": {
                key: str(value)
                for key, value in outputs.__dict__.items()
            },
        },
    )
    return outputs


def _optional_rows(path: Path | None) -> list[dict[str, object]]:
    return read_csv_rows(path) if path else []


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = json.dumps(payload, indent=2, sort_keys=True)
    except TypeError as exc:
        raise ProtocolError(f"Pipeline manifest payload is not JSON serializable: {path}") from exc
    path.write_text(encoded + "\n", encoding="utf-8")
