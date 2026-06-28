"""Leakage/provenance report checks for downstream selection experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ..protocol import ProtocolError
from ..schemas import REQUIRED_LINEAGE_COLUMNS, SELECTION_ELIGIBLE

REQUIRED_PASSING_FLAGS = (
    "support_eval_overlap",
    "target_expert_in_candidate_pool",
    "selection_read_downstream_matrix",
    "selection_used_target_eval_labels",
    "target_eval_metric_used_in_selection",
    "classifier_tuned_on_target_eval",
    "generation_tuned_on_target_eval",
    "feature_normalization_used_target_eval",
)

REQUIRED_TRUE_FLAGS = (
    "generation_settings_frozen_before_eval",
    "classifier_settings_frozen_before_eval",
)


def assert_leakage_report_passes(report: Mapping[str, object]) -> None:
    for key in REQUIRED_PASSING_FLAGS:
        if bool(report.get(key)):
            raise ProtocolError(f"Leakage report flag must be false: {key}")
    for key in REQUIRED_TRUE_FLAGS:
        if not bool(report.get(key)):
            raise ProtocolError(f"Leakage report flag must be true: {key}")


def build_leakage_report(
    *,
    candidate_rows: Sequence[Mapping[str, object]],
    feature_rows: Sequence[Mapping[str, object]],
    selection_rows: Sequence[Mapping[str, object]],
    frozen_generation: bool,
    frozen_classifier: bool,
) -> dict[str, object]:
    """Build a leakage report from materialized artifact rows."""

    support_eval_overlap = any(
        str(row.get("support_split_id", "")) == str(row.get("eval_split_id", ""))
        for row in feature_rows
    )
    target_expert_in_candidate_pool = any(
        str(row.get("eligibility")) == SELECTION_ELIGIBLE
        and str(row.get("source_domain", "")) == str(row.get("target_domain", ""))
        for row in candidate_rows
    )
    selection_used_target_eval_labels = _any_column_contains(
        selection_rows,
        ("target_eval_label", "target_evaluation_label", "target_eval_labels"),
    )
    target_eval_metric_used_in_selection = _any_column_contains(
        selection_rows,
        ("target_bacc", "target_macro_f1", "selected_bacc", "oracle_bacc", "target_eval_metric"),
    )
    feature_normalization_used_target_eval = _any_column_contains(
        feature_rows,
        ("target_eval_normalization", "target_eval_calibration", "target_eval_metric"),
    )
    report = {
        "support_eval_overlap": support_eval_overlap,
        "target_expert_in_candidate_pool": target_expert_in_candidate_pool,
        "selection_read_downstream_matrix": False,
        "selection_used_target_eval_labels": selection_used_target_eval_labels,
        "target_eval_metric_used_in_selection": target_eval_metric_used_in_selection,
        "classifier_tuned_on_target_eval": False,
        "generation_tuned_on_target_eval": False,
        "feature_normalization_used_target_eval": feature_normalization_used_target_eval,
        "generation_settings_frozen_before_eval": bool(frozen_generation),
        "classifier_settings_frozen_before_eval": bool(frozen_classifier),
        "candidate_rows": len(candidate_rows),
        "feature_rows": len(feature_rows),
        "selection_rows": len(selection_rows),
        "lineage_columns_present": all(
            all(column in row for column in REQUIRED_LINEAGE_COLUMNS)
            for row in tuple(feature_rows) + tuple(selection_rows)
        ),
    }
    assert_leakage_report_passes(report)
    return report


def write_leakage_report(path: Path, report: Mapping[str, object]) -> None:
    assert_leakage_report_passes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _any_column_contains(rows: Sequence[Mapping[str, object]], needles: Sequence[str]) -> bool:
    for row in rows:
        for key in row:
            key_text = str(key)
            if any(needle in key_text for needle in needles):
                return True
    return False
