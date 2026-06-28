"""Candidate selection wrappers for adoption-eligible feature rows."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from ..features import assert_allowed_feature_table
from ..protocol import ProtocolError
from ..schemas import (
    ADOPTION_ELIGIBLE_SELECTION_COLUMNS,
    REQUIRED_LINEAGE_COLUMNS,
    SELECTION_ELIGIBLE,
)
from . import CompatibilityPrediction, select_top1

__all__ = [
    "CompatibilityPrediction",
    "select_top1",
    "build_top1_selection_rows",
    "build_baseline_selection_rows",
    "write_selection_rows",
]


def build_top1_selection_rows(
    feature_rows: Sequence[Mapping[str, object]],
    *,
    method: str = "learned_downstream_utility_top1",
    score_column: str = "predicted_primary_utility",
    support_nelbo_column: str = "support_nelbo",
    stability_column: str = "source_inner_stability",
) -> list[dict[str, object]]:
    """Select one candidate per target-support context using allowed features only."""

    assert_allowed_feature_table(feature_rows)
    groups: dict[tuple[str, str, str, str, str], list[Mapping[str, object]]] = {}
    for row in feature_rows:
        if str(row.get("eligibility")) != SELECTION_ELIGIBLE:
            continue
        key = (
            str(row["fold_id"]),
            str(row["experiment_seed"]),
            str(row["target_domain"]),
            str(row["support_split_id"]),
            str(row["eval_split_id"]),
        )
        groups.setdefault(key, []).append(row)
    if not groups:
        raise ProtocolError("No selection-eligible feature rows available.")

    selections: list[dict[str, object]] = []
    for key, rows in sorted(groups.items()):
        predictions = [
            CompatibilityPrediction(
                candidate_id=str(row["candidate_id"]),
                predicted_primary_utility=float(row.get(score_column, float("nan"))),
                support_nelbo=float(row.get(support_nelbo_column, float("inf"))),
                source_inner_stability=float(row.get(stability_column, 0.0)),
            )
            for row in rows
        ]
        selected_prediction = select_top1(predictions)
        selected_row = next(row for row in rows if str(row["candidate_id"]) == selected_prediction.candidate_id)
        selection = {column: selected_row[column] for column in REQUIRED_LINEAGE_COLUMNS}
        selection.update(
            {
                "method": method,
                "predicted_primary_utility": selected_prediction.predicted_primary_utility,
                "support_nelbo": selected_prediction.support_nelbo,
                "source_inner_stability": selected_prediction.source_inner_stability,
                "selection_rank": 1,
                "aggregation_weight": 1.0,
            }
        )
        selections.append(selection)
    return selections


def build_baseline_selection_rows(
    feature_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Materialize required non-oracle top-1 baselines from allowed features only."""

    assert_allowed_feature_table(feature_rows)
    groups = _selection_groups(feature_rows)
    selections: list[dict[str, object]] = []
    for key, rows in sorted(groups.items()):
        selections.append(_selection_from_row(_select_metadata_top1(rows), method="metadata_top1"))
        selections.append(_selection_from_row(_select_support_nelbo_top1(rows), method="support_nelbo_top1"))
        selections.append(_selection_from_row(_select_deterministic_random(rows, key), method="random_expert"))
    return selections


def _selection_groups(feature_rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, str, str, str], list[Mapping[str, object]]]:
    groups: dict[tuple[str, str, str, str, str], list[Mapping[str, object]]] = {}
    for row in feature_rows:
        if str(row.get("eligibility")) != SELECTION_ELIGIBLE:
            continue
        key = (
            str(row["fold_id"]),
            str(row["experiment_seed"]),
            str(row["target_domain"]),
            str(row["support_split_id"]),
            str(row["eval_split_id"]),
        )
        groups.setdefault(key, []).append(row)
    if not groups:
        raise ProtocolError("No selection-eligible feature rows available.")
    return groups


def _select_metadata_top1(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    return max(
        rows,
        key=lambda row: (
            _float_or_default(row.get("metadata_match"), 0.0),
            -_float_or_default(row.get("support_nelbo"), float("inf")),
            str(row.get("candidate_id", "")),
        ),
    )


def _select_support_nelbo_top1(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    return min(
        rows,
        key=lambda row: (
            _float_or_default(row.get("support_nelbo"), float("inf")),
            str(row.get("candidate_id", "")),
        ),
    )


def _select_deterministic_random(
    rows: Sequence[Mapping[str, object]],
    context_key: tuple[str, str, str, str, str],
) -> Mapping[str, object]:
    context = "|".join(context_key)
    return min(
        rows,
        key=lambda row: hashlib.sha256(f"{context}|{row.get('candidate_id', '')}".encode("utf-8")).hexdigest(),
    )


def _selection_from_row(row: Mapping[str, object], *, method: str) -> dict[str, object]:
    selection = {column: row[column] for column in REQUIRED_LINEAGE_COLUMNS}
    selection.update(
        {
            "method": method,
            "predicted_primary_utility": row.get("predicted_primary_utility", ""),
            "support_nelbo": row.get("support_nelbo", ""),
            "source_inner_stability": row.get("source_inner_stability", ""),
            "selection_rank": 1,
            "aggregation_weight": 1.0,
        }
    )
    return selection


def _float_or_default(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def write_selection_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ADOPTION_ELIGIBLE_SELECTION_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in ADOPTION_ELIGIBLE_SELECTION_COLUMNS})
