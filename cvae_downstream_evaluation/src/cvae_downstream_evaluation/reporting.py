"""Report contracts for thesis-facing downstream evaluation summaries."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

from .downstream import CandidateDownstreamRow, compute_single_expert_oracles, spearman
from .protocol import ProtocolError
from .routing import SupportSelectionUnit
from .schemas import (
    BASELINE_COMPARISON_COLUMNS,
    DECISION_CLASSIFICATIONS,
    ENSEMBLE_METHOD,
    METHODS_WITH_FULL_RANKING,
    METADATA_METHOD,
    METHOD_BASELINE_ROW_TYPE,
    PRIMARY_BUDGET_PER_CLASS,
    PRIMARY_GENERATION_MODE,
    ROUTING_ALIGNMENT_COLUMNS,
    STABILITY_COLUMNS,
    SUPPORT_NELBO_METHOD,
    SUPPORT_SIZE_SUMMARY_COLUMNS,
)

@dataclass(frozen=True)
class DecisionSummary:
    classification: str
    primary_method: str
    claim_boundary: str
    metrics: Mapping[str, float]


VALID_DECISIONS = DECISION_CLASSIFICATIONS


def assert_valid_decision(summary: DecisionSummary) -> None:
    if summary.classification not in VALID_DECISIONS:
        raise ValueError(f"Unknown decision classification: {summary.classification}")


def build_routing_alignment_rows(
    *,
    selections: Sequence[SupportSelectionUnit],
    downstream_rows: Sequence[CandidateDownstreamRow],
) -> list[dict[str, object]]:
    """Join support selections to candidate downstream utility rows."""

    single_rows = {
        (
            int(row.experiment_seed),
            row.heldout_center,
            row.candidate_expert,
            row.generation_mode,
            int(row.budget_per_class),
            int(row.generation_seed),
            int(row.classifier_seed),
        ): row
        for row in downstream_rows
        if row.row_type == "single_expert" and row.status == "ok"
    }
    oracles = compute_single_expert_oracles(downstream_rows)
    metadata_by_support = {
        _selection_key(unit): unit
        for unit in selections
        if unit.method == METADATA_METHOD
    }
    rows: list[dict[str, object]] = []
    contexts = sorted(
        {
            row.oracle_key()
            for row in downstream_rows
            if row.is_oracle_eligible()
            and int(row.budget_per_class) == PRIMARY_BUDGET_PER_CLASS
        }
    )
    for unit in selections:
        if unit.method == ENSEMBLE_METHOD:
            continue
        metadata_unit = metadata_by_support.get(_selection_key(unit))
        for context_key in contexts:
            experiment_seed, heldout, generation_mode, budget, generation_seed, classifier_seed = context_key
            if int(experiment_seed) != int(unit.experiment_seed) or heldout != unit.heldout_center:
                continue
            selected_key = (
                int(unit.experiment_seed),
                unit.heldout_center,
                unit.selected_expert,
                generation_mode,
                int(budget),
                int(generation_seed),
                int(classifier_seed),
            )
            selected = single_rows.get(selected_key)
            if selected is None:
                raise ProtocolError(f"Missing downstream row for selected expert key {selected_key}")
            oracle = oracles.get(context_key)
            if oracle is None:
                raise ProtocolError(f"Missing downstream oracle for key {context_key}")
            metadata_bacc = math.nan
            if metadata_unit is not None:
                metadata_key = (
                    int(metadata_unit.experiment_seed),
                    metadata_unit.heldout_center,
                    metadata_unit.selected_expert,
                    generation_mode,
                    int(budget),
                    int(generation_seed),
                    int(classifier_seed),
                )
                metadata_row = single_rows.get(metadata_key)
                if metadata_row is not None:
                    metadata_bacc = float(metadata_row.bacc)
            spearman_value = math.nan
            if unit.method in METHODS_WITH_FULL_RANKING:
                spearman_value = _spearman_neg_nelbo_vs_bacc(unit, single_rows, selected)
            oracle_gap_bacc = float(oracle.bacc) - float(selected.bacc)
            row = {
                "heldout_center": unit.heldout_center,
                "experiment_seed": unit.experiment_seed,
                "support_size": unit.support_size,
                "support_seed": unit.support_seed,
                "generation_seed": generation_seed,
                "classifier_seed": classifier_seed,
                "method": unit.method,
                "selected_expert": unit.selected_expert,
                "selected_bacc": float(selected.bacc),
                "selected_macro_f1": float(selected.macro_f1),
                "downstream_oracle_expert": oracle.expert,
                "oracle_bacc": float(oracle.bacc),
                "oracle_macro_f1": float(oracle.macro_f1),
                "downstream_oracle_gap_bacc": oracle_gap_bacc,
                "downstream_oracle_gap_macro_f1": float(oracle.macro_f1) - float(selected.macro_f1),
                "relative_downstream_oracle_gap_pct": _relative_gap_pct(oracle_gap_bacc, float(oracle.bacc)),
                "top1_downstream_hit": int(unit.selected_expert == oracle.expert),
                "spearman_neg_nelbo_vs_bacc": spearman_value,
                "metadata_bacc": metadata_bacc,
                "delta_vs_metadata": float(selected.bacc) - metadata_bacc if not math.isnan(metadata_bacc) else math.nan,
            }
            rows.append(row)
    return rows


def classify_decision(alignment_rows: Sequence[Mapping[str, object]]) -> DecisionSummary:
    """Classify PASS/WEAK_PASS/DIAGNOSTIC_ONLY/FAIL from primary rows."""

    primary = [
        row
        for row in alignment_rows
        if row.get("method") in {SUPPORT_NELBO_METHOD, METADATA_METHOD}
    ]
    if not primary:
        raise ProtocolError("No primary alignment rows available for decision.")

    support_rows = [row for row in alignment_rows if row.get("method") == SUPPORT_NELBO_METHOD]
    metadata_rows = [row for row in alignment_rows if row.get("method") == METADATA_METHOD]
    if not support_rows or not metadata_rows:
        raise ProtocolError("Decision requires both support-NELBO and metadata rows.")

    mean_delta = _nanmean(float(row["delta_vs_metadata"]) for row in support_rows)
    support_gap = _nanmean(float(row["downstream_oracle_gap_bacc"]) for row in support_rows)
    metadata_gap = _nanmean(float(row["downstream_oracle_gap_bacc"]) for row in metadata_rows)
    mean_spearman = _nanmean(float(row["spearman_neg_nelbo_vs_bacc"]) for row in support_rows)
    center_pass_count = sum(1 for value in center_passes(alignment_rows).values() if value)

    if mean_delta > 0.0 and support_gap < metadata_gap and mean_spearman > 0.0 and center_pass_count >= 4:
        classification = "PASS"
    elif mean_delta > 0.0 or support_gap < metadata_gap:
        classification = "WEAK_PASS"
    elif math.isnan(mean_delta) or math.isnan(support_gap):
        classification = "DIAGNOSTIC_ONLY"
    else:
        classification = "FAIL"
    return DecisionSummary(
        classification=classification,
        primary_method=SUPPORT_NELBO_METHOD,
        claim_boundary=(
            "Direct support-NELBO is evaluated for downstream utility transfer; "
            "lower NELBO alone is not treated as generative-quality evidence."
        ),
        metrics={
            "mean_delta_bacc_vs_metadata": mean_delta,
            "mean_support_nelbo_oracle_gap_bacc": support_gap,
            "mean_metadata_oracle_gap_bacc": metadata_gap,
            "mean_spearman_neg_nelbo_vs_bacc": mean_spearman,
            "center_pass_count": float(center_pass_count),
        },
    )


def center_passes(alignment_rows: Sequence[Mapping[str, object]]) -> dict[str, bool]:
    centers = sorted({str(row["heldout_center"]) for row in alignment_rows})
    result: dict[str, bool] = {}
    for center in centers:
        support_rows = [
            row for row in alignment_rows if str(row["heldout_center"]) == center and row.get("method") == SUPPORT_NELBO_METHOD
        ]
        metadata_rows = [
            row for row in alignment_rows if str(row["heldout_center"]) == center and row.get("method") == METADATA_METHOD
        ]
        if not support_rows or not metadata_rows:
            result[center] = False
            continue
        delta = _nanmean(float(row["delta_vs_metadata"]) for row in support_rows)
        support_gap = _nanmean(float(row["downstream_oracle_gap_bacc"]) for row in support_rows)
        metadata_gap = _nanmean(float(row["downstream_oracle_gap_bacc"]) for row in metadata_rows)
        result[center] = bool(delta >= 0.0 and support_gap <= metadata_gap)
    return result


def support_size_stratified_summary(alignment_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sizes = sorted({int(row["support_size"]) for row in alignment_rows})
    methods = sorted({str(row["method"]) for row in alignment_rows})
    for support_size in sizes:
        for method in methods:
            subset = [
                row for row in alignment_rows if int(row["support_size"]) == support_size and str(row["method"]) == method
            ]
            if not subset:
                continue
            rows.append(
                {
                    "support_size": support_size,
                    "method": method,
                    "mean_bacc": _nanmean(float(row["selected_bacc"]) for row in subset),
                    "mean_macro_f1": _nanmean(float(row["selected_macro_f1"]) for row in subset),
                    "mean_delta_bacc_vs_metadata": _nanmean(float(row["delta_vs_metadata"]) for row in subset),
                    "mean_downstream_oracle_gap_bacc": _nanmean(
                        float(row["downstream_oracle_gap_bacc"]) for row in subset
                    ),
                    "mean_spearman_neg_nelbo_vs_bacc": _nanmean(
                        float(row["spearman_neg_nelbo_vs_bacc"]) for row in subset
                    ),
                    "center_pass_count": _support_size_center_pass_count(
                        alignment_rows,
                        support_size=support_size,
                        method=method,
                    ),
                }
            )
    return rows


def baseline_comparison_rows(
    *,
    alignment_rows: Sequence[Mapping[str, object]],
    downstream_rows: Sequence[CandidateDownstreamRow],
) -> list[dict[str, object]]:
    """Summarize single-expert methods plus explicitly tagged method baselines."""

    rows: list[dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in alignment_rows}):
        subset = [row for row in alignment_rows if str(row["method"]) == method]
        rows.append(
            {
                "method": method,
                "row_type": "selection_method",
                "mean_bacc": _nanmean(float(row["selected_bacc"]) for row in subset),
                "mean_macro_f1": _nanmean(float(row["selected_macro_f1"]) for row in subset),
                "mean_delta_bacc_vs_metadata": _nanmean(float(row["delta_vs_metadata"]) for row in subset),
                "mean_downstream_oracle_gap_bacc": _nanmean(
                    float(row["downstream_oracle_gap_bacc"]) for row in subset
                ),
                "top1_downstream_hit_rate": _nanmean(float(row["top1_downstream_hit"]) for row in subset),
            }
        )

    ensemble_rows = [
        row
        for row in downstream_rows
        if row.row_type == METHOD_BASELINE_ROW_TYPE
        and row.generation_mode == PRIMARY_GENERATION_MODE
        and int(row.budget_per_class) == PRIMARY_BUDGET_PER_CLASS
        and row.status == "ok"
    ]
    if ensemble_rows:
        rows.append(
            {
                "method": ENSEMBLE_METHOD,
                "row_type": METHOD_BASELINE_ROW_TYPE,
                "mean_bacc": _nanmean(row.bacc for row in ensemble_rows),
                "mean_macro_f1": _nanmean(row.macro_f1 for row in ensemble_rows),
                "mean_delta_bacc_vs_metadata": math.nan,
                "mean_downstream_oracle_gap_bacc": math.nan,
                "top1_downstream_hit_rate": math.nan,
            }
        )
    return rows


def stability_rows(alignment_rows: Sequence[Mapping[str, object]], *, group: str) -> list[dict[str, object]]:
    """Build lightweight stability summaries for selection or generation seeds."""

    rows: list[dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in alignment_rows}):
        subset = [row for row in alignment_rows if str(row["method"]) == method]
        baccs = [float(row["selected_bacc"]) for row in subset if not math.isnan(float(row["selected_bacc"]))]
        center_means = [
            _nanmean(float(row["selected_bacc"]) for row in subset if str(row["heldout_center"]) == center)
            for center in sorted({str(row["heldout_center"]) for row in subset})
        ]
        rows.append(
            {
                "method": method,
                "group": group,
                "mean_bacc": _nanmean(baccs),
                "std_bacc": _std(baccs),
                "worst_center_bacc": min(center_means) if center_means else math.nan,
            }
        )
    return rows


def write_alignment_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, ROUTING_ALIGNMENT_COLUMNS, rows)


def write_support_size_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, SUPPORT_SIZE_SUMMARY_COLUMNS, rows)


def write_baseline_comparison_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, BASELINE_COMPARISON_COLUMNS, rows)


def write_stability_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, STABILITY_COLUMNS, rows)


def write_decision_summary(path: Path, summary: DecisionSummary) -> None:
    assert_valid_decision(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Downstream Decision Summary",
        "",
        f"Decision: `{summary.classification}`",
        "",
        "Primary PASS is based on the predefined all-support-size aggregation.",
        "Support-size-stratified results are descriptive and identify reliability regimes; they do not rescue or overturn the primary decision.",
        "",
        "## Metrics",
    ]
    for key, value in summary.metrics.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            summary.claim_boundary,
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _spearman_neg_nelbo_vs_bacc(
    unit: SupportSelectionUnit,
    rows_by_key: Mapping[tuple[int, str, str, str, int, int, int], CandidateDownstreamRow],
    downstream_context: CandidateDownstreamRow,
) -> float:
    neg_nelbo: list[float] = []
    baccs: list[float] = []
    for expert in unit.candidate_experts:
        if expert not in unit.support_nelbo_by_expert:
            continue
        key = (
            int(unit.experiment_seed),
            unit.heldout_center,
            expert,
            downstream_context.generation_mode,
            int(downstream_context.budget_per_class),
            int(downstream_context.generation_seed),
            int(downstream_context.classifier_seed),
        )
        row = rows_by_key.get(key)
        if row is None:
            continue
        neg_nelbo.append(-float(unit.support_nelbo_by_expert[expert]))
        baccs.append(float(row.bacc))
    return spearman(neg_nelbo, baccs) if len(neg_nelbo) >= 2 else math.nan


def _selection_key(unit: SupportSelectionUnit) -> tuple[str, int, int, int, str]:
    return (
        unit.heldout_center,
        int(unit.experiment_seed),
        int(unit.support_size),
        int(unit.support_seed),
        unit.support_eval_split_id,
    )


def _relative_gap_pct(gap: float, oracle: float) -> float:
    denom = max(abs(float(oracle)), 1e-12)
    return 100.0 * float(gap) / denom


def _nanmean(values: Sequence[float] | object) -> float:
    cleaned = [float(v) for v in values if not math.isnan(float(v))]
    return float(mean(cleaned)) if cleaned else math.nan


def _std(values: Sequence[float]) -> float:
    cleaned = [float(v) for v in values if not math.isnan(float(v))]
    if len(cleaned) < 2:
        return 0.0 if cleaned else math.nan
    avg = sum(cleaned) / float(len(cleaned))
    return math.sqrt(sum((v - avg) ** 2 for v in cleaned) / float(len(cleaned) - 1))


def _support_size_center_pass_count(
    rows: Sequence[Mapping[str, object]],
    *,
    support_size: int,
    method: str,
) -> float:
    if method != SUPPORT_NELBO_METHOD:
        return math.nan
    count = 0
    centers = sorted({str(row["heldout_center"]) for row in rows if int(row["support_size"]) == support_size})
    for center in centers:
        support_rows = [
            row
            for row in rows
            if int(row["support_size"]) == support_size
            and str(row["heldout_center"]) == center
            and row.get("method") == SUPPORT_NELBO_METHOD
        ]
        metadata_rows = [
            row
            for row in rows
            if int(row["support_size"]) == support_size
            and str(row["heldout_center"]) == center
            and row.get("method") == METADATA_METHOD
        ]
        if not support_rows or not metadata_rows:
            continue
        delta = _nanmean(float(row["delta_vs_metadata"]) for row in support_rows)
        support_gap = _nanmean(float(row["downstream_oracle_gap_bacc"]) for row in support_rows)
        metadata_gap = _nanmean(float(row["downstream_oracle_gap_bacc"]) for row in metadata_rows)
        if delta >= 0.0 and support_gap <= metadata_gap:
            count += 1
    return float(count)


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
