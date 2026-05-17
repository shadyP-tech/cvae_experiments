"""Post-hoc source-global gated support-NELBO routing reports.

This module consumes frozen v1 downstream artifacts only. It derives a
deployment-risk gate over existing selectors; it does not train CVAEs,
regenerate embeddings, or rerun downstream classifiers.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

from .downstream import (
    CandidateDownstreamRow,
    assert_matrix_schema,
    compute_single_expert_oracles,
    read_candidate_downstream_matrix,
)
from .protocol import ProtocolError
from .reporting import build_routing_alignment_rows
from .routing import SupportSelectionUnit, support_units_from_csv
from .schemas import (
    MATRIX_SCHEMA_VERSION,
    METADATA_METHOD,
    PRIMARY_BUDGET_PER_CLASS,
    PRIMARY_GENERATION_MODE,
    SINGLE_EXPERT_ROW_TYPE,
    SOURCE_GLOBAL_GATED_METHOD_PREFIX,
    SOURCE_GLOBAL_METHOD,
    SUPPORT_NELBO_METHOD,
)


PRIMARY_TAU = 0.10
DIAGNOSTIC_TAUS = (0.00, 0.05, 0.10, 0.20, 0.30)
EPS = 1e-12

GATED_ROUTING_COLUMNS = (
    "heldout_center",
    "experiment_seed",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "tau",
    "method",
    "best_expert",
    "global_expert",
    "selected_expert",
    "best_score",
    "global_score",
    "score_min",
    "score_max",
    "score_range",
    "normalized_gain_vs_global",
    "same_as_global",
    "eligible_switch",
    "switched_from_global",
    "support_nelbo_rank_of_global",
    "support_nelbo_rank_of_selected",
    "candidate_experts",
    "target_expert_excluded",
)

GATED_ALIGNMENT_COLUMNS = GATED_ROUTING_COLUMNS + (
    "generation_seed",
    "classifier_seed",
    "selected_bacc",
    "selected_macro_f1",
    "downstream_oracle_expert",
    "oracle_bacc",
    "oracle_macro_f1",
    "downstream_oracle_gap_bacc",
    "downstream_oracle_gap_macro_f1",
    "relative_downstream_oracle_gap_pct",
    "top1_downstream_hit",
)

GATED_COMPARISON_COLUMNS = (
    "method",
    "tau",
    "mean_bacc",
    "mean_macro_f1",
    "mean_delta_bacc_vs_metadata",
    "mean_delta_bacc_vs_support_nelbo",
    "mean_delta_bacc_vs_source_global",
    "mean_downstream_oracle_gap_bacc",
    "mean_delta_oracle_gap_vs_metadata",
    "mean_delta_oracle_gap_vs_support_nelbo",
    "mean_delta_oracle_gap_vs_source_global",
    "top1_downstream_hit_rate",
    "same_as_global_rate",
    "eligible_switch_rate",
    "actual_switch_rate",
    "center_pass_vs_source_global_count",
)


@dataclass(frozen=True)
class GatedRoutingUnit:
    heldout_center: str
    experiment_seed: int
    support_size: int
    support_seed: int
    support_eval_split_id: str
    tau: float
    method: str
    best_expert: str
    global_expert: str
    selected_expert: str
    best_score: float
    global_score: float
    score_min: float
    score_max: float
    score_range: float
    normalized_gain_vs_global: float
    same_as_global: bool
    eligible_switch: bool
    switched_from_global: bool
    support_nelbo_rank_of_global: int
    support_nelbo_rank_of_selected: int
    candidate_experts: tuple[str, ...]
    target_expert_excluded: bool
    support_nelbo_by_expert: Mapping[str, float]

    def to_csv_row(self) -> dict[str, object]:
        return {
            "heldout_center": self.heldout_center,
            "experiment_seed": self.experiment_seed,
            "support_size": self.support_size,
            "support_seed": self.support_seed,
            "support_eval_split_id": self.support_eval_split_id,
            "tau": self.tau,
            "method": self.method,
            "best_expert": self.best_expert,
            "global_expert": self.global_expert,
            "selected_expert": self.selected_expert,
            "best_score": self.best_score,
            "global_score": self.global_score,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "score_range": self.score_range,
            "normalized_gain_vs_global": self.normalized_gain_vs_global,
            "same_as_global": str(self.same_as_global).lower(),
            "eligible_switch": str(self.eligible_switch).lower(),
            "switched_from_global": str(self.switched_from_global).lower(),
            "support_nelbo_rank_of_global": self.support_nelbo_rank_of_global,
            "support_nelbo_rank_of_selected": self.support_nelbo_rank_of_selected,
            "candidate_experts": "|".join(self.candidate_experts),
            "target_expert_excluded": str(self.target_expert_excluded).lower(),
        }

    def as_selection_unit(self) -> SupportSelectionUnit:
        return SupportSelectionUnit(
            heldout_center=self.heldout_center,
            experiment_seed=self.experiment_seed,
            support_size=self.support_size,
            support_seed=self.support_seed,
            method=self.method,
            selected_expert=self.selected_expert,
            candidate_experts=self.candidate_experts,
            support_nelbo_by_expert=dict(self.support_nelbo_by_expert),
            target_expert_excluded=self.target_expert_excluded,
            support_eval_split_id=self.support_eval_split_id,
        )


def gated_method_name(tau: float) -> str:
    return f"{SOURCE_GLOBAL_GATED_METHOD_PREFIX}_tau{int(round(float(tau) * 100.0)):02d}"


def derive_source_global_gated_units(
    selections: Sequence[SupportSelectionUnit],
    *,
    taus: Sequence[float] = DIAGNOSTIC_TAUS,
) -> list[GatedRoutingUnit]:
    """Derive gated source-global fallback units from existing support units."""

    by_key_method: dict[tuple[object, ...], SupportSelectionUnit] = {}
    for unit in selections:
        if unit.method not in {SUPPORT_NELBO_METHOD, SOURCE_GLOBAL_METHOD}:
            continue
        key = _selection_method_key(unit)
        if key in by_key_method:
            raise ProtocolError(f"Duplicate support selection unit for {key}")
        by_key_method[key] = unit

    support_units = [unit for unit in selections if unit.method == SUPPORT_NELBO_METHOD]
    if not support_units:
        raise ProtocolError("No support-NELBO units available for gated routing.")

    gated: list[GatedRoutingUnit] = []
    for support_unit in sorted(support_units, key=_selection_sort_key):
        global_unit = by_key_method.get(
            _selection_method_key(support_unit, method=SOURCE_GLOBAL_METHOD)
        )
        if global_unit is None:
            raise ProtocolError(f"Missing source-global unit for {_selection_key(support_unit)}")
        for tau in taus:
            gated.append(_derive_one_gated_unit(support_unit, global_unit, float(tau)))
    return gated


def build_source_global_gated_alignment_rows(
    *,
    gated_units: Sequence[GatedRoutingUnit],
    downstream_rows: Sequence[CandidateDownstreamRow],
) -> list[dict[str, object]]:
    """Join gated selections to primary single-expert downstream rows."""

    matrix_by_key = {
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
        if row.row_type == SINGLE_EXPERT_ROW_TYPE
        and row.status == "ok"
        and row.generation_mode == PRIMARY_GENERATION_MODE
        and int(row.budget_per_class) == PRIMARY_BUDGET_PER_CLASS
    }
    oracles = compute_single_expert_oracles(downstream_rows)
    contexts = sorted(
        {
            row.oracle_key()
            for row in downstream_rows
            if row.is_oracle_eligible()
            and row.generation_mode == PRIMARY_GENERATION_MODE
            and int(row.budget_per_class) == PRIMARY_BUDGET_PER_CLASS
        }
    )

    rows: list[dict[str, object]] = []
    for unit in gated_units:
        for context_key in contexts:
            experiment_seed, heldout, generation_mode, budget, generation_seed, classifier_seed = context_key
            if int(experiment_seed) != int(unit.experiment_seed) or heldout != unit.heldout_center:
                continue
            matrix_key = (
                int(unit.experiment_seed),
                unit.heldout_center,
                unit.selected_expert,
                generation_mode,
                int(budget),
                int(generation_seed),
                int(classifier_seed),
            )
            selected = matrix_by_key.get(matrix_key)
            if selected is None:
                raise ProtocolError(
                    "Missing downstream matrix row for gated selection; expected "
                    f"selected_expert={unit.selected_expert!r} to match matrix.candidate_expert "
                    f"with key {matrix_key}."
                )
            oracle = oracles.get(context_key)
            if oracle is None:
                raise ProtocolError(f"Missing downstream oracle for key {context_key}")
            oracle_gap_bacc = float(oracle.bacc) - float(selected.bacc)
            row = unit.to_csv_row()
            row.update(
                {
                    "generation_seed": generation_seed,
                    "classifier_seed": classifier_seed,
                    "selected_bacc": float(selected.bacc),
                    "selected_macro_f1": float(selected.macro_f1),
                    "downstream_oracle_expert": oracle.expert,
                    "oracle_bacc": float(oracle.bacc),
                    "oracle_macro_f1": float(oracle.macro_f1),
                    "downstream_oracle_gap_bacc": oracle_gap_bacc,
                    "downstream_oracle_gap_macro_f1": float(oracle.macro_f1) - float(selected.macro_f1),
                    "relative_downstream_oracle_gap_pct": _relative_gap_pct(oracle_gap_bacc, float(oracle.bacc)),
                    "top1_downstream_hit": int(unit.selected_expert == oracle.expert),
                }
            )
            rows.append(row)
    return rows


def source_global_gated_comparison_rows(
    *,
    gated_alignment_rows: Sequence[Mapping[str, object]],
    baseline_alignment_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Compute paired gated-vs-baseline deltas before equal-center aggregation."""

    baseline_by_key = {
        (str(row["method"]), _comparison_key(row)): row
        for row in baseline_alignment_rows
        if str(row.get("method")) in {METADATA_METHOD, SUPPORT_NELBO_METHOD, SOURCE_GLOBAL_METHOD}
    }
    paired_rows: list[dict[str, object]] = []
    for row in gated_alignment_rows:
        context_key = _comparison_key(row)
        metadata = _require_baseline(baseline_by_key, METADATA_METHOD, context_key)
        support = _require_baseline(baseline_by_key, SUPPORT_NELBO_METHOD, context_key)
        source_global = _require_baseline(baseline_by_key, SOURCE_GLOBAL_METHOD, context_key)
        gated_gap = float(row["downstream_oracle_gap_bacc"])
        paired_rows.append(
            {
                **row,
                "delta_bacc_vs_metadata": float(row["selected_bacc"]) - float(metadata["selected_bacc"]),
                "delta_bacc_vs_support_nelbo": float(row["selected_bacc"]) - float(support["selected_bacc"]),
                "delta_bacc_vs_source_global": float(row["selected_bacc"]) - float(source_global["selected_bacc"]),
                "delta_oracle_gap_vs_metadata": float(metadata["downstream_oracle_gap_bacc"]) - gated_gap,
                "delta_oracle_gap_vs_support_nelbo": float(support["downstream_oracle_gap_bacc"]) - gated_gap,
                "delta_oracle_gap_vs_source_global": float(source_global["downstream_oracle_gap_bacc"]) - gated_gap,
            }
        )

    summaries: list[dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in paired_rows}):
        method_rows = [row for row in paired_rows if str(row["method"]) == method]
        tau_values = sorted({float(row["tau"]) for row in method_rows})
        for tau in tau_values:
            subset = [row for row in method_rows if float(row["tau"]) == tau]
            summaries.append(_aggregate_paired_rows(method=method, tau=tau, rows=subset))
    return summaries


def classify_source_global_gated_decision(
    comparison_rows: Sequence[Mapping[str, object]],
    *,
    primary_tau: float = PRIMARY_TAU,
) -> tuple[str, Mapping[str, float]]:
    primary_method = gated_method_name(primary_tau)
    rows = [
        row
        for row in comparison_rows
        if str(row.get("method")) == primary_method and abs(float(row.get("tau", math.nan)) - primary_tau) <= EPS
    ]
    if len(rows) != 1:
        raise ProtocolError(f"Expected one primary gated comparison row for {primary_method}, got {len(rows)}.")
    row = rows[0]
    metrics = {
        "mean_bacc": float(row["mean_bacc"]),
        "mean_macro_f1": float(row["mean_macro_f1"]),
        "mean_delta_bacc_vs_support_nelbo": float(row["mean_delta_bacc_vs_support_nelbo"]),
        "mean_delta_bacc_vs_source_global": float(row["mean_delta_bacc_vs_source_global"]),
        "mean_downstream_oracle_gap_bacc": float(row["mean_downstream_oracle_gap_bacc"]),
        "mean_delta_oracle_gap_vs_support_nelbo": float(row["mean_delta_oracle_gap_vs_support_nelbo"]),
        "mean_delta_oracle_gap_vs_source_global": float(row["mean_delta_oracle_gap_vs_source_global"]),
        "center_pass_vs_source_global_count": float(row["center_pass_vs_source_global_count"]),
        "same_as_global_rate": float(row["same_as_global_rate"]),
        "eligible_switch_rate": float(row["eligible_switch_rate"]),
        "actual_switch_rate": float(row["actual_switch_rate"]),
    }
    pass_gates = (
        metrics["mean_delta_bacc_vs_support_nelbo"] > 0.0
        and metrics["mean_delta_bacc_vs_source_global"] > 0.0
        and metrics["mean_delta_oracle_gap_vs_support_nelbo"] > 0.0
        and metrics["mean_delta_oracle_gap_vs_source_global"] > 0.0
        and metrics["center_pass_vs_source_global_count"] >= 4.0
    )
    improves_any = any(
        metrics[key] > 0.0
        for key in (
            "mean_delta_bacc_vs_support_nelbo",
            "mean_delta_bacc_vs_source_global",
            "mean_delta_oracle_gap_vs_support_nelbo",
            "mean_delta_oracle_gap_vs_source_global",
        )
    )
    worse_than_both = (
        metrics["mean_delta_bacc_vs_support_nelbo"] < 0.0
        and metrics["mean_delta_bacc_vs_source_global"] < 0.0
        and metrics["mean_delta_oracle_gap_vs_support_nelbo"] < 0.0
        and metrics["mean_delta_oracle_gap_vs_source_global"] < 0.0
    )
    if pass_gates:
        classification = "PASS"
    elif improves_any:
        classification = "WEAK_PASS"
    elif worse_than_both:
        classification = "FAIL"
    else:
        classification = "DIAGNOSTIC_ONLY"
    return classification, metrics


def build_source_global_gated_report(artifacts_root: Path) -> dict[str, Path]:
    """Build all source-global gated v2 artifacts from completed v1 outputs."""

    artifacts_root = Path(artifacts_root)
    tables = artifacts_root / "tables"
    reports = artifacts_root / "reports"
    support_path = tables / "support_selection_units.csv"
    matrix_path = tables / "all_expert_downstream_matrix.csv"

    assert_matrix_schema(matrix_path)
    selections = support_units_from_csv(support_path)
    downstream_rows = read_candidate_downstream_matrix(matrix_path)
    gated_units = derive_source_global_gated_units(selections, taus=DIAGNOSTIC_TAUS)
    gated_alignment = build_source_global_gated_alignment_rows(
        gated_units=gated_units,
        downstream_rows=downstream_rows,
    )
    baseline_alignment = build_routing_alignment_rows(
        selections=selections,
        downstream_rows=downstream_rows,
    )
    comparison = source_global_gated_comparison_rows(
        gated_alignment_rows=gated_alignment,
        baseline_alignment_rows=baseline_alignment,
    )

    paths = {
        "routing_units": tables / "source_global_gated_routing_units.csv",
        "alignment": tables / "source_global_gated_alignment.csv",
        "comparison": tables / "source_global_gated_comparison.csv",
        "threshold_sensitivity": tables / "source_global_gated_threshold_sensitivity.csv",
        "decision_summary": reports / "source_global_gated_decision_summary.md",
        "provenance": reports / "source_global_gated_provenance.json",
    }
    _write_csv(paths["routing_units"], GATED_ROUTING_COLUMNS, [unit.to_csv_row() for unit in gated_units])
    _write_csv(paths["alignment"], GATED_ALIGNMENT_COLUMNS, gated_alignment)
    _write_csv(paths["comparison"], GATED_COMPARISON_COLUMNS, comparison)
    _write_csv(paths["threshold_sensitivity"], GATED_COMPARISON_COLUMNS, comparison)
    _write_decision_summary(paths["decision_summary"], comparison)
    _write_provenance(
        paths["provenance"],
        matrix_path=matrix_path,
        support_path=support_path,
        primary_tau=PRIMARY_TAU,
        diagnostic_taus=DIAGNOSTIC_TAUS,
    )
    return paths


def _derive_one_gated_unit(
    support_unit: SupportSelectionUnit,
    global_unit: SupportSelectionUnit,
    tau: float,
) -> GatedRoutingUnit:
    scores = {str(k): float(v) for k, v in support_unit.support_nelbo_by_expert.items()}
    best_expert = str(support_unit.selected_expert)
    global_expert = str(global_unit.selected_expert)
    if best_expert not in scores:
        raise ProtocolError(f"Support-NELBO selected expert {best_expert!r} is absent from support scores.")
    if global_expert not in scores:
        raise ProtocolError(
            f"Source-global expert {global_expert!r} is absent from support-NELBO scores "
            f"for {_selection_key(support_unit)}."
        )
    if global_expert not in support_unit.candidate_experts:
        raise ProtocolError(
            f"Source-global expert {global_expert!r} is absent from candidate experts "
            f"{support_unit.candidate_experts} for {_selection_key(support_unit)}."
        )

    values = list(scores.values())
    score_min = min(values)
    score_max = max(values)
    score_range = score_max - score_min
    normalized_gain = 0.0 if score_range <= EPS else (scores[global_expert] - scores[best_expert]) / score_range
    same_as_global = best_expert == global_expert
    eligible_switch = not same_as_global
    switched = bool(eligible_switch and score_range > EPS and normalized_gain >= float(tau))
    selected = best_expert if switched else global_expert
    ranks = _support_score_ranks(scores)
    return GatedRoutingUnit(
        heldout_center=support_unit.heldout_center,
        experiment_seed=support_unit.experiment_seed,
        support_size=support_unit.support_size,
        support_seed=support_unit.support_seed,
        support_eval_split_id=support_unit.support_eval_split_id,
        tau=float(tau),
        method=gated_method_name(float(tau)),
        best_expert=best_expert,
        global_expert=global_expert,
        selected_expert=selected,
        best_score=float(scores[best_expert]),
        global_score=float(scores[global_expert]),
        score_min=float(score_min),
        score_max=float(score_max),
        score_range=float(score_range),
        normalized_gain_vs_global=float(normalized_gain),
        same_as_global=same_as_global,
        eligible_switch=eligible_switch,
        switched_from_global=switched,
        support_nelbo_rank_of_global=int(ranks[global_expert]),
        support_nelbo_rank_of_selected=int(ranks[selected]),
        candidate_experts=tuple(support_unit.candidate_experts),
        target_expert_excluded=bool(support_unit.target_expert_excluded and global_unit.target_expert_excluded),
        support_nelbo_by_expert=scores,
    )


def _aggregate_paired_rows(*, method: str, tau: float, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    centers = sorted({str(row["heldout_center"]) for row in rows})
    center_metrics: dict[str, dict[str, float]] = {}
    for center in centers:
        subset = [row for row in rows if str(row["heldout_center"]) == center]
        center_metrics[center] = {
            "mean_bacc": _nanmean(float(row["selected_bacc"]) for row in subset),
            "mean_macro_f1": _nanmean(float(row["selected_macro_f1"]) for row in subset),
            "mean_delta_bacc_vs_metadata": _nanmean(float(row["delta_bacc_vs_metadata"]) for row in subset),
            "mean_delta_bacc_vs_support_nelbo": _nanmean(float(row["delta_bacc_vs_support_nelbo"]) for row in subset),
            "mean_delta_bacc_vs_source_global": _nanmean(float(row["delta_bacc_vs_source_global"]) for row in subset),
            "mean_downstream_oracle_gap_bacc": _nanmean(float(row["downstream_oracle_gap_bacc"]) for row in subset),
            "mean_delta_oracle_gap_vs_metadata": _nanmean(float(row["delta_oracle_gap_vs_metadata"]) for row in subset),
            "mean_delta_oracle_gap_vs_support_nelbo": _nanmean(
                float(row["delta_oracle_gap_vs_support_nelbo"]) for row in subset
            ),
            "mean_delta_oracle_gap_vs_source_global": _nanmean(
                float(row["delta_oracle_gap_vs_source_global"]) for row in subset
            ),
            "top1_downstream_hit_rate": _nanmean(float(row["top1_downstream_hit"]) for row in subset),
            "same_as_global_rate": _nanmean(_bool_float(row["same_as_global"]) for row in subset),
            "eligible_switch_rate": _nanmean(_bool_float(row["eligible_switch"]) for row in subset),
            "actual_switch_rate": _nanmean(_bool_float(row["switched_from_global"]) for row in subset),
        }
    center_pass_count = sum(
        1
        for values in center_metrics.values()
        if values["mean_delta_bacc_vs_source_global"] >= 0.0
        and values["mean_delta_oracle_gap_vs_source_global"] >= 0.0
    )
    return {
        "method": method,
        "tau": tau,
        **{
            key: _nanmean(values[key] for values in center_metrics.values())
            for key in (
                "mean_bacc",
                "mean_macro_f1",
                "mean_delta_bacc_vs_metadata",
                "mean_delta_bacc_vs_support_nelbo",
                "mean_delta_bacc_vs_source_global",
                "mean_downstream_oracle_gap_bacc",
                "mean_delta_oracle_gap_vs_metadata",
                "mean_delta_oracle_gap_vs_support_nelbo",
                "mean_delta_oracle_gap_vs_source_global",
                "top1_downstream_hit_rate",
                "same_as_global_rate",
                "eligible_switch_rate",
                "actual_switch_rate",
            )
        },
        "center_pass_vs_source_global_count": float(center_pass_count),
    }


def _write_decision_summary(path: Path, comparison_rows: Sequence[Mapping[str, object]]) -> None:
    decision, metrics = classify_source_global_gated_decision(comparison_rows, primary_tau=PRIMARY_TAU)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Source-Global Gated Router Decision Summary",
        "",
        f"Decision: `{decision}`",
        "",
        f"Primary method: `{gated_method_name(PRIMARY_TAU)}`",
        f"Primary tau: `{PRIMARY_TAU}`",
        "",
        "Diagnostic thresholds are descriptive only. This report does not select or recommend a best tau from target downstream performance.",
        "",
        "## Metrics",
    ]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            (
                "This method is a deployment gate over existing selectors, not a better compatibility estimator. "
                "It tests whether a source-global fallback gate reduces harmful support-NELBO switches while "
                "preserving useful high-confidence switches."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_provenance(
    path: Path,
    *,
    matrix_path: Path,
    support_path: Path,
    primary_tau: float,
    diagnostic_taus: Sequence[float],
) -> None:
    schema_path = matrix_path.with_suffix(".schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    payload = {
        "source_matrix_schema_version": schema.get("schema_version", MATRIX_SCHEMA_VERSION),
        "source_matrix_path": str(matrix_path),
        "support_selection_units_path": str(support_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_tau": float(primary_tau),
        "diagnostic_taus": [float(value) for value in diagnostic_taus],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _support_score_ranks(scores: Mapping[str, float]) -> dict[str, int]:
    ordered = sorted(scores.items(), key=lambda item: (float(item[1]), str(item[0])))
    return {str(expert): rank for rank, (expert, _score) in enumerate(ordered, start=1)}


def _selection_key(unit: SupportSelectionUnit) -> tuple[str, int, int, int, str]:
    return (
        unit.heldout_center,
        int(unit.experiment_seed),
        int(unit.support_size),
        int(unit.support_seed),
        unit.support_eval_split_id,
    )


def _selection_method_key(unit: SupportSelectionUnit, *, method: str | None = None) -> tuple[object, ...]:
    return (*_selection_key(unit), method or unit.method)


def _selection_sort_key(unit: SupportSelectionUnit) -> tuple[object, ...]:
    return _selection_key(unit)


def _comparison_key(row: Mapping[str, object]) -> tuple[str, int, int, int, int, int]:
    return (
        str(row["heldout_center"]),
        int(row["experiment_seed"]),
        int(row["support_size"]),
        int(row["support_seed"]),
        int(row["generation_seed"]),
        int(row["classifier_seed"]),
    )


def _require_baseline(
    rows: Mapping[tuple[str, tuple[str, int, int, int, int, int]], Mapping[str, object]],
    method: str,
    context_key: tuple[str, int, int, int, int, int],
) -> Mapping[str, object]:
    row = rows.get((method, context_key))
    if row is None:
        raise ProtocolError(f"Missing paired baseline row for method={method}, context={context_key}")
    return row


def _relative_gap_pct(gap: float, oracle: float) -> float:
    denom = max(abs(float(oracle)), EPS)
    return 100.0 * float(gap) / denom


def _nanmean(values: Sequence[float] | object) -> float:
    cleaned = [float(value) for value in values if not math.isnan(float(value))]
    return float(mean(cleaned)) if cleaned else math.nan


def _bool_float(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return 1.0 if str(value).strip().lower() in {"1", "true", "yes", "y"} else 0.0
