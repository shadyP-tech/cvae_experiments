"""Build MIDOG++ phase-2 downstream reports from frozen routing decisions.

This script does not run classifiers or change routing decisions. It joins the
phase-2 freeze artifacts to an already-materialized diagnostic downstream
matrix and writes post-freeze diagnostic summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.artifacts import stable_hash  # noqa: E402
from cvae_downstream_evaluation.artifacts.midogpp_phase2 import validate_phase2_preflight_freeze  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


DEFAULT_PHASE2_ROOT = (
    "cvae_downstream_evaluation/artifacts/midogpp/phase2_target_support_adaptation_virchow2_seed42"
)
DEFAULT_PHASE1_MATRIX = (
    "cvae_downstream_evaluation/artifacts/midogpp/phase1_virchow2_late_import_seed42/"
    "tables/diagnostic_downstream_utility.csv"
)
DEFAULT_CANDIDATE_METHOD = "dense_late_all_sources_reliability_shrink050_geom"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build MIDOG++ phase-2 downstream alignment reports from frozen selections."
    )
    parser.add_argument("--phase2-root", default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--phase1-matrix", default=DEFAULT_PHASE1_MATRIX)
    parser.add_argument("--candidate-method", default=DEFAULT_CANDIDATE_METHOD)
    parser.add_argument("--experiment-seed", default="42")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate joins and print report counts without writing outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    phase2_root = Path(args.phase2_root)
    phase1_matrix = Path(args.phase1_matrix)
    candidate_method = str(args.candidate_method)
    experiment_seed = str(args.experiment_seed)

    preflight = validate_phase2_preflight_freeze(phase2_root)
    if preflight.get("status") != "PASS":
        raise ProtocolError(f"Phase-2 preflight freeze is not PASS: {preflight}")

    selected_rows = _read_csv(phase2_root / "tables" / "selected_sources.csv")
    decision_rows = _read_csv(phase2_root / "tables" / "routing_decisions.csv")
    support_score_rows = _read_csv(phase2_root / "tables" / "support_score_matrix.csv")
    downstream_rows = _filter_downstream_rows(
        _read_csv(phase1_matrix),
        candidate_method=candidate_method,
        experiment_seed=experiment_seed,
    )
    single_rows = [row for row in downstream_rows if row.get("row_type") == "single_source"]
    baseline_rows = [row for row in downstream_rows if row.get("row_type") == "method_baseline"]
    if not single_rows:
        raise ProtocolError(f"No single-source downstream rows for method={candidate_method!r}.")

    alignment_rows = _alignment_rows(
        selected_rows=selected_rows,
        decision_rows=decision_rows,
        support_score_rows=support_score_rows,
        single_rows=single_rows,
        candidate_method=candidate_method,
    )
    oracle_gap_rows = _selected_vs_oracle_gap_rows(alignment_rows)
    baseline_comparison = _baseline_comparison_rows(alignment_rows, baseline_rows)
    support_summary = _summary_rows(alignment_rows, group_keys=("support_seed",))
    heldout_summary = _summary_rows(alignment_rows, group_keys=("heldout_center",))
    report = _validation_report(
        phase2_root=phase2_root,
        phase1_matrix=phase1_matrix,
        candidate_method=candidate_method,
        preflight=preflight,
        downstream_rows=downstream_rows,
        alignment_rows=alignment_rows,
        oracle_gap_rows=oracle_gap_rows,
        baseline_rows=baseline_rows,
    )
    summary_md = _decision_summary(report, alignment_rows, baseline_comparison)

    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    _write_csv(phase2_root / "tables" / "diagnostic_downstream_utility.csv", downstream_rows)
    _write_csv(phase2_root / "tables" / "routing_to_downstream_alignment.csv", alignment_rows)
    _write_csv(phase2_root / "tables" / "selected_vs_oracle_gap.csv", oracle_gap_rows)
    _write_csv(phase2_root / "tables" / "baseline_comparison.csv", baseline_comparison)
    _write_csv(phase2_root / "tables" / "support_seed_summary.csv", support_summary)
    _write_csv(phase2_root / "tables" / "heldout_center_summary.csv", heldout_summary)
    _write_json(phase2_root / "reports" / "phase2_validation_report.json", report)
    summary_path = phase2_root / "reports" / "decision_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_md, encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def _filter_downstream_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    candidate_method: str,
    experiment_seed: str,
) -> list[dict[str, str]]:
    out = [
        dict(row)
        for row in rows
        if str(row.get("candidate_method", "")) == candidate_method
        and str(row.get("experiment_seed", "")) == experiment_seed
        and str(row.get("status", "")) == "ok"
    ]
    if not out:
        raise ProtocolError(f"No downstream rows found for method={candidate_method!r}, seed={experiment_seed!r}.")
    return out


def _alignment_rows(
    *,
    selected_rows: Sequence[Mapping[str, str]],
    decision_rows: Sequence[Mapping[str, str]],
    support_score_rows: Sequence[Mapping[str, str]],
    single_rows: Sequence[Mapping[str, str]],
    candidate_method: str,
) -> list[dict[str, object]]:
    downstream_by_key = {
        _downstream_candidate_key(row): row
        for row in single_rows
    }
    oracle_by_context = _oracle_by_context(single_rows)
    decision_by_context = {
        _selection_context_key(row): row
        for row in decision_rows
    }
    support_scores_by_context = defaultdict(list)
    for row in support_score_rows:
        support_scores_by_context[_selection_context_key(row)].append(row)

    rows: list[dict[str, object]] = []
    for selection in selected_rows:
        context_key = _selection_context_key(selection)
        heldout = str(selection["heldout_center"])
        selected_source = str(selection["selected_source_center"])
        decision = decision_by_context.get(context_key)
        if decision is None:
            raise ProtocolError(f"Missing routing decision for selected source context: {context_key}")
        support_scores = support_scores_by_context.get(context_key, [])
        if not support_scores:
            raise ProtocolError(f"Missing support scores for context: {context_key}")
        selected_support = _selected_support_row(support_scores, selected_source)

        downstream_contexts = sorted({
            _downstream_context_key(row)
            for row in single_rows
            if str(row.get("heldout_center", "")) == heldout
        })
        if not downstream_contexts:
            raise ProtocolError(f"No downstream contexts for heldout={heldout!r}.")
        for downstream_context in downstream_contexts:
            selected = downstream_by_key.get(downstream_context + (selected_source,))
            if selected is None:
                raise ProtocolError(
                    f"Missing selected downstream row: context={downstream_context}, source={selected_source}"
                )
            oracle = oracle_by_context.get(downstream_context)
            if oracle is None:
                raise ProtocolError(f"Missing downstream oracle for context={downstream_context}.")
            rows.append(
                {
                    "schema_version": "midogpp_phase2_downstream_alignment_v1",
                    "heldout_center": heldout,
                    "support_seed": str(selection["support_seed"]),
                    "replicate": str(selection["replicate"]),
                    "support_split_id": str(selection["support_split_id"]),
                    "candidate_method": candidate_method,
                    "selected_candidate_id": str(selection["selected_candidate_id"]),
                    "selected_source_center": selected_source,
                    "selected_support_score": _float(decision["selected_score"]),
                    "support_candidate_rank": _support_rank(support_scores, selected_source),
                    "downstream_replicate_seed": str(selected["replicate_seed"]),
                    "generation_seed": str(selected["generation_seed"]),
                    "classifier_seed": str(selected["classifier_seed"]),
                    "eval_set_id": str(selected["eval_set_id"]),
                    "selected_downstream_candidate_id": str(selected["candidate_id"]),
                    "selected_bacc": _float(selected["bacc"]),
                    "selected_macro_f1": _float(selected["macro_f1"]),
                    "oracle_source_center": str(oracle["candidate_source_center"]),
                    "oracle_candidate_id": str(oracle["candidate_id"]),
                    "oracle_bacc": _float(oracle["bacc"]),
                    "oracle_macro_f1": _float(oracle["macro_f1"]),
                    "downstream_oracle_gap_bacc": _float(oracle["bacc"]) - _float(selected["bacc"]),
                    "downstream_oracle_gap_macro_f1": _float(oracle["macro_f1"]) - _float(selected["macro_f1"]),
                    "top1_downstream_oracle_hit": int(
                        str(oracle["candidate_source_center"]) == selected_source
                    ),
                    "support_score_selected_raw": _float(selected_support["support_score"]),
                    "support_labels_used": False,
                    "selection_used_target_labels": False,
                    "target_eval_labels_used_for_scoring_only": True,
                    "claim_role": "post_freeze_downstream_diagnostic",
                }
            )
    return rows


def _selected_vs_oracle_gap_rows(alignment_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "heldout_center": row["heldout_center"],
            "support_seed": row["support_seed"],
            "replicate": row["replicate"],
            "support_split_id": row["support_split_id"],
            "candidate_method": row["candidate_method"],
            "downstream_replicate_seed": row["downstream_replicate_seed"],
            "generation_seed": row["generation_seed"],
            "classifier_seed": row["classifier_seed"],
            "selected_source_center": row["selected_source_center"],
            "oracle_source_center": row["oracle_source_center"],
            "selected_bacc": row["selected_bacc"],
            "oracle_bacc": row["oracle_bacc"],
            "oracle_gap_bacc": row["downstream_oracle_gap_bacc"],
            "selected_macro_f1": row["selected_macro_f1"],
            "oracle_macro_f1": row["oracle_macro_f1"],
            "oracle_gap_macro_f1": row["downstream_oracle_gap_macro_f1"],
            "top1_downstream_oracle_hit": row["top1_downstream_oracle_hit"],
        }
        for row in alignment_rows
    ]


def _baseline_comparison_rows(
    alignment_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    baseline_by_context = {
        _downstream_context_key(row): row
        for row in baseline_rows
    }
    rows = []
    selected_by_context: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in alignment_rows:
        key = (str(row["heldout_center"]), str(row["support_seed"]), str(row["candidate_method"]))
        selected_by_context[key].append(row)
    for key, subset in sorted(selected_by_context.items()):
        baseline_matches = []
        for row in subset:
            ctx = (
                str(row["heldout_center"]),
                str(row["downstream_replicate_seed"]),
                str(row["generation_seed"]),
                str(row["classifier_seed"]),
            )
            baseline = baseline_by_context.get(ctx)
            if baseline is not None:
                baseline_matches.append(baseline)
        rows.append(
            {
                "heldout_center": key[0],
                "support_seed": key[1],
                "candidate_method": key[2],
                "mean_selected_bacc": _mean(_float(row["selected_bacc"]) for row in subset),
                "mean_selected_macro_f1": _mean(_float(row["selected_macro_f1"]) for row in subset),
                "mean_oracle_bacc": _mean(_float(row["oracle_bacc"]) for row in subset),
                "mean_oracle_gap_bacc": _mean(_float(row["downstream_oracle_gap_bacc"]) for row in subset),
                "top1_downstream_oracle_hit_rate": _mean(
                    _float(row["top1_downstream_oracle_hit"]) for row in subset
                ),
                "mean_baseline_bacc": _mean(_float(row["bacc"]) for row in baseline_matches),
                "mean_baseline_macro_f1": _mean(_float(row["macro_f1"]) for row in baseline_matches),
                "mean_delta_bacc_vs_method_baseline": _mean(_float(row["selected_bacc"]) for row in subset)
                - _mean(_float(row["bacc"]) for row in baseline_matches),
                "baseline_rows": len(baseline_matches),
                "alignment_rows": len(subset),
            }
        )
    return rows


def _summary_rows(
    alignment_rows: Sequence[Mapping[str, object]],
    *,
    group_keys: Sequence[str],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in alignment_rows:
        groups[tuple(str(row[key]) for key in group_keys)].append(row)
    rows = []
    for key, subset in sorted(groups.items()):
        out = {name: value for name, value in zip(group_keys, key)}
        out.update(
            {
                "alignment_rows": len(subset),
                "mean_selected_bacc": _mean(_float(row["selected_bacc"]) for row in subset),
                "mean_selected_macro_f1": _mean(_float(row["selected_macro_f1"]) for row in subset),
                "mean_oracle_bacc": _mean(_float(row["oracle_bacc"]) for row in subset),
                "mean_oracle_macro_f1": _mean(_float(row["oracle_macro_f1"]) for row in subset),
                "mean_oracle_gap_bacc": _mean(_float(row["downstream_oracle_gap_bacc"]) for row in subset),
                "mean_oracle_gap_macro_f1": _mean(_float(row["downstream_oracle_gap_macro_f1"]) for row in subset),
                "top1_downstream_oracle_hit_rate": _mean(
                    _float(row["top1_downstream_oracle_hit"]) for row in subset
                ),
            }
        )
        rows.append(out)
    return rows


def _validation_report(
    *,
    phase2_root: Path,
    phase1_matrix: Path,
    candidate_method: str,
    preflight: Mapping[str, object],
    downstream_rows: Sequence[Mapping[str, str]],
    alignment_rows: Sequence[Mapping[str, object]],
    oracle_gap_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    if not alignment_rows:
        raise ProtocolError("No phase-2 downstream alignment rows were built.")
    return {
        "schema_version": "midogpp_phase2_validation_report_v1",
        "artifacts_root": str(phase2_root),
        "status": "PASS",
        "candidate_method": candidate_method,
        "checks": {
            "preflight_status": preflight.get("status"),
            "phase1_matrix": str(phase1_matrix),
            "downstream_rows": len(downstream_rows),
            "baseline_rows": len(baseline_rows),
            "alignment_rows": len(alignment_rows),
            "selected_vs_oracle_gap_rows": len(oracle_gap_rows),
            "heldout_centers": sorted({str(row["heldout_center"]) for row in alignment_rows}),
            "support_seeds": sorted({str(row["support_seed"]) for row in alignment_rows}),
            "downstream_replicate_seeds": sorted({str(row["downstream_replicate_seed"]) for row in alignment_rows}),
            "generation_seeds": sorted({str(row["generation_seed"]) for row in alignment_rows}),
            "classifier_seeds": sorted({str(row["classifier_seed"]) for row in alignment_rows}),
            "mean_selected_bacc": _mean(_float(row["selected_bacc"]) for row in alignment_rows),
            "mean_oracle_bacc": _mean(_float(row["oracle_bacc"]) for row in alignment_rows),
            "mean_oracle_gap_bacc": _mean(_float(row["downstream_oracle_gap_bacc"]) for row in alignment_rows),
            "top1_downstream_oracle_hit_rate": _mean(
                _float(row["top1_downstream_oracle_hit"]) for row in alignment_rows
            ),
            "routing_freeze_unchanged": True,
            "claim_boundary": "post-freeze downstream diagnostic alignment only",
        },
        "input_hashes": {
            "phase1_matrix_hash": _file_hash(phase1_matrix),
            "selected_sources_hash": _file_hash(phase2_root / "tables" / "selected_sources.csv"),
            "routing_decisions_hash": _file_hash(phase2_root / "tables" / "routing_decisions.csv"),
            "support_score_matrix_hash": _file_hash(phase2_root / "tables" / "support_score_matrix.csv"),
            "frozen_protocol_snapshot_hash": _file_hash(phase2_root / "configs" / "frozen_protocol_snapshot.json"),
        },
    }


def _decision_summary(
    report: Mapping[str, object],
    alignment_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> str:
    checks = report["checks"]
    assert isinstance(checks, Mapping)
    baseline_delta = _mean(
        _float(row["mean_delta_bacc_vs_method_baseline"])
        for row in baseline_rows
        if not math.isnan(_float(row["mean_delta_bacc_vs_method_baseline"]))
    )
    return "\n".join(
        [
            "# MIDOG++ Phase-2 Downstream Diagnostic Summary",
            "",
            f"- Status: {report['status']}",
            f"- Candidate method: {report['candidate_method']}",
            f"- Alignment rows: {checks['alignment_rows']}",
            f"- Mean selected BACC: {_format_float(checks['mean_selected_bacc'])}",
            f"- Mean oracle BACC: {_format_float(checks['mean_oracle_bacc'])}",
            f"- Mean oracle gap BACC: {_format_float(checks['mean_oracle_gap_bacc'])}",
            f"- Top-1 downstream oracle hit rate: {_format_float(checks['top1_downstream_oracle_hit_rate'])}",
            f"- Mean delta BACC vs method baseline: {_format_float(baseline_delta)}",
            "",
            "Claim boundary: frozen support-NELBO routing is evaluated only after the routing freeze. "
            "Target downstream labels are diagnostic/final-scoring only.",
            "",
        ]
    )


def _oracle_by_context(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, str, str], Mapping[str, str]]:
    out: dict[tuple[str, str, str, str], Mapping[str, str]] = {}
    for row in rows:
        key = _downstream_context_key(row)
        current = out.get(key)
        if current is None or (
            _float(row["bacc"]),
            _float(row["macro_f1"]),
            str(row["candidate_source_center"]),
        ) > (
            _float(current["bacc"]),
            _float(current["macro_f1"]),
            str(current["candidate_source_center"]),
        ):
            out[key] = row
    return out


def _selected_support_row(rows: Sequence[Mapping[str, str]], selected_source: str) -> Mapping[str, str]:
    for row in rows:
        if str(row.get("candidate_source_center", "")) == selected_source:
            return row
    raise ProtocolError(f"Selected source {selected_source!r} is missing from support scores.")


def _support_rank(rows: Sequence[Mapping[str, str]], selected_source: str) -> int:
    ranked = sorted(rows, key=lambda row: (_float(row["support_score"]), str(row["stable_candidate_id"])))
    for idx, row in enumerate(ranked, start=1):
        if str(row.get("candidate_source_center", "")) == selected_source:
            return idx
    raise ProtocolError(f"Selected source {selected_source!r} is missing from ranked support scores.")


def _selection_context_key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row["heldout_center"]),
        str(row["support_seed"]),
        str(row["replicate"]),
        str(row["support_split_id"]),
    )


def _downstream_context_key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row["heldout_center"]),
        str(row["replicate_seed"]),
        str(row["generation_seed"]),
        str(row["classifier_seed"]),
    )


def _downstream_candidate_key(row: Mapping[str, object]) -> tuple[str, str, str, str, str]:
    return _downstream_context_key(row) + (str(row["candidate_source_center"]),)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ProtocolError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_hash(path: Path) -> str:
    return stable_hash(Path(path).read_text(encoding="utf-8"))


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _format_float(value: object) -> str:
    val = _float(value)
    return "nan" if math.isnan(val) else f"{val:.6f}"


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
