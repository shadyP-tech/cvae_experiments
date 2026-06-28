"""Build downstream alignment and decision report tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.downstream import assert_matrix_schema, read_candidate_downstream_matrix
from cvae_downstream_evaluation.reporting import (
    baseline_comparison_rows,
    build_routing_alignment_rows,
    classify_decision,
    stability_rows,
    support_size_stratified_summary,
    write_alignment_csv,
    write_baseline_comparison_csv,
    write_decision_summary,
    write_stability_csv,
    write_support_size_summary_csv,
)
from cvae_downstream_evaluation.routing import support_units_from_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build thesis-facing downstream evaluation report tables."
    )
    parser.add_argument("--artifacts-root", required=True, help="Run artifact root.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output markdown report path. Defaults to reports/decision_summary.md.",
    )
    parser.add_argument(
        "--matrix",
        default=None,
        help="Matrix CSV path. Defaults to diagnostic_downstream_utility.csv if present, otherwise all_expert_downstream_matrix.csv.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    artifacts_root = Path(args.artifacts_root)
    support_path = artifacts_root / "tables" / "support_selection_units.csv"
    if args.matrix:
        matrix_path = Path(args.matrix)
    else:
        diagnostic_path = artifacts_root / "tables" / "diagnostic_downstream_utility.csv"
        matrix_path = diagnostic_path if diagnostic_path.exists() else artifacts_root / "tables" / "all_expert_downstream_matrix.csv"
    alignment_path = artifacts_root / "tables" / "routing_to_downstream_alignment.csv"
    baseline_path = artifacts_root / "tables" / "baseline_comparison.csv"
    support_size_path = artifacts_root / "tables" / "support_size_stratified_downstream_summary.csv"
    selection_stability_path = artifacts_root / "tables" / "selection_stability.csv"
    generation_stability_path = artifacts_root / "tables" / "generation_classifier_stability.csv"
    report_path = Path(args.out) if args.out else artifacts_root / "reports" / "decision_summary.md"

    selections = support_units_from_csv(support_path)
    assert_matrix_schema(matrix_path)
    downstream_rows = read_candidate_downstream_matrix(matrix_path)
    alignment_rows = build_routing_alignment_rows(selections=selections, downstream_rows=downstream_rows)
    write_alignment_csv(alignment_path, alignment_rows)
    write_baseline_comparison_csv(
        baseline_path,
        baseline_comparison_rows(alignment_rows=alignment_rows, downstream_rows=downstream_rows),
    )
    write_support_size_summary_csv(support_size_path, support_size_stratified_summary(alignment_rows))
    write_stability_csv(selection_stability_path, stability_rows(alignment_rows, group="selection_support"))
    write_stability_csv(generation_stability_path, stability_rows(alignment_rows, group="generation_classifier"))
    write_decision_summary(report_path, classify_decision(alignment_rows))
    print(f"Wrote alignment table: {alignment_path}")
    print(f"Wrote baseline comparison: {baseline_path}")
    print(f"Wrote support-size summary: {support_size_path}")
    print(f"Wrote selection stability: {selection_stability_path}")
    print(f"Wrote generation/classifier stability: {generation_stability_path}")
    print(f"Wrote decision summary: {report_path}")


if __name__ == "__main__":
    main()
