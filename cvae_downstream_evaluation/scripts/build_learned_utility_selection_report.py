"""Build learned-utility selection and diagnostic alignment tables.

This script preserves the firewall: selection is built from allowed feature
rows only. The diagnostic downstream utility matrix is read only after
selection rows already exist, for reporting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.compatibility.select_candidates import (  # noqa: E402
    build_top1_selection_rows,
    write_selection_rows,
)
from cvae_downstream_evaluation.downstream import read_candidate_downstream_matrix  # noqa: E402
from cvae_downstream_evaluation.features.feature_table_builder import build_allowed_feature_table  # noqa: E402
from cvae_downstream_evaluation.reports.rank_metrics import (  # noqa: E402
    build_learned_utility_alignment_rows,
    learned_utility_alignment_columns,
)
from cvae_downstream_evaluation.reports.tables import write_rows  # noqa: E402
from cvae_downstream_evaluation.utility_matrix import (  # noqa: E402
    assert_diagnostic_matrix_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build adoption-eligible learned utility selections and diagnostic alignment."
    )
    parser.add_argument("--features", required=True, help="Allowed pre-evaluation feature CSV.")
    parser.add_argument(
        "--diagnostic-matrix",
        required=True,
        help="Diagnostic downstream utility matrix CSV, named diagnostic_downstream_utility.*.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for selections and reports.")
    parser.add_argument("--method", default="learned_downstream_utility_top1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    feature_rows = build_allowed_feature_table(_read_csv(Path(args.features)))
    selection_rows = build_top1_selection_rows(feature_rows, method=str(args.method))

    out_dir = Path(args.out_dir)
    selection_path = out_dir / "selections" / "adoption_eligible_predictions.csv"
    write_selection_rows(selection_path, selection_rows)

    matrix_path = Path(args.diagnostic_matrix)
    assert_diagnostic_matrix_path(matrix_path)
    downstream_rows = read_candidate_downstream_matrix(matrix_path)
    alignment_rows = build_learned_utility_alignment_rows(
        selection_rows=selection_rows,
        downstream_rows=downstream_rows,
    )
    alignment_path = out_dir / "reports" / "learned_utility_alignment.csv"
    write_rows(alignment_path, learned_utility_alignment_columns(), alignment_rows)

    print(f"Wrote adoption-eligible selections: {selection_path}")
    print(f"Wrote learned utility diagnostic alignment: {alignment_path}")


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


if __name__ == "__main__":
    main()
