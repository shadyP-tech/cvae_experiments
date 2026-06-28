"""Apply a source-inner learned utility estimator to allowed feature rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.compatibility.estimators import load_estimator, predict_rows  # noqa: E402
from cvae_downstream_evaluation.features.feature_table_builder import (  # noqa: E402
    build_allowed_feature_table,
    write_allowed_feature_table,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict learned downstream utility for allowed features.")
    parser.add_argument("--model", required=True, help="Estimator JSON.")
    parser.add_argument("--features", required=True, help="Allowed pre-evaluation feature CSV.")
    parser.add_argument("--out", required=True, help="Feature CSV with predicted_primary_utility.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    estimator = load_estimator(Path(args.model))
    rows = build_allowed_feature_table(_read_csv(Path(args.features)))
    predicted = predict_rows(estimator, rows)
    write_allowed_feature_table(Path(args.out), predicted)
    print(f"Wrote learned utility predictions: {args.out}")


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
