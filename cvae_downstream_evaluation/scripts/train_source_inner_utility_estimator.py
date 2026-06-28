"""Train a source-inner learned downstream utility estimator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.compatibility.diagnostics import estimator_diagnostics  # noqa: E402
from cvae_downstream_evaluation.compatibility.estimators import predict_rows, save_estimator  # noqa: E402
from cvae_downstream_evaluation.compatibility.train_source_inner import train_linear_utility_estimator  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train source-inner downstream utility estimator.")
    parser.add_argument("--source-inner", required=True, help="Source-inner training CSV.")
    parser.add_argument("--model-out", required=True, help="Estimator JSON output.")
    parser.add_argument("--diagnostics-out", required=True, help="Training diagnostics JSON output.")
    parser.add_argument("--features", required=True, help="Comma-separated deployable feature columns.")
    parser.add_argument("--label", default="source_inner_heldout_bacc")
    parser.add_argument("--ridge-lambda", type=float, default=1e-6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = _read_csv(Path(args.source_inner))
    feature_columns = tuple(part.strip() for part in str(args.features).split(",") if part.strip())
    estimator = train_linear_utility_estimator(
        rows,
        feature_columns=feature_columns,
        label=str(args.label),
        ridge_lambda=float(args.ridge_lambda),
    )
    save_estimator(Path(args.model_out), estimator)
    predicted = predict_rows(estimator, rows)
    diagnostics = estimator_diagnostics(predicted, label_column=str(args.label))
    out = Path(args.diagnostics_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote source-inner estimator: {args.model_out}")
    print(f"Wrote source-inner diagnostics: {args.diagnostics_out}")


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
