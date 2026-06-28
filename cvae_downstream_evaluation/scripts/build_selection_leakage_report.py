"""Build leakage/provenance report for learned-utility selection artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.reports.leakage_report import (  # noqa: E402
    build_leakage_report,
    write_leakage_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build learned-utility selection leakage report.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--selections", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--generation-frozen", action="store_true")
    parser.add_argument("--classifier-frozen", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_leakage_report(
        candidate_rows=_read_csv(Path(args.candidates)),
        feature_rows=_read_csv(Path(args.features)),
        selection_rows=_read_csv(Path(args.selections)),
        frozen_generation=bool(args.generation_frozen),
        frozen_classifier=bool(args.classifier_frozen),
    )
    write_leakage_report(Path(args.out), report)
    print(f"Wrote leakage report: {args.out}")


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
