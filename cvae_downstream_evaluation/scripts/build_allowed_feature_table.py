"""Build allowed pre-evaluation feature tables from artifact CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.features.feature_table_builder import (  # noqa: E402
    build_allowed_feature_table_from_artifacts,
    read_csv_rows,
    write_allowed_feature_table,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build allowed_pre_eval_features.csv from candidate and feature artifact CSVs."
    )
    parser.add_argument("--candidates", required=True, help="Candidate manifest CSV.")
    parser.add_argument("--out", required=True, help="Output allowed feature CSV.")
    parser.add_argument("--support-features", default=None)
    parser.add_argument("--source-inner-features", default=None)
    parser.add_argument("--metadata-features", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = build_allowed_feature_table_from_artifacts(
        candidate_rows=read_csv_rows(Path(args.candidates)),
        support_feature_rows=_optional_rows(args.support_features),
        source_inner_rows=_optional_rows(args.source_inner_features),
        metadata_rows=_optional_rows(args.metadata_features),
    )
    write_allowed_feature_table(Path(args.out), rows)
    print(f"Wrote allowed pre-evaluation feature table: {args.out}")


def _optional_rows(raw: str | None) -> list[dict[str, object]]:
    return read_csv_rows(Path(raw)) if raw else []


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
