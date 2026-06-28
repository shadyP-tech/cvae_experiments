"""Materialize MIDOG++ phase-1 diagnostic artifacts from pre-scored rows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.adapters.midogpp import (  # noqa: E402
    read_candidate_manifest_rows,
    read_midogpp_scored_rows,
    write_midogpp_phase1_artifacts,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.schemas.midogpp import assert_midogpp_frozen_config_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write MIDOG++ phase-1 diagnostic matrix, oracle summary, baseline "
            "comparison, leakage report, and decision summary from pre-scored rows."
        )
    )
    parser.add_argument("--scored-rows", required=True, help="CSV containing MIDOG++ scored diagnostic rows.")
    parser.add_argument("--candidate-manifest", required=True, help="Selection-eligible candidate manifest CSV.")
    parser.add_argument("--out-dir", required=True, help="Output artifact root.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "utility_matrix" / "virchow2_midogpp_all_candidates.yaml"),
        help="Frozen MIDOG++ config to validate before writing artifacts.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assert_midogpp_frozen_config_file(Path(args.config))
    outputs = write_midogpp_phase1_artifacts(
        Path(args.out_dir),
        rows=read_midogpp_scored_rows(Path(args.scored_rows)),
        candidate_manifest_rows=read_candidate_manifest_rows(Path(args.candidate_manifest)),
    )
    for label, path in outputs.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
