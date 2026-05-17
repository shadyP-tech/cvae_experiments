"""Build source-global gated support-NELBO v2 report artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.source_global_gated import build_source_global_gated_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build post-hoc source-global gated support-NELBO report tables."
    )
    parser.add_argument(
        "--artifacts-root",
        required=True,
        help="Completed v1 downstream artifact root containing tables/ and reports/.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = build_source_global_gated_report(Path(args.artifacts_root))
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
