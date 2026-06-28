"""Run learned-utility pipeline over legacy C5.2 contexts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.compatibility.legacy_adapters import run_c52_legacy_batch  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run C5.2 legacy learned-utility batch.")
    parser.add_argument("--router-training-examples", required=True)
    parser.add_argument("--downstream-matrix", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--target-domains", default=None)
    parser.add_argument("--support-sizes", default=None)
    parser.add_argument("--support-seeds", default=None)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_c52_legacy_batch(
        router_training_examples=Path(args.router_training_examples),
        downstream_matrix=Path(args.downstream_matrix),
        out_dir=Path(args.out_dir),
        feature_columns=_split_str(args.features),
        target_domains=_split_str(args.target_domains) if args.target_domains else None,
        support_sizes=_split_int(args.support_sizes) if args.support_sizes else None,
        support_seeds=_split_int(args.support_seeds) if args.support_seeds else None,
        ridge_lambda=float(args.ridge_lambda),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


def _split_str(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _split_int(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
