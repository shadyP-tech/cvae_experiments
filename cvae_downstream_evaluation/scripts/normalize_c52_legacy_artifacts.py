"""Normalize legacy C5.2 artifacts for the learned-utility pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.compatibility.legacy_adapters import normalize_c52_legacy_artifacts  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize C5.2 legacy artifacts for learned utility pipeline.")
    parser.add_argument("--router-training-examples", required=True)
    parser.add_argument("--downstream-matrix", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--support-size", type=int, default=None)
    parser.add_argument("--support-seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = normalize_c52_legacy_artifacts(
        router_training_examples=Path(args.router_training_examples),
        downstream_matrix=Path(args.downstream_matrix),
        target_domain=str(args.target_domain),
        out_dir=Path(args.out_dir),
        support_size=args.support_size,
        support_seed=args.support_seed,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
