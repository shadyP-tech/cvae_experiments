"""CLI for independently trained CVAE expert-bank preparation and pilots."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="surface", required=True)
    pilot = sub.add_parser(
        "uniform-b-adaptation-pilot",
        help="Run the bounded canonical-B source-expert adaptation pilot.",
    )
    pilot.add_argument("--config", required=True)
    pilot.add_argument("--artifact-root", default=None)
    stability = sub.add_parser(
        "uniform-b-block-tail-average-stability-probe",
        help="Run the bounded v2 B-block last-quarter averaging diagnostic.",
    )
    stability.add_argument("--config", required=True)
    stability.add_argument("--artifact-root", default=None)
    args = parser.parse_args(argv)
    if args.surface == "uniform-b-adaptation-pilot":
        from .b_adaptation_pilot import load_pilot_config, run_pilot

        config = load_pilot_config(args.config)
        output = run_pilot(
            config,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(output)
        return 0
    if args.surface == "uniform-b-block-tail-average-stability-probe":
        from .b_stability_probe import load_stability_config, run_stability_probe

        config = load_stability_config(args.config)
        output = run_stability_probe(
            config,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(output)
        return 0
    raise AssertionError(f"Unknown expert-bank surface: {args.surface}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
