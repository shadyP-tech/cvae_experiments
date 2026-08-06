"""CLI for non-deployable CVAE diagnostic snapshots and audits."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the import-light diagnostic command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="surface", required=True)

    snapshot = sub.add_parser(
        "build-b-paired-reparameterization-snapshot",
        help="Build the portable canonical-B paired-replay snapshot.",
    )
    snapshot.add_argument("--config", required=True)
    snapshot.add_argument("--artifact-root", required=True)

    audit = sub.add_parser(
        "b-paired-reparameterization-audit",
        help="Run the bounded canonical-B paired reparameterization audit.",
    )
    audit.add_argument("--config", required=True)
    audit.add_argument("--artifact-root", required=True)

    residual = sub.add_parser(
        "dense-residual-router-diagnostic",
        help=(
            "Run the consumed-validation Stage-90 dense residual router "
            "diagnostic."
        ),
    )
    residual.add_argument("--config", required=True)
    residual.add_argument("--artifact-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_root = Path(args.artifact_root)

    if args.surface == "build-b-paired-reparameterization-snapshot":
        from .b_paired_reparameterization_audit import build_snapshot_from_config

        output = build_snapshot_from_config(
            args.config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "b-paired-reparameterization-audit":
        from .b_paired_reparameterization_audit import (
            load_audit_config,
            run_b_paired_reparameterization_audit,
        )

        config = load_audit_config(args.config)
        output = run_b_paired_reparameterization_audit(
            config,
            artifact_root=artifact_root,
            resolved_config_path=args.config,
        )
        print(output)
        return 0

    if args.surface == "dense-residual-router-diagnostic":
        from .dense_residual_router.config import (
            load_dense_residual_diagnostic_config,
        )
        from .dense_residual_router.runner import (
            run_dense_residual_router_diagnostic,
        )

        config = load_dense_residual_diagnostic_config(args.config)
        output = run_dense_residual_router_diagnostic(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    raise AssertionError(f"Unknown CVAE diagnostic surface: {args.surface}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
