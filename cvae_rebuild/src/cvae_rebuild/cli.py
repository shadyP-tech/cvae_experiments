from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import run_artifact_contract_smoke, run_real_cache_backed, run_synthetic_smoke
from .preservation import load_preservation_config, run_preservation_diagnosis
from .preservation_repair import load_repair_config, run_preservation_repair
from .preservation_sampling import load_sampling_config, run_preservation_sampling


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Virchow2-CVAE rebuild runner.")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config", help="Validate a locked rebuild config.")
    validate.add_argument("--config", required=True)

    smoke = sub.add_parser("smoke-artifacts", help="Write empty artifact-contract outputs.")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--artifact-root", default=None)

    run = sub.add_parser("run", help="Run the rebuild pipeline or a synthetic smoke run.")
    run.add_argument("--config", required=True)
    run.add_argument("--artifact-root", default=None)
    run.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run a tiny end-to-end synthetic train/routing/downstream smoke.",
    )

    diagnose = sub.add_parser("diagnose-preservation", help="Run the Virchow2-CVAE preservation diagnosis.")
    diagnose.add_argument("--config", required=True)
    diagnose.add_argument("--artifact-root", default=None)

    repair = sub.add_parser("diagnose-preservation-repair", help="Run the Virchow2-CVAE preservation repair diagnosis.")
    repair.add_argument("--config", required=True)
    repair.add_argument("--artifact-root", default=None)

    sampling = sub.add_parser("diagnose-preservation-sampling", help="Run the Virchow2-CVAE PCA64 sampling continuation.")
    sampling.add_argument("--config", required=True)
    sampling.add_argument("--artifact-root", default=None)

    args = parser.parse_args(argv)
    if args.command == "diagnose-preservation-sampling":
        cfg = load_sampling_config(args.config)
        root = run_preservation_sampling(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-preservation-repair":
        cfg = load_repair_config(args.config)
        root = run_preservation_repair(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-preservation":
        cfg = load_preservation_config(args.config)
        root = run_preservation_diagnosis(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    cfg = load_config(args.config)
    if args.command == "validate-config":
        print(f"OK: {cfg.name}")
        return 0
    if args.command == "smoke-artifacts":
        root = run_artifact_contract_smoke(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "run":
        if args.synthetic_smoke:
            root = run_synthetic_smoke(
                cfg,
                artifact_root=Path(args.artifact_root) if args.artifact_root else None,
            )
            print(root)
            return 0
        root = run_real_cache_backed(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
