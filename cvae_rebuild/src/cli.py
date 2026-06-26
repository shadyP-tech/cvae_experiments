from __future__ import annotations

import argparse
from pathlib import Path

from cli_registry import COMMANDS_BY_NAME, DIAGNOSIS_COMMANDS, load_config_for_validation
from config import load_config
from pipeline import run_artifact_contract_smoke, run_real_cache_backed, run_synthetic_smoke

_load_config_for_validation = load_config_for_validation


def _add_config_artifact_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-root", default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Virchow2-CVAE rebuild runner.")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config", help="Validate a locked rebuild config.")
    validate.add_argument("--config", required=True)

    smoke = sub.add_parser("smoke-artifacts", help="Write empty artifact-contract outputs.")
    _add_config_artifact_args(smoke)

    run = sub.add_parser("run", help="Run the rebuild pipeline or a synthetic smoke run.")
    _add_config_artifact_args(run)
    run.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run a tiny end-to-end synthetic train/routing/downstream smoke.",
    )

    for command in DIAGNOSIS_COMMANDS:
        diagnose = sub.add_parser(command.command, help=command.help)
        _add_config_artifact_args(diagnose)

    args = parser.parse_args(argv)
    artifact_root = Path(args.artifact_root) if getattr(args, "artifact_root", None) else None

    if args.command in COMMANDS_BY_NAME:
        command = COMMANDS_BY_NAME[args.command]
        cfg = command.load_config(args.config)
        root = command.run(cfg, artifact_root=artifact_root)
        print(root)
        return 0

    if args.command == "validate-config":
        cfg = load_config_for_validation(args.config)
        print(f"OK: {cfg.name}")
        return 0

    cfg = load_config(args.config)
    if args.command == "smoke-artifacts":
        root = run_artifact_contract_smoke(cfg, artifact_root=artifact_root)
        print(root)
        return 0
    if args.command == "run":
        if args.synthetic_smoke:
            root = run_synthetic_smoke(cfg, artifact_root=artifact_root)
            print(root)
            return 0
        root = run_real_cache_backed(cfg, artifact_root=artifact_root)
        print(root)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
