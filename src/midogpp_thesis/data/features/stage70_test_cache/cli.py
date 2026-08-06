"""Package-local CLI for the catalog-only Stage-70 derived feature cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .builder import build_stage70_test_cache
from .config import load_stage70_test_cache_config
from .validation import validate_stage70_test_cache


COMMAND_NAME = "stage70-test-cache"


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Frozen Stage-70 descriptive test-cache YAML config.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the existing catalog cache without opening source JPEGs.",
    )
    parser.set_defaults(_stage70_test_cache_handler=run_from_args)
    return parser


def build_parser() -> argparse.ArgumentParser:
    return add_arguments(
        argparse.ArgumentParser(
            prog=COMMAND_NAME,
            description=(
                "Build or validate the label-sealed, descriptive Stage-70 "
                "Virchow2 test cache."
            ),
        )
    )


def register_subparser(subparsers: object) -> argparse.ArgumentParser:
    """Register the package CLI under a caller-owned top-level parser."""

    add_parser = getattr(subparsers, "add_parser", None)
    if not callable(add_parser):
        raise TypeError("Stage-70 CLI registration requires argparse subparsers.")
    parser = add_parser(
        COMMAND_NAME,
        help="Build/validate the catalog-only descriptive Stage-70 test cache.",
    )
    return add_arguments(parser)


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    config = load_stage70_test_cache_config(args.config)
    if bool(args.validate_only):
        return validate_stage70_test_cache(
            config.cache_root,
            expected_config=config,
        )
    root = build_stage70_test_cache(config)
    return validate_stage70_test_cache(root, expected_config=config)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_from_args(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual workstation surface
    raise SystemExit(main())


__all__ = (
    "COMMAND_NAME",
    "add_arguments",
    "build_parser",
    "main",
    "register_subparser",
    "run_from_args",
)
