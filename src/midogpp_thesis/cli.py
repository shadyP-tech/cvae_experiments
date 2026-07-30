"""Lazy command dispatcher for the canonical MIDOG++ package."""

from __future__ import annotations

import argparse
from importlib import import_module
import sys
from typing import Callable, Sequence


Handler = Callable[[list[str] | None], int]

COMMANDS: dict[str, tuple[str, str]] = {
    "dataset-build": (
        "midogpp_thesis.data.contract.commands.build:main",
        "Build the annotation-patch dataset contract.",
    ),
    "dataset-validate": (
        "midogpp_thesis.data.contract.commands.validate:main",
        "Validate a frozen dataset contract.",
    ),
    "dataset-inspect": (
        "midogpp_thesis.data.contract.commands.inspect:main",
        "Inspect contract/cache domain alignment.",
    ),
    "dataset-physical-multiscale": (
        "midogpp_thesis.data.physical_multiscale.cli:main",
        "Audit and build physical-multiscale dataset contracts and caches.",
    ),
    "real-features": (
        "midogpp_thesis.real_features.sail.cli:main",
        "Build Virchow2 caches or run real-feature controls.",
    ),
    "real-feature-classifier": (
        "midogpp_thesis.real_features.classifier_reference.cli:main",
        "Run real-feature classifier references and diagnostics.",
    ),
    "cvae-preservation": (
        "midogpp_thesis.cvae.preservation.cli:main",
        "Run a CVAE preservation surface.",
    ),
    "cvae-expert-bank": (
        "midogpp_thesis.cvae.expert_bank.cli:main",
        "Run CVAE source-expert adaptation pilots or expert-bank construction.",
    ),
    "workspace": (
        "midogpp_thesis.workspace.cli:main",
        "Validate, inspect, prepare, or run registered experiments.",
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="midogpp-thesis",
        description="Protocol-safe MIDOG++ thesis experiment commands.",
    )
    parser.add_argument("command", nargs="?", choices=tuple(COMMANDS))
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command = raw[0]
    if command not in COMMANDS:
        parser.error(f"invalid choice: {command!r}")
    target, _ = COMMANDS[command]
    module_name, function_name = target.split(":", 1)
    handler = getattr(import_module(module_name), function_name)
    return int(handler(raw[1:]))


def command_help() -> dict[str, str]:
    """Return command descriptions without importing any experiment module."""

    return {name: description for name, (_, description) in COMMANDS.items()}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
