"""CLI for the four MIDOG++ CVAE preservation-only surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable


Loader = Callable[[str | Path], object]
Runner = Callable[..., Path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="surface", required=True)
    for name, help_text in (
        ("sanity", "Run the preservation mechanics sanity surface."),
        ("gate", "Run the PCA128 preservation gate."),
        ("condition-audit", "Run the condition-capacity audit."),
        ("tuned-classifier", "Run tuned-classifier preservation."),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", required=True)
        command.add_argument("--artifact-root", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loader, runner = _surface_handler(args.surface)
    config = loader(args.config)
    artifact_root = Path(args.artifact_root) if args.artifact_root else None
    output = runner(config, artifact_root=artifact_root)
    print(output)
    return 0


def _surface_handler(surface: str) -> tuple[Loader, Runner]:
    if surface == "sanity":
        from .sanity import (
            load_midogpp_preservation_sanity_config,
            run_midogpp_preservation_sanity,
        )

        return load_midogpp_preservation_sanity_config, run_midogpp_preservation_sanity
    if surface == "gate":
        from .gate import (
            load_midogpp_preservation_gate_config,
            run_midogpp_preservation_gate,
        )

        return load_midogpp_preservation_gate_config, run_midogpp_preservation_gate
    if surface == "condition-audit":
        from .condition_audit import (
            load_midogpp_condition_audit_config,
            run_midogpp_condition_audit,
        )

        return load_midogpp_condition_audit_config, run_midogpp_condition_audit
    if surface == "tuned-classifier":
        from .tuned_classifier import (
            load_midogpp_tuned_classifier_preservation_config,
            run_midogpp_tuned_classifier_preservation,
        )

        return (
            load_midogpp_tuned_classifier_preservation_config,
            run_midogpp_tuned_classifier_preservation,
        )
    raise AssertionError(f"Unknown preservation surface: {surface}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
