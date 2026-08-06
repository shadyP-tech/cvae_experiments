"""CLI for frozen CVAE generation contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..protocol import ProtocolError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="surface", required=True)
    generation_lock = sub.add_parser(
        "uniform-b-v2-generation-lock",
        help="Freeze and health-check the routing-authorized Uniform-B v2 generator.",
    )
    generation_lock.add_argument("--config", required=True)
    generation_lock.add_argument("--artifact-root", default=None)
    args = parser.parse_args(argv)
    if args.surface == "uniform-b-v2-generation-lock":
        from .config import load_generation_lock_config
        from .runner import run_generation_lock
        from .workspace_binding import validate_production_workspace_binding

        config = load_generation_lock_config(args.config)
        validate_production_workspace_binding(config)
        requested_root = Path(args.artifact_root) if args.artifact_root else config.artifact_root
        if requested_root.resolve() != config.artifact_root.resolve():
            raise ProtocolError(
                "Uniform-B v2 GenerationLock output must remain at its canonical workspace path."
            )
        output = run_generation_lock(
            config,
            artifact_root=requested_root,
        )
        print(output)
        return 0
    raise AssertionError(f"Unknown generation surface: {args.surface}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
