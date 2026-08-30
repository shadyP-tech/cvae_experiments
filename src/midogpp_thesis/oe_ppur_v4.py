"""Import-light, preparation-only executable for OE-PPUR v4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m midogpp_thesis.oe_ppur_v4",
        description=(
            "Workspace-sealed preparation lifecycle for the terminal-only "
            "OE-PPUR v4 successor."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect the path-free plan.")
    inspect.add_argument("--config", type=Path)
    preflight = commands.add_parser(
        "preflight", help="Build a mutation-free prospective workspace plan."
    )
    preflight.add_argument("--repository-root", type=Path, required=True)
    preflight.add_argument("--scratch-root", type=Path)
    preflight.add_argument("--host-id")
    authorize = commands.add_parser(
        "authorize",
        help="Publish only an exact replayed v4 amendment; never launch.",
    )
    authorize.add_argument("--repository-root", type=Path, required=True)
    authorize.add_argument("--preflight-receipt", type=Path, required=True)
    authorize.add_argument("--scratch-root", type=Path)
    authorize.add_argument("--host-id")
    run = commands.add_parser("run", help="Fail closed without separate launch authority.")
    run.add_argument("--repository-root", type=Path, required=True)
    run.add_argument("--authority", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        from .cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.config import (
            build_planned_config,
            load_config,
        )
        from .cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.runner import (
            inspect_planned_router,
        )

        config = build_planned_config() if args.config is None else load_config(args.config)
        payload = inspect_planned_router(config)
    elif args.command == "preflight":
        from .cvae.diagnostics.oe_ppur_v4_preparation.workspace import (
            DEFAULT_SCRATCH_ROOT,
            build_workspace_preparation_context,
            preflight_document,
            replay_prepublication,
        )

        context = build_workspace_preparation_context(
            args.repository_root,
            scratch_root=(
                DEFAULT_SCRATCH_ROOT
                if args.scratch_root is None
                else args.scratch_root
            ),
            host_id=args.host_id,
        )
        receipt = replay_prepublication(context)
        payload = preflight_document(context, receipt)
    elif args.command == "authorize":
        from .cvae.diagnostics.oe_ppur_v4_preparation.publish import (
            publish_amendment_only,
        )
        from .cvae.diagnostics.oe_ppur_v4_preparation.workspace import (
            DEFAULT_SCRATCH_ROOT,
            build_workspace_preparation_context,
        )

        context = build_workspace_preparation_context(
            args.repository_root,
            scratch_root=(
                DEFAULT_SCRATCH_ROOT
                if args.scratch_root is None
                else args.scratch_root
            ),
            host_id=args.host_id,
        )
        try:
            preflight_raw = args.preflight_receipt.read_bytes()
        except OSError as exc:
            raise RuntimeError("OE-PPUR v4 preflight receipt is unavailable.") from exc
        payload = publish_amendment_only(
            context,
            preflight_raw=preflight_raw,
        ).to_payload()
    elif args.command == "run":
        raise RuntimeError(
            "OE-PPUR v4 execution is outside the current authorization."
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("build_parser", "main")
