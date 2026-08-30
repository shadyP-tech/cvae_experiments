"""Import-light preparation and explicit real-launch executable for OE-PPUR v4."""

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
    render = commands.add_parser(
        "render-launch-authority",
        help="Render a separate hash-bound launch capability to stdout.",
    )
    render.add_argument("--repository-root", type=Path, required=True)
    render.add_argument("--preflight-receipt", type=Path, required=True)
    render.add_argument("--authorization-phrase", required=True)
    render.add_argument("--authorization-nonce")
    render.add_argument("--scratch-root", type=Path)
    render.add_argument("--host-id")
    dry_run = commands.add_parser(
        "dry-run",
        help="Replay every read-only launch gate without claiming the lease.",
    )
    dry_run.add_argument("--repository-root", type=Path, required=True)
    dry_run.add_argument("--preflight-receipt", type=Path, required=True)
    dry_run.add_argument("--authority", type=Path, required=True)
    dry_run.add_argument("--scratch-root", type=Path)
    dry_run.add_argument("--host-id")
    run = commands.add_parser(
        "run",
        help="Consume the single-use authority and execute the terminal diagnostic.",
    )
    run.add_argument("--repository-root", type=Path, required=True)
    run.add_argument("--preflight-receipt", type=Path)
    run.add_argument("--authority", type=Path)
    run.add_argument("--scratch-root", type=Path)
    run.add_argument("--host-id")
    run.add_argument("--confirm")
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
    elif args.command == "render-launch-authority":
        from .cvae.diagnostics.oe_ppur_v4_preparation.workspace import (
            DEFAULT_SCRATCH_ROOT,
        )
        from .cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.execution.sealed_replay import (
            build_launch_authority_from_replay,
            replay_sealed_execution,
        )

        replay = replay_sealed_execution(
            args.repository_root,
            preflight_receipt_path=args.preflight_receipt,
            scratch_root=(
                DEFAULT_SCRATCH_ROOT
                if args.scratch_root is None
                else args.scratch_root
            ),
            host_id=args.host_id,
        )
        payload = build_launch_authority_from_replay(
            replay,
            authorization_phrase=args.authorization_phrase,
            authorization_nonce=args.authorization_nonce,
        ).to_payload()
    elif args.command == "dry-run":
        from .cvae.diagnostics.oe_ppur_v4_preparation.workspace import (
            DEFAULT_SCRATCH_ROOT,
        )
        from .cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.execution.dry_run import (
            dry_run_real_launch,
        )

        payload = dry_run_real_launch(
            args.repository_root,
            preflight_receipt_path=args.preflight_receipt,
            launch_authority_path=args.authority,
            scratch_root=(
                DEFAULT_SCRATCH_ROOT
                if args.scratch_root is None
                else args.scratch_root
            ),
            host_id=args.host_id,
        ).to_payload()
    elif args.command == "run":
        if (
            args.preflight_receipt is None
            or args.authority is None
            or args.confirm is None
        ):
            raise RuntimeError(
                "OE-PPUR v4 execution is outside the current authorization: "
                "sealed preflight, separate launch authority, and explicit "
                "terminal confirmation are required."
            )
        if args.confirm != "RUN_TERMINAL_CONSUMED_TEST":
            raise RuntimeError(
                "OE-PPUR v4 run requires --confirm RUN_TERMINAL_CONSUMED_TEST."
            )
        from .cvae.diagnostics.oe_ppur_v4_preparation.workspace import (
            DEFAULT_SCRATCH_ROOT,
        )
        from .cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.runner import (
            run_real_oe_ppur_v4,
        )

        root = run_real_oe_ppur_v4(
            args.repository_root,
            preflight_receipt_path=args.preflight_receipt,
            launch_authority_path=args.authority,
            scratch_root=(
                DEFAULT_SCRATCH_ROOT
                if args.scratch_root is None
                else args.scratch_root
            ),
            host_id=args.host_id,
        )
        payload = {
            "schema_version": "oe_ppur_v4_real_launch_result_v1",
            "artifact_root": root.as_posix(),
            "status": "COMPLETE",
            "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
            "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        }
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
