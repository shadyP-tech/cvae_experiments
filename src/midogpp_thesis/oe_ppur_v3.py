"""Import-safe executable lifecycle for the OE-PPUR v3 experiment.

This entrypoint lives above ``midogpp_thesis.cvae`` so source-worker CUDA and
BLAS variables can be fixed before that package imports torch or NumPy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence


SOURCE_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
DEFAULT_SOURCE_SCRATCH_ROOT = Path(
    "/data/local/oe_ppur_v3_source_training_supervision_work"
)
DEFAULT_EXECUTION_SCRATCH_ROOT = Path(
    "/data/local/fixed_bank_p_anchored_opportunity_equivalence_"
    "pairwise_primitive_utility_router_v3"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m midogpp_thesis.oe_ppur_v3",
        description=(
            "Specialized source, authorization, and single-use execution "
            "lifecycle for OE-PPUR v3."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    source = subcommands.add_parser(
        "materialize-source",
        help="Materialize and reconstructively validate direct input #3 only.",
    )
    source.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SOURCE_SCRATCH_ROOT,
    )
    authorize = subcommands.add_parser(
        "authorize",
        help=(
            "Issue direct input #7 once and render the resolved launch "
            "envelope without consuming the lease."
        ),
    )
    authorize.add_argument("--repository-root", type=Path, required=True)
    authorize.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_EXECUTION_SCRATCH_ROOT,
    )
    recover = subcommands.add_parser(
        "render-existing",
        help=(
            "Validate an already issued input #7 and retry only launch-envelope "
            "rendering; never rewrite the amendment or claim the lease."
        ),
    )
    recover.add_argument("--repository-root", type=Path, required=True)
    recover.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_EXECUTION_SCRATCH_ROOT,
    )
    run = subcommands.add_parser(
        "run",
        help="Consume the single-use lease and run the prepared router.",
    )
    run.add_argument("--repository-root", type=Path, required=True)
    run.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_EXECUTION_SCRATCH_ROOT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Every command may import NumPy/torch through the sealed router package.
    # Fix the workstation process contract before any such import, including
    # authorization-only parsing and the final dedicated launch edge.
    _apply_source_environment()
    if args.command == "materialize-source":
        from .cvae.diagnostics.oe_ppur_v3_preparation.source_runner import (
            materialize_source_input,
        )

        payload = materialize_source_input(
            scratch_root=args.scratch_root,
        ).to_payload()
    elif args.command == "authorize":
        from .cvae.diagnostics.oe_ppur_v3_preparation.authorization_preparation import (
            authorize_and_render,
        )

        payload = authorize_and_render(
            args.repository_root,
            scratch_root=args.scratch_root,
        ).to_payload()
    elif args.command == "render-existing":
        from .cvae.diagnostics.oe_ppur_v3_preparation.authorization_preparation import (
            render_existing_authorization,
        )

        payload = render_existing_authorization(
            args.repository_root,
            scratch_root=args.scratch_root,
        ).to_payload()
    elif args.command == "run":
        from .cvae.diagnostics.oe_ppur_v3_preparation.authorized_runner import (
            run_authorized_experiment,
        )

        root = run_authorized_experiment(
            args.repository_root,
            scratch_root=args.scratch_root,
        )
        payload = {
            "schema_version": "oe_ppur_v3_dedicated_launch_result_v1",
            "status": "COMPLETE",
            "artifact_root": root.as_posix(),
        }
    else:  # pragma: no cover - argparse owns the closed command set
        raise AssertionError(args.command)
    print(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        flush=True,
    )
    return 0


def _apply_source_environment() -> None:
    for name, expected in SOURCE_ENVIRONMENT.items():
        observed = os.environ.get(name)
        if observed not in (None, expected):
            raise RuntimeError(
                f"OE-PPUR v3 source environment {name} must equal {expected!r}."
            )
        os.environ[name] = expected


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "DEFAULT_EXECUTION_SCRATCH_ROOT",
    "DEFAULT_SOURCE_SCRATCH_ROOT",
    "SOURCE_ENVIRONMENT",
    "build_parser",
    "main",
)
