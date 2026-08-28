"""Internal source-only adapter for OE-PPUR v3 direct input number three.

The canonical executable is ``python -m midogpp_thesis.oe_ppur_v3
materialize-source``.  Invoking this deeper module directly is supported only
when every exact CUDA/BLAS variable has already been exported, because Python
imports :mod:`midogpp_thesis.cvae` before this module can run.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

from midogpp_thesis.oe_ppur_v3 import SOURCE_ENVIRONMENT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oe-ppur-v3-source-preparation",
        description=(
            "Materialize and reconstructively validate OE-PPUR v3 direct "
            "input #3 from the canonical source-only cache, bank, and lock."
        ),
    )
    parser.add_argument(
        "--scratch-root",
        required=True,
        help=(
            "Fresh or exact-resumable absolute source-preparation scratch "
            "root. This is distinct from the terminal router scratch root."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_preexported_source_environment()
    # Keep the producer import lazy for callers routed through the canonical
    # import-light top-level executable.
    from .source_runner import materialize_source_input

    result = materialize_source_input(scratch_root=args.scratch_root)
    print(
        json.dumps(
            result.to_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def _require_preexported_source_environment() -> None:
    for name, expected in SOURCE_ENVIRONMENT.items():
        observed = os.environ.get(name)
        if observed != expected:
            raise RuntimeError(
                "OE-PPUR v3 internal source adapter requires pre-exported "
                f"{name}={expected!r}; use the canonical top-level executable."
            )


if __name__ == "__main__":  # pragma: no cover - workstation entry point
    raise SystemExit(main())


__all__ = ("build_parser", "main")
