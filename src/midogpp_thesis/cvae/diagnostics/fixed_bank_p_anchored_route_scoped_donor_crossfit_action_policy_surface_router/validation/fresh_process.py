"""Sequential fresh-process bundle validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

from ....protocol import ProtocolError
from ..identity import canonical_hash
from ..persistence.bundle import verify_content_index
from .protocol import validate_frozen_protocol, validate_no_sibling_imports


def validate_bundle(root: Path) -> dict[str, object]:
    content = verify_content_index(Path(root))
    protocol = validate_frozen_protocol()
    imports = validate_no_sibling_imports()
    base = {
        "schema_version": "pdcaps_fresh_validation_v1",
        "status": "PASS",
        "content_index_hash": content["content_index_hash"],
        "protocol_hash": protocol["protocol_hash"],
        "python_file_count": imports["python_file_count"],
        "optimizer_refit_count": 0,
    }
    return {**base, "checks_hash": canonical_hash(base)}


def require_two_fresh_process_validations(
    root: Path,
    *,
    expected_checks: Mapping[str, object],
) -> dict[str, object]:
    rows = []
    for _ in range(2):
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                (
                    "midogpp_thesis.cvae.diagnostics."
                    "fixed_bank_p_anchored_route_scoped_donor_crossfit_"
                    "action_policy_surface_router.validation.fresh_process"
                ),
                "--root",
                str(Path(root).resolve()),
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ProtocolError(
                f"P-DCAPS fresh validation failed: {completed.stderr.strip()}."
            )
        try:
            row = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ProtocolError("P-DCAPS fresh validator returned malformed JSON.") from exc
        if row.get("checks") != dict(expected_checks):
            raise ProtocolError("P-DCAPS fresh-process checks drifted.")
        rows.append(row)
    pids = tuple(int(row["pid"]) for row in rows)
    if len(set(pids)) != 2 or os.getpid() in pids:
        raise ProtocolError("P-DCAPS validations were not two distinct fresh processes.")
    base = {
        "schema_version": "pdcaps_two_fresh_process_attestation_v1",
        "status": "PASS",
        "pids": list(pids),
        "checks_hash": expected_checks["checks_hash"],
        "sequential": True,
    }
    return {**base, "attestation_hash": canonical_hash(base)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    print(json.dumps({"pid": os.getpid(), "checks": validate_bundle(Path(args.root))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("require_two_fresh_process_validations", "validate_bundle")
