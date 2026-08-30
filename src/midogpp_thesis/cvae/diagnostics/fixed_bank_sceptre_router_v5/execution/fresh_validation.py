"""Independent subprocess barriers for SCEPTRE v5 preterminal and final state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

from ....protocol import ProtocolError
from ...fixed_bank_sceptre_router.seals import (
    DurablePreterminalAttestation,
    FreshProcessValidation,
)
from .validation import validate_final_bundle, validate_preterminal_bundle


def require_two_fresh_preterminal_validations(
    root: str | Path,
) -> DurablePreterminalAttestation:
    rows = _run_two(Path(root), phase="preterminal")
    validations = tuple(
        FreshProcessValidation(
            process_id=int(row["process_id"]),
            policy_seal_hash=str(row["policy_seal_hash"]),
            source_tree_sha256=str(row["source_tree_sha256"]),
            reconstruction_hash=str(row["reconstruction_hash"]),
            receipt_hash=str(row["receipt_hash"]),
        )
        for row in rows
    )
    return DurablePreterminalAttestation(
        policy_seal_hash=validations[0].policy_seal_hash,
        validations=validations,  # type: ignore[arg-type]
    )


def require_two_fresh_final_validations(
    root: str | Path,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    first, second = _run_two(Path(root), phase="final")
    if first["reconstruction_hash"] != second["reconstruction_hash"]:
        raise ProtocolError("SCEPTRE v5 final fresh validators disagree.")
    return first, second


def _run_two(root: Path, *, phase: str) -> tuple[Mapping[str, object], Mapping[str, object]]:
    command = [
        sys.executable,
        "-m",
        "midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5.execution.fresh_validation",
        "--phase",
        phase,
        "--root",
        str(root.resolve()),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    rows = []
    for _ in range(2):
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=180,
        )
        if completed.returncode != 0:
            raise ProtocolError(
                "SCEPTRE v5 fresh validation failed: "
                + completed.stderr.strip()[-1000:]
            )
        try:
            row = json.loads(completed.stdout.strip())
        except json.JSONDecodeError as exc:
            raise ProtocolError("SCEPTRE v5 fresh validation output drifted.") from exc
        if not isinstance(row, dict):
            raise ProtocolError("SCEPTRE v5 fresh validation output is malformed.")
        rows.append(row)
    if rows[0].get("process_id") == rows[1].get("process_id"):
        raise ProtocolError("SCEPTRE v5 fresh validators reused a process.")
    return rows[0], rows[1]


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preterminal", "final"), required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    if args.phase == "preterminal":
        reconstructed = dict(validate_preterminal_bundle(args.root))
        validation = FreshProcessValidation(
            process_id=os.getpid(),
            policy_seal_hash=str(reconstructed["policy_seal_hash"]),
            source_tree_sha256=str(reconstructed["source_tree_sha256"]),
            reconstruction_hash=str(reconstructed["reconstruction_hash"]),
        )
        payload = {
            "process_id": validation.process_id,
            "policy_seal_hash": validation.policy_seal_hash,
            "source_tree_sha256": validation.source_tree_sha256,
            "reconstruction_hash": validation.reconstruction_hash,
            "receipt_hash": validation.receipt_hash,
        }
    else:
        reconstructed = dict(validate_final_bundle(args.root))
        payload = {
            "process_id": os.getpid(),
            "phase": "final",
            "reconstruction_hash": reconstructed["reconstruction_hash"],
            "terminal_result_hash": reconstructed["terminal_result_hash"],
            "cuda_hidden": True,
            "thread_count": 1,
        }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entrypoint
    raise SystemExit(_main())


__all__ = (
    "require_two_fresh_final_validations",
    "require_two_fresh_preterminal_validations",
)
