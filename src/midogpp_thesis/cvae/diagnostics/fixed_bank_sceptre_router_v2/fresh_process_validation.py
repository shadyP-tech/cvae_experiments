"""Two sequential CUDA-hidden fresh-process validators for SCEPTRE v2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from midogpp_thesis.cvae.protocol import ProtocolError

from ..fixed_bank_sceptre_router.hashing import canonical_hash
from ..fixed_bank_sceptre_router.seals import (
    DurablePreterminalAttestation,
    FreshProcessValidation,
)
from .validation import validate_final_bundle, validate_preterminal_bundle


WORKER_MODULE = (
    "midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2."
    "fresh_process_validation"
)
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def require_two_fresh_preterminal_validations(
    root: str | Path,
) -> DurablePreterminalAttestation:
    checks, children = _run_two(Path(root), phase="preterminal")
    validations = tuple(
        FreshProcessValidation(
            process_id=int(row["process_id"]),
            policy_seal_hash=str(checks["policy_seal_hash"]),
            source_tree_sha256=str(checks["source_tree_sha256"]),
            reconstruction_hash=str(checks["reconstruction_hash"]),
        )
        for row in children
    )
    return DurablePreterminalAttestation(
        policy_seal_hash=str(checks["policy_seal_hash"]),
        validations=validations,  # type: ignore[arg-type]
    )


def require_two_fresh_final_validations(root: str | Path) -> dict[str, object]:
    checks, children = _run_two(Path(root), phase="final")
    base = {
        "schema_version": "sceptre_v2_final_fresh_process_attestation_v1",
        "status": "PASS",
        "validator_process_ids": [row["process_id"] for row in children],
        "validator_result_hashes": [row["result_hash"] for row in children],
        "checks": checks,
        "checks_hash": canonical_hash(checks),
        "fresh_process_count": 2,
        "process_launches_sequential": True,
        "cuda_hidden": True,
        "thread_count": 1,
        "semantic_reconstruction_without_refit": True,
        "raw_labels_read": False,
    }
    return {**base, "attestation_hash": canonical_hash(base)}


def _run_two(
    root: Path, *, phase: str
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    environment = dict(os.environ)
    environment.update(THREAD_ENVIRONMENT)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONHASHSEED": "0",
            "SCEPTRE_NO_REFIT": "1",
        }
    )
    observed: dict[str, object] | None = None
    children: list[dict[str, object]] = []
    for ordinal in (1, 2):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                WORKER_MODULE,
                "--worker",
                str(root.resolve()),
                "--phase",
                phase,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=21_600,
        )
        if completed.returncode != 0:
            raise ProtocolError(
                f"SCEPTRE v2 fresh validator {ordinal} failed: "
                f"{completed.stderr.strip()[-2000:]}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ProtocolError("SCEPTRE v2 validator emitted invalid JSON.") from exc
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        checks = payload.get("checks") if isinstance(payload, dict) else None
        process_id = payload.get("process_id") if isinstance(payload, dict) else None
        if (
            completed.stdout.strip() != canonical
            or not isinstance(checks, dict)
            or not isinstance(process_id, int)
            or process_id == os.getpid()
            or payload.get("validation_phase") != phase
            or (observed is not None and checks != observed)
        ):
            raise ProtocolError("SCEPTRE v2 fresh validators disagreed.")
        observed = checks
        children.append(
            {
                "ordinal": ordinal,
                "process_id": process_id,
                "result_hash": canonical_hash(payload),
            }
        )
    if observed is None or len({row["process_id"] for row in children}) != 2:
        raise ProtocolError("SCEPTRE v2 validators were not independent.")
    return observed, tuple(children)


def _worker(root: Path, *, phase: str) -> int:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("SCEPTRE_NO_REFIT") != "1"
        or any(
            os.environ.get(key) != value
            for key, value in THREAD_ENVIRONMENT.items()
        )
    ):
        raise ProtocolError("SCEPTRE v2 validator environment drifted.")
    checks = (
        validate_preterminal_bundle(root)
        if phase == "preterminal"
        else validate_final_bundle(root)
    )
    payload = {
        "process_id": os.getpid(),
        "validation_phase": phase,
        "checks": checks,
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--phase", choices=("preterminal", "final"), required=True)
    args = parser.parse_args(argv)
    return _worker(args.worker.resolve(), phase=args.phase)


if __name__ == "__main__":  # pragma: no cover - subprocess entrypoint
    raise SystemExit(main())


__all__ = (
    "require_two_fresh_final_validations",
    "require_two_fresh_preterminal_validations",
)
