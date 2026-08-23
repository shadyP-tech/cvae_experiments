"""Two sequential CUDA-free fresh-process validations for P-DCAPS v4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.lifecycle import (
    DurablePreterminalAttestation,
)
from .identity import canonical_hash
from .validation import validate_final_bundle, validate_preterminal_bundle


WORKER_MODULE = (
    "midogpp_thesis.cvae.diagnostics."
    "fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_"
    "router_v4.fresh_process_validation"
)
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def require_two_fresh_preterminal_validations(
    root: Path,
) -> DurablePreterminalAttestation:
    checks, children = _run_two(Path(root), phase="preterminal")
    return DurablePreterminalAttestation(
        preterminal_seal_hash=str(checks["preterminal_seal_hash"]),
        validator_process_ids=tuple(row["process_id"] for row in children),
        validator_result_hashes=tuple(row["result_hash"] for row in children),
        durable_bundle_hash=str(checks["content_index_hash"]),
    )


def require_two_fresh_final_validations(root: Path) -> dict[str, object]:
    checks, children = _run_two(Path(root), phase="final")
    base = {
        "schema_version": "pdcaps_v4_final_fresh_process_attestation_v1",
        "status": "PASS",
        "fresh_python_process_count": 2,
        "independent_fresh_python_processes": True,
        "process_launches_sequential": True,
        "cuda_visible_devices": "",
        "worker_thread_environment": THREAD_ENVIRONMENT,
        "parent_process_id": os.getpid(),
        "validator_process_ids": [row["process_id"] for row in children],
        "validator_result_hashes": [row["result_hash"] for row in children],
        "reconstructed_checks": checks,
        "reconstructed_checks_hash": canonical_hash(checks),
        "semantic_reconstruction_without_refit": True,
        "formal_claim_authorized": False,
    }
    payload = {**base, "attestation_hash": canonical_hash(base)}
    atomic_json(
        Path(root) / "reports/final_fresh_process_attestation.json", payload
    )
    return payload


def _run_two(
    root: Path, *, phase: str
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    path = root.resolve()
    environment = dict(os.environ)
    environment.update(THREAD_ENVIRONMENT)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    observed_checks: dict[str, object] | None = None
    children: list[dict[str, object]] = []
    for ordinal in (1, 2):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                WORKER_MODULE,
                "--worker",
                str(path),
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
                f"P-DCAPS v4 fresh validation {ordinal} failed: "
                f"{completed.stderr.strip()[-2000:]}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ProtocolError(
                "P-DCAPS v4 fresh validator emitted invalid JSON."
            ) from exc
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        checks = payload.get("checks") if isinstance(payload, dict) else None
        process_id = payload.get("process_id") if isinstance(payload, dict) else None
        if (
            completed.stdout.strip() != canonical
            or not isinstance(checks, dict)
            or not isinstance(process_id, int)
            or process_id == os.getpid()
            or payload.get("validation_phase") != phase
            or (observed_checks is not None and checks != observed_checks)
        ):
            raise ProtocolError("P-DCAPS v4 fresh validators disagreed.")
        observed_checks = checks
        children.append(
            {
                "ordinal": ordinal,
                "process_id": process_id,
                "result_hash": canonical_hash(payload),
            }
        )
    if (
        observed_checks is None
        or len({row["process_id"] for row in children}) != 2
    ):
        raise ProtocolError("P-DCAPS v4 validators were not independent.")
    return observed_checks, tuple(children)


def _worker(root: Path, *, phase: str) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(
        os.environ.get(key) != value for key, value in THREAD_ENVIRONMENT.items()
    ):
        raise ProtocolError("P-DCAPS v4 fresh validator environment drifted.")
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


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())


__all__ = (
    "require_two_fresh_final_validations",
    "require_two_fresh_preterminal_validations",
)
