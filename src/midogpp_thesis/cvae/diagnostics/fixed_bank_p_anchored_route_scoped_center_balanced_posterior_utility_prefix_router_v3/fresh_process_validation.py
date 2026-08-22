"""Two sequential CUDA-free fresh-process bundle reconstructions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

from ...protocol import ProtocolError
from .artifact_io import persist_json
from .hashing import canonical_hash, canonical_json
from .preterminal_gate import FRESH_WORKER_THREAD_ENVIRONMENT
from .preterminal_validation import validate_preterminal_bundle
from .validation import (
    validate_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_bundle,
)


WORKER_MODULE = (
    "midogpp_thesis.cvae.diagnostics."
    "fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3."
    "fresh_process_validation"
)
THREAD_ENVIRONMENT = FRESH_WORKER_THREAD_ENVIRONMENT


def require_two_fresh_process_validations(
    root: str | Path, *, expected_checks: Mapping[str, object]
) -> dict[str, object]:
    return _require_two_fresh_process_validations(
        Path(root).resolve(), expected_checks=expected_checks, phase="final"
    )


def require_two_fresh_preterminal_process_validations(
    root: str | Path, *, expected_checks: Mapping[str, object]
) -> dict[str, object]:
    return _require_two_fresh_process_validations(
        Path(root).resolve(), expected_checks=expected_checks, phase="preterminal"
    )


def _require_two_fresh_process_validations(
    path: Path,
    *,
    expected_checks: Mapping[str, object],
    phase: str,
) -> dict[str, object]:
    expected = dict(expected_checks)
    expected_hash = canonical_hash(expected)
    children = []
    environment = dict(os.environ)
    environment.update(THREAD_ENVIRONMENT)
    environment["CUDA_VISIBLE_DEVICES"] = ""
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
                f"CBPUPR fresh validation {ordinal} failed: "
                f"{completed.stderr.strip()[-2000:]}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ProtocolError("CBPUPR fresh validator emitted invalid JSON.") from exc
        if (
            not isinstance(payload, dict)
            or completed.stdout.strip() != canonical_json(payload)
            or payload.get("checks") != expected
            or canonical_hash(payload["checks"]) != expected_hash
            or not isinstance(payload.get("process_id"), int)
            or (
                phase == "preterminal"
                and payload.get("validation_phase") != "preterminal"
            )
            or (
                phase == "final"
                and "validation_phase" in payload
            )
        ):
            raise ProtocolError("CBPUPR fresh validator disagreed with parent.")
        children.append(
            {
                "ordinal": ordinal,
                "process_id": payload["process_id"],
                "exit_code": 0,
                "result_hash": canonical_hash(payload),
            }
        )
    pids = [row["process_id"] for row in children]
    if len(set(pids)) != 2 or os.getpid() in pids:
        raise ProtocolError("CBPUPR validators were not independent processes.")
    base = {
        "schema_version": (
            "fixed_bank_cbpupr_preterminal_fresh_process_attestation_v1"
            if phase == "preterminal"
            else "fixed_bank_cbpupr_fresh_process_attestation_v1"
        ),
        "status": "PASS",
        "fresh_python_process_count": 2,
        "independent_fresh_python_processes": True,
        "process_launches_sequential": True,
        "cuda_visible_devices": "",
        "worker_thread_environment": THREAD_ENVIRONMENT,
        "parent_process_id": os.getpid(),
        "child_process_ids": pids,
        "child_process_results": children,
        "reconstructed_checks_exactly_equal": True,
        "reconstructed_checks_hash": expected_hash,
        "validator_entrypoint": (
            "validate_preterminal_bundle"
            if phase == "preterminal"
            else (
                "validate_p_anchored_route_scoped_center_balanced_"
                "posterior_utility_prefix_router_bundle"
            )
        ),
    }
    if phase == "preterminal":
        base = {
            **base,
            "validation_phase": "preterminal",
            "terminal_opened": False,
        }
    attestation = {**base, "attestation_hash": canonical_hash(base)}
    attestation_path = (
        path / "reports/preterminal_fresh_process_attestation.json"
        if phase == "preterminal"
        else path / "reports/fresh_process_attestation.json"
    )
    persist_json(attestation_path, attestation)
    return attestation


def validation_report_payload(
    checks: Mapping[str, object], attestation: Mapping[str, object]
) -> dict[str, object]:
    payload = {
        "schema_version": "fixed_bank_cbpupr_validation_report_v1",
        "status": "PASS",
        "checks": dict(checks),
        "fresh_process_attestation_hash": attestation["attestation_hash"],
        "formal_claim_authorized": False,
    }
    return {**payload, "validation_report_hash": canonical_hash(payload)}


def _worker(root: Path, *, phase: str) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(
        os.environ.get(key) != value for key, value in THREAD_ENVIRONMENT.items()
    ):
        raise ProtocolError("CBPUPR fresh worker environment drifted.")
    if phase == "preterminal":
        checks = validate_preterminal_bundle(root, require_attested=False)
        payload = {
            "process_id": os.getpid(),
            "validation_phase": "preterminal",
            "checks": checks,
        }
    elif phase == "final":
        checks = validate_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_bundle(
            root, require_final=False
        )
        payload = {"process_id": os.getpid(), "checks": checks}
    else:  # pragma: no cover - argparse constrains the public entrypoint
        raise ProtocolError("CBPUPR fresh validation phase drifted.")
    sys.stdout.write(canonical_json(payload))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("preterminal", "final"), default="final"
    )
    args = parser.parse_args(argv)
    return _worker(args.worker.resolve(), phase=args.phase)


if __name__ == "__main__":  # pragma: no cover - exercised through spawn
    raise SystemExit(main())


__all__ = (
    "require_two_fresh_preterminal_process_validations",
    "require_two_fresh_process_validations",
    "validation_report_payload",
)
