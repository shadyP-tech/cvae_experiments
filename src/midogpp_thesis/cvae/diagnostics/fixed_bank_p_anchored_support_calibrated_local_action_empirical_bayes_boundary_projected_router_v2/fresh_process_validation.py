"""Two sequential independent artifact-only v2 validation attestations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

from .artifacts.hashing import canonical_hash, canonical_json
from .artifacts.io import atomic_json, member_path, read_json_object
from .protocol import GovernanceError
from .execution.workstation import NO_REFIT_ENV, THREAD_ENVIRONMENT


WORKER_MODULE = (
    "midogpp_thesis.cvae.diagnostics."
    "fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_"
    "boundary_projected_router_v2.fresh_process_validation"
)
FRESH_PROCESS_COUNT = 2
FRESH_PROCESS_TIMEOUT_SECONDS = 21_600


def require_two_fresh_process_attestations(
    root: str | Path,
    *,
    phase: str = "final",
    expected_checks: Mapping[str, object] | None = None,
    timeout_seconds: int = FRESH_PROCESS_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Run two new CUDA-free Python interpreters; neither may refit science."""

    path = Path(root).resolve()
    if phase not in {"preterminal", "final"}:
        raise GovernanceError("SCALE-BP v2 fresh-process phase drifted.")
    expected = (
        _parent_checks(path, phase=phase)
        if expected_checks is None
        else dict(expected_checks)
    )
    if expected.get("status") != "PASS" or expected.get("scientific_refit_performed") is not False:
        raise GovernanceError("SCALE-BP v2 cannot attest non-PASS/refit checks.")
    expected_hash = canonical_hash(expected)
    results: list[dict[str, object]] = []
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONHASHSEED"] = "0"
    environment[NO_REFIT_ENV] = "1"
    environment.update(THREAD_ENVIRONMENT)
    for ordinal in range(1, FRESH_PROCESS_COUNT + 1):
        try:
            completed = subprocess.run(
                (
                    sys.executable,
                    "-m",
                    WORKER_MODULE,
                    "--worker",
                    str(path),
                    "--phase",
                    phase,
                ),
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=int(timeout_seconds),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GovernanceError(
                "SCALE-BP v2 fresh-process validation could not complete."
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-2_000:]
            raise GovernanceError(
                f"SCALE-BP v2 fresh validator {ordinal} failed: {detail or 'no stderr'}."
            )
        payload = _worker_json(completed.stdout, ordinal=ordinal)
        checks = payload.get("checks")
        process_id = payload.get("process_id")
        if (
            not isinstance(checks, Mapping)
            or dict(checks) != expected
            or canonical_hash(checks) != expected_hash
            or type(process_id) is not int
            or payload.get("phase") != phase
            or payload.get("scientific_refit_performed") is not False
        ):
            raise GovernanceError("SCALE-BP v2 fresh validator disagreed with parent.")
        results.append(
            {
                "ordinal": ordinal,
                "process_id": process_id,
                "exit_code": 0,
                "result_hash": canonical_hash(payload),
                "reconstructed_checks_hash": expected_hash,
            }
        )
    child_pids = [int(row["process_id"]) for row in results]
    if len(set(child_pids)) != FRESH_PROCESS_COUNT or os.getpid() in child_pids:
        raise GovernanceError("SCALE-BP v2 validators are not independent processes.")
    body = {
        "schema_version": f"scale_bp_v2_{phase}_fresh_process_attestation_v1",
        "status": "PASS",
        "phase": phase,
        "fresh_python_process_count": FRESH_PROCESS_COUNT,
        "independent_fresh_python_processes": True,
        "process_launches_sequential": True,
        "parent_process_id": os.getpid(),
        "child_process_ids": child_pids,
        "child_process_results": results,
        "subprocess_exit_codes": [0, 0],
        "cuda_visible_devices": "",
        "worker_thread_environment": dict(THREAD_ENVIRONMENT),
        "pythonhashseed": "0",
        "artifact_only_reconstruction": True,
        "scientific_refit_performed": False,
        "fitted_model_deserialization_performed": False,
        "reconstructed_checks_exactly_equal": True,
        "reconstructed_checks_hash": expected_hash,
        "validator_entrypoint": (
            "validate_preterminal_bundle" if phase == "preterminal" else "validate_final_bundle"
        ),
    }
    attestation = {**body, "attestation_hash": canonical_hash(body)}
    atomic_json(
        member_path(path, f"reports/{phase}_fresh_process_attestation.json"),
        attestation,
    )
    return attestation


def validate_fresh_process_attestation(
    root: str | Path,
    *,
    phase: str,
    expected_checks: Mapping[str, object],
) -> dict[str, object]:
    if phase not in {"preterminal", "final"}:
        raise GovernanceError("SCALE-BP v2 attestation phase drifted.")
    payload = read_json_object(
        member_path(root, f"reports/{phase}_fresh_process_attestation.json")
    )
    body = {key: value for key, value in payload.items() if key != "attestation_hash"}
    results = payload.get("child_process_results")
    pids = payload.get("child_process_ids")
    expected_hash = canonical_hash(dict(expected_checks))
    if (
        payload.get("schema_version") != f"scale_bp_v2_{phase}_fresh_process_attestation_v1"
        or payload.get("status") != "PASS"
        or payload.get("phase") != phase
        or payload.get("fresh_python_process_count") != FRESH_PROCESS_COUNT
        or payload.get("independent_fresh_python_processes") is not True
        or payload.get("process_launches_sequential") is not True
        or not isinstance(pids, list)
        or len(pids) != FRESH_PROCESS_COUNT
        or len(set(pids)) != FRESH_PROCESS_COUNT
        or not isinstance(results, list)
        or len(results) != FRESH_PROCESS_COUNT
        or [row.get("process_id") for row in results if isinstance(row, Mapping)] != pids
        or any(
            not isinstance(row, Mapping)
            or row.get("ordinal") != ordinal
            or row.get("exit_code") != 0
            or row.get("reconstructed_checks_hash") != expected_hash
            for ordinal, row in enumerate(results, start=1)
        )
        or payload.get("subprocess_exit_codes") != [0, 0]
        or payload.get("cuda_visible_devices") != ""
        or payload.get("worker_thread_environment") != THREAD_ENVIRONMENT
        or payload.get("pythonhashseed") != "0"
        or payload.get("artifact_only_reconstruction") is not True
        or payload.get("scientific_refit_performed") is not False
        or payload.get("fitted_model_deserialization_performed") is not False
        or payload.get("reconstructed_checks_exactly_equal") is not True
        or payload.get("reconstructed_checks_hash") != expected_hash
        or payload.get("attestation_hash") != canonical_hash(body)
    ):
        raise GovernanceError("SCALE-BP v2 fresh-process attestation drifted.")
    return payload


def _parent_checks(root: Path, *, phase: str) -> dict[str, object]:
    if phase == "preterminal":
        from .validation import validate_preterminal_bundle

        return validate_preterminal_bundle(root, no_refit=True)
    from .validation import validate_final_bundle

    return validate_final_bundle(root, no_refit=True, require_fresh_attestation=False)


def _worker_payload(root: Path, *, phase: str) -> dict[str, object]:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get(NO_REFIT_ENV) != "1"
        or any(os.environ.get(key) != value for key, value in THREAD_ENVIRONMENT.items())
    ):
        raise GovernanceError("SCALE-BP v2 fresh worker environment drifted.")
    checks = _parent_checks(root.resolve(), phase=phase)
    return {
        "process_id": os.getpid(),
        "phase": phase,
        "checks": checks,
        "scientific_refit_performed": False,
    }


def _worker_json(stdout: str, *, ordinal: int) -> dict[str, object]:
    try:
        payload = json.loads(stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise GovernanceError(
            f"SCALE-BP v2 fresh validator {ordinal} emitted invalid JSON."
        ) from exc
    if not isinstance(payload, dict) or stdout.strip() != canonical_json(payload):
        raise GovernanceError(
            f"SCALE-BP v2 fresh validator {ordinal} JSON is not canonical."
        )
    return payload


def _main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--phase", choices=("preterminal", "final"), required=True)
    args = parser.parse_args(tuple(argv))
    payload = _worker_payload(Path(args.worker), phase=args.phase)
    sys.stdout.write(canonical_json(payload) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a fresh process
    raise SystemExit(_main(sys.argv[1:]))


__all__ = (
    "FRESH_PROCESS_COUNT",
    "require_two_fresh_process_attestations",
    "validate_fresh_process_attestation",
)
