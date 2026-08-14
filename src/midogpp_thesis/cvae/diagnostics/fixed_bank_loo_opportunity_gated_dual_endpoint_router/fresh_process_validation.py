"""Two sequential, independent, CUDA-free full-bundle validation replays."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from .hashing import canonical_hash, canonical_json


ATTESTATION_KEY = "fresh_process_validation_attestation"
ATTESTATION_SCHEMA = "fixed_bank_dual_endpoint_fresh_process_validation_v1"
WORKER_MODULE = (
    "midogpp_thesis.cvae.diagnostics."
    "fixed_bank_loo_opportunity_gated_dual_endpoint_router.fresh_process_validation"
)
VALIDATOR_ENTRYPOINT = (
    "validate_fixed_bank_loo_opportunity_gated_dual_endpoint_router_bundle"
)
FRESH_PROCESS_TIMEOUT_SECONDS = 21_600
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
ATTESTATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "fresh_python_process_count",
        "independent_fresh_python_processes",
        "process_launches_sequential",
        "persisted_resolved_config_loaded_by_each_process",
        "full_scientific_reconstruction_called_by_each_process",
        "pending_validation_allowed",
        "cuda_visible_devices",
        "outer_blas_threads",
        "fitted_reconstruction_blas_threads",
        "worker_thread_environment",
        "parent_process_id",
        "child_process_ids",
        "child_process_results",
        "subprocess_exit_codes",
        "reconstructed_check_payloads_exactly_equal",
        "reconstructed_check_hash",
        "validator_entrypoint",
        "attestation_hash",
    }
)
CHILD_RESULT_KEYS = frozenset(
    {"ordinal", "process_id", "exit_code", "result_hash", "reconstructed_check_hash"}
)


def require_two_fresh_process_validations(
    root: str | Path,
    *,
    expected_checks: Mapping[str, object],
) -> Mapping[str, object]:
    path = Path(root).resolve()
    expected = _base_checks(expected_checks)
    expected_hash = canonical_hash(expected)
    results: list[dict[str, object]] = []
    for ordinal in (1, 2):
        completed = _run_worker(path)
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-2_000:]
            raise ProtocolError(
                f"Dual-endpoint fresh replay {ordinal} failed: {detail or 'no stderr'}."
            )
        payload = _worker_json(completed.stdout, ordinal=ordinal)
        checks = payload.get("checks")
        pid = payload.get("process_id")
        if (
            not isinstance(checks, Mapping)
            or not isinstance(pid, int)
            or isinstance(pid, bool)
            or dict(checks) != expected
            or canonical_hash(checks) != expected_hash
        ):
            raise ProtocolError("Dual-endpoint fresh replay disagreed with parent.")
        results.append(
            {
                "ordinal": ordinal,
                "process_id": pid,
                "exit_code": 0,
                "result_hash": canonical_hash(payload),
                "reconstructed_check_hash": expected_hash,
            }
        )
    child_pids = [int(row["process_id"]) for row in results]
    if len(set(child_pids)) != 2 or os.getpid() in child_pids:
        raise ProtocolError("Dual-endpoint fresh replay PIDs are not independent.")
    unhashed = {
        "schema_version": ATTESTATION_SCHEMA,
        "status": "PASS",
        "fresh_python_process_count": 2,
        "independent_fresh_python_processes": True,
        "process_launches_sequential": True,
        "persisted_resolved_config_loaded_by_each_process": True,
        "full_scientific_reconstruction_called_by_each_process": True,
        "pending_validation_allowed": True,
        "cuda_visible_devices": "",
        "outer_blas_threads": 1,
        "fitted_reconstruction_blas_threads": 3,
        "worker_thread_environment": dict(THREAD_ENVIRONMENT),
        "parent_process_id": os.getpid(),
        "child_process_ids": child_pids,
        "child_process_results": results,
        "subprocess_exit_codes": [0, 0],
        "reconstructed_check_payloads_exactly_equal": True,
        "reconstructed_check_hash": expected_hash,
        "validator_entrypoint": VALIDATOR_ENTRYPOINT,
    }
    attestation = {**unhashed, "attestation_hash": canonical_hash(unhashed)}
    atomic_json(path / "reports/fresh_process_attestation.json", attestation)
    return {**expected, ATTESTATION_KEY: attestation}


def verify_attested_validation_checks(
    checks: Mapping[str, object],
    *,
    expected_reconstructed_checks: Mapping[str, object],
    persisted_attestation: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    payload = dict(checks)
    attestation = payload.pop(ATTESTATION_KEY, None)
    expected = _base_checks(expected_reconstructed_checks)
    if payload != expected or not isinstance(attestation, Mapping):
        raise ProtocolError("Dual-endpoint validation report is not reconstructive.")
    unhashed = {
        key: value for key, value in attestation.items() if key != "attestation_hash"
    }
    pids = attestation.get("child_process_ids")
    results = attestation.get("child_process_results")
    expected_hash = canonical_hash(expected)
    if (
        set(attestation) != ATTESTATION_KEYS
        or attestation.get("schema_version") != ATTESTATION_SCHEMA
        or attestation.get("status") != "PASS"
        or attestation.get("fresh_python_process_count") != 2
        or attestation.get("independent_fresh_python_processes") is not True
        or attestation.get("process_launches_sequential") is not True
        or attestation.get("persisted_resolved_config_loaded_by_each_process")
        is not True
        or attestation.get("full_scientific_reconstruction_called_by_each_process")
        is not True
        or attestation.get("pending_validation_allowed") is not True
        or attestation.get("cuda_visible_devices") != ""
        or attestation.get("outer_blas_threads") != 1
        or attestation.get("fitted_reconstruction_blas_threads") != 3
        or attestation.get("worker_thread_environment") != THREAD_ENVIRONMENT
        or not isinstance(pids, list)
        or len(pids) != 2
        or any(not isinstance(pid, int) or isinstance(pid, bool) for pid in pids)
        or len(set(pids)) != 2
        or not isinstance(attestation.get("parent_process_id"), int)
        or isinstance(attestation.get("parent_process_id"), bool)
        or attestation.get("parent_process_id") in pids
        or not isinstance(results, list)
        or len(results) != 2
        or [row.get("process_id") for row in results if isinstance(row, Mapping)]
        != pids
        or any(
            not isinstance(row, Mapping)
            or set(row) != CHILD_RESULT_KEYS
            or row.get("ordinal") != ordinal
            or row.get("exit_code") != 0
            or row.get("reconstructed_check_hash") != expected_hash
            or row.get("result_hash")
            != canonical_hash(
                {"process_id": row.get("process_id"), "checks": expected}
            )
            for ordinal, row in enumerate(results, start=1)
        )
        or attestation.get("subprocess_exit_codes") != [0, 0]
        or attestation.get("reconstructed_check_payloads_exactly_equal") is not True
        or attestation.get("reconstructed_check_hash") != expected_hash
        or attestation.get("validator_entrypoint") != VALIDATOR_ENTRYPOINT
        or attestation.get("attestation_hash") != canonical_hash(unhashed)
        or (
            persisted_attestation is not None
            and dict(attestation) != dict(persisted_attestation)
        )
    ):
        raise ProtocolError("Dual-endpoint fresh-process attestation drifted.")
    return {**expected, ATTESTATION_KEY: dict(attestation)}


def _run_worker(root: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.update(THREAD_ENVIRONMENT)
    environment["PYTHONHASHSEED"] = "0"
    try:
        return subprocess.run(
            (sys.executable, "-m", WORKER_MODULE, "--worker", str(root)),
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=FRESH_PROCESS_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProtocolError("Dual-endpoint fresh replay could not complete.") from exc


def _worker_json(stdout: str, *, ordinal: int) -> Mapping[str, object]:
    try:
        payload = json.loads(stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            f"Dual-endpoint fresh replay {ordinal} emitted invalid JSON."
        ) from exc
    if not isinstance(payload, Mapping) or stdout.strip() != canonical_json(payload):
        raise ProtocolError(
            f"Dual-endpoint fresh replay {ordinal} JSON is not canonical."
        )
    return payload


def _base_checks(checks: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(checks, Mapping) or ATTESTATION_KEY in checks:
        raise ProtocolError("Dual-endpoint pending checks are malformed.")
    payload = dict(checks)
    if payload.get("status") != "PASS":
        raise ProtocolError("Dual-endpoint cannot attest non-PASS checks.")
    canonical_json(payload)
    return payload


def _worker_payload(root: Path) -> Mapping[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(
        os.environ.get(key) != value for key, value in THREAD_ENVIRONMENT.items()
    ):
        raise ProtocolError("Dual-endpoint fresh worker environment drifted.")
    path = root.resolve()
    config_path = path / "config.resolved.yaml"
    from .config import load_fixed_bank_loo_opportunity_gated_dual_endpoint_router_config
    from .validation import validate_fixed_bank_loo_opportunity_gated_dual_endpoint_router_bundle

    config = load_fixed_bank_loo_opportunity_gated_dual_endpoint_router_config(config_path)
    if (
        Path(getattr(config, "source_path")).resolve() != config_path
        or Path(getattr(config, "artifact_root")).resolve() != path
    ):
        raise ProtocolError("Dual-endpoint fresh config/root binding drifted.")
    checks = validate_fixed_bank_loo_opportunity_gated_dual_endpoint_router_bundle(
        path, config=config, allow_pending_validation=True
    )
    return {"process_id": os.getpid(), "checks": checks}


def _main(argv: Sequence[str]) -> int:
    if len(argv) != 2 or argv[0] != "--worker":
        raise ProtocolError("Dual-endpoint fresh module is worker-only.")
    payload = _worker_payload(Path(argv[1]))
    sys.stdout.write(canonical_json(payload) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))


__all__ = (
    "ATTESTATION_KEY",
    "ATTESTATION_SCHEMA",
    "require_two_fresh_process_validations",
    "verify_attested_validation_checks",
)
