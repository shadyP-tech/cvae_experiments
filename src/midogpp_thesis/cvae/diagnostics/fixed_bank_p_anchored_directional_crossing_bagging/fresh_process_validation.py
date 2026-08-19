"""Two sequential CUDA-free reconstruction attestations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .artifact_io import persist_json
from .hashing import canonical_hash, canonical_json


ATTESTATION_KEY = "fresh_process_validation_attestation"
ATTESTATION_SCHEMA = "fixed_bank_pdcb_fresh_process_validation_v1"
WORKER_MODULE = (
    "midogpp_thesis.cvae.diagnostics."
    "fixed_bank_p_anchored_directional_crossing_bagging.fresh_process_validation"
)
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
TIMEOUT_SECONDS = 21_600
VALIDATOR_ENTRYPOINT = "validate_p_anchored_directional_crossing_bagging_bundle"
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
        "route_worker_blas_threads",
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
    {
        "ordinal",
        "process_id",
        "exit_code",
        "result_hash",
        "reconstructed_check_hash",
    }
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
                f"PDCB fresh replay {ordinal} failed: {detail or 'no stderr'}."
            )
        try:
            payload = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ProtocolError("PDCB fresh replay emitted invalid JSON.") from exc
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"process_id", "checks"}
            or completed.stdout.strip() != canonical_json(payload)
            or not isinstance(payload.get("process_id"), int)
            or isinstance(payload.get("process_id"), bool)
            or not isinstance(payload.get("checks"), Mapping)
            or canonical_hash(payload["checks"]) != expected_hash
            or dict(payload["checks"]) != expected
        ):
            raise ProtocolError("PDCB fresh replay disagreed with parent.")
        results.append(
            {
                "ordinal": ordinal,
                "process_id": int(payload["process_id"]),
                "exit_code": 0,
                "result_hash": canonical_hash(payload),
                "reconstructed_check_hash": expected_hash,
            }
        )
    child_pids = [int(row["process_id"]) for row in results]
    if len(set(child_pids)) != 2 or os.getpid() in child_pids:
        raise ProtocolError("PDCB validation processes are not independent.")
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
        "route_worker_blas_threads": 3,
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
    persist_json(path / "reports/fresh_process_attestation.json", attestation)
    return {**expected, ATTESTATION_KEY: attestation}


def verify_attested_validation(
    report: Mapping[str, object],
    *,
    expected_checks: Mapping[str, object],
    persisted_attestation: Mapping[str, object],
) -> Mapping[str, object]:
    expected = _base_checks(expected_checks)
    payload = dict(report)
    raw_attestation = payload.pop(ATTESTATION_KEY, None)
    if payload != expected or not isinstance(raw_attestation, Mapping):
        raise ProtocolError("PDCB validation report is not reconstructive.")
    attestation = dict(raw_attestation)
    if set(attestation) != ATTESTATION_KEYS:
        raise ProtocolError("PDCB fresh-process attestation schema drifted.")
    unhashed = {
        key: value for key, value in attestation.items() if key != "attestation_hash"
    }
    results = attestation.get("child_process_results")
    pids = attestation.get("child_process_ids")
    expected_hash = canonical_hash(expected)
    if (
        attestation != dict(persisted_attestation)
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
        or attestation.get("route_worker_blas_threads") != 3
        or attestation.get("worker_thread_environment") != THREAD_ENVIRONMENT
        or not isinstance(attestation.get("parent_process_id"), int)
        or isinstance(attestation.get("parent_process_id"), bool)
        or not isinstance(pids, list)
        or len(pids) != 2
        or any(not isinstance(pid, int) or isinstance(pid, bool) for pid in pids)
        or len(set(pids)) != 2
        or attestation.get("parent_process_id") in pids
        or not isinstance(results, list)
        or len(results) != 2
        or [row.get("process_id") for row in results if isinstance(row, Mapping)]
        != pids
        or attestation.get("subprocess_exit_codes") != [0, 0]
        or attestation.get("reconstructed_check_hash") != expected_hash
        or attestation.get("reconstructed_check_payloads_exactly_equal") is not True
        or attestation.get("validator_entrypoint") != VALIDATOR_ENTRYPOINT
        or attestation.get("attestation_hash") != canonical_hash(unhashed)
    ):
        raise ProtocolError("PDCB fresh-process attestation drifted.")
    for ordinal, row in enumerate(results, start=1):
        if (
            not isinstance(row, Mapping)
            or set(row) != CHILD_RESULT_KEYS
            or row.get("ordinal") != ordinal
            or row.get("exit_code") != 0
            or row.get("reconstructed_check_hash") != expected_hash
            or row.get("process_id") != pids[ordinal - 1]
            or row.get("result_hash")
            != canonical_hash(
                {
                    "process_id": row.get("process_id"),
                    "checks": expected,
                }
            )
        ):
            raise ProtocolError("PDCB fresh-process result drifted.")
    return {**expected, ATTESTATION_KEY: attestation}


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
            timeout=TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProtocolError("PDCB fresh replay could not complete.") from exc


def _base_checks(checks: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(checks, Mapping) or ATTESTATION_KEY in checks:
        raise ProtocolError("PDCB pending checks are malformed.")
    payload = dict(checks)
    if payload.get("status") != "PASS":
        raise ProtocolError("PDCB cannot attest non-PASS checks.")
    canonical_json(payload)
    return payload


def _worker_payload(root: Path) -> Mapping[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(
        os.environ.get(key) != value for key, value in THREAD_ENVIRONMENT.items()
    ):
        raise ProtocolError("PDCB fresh worker environment drifted.")
    path = root.resolve()
    from .config import load_p_anchored_directional_crossing_bagging_config
    from .validation import validate_p_anchored_directional_crossing_bagging_bundle

    config = load_p_anchored_directional_crossing_bagging_config(
        path / "config.resolved.yaml"
    )
    if (
        Path(getattr(config, "source_path")).resolve()
        != (path / "config.resolved.yaml").resolve()
        or Path(getattr(config, "artifact_root")).resolve() != path
    ):
        raise ProtocolError("PDCB fresh config/root binding drifted.")
    checks = validate_p_anchored_directional_crossing_bagging_bundle(
        path, config=config, allow_pending_validation=True
    )
    return {"process_id": os.getpid(), "checks": checks}


def _main(argv: Sequence[str]) -> int:
    if len(argv) != 2 or argv[0] != "--worker":
        raise ProtocolError("PDCB fresh module is worker-only.")
    sys.stdout.write(canonical_json(_worker_payload(Path(argv[1]))) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))


__all__ = (
    "ATTESTATION_KEY",
    "require_two_fresh_process_validations",
    "verify_attested_validation",
)
