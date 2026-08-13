"""Two independent CUDA-free interpreter replays for the DCSE bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .hashing import canonical_hash, canonical_json


ATTESTATION_KEY = "fresh_process_validation_attestation"
ATTESTATION_SCHEMA = "fixed_bank_dcse_fresh_process_validation_v1"
WORKER_MODULE = (
    "midogpp_thesis.cvae.diagnostics."
    "fixed_bank_loo_directional_shrinkage_ensemble.fresh_process_validation"
)
VALIDATOR_ENTRYPOINT = (
    "validate_fixed_bank_loo_directional_shrinkage_ensemble_bundle"
)
FRESH_PROCESS_TIMEOUT_SECONDS = 21_600
_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
_ATTESTATION_KEYS = frozenset(
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
_CHILD_RESULT_KEYS = frozenset(
    {
        "ordinal",
        "process_id",
        "exit_code",
        "reconstructed_check_hash",
        "result_payload_hash",
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
                "Directional-shrinkage fresh replay failed "
                f"for child {ordinal}: {detail or 'no stderr'}."
            )
        payload = _worker_json(completed.stdout, ordinal=ordinal)
        checks = payload.get("checks")
        pid = payload.get("process_id")
        if (
            not isinstance(checks, Mapping)
            or not isinstance(pid, int)
            or isinstance(pid, bool)
            or canonical_hash(checks) != expected_hash
            or dict(checks) != expected
        ):
            raise ProtocolError(
                "Directional-shrinkage fresh replay disagreed with the parent."
            )
        results.append(
            {
                "ordinal": ordinal,
                "process_id": pid,
                "exit_code": completed.returncode,
                "reconstructed_check_hash": canonical_hash(checks),
                "result_payload_hash": canonical_hash(payload),
            }
        )
    child_pids = [int(result["process_id"]) for result in results]
    if len(set(child_pids)) != 2 or os.getpid() in child_pids:
        raise ProtocolError(
            "Directional-shrinkage fresh replay process identities are not independent."
        )
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
        "worker_thread_environment": dict(_THREAD_ENVIRONMENT),
        "parent_process_id": os.getpid(),
        "child_process_ids": child_pids,
        "child_process_results": results,
        "subprocess_exit_codes": [0, 0],
        "reconstructed_check_payloads_exactly_equal": True,
        "reconstructed_check_hash": expected_hash,
        "validator_entrypoint": VALIDATOR_ENTRYPOINT,
    }
    attestation = {**unhashed, "attestation_hash": canonical_hash(unhashed)}
    return {**expected, ATTESTATION_KEY: attestation}


def verify_attested_validation_checks(
    checks: Mapping[str, object],
    *,
    expected_reconstructed_checks: Mapping[str, object],
) -> Mapping[str, object]:
    payload = dict(checks)
    attestation = payload.pop(ATTESTATION_KEY, None)
    expected = _base_checks(expected_reconstructed_checks)
    if payload != expected or not isinstance(attestation, Mapping):
        raise ProtocolError(
            "Directional-shrinkage validation report is not reconstructive."
        )
    if set(attestation) != _ATTESTATION_KEYS:
        raise ProtocolError(
            "Directional-shrinkage fresh-process attestation schema drifted."
        )
    unhashed = {
        key: value for key, value in attestation.items() if key != "attestation_hash"
    }
    child_pids = attestation.get("child_process_ids")
    results = attestation.get("child_process_results")
    expected_hash = canonical_hash(expected)
    if (
        attestation.get("schema_version") != ATTESTATION_SCHEMA
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
        or attestation.get("worker_thread_environment") != _THREAD_ENVIRONMENT
        or not isinstance(attestation.get("parent_process_id"), int)
        or isinstance(attestation.get("parent_process_id"), bool)
        or not isinstance(child_pids, list)
        or len(child_pids) != 2
        or any(not isinstance(pid, int) or isinstance(pid, bool) for pid in child_pids)
        or len(set(child_pids)) != 2
        or attestation.get("parent_process_id") in child_pids
        or not isinstance(results, list)
        or len(results) != 2
        or [row.get("process_id") for row in results if isinstance(row, Mapping)]
        != child_pids
        or any(
            not isinstance(row, Mapping)
            or set(row) != _CHILD_RESULT_KEYS
            or row.get("ordinal") != ordinal
            or row.get("exit_code") != 0
            or row.get("reconstructed_check_hash") != expected_hash
            or row.get("result_payload_hash")
            != canonical_hash(
                {
                    "process_id": row.get("process_id"),
                    "checks": expected,
                }
            )
            for ordinal, row in enumerate(results, start=1)
        )
        or attestation.get("subprocess_exit_codes") != [0, 0]
        or attestation.get("reconstructed_check_payloads_exactly_equal") is not True
        or attestation.get("reconstructed_check_hash") != expected_hash
        or attestation.get("validator_entrypoint") != VALIDATOR_ENTRYPOINT
        or attestation.get("attestation_hash") != canonical_hash(unhashed)
    ):
        raise ProtocolError(
            "Directional-shrinkage fresh-process attestation drifted."
        )
    return {**expected, ATTESTATION_KEY: dict(attestation)}


def _run_worker(root: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.update(_THREAD_ENVIRONMENT)
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
        raise ProtocolError(
            "Directional-shrinkage fresh-process validation could not complete."
        ) from exc


def _worker_json(stdout: str, *, ordinal: int) -> Mapping[str, object]:
    try:
        payload = json.loads(stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            f"Directional-shrinkage fresh replay {ordinal} emitted invalid JSON."
        ) from exc
    if not isinstance(payload, Mapping) or stdout.strip().encode("utf-8") != canonical_json(payload):
        raise ProtocolError(
            f"Directional-shrinkage fresh replay {ordinal} JSON is not canonical."
        )
    return payload


def _base_checks(checks: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(checks, Mapping) or ATTESTATION_KEY in checks:
        raise ProtocolError("Directional-shrinkage pending checks are malformed.")
    payload = dict(checks)
    if payload.get("status") != "PASS":
        raise ProtocolError("Directional-shrinkage cannot attest non-PASS checks.")
    canonical_json(payload)
    return payload


def _worker_payload(root: Path) -> Mapping[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(
        os.environ.get(key) != value for key, value in _THREAD_ENVIRONMENT.items()
    ):
        raise ProtocolError(
            "Directional-shrinkage fresh worker environment is not CUDA-free/bounded."
        )
    path = root.resolve()
    config_path = path / "config.resolved.yaml"
    if config_path.is_symlink() or not config_path.is_file():
        raise ProtocolError(
            "Directional-shrinkage fresh worker config is absent or unsafe."
        )
    from .config import load_fixed_bank_loo_directional_shrinkage_ensemble_config

    config = load_fixed_bank_loo_directional_shrinkage_ensemble_config(config_path)
    if (
        Path(getattr(config, "source_path")).resolve() != config_path
        or Path(getattr(config, "artifact_root")).resolve() != path
    ):
        raise ProtocolError(
            "Directional-shrinkage fresh worker config/root binding drifted."
        )
    from .validation import (
        validate_fixed_bank_loo_directional_shrinkage_ensemble_bundle,
    )

    checks = validate_fixed_bank_loo_directional_shrinkage_ensemble_bundle(
        path,
        config=config,
        allow_pending_validation=True,
    )
    return {"process_id": os.getpid(), "checks": checks}


def _main(argv: Sequence[str]) -> int:
    if len(argv) != 2 or argv[0] != "--worker":
        raise ProtocolError("Directional-shrinkage fresh module is worker-only.")
    payload = _worker_payload(Path(argv[1]))
    sys.stdout.buffer.write(canonical_json(payload) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entrypoint
    raise SystemExit(_main(sys.argv[1:]))


__all__ = (
    "ATTESTATION_KEY",
    "ATTESTATION_SCHEMA",
    "require_two_fresh_process_validations",
    "verify_attested_validation_checks",
)
