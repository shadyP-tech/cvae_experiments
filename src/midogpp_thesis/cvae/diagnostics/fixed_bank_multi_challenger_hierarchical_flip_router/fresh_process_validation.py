"""Two-process replay gate for accepting a terminal validation report.

The parent process first reconstructs the pending bundle checks itself.  This
module then launches two sequential Python interpreters.  Each interpreter
loads only the persisted resolved config and invokes the complete bundle
validator in pending-report mode.  The parent accepts the result only when
both canonical JSON payloads exactly match each other and its own replay.

The persisted attestation deliberately excludes process IDs and timestamps.
It is therefore deterministic and can be verified reconstructively by normal
completed-bundle validation without launching another process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .hashing import canonical_hash


ATTESTATION_KEY = "fresh_process_validation_attestation"
ATTESTATION_SCHEMA = "fixed_bank_multi_challenger_fresh_process_validation_v1"
WORKER_MODULE = (
    "midogpp_thesis.cvae.diagnostics."
    "fixed_bank_multi_challenger_hierarchical_flip_router."
    "fresh_process_validation"
)
VALIDATOR_ENTRYPOINT = (
    "validate_fixed_bank_multi_challenger_hierarchical_flip_router_bundle"
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
        "full_bundle_validation_called_by_each_process",
        "pending_validation_allowed",
        "cuda_visible_devices",
        "worker_thread_environment",
        "subprocess_exit_codes",
        "valid_json_payload_count",
        "reconstructed_check_payloads_exactly_equal",
        "reconstructed_check_hash",
        "replay_check_hashes",
        "validator_entrypoint",
        "attestation_hash",
    }
)


def require_two_fresh_process_validations(
    root: str | Path,
    *,
    expected_checks: Mapping[str, object],
) -> Mapping[str, object]:
    """Replay ``root`` twice and return checks carrying a sealed attestation."""

    path = Path(root).resolve()
    expected = _base_checks(expected_checks)
    expected_json = _canonical_json(expected, role="parent validation checks")
    replay_json: list[str] = []
    for ordinal in range(2):
        completed = _run_worker(path)
        if completed.returncode != 0:
            detail = completed.stderr.strip()
            if len(detail) > 2_000:
                detail = detail[-2_000:]
            raise ProtocolError(
                "Multi-challenger fresh-process validation failed "
                f"for replay {ordinal + 1} with exit code "
                f"{completed.returncode}: {detail or 'no stderr'}"
            )
        replay_json.append(
            _validated_worker_json(completed.stdout, ordinal=ordinal + 1)
        )
    if replay_json[0] != replay_json[1]:
        raise ProtocolError(
            "Multi-challenger fresh-process validation replays disagreed."
        )
    if replay_json[0] != expected_json:
        raise ProtocolError(
            "Multi-challenger fresh-process validation disagreed with the "
            "parent reconstruction."
        )
    check_hash = canonical_hash(expected)
    unhashed = _attestation_payload(check_hash)
    attestation = {**unhashed, "attestation_hash": canonical_hash(unhashed)}
    return {**expected, ATTESTATION_KEY: attestation}


def verify_attested_validation_checks(
    checks: Mapping[str, object],
    *,
    expected_reconstructed_checks: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Verify the deterministic replay attestation without spawning."""

    if not isinstance(checks, Mapping):
        raise ProtocolError("Multi-challenger validation checks are not a mapping.")
    payload = dict(checks)
    attestation = payload.pop(ATTESTATION_KEY, None)
    if not isinstance(attestation, Mapping) or set(attestation) != _ATTESTATION_KEYS:
        raise ProtocolError(
            "Multi-challenger fresh-process validation attestation is absent "
            "or malformed."
        )
    base = _base_checks(payload)
    if expected_reconstructed_checks is not None:
        expected = _base_checks(expected_reconstructed_checks)
        if _canonical_json(base, role="attested validation checks") != _canonical_json(
            expected, role="reconstructed validation checks"
        ):
            raise ProtocolError(
                "Multi-challenger attested checks do not match reconstruction."
            )
    check_hash = canonical_hash(base)
    unhashed = {
        key: value for key, value in attestation.items() if key != "attestation_hash"
    }
    expected_unhashed = _attestation_payload(check_hash)
    if (
        _canonical_json(unhashed, role="fresh-process attestation")
        != _canonical_json(expected_unhashed, role="expected fresh-process attestation")
        or attestation.get("attestation_hash") != canonical_hash(unhashed)
    ):
        raise ProtocolError(
            "Multi-challenger fresh-process validation attestation drifted."
        )
    return {**base, ATTESTATION_KEY: dict(attestation)}


def _attestation_payload(check_hash: str) -> dict[str, object]:
    return {
        "schema_version": ATTESTATION_SCHEMA,
        "status": "PASS",
        "fresh_python_process_count": 2,
        "independent_fresh_python_processes": True,
        "process_launches_sequential": True,
        "persisted_resolved_config_loaded_by_each_process": True,
        "full_bundle_validation_called_by_each_process": True,
        "pending_validation_allowed": True,
        "cuda_visible_devices": "",
        "worker_thread_environment": dict(_THREAD_ENVIRONMENT),
        "subprocess_exit_codes": [0, 0],
        "valid_json_payload_count": 2,
        "reconstructed_check_payloads_exactly_equal": True,
        "reconstructed_check_hash": check_hash,
        "replay_check_hashes": [check_hash, check_hash],
        "validator_entrypoint": VALIDATOR_ENTRYPOINT,
    }


def _run_worker(root: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.update(_THREAD_ENVIRONMENT)
    environment["PYTHONHASHSEED"] = "0"
    command = (sys.executable, "-m", WORKER_MODULE, "--worker", str(root))
    try:
        return subprocess.run(
            command,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=FRESH_PROCESS_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProtocolError(
            "Multi-challenger fresh-process validation could not complete."
        ) from exc


def _validated_worker_json(stdout: str, *, ordinal: int) -> str:
    try:
        payload = json.loads(stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            "Multi-challenger fresh-process validation emitted invalid JSON "
            f"for replay {ordinal}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError(
            "Multi-challenger fresh-process validation emitted a non-object "
            f"payload for replay {ordinal}."
        )
    canonical = _canonical_json(payload, role=f"fresh replay {ordinal}")
    if stdout.strip() != canonical:
        raise ProtocolError(
            "Multi-challenger fresh-process validation JSON is not canonical "
            f"for replay {ordinal}."
        )
    return canonical


def _base_checks(checks: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(checks, Mapping) or ATTESTATION_KEY in checks:
        raise ProtocolError(
            "Multi-challenger pending validation checks have invalid attestation state."
        )
    payload = dict(checks)
    if payload.get("status") != "PASS":
        raise ProtocolError(
            "Multi-challenger cannot attest validation checks without PASS status."
        )
    _canonical_json(payload, role="validation checks")
    return payload


def _canonical_json(payload: object, *, role: str) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Multi-challenger {role} is not strict JSON.") from exc


def _worker_payload(root: Path) -> Mapping[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(
        os.environ.get(key) != value for key, value in _THREAD_ENVIRONMENT.items()
    ):
        raise ProtocolError(
            "Multi-challenger fresh validation worker environment is unbounded."
        )
    path = root.resolve()
    config_path = path / "config.resolved.yaml"
    if config_path.is_symlink() or not config_path.is_file():
        raise ProtocolError(
            "Multi-challenger fresh validation worker resolved config is absent "
            "or unsafe."
        )
    from .config import (
        load_fixed_bank_multi_challenger_hierarchical_flip_router_config,
    )

    config = load_fixed_bank_multi_challenger_hierarchical_flip_router_config(
        config_path
    )
    if (
        Path(getattr(config, "source_path")).resolve() != config_path.resolve()
        or Path(getattr(config, "artifact_root")).resolve() != path
    ):
        raise ProtocolError(
            "Multi-challenger fresh validation worker is not bound to the "
            "persisted resolved config."
        )
    from .validation import (
        validate_fixed_bank_multi_challenger_hierarchical_flip_router_bundle,
    )

    return validate_fixed_bank_multi_challenger_hierarchical_flip_router_bundle(
        path,
        config=config,
        allow_pending_validation=True,
    )


def _main(argv: Sequence[str]) -> int:
    if len(argv) != 2 or argv[0] != "--worker":
        raise ProtocolError(
            "Multi-challenger fresh-process module is worker-only."
        )
    payload = _worker_payload(Path(argv[1]))
    sys.stdout.write(_canonical_json(payload, role="worker validation checks") + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(_main(sys.argv[1:]))


__all__ = (
    "ATTESTATION_KEY",
    "ATTESTATION_SCHEMA",
    "require_two_fresh_process_validations",
    "verify_attested_validation_checks",
)
