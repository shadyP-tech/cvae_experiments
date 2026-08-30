"""One-shot spawn runners for OE-PPUR v4 artifact-only attestation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
import inspect
import multiprocessing
import os
from pathlib import Path
from typing import Any

from ....protocol import ProtocolError
from ..hashing import canonical_hash
from ..terminal.contracts import (
    AggregateOnlyTerminalReceipt,
    ArtifactOnlyPreterminalAttestationReceipt,
    _ATTESTATION_TOKEN,
    _issue_artifact_only_preterminal_attestation,
)
from .attestation_contracts import (
    FinalAggregateAttestationReceipt,
    _FINAL_ATTESTATION_TOKEN,
    _issue_final_aggregate_attestation,
)
from .attestation_validation import (
    _validate_preterminal_files,
    _validate_terminal_aggregate_file,
)
from .preterminal_persistence import (
    PersistedPreterminalArtifact,
    _fsync_preterminal_tree,
    _fsync_regular_file_and_parent,
    _read_regular_bytes_nofollow,
    _sha256_regular_file,
)


_FORBIDDEN_REQUEST_KEYS = frozenset(
    {
        "label",
        "labels",
        "row_label",
        "row_labels",
        "case_label",
        "case_labels",
        "raw_label",
        "raw_labels",
        "outcome",
        "outcomes",
    }
)


def attest_preterminal_artifact_twice(
    artifact: PersistedPreterminalArtifact,
    *,
    timeout_seconds: float = 120.0,
) -> tuple[
    ArtifactOnlyPreterminalAttestationReceipt,
    ArtifactOnlyPreterminalAttestationReceipt,
]:
    """Use two distinct one-shot spawn executors over artifact bytes only."""

    if type(artifact) is not PersistedPreterminalArtifact:
        raise ProtocolError("OE-PPUR v4 preterminal artifact descriptor is untyped.")
    timeout = _validate_timeout(
        timeout_seconds,
        role="attestation",
    )
    _fsync_preterminal_tree(
        artifact.root,
        matrix_path=artifact.matrix_path,
        manifest_path=artifact.manifest_path,
    )
    request = {
        "manifest_path": artifact.manifest_path.as_posix(),
        "matrix_path": artifact.matrix_path.as_posix(),
        "expected_ledger_hash": artifact.decision_ledger_hash,
        "expected_result_hash": artifact.result_hash,
        "expected_artifact_file_sha256": artifact.artifact_file_sha256,
        "expected_artifact_file_identity_sha256": (
            artifact.artifact_file_identity_sha256
        ),
    }
    raw = _run_two_fresh_spawn_workers(
        _spawn_attestation_worker,
        request,
        timeout_seconds=timeout,
    )
    if (
        len({int(row["process_pid"]) for row in raw}) != 2
        or any(int(row["process_pid"]) == os.getpid() for row in raw)
    ):
        raise ProtocolError("OE-PPUR v4 validators were not two fresh processes.")
    receipts = tuple(
        _issue_artifact_only_preterminal_attestation(
            sealed_ledger_receipt_hash=str(row["sealed_ledger_receipt_hash"]),
            artifact_file_sha256=str(row["artifact_file_sha256"]),
            artifact_file_identity_sha256=str(
                row["artifact_file_identity_sha256"]
            ),
            validator_runtime_sha256=str(row["validator_runtime_sha256"]),
            process_pid=int(row["process_pid"]),
            _validator_token=_ATTESTATION_TOKEN,
        )
        for row in raw
    )
    return receipts  # type: ignore[return-value]


def attest_terminal_aggregate_twice(
    path: Path,
    receipt: AggregateOnlyTerminalReceipt,
    *,
    timeout_seconds: float = 120.0,
) -> FinalAggregateAttestationReceipt:
    """Revalidate the persisted aggregate-only terminal receipt twice."""

    if type(receipt) is not AggregateOnlyTerminalReceipt:
        raise ProtocolError("OE-PPUR v4 terminal aggregate receipt is untyped.")
    timeout = _validate_timeout(
        timeout_seconds,
        role="final attestation",
    )
    aggregate_path = Path(path)
    _fsync_regular_file_and_parent(aggregate_path)
    first = _validate_terminal_aggregate_file(
        aggregate_path,
        expected_receipt_hash=receipt.receipt_hash,
    )
    request = {
        "path": aggregate_path.as_posix(),
        "expected_receipt_hash": receipt.receipt_hash,
        "expected_file_sha256": first["terminal_file_sha256"],
        "expected_file_identity_sha256": first["terminal_file_identity_sha256"],
    }
    raw = _run_two_fresh_spawn_workers(
        _spawn_final_attestation_worker,
        request,
        timeout_seconds=timeout,
    )
    if (
        len({int(row["process_pid"]) for row in raw}) != 2
        or len({str(row["validator_runtime_sha256"]) for row in raw}) != 1
        or any(int(row["process_pid"]) == os.getpid() for row in raw)
    ):
        raise ProtocolError("OE-PPUR v4 final validators were not fresh processes.")
    return _issue_final_aggregate_attestation(
        terminal_receipt_hash=receipt.receipt_hash,
        terminal_file_sha256=str(first["terminal_file_sha256"]),
        terminal_file_identity_sha256=str(first["terminal_file_identity_sha256"]),
        validator_runtime_sha256=str(raw[0]["validator_runtime_sha256"]),
        validator_process_pids=tuple(int(row["process_pid"]) for row in raw),
        worker_attestation_hashes=tuple(
            str(row["worker_attestation_hash"]) for row in raw
        ),
        _validator_token=_FINAL_ATTESTATION_TOKEN,
    )


def _run_two_fresh_spawn_workers(
    worker: Callable[[Mapping[str, object]], dict[str, object]],
    request: Mapping[str, object],
    *,
    timeout_seconds: float,
) -> tuple[dict[str, object], dict[str, object]]:
    _assert_label_free_request(request)
    context = multiprocessing.get_context("spawn")
    rows: list[dict[str, object]] = []
    for _ in range(2):
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            result = executor.submit(worker, dict(request)).result(
                timeout=timeout_seconds
            )
        if type(result) is not dict:
            raise ProtocolError("OE-PPUR v4 fresh attestation worker is untyped.")
        rows.append(result)
    return rows[0], rows[1]


def _spawn_attestation_worker(request: Mapping[str, object]) -> dict[str, object]:
    _seal_label_free_worker_environment()
    manifest = Path(str(request["manifest_path"]))
    matrix = Path(str(request["matrix_path"]))
    result = _validate_preterminal_files(
        manifest,
        matrix,
        expected_ledger_hash=str(request["expected_ledger_hash"]),
        expected_result_hash=str(request["expected_result_hash"]),
    )
    if (
        result["artifact_file_sha256"]
        != request["expected_artifact_file_sha256"]
        or result["artifact_file_identity_sha256"]
        != request["expected_artifact_file_identity_sha256"]
    ):
        raise ProtocolError("OE-PPUR v4 attested artifact changed after persistence.")
    return {
        **result,
        "process_pid": os.getpid(),
        "validator_runtime_sha256": _validator_runtime_sha256(),
    }


def _spawn_final_attestation_worker(
    request: Mapping[str, object],
) -> dict[str, object]:
    _seal_label_free_worker_environment()
    result = _validate_terminal_aggregate_file(
        Path(str(request["path"])),
        expected_receipt_hash=str(request["expected_receipt_hash"]),
    )
    if (
        result["terminal_file_sha256"] != request["expected_file_sha256"]
        or result["terminal_file_identity_sha256"]
        != request["expected_file_identity_sha256"]
    ):
        raise ProtocolError("OE-PPUR v4 terminal aggregate changed after scoring.")
    runtime_hash = _validator_runtime_sha256()
    pid = os.getpid()
    return {
        **result,
        "process_pid": pid,
        "validator_runtime_sha256": runtime_hash,
        "worker_attestation_hash": canonical_hash(
            {
                "schema_version": (
                    "oe_ppur_v4_final_aggregate_worker_attestation_v1"
                ),
                **result,
                "process_pid": pid,
                "validator_runtime_sha256": runtime_hash,
                "aggregate_only": True,
            }
        ),
    }


def _seal_label_free_worker_environment() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _validator_runtime_sha256() -> str:
    """Bind workers to validation, spawn, and durable reader implementations."""

    worker_source = Path(inspect.getsourcefile(_validator_runtime_sha256) or "")
    validation_source = Path(
        inspect.getsourcefile(_validate_preterminal_files) or ""
    )
    persistence_source = Path(
        inspect.getsourcefile(_read_regular_bytes_nofollow) or ""
    )
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v4_split_validator_runtime_v2",
            "attestation_worker_source_sha256": _sha256_regular_file(worker_source),
            "attestation_validation_source_sha256": _sha256_regular_file(
                validation_source
            ),
            "preterminal_persistence_source_sha256": _sha256_regular_file(
                persistence_source
            ),
        }
    )


def _assert_label_free_request(request: Mapping[str, object]) -> None:
    if (
        type(request) is not dict
        or not request
        or any(str(key).lower() in _FORBIDDEN_REQUEST_KEYS for key in request)
        or any(
            not isinstance(value, (str, int, float, bool, type(None)))
            for value in request.values()
        )
    ):
        raise ProtocolError("OE-PPUR v4 attestation request crossed the label firewall.")


def _validate_timeout(value: Any, *, role: str) -> float:
    timeout = float(value)
    if not 1.0 <= timeout <= 600.0:
        raise ProtocolError(f"OE-PPUR v4 {role} timeout is unsafe.")
    return timeout


__all__ = (
    "_run_two_fresh_spawn_workers",
    "_spawn_attestation_worker",
    "_spawn_final_attestation_worker",
    "_validator_runtime_sha256",
    "attest_preterminal_artifact_twice",
    "attest_terminal_aggregate_twice",
)
