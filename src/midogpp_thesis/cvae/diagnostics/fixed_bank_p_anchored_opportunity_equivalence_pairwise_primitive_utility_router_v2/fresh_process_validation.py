"""Artifact-only fresh-process attestation for OE-PPUR v2.

This module deliberately has one narrow job: two independent Python
interpreters, created with the ``spawn`` start method, reopen one already
persisted JSON receipt and prove that its canonical bytes, receipt hash, file
identity, and validator source identity have not drifted.  The workers receive
no workspace inputs, models, predictions, or terminal-label capability, so the
attestation cannot refit science or read labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import InitVar, dataclass, field
import fcntl
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
from pathlib import Path
import stat
from typing import Iterator

from ...protocol import ProtocolError
from .hashing import canonical_hash, canonical_json_bytes, require_sha256
from .run_paths import assert_no_symlink_chain, validate_absolute_path


WORKER_MODULE = (
    "midogpp_thesis.cvae.diagnostics."
    "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_"
    "utility_router_v2.fresh_process_validation"
)
FRESH_PROCESS_COUNT = 2
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
MAX_WORKER_MESSAGE_BYTES = 256 * 1024
ARTIFACT_ONLY_ENV = "OE_PPUR_V2_ARTIFACT_ONLY_VALIDATOR"
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
_SPAWN_ENVIRONMENT = {
    **THREAD_ENVIRONMENT,
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONHASHSEED": "0",
    ARTIFACT_ONLY_ENV: "1",
}
_ATTESTATION_FACTORY_TOKEN = object()
_VALID_PHASES = frozenset({"preterminal", "final"})


@dataclass(frozen=True, slots=True)
class ArtifactFreshProcessAttestationReceipt:
    """Guarded proof that two actual spawned validators agreed on one file."""

    phase: str
    sealed_receipt_hash: str
    sealed_file_sha256: str
    sealed_file_identity_sha256: str
    parent_process_id: int
    process_ids: tuple[int, int]
    validator_source_module: str
    validator_source_sha256: str
    validator_source_identity_sha256s: tuple[str, str]
    validator_result_hashes: tuple[str, str]
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _ATTESTATION_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR v2 fresh-process receipt bypassed spawned validation."
            )
        phase = str(self.phase)
        process_ids = tuple(self.process_ids)
        source_identities = tuple(
            require_sha256(value, "validator source-identity hash")
            for value in self.validator_source_identity_sha256s
        )
        result_hashes = tuple(
            require_sha256(value, "validator result hash")
            for value in self.validator_result_hashes
        )
        if (
            phase not in _VALID_PHASES
            or type(self.parent_process_id) is not int
            or self.parent_process_id <= 0
            or len(process_ids) != FRESH_PROCESS_COUNT
            or any(type(value) is not int or value <= 0 for value in process_ids)
            or len(set(process_ids)) != FRESH_PROCESS_COUNT
            or self.parent_process_id in process_ids
            or self.validator_source_module != WORKER_MODULE
            or len(source_identities) != FRESH_PROCESS_COUNT
            or len(set(source_identities)) != 1
            or len(result_hashes) != FRESH_PROCESS_COUNT
            or len(set(result_hashes)) != FRESH_PROCESS_COUNT
        ):
            raise ProtocolError(
                "OE-PPUR v2 fresh-process attestation topology drifted."
            )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "process_ids", process_ids)
        object.__setattr__(
            self, "validator_source_identity_sha256s", source_identities
        )
        object.__setattr__(self, "validator_result_hashes", result_hashes)
        for name in (
            "sealed_receipt_hash",
            "sealed_file_sha256",
            "sealed_file_identity_sha256",
            "validator_source_sha256",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name.replace("_", " ")),
            )
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v2_artifact_fresh_process_attestation_v1",
            "phase": self.phase,
            "sealed_receipt_hash": self.sealed_receipt_hash,
            "sealed_file_sha256": self.sealed_file_sha256,
            "sealed_file_identity_sha256": self.sealed_file_identity_sha256,
            "parent_process_id": self.parent_process_id,
            "process_ids": list(self.process_ids),
            "validator_source_module": self.validator_source_module,
            "validator_source_sha256": self.validator_source_sha256,
            "validator_source_identity_sha256s": list(
                self.validator_source_identity_sha256s
            ),
            "validator_result_hashes": list(self.validator_result_hashes),
            "fresh_python_process_count": FRESH_PROCESS_COUNT,
            "multiprocessing_start_method": "spawn",
            "process_launches_sequential": True,
            "cuda_visible_devices": "",
            "worker_thread_environment": dict(THREAD_ENVIRONMENT),
            "artifact_only_validation": True,
            "scientific_refit_performed": False,
            "labels_opened": False,
            "terminal_capability_opened": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class _StableFileObservation:
    content: bytes
    content_sha256: str
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class _SealedReceiptObservation:
    payload: dict[str, object]
    sealed_receipt_hash: str
    file_sha256: str
    file_identity_sha256: str


@dataclass(frozen=True, slots=True)
class _ChildObservation:
    process_id: int
    payload: dict[str, object]
    result_hash: str


def require_two_fresh_artifact_attestations(
    receipt_path: str | Path,
    *,
    phase: str,
    expected_sealed_receipt_hash: object,
    expected_file_sha256: object,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> ArtifactFreshProcessAttestationReceipt:
    """Reopen and attest one sealed JSON receipt in exactly two processes."""

    if phase not in _VALID_PHASES:
        raise ProtocolError("OE-PPUR v2 fresh-process phase drifted.")
    sealed_hash = require_sha256(
        expected_sealed_receipt_hash, "expected sealed receipt hash"
    )
    file_hash = require_sha256(expected_file_sha256, "expected receipt-file hash")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0.0 < float(timeout_seconds) <= 3_600.0
    ):
        raise ProtocolError("OE-PPUR v2 fresh-process timeout drifted.")
    path = validate_absolute_path(receipt_path, role="sealed receipt path")
    assert_no_symlink_chain(path)

    parent_before = _read_sealed_receipt(
        path,
        expected_sealed_receipt_hash=sealed_hash,
        expected_file_sha256=file_hash,
    )
    source_before = _validator_source_observation()
    parent_pid = os.getpid()
    request = {
        "schema_version": "oe_ppur_v2_artifact_validator_request_v1",
        "phase": phase,
        "receipt_path": path.as_posix(),
        "expected_sealed_receipt_hash": sealed_hash,
        "expected_file_sha256": file_hash,
        "expected_file_identity_sha256": parent_before.file_identity_sha256,
        "expected_validator_source_sha256": source_before.content_sha256,
        "expected_validator_source_identity_sha256": source_before.identity_sha256,
        "parent_process_id": parent_pid,
    }

    children = tuple(
        _launch_spawn_validator(
            request,
            ordinal=ordinal,
            timeout_seconds=float(timeout_seconds),
        )
        for ordinal in range(1, FRESH_PROCESS_COUNT + 1)
    )
    _validate_children(
        children,
        request=request,
        parent_process_id=parent_pid,
    )

    # Close the window across both subprocess launches.  Replacing even an
    # equal-byte file is rejected because its physical identity would change.
    parent_after = _read_sealed_receipt(
        path,
        expected_sealed_receipt_hash=sealed_hash,
        expected_file_sha256=file_hash,
    )
    source_after = _validator_source_observation()
    if parent_after != parent_before or source_after != source_before:
        raise ProtocolError(
            "OE-PPUR v2 attested artifact or validator source changed during validation."
        )

    return _issue_attestation_receipt(
        phase=phase,
        sealed_receipt_hash=sealed_hash,
        sealed_file_sha256=file_hash,
        sealed_file_identity_sha256=parent_before.file_identity_sha256,
        parent_process_id=parent_pid,
        process_ids=tuple(row.process_id for row in children),
        validator_source_module=WORKER_MODULE,
        validator_source_sha256=source_before.content_sha256,
        validator_source_identity_sha256s=tuple(
            str(row.payload["validator_source_identity_sha256"])
            for row in children
        ),
        validator_result_hashes=tuple(row.result_hash for row in children),
    )


def validate_artifact_fresh_process_attestation(
    receipt: object,
    *,
    expected_phase: str | None = None,
    expected_sealed_receipt_hash: object | None = None,
    expected_file_sha256: object | None = None,
) -> ArtifactFreshProcessAttestationReceipt:
    """Rebuild one guarded receipt and optionally enforce its parent lineage."""

    if not isinstance(receipt, ArtifactFreshProcessAttestationReceipt):
        raise ProtocolError("OE-PPUR v2 fresh-process attestation is untyped.")
    rebuilt = _issue_attestation_receipt(
        phase=receipt.phase,
        sealed_receipt_hash=receipt.sealed_receipt_hash,
        sealed_file_sha256=receipt.sealed_file_sha256,
        sealed_file_identity_sha256=receipt.sealed_file_identity_sha256,
        parent_process_id=receipt.parent_process_id,
        process_ids=receipt.process_ids,
        validator_source_module=receipt.validator_source_module,
        validator_source_sha256=receipt.validator_source_sha256,
        validator_source_identity_sha256s=(
            receipt.validator_source_identity_sha256s
        ),
        validator_result_hashes=receipt.validator_result_hashes,
    )
    if rebuilt != receipt:
        raise ProtocolError("OE-PPUR v2 fresh-process receipt hash drifted.")
    if expected_phase is not None and receipt.phase != expected_phase:
        raise ProtocolError("OE-PPUR v2 fresh-process attestation phase mismatched.")
    if expected_sealed_receipt_hash is not None and (
        receipt.sealed_receipt_hash
        != require_sha256(expected_sealed_receipt_hash, "sealed receipt hash")
    ):
        raise ProtocolError("OE-PPUR v2 fresh-process sealed lineage mismatched.")
    if expected_file_sha256 is not None and (
        receipt.sealed_file_sha256
        != require_sha256(expected_file_sha256, "receipt-file hash")
    ):
        raise ProtocolError("OE-PPUR v2 fresh-process file lineage mismatched.")
    return receipt


def _issue_attestation_receipt(**fields: object) -> ArtifactFreshProcessAttestationReceipt:
    return ArtifactFreshProcessAttestationReceipt(
        **fields,
        _factory_token=_ATTESTATION_FACTORY_TOKEN,
    )


def _launch_spawn_validator(
    request: Mapping[str, object],
    *,
    ordinal: int,
    timeout_seconds: float,
) -> _ChildObservation:
    context = mp.get_context("spawn")
    if context.get_start_method() != "spawn":
        raise ProtocolError("OE-PPUR v2 fresh validator did not select spawn.")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_spawn_validator_entrypoint,
        args=(sender, dict(request)),
        name=f"oe-ppur-v2-artifact-validator-{ordinal}",
        daemon=False,
    )
    actual_pid: int | None = None
    exit_code: int | None = None
    raw_message: bytes | None = None
    try:
        with _bounded_spawn_environment():
            process.start()
        actual_pid = process.pid
        sender.close()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(5.0)
            raise ProtocolError(
                f"OE-PPUR v2 fresh validator {ordinal} timed out."
            )
        exit_code = process.exitcode
        if receiver.poll():
            try:
                raw_message = receiver.recv_bytes(MAX_WORKER_MESSAGE_BYTES)
            except (EOFError, OSError) as exc:
                raise ProtocolError(
                    f"OE-PPUR v2 fresh validator {ordinal} response drifted."
                ) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise ProtocolError(
            f"OE-PPUR v2 fresh validator {ordinal} could not launch."
        ) from exc
    finally:
        try:
            sender.close()
        except OSError:
            pass
        receiver.close()
        try:
            process.close()
        except (OSError, ValueError):
            pass
    if exit_code != 0:
        raise ProtocolError(
            f"OE-PPUR v2 fresh validator {ordinal} exited nonzero ({exit_code})."
        )
    if type(actual_pid) is not int or actual_pid <= 0 or raw_message is None:
        raise ProtocolError(
            f"OE-PPUR v2 fresh validator {ordinal} emitted no usable result."
        )
    payload = _canonical_json_object(raw_message, role="validator result")
    reported_pid = payload.get("process_id")
    if type(reported_pid) is not int or reported_pid != actual_pid:
        raise ProtocolError(
            f"OE-PPUR v2 fresh validator {ordinal} reported a fake process ID."
        )
    return _ChildObservation(
        process_id=actual_pid,
        payload=payload,
        result_hash=canonical_hash(payload),
    )


def _spawn_validator_entrypoint(
    sender: Connection,
    request: dict[str, object],
) -> None:
    """Spawn-only child entrypoint; communicate one canonical primitive payload."""

    try:
        payload = _worker_payload(request)
        sender.send_bytes(canonical_json_bytes(payload))
    except BaseException as exc:
        try:
            sender.send_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": "oe_ppur_v2_artifact_validator_error_v1",
                        "process_id": os.getpid(),
                        "error_type": type(exc).__name__,
                    }
                )
            )
        except BaseException:
            pass
        raise SystemExit(91) from None
    finally:
        sender.close()


def _worker_payload(request: Mapping[str, object]) -> dict[str, object]:
    _validate_worker_environment()
    if mp.get_start_method(allow_none=False) != "spawn":
        raise ProtocolError("OE-PPUR v2 validator is not a spawned interpreter.")
    expected_keys = {
        "schema_version",
        "phase",
        "receipt_path",
        "expected_sealed_receipt_hash",
        "expected_file_sha256",
        "expected_file_identity_sha256",
        "expected_validator_source_sha256",
        "expected_validator_source_identity_sha256",
        "parent_process_id",
    }
    if (
        set(request) != expected_keys
        or request.get("schema_version")
        != "oe_ppur_v2_artifact_validator_request_v1"
        or request.get("phase") not in _VALID_PHASES
        or type(request.get("parent_process_id")) is not int
        or request.get("parent_process_id") != os.getppid()
    ):
        raise ProtocolError("OE-PPUR v2 validator request drifted.")
    source = _validator_source_observation()
    if (
        source.content_sha256
        != require_sha256(
            request["expected_validator_source_sha256"],
            "expected validator-source hash",
        )
        or source.identity_sha256
        != require_sha256(
            request["expected_validator_source_identity_sha256"],
            "expected validator-source identity hash",
        )
    ):
        raise ProtocolError("OE-PPUR v2 validator source identity drifted.")
    receipt = _read_sealed_receipt(
        validate_absolute_path(
            str(request["receipt_path"]), role="sealed receipt path"
        ),
        expected_sealed_receipt_hash=require_sha256(
            request["expected_sealed_receipt_hash"], "sealed receipt hash"
        ),
        expected_file_sha256=require_sha256(
            request["expected_file_sha256"], "receipt-file hash"
        ),
    )
    if receipt.file_identity_sha256 != require_sha256(
        request["expected_file_identity_sha256"], "receipt-file identity hash"
    ):
        raise ProtocolError("OE-PPUR v2 sealed receipt physical identity drifted.")
    return {
        "schema_version": "oe_ppur_v2_artifact_validator_result_v1",
        "phase": request["phase"],
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "multiprocessing_start_method": "spawn",
        "sealed_receipt_hash": receipt.sealed_receipt_hash,
        "sealed_file_sha256": receipt.file_sha256,
        "sealed_file_identity_sha256": receipt.file_identity_sha256,
        "validator_source_module": WORKER_MODULE,
        "validator_source_sha256": source.content_sha256,
        "validator_source_identity_sha256": source.identity_sha256,
        "environment": dict(_SPAWN_ENVIRONMENT),
        "descriptor_read_only": True,
        "no_follow_used": True,
        "stable_identity_revalidated": True,
        "canonical_json_validated": True,
        "artifact_only_validation": True,
        "scientific_refit_performed": False,
        "labels_opened": False,
        "terminal_capability_opened": False,
    }


def _validate_children(
    children: Sequence[_ChildObservation],
    *,
    request: Mapping[str, object],
    parent_process_id: int,
) -> None:
    if len(children) != FRESH_PROCESS_COUNT:
        raise ProtocolError("OE-PPUR v2 requires exactly two fresh validators.")
    pids = tuple(row.process_id for row in children)
    if len(set(pids)) != FRESH_PROCESS_COUNT or parent_process_id in pids:
        raise ProtocolError("OE-PPUR v2 validators were not fresh processes.")
    for row in children:
        payload = row.payload
        if (
            payload.get("schema_version")
            != "oe_ppur_v2_artifact_validator_result_v1"
            or payload.get("phase") != request["phase"]
            or payload.get("process_id") != row.process_id
            or payload.get("parent_process_id") != parent_process_id
            or payload.get("multiprocessing_start_method") != "spawn"
            or payload.get("sealed_receipt_hash")
            != request["expected_sealed_receipt_hash"]
            or payload.get("sealed_file_sha256")
            != request["expected_file_sha256"]
            or payload.get("sealed_file_identity_sha256")
            != request["expected_file_identity_sha256"]
            or payload.get("validator_source_module") != WORKER_MODULE
            or payload.get("validator_source_sha256")
            != request["expected_validator_source_sha256"]
            or payload.get("validator_source_identity_sha256")
            != request["expected_validator_source_identity_sha256"]
            or payload.get("environment") != _SPAWN_ENVIRONMENT
            or payload.get("descriptor_read_only") is not True
            or payload.get("no_follow_used") is not True
            or payload.get("stable_identity_revalidated") is not True
            or payload.get("canonical_json_validated") is not True
            or payload.get("artifact_only_validation") is not True
            or payload.get("scientific_refit_performed") is not False
            or payload.get("labels_opened") is not False
            or payload.get("terminal_capability_opened") is not False
            or row.result_hash != canonical_hash(payload)
        ):
            raise ProtocolError("OE-PPUR v2 fresh validator result drifted.")
    if len({row.result_hash for row in children}) != FRESH_PROCESS_COUNT:
        raise ProtocolError("OE-PPUR v2 fresh validators are not independently identified.")


def _read_sealed_receipt(
    path: Path,
    *,
    expected_sealed_receipt_hash: str,
    expected_file_sha256: str,
) -> _SealedReceiptObservation:
    observed = _read_stable_regular_file(path, role="sealed receipt")
    if observed.content_sha256 != expected_file_sha256:
        raise ProtocolError("OE-PPUR v2 sealed receipt file hash drifted.")
    payload = _canonical_json_object(
        observed.content,
        role="sealed receipt",
        require_trailing_newline=True,
    )
    receipt_hash = require_sha256(payload.get("receipt_hash"), "sealed receipt hash")
    body = dict(payload)
    del body["receipt_hash"]
    if (
        receipt_hash != expected_sealed_receipt_hash
        or canonical_hash(body) != receipt_hash
    ):
        raise ProtocolError("OE-PPUR v2 sealed receipt hash drifted.")
    _assert_no_refit_or_label_access_claim(payload)
    return _SealedReceiptObservation(
        payload=payload,
        sealed_receipt_hash=receipt_hash,
        file_sha256=observed.content_sha256,
        file_identity_sha256=observed.identity_sha256,
    )


def _validator_source_observation() -> _StableFileObservation:
    source_path = validate_absolute_path(Path(__file__), role="validator source path")
    return _read_stable_regular_file(source_path, role="validator source")


def _read_stable_regular_file(path: Path, *, role: str) -> _StableFileObservation:
    assert_no_symlink_chain(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise ProtocolError("OE-PPUR v2 platform lacks no-follow file opening.")
    try:
        before = path.lstat()
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v2 {role} is absent.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ProtocolError(f"OE-PPUR v2 {role} is not a regular file.")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stable_identity(opened) != _stable_identity(before)
            or fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
            != os.O_RDONLY
        ):
            raise ProtocolError(f"OE-PPUR v2 {role} changed before reading.")
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, MAX_RECEIPT_BYTES + 1))
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > MAX_RECEIPT_BYTES:
                raise ProtocolError(f"OE-PPUR v2 {role} exceeds its byte limit.")
            chunks.append(chunk)
        after_descriptor = os.fstat(descriptor)
        after_path = path.lstat()
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v2 {role} could not be read safely.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        _stable_identity(before) != _stable_identity(opened)
        or _stable_identity(opened) != _stable_identity(after_descriptor)
        or _stable_identity(after_descriptor) != _stable_identity(after_path)
        or stat.S_ISLNK(after_path.st_mode)
    ):
        raise ProtocolError(f"OE-PPUR v2 {role} changed while reading.")
    content = b"".join(chunks)
    return _StableFileObservation(
        content=content,
        content_sha256=hashlib.sha256(content).hexdigest(),
        identity_sha256=canonical_hash(
            {
                "path": path.as_posix(),
                "stable_identity": list(_stable_identity(after_descriptor)),
            }
        ),
    )


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _canonical_json_object(
    raw: bytes,
    *,
    role: str,
    require_trailing_newline: bool = False,
) -> dict[str, object]:
    def reject_duplicate_pairs(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError(f"OE-PPUR v2 {role} has duplicate JSON keys.")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ProtocolError(f"OE-PPUR v2 {role} contains {value}.")

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"OE-PPUR v2 {role} is invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"OE-PPUR v2 {role} is not a JSON object.")
    expected = canonical_json_bytes(payload) + (b"\n" if require_trailing_newline else b"")
    if raw != expected:
        raise ProtocolError(f"OE-PPUR v2 {role} is not canonical JSON.")
    return payload


def _assert_no_refit_or_label_access_claim(payload: Mapping[str, object]) -> None:
    label_actions = (
        "access",
        "load",
        "open",
        "persist",
        "read",
        "use",
        "materializ",
        "consum",
    )

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                folded = str(key).casefold()
                forbidden = "refit" in folded or (
                    "label" in folded
                    and any(action in folded for action in label_actions)
                )
                if forbidden and _claim_is_positive(nested):
                    raise ProtocolError(
                        "OE-PPUR v2 sealed receipt claims refit or label access."
                    )
                visit(nested)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for nested in value:
                visit(nested)

    visit(payload)


def _claim_is_positive(value: object) -> bool:
    return value not in (False, None, 0, "", (), [], {})


def _validate_worker_environment() -> None:
    if any(os.environ.get(key) != value for key, value in _SPAWN_ENVIRONMENT.items()):
        raise ProtocolError("OE-PPUR v2 fresh validator environment drifted.")


@contextmanager
def _bounded_spawn_environment() -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in _SPAWN_ENVIRONMENT}
    try:
        os.environ.update(_SPAWN_ENVIRONMENT)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


__all__ = (
    "ArtifactFreshProcessAttestationReceipt",
    "FRESH_PROCESS_COUNT",
    "THREAD_ENVIRONMENT",
    "WORKER_MODULE",
    "require_two_fresh_artifact_attestations",
    "validate_artifact_fresh_process_attestation",
)
