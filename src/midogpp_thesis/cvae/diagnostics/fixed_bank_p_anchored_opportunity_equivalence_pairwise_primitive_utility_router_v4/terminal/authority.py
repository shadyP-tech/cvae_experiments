"""Resolved v4 authority validation and race-safe terminal input reads."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import stat

from ....protocol import ProtocolError
from ..config import ResolvedV4ConfigBundle
from ..hashing import require_sha256
from ..identity import (
    EXPERIMENT_ID,
    EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
    OUTPUT_ARTIFACT_ID,
)
from ..lifecycle_source_seal import (
    LifecycleSourceSealReceipt,
    build_lifecycle_source_seal,
    validate_lifecycle_source_seal,
)


def validate_resolved_terminal_authority(
    bundle: ResolvedV4ConfigBundle,
    *,
    source_training_surface_receipt_hash: str,
) -> LifecycleSourceSealReceipt:
    """Reopen the workspace amendment before the one-shot label read.

    The amendment remains preparation authority (`execution_authorized=false`).
    The separate launch-authority hash is bound by ``ResolvedV4ConfigBundle``
    and is checked independently here; config.launch_authorized is never used.
    """

    if type(bundle) is not ResolvedV4ConfigBundle:
        raise ProtocolError("OE-PPUR v4 terminal authority bundle is untyped.")
    source_receipt = require_sha256(
        source_training_surface_receipt_hash,
        "source training surface receipt hash",
    )
    launch_hash = require_sha256(
        bundle.execution_launch_authority_sha256,
        "execution launch authority",
    )
    if launch_hash == "0" * 64:
        raise ProtocolError("OE-PPUR v4 separate launch authority is absent.")
    lifecycle = build_lifecycle_source_seal()
    amendment_raw = _read_regular_file_bytes(
        bundle.input_bindings[6].path,
        maximum_bytes=1024 * 1024,
        role="authorization amendment",
    )
    return _validate_terminal_authority_bytes(
        bundle,
        source_training_surface_receipt_hash=source_receipt,
        amendment_raw=amendment_raw,
        lifecycle=lifecycle,
    )


def _validate_terminal_authority_bytes(
    bundle: ResolvedV4ConfigBundle,
    *,
    source_training_surface_receipt_hash: str,
    amendment_raw: bytes,
    lifecycle: LifecycleSourceSealReceipt,
) -> LifecycleSourceSealReceipt:
    parent_raw = _read_regular_file_bytes(
        bundle.input_bindings[5].path,
        maximum_bytes=1024 * 1024,
        role="original parent ledger",
    )
    if hashlib.sha256(parent_raw).hexdigest() != EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256:
        raise ProtocolError("OE-PPUR v4 original parent ledger bytes drifted.")
    parent = _decode_unique_json(parent_raw, role="original parent ledger")
    if (
        parent.get("status") != "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        or parent.get("split") != "test"
    ):
        raise ProtocolError("OE-PPUR v4 original parent ledger identity drifted.")
    observed = hashlib.sha256(amendment_raw).hexdigest()
    if observed != bundle.config.authorization_amendment_sha256:
        raise ProtocolError("OE-PPUR v4 amendment bytes drifted from config.")
    _validate_amendment_payload(
        _decode_unique_json(amendment_raw, role="authorization amendment"),
        bundle=bundle,
        source_training_surface_receipt_hash=source_training_surface_receipt_hash,
        lifecycle_source_seal_sha256=lifecycle.lifecycle_source_seal_sha256,
    )
    return validate_lifecycle_source_seal(
        lifecycle,
        expected_sha256=lifecycle.lifecycle_source_seal_sha256,
    )


def _validate_amendment_payload(
    payload: Mapping[str, object],
    *,
    bundle: ResolvedV4ConfigBundle,
    source_training_surface_receipt_hash: str,
    lifecycle_source_seal_sha256: str,
) -> None:
    direct_ids = payload.get("direct_input_artifact_ids")
    if (
        payload.get("schema_version")
        != "oe_ppur_v4_workspace_sealed_authorization_amendment_v1"
        or payload.get("consumer_experiment_id") != EXPERIMENT_ID
        or payload.get("consumer_output_artifact_id") != OUTPUT_ARTIFACT_ID
        or payload.get("authorized_run_count") != 1
        or payload.get("execution_authorized") is not False
        or payload.get("separate_launch_authority_required") is not True
        or payload.get("single_use_execution_identity") is not True
        or payload.get("cross_run_recovery_allowed") is not False
        or payload.get("target_labels_open_only_after_durable_preterminal_attestation")
        is not True
        or payload.get("pre_amendment_plan_sha256")
        != bundle.config.workspace_plan_sha256
        or payload.get("lifecycle_seal_sha256") != lifecycle_source_seal_sha256
        or payload.get("previous_stage90_operational_outputs_used") is not False
        or payload.get("previous_stage90_run_state_or_scratch_used") is not False
        or not isinstance(direct_ids, list)
        or len(direct_ids) != 7
    ):
        raise ProtocolError("OE-PPUR v4 workspace amendment authority drifted.")
    # The source receipt is already exact-hash bound in admission; include it
    # in the live lifecycle validation even though the preparation amendment
    # intentionally binds the broader scientific source seal instead.
    require_sha256(
        source_training_surface_receipt_hash,
        "source training surface receipt hash",
    )


def _decode_unique_json(raw: bytes, *, role: str) -> dict[str, object]:
    def unique(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError(f"OE-PPUR v4 {role} is unreadable.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"OE-PPUR v4 {role} is not an object.")
    return payload


def _read_regular_file_bytes(
    path: Path,
    *,
    maximum_bytes: int,
    role: str,
) -> bytes:
    candidate = Path(os.path.abspath(path))
    current = candidate
    while True:
        if current.is_symlink():
            raise ProtocolError(f"OE-PPUR v4 {role} path contains a symlink.")
        if current == current.parent:
            break
        current = current.parent
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v4 {role} is absent or unsafe.") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum_bytes
        ):
            raise ProtocolError(f"OE-PPUR v4 {role} is not a bounded regular file.")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ProtocolError(f"OE-PPUR v4 {role} is oversized.")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_id != after_id:
        raise ProtocolError(f"OE-PPUR v4 {role} changed while read.")
    return b"".join(chunks)


__all__ = ("validate_resolved_terminal_authority",)
