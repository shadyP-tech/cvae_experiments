"""Resolved amendment validation and race-safe terminal input reads."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import stat

from ....protocol import ProtocolError
from ..authorization_contract import validate_authorization_amendment_payload
from ..config import ResolvedV3ConfigBundle
from ..hashing import require_sha256
from ..identity import EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256
from ..lifecycle_source_seal import (
    LifecycleSourceSealReceipt,
    build_lifecycle_source_seal,
    validate_lifecycle_source_seal,
)


def validate_resolved_terminal_authority(
    bundle: ResolvedV3ConfigBundle,
    *,
    source_training_surface_receipt_hash: str,
) -> LifecycleSourceSealReceipt:
    """Validate amendment authority before the physical reader opens labels."""

    if type(bundle) is not ResolvedV3ConfigBundle:
        raise ProtocolError("OE-PPUR v3 terminal authority bundle is untyped.")
    source_receipt = require_sha256(
        source_training_surface_receipt_hash,
        "source training surface receipt hash",
    )
    lifecycle = build_lifecycle_source_seal()
    amendment_path = bundle.input_bindings[6].path
    amendment_raw = _read_regular_file_bytes(
        amendment_path, maximum_bytes=1024 * 1024, role="authorization amendment"
    )
    return _validate_terminal_authority_bytes(
        bundle,
        source_training_surface_receipt_hash=source_receipt,
        amendment_raw=amendment_raw,
        lifecycle=lifecycle,
    )


def validate_prospective_terminal_authority(
    bundle: ResolvedV3ConfigBundle,
    *,
    source_training_surface_receipt_hash: str,
    amendment_raw: bytes,
    lifecycle_source_seal: LifecycleSourceSealReceipt,
) -> LifecycleSourceSealReceipt:
    """Validate in-memory amendment authority before publishing input #7."""

    if type(bundle) is not ResolvedV3ConfigBundle:
        raise ProtocolError("OE-PPUR v3 prospective authority bundle is untyped.")
    if type(amendment_raw) is not bytes or len(amendment_raw) > 1024 * 1024:
        raise ProtocolError("OE-PPUR v3 prospective amendment bytes are untyped.")
    source_receipt = require_sha256(
        source_training_surface_receipt_hash,
        "source training surface receipt hash",
    )
    lifecycle = validate_lifecycle_source_seal(lifecycle_source_seal)
    return _validate_terminal_authority_bytes(
        bundle,
        source_training_surface_receipt_hash=source_receipt,
        amendment_raw=amendment_raw,
        lifecycle=lifecycle,
    )


def _validate_terminal_authority_bytes(
    bundle: ResolvedV3ConfigBundle,
    *,
    source_training_surface_receipt_hash: str,
    amendment_raw: bytes,
    lifecycle: LifecycleSourceSealReceipt,
) -> LifecycleSourceSealReceipt:
    parent_path = bundle.input_bindings[5].path
    parent_raw = _read_regular_file_bytes(
        parent_path, maximum_bytes=1024 * 1024, role="original parent ledger"
    )
    if hashlib.sha256(parent_raw).hexdigest() != EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256:
        raise ProtocolError("OE-PPUR v3 original parent ledger bytes drifted.")
    parent = _decode_unique_json(parent_raw, role="original parent ledger")
    if (
        parent.get("status") != "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        or parent.get("split") != "test"
    ):
        raise ProtocolError("OE-PPUR v3 original parent ledger identity drifted.")
    observed = hashlib.sha256(amendment_raw).hexdigest()
    if observed != bundle.config.authorization_amendment_sha256:
        raise ProtocolError("OE-PPUR v3 amendment bytes drifted from config.")
    _validate_amendment_payload(
        _decode_unique_json(amendment_raw, role="authorization amendment"),
        bundle=bundle,
        source_training_surface_receipt_hash=(
            source_training_surface_receipt_hash
        ),
        lifecycle_source_seal_sha256=(
            lifecycle.lifecycle_source_seal_sha256
        ),
    )
    return validate_lifecycle_source_seal(
        lifecycle,
        expected_sha256=lifecycle.lifecycle_source_seal_sha256,
    )


def _validate_amendment_payload(
    payload: Mapping[str, object],
    *,
    bundle: ResolvedV3ConfigBundle,
    source_training_surface_receipt_hash: str,
    lifecycle_source_seal_sha256: str,
) -> None:
    validate_authorization_amendment_payload(
        payload,
        source_contract_hash=source_training_surface_receipt_hash,
        protocol_hash=bundle.config.protocol_hash,
        lifecycle_source_seal_sha256=lifecycle_source_seal_sha256,
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
        raise ProtocolError(f"OE-PPUR v3 {role} is unreadable.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"OE-PPUR v3 {role} is not an object.")
    return payload


def _read_regular_file_bytes(
    path: Path,
    *,
    maximum_bytes: int,
    role: str,
) -> bytes:
    """Read one bounded regular file without following a swapped symlink."""

    candidate = Path(os.path.abspath(path))
    current = candidate
    while True:
        if current.is_symlink():
            raise ProtocolError(f"OE-PPUR v3 {role} path contains a symlink.")
        if current == current.parent:
            break
        current = current.parent
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v3 {role} is absent or unsafe.") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum_bytes
        ):
            raise ProtocolError(f"OE-PPUR v3 {role} is not a bounded regular file.")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ProtocolError(f"OE-PPUR v3 {role} is oversized.")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ProtocolError(f"OE-PPUR v3 {role} changed while read.")
    return b"".join(chunks)
