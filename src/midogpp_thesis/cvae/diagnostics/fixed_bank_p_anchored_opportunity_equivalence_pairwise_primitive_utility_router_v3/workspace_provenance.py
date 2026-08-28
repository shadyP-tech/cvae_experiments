"""Exact, no-follow workspace input-manifest admission for OE-PPUR v3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import stat

from ...protocol import ProtocolError
from .execution.inputs import (
    ResolvedDirectInput,
    hash_resolved_input_locations,
    validate_exact_resolved_input_bindings,
)
from .hashing import canonical_hash, require_sha256
from .identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    CLAIM_SCOPE,
    DIRECT_INPUT_ARTIFACT_IDS,
    EXPERIMENT_ID,
    INPUT_RELATIVE_MEMBERS,
    SOURCE_SUPERVISION_ARTIFACT_ID,
)
from .run_paths import assert_no_symlink_chain


WORKSPACE_INPUT_MANIFEST_MEMBER = "provenance/input_artifacts.json"


def build_authorized_input_semantics(
    *,
    source_contract_hash: str,
    source_row_order_sha256: str,
    source_producer_seal_sha256: str,
    source_recomputation_receipt_sha256: str,
    authorization_amendment_sha256: str,
    protocol_hash: str,
    lifecycle_source_seal_sha256: str,
) -> dict[str, dict[str, str]]:
    """Return the receipt-derived lifecycle facts for direct inputs #3/#7.

    The workspace catalog intentionally records the pre-authorization state.
    A resolved launch envelope must therefore replace those two stale state
    projections with facts derived from the materialized source receipt and
    exact amendment bytes.  All values remain strings to match the catalog
    manifest schema.
    """

    source = require_sha256(source_contract_hash, "source contract hash")
    row_order = require_sha256(source_row_order_sha256, "source row order")
    producer = require_sha256(source_producer_seal_sha256, "source producer seal")
    recomputation = require_sha256(
        source_recomputation_receipt_sha256,
        "source recomputation receipt",
    )
    amendment = require_sha256(
        authorization_amendment_sha256,
        "authorization amendment",
    )
    protocol = require_sha256(protocol_hash, "protocol hash")
    lifecycle = require_sha256(
        lifecycle_source_seal_sha256,
        "lifecycle source seal",
    )
    return {
        SOURCE_SUPERVISION_ARTIFACT_ID: {
            "source_bundle_materialized": "true",
            "source_contract_hash": source,
            "source_row_order_sha256": row_order,
            "producer_source_seal_sha256": producer,
            "compiler_recomputation_receipt_sha256": recomputation,
            "target_rows_present": "false",
            "target_labels_used": "false",
            "execution_authorized": "false",
            "consumed_test_reuse_authorized": "false",
            "execution_authorized_by_this_artifact": "false",
            "consumed_test_reuse_authorized_by_this_artifact": "false",
        },
        AUTHORIZATION_AMENDMENT_ARTIFACT_ID: {
            "amendment_status": "AUTHORIZED_SINGLE_USE_NOT_CONSUMED",
            "amendment_file_present": "true",
            "expected_amendment_sha256_present": "true",
            "authorization_amendment_sha256": amendment,
            "source_contract_hash": source,
            "protocol_hash": protocol,
            "lifecycle_source_seal_sha256": lifecycle,
            "execution_authorized": "true",
            "consumed_test_reuse_authorized": "true",
            "single_use_execution_identity": "true",
            "implementation_authorizes_execution": "true",
            "authorization_exhausted": "false",
            "previous_stage90_outputs_used": "false",
            "previous_stage90_amendments_used": "false",
            "previous_stage90_run_state_or_scratch_used": "false",
            "cross_run_recovery_allowed": "false",
        },
    }


@dataclass(frozen=True, slots=True)
class WorkspaceInputProvenanceReceipt:
    manifest_file_sha256: str
    manifest_file_identity_sha256: str
    input_location_binding_hash: str
    input_artifact_count: int
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for role in (
            "manifest_file_sha256",
            "manifest_file_identity_sha256",
            "input_location_binding_hash",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        if self.input_artifact_count != 7:
            raise ProtocolError("OE-PPUR v3 workspace provenance count drifted.")
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_workspace_input_provenance_receipt_v1",
            "manifest_file_sha256": self.manifest_file_sha256,
            "manifest_file_identity_sha256": self.manifest_file_identity_sha256,
            "input_location_binding_hash": self.input_location_binding_hash,
            "input_artifact_count": 7,
            "exact_input_inventory": True,
            "manifest_read_nofollow": True,
            "target_selection_artifacts_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def validate_workspace_input_provenance(
    artifact_root: Path,
    bindings: Sequence[ResolvedDirectInput],
    *,
    expected_authorized_semantics: Mapping[str, Mapping[str, str]] | None = None,
) -> WorkspaceInputProvenanceReceipt:
    """Bind the workspace-rendered manifest to the exact seven resolved inputs."""

    rows = validate_exact_resolved_input_bindings(bindings)
    root = Path(artifact_root)
    manifest_path = root / WORKSPACE_INPUT_MANIFEST_MEMBER
    raw, metadata = _read_unique_regular_file(manifest_path)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("OE-PPUR v3 workspace provenance is unreadable.") from exc
    validate_workspace_input_provenance_payload(
        payload,
        rows,
        expected_authorized_semantics=expected_authorized_semantics,
    )
    return WorkspaceInputProvenanceReceipt(
        manifest_file_sha256=hashlib.sha256(raw).hexdigest(),
        manifest_file_identity_sha256=canonical_hash(
            {
                "schema_version": "oe_ppur_v3_workspace_manifest_file_identity_v1",
                "stat": _stat_payload(metadata),
            }
        ),
        input_location_binding_hash=hash_resolved_input_locations(rows),
        input_artifact_count=7,
    )


def validate_workspace_input_provenance_payload(
    payload: object,
    bindings: Sequence[ResolvedDirectInput],
    *,
    expected_authorized_semantics: Mapping[str, Mapping[str, str]] | None = None,
    allow_missing_amendment: bool = False,
) -> None:
    """Validate a rendered or prospective manifest without opening its file."""

    rows = validate_exact_resolved_input_bindings(bindings)
    if type(allow_missing_amendment) is not bool:
        raise ProtocolError("OE-PPUR v3 prospective provenance flag is untyped.")
    if not isinstance(payload, Mapping):
        raise ProtocolError("OE-PPUR v3 workspace provenance is malformed.")
    if (
        payload.get("schema_version") != "midogpp_input_artifacts_v2"
        or payload.get("dataset_id") != "midogpp"
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("stage") != "90_oracles_and_diagnostics"
        or payload.get("claim_scope") != CLAIM_SCOPE
        or payload.get("selection_used_target_eval_artifacts") is not False
    ):
        raise ProtocolError("OE-PPUR v3 workspace provenance header drifted.")
    raw_rows = payload.get("input_artifacts")
    if not isinstance(raw_rows, list) or not all(
        isinstance(row, Mapping) for row in raw_rows
    ):
        raise ProtocolError("OE-PPUR v3 workspace provenance rows are malformed.")
    artifact_ids = tuple(str(row.get("artifact_id", "")) for row in raw_rows)
    if (
        len(artifact_ids) != 7
        or len(set(artifact_ids)) != 7
        or artifact_ids != tuple(sorted(DIRECT_INPUT_ARTIFACT_IDS))
    ):
        raise ProtocolError("OE-PPUR v3 workspace provenance coverage drifted.")
    expected_semantics = _normalize_expected_authorized_semantics(
        expected_authorized_semantics
    )
    by_id = {row.artifact_id: row for row in rows}
    for raw_row in raw_rows:
        artifact_id = str(raw_row["artifact_id"])
        binding = by_id[artifact_id]
        expected_root = _artifact_root_for_binding(binding)
        resolved_path = raw_row.get("resolved_path")
        if not isinstance(resolved_path, str):
            raise ProtocolError("OE-PPUR v3 workspace provenance path drifted.")
        rendered_root = Path(resolved_path)
        semantic_identities = raw_row.get("semantic_identities")
        if (
            not rendered_root.is_absolute()
            or rendered_root != expected_root
            or raw_row.get("exists") is not True
            or raw_row.get("semantic_identities_are_file_hashes") is not False
            or not isinstance(semantic_identities, Mapping)
            or not isinstance(raw_row.get("file_integrity"), Mapping)
        ):
            raise ProtocolError(
                f"OE-PPUR v3 workspace provenance drifted: {artifact_id}."
            )
        required = expected_semantics.get(artifact_id)
        if required is not None and any(
            semantic_identities.get(key) != value
            for key, value in required.items()
        ):
            raise ProtocolError(
                f"OE-PPUR v3 authorized provenance drifted: {artifact_id}."
            )
        prospective_amendment = (
            allow_missing_amendment
            and artifact_id == AUTHORIZATION_AMENDMENT_ARTIFACT_ID
            and not rendered_root.exists()
            and not rendered_root.is_symlink()
        )
        assert_no_symlink_chain(
            rendered_root,
            allow_missing_leaf=prospective_amendment,
        )
        if prospective_amendment:
            if (
                not rendered_root.parent.is_dir()
                or rendered_root.parent.is_symlink()
            ):
                raise ProtocolError(
                    "OE-PPUR v3 prospective amendment root is unsafe."
                )
            continue
        try:
            canonical_root = rendered_root.resolve(strict=True)
        except OSError as exc:
            raise ProtocolError("OE-PPUR v3 workspace provenance input is absent.") from exc
        if canonical_root != rendered_root or not rendered_root.is_dir():
            raise ProtocolError("OE-PPUR v3 workspace provenance root is unsafe.")


def _normalize_expected_authorized_semantics(
    value: Mapping[str, Mapping[str, str]] | None,
) -> dict[str, dict[str, str]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or set(value) != {
        SOURCE_SUPERVISION_ARTIFACT_ID,
        AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    }:
        raise ProtocolError("OE-PPUR v3 authorized provenance inventory drifted.")
    normalized: dict[str, dict[str, str]] = {}
    for artifact_id, identities in value.items():
        if (
            not isinstance(identities, Mapping)
            or not identities
            or not all(
                isinstance(key, str)
                and key
                and isinstance(item, str)
                and item
                for key, item in identities.items()
            )
        ):
            raise ProtocolError("OE-PPUR v3 authorized provenance is malformed.")
        normalized[str(artifact_id)] = dict(identities)
    return normalized


def _artifact_root_for_binding(binding: ResolvedDirectInput) -> Path:
    relative = Path(
        INPUT_RELATIVE_MEMBERS[DIRECT_INPUT_ARTIFACT_IDS.index(binding.artifact_id)]
    )
    root = binding.path
    for _part in relative.parts:
        root = root.parent
    return root


def _read_unique_regular_file(path: Path) -> tuple[bytes, os.stat_result]:
    candidate = Path(os.path.abspath(path))
    if candidate != path:
        raise ProtocolError("OE-PPUR v3 workspace provenance path is unsafe.")
    assert_no_symlink_chain(candidate)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v3 workspace provenance is unsafe.") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ProtocolError(
                "OE-PPUR v3 workspace provenance is not a unique regular file."
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if _stat_payload(before) != _stat_payload(after) or len(raw) != before.st_size:
        raise ProtocolError("OE-PPUR v3 workspace provenance changed while read.")
    return raw, before


def _stat_payload(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


__all__ = (
    "WORKSPACE_INPUT_MANIFEST_MEMBER",
    "WorkspaceInputProvenanceReceipt",
    "build_authorized_input_semantics",
    "validate_workspace_input_provenance",
    "validate_workspace_input_provenance_payload",
)
