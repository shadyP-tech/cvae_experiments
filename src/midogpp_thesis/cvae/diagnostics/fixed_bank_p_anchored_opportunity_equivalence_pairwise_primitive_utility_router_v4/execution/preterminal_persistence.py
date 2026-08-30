"""Durable, exclusive preterminal file persistence for OE-PPUR v4."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from ....protocol import ProtocolError
from .. import durable_io as _durable_io
from ..hashing import require_sha256
from ..identity import CENTERS, EXPECTED_TERMINAL_CASE_INVENTORY_SHA256
from ..science.target_inventory import CANONICAL_TARGET_CASE_INVENTORY
from .services import CanonicalPreterminalResult, CanonicalRouterExecutionRequest


MATRIX_MEMBER = "arrays/preterminal_probability_matrix.npy"
MANIFEST_MEMBER = "manifests/preterminal_result.json"
PRETERMINAL_ATTESTATION_MEMBER = (
    "reports/preterminal_fresh_process_attestation.json"
)
FINAL_ATTESTATION_MEMBER = "reports/final_fresh_process_attestation.json"
ATTESTATION_MEMBERS = (
    PRETERMINAL_ATTESTATION_MEMBER,
    FINAL_ATTESTATION_MEMBER,
)
_MANIFEST_SCHEMA = "oe_ppur_v4_persisted_preterminal_result_v1"


@dataclass(frozen=True, slots=True)
class PersistedPreterminalArtifact:
    root: Path
    matrix_path: Path
    manifest_path: Path
    artifact_file_sha256: str
    artifact_file_identity_sha256: str
    decision_ledger_hash: str
    result_hash: str

    def __post_init__(self) -> None:
        root = Path(self.root)
        matrix = Path(self.matrix_path)
        manifest = Path(self.manifest_path)
        if (
            not root.is_absolute()
            or matrix != root / MATRIX_MEMBER
            or manifest != root / MANIFEST_MEMBER
            or root.is_symlink()
            or matrix.is_symlink()
            or manifest.is_symlink()
        ):
            raise ProtocolError("OE-PPUR v4 persisted preterminal paths drifted.")
        for role in (
            "artifact_file_sha256",
            "artifact_file_identity_sha256",
            "decision_ledger_hash",
            "result_hash",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )


def persist_preterminal_files(
    root: Path,
    result: CanonicalPreterminalResult,
    request: CanonicalRouterExecutionRequest,
) -> tuple[Path, Path, Path]:
    """Write and durably seal the canonical matrix and manifest only."""

    if (
        type(result) is not CanonicalPreterminalResult
        or type(request) is not CanonicalRouterExecutionRequest
        or result.request_hash != request.request_hash
    ):
        raise ProtocolError("OE-PPUR v4 preterminal persistence input drifted.")
    artifact_root = _existing_safe_root(root)
    matrix_path = _fresh_member(artifact_root, MATRIX_MEMBER)
    manifest_path = _fresh_member(artifact_root, MANIFEST_MEMBER)
    _write_npy_exclusive(matrix_path, result.probability_matrix.values)
    _write_json_exclusive(
        manifest_path,
        _manifest_payload(result, request=request),
    )
    _fsync_preterminal_tree(
        artifact_root,
        matrix_path=matrix_path,
        manifest_path=manifest_path,
    )
    return artifact_root, matrix_path, manifest_path


def _manifest_payload(
    result: CanonicalPreterminalResult,
    *,
    request: CanonicalRouterExecutionRequest,
) -> dict[str, object]:
    matrix = result.probability_matrix
    row_bindings = tuple(
        (row.sample_id, row.center, row.case_id) for row in request.frame.rows
    )
    decisions = []
    for row in result.decision_ledger.decisions:
        admission = row.admission_decision_receipt
        decisions.append(
            {
                "center_id": row.center_id,
                "case_id": row.case_id,
                "selected_action_id": row.selected_action_id,
                "reason": row.reason,
                "row_indices": list(row.row_indices),
                "row_manifest_hash": row.row_manifest_hash,
                "outer_result_hash": row.outer_result_hash,
                "predicted_action_scores": [
                    list(value) for value in row.predicted_action_scores
                ],
                "rank_available": row.rank_available,
                "admission_decision_receipt_hash": (
                    None if admission is None else admission.receipt_hash
                ),
                "selection_decision_hash": (
                    None
                    if admission is None
                    else admission.selection_decision.decision_hash
                ),
                "decision_hash": row.decision_hash,
            }
        )
    return {
        "schema_version": _MANIFEST_SCHEMA,
        "result_hash": result.result_hash,
        "request_hash": result.request_hash,
        "service_factory_identity_hash": result.service_factory_identity_hash,
        "seven_input_contract_hash": result.seven_input_contract_hash,
        "source_seal_hash": result.source_seal_hash,
        "source_training_surface_receipt_hash": (
            result.source_training_surface_receipt_hash
        ),
        "final_pool_receipt_hashes": [
            row.receipt_hash for row in result.final_pool_receipts
        ],
        "outer_science_result_hashes": [
            row.result_hash for row in result.outer_science_results
        ],
        "final_surface_hashes": [
            row.surface_hash for row in result.final_surfaces
        ],
        "probability_matrix_hash": matrix.matrix_hash,
        "matrix_shape": list(matrix.values.shape),
        "matrix_dtype": matrix.values.dtype.str,
        "matrix_f4_sha256": _array_sha256(matrix.values),
        "matrix_row_ids": list(matrix.row_ids),
        "matrix_center_offsets": {
            center: list(matrix.center_offsets[center]) for center in CENTERS
        },
        "matrix_action_ids": list(matrix.action_ids),
        "matrix_surface_hashes": [
            list(value) for value in matrix.surface_hashes
        ],
        "row_bindings": [list(value) for value in row_bindings],
        "case_inventory": [
            list(value) for value in CANONICAL_TARGET_CASE_INVENTORY
        ],
        "case_inventory_sha256": EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
        "decisions": decisions,
        "decision_ledger_hash": result.decision_ledger.ledger_hash,
        "exact_p_count": result.decision_ledger.exact_p_count,
        "rank_unavailable_count": result.decision_ledger.rank_unavailable_count,
        "target_labels_opened": False,
    }


def persist_attestation_json_exclusive(
    root: Path,
    relative_member: str,
    payload: Mapping[str, object],
) -> Path:
    """Durably persist one of the two canonical attestation receipts."""

    artifact_root = _existing_safe_root(root)
    if relative_member not in ATTESTATION_MEMBERS:
        raise ProtocolError("OE-PPUR v4 attestation member identity drifted.")
    member = _fresh_member(artifact_root, relative_member)
    _write_json_exclusive(member, payload)
    _fsync_directory(artifact_root)
    return member


def _existing_safe_root(value: Path) -> Path:
    candidate = Path(os.path.abspath(Path(value)))
    _reject_symlink_chain(candidate)
    try:
        root = Path(value).resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v4 artifact root is absent.") from exc
    if (
        root != candidate
        or root.is_symlink()
        or not root.is_dir()
        or root == Path(root.anchor)
    ):
        raise ProtocolError("OE-PPUR v4 artifact root is unsafe.")
    return root


def _fresh_member(root: Path, relative: str) -> Path:
    member = root / relative
    if member.exists() or member.is_symlink():
        raise ProtocolError("OE-PPUR v4 preterminal member is not fresh.")
    parent = member.parent
    if parent.exists():
        if parent.is_symlink() or not parent.is_dir():
            raise ProtocolError("OE-PPUR v4 preterminal directory is unsafe.")
    else:
        parent.mkdir(parents=True, exist_ok=False)
    _reject_symlink_chain(parent, stop=root)
    return member


def _write_npy_exclusive(path: Path, values: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(
        buffer,
        np.ascontiguousarray(values, dtype="<f4"),
        allow_pickle=False,
    )
    _durable_io.write_bytes_exclusive(
        Path(path),
        buffer.getvalue(),
        role="preterminal probability matrix",
    )


def _write_json_exclusive(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    _durable_io.write_canonical_json_exclusive(
        Path(path),
        payload,
        role="preterminal JSON member",
    )


def _fsync_preterminal_tree(
    root: Path,
    *,
    matrix_path: Path,
    manifest_path: Path,
) -> None:
    artifact_root = _existing_safe_root(root)
    if (
        matrix_path != artifact_root / MATRIX_MEMBER
        or manifest_path != artifact_root / MANIFEST_MEMBER
    ):
        raise ProtocolError("OE-PPUR v4 preterminal durability paths drifted.")
    _fsync_regular_file_and_parent(matrix_path)
    _fsync_regular_file_and_parent(manifest_path)
    _fsync_directory(artifact_root)


def _fsync_regular_file_and_parent(path: Path) -> None:
    _durable_io.fsync_regular_file_and_parent(
        Path(path),
        role="durable member",
    )


def _fsync_directory(path: Path) -> None:
    _durable_io.fsync_directory(Path(path))


def _read_regular_bytes_nofollow(
    path: Path,
) -> tuple[bytes, os.stat_result]:
    return _durable_io.read_regular_bytes_nofollow(
        Path(path),
        role="preterminal member",
    )


def _stat_payload(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return _durable_io.stat_payload(value)


def _reject_symlink_chain(path: Path, *, stop: Path | None = None) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ProtocolError("OE-PPUR v4 preterminal path contains a symlink.")
        if stop is not None and current == stop:
            return
        if current == current.parent:
            return
        current = current.parent


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype("<f4"))
    header = f"{array.dtype.str}|{array.shape}".encode("ascii")
    return hashlib.sha256(header + memoryview(array).cast("B")).hexdigest()


def _sha256_regular_file(path: Path) -> str:
    payload, _identity = _read_regular_bytes_nofollow(path)
    return hashlib.sha256(payload).hexdigest()


__all__ = (
    "ATTESTATION_MEMBERS",
    "FINAL_ATTESTATION_MEMBER",
    "MANIFEST_MEMBER",
    "MATRIX_MEMBER",
    "PRETERMINAL_ATTESTATION_MEMBER",
    "PersistedPreterminalArtifact",
    "persist_attestation_json_exclusive",
    "persist_preterminal_files",
)
