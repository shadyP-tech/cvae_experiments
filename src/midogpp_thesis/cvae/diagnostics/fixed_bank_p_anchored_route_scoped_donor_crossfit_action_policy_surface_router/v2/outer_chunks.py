"""No-recovery atomic integrity chunks for complete outer-H DTOs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ....protocol import ProtocolError
from ....runtime.artifact_io import atomic_json, read_json
from .identity import canonical_hash
from .outer_runtime import OuterRuntimeResult
from .scratch import OUTER_DIRECTORY, ScratchLease


def persist_and_verify_outer_chunks(
    lease: ScratchLease,
    rows: Sequence[OuterRuntimeResult],
) -> dict[str, object]:
    """Serialize each complete H result atomically, then verify exact bytes.

    Chunks are an in-attempt integrity barrier only. They are never admitted as
    recovery input and are deleted with the lease after success or failure.
    """

    values = tuple(rows)
    if (
        not isinstance(lease, ScratchLease)
        or not values
        or len({row.outer_center for row in values}) != len(values)
    ):
        raise ProtocolError("P-DCAPS v2 outer chunk inventory drifted.")
    directory = lease.root / OUTER_DIRECTORY
    if directory.exists() or directory.is_symlink():
        raise ProtocolError("P-DCAPS v2 refuses outer chunk recovery or repair.")
    directory.mkdir(parents=False, exist_ok=False)
    chunk_rows: list[dict[str, object]] = []
    for ordinal, row in enumerate(values):
        payload = row.to_payload()
        base = {
            "schema_version": "pdcaps_v2_complete_outer_h_chunk_v1",
            "ordinal": ordinal,
            "outer_center": row.outer_center,
            "outer_result": payload,
            "outer_result_hash": row.result_hash,
            "cross_run_recovery_allowed": False,
            "target_labels_used": False,
        }
        chunk = {**base, "chunk_hash": canonical_hash(base)}
        member = f"outer_{ordinal:02d}_center_{row.outer_center}.json"
        target = directory / member
        atomic_json(target, chunk)
        if read_json(target) != chunk:
            raise ProtocolError("P-DCAPS v2 outer chunk verification failed.")
        chunk_rows.append(
            {
                "ordinal": ordinal,
                "outer_center": row.outer_center,
                "member_id": member,
                "outer_result_hash": row.result_hash,
                "chunk_hash": chunk["chunk_hash"],
            }
        )
    manifest_base = {
        "schema_version": "pdcaps_v2_outer_h_chunk_manifest_v1",
        "chunks": chunk_rows,
        "chunk_count": len(chunk_rows),
        "written_atomically": True,
        "verified_after_write": True,
        "scratch_only": True,
        "cross_run_recovery_allowed": False,
        "target_labels_used": False,
    }
    return {**manifest_base, "manifest_hash": canonical_hash(manifest_base)}


__all__ = ("persist_and_verify_outer_chunks",)
