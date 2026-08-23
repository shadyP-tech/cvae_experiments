"""Pure phase-seal computation and verification."""

from __future__ import annotations

from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .identity import canonical_hash


def compute_phase_seal(
    *,
    phase: str,
    row_hashes: Sequence[str],
    upstream_seal_hashes: Sequence[str],
    protocol_hash: str,
    target_labels_opened: bool,
) -> dict[str, object]:
    rows = tuple(str(value) for value in row_hashes)
    upstream = tuple(str(value) for value in upstream_seal_hashes)
    if not phase or len(rows) != len(set(rows)):
        raise ProtocolError("P-DCAPS phase seal inventory drifted.")
    base = {
        "schema_version": "pdcaps_phase_seal_v1",
        "phase": str(phase),
        "row_hashes": list(rows),
        "row_count": len(rows),
        "upstream_seal_hashes": list(upstream),
        "protocol_hash": str(protocol_hash),
        "target_labels_opened": bool(target_labels_opened),
    }
    return {**base, "seal_hash": canonical_hash(base)}


def verify_phase_seal(payload: Mapping[str, object]) -> str:
    base = {key: value for key, value in payload.items() if key != "seal_hash"}
    rows = base.get("row_hashes")
    if (
        not isinstance(rows, list)
        or base.get("row_count") != len(rows)
        or len(rows) != len(set(str(value) for value in rows))
        or payload.get("seal_hash") != canonical_hash(base)
    ):
        raise ProtocolError("P-DCAPS phase seal drifted.")
    return str(payload["seal_hash"])


__all__ = ("compute_phase_seal", "verify_phase_seal")
