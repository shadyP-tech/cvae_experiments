"""Small deterministic report writer."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....protocol import ProtocolError
from ....runtime.artifact_io import atomic_json, read_json
from ..identity import canonical_hash
from .safety import reject_forbidden_persisted_values


def persist_report(
    path: Path,
    payload: Mapping[str, object],
    *,
    report_role: str,
) -> dict[str, object]:
    reject_forbidden_persisted_values(payload)
    base = {
        "schema_version": "pdcaps_report_v1",
        "report_role": str(report_role),
        **dict(payload),
    }
    report = {**base, "report_hash": canonical_hash(base)}
    target = Path(path)
    if target.exists():
        if read_json(target) != report:
            raise ProtocolError("P-DCAPS refuses to repair a different report.")
        return report
    atomic_json(target, report)
    return report


__all__ = ("persist_report",)
