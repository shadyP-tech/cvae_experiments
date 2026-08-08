"""Package-local closed-world and content-index IO for Stage 60."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ...protocol import ProtocolError
from ...reporting import write_json
from ..residual_topup.hashing import canonical_sha256
from .bundle import CONTENT_INDEX_MEMBERS, REQUIRED_FILES
from .config import CLAIM_SCOPE


def write_content_index(root: Path) -> None:
    records: list[dict[str, object]] = []
    for relative in CONTENT_INDEX_MEMBERS:
        member = root / relative
        if not member.is_file():
            raise ProtocolError(f"Residual-topup content member is missing: {relative}.")
        records.append(
            {
                "relative_path": relative,
                "sha256": sha256_file(member),
                "size_bytes": member.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_b_u_g_s_content_index_v1",
        "records": records,
    }
    payload["content_hash"] = canonical_sha256(payload)
    write_json(root / "manifests/content_index.json", payload)


def assert_closed_world(root: Path) -> None:
    if not root.exists():
        raise ProtocolError("Residual-topup output root was not prepared by the workspace.")
    actual = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file()
    }
    unexpected = sorted(actual.difference(REQUIRED_FILES))
    if unexpected:
        raise ProtocolError(f"Residual-topup artifact contains unexpected files: {unexpected}.")


def write_state(root: Path, status: str) -> None:
    write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_residual_topup_b_u_g_s_run_state_v1",
            "status": status,
            "claim_scope": CLAIM_SCOPE,
        },
    )


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Residual-topup JSON must be an object: {path}.")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "assert_closed_world",
    "read_json",
    "sha256_file",
    "write_content_index",
    "write_state",
)
