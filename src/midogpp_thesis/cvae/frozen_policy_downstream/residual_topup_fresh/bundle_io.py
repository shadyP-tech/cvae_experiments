"""Atomic serialization and closed-world indexing for fresh Stage 70."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .bundle_schema import CONTENT_INDEX_EXCLUSIONS
from .config import INPUT_ARTIFACT_IDS, ResidualTopupFreshConfig


def write_content_index(root: str | Path) -> dict[str, object]:
    output = Path(root)
    files: list[dict[str, object]] = []
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ProtocolError("Fresh Stage-70 bundles cannot contain symlinks.")
        if not path.is_file():
            continue
        relative = str(path.relative_to(output))
        if relative in CONTENT_INDEX_EXCLUSIONS:
            continue
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload = {
        "schema_version": "midogpp_residual_topup_fresh_content_index_v1",
        "status": "COMPLETE",
        "files": files,
        "file_count": len(files),
        "scratch_authoritative": False,
    }
    payload["content_hash"] = stable_hash(payload)
    atomic_json(output / "manifests/content_index.json", payload)
    return payload


def write_validation_report(
    root: str | Path,
    checks: Mapping[str, object],
) -> None:
    atomic_json(
        Path(root) / "reports/validation_report.json",
        {
            "schema_version": "midogpp_residual_topup_fresh_validation_v1",
            "status": "PASS",
            "validator": "validate_residual_topup_fresh_bundle",
            "checks": dict(checks),
        },
    )


def load_workspace_provenance(
    path: Path,
    *,
    config: ResidualTopupFreshConfig,
) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            "Fresh Stage-70 requires workspace-prepared input provenance."
        ) from exc
    rows = payload.get("input_artifacts") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, dict)
        or payload.get("experiment_id") != config.experiment_id
        or payload.get("stage") != "70_frozen_policy_downstream"
        or payload.get("claim_scope") != "synthetic_downstream_utility"
        or payload.get("selection_used_target_eval_artifacts") is not False
        or not isinstance(rows, list)
        or len(rows) != len(INPUT_ARTIFACT_IDS)
        or {
            str(row.get("artifact_id", ""))
            for row in rows
            if isinstance(row, Mapping)
        }
        != set(INPUT_ARTIFACT_IDS)
    ):
        raise ProtocolError("Fresh Stage-70 workspace provenance drifted.")
    return payload


def write_resolved_config(path: Path, source: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve() == source.resolve():
        return
    try:
        source_bytes = source.read_bytes()
    except OSError as exc:
        raise ProtocolError("Cannot read fresh Stage-70 resolved config.") from exc
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(source_bytes)
    os.replace(temporary, path)


def write_dataclass_csv(
    path: Path,
    rows: Sequence[object],
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            payload: dict[str, object] = {}
            for column in columns:
                value = getattr(row, column)
                if isinstance(value, float) and not math.isfinite(value):
                    value = ""
                payload[column] = value
            writer.writerow(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "atomic_json",
    "load_workspace_provenance",
    "sha256_file",
    "write_content_index",
    "write_dataclass_csv",
    "write_resolved_config",
    "write_validation_report",
)
