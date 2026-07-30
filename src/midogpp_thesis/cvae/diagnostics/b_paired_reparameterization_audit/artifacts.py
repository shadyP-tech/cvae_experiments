"""Atomic artifact writers for the Stage-90 paired audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError


CONTENT_INDEX_EXCLUSIONS = frozenset(
    {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
        ".run.lock",
    }
)


def ensure_audit_directories(root: str | Path) -> Path:
    output = Path(root).resolve()
    for relative in (
        "checkpoints",
        "jobs",
        "manifests",
        "provenance",
        "reports",
        "tables",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = tuple(
        fieldnames
        or dict.fromkeys(str(key) for row in rows for key in row).keys()
    )
    if not columns:
        raise ProtocolError(f"Cannot write an empty-schema CSV: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    temporary.replace(output)


def torch_save(path: str | Path, payload: Mapping[str, object]) -> None:
    import torch

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(output)


def read_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected a JSON object: {path}")
    return value


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_rows_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    encoded = json.dumps(
        [dict(row) for row in rows],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_content_index(root: str | Path) -> dict[str, object]:
    output = Path(root).resolve()
    files = []
    for member in sorted(output.rglob("*")):
        if not member.is_file() or ".tmp" in member.name:
            continue
        relative = str(member.relative_to(output))
        if relative in CONTENT_INDEX_EXCLUSIONS:
            continue
        files.append(
            {
                "path": relative,
                "sha256": file_sha256(member),
                "size_bytes": int(member.stat().st_size),
            }
        )
    payload: dict[str, object] = {
        "schema_version": "midogpp_b_paired_reparameterization_content_index_v1",
        "files": files,
        "indexed_file_count": len(files),
        "claim_scope": "diagnostic_only",
        "may_feed_deployable_selection": False,
    }
    write_json(output / "manifests/content_index.json", payload)
    return payload


def validate_content_index(root: str | Path) -> list[str]:
    output = Path(root).resolve()
    errors: list[str] = []
    index = read_json(output / "manifests/content_index.json")
    records = index.get("files", ())
    if not isinstance(records, list):
        return ["content index files must be a list"]
    indexed = {
        str(record.get("path", "")): record
        for record in records
        if isinstance(record, Mapping)
    }
    actual = {
        str(member.relative_to(output)): member
        for member in output.rglob("*")
        if member.is_file()
        and ".tmp" not in member.name
        and str(member.relative_to(output)) not in CONTENT_INDEX_EXCLUSIONS
    }
    if set(indexed) != set(actual):
        errors.append("content-index coverage differs from bundle members")
    for relative, member in actual.items():
        record = indexed.get(relative, {})
        if (
            record.get("sha256") != file_sha256(member)
            or int(record.get("size_bytes", -1)) != member.stat().st_size
        ):
            errors.append(f"content-index hash/size mismatch: {relative}")
    return errors


__all__ = (
    "CONTENT_INDEX_EXCLUSIONS",
    "canonical_rows_sha256",
    "ensure_audit_directories",
    "file_sha256",
    "read_csv",
    "read_json",
    "torch_save",
    "validate_content_index",
    "write_content_index",
    "write_csv",
    "write_json",
)
