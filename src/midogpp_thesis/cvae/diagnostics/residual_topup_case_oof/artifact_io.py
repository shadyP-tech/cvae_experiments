"""Atomic and closed-world persistence for the case-OOF diagnostic."""

from __future__ import annotations

from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
from io import StringIO
from typing import Iterator, Mapping, Sequence

from ...protocol import ProtocolError


def json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(columns), extrasaction="raise"
            )
            writer.writeheader()
            for row in rows:
                if set(row) != set(columns):
                    raise ProtocolError(
                        f"Case-OOF CSV schema drifted for {path.name}."
                    )
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_save_npz(path: Path, **arrays: object) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_save_npy(path: Path, values: object) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(values), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read case-OOF JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Case-OOF JSON must be an object.")
    return payload


def read_csv_rows(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ProtocolError("Case-OOF CSV lacks a header.")
            return tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ProtocolError(f"Cannot read case-OOF CSV: {path}.") from exc


def read_utf8_text_exact(path: Path) -> str:
    """Read UTF-8 text without universal-newline translation."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except (OSError, UnicodeError) as exc:
        raise ProtocolError(f"Cannot read case-OOF text: {path}.") from exc


def persist_or_validate_json(path: Path, payload: Mapping[str, object]) -> None:
    """Create an immutable JSON member, or fail if a resumed copy drifted."""

    expected = json_ready(payload)
    if path.is_file():
        if read_json(path) != expected:
            raise ProtocolError(f"Case-OOF resumed JSON drifted: {path}.")
        return
    atomic_write_json(path, payload)


def persist_or_validate_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str],
) -> None:
    """Create an immutable CSV member, or fail on byte-level resume drift."""

    expected = _render_csv(rows, columns=columns)
    if path.is_file():
        observed = read_utf8_text_exact(path)
        if observed != expected:
            raise ProtocolError(f"Case-OOF resumed CSV drifted: {path}.")
        return
    atomic_write_csv_rows(path, rows, columns=columns)


def _render_csv(
    rows: Sequence[Mapping[str, object]], *, columns: Sequence[str]
) -> str:
    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
    writer.writeheader()
    for row in rows:
        if set(row) != set(columns):
            raise ProtocolError("Case-OOF CSV schema drifted during validation.")
        writer.writerow(row)
    return handle.getvalue()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_closed_world(
    root: Path,
    *,
    required_files: Sequence[str],
    allow_incomplete: bool,
) -> None:
    allowed = set(required_files)
    unexpected: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in allowed or relative == ".run.lock":
            continue
        if allow_incomplete and (
            relative.startswith("checkpoints/") or ".tmp" in path.name
        ):
            continue
        unexpected.append(relative)
    if unexpected:
        raise ProtocolError(
            f"Case-OOF artifact contains unexpected files: {sorted(unexpected)}."
        )


def prune_stale_temp_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and ".tmp" in path.name:
            path.unlink()


@contextmanager
def exclusive_run_lock(root: Path) -> Iterator[None]:
    path = root / ".run.lock"
    handle = path.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            raise ProtocolError(
                "Another case-OOF runner owns this artifact."
            ) from exc
        yield
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


__all__ = (
    "assert_closed_world",
    "atomic_save_npy",
    "atomic_save_npz",
    "atomic_write_csv_rows",
    "atomic_write_json",
    "exclusive_run_lock",
    "json_ready",
    "prune_stale_temp_files",
    "persist_or_validate_csv",
    "persist_or_validate_json",
    "read_csv_rows",
    "read_json",
    "read_utf8_text_exact",
    "sha256_file",
)
