"""Non-repairing artifact helpers for the prediction-only diagnostic."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, atomic_npz, read_json, sha256_file


def persist_or_validate_json(path: Path, payload: Mapping[str, object]) -> None:
    """Publish one canonical JSON value, or validate the existing value."""

    expected = dict(payload)
    if path.is_file():
        if read_json(path) != expected:
            raise ProtocolError(
                f"Existing prediction-only JSON differs and will not be repaired: {path}."
            )
        return
    atomic_json(path, expected)


def persist_or_validate_csv(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Publish deterministic CSV without silently replacing prior evidence."""

    names = tuple(str(value) for value in fieldnames)
    expected_rows = tuple({name: row.get(name) for name in names} for row in rows)
    if path.is_file():
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                observed = tuple(dict(row) for row in csv.DictReader(handle))
        except OSError as exc:
            raise ProtocolError(f"Cannot read prediction-only CSV: {path}.") from exc
        normalized = tuple(
            {name: _csv_text(row.get(name)) for name in names} for row in expected_rows
        )
        if observed != normalized:
            raise ProtocolError(
                f"Existing prediction-only CSV differs and will not be repaired: {path}."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
            writer.writeheader()
            writer.writerows(expected_rows)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def persist_or_validate_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Publish an exact closed archive or validate all existing members."""

    expected = {
        str(name): np.ascontiguousarray(value) for name, value in arrays.items()
    }
    if path.is_file():
        try:
            archive = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ProtocolError(f"Cannot read prediction-only NPZ: {path}.") from exc
        with archive:
            if tuple(archive.files) != tuple(expected) or any(
                not np.array_equal(archive[name], value)
                for name, value in expected.items()
            ):
                raise ProtocolError(
                    f"Existing prediction-only NPZ differs and will not be repaired: {path}."
                )
        return
    atomic_npz(path, **expected)


def relative_files(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != ".run.lock"
        )
    )


def _csv_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


__all__ = (
    "persist_or_validate_csv",
    "persist_or_validate_json",
    "persist_or_validate_npz",
    "read_json",
    "relative_files",
    "sha256_file",
)
