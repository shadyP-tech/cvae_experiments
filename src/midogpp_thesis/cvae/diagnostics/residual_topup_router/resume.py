"""Durable artifact reuse for interrupted residual top-up runs.

Completed scientific members are immutable resume boundaries.  A restart
reconstructs their expected payloads and reuses the existing bytes only when
they agree; it never silently overwrites a drifted post-seal decision surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .artifact_io import (
    atomic_write_csv_rows,
    atomic_write_json,
    read_csv_rows,
    read_json,
)


def persist_or_validate_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_file():
        if read_json(path) != dict(payload):
            raise ProtocolError(
                f"Residual top-up durable JSON drifted during resume: {path}."
            )
        return
    atomic_write_json(path, payload)


def persist_or_validate_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str],
) -> None:
    if not path.is_file():
        atomic_write_csv_rows(path, rows, columns=columns)
        return
    observed = read_csv_rows(path)
    if len(observed) != len(rows):
        raise ProtocolError(
            f"Residual top-up durable CSV row count drifted: {path}."
        )
    for left, right in zip(observed, rows, strict=True):
        if tuple(left) != tuple(columns) or set(right) != set(columns):
            raise ProtocolError(
                f"Residual top-up durable CSV schema drifted: {path}."
            )
        for key in columns:
            expected = right[key]
            raw = left[key]
            if isinstance(expected, bool):
                equal = raw.strip().lower() == str(expected).lower()
            elif isinstance(expected, (int, float)) and not isinstance(
                expected, bool
            ):
                try:
                    equal = bool(
                        np.isclose(
                            float(raw),
                            float(expected),
                            rtol=1e-12,
                            atol=1e-12,
                        )
                    )
                except ValueError:
                    equal = False
            else:
                equal = raw == str(expected)
            if not equal:
                raise ProtocolError(
                    f"Residual top-up durable CSV value drifted at {key!r}: {path}."
                )


__all__ = ("persist_or_validate_csv", "persist_or_validate_json")
