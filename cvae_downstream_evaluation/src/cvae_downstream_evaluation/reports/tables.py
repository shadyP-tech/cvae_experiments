"""CSV table validation helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence


def require_columns(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    missing = [column for column in columns if any(column not in row for row in rows)]
    if missing:
        raise ValueError(f"Rows are missing required columns: {sorted(set(missing))}")


def write_rows(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    require_columns(rows, columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
