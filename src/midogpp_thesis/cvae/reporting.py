"""Small artifact-writing helpers shared by preservation-only runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence


ARTIFACT_DIRS = ("tables", "manifests", "reports")


def prepare_artifact_dirs(root: str | Path) -> Path:
    artifact_root = Path(root)
    for relative in ARTIFACT_DIRS:
        (artifact_root / relative).mkdir(parents=True, exist_ok=True)
    return artifact_root


def write_csv_rows(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        inferred: list[str] = []
        for row in rows:
            for key in row:
                if key not in inferred:
                    inferred.append(str(key))
        columns = tuple(inferred)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
