from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from reporting import write_csv_rows


@dataclass(frozen=True)
class TableOutput:
    path: str
    rows: Sequence[Mapping[str, object]]
    columns: Sequence[str] | None = None


def write_table_outputs(root: Path, specs: Sequence[TableOutput]) -> None:
    for spec in specs:
        write_csv_rows(root / spec.path, spec.rows, columns=spec.columns)
