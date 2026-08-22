"""Small shared primitives for persisted CBPUPR validation modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json


Row = Mapping[str, object]


def index_rows(
    rows: Sequence[Row], fields: Sequence[str], role: str
) -> dict[tuple[str, ...], Row]:
    result: dict[tuple[str, ...], Row] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            fail(role)
        key = tuple(str(row.get(field, "")) for field in fields)
        if any(not value for value in key) or key in result:
            fail(f"{role} unique keys")
        result[key] = row
    return result


def string_list(row: Row, key: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    value = row.get(key)
    if not isinstance(value, list) or (not value and not allow_empty):
        fail(f"{key} list")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        fail(f"{key} identities")
    return result


def mapping_field(row: Row, key: str) -> Row:
    value = row.get(key)
    if not isinstance(value, Mapping):
        fail(f"{key} mapping")
    return value


def table_rows(root: Path, name: str) -> list[Row]:
    payload = read_json(root / "tables" / f"{name}.json")
    rows = payload.get("rows")
    if (
        not isinstance(rows, list)
        or payload.get("row_count") != len(rows)
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        fail(f"{name} table")
    return rows


def support_identities(
    plans: Mapping[tuple[str, str], Row], center: str, held_case: str
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            (center, case, sample)
            for (observed, case), row in plans.items()
            if observed == center and case != held_case
            for sample in string_list(row, "evaluation_sample_ids")
        )
    )


def fail(role: str) -> None:
    raise ProtocolError(f"CBPUPR persisted {role} drifted.")


__all__ = (
    "Row",
    "fail",
    "index_rows",
    "mapping_field",
    "string_list",
    "support_identities",
    "table_rows",
)
