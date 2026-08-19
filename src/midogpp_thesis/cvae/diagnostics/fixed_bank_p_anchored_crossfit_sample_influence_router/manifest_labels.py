"""Capability-scoped decoder for the canonical MIDOG++ manifest."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from ....data.contract.stage70_target_evaluation.contracts import evaluation_row_id
from ...protocol import ProtocolError
from .contracts import BinaryLabel


def read_scoped_manifest_labels(
    config: object,
    frame: object,
    *,
    allowed_keys: frozenset[tuple[str, str, str]],
    role: str,
) -> Sequence[BinaryLabel]:
    """Decode only explicitly granted rows; skipped CSV rows are never parsed."""

    frame_rows = tuple(getattr(frame, "rows"))
    universe = {(row.center, row.case_id, row.sample_id): row for row in frame_rows}
    frame_by_ordinal = {
        row.manifest_row_index: (row.center, row.case_id, row.sample_id)
        for row in frame_rows
    }
    if (
        len(universe) != len(frame_rows)
        or len(frame_by_ordinal) != len(frame_rows)
        or not allowed_keys
        or not set(allowed_keys) <= set(universe)
    ):
        raise ProtocolError("PCSI label grant escapes sealed rows.")
    ordered = tuple(
        key
        for row in frame_rows
        if (key := (row.center, row.case_id, row.sample_id)) in allowed_keys
    )
    requested_by_ordinal = {
        universe[key].manifest_row_index: key for key in ordered
    }
    found: dict[tuple[str, str, str], BinaryLabel] = {}
    seen_frame_ordinals: set[int] = set()
    manifest_path = Path(getattr(config, "test_manifest_path"))
    manifest_hash = str(getattr(config, "expected_manifest_sha256"))
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            header = tuple(next(csv.reader((handle.readline(),))))
            required = ("center", "case_id", "label")
            if any(column not in header for column in required):
                raise ProtocolError("PCSI manifest header drifted.")
            positions = {column: header.index(column) for column in required}
            for ordinal, raw_line in enumerate(handle):
                if ordinal in frame_by_ordinal:
                    seen_frame_ordinals.add(ordinal)
                expected_key = requested_by_ordinal.get(ordinal)
                if expected_key is None:
                    continue
                values = tuple(next(csv.reader((raw_line,))))
                if len(values) != len(header):
                    raise ProtocolError("PCSI granted manifest row drifted.")
                key = (
                    values[positions["center"]],
                    values[positions["case_id"]],
                    evaluation_row_id(manifest_hash, ordinal),
                )
                if key != expected_key or key in found:
                    raise ProtocolError("PCSI manifest order drifted.")
                found[key] = BinaryLabel(
                    *key, int(values[positions["label"]]), role
                )
    except ProtocolError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Cannot load PCSI scoped labels.") from exc
    if seen_frame_ordinals != set(frame_by_ordinal) or set(found) != set(ordered):
        raise ProtocolError("PCSI manifest coverage drifted.")
    return tuple(found[key] for key in ordered)


__all__ = ("read_scoped_manifest_labels",)
