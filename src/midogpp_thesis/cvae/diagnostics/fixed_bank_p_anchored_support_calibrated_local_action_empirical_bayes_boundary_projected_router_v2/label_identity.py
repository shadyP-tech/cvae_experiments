"""Compact label-free row identity shared with spawned outer-center workers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .artifacts.io import atomic_json
from .hashing import canonical_hash, require_sha256
from .identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_ROW_COUNT,
    GovernanceError,
)
from .input_contracts import LabelFreeTestFrame, TestRowIdentity


IDENTITY_INDEX_SCHEMA = "scale_bp_v2_worker_label_identity_index_v1"


@dataclass(frozen=True, slots=True)
class LabelIdentityFrame:
    """Embedding-free frame sufficient for capability-scoped manifest decoding."""

    rows: tuple[TestRowIdentity, ...]
    rows_by_center: Mapping[str, tuple[TestRowIdentity, ...]]
    frame_hash: str
    identity_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        by_center = {
            str(center): tuple(center_rows)
            for center, center_rows in self.rows_by_center.items()
        }
        expected_cases = dict(EXPECTED_CASE_COUNTS_BY_CENTER)
        if (
            len(rows) != EXPECTED_TEST_ROW_COUNT
            or tuple(by_center) != CENTERS
            or tuple(row for center in CENTERS for row in by_center[center]) != rows
            or tuple(row.row_ordinal for row in rows) != tuple(range(len(rows)))
            or len({row.sample_id for row in rows}) != len(rows)
            or len({row.manifest_row_index for row in rows}) != len(rows)
            or len({(row.center, row.case_id) for row in rows}) != EXPECTED_CASE_COUNT
            or any(
                len({row.case_id for row in by_center[center]})
                != expected_cases[center]
                for center in CENTERS
            )
        ):
            raise GovernanceError("SCALE-BP v2 worker label identity frame drifted.")
        frame_hash = require_sha256(self.frame_hash, "label-free frame hash")
        body = {
            "schema_version": IDENTITY_INDEX_SCHEMA,
            "frame_hash": frame_hash,
            "cache_content_hash": EXPECTED_TEST_CACHE_CONTENT_HASH,
            "row_order_hash": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
            "row_count": len(rows),
            "case_count": EXPECTED_CASE_COUNT,
            "centers": list(CENTERS),
            "rows": [row.to_payload() for row in rows],
            "embeddings_embedded": False,
            "labels_embedded": False,
            "sample_paths_embedded": False,
        }
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "rows_by_center", MappingProxyType(by_center))
        object.__setattr__(self, "frame_hash", frame_hash)
        object.__setattr__(self, "identity_hash", canonical_hash(body))

    @classmethod
    def from_frame(cls, frame: LabelFreeTestFrame) -> "LabelIdentityFrame":
        if not isinstance(frame, LabelFreeTestFrame):
            raise GovernanceError("SCALE-BP v2 identity index requires its cache frame.")
        return cls(frame.rows, frame.rows_by_center, frame.frame_hash)

    def to_payload(self) -> dict[str, object]:
        body = {
            "schema_version": IDENTITY_INDEX_SCHEMA,
            "frame_hash": self.frame_hash,
            "cache_content_hash": EXPECTED_TEST_CACHE_CONTENT_HASH,
            "row_order_hash": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
            "row_count": len(self.rows),
            "case_count": EXPECTED_CASE_COUNT,
            "centers": list(CENTERS),
            "rows": [row.to_payload() for row in self.rows],
            "embeddings_embedded": False,
            "labels_embedded": False,
            "sample_paths_embedded": False,
        }
        return {**body, "identity_hash": self.identity_hash}


def persist_label_identity_index(
    frame: LabelFreeTestFrame, *, path: str | Path
) -> LabelIdentityFrame:
    identity = LabelIdentityFrame.from_frame(frame)
    output = Path(path)
    if not output.is_absolute():
        raise GovernanceError("SCALE-BP v2 label identity index path is not absolute.")
    atomic_json(output, identity.to_payload())
    return identity


def load_label_identity_index(
    path: str | Path, *, expected_identity_hash: object
) -> LabelIdentityFrame:
    source = Path(path)
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise GovernanceError("SCALE-BP v2 label identity index is unsafe.")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GovernanceError("SCALE-BP v2 label identity index is unreadable.") from exc
    rows_payload = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows_payload, list):
        raise GovernanceError("SCALE-BP v2 label identity rows are malformed.")
    rows = tuple(
        TestRowIdentity(
            row_ordinal=int(row["row_ordinal"]),
            manifest_row_index=int(row["manifest_row_index"]),
            sample_id=str(row["sample_id"]),
            case_id=str(row["case_id"]),
            center=str(row["center"]),
            patient_slide_group_id=str(row["patient_slide_group_id"]),
            split=str(row["split"]),
        )
        for row in rows_payload
        if isinstance(row, Mapping)
    )
    by_center = {
        center: tuple(row for row in rows if row.center == center) for center in CENTERS
    }
    identity = LabelIdentityFrame(rows, by_center, str(payload.get("frame_hash", "")))
    expected = require_sha256(expected_identity_hash, "expected label identity hash")
    if payload != identity.to_payload() or identity.identity_hash != expected:
        raise GovernanceError("SCALE-BP v2 label identity index hash drifted.")
    return identity


__all__ = (
    "IDENTITY_INDEX_SCHEMA",
    "LabelIdentityFrame",
    "load_label_identity_index",
    "persist_label_identity_index",
)
