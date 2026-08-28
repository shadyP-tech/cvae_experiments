"""V3-owned label-free consumed-test frame contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....protocol import ProtocolError
from ..hashing import canonical_hash, require_sha256
from ..identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    FEATURE_DIM,
)


@dataclass(frozen=True, order=True, slots=True)
class TestRowIdentity:
    row_ordinal: int
    manifest_row_index: int
    sample_id: str
    case_id: str
    center: str
    split: str = "test"

    def __post_init__(self) -> None:
        if (
            type(self.row_ordinal) is not int
            or self.row_ordinal < 0
            or type(self.manifest_row_index) is not int
            or self.manifest_row_index < 0
            or self.center not in CENTERS
            or self.split != "test"
            or not self.sample_id
            or not self.case_id
        ):
            raise ProtocolError("OE-PPUR v3 test-row identity drifted.")

    @property
    def evaluation_row_id(self) -> str:
        return self.sample_id

    def to_payload(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "manifest_row_index": self.manifest_row_index,
            "sample_id": self.sample_id,
            "case_id": self.case_id,
            "center": self.center,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class LabelFreeTestFrame:
    embeddings: np.ndarray
    rows: tuple[TestRowIdentity, ...]
    rows_by_center: Mapping[str, tuple[TestRowIdentity, ...]]
    cache_binding: Mapping[str, object]
    frame_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.embeddings)
        rows = tuple(self.rows)
        by_center = {
            str(center): tuple(center_rows)
            for center, center_rows in self.rows_by_center.items()
        }
        if (
            values.shape != (EXPECTED_TEST_ROW_COUNT, FEATURE_DIM)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or len(rows) != EXPECTED_TEST_ROW_COUNT
            or tuple(by_center) != CENTERS
            or tuple(row for center in CENTERS for row in by_center[center]) != rows
            or tuple(row.row_ordinal for row in rows) != tuple(range(len(rows)))
            or len({row.sample_id for row in rows}) != len(rows)
            or len({row.manifest_row_index for row in rows}) != len(rows)
            or len({(row.center, row.case_id) for row in rows}) != EXPECTED_CASE_COUNT
            or any(row.center != center for center in CENTERS for row in by_center[center])
        ):
            raise ProtocolError("OE-PPUR v3 label-free test frame drifted.")
        frozen = np.ascontiguousarray(values, dtype=np.float32)
        frozen.setflags(write=False)
        binding = MappingProxyType(dict(self.cache_binding))
        content = require_sha256(binding.get("cache_content_hash"), "cache content hash")
        row_order = require_sha256(binding.get("row_order_hash"), "row order hash")
        object.__setattr__(self, "embeddings", frozen)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "rows_by_center", MappingProxyType(by_center))
        object.__setattr__(self, "cache_binding", binding)
        object.__setattr__(
            self,
            "frame_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_label_free_test_frame_v1",
                    "cache_content_hash": content,
                    "row_order_hash": row_order,
                    "row_identity": [row.to_payload() for row in rows],
                    "row_count": EXPECTED_TEST_ROW_COUNT,
                    "case_count": EXPECTED_CASE_COUNT,
                    "labels_present": False,
                    "sample_paths_present": False,
                }
            ),
        )

    @property
    def cache_binding_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": "oe_ppur_v3_test_cache_binding_v1",
                "cache_binding": dict(self.cache_binding),
            }
        )

    def embeddings_for(self, rows: Sequence[TestRowIdentity]) -> np.ndarray:
        selected = tuple(rows)
        ordinals = np.asarray([row.row_ordinal for row in selected], dtype=np.int64)
        if (
            not len(ordinals)
            or np.any(ordinals < 0)
            or np.any(ordinals >= len(self.rows))
            or tuple(self.rows[int(index)].sample_id for index in ordinals)
            != tuple(row.sample_id for row in selected)
        ):
            raise ProtocolError("OE-PPUR v3 embedding identity drifted.")
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


__all__ = ("LabelFreeTestFrame", "TestRowIdentity")
