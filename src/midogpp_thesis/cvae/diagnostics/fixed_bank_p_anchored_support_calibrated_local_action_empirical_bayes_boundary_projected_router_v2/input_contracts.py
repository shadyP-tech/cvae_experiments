"""Label-free consumed-test DTOs for the SCALE-BP v2 input boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .hashing import canonical_hash, require_sha256
from .identity import CENTERS, FEATURE_DIM, GovernanceError


@dataclass(frozen=True, order=True, slots=True)
class TestRowIdentity:
    """One stable test row without a path or target-label capability."""

    row_ordinal: int
    manifest_row_index: int
    sample_id: str
    case_id: str
    center: str
    patient_slide_group_id: str
    split: str = "test"

    def __post_init__(self) -> None:
        if (
            isinstance(self.row_ordinal, bool)
            or int(self.row_ordinal) < 0
            or isinstance(self.manifest_row_index, bool)
            or int(self.manifest_row_index) < 0
            or str(self.center) not in CENTERS
            or self.split != "test"
            or not all(
                str(value)
                for value in (
                    self.sample_id,
                    self.case_id,
                    self.patient_slide_group_id,
                )
            )
        ):
            raise GovernanceError("SCALE-BP v2 test-row identity drifted.")
        object.__setattr__(self, "row_ordinal", int(self.row_ordinal))
        object.__setattr__(self, "manifest_row_index", int(self.manifest_row_index))
        object.__setattr__(self, "sample_id", str(self.sample_id))
        object.__setattr__(self, "case_id", str(self.case_id))
        object.__setattr__(self, "center", str(self.center))
        object.__setattr__(
            self, "patient_slide_group_id", str(self.patient_slide_group_id)
        )

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
            "patient_slide_group_id": self.patient_slide_group_id,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class LabelFreeTestFrame:
    """Immutable float32 embeddings with no label or sample-path field."""

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
            values.shape != (len(rows), FEATURE_DIM)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or tuple(by_center) != tuple(CENTERS)
            or tuple(row for center in CENTERS for row in by_center[center]) != rows
            or tuple(row.row_ordinal for row in rows) != tuple(range(len(rows)))
            or len({row.sample_id for row in rows}) != len(rows)
            or len({row.manifest_row_index for row in rows}) != len(rows)
            or len({(row.center, row.case_id) for row in rows}) != 218
            or any(row.center != center for center in CENTERS for row in by_center[center])
        ):
            raise GovernanceError("SCALE-BP v2 label-free test frame drifted.")
        frozen = np.ascontiguousarray(values, dtype=np.float32)
        frozen.setflags(write=False)
        binding = MappingProxyType(dict(self.cache_binding))
        binding_hash = require_sha256(
            binding.get("cache_content_hash"), "cache-content hash"
        )
        row_order_hash = require_sha256(
            binding.get("row_order_hash"), "row-order hash"
        )
        payload = {
            "schema_version": "scale_bp_v2_label_free_test_frame_v1",
            "row_count": len(rows),
            "case_count": 218,
            "centers": list(CENTERS),
            "cache_content_hash": binding_hash,
            "row_order_hash": row_order_hash,
            "row_identity_hash": canonical_hash([row.to_payload() for row in rows]),
            "labels_present": False,
            "sample_paths_present": False,
        }
        object.__setattr__(self, "embeddings", frozen)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "rows_by_center", MappingProxyType(by_center))
        object.__setattr__(self, "cache_binding", binding)
        object.__setattr__(self, "frame_hash", canonical_hash(payload))

    @property
    def cache_binding_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": "scale_bp_v2_test_cache_binding_v1",
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
            raise GovernanceError("SCALE-BP v2 embedding identity drifted.")
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


__all__ = ("LabelFreeTestFrame", "TestRowIdentity")
