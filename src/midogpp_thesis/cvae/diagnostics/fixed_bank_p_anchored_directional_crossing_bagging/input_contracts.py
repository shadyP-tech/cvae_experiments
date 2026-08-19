"""Path-free, label-free contracts for the consumed MIDOG++ test cache."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from .constants import CENTERS, EXPECTED_TEST_ROW_COUNT
from .hashing import canonical_hash


@dataclass(frozen=True, order=True)
class TestRowIdentity:
    """One stable cache row without its filesystem path or target label."""

    row_ordinal: int
    manifest_row_index: int
    sample_id: str
    case_id: str
    center: str
    split: str = "test"

    def __post_init__(self) -> None:
        if (
            isinstance(self.row_ordinal, bool)
            or int(self.row_ordinal) < 0
            or isinstance(self.manifest_row_index, bool)
            or int(self.manifest_row_index) < 0
            or str(self.center) not in CENTERS
            or self.split != "test"
            or not str(self.sample_id)
            or not str(self.case_id)
        ):
            raise ProtocolError("PDCB test-row identity drifted.")
        object.__setattr__(self, "row_ordinal", int(self.row_ordinal))
        object.__setattr__(self, "manifest_row_index", int(self.manifest_row_index))
        object.__setattr__(self, "sample_id", str(self.sample_id))
        object.__setattr__(self, "case_id", str(self.case_id))
        object.__setattr__(self, "center", str(self.center))

    @property
    def evaluation_row_id(self) -> str:
        return self.sample_id

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class LabelFreeTestFrame:
    """Immutable float32 embeddings with no label or sample-path capability."""

    embeddings: np.ndarray
    rows: tuple[TestRowIdentity, ...]
    rows_by_center: Mapping[str, tuple[TestRowIdentity, ...]]
    cache_binding: Mapping[str, object]

    def __post_init__(self) -> None:
        values = np.asarray(self.embeddings)
        rows = tuple(self.rows)
        by_center = {
            str(center): tuple(center_rows)
            for center, center_rows in self.rows_by_center.items()
        }
        if (
            values.shape != (len(rows), COMMON_OUTPUT_DIM)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or tuple(by_center) != CENTERS
            or tuple(row for center in CENTERS for row in by_center[center]) != rows
            or tuple(row.row_ordinal for row in rows) != tuple(range(len(rows)))
            or len({row.sample_id for row in rows}) != len(rows)
            or len({row.manifest_row_index for row in rows}) != len(rows)
            or (len(rows) > 1_000 and len(rows) != EXPECTED_TEST_ROW_COUNT)
        ):
            raise ProtocolError("PDCB label-free frame drifted.")
        frozen = np.ascontiguousarray(values, dtype=np.float32)
        frozen.setflags(write=False)
        object.__setattr__(self, "embeddings", frozen)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "rows_by_center", MappingProxyType(by_center))
        object.__setattr__(
            self, "cache_binding", MappingProxyType(dict(self.cache_binding))
        )

    @property
    def cache_binding_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": "fixed_bank_pdcb_test_cache_binding_v1",
                "cache_binding": dict(self.cache_binding),
            }
        )

    def embeddings_for(self, rows: Sequence[TestRowIdentity]) -> np.ndarray:
        ordinals = np.asarray([row.row_ordinal for row in rows], dtype=np.int64)
        if (
            not len(ordinals)
            or np.any(ordinals < 0)
            or np.any(ordinals >= len(self.rows))
            or tuple(self.rows[int(index)].sample_id for index in ordinals)
            != tuple(row.sample_id for row in rows)
        ):
            raise ProtocolError("PDCB embedding identity drifted.")
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


__all__ = ("LabelFreeTestFrame", "TestRowIdentity")
