"""Label-free consumed-test identities for the case-OOF ceiling."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from .experiment_contracts import CENTERS, EVALUATION_SPLIT, EXPECTED_TEST_ROW_COUNT


@dataclass(frozen=True)
class TestRowIdentity:
    row_ordinal: int
    manifest_row_index: int
    evaluation_row_id: str
    case_id: str
    center: str
    split: str = EVALUATION_SPLIT

    def __post_init__(self) -> None:
        if (
            type(self.row_ordinal) is not int
            or self.row_ordinal < 0
            or type(self.manifest_row_index) is not int
            or self.manifest_row_index < 0
            or not self.evaluation_row_id
            or not self.case_id
            or self.center not in CENTERS
            or self.split != EVALUATION_SPLIT
        ):
            raise ProtocolError("Label-aware test row identity drifted.")

    @property
    def sample_id(self) -> str:
        return self.evaluation_row_id

    def identity_payload(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "manifest_row_index": self.manifest_row_index,
            "evaluation_row_id": self.evaluation_row_id,
            "case_id": self.case_id,
            "center": self.center,
            "split": self.split,
        }


def row_identity_hash(rows: Sequence[TestRowIdentity]) -> str:
    return stable_hash([row.identity_payload() for row in rows])


@dataclass(frozen=True)
class LabelFreeTestFrame:
    embeddings: np.ndarray
    rows: tuple[TestRowIdentity, ...]
    rows_by_center: Mapping[str, tuple[TestRowIdentity, ...]]
    cache_binding: Mapping[str, object]

    def __post_init__(self) -> None:
        values = np.asarray(self.embeddings)
        by_center = {str(center): tuple(rows) for center, rows in self.rows_by_center.items()}
        if (
            values.shape != (len(self.rows), COMMON_OUTPUT_DIM)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or tuple(by_center) != CENTERS
            or tuple(row for center in CENTERS for row in by_center[center]) != self.rows
            or len({row.evaluation_row_id for row in self.rows}) != len(self.rows)
            or len({row.manifest_row_index for row in self.rows}) != len(self.rows)
        ):
            raise ProtocolError("Label-aware label-free test frame is malformed.")
        if len(self.rows) > 1_000 and len(self.rows) != EXPECTED_TEST_ROW_COUNT:
            raise ProtocolError("Label-aware consumed-test row count drifted.")
        frozen = np.ascontiguousarray(values, dtype=np.float32)
        frozen.setflags(write=False)
        object.__setattr__(self, "embeddings", frozen)
        object.__setattr__(self, "rows_by_center", MappingProxyType(by_center))
        object.__setattr__(self, "cache_binding", MappingProxyType(dict(self.cache_binding)))

    @property
    def cache_binding_hash(self) -> str:
        return stable_hash(dict(self.cache_binding))

    def embeddings_for(self, rows: Sequence[TestRowIdentity]) -> np.ndarray:
        ordinals = np.asarray([row.row_ordinal for row in rows], dtype=np.int64)
        if (
            not len(ordinals)
            or np.any(ordinals < 0)
            or np.any(ordinals >= len(self.rows))
            or tuple(self.rows[int(index)].evaluation_row_id for index in ordinals)
            != tuple(row.evaluation_row_id for row in rows)
        ):
            raise ProtocolError("Label-aware test-row slice drifted.")
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


__all__ = ("LabelFreeTestFrame", "TestRowIdentity", "row_identity_hash")
