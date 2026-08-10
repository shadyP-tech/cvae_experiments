"""Package-owned label-free consumed-test identities and embedding frame."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from .constants import MIDOGPP_CENTERS
from .experiment_contracts import EXPECTED_TEST_ROW_COUNT
from .hashing import canonical_hash, nonempty_text


@dataclass(frozen=True, order=True)
class TestRowIdentity:
    row_ordinal: int
    manifest_row_index: int
    evaluation_row_id: str
    case_id: str
    center: str
    split: str = "test"

    def __post_init__(self) -> None:
        if (
            type(self.row_ordinal) is not int
            or self.row_ordinal < 0
            or type(self.manifest_row_index) is not int
            or self.manifest_row_index < 0
            or self.center not in MIDOGPP_CENTERS
            or self.split != "test"
        ):
            raise ProtocolError("Actionability test-row identity drifted.")
        nonempty_text(self.evaluation_row_id, "evaluation_row_id")
        nonempty_text(self.case_id, "case_id")

    @property
    def sample_id(self) -> str:
        return self.evaluation_row_id

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


def row_identity_hash(rows: Sequence[TestRowIdentity]) -> str:
    return canonical_hash([row.to_payload() for row in rows])


@dataclass(frozen=True)
class LabelFreeTestFrame:
    embeddings: np.ndarray
    rows: tuple[TestRowIdentity, ...]
    rows_by_center: Mapping[str, tuple[TestRowIdentity, ...]]
    cache_binding: Mapping[str, object]

    def __post_init__(self) -> None:
        values = np.asarray(self.embeddings)
        rows = tuple(self.rows)
        by_center = {
            str(key): tuple(value) for key, value in self.rows_by_center.items()
        }
        if (
            values.shape != (len(rows), COMMON_OUTPUT_DIM)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or tuple(by_center) != MIDOGPP_CENTERS
            or tuple(row for center in MIDOGPP_CENTERS for row in by_center[center])
            != rows
            or len({row.evaluation_row_id for row in rows}) != len(rows)
            or len({row.manifest_row_index for row in rows}) != len(rows)
        ):
            raise ProtocolError("Actionability label-free test frame is malformed.")
        if len(rows) > 1_000 and len(rows) != EXPECTED_TEST_ROW_COUNT:
            raise ProtocolError("Actionability consumed-test row count drifted.")
        frozen = np.ascontiguousarray(values, dtype=np.float32)
        frozen.setflags(write=False)
        object.__setattr__(self, "embeddings", frozen)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "rows_by_center", MappingProxyType(by_center))
        object.__setattr__(self, "cache_binding", MappingProxyType(dict(self.cache_binding)))

    @property
    def cache_binding_hash(self) -> str:
        return canonical_hash(dict(self.cache_binding))

    def embeddings_for(self, rows: Sequence[TestRowIdentity]) -> np.ndarray:
        ordinals = np.asarray([row.row_ordinal for row in rows], dtype=np.int64)
        if (
            not len(ordinals)
            or np.any(ordinals < 0)
            or np.any(ordinals >= len(self.rows))
            or tuple(self.rows[int(index)].evaluation_row_id for index in ordinals)
            != tuple(row.evaluation_row_id for row in rows)
        ):
            raise ProtocolError("Actionability test-row slice drifted.")
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


__all__ = ("LabelFreeTestFrame", "TestRowIdentity", "row_identity_hash")
