"""Opaque consumed-test identities and the label-free S4 embedding frame."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from .experiment_contracts import CENTERS, EXPECTED_TEST_ROW_COUNT


def canonical_hash(value: object) -> str:
    """Return the bundle-wide canonical SHA-256 digest."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, order=True)
class TestRowIdentity:
    """Identity-only view of one consumed test row."""

    row_ordinal: int
    manifest_row_index: int
    evaluation_row_id: str
    case_id: str
    center: str
    split: str = "test"

    def __post_init__(self) -> None:
        if (
            isinstance(self.row_ordinal, bool)
            or self.row_ordinal < 0
            or isinstance(self.manifest_row_index, bool)
            or self.manifest_row_index < 0
            or not self.evaluation_row_id
            or not self.case_id
            or self.center not in CENTERS
            or self.split != "test"
        ):
            raise ProtocolError("S4 test-row identity drifted.")

    @property
    def sample_id(self) -> str:
        return self.evaluation_row_id

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class LabelFreeTestFrame:
    """Immutable label-free cache surface accepted by neutral prediction code."""

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
            or len(rows) != EXPECTED_TEST_ROW_COUNT
            or tuple(by_center) != CENTERS
            or tuple(row for center in CENTERS for row in by_center[center]) != rows
            or tuple(row.row_ordinal for row in rows) != tuple(range(len(rows)))
            or len({row.evaluation_row_id for row in rows}) != len(rows)
            or any(row.center != center for center in CENTERS for row in by_center[center])
        ):
            raise ProtocolError("S4 label-free frame is malformed.")
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
            raise ProtocolError("S4 embedding row identity drifted.")
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


__all__ = ("LabelFreeTestFrame", "TestRowIdentity", "canonical_hash")
