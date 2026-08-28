"""Immutable, label-free consumed-test DTOs for SCEPTRE v4."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..experiment_contracts import (
    EXPECTED_TEST_CASES,
    EXPECTED_TEST_CASES_BY_CENTER,
    EXPECTED_TEST_FEATURE_DIM,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
)
from ..identity import canonical_hash


@dataclass(frozen=True, order=True, slots=True)
class TestRowIdentity:
    """One neutral row identity; neither target label nor sample path is stored."""

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
            or not self.evaluation_row_id.startswith("eval_")
            or len(self.evaluation_row_id) != 69
            or any(ch not in "0123456789abcdef" for ch in self.evaluation_row_id[5:])
            or not self.case_id
            or self.center not in CENTERS
            or self.split != "test"
        ):
            raise ProtocolError("SCEPTRE v4 test-row identity drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "manifest_row_index": self.manifest_row_index,
            "evaluation_row_id": self.evaluation_row_id,
            "case_id": self.case_id,
            "center": self.center,
            "split": self.split,
        }

    @property
    def sample_id(self) -> str:
        """Compatibility alias for the neutral Stage-70 evaluation-row ID."""

        return self.evaluation_row_id


@dataclass(frozen=True, slots=True)
class LabelFreeTestFrame:
    """Read-only float32 embeddings and whole-case identities only."""

    embeddings: np.ndarray
    rows: tuple[TestRowIdentity, ...]
    rows_by_center: Mapping[str, tuple[TestRowIdentity, ...]]
    cases_by_center: Mapping[str, tuple[str, ...]]
    cache_binding: Mapping[str, object]
    canonical_coverage: bool = True

    def __post_init__(self) -> None:
        values = np.asarray(self.embeddings)
        rows = tuple(self.rows)
        by_center = {
            str(center): tuple(center_rows)
            for center, center_rows in self.rows_by_center.items()
        }
        cases = {
            str(center): tuple(str(case_id) for case_id in center_cases)
            for center, center_cases in self.cases_by_center.items()
        }
        expected_rows = EXPECTED_TEST_ROWS if self.canonical_coverage else len(rows)
        if (
            values.shape != (len(rows), EXPECTED_TEST_FEATURE_DIM)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or len(rows) != expected_rows
            or tuple(by_center) != tuple(CENTERS)
            or tuple(cases) != tuple(CENTERS)
            or tuple(row for center in CENTERS for row in by_center[center]) != rows
            or tuple(row.row_ordinal for row in rows) != tuple(range(len(rows)))
            or len({row.evaluation_row_id for row in rows}) != len(rows)
            or len({row.manifest_row_index for row in rows}) != len(rows)
            or any(
                tuple(dict.fromkeys(row.case_id for row in by_center[center]))
                != cases[center]
                for center in CENTERS
            )
            or len({(row.center, row.case_id) for row in rows})
            != sum(len(cases[center]) for center in CENTERS)
        ):
            raise ProtocolError("SCEPTRE v4 label-free frame drifted.")
        if self.canonical_coverage and (
            {center: len(by_center[center]) for center in CENTERS}
            != EXPECTED_TEST_ROWS_BY_CENTER
            or {center: len(cases[center]) for center in CENTERS}
            != EXPECTED_TEST_CASES_BY_CENTER
            or sum(len(value) for value in cases.values()) != EXPECTED_TEST_CASES
        ):
            raise ProtocolError("SCEPTRE v4 canonical whole-case coverage drifted.")
        frozen = np.array(values, dtype=np.float32, order="C", copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "embeddings", frozen)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "rows_by_center", MappingProxyType(by_center))
        object.__setattr__(self, "cases_by_center", MappingProxyType(cases))
        object.__setattr__(
            self, "cache_binding", MappingProxyType(dict(self.cache_binding))
        )

    @property
    def cache_binding_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": "sceptre_v4_test_cache_binding_v1",
                "cache_binding": dict(self.cache_binding),
            }
        )

    @property
    def case_count(self) -> int:
        return sum(len(value) for value in self.cases_by_center.values())

    def embeddings_for(self, rows: Sequence[TestRowIdentity]) -> np.ndarray:
        ordinals = np.asarray([row.row_ordinal for row in rows], dtype=np.int64)
        if (
            not len(ordinals)
            or np.any(ordinals < 0)
            or np.any(ordinals >= len(self.rows))
            or tuple(self.rows[int(index)].evaluation_row_id for index in ordinals)
            != tuple(row.evaluation_row_id for row in rows)
        ):
            raise ProtocolError("SCEPTRE v4 embedding identity drifted.")
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


__all__ = ("LabelFreeTestFrame", "TestRowIdentity")

