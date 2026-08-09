"""Label-free consumed-test identities for the case-aware Stage-90 audit."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError


@dataclass(frozen=True)
class TestRowIdentity:
    """Opaque cache identity; it deliberately carries no outcome."""

    row_ordinal: int
    manifest_row_index: int
    evaluation_row_id: str
    case_id: str
    center: str
    partition_role: str = "unassigned"
    split: str = "test"

    def __post_init__(self) -> None:
        if (
            type(self.row_ordinal) is not int
            or self.row_ordinal < 0
            or type(self.manifest_row_index) is not int
            or self.manifest_row_index < 0
            or not self.evaluation_row_id
            or not self.case_id
            or self.center not in CENTERS
            or self.partition_role not in {"unassigned", "support", "evaluation"}
            or self.split != "test"
        ):
            raise ProtocolError("Case-aware audit test-row identity drifted.")

    @property
    def sample_id(self) -> str:
        """Compatibility name for label-free execution leaves.

        The value remains the cache's opaque evaluation-row identifier, never
        a sample filename or label-bearing manifest field.
        """

        return self.evaluation_row_id

    def identity_payload(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "manifest_row_index": self.manifest_row_index,
            "evaluation_row_id": self.evaluation_row_id,
            "case_id": self.case_id,
            "center": self.center,
            "split": self.split,
            "partition_role": self.partition_role,
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
        by_center = {
            str(center): tuple(rows) for center, rows in self.rows_by_center.items()
        }
        if (
            values.shape != (len(self.rows), 3_840)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or tuple(by_center) != CENTERS
            or tuple(row for center in CENTERS for row in by_center[center])
            != self.rows
            or len({row.evaluation_row_id for row in self.rows}) != len(self.rows)
            or len({row.manifest_row_index for row in self.rows}) != len(self.rows)
            or any(row.partition_role != "unassigned" for row in self.rows)
        ):
            raise ProtocolError("Case-aware audit label-free test frame is malformed.")
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
            raise ProtocolError("Case-aware audit row slice drifted.")
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


@dataclass(frozen=True)
class FixedTestPartitionSurface:
    support_rows_by_center: Mapping[str, tuple[TestRowIdentity, ...]]
    evaluation_rows_by_center: Mapping[str, tuple[TestRowIdentity, ...]]
    table_rows: tuple[Mapping[str, object], ...]
    lock_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        support = {
            str(center): tuple(rows)
            for center, rows in self.support_rows_by_center.items()
        }
        evaluation = {
            str(center): tuple(rows)
            for center, rows in self.evaluation_rows_by_center.items()
        }
        if tuple(support) != CENTERS or tuple(evaluation) != CENTERS:
            raise ProtocolError("Case-aware audit partition center coverage drifted.")
        object.__setattr__(self, "support_rows_by_center", MappingProxyType(support))
        object.__setattr__(self, "evaluation_rows_by_center", MappingProxyType(evaluation))
        object.__setattr__(
            self,
            "table_rows",
            tuple(MappingProxyType(dict(row)) for row in self.table_rows),
        )
        object.__setattr__(self, "lock_payload", MappingProxyType(dict(self.lock_payload)))

    @property
    def lock_hash(self) -> str:
        return str(self.lock_payload["support_partition_lock_hash"])


__all__ = (
    "FixedTestPartitionSurface",
    "LabelFreeTestFrame",
    "TestRowIdentity",
    "row_identity_hash",
)
