"""Label-free MIDOG++ validation and fixed-support contracts.

These contracts belong to the consumed Stage-90 experiment.  They carry row
identities and embeddings only; labels are deliberately absent from every
constructor in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import CENTERS


@dataclass(frozen=True)
class ValidationRowIdentity:
    row_ordinal: int
    manifest_row_index: int
    sample_id: str
    case_id: str
    center: str
    partition_role: str = "unassigned"
    split: str = "val"

    def __post_init__(self) -> None:
        if (
            isinstance(self.row_ordinal, bool)
            or not isinstance(self.row_ordinal, int)
            or self.row_ordinal < 0
            or isinstance(self.manifest_row_index, bool)
            or not isinstance(self.manifest_row_index, int)
            or self.manifest_row_index < 0
            or not self.sample_id
            or not self.case_id
            or self.center not in CENTERS
            or self.partition_role not in {"unassigned", "support", "evaluation"}
            or self.split != "val"
        ):
            raise ProtocolError("Utility-aligned Stage-90 row identity drifted.")

    def identity_payload(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "manifest_row_index": self.manifest_row_index,
            "sample_id": self.sample_id,
            "case_id": self.case_id,
            "center": self.center,
            "split": self.split,
            "partition_role": self.partition_role,
        }


def row_identity_hash(rows: Sequence[ValidationRowIdentity]) -> str:
    return stable_hash([row.identity_payload() for row in rows])


@dataclass(frozen=True)
class LabelFreeValidationFrame:
    embeddings: np.ndarray
    rows: tuple[ValidationRowIdentity, ...]
    rows_by_center: Mapping[str, tuple[ValidationRowIdentity, ...]]
    cache_binding: Mapping[str, object]

    def __post_init__(self) -> None:
        values = np.asarray(self.embeddings)
        by_center = {
            str(center): tuple(rows) for center, rows in self.rows_by_center.items()
        }
        if (
            values.shape != (len(self.rows), 3840)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or tuple(by_center) != CENTERS
            or tuple(row for center in CENTERS for row in by_center[center])
            != self.rows
            or len({row.sample_id for row in self.rows}) != len(self.rows)
            or any(row.partition_role != "unassigned" for row in self.rows)
        ):
            raise ProtocolError("Utility-aligned Stage-90 frame is malformed.")
        values.setflags(write=False)
        object.__setattr__(self, "embeddings", values)
        object.__setattr__(self, "rows_by_center", MappingProxyType(by_center))
        object.__setattr__(self, "cache_binding", MappingProxyType(dict(self.cache_binding)))

    @property
    def cache_binding_hash(self) -> str:
        return stable_hash(dict(self.cache_binding))

    def embeddings_for(self, rows: Sequence[ValidationRowIdentity]) -> np.ndarray:
        ordinals = np.asarray([row.row_ordinal for row in rows], dtype=np.int64)
        if (
            not len(ordinals)
            or np.any(ordinals < 0)
            or np.any(ordinals >= len(self.rows))
            or tuple(self.rows[int(index)].sample_id for index in ordinals)
            != tuple(row.sample_id for row in rows)
        ):
            raise ProtocolError("Utility-aligned Stage-90 row slice drifted.")
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


@dataclass(frozen=True)
class FixedPartitionSurface:
    support_rows_by_center: Mapping[str, tuple[ValidationRowIdentity, ...]]
    evaluation_rows_by_center: Mapping[str, tuple[ValidationRowIdentity, ...]]
    table_rows: tuple[Mapping[str, object], ...]
    lock_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        support = {str(key): tuple(value) for key, value in self.support_rows_by_center.items()}
        evaluation = {
            str(key): tuple(value) for key, value in self.evaluation_rows_by_center.items()
        }
        if tuple(support) != CENTERS or tuple(evaluation) != CENTERS:
            raise ProtocolError("Utility-aligned partition center coverage drifted.")
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
    "FixedPartitionSurface",
    "LabelFreeValidationFrame",
    "ValidationRowIdentity",
    "row_identity_hash",
)
