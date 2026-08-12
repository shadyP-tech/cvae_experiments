"""Label-free input contracts for the consumed-test endpoint router."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .contracts import CENTERS, EXPECTED_TEST_ROW_COUNT
from .partitions import LabelFreeCaseRow


FEATURE_DIM = 3_840


def row_identity_hash(rows: Sequence[LabelFreeCaseRow]) -> str:
    """Hash ordered, label-free row identities at a phase boundary."""

    return canonical_sha256([row.identity_payload() for row in rows])


@dataclass(frozen=True)
class LabelFreeTestFrame:
    """Read-only embeddings and opaque identities; outcomes cannot be attached."""

    embeddings: np.ndarray
    rows: tuple[LabelFreeCaseRow, ...]
    rows_by_center: Mapping[str, tuple[LabelFreeCaseRow, ...]]
    cache_binding: Mapping[str, object]

    def __post_init__(self) -> None:
        values = np.asarray(self.embeddings)
        rows = tuple(self.rows)
        grouped = {
            str(center): tuple(center_rows)
            for center, center_rows in self.rows_by_center.items()
        }
        if (
            values.shape != (EXPECTED_TEST_ROW_COUNT, FEATURE_DIM)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or len(rows) != EXPECTED_TEST_ROW_COUNT
            or tuple(grouped) != CENTERS
            or tuple(row for center in CENTERS for row in grouped[center]) != rows
            or tuple(row.row_ordinal for row in rows)
            != tuple(range(EXPECTED_TEST_ROW_COUNT))
            or len({row.evaluation_row_id for row in rows}) != len(rows)
            or len({row.manifest_row_index for row in rows}) != len(rows)
            or any(row.partition_role != "unassigned" for row in rows)
        ):
            raise ProtocolError("Consumed-test label-free frame geometry drifted.")
        frozen = np.ascontiguousarray(values, dtype=np.float32)
        frozen.setflags(write=False)
        object.__setattr__(self, "embeddings", frozen)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "rows_by_center", MappingProxyType(grouped))
        object.__setattr__(
            self, "cache_binding", MappingProxyType(dict(self.cache_binding))
        )

    @property
    def cache_binding_hash(self) -> str:
        return canonical_sha256(dict(self.cache_binding))

    def embeddings_for(self, rows: Sequence[LabelFreeCaseRow]) -> np.ndarray:
        selected = tuple(rows)
        if not selected:
            raise ProtocolError("Consumed-test embedding slice must be nonempty.")
        ordinals = np.asarray([row.row_ordinal for row in selected], dtype=np.int64)
        if (
            np.any(ordinals < 0)
            or np.any(ordinals >= len(self.rows))
            or len(set(map(int, ordinals))) != len(ordinals)
            or tuple(
                self.rows[int(index)].cache_identity_payload() for index in ordinals
            )
            != tuple(row.cache_identity_payload() for row in selected)
        ):
            raise ProtocolError("Consumed-test embedding row identity drifted.")
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


@dataclass(frozen=True)
class MetadataCompatibilityGrid:
    """Manifest-local, label-free 9x8 source-control surface."""

    by_target: Mapping[str, Mapping[str, float]]
    domain_mapping_sha256: str
    grid_hash: str

    def __post_init__(self) -> None:
        normalized = {
            str(target): MappingProxyType(
                {str(source): float(value) for source, value in scores.items()}
            )
            for target, scores in self.by_target.items()
        }
        if (
            tuple(normalized) != CENTERS
            or any(
                set(normalized[target]) != set(CENTERS).difference({target})
                for target in CENTERS
            )
            or any(
                not np.isfinite(value)
                for scores in normalized.values()
                for value in scores.values()
            )
        ):
            raise ProtocolError("Metadata compatibility grid coverage drifted.")
        unhashed = {
            "schema_version": "midogpp_endpoint_router_metadata_grid_v1",
            "domain_mapping_sha256": self.domain_mapping_sha256,
            "by_target": {
                target: dict(normalized[target]) for target in CENTERS
            },
            "label_fields_consumed": False,
            "identity_predictors_emitted": False,
        }
        if self.grid_hash != canonical_sha256(unhashed):
            raise ProtocolError("Metadata compatibility grid hash drifted.")
        object.__setattr__(self, "by_target", MappingProxyType(normalized))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_endpoint_router_metadata_grid_v1",
            "domain_mapping_sha256": self.domain_mapping_sha256,
            "by_target": {
                target: dict(self.by_target[target]) for target in CENTERS
            },
            "label_fields_consumed": False,
            "identity_predictors_emitted": False,
            "grid_hash": self.grid_hash,
        }


__all__ = (
    "FEATURE_DIM",
    "LabelFreeTestFrame",
    "MetadataCompatibilityGrid",
    "row_identity_hash",
)
