"""Canonical read-only 9928-by-7 target probability-matrix assembly."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import hashlib
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....protocol import ProtocolError
from ..action_compiler import CompiledActionSurface
from ..candidate_pools import ALL_ACTION_IDS
from ..hashing import canonical_hash, require_sha256
from ..identity import (
    CENTERS,
    EXPECTED_PROBABILITY_MATRIX_SHAPE,
    EXPECTED_TEST_ROWS_BY_CENTER,
)


_MATRIX_TOKEN = object()


@dataclass(frozen=True, slots=True)
class CompiledProbabilityMatrix:
    row_ids: tuple[str, ...]
    center_offsets: Mapping[str, tuple[int, int]]
    action_ids: tuple[str, ...]
    values: np.ndarray
    surface_hashes: tuple[tuple[str, str], ...]
    _factory_token: InitVar[object | None] = None
    matrix_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        rows = tuple(str(value) for value in self.row_ids)
        offsets = {
            str(center): (int(bounds[0]), int(bounds[1]))
            for center, bounds in self.center_offsets.items()
        }
        values = np.asarray(self.values)
        surfaces = tuple(
            (str(center), require_sha256(digest, "compiled surface hash"))
            for center, digest in self.surface_hashes
        )
        if (
            _factory_token is not _MATRIX_TOKEN
            or tuple(offsets) != CENTERS
            or tuple(self.action_ids) != ALL_ACTION_IDS
            or values.shape != EXPECTED_PROBABILITY_MATRIX_SHAPE
            or values.dtype != np.dtype("<f4")
            or not values.flags.c_contiguous
            or not np.isfinite(values).all()
            or np.any((values < 0.0) | (values > 1.0))
            or len(rows) != EXPECTED_PROBABILITY_MATRIX_SHAPE[0]
            or len(set(rows)) != len(rows)
            or tuple(center for center, _ in surfaces) != CENTERS
        ):
            raise ProtocolError("OE-PPUR v3 compiled probability matrix drifted.")
        cursor = 0
        expected_counts = dict(EXPECTED_TEST_ROWS_BY_CENTER)
        for center in CENTERS:
            start, stop = offsets[center]
            if start != cursor or stop - start != expected_counts[center]:
                raise ProtocolError("OE-PPUR v3 compiled center offsets drifted.")
            cursor = stop
        if cursor != len(rows):
            raise ProtocolError("OE-PPUR v3 compiled matrix row coverage drifted.")
        frozen = np.ascontiguousarray(values, dtype=np.dtype("<f4"))
        frozen.setflags(write=False)
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "center_offsets", MappingProxyType(offsets))
        object.__setattr__(self, "action_ids", ALL_ACTION_IDS)
        object.__setattr__(self, "values", frozen)
        object.__setattr__(self, "surface_hashes", surfaces)
        object.__setattr__(
            self,
            "matrix_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_compiled_probability_matrix_v1",
                    "shape": list(frozen.shape),
                    "dtype": frozen.dtype.str,
                    "row_ids_sha256": canonical_hash(rows),
                    "center_offsets": offsets,
                    "action_ids": ALL_ACTION_IDS,
                    "matrix_f4_sha256": _array_sha256(frozen),
                    "surface_hashes": surfaces,
                    "labels_present": False,
                }
            ),
        )

    def center_matrix(self, center: object) -> np.ndarray:
        key = str(center)
        try:
            start, stop = self.center_offsets[key]
        except KeyError as exc:
            raise ProtocolError("OE-PPUR v3 compiled matrix center is unknown.") from exc
        result = self.values[start:stop]
        result.setflags(write=False)
        return result


def assemble_compiled_probability_matrix(
    surfaces: Sequence[CompiledActionSurface],
) -> CompiledProbabilityMatrix:
    rows = tuple(surfaces)
    if (
        len(rows) != len(CENTERS)
        or any(type(value) is not CompiledActionSurface for value in rows)
        or tuple(value.receipt.outer_target_center for value in rows) != CENTERS
        or any(value.receipt.evaluated_center != center for center, value in zip(CENTERS, rows, strict=True))
    ):
        raise ProtocolError("OE-PPUR v3 compiled target-surface inventory drifted.")
    expected_counts = dict(EXPECTED_TEST_ROWS_BY_CENTER)
    row_ids = []
    matrices = []
    offsets = {}
    surface_hashes = []
    cursor = 0
    for center, surface in zip(CENTERS, rows, strict=True):
        matrix = surface.probability_matrix(dtype="<f4")
        if matrix.shape != (expected_counts[center], len(ALL_ACTION_IDS)):
            raise ProtocolError("OE-PPUR v3 compiled target-surface geometry drifted.")
        start, stop = cursor, cursor + len(surface.row_ids)
        offsets[center] = (start, stop)
        cursor = stop
        row_ids.extend(surface.row_ids)
        matrices.append(matrix)
        surface_hashes.append((center, surface.surface_hash))
    return CompiledProbabilityMatrix(
        tuple(row_ids),
        offsets,
        ALL_ACTION_IDS,
        np.ascontiguousarray(np.concatenate(matrices, axis=0), dtype=np.dtype("<f4")),
        tuple(surface_hashes),
        _factory_token=_MATRIX_TOKEN,
    )


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype("<f4"))
    header = f"{array.dtype.str}|{array.shape}".encode("ascii")
    return hashlib.sha256(header + memoryview(array).cast("B")).hexdigest()


__all__ = ("CompiledProbabilityMatrix", "assemble_compiled_probability_matrix")
