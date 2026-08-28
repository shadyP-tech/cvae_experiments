"""Target-excluded synthetic training composition for SCEPTRE v4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_array

from .prediction_contracts import (
    CANDIDATE_EXCLUSION_SENTINEL,
    PRODUCTION_PREDICTION_GEOMETRY,
    PredictionGeometry,
)
from .prediction_io import canonical_sha256


def exact_b_source_centers(
    target_center: str,
    *,
    geometry: PredictionGeometry = PRODUCTION_PREDICTION_GEOMETRY,
) -> tuple[str, ...]:
    """Return the exact ordered ``C - H`` source inventory for B."""

    target = str(target_center)
    if target not in geometry.centers:
        raise ProtocolError("SCEPTRE v4 exact-B target center is unknown.")
    sources = tuple(center for center in geometry.centers if center != target)
    if target in sources or len(sources) != len(geometry.centers) - 1:
        raise ProtocolError("SCEPTRE v4 exact-B source exclusion failed.")
    return sources


def candidate_exclusion_is_valid(
    values: np.ndarray,
    *,
    geometry: PredictionGeometry,
) -> bool:
    """Authenticate the physical H-on-H exclusion mask."""

    candidate = np.asarray(values)
    if (
        candidate.ndim != 3
        or candidate.shape[0] <= 0
        or candidate.shape[1:]
        != (len(geometry.centers), geometry.evaluation_rows)
        or not np.isfinite(candidate).all()
    ):
        return False
    for source_ordinal, source in enumerate(geometry.centers):
        forbidden = geometry.row_slice(source)
        if not np.all(
            candidate[:, source_ordinal, forbidden]
            == CANDIDATE_EXCLUSION_SENTINEL
        ):
            return False
        allowed = np.concatenate(
            (
                candidate[:, source_ordinal, : forbidden.start].reshape(-1),
                candidate[:, source_ordinal, forbidden.stop :].reshape(-1),
            )
        )
        if np.any((allowed < 0.0) | (allowed > 1.0)):
            return False
    return True


def compose_single_source(
    block: np.ndarray,
    *,
    source: str,
    geometry: PredictionGeometry,
) -> tuple[np.ndarray, np.ndarray, str]:
    values = np.ascontiguousarray(block, dtype=np.float32)
    if (
        source not in geometry.centers
        or values.shape != (2 * geometry.source_rows_per_class, geometry.feature_dim)
        or not np.isfinite(values).all()
    ):
        raise ProtocolError("SCEPTRE v4 single-source training block drifted.")
    truth = synthetic_truth(geometry.source_rows_per_class)
    composition_hash = canonical_sha256(
        {
            "family": "single_source",
            "source_center": source,
            "rows_per_class": geometry.source_rows_per_class,
            "embedding_sha256": sha256_array(values),
        }
    )
    return values, truth, composition_hash


def compose_exact_b(
    blocks: Mapping[str, np.ndarray],
    *,
    target_center: str,
    source_order: Sequence[str],
    geometry: PredictionGeometry,
) -> tuple[np.ndarray, np.ndarray, str]:
    target = str(target_center)
    sources = tuple(str(value) for value in source_order)
    expected = exact_b_source_centers(target, geometry=geometry)
    if sources != expected or target in sources:
        raise ProtocolError("SCEPTRE v4 exact-B composition included H or changed C-H order.")
    chunks: list[np.ndarray] = []
    for class_index in (0, 1):
        class_start = class_index * geometry.source_rows_per_class
        for source in sources:
            block = np.asarray(blocks[source])
            if block.shape != (
                2 * geometry.source_rows_per_class,
                geometry.feature_dim,
            ):
                raise ProtocolError("SCEPTRE v4 exact-B source block geometry drifted.")
            chunks.append(
                np.asarray(
                    block[
                        class_start : class_start
                        + geometry.exact_b_prefix_per_source_class
                    ],
                    dtype=np.float32,
                )
            )
    values = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
    expected_rows = 2 * geometry.source_rows_per_class
    if (
        values.shape != (expected_rows, geometry.feature_dim)
        or not np.isfinite(values).all()
    ):
        raise ProtocolError("SCEPTRE v4 exact-B composition geometry drifted.")
    truth = synthetic_truth(geometry.source_rows_per_class)
    composition_hash = canonical_sha256(
        {
            "family": "exact_B",
            "target_center": target,
            "source_centers": list(sources),
            "prefix_per_source_class": geometry.exact_b_prefix_per_source_class,
            "embedding_sha256": sha256_array(values),
            "target_expert_excluded": True,
        }
    )
    return values, truth, composition_hash


def synthetic_truth(rows_per_class: int) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate(
            (
                np.zeros(rows_per_class, dtype=np.uint8),
                np.ones(rows_per_class, dtype=np.uint8),
            )
        ),
        dtype=np.uint8,
    )


__all__ = (
    "candidate_exclusion_is_valid",
    "compose_exact_b",
    "compose_single_source",
    "exact_b_source_centers",
    "synthetic_truth",
)
