"""Faithful fixed-bank controls and strict fold geometry."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import CENTERS, candidate_sources, expected_training_row_count
from .model_contracts import FamilyDesign
from .row_contracts import FixedBankDataset


SOURCE_FEATURE_NAMES = tuple(f"candidate_source::{source}" for source in CENTERS)


def strict_training_indices(
    dataset: FixedBankDataset,
    outer_target_id: str,
    query_id: str,
) -> np.ndarray:
    """Exclude held ``H`` and ``q`` from every H'/q'/e' role."""

    candidate_sources(outer_target_id, query_id)
    held = {outer_target_id, query_id}
    indices = np.fromiter(
        (
            index
            for index, key in enumerate(dataset.row_keys)
            if held.isdisjoint(key)
        ),
        dtype=np.int64,
    )
    if indices.shape != (expected_training_row_count(),):
        raise ProtocolError("Strict fixed-bank H/q training geometry drifted.")
    indices.setflags(write=False)
    return indices


def held_query_indices(
    dataset: FixedBankDataset,
    outer_target_id: str,
    query_id: str,
) -> np.ndarray:
    sources = candidate_sources(outer_target_id, query_id)
    index_by_key = {key: index for index, key in enumerate(dataset.row_keys)}
    indices = np.asarray(
        [index_by_key[(outer_target_id, query_id, source)] for source in sources],
        dtype=np.int64,
    )
    indices.setflags(write=False)
    return indices


def augmented_model_matrix(
    design: FamilyDesign,
    row_indices: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Add the faithful candidate-source G block to a local family design."""

    indices = np.asarray(row_indices, dtype=np.int64)
    local = design.values[indices]
    if design.spec.source_effects_included:
        source_position = {source: index for index, source in enumerate(CENTERS)}
        categorical = np.zeros((len(indices), len(CENTERS)), dtype=np.float64)
        for row_index, design_index in enumerate(indices.tolist()):
            source = design.row_keys[design_index][2]
            categorical[row_index, source_position[source]] = 1.0
        matrix = np.column_stack((categorical, local))
        names = (*SOURCE_FEATURE_NAMES, *design.spec.predictor_names)
    else:
        matrix = local.copy()
        names = design.spec.predictor_names
    if matrix.shape != (len(indices), len(names)) or not np.isfinite(matrix).all():
        raise ProtocolError("Augmented fixed-bank model matrix drifted.")
    matrix.setflags(write=False)
    return matrix, tuple(names)


def legal_candidate_history_counts(
    dataset: FixedBankDataset,
    training_indices: Sequence[int] | np.ndarray,
    outer_target_id: str,
    query_id: str,
) -> tuple[tuple[str, int], ...]:
    keys = tuple(dataset.row_keys[int(index)] for index in training_indices)
    counts = {
        source: sum(key[2] == source for key in keys)
        for source in candidate_sources(outer_target_id, query_id)
    }
    return tuple(counts.items())


__all__ = (
    "SOURCE_FEATURE_NAMES",
    "augmented_model_matrix",
    "held_query_indices",
    "legal_candidate_history_counts",
    "strict_training_indices",
)
