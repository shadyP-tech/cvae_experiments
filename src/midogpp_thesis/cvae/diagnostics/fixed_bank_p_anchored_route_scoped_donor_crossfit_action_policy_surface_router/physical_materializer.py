"""Neutral fixed-bank materialization and canonical seed-lattice adaptation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from ...runtime.fixed_bank_a1_action_predictions import (
    materialize_fixed_bank_a1_action_predictions,
)
from ...runtime.frozen_source_streams import materialize_frozen_source_streams
from .identity import canonical_hash
from .physical_actions import action_library_by_target
from .physical_contracts import (
    CenterPhysicalSurface,
    MaterializedPhysicalBank,
    PhysicalSurface,
)


def _canonical_seed_keys() -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(training_seed), int(generation_seed))
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )


def _canonical_action_probabilities(
    cells: tuple[object, ...],
    *,
    target_center: str,
    action_id: str,
) -> np.ndarray:
    selected = tuple(
        cell
        for cell in cells
        if str(getattr(cell, "target_center")) == target_center
        and str(getattr(cell, "action_id")) == action_id
    )
    expected_seed_keys = _canonical_seed_keys()
    by_seed_key = {
        (
            int(getattr(cell, "training_seed")),
            int(getattr(cell, "generation_seed")),
        ): cell
        for cell in selected
    }
    if (
        len(selected) != len(expected_seed_keys)
        or len(by_seed_key) != len(selected)
        or set(by_seed_key) != set(expected_seed_keys)
    ):
        raise ProtocolError("P-DCAPS exact-nine physical cell coverage drifted.")
    return np.stack(
        [
            np.asarray(
                getattr(by_seed_key[seed_key], "probabilities"),
                dtype=np.float32,
            )
            for seed_key in expected_seed_keys
        ]
    )


def build_physical_surface(prediction_store: object) -> PhysicalSurface:
    """Adapt a neutral store using canonical center, action, and seed ordering."""

    cells = tuple(getattr(prediction_store, "cells"))
    store_hash = str(getattr(prediction_store, "store_hash"))
    rows_by_center = getattr(prediction_store, "rows_by_center")
    cases_by_center = getattr(prediction_store, "case_ids_by_center")
    centers: list[CenterPhysicalSurface] = []
    for center in CENTERS:
        action_rows = tuple(
            (
                action.action_id,
                _canonical_action_probabilities(
                    cells,
                    target_center=center,
                    action_id=action.action_id,
                ),
            )
            for action in action_library_by_target()[center]
        )
        centers.append(
            CenterPhysicalSurface(
                center,
                tuple(rows_by_center[center]),
                tuple(cases_by_center[center]),
                action_rows,
                store_hash,
            )
        )
    return PhysicalSurface(tuple(centers), store_hash)


def physical_partition_hash(frame: object) -> str:
    rows = tuple(getattr(frame, "rows"))
    return canonical_hash(
        {
            "schema_version": "pdcaps_physical_partition_v1",
            "rows": [
                [
                    str(getattr(row, "center")),
                    str(getattr(row, "case_id")),
                    str(getattr(row, "sample_id")),
                ]
                for row in rows
            ],
            "labels_used": False,
        }
    )


def materialize_physical_bank(
    config: object,
    generation_lock: object,
    frame: object,
    *,
    root: Path,
    prediction_scratch_root: Path,
) -> MaterializedPhysicalBank:
    """Materialize only neutral source streams and fresh physical predictions."""

    source_root = root / "source_runtime"
    source_cache = materialize_frozen_source_streams(
        config, generation_lock, root=source_root
    )
    prediction = materialize_fixed_bank_a1_action_predictions(
        config,
        source_cache,
        frame,
        partition_hash=physical_partition_hash(frame),
        action_library=action_library_by_target(),
        root=root,
        scratch_root=prediction_scratch_root,
    )
    return MaterializedPhysicalBank(
        source_cache,
        prediction,
        build_physical_surface(prediction.store),
    )


__all__ = (
    "build_physical_surface",
    "materialize_physical_bank",
    "physical_partition_hash",
)
