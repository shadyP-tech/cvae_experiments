"""Single materialization bridge from neutral runtime to SCALE-BP v2 science."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...runtime.fixed_bank_a1_action_predictions import (
    materialize_fixed_bank_a1_action_predictions,
)
from ...runtime.frozen_source_streams import materialize_frozen_source_streams

from .hashing import canonical_hash
from .identity import EXPECTED_PHYSICAL_CELL_COUNT, GovernanceError
from .physical import (
    PhysicalStoreAdapter,
    action_library_by_target,
    adapt_prediction_store,
)


@dataclass(frozen=True, slots=True)
class MaterializedPhysicalBank:
    source_cache: object
    prediction_seal: object
    store: PhysicalStoreAdapter
    partition_hash: str

    @property
    def physical_receipt(self) -> dict[str, object]:
        raw_store = getattr(self.store, "store")
        body = {
            "schema_version": "scale_bp_v2_materialized_physical_bank_v1",
            "source_stream_lock_hash": str(getattr(self.source_cache, "lock_hash")),
            "prediction_seal_hash": str(getattr(self.prediction_seal, "seal_hash")),
            "prediction_store_hash": str(getattr(raw_store, "store_hash")),
            "physical_adapter_hash": self.store.adapter_hash,
            "partition_hash": self.partition_hash,
            "physical_cell_count": len(tuple(getattr(raw_store, "cells"))),
            "target_expert_used": False,
            "labels_opened": False,
            "stage90_prediction_reused": False,
        }
        return {**body, "receipt_hash": canonical_hash(body)}


def physical_partition_hash(frame: object) -> str:
    rows = tuple(getattr(frame, "rows"))
    if not rows:
        raise GovernanceError("SCALE-BP v2 physical partition is empty.")
    return canonical_hash(
        {
            "schema_version": "scale_bp_v2_physical_partition_v1",
            "frame_hash": str(getattr(frame, "frame_hash")),
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
    root: str | Path,
    scratch_root: str | Path,
) -> MaterializedPhysicalBank:
    """Generate source streams once, then exactly 810 label-free cells once."""

    artifact = Path(root)
    scratch = Path(scratch_root)
    source_root = artifact / "physical/source_runtime"
    prediction_root = artifact / "physical/prediction_store"
    prediction_scratch = scratch / "prediction_runtime"
    source_cache = materialize_frozen_source_streams(
        config,
        generation_lock,
        root=source_root,
    )
    partition = physical_partition_hash(frame)
    prediction = materialize_fixed_bank_a1_action_predictions(
        config,
        source_cache,
        frame,
        partition_hash=partition,
        action_library=action_library_by_target(),
        root=prediction_root,
        scratch_root=prediction_scratch,
    )
    store = adapt_prediction_store(getattr(prediction, "store"))
    if len(tuple(getattr(store.store, "cells"))) != EXPECTED_PHYSICAL_CELL_COUNT:
        raise GovernanceError("SCALE-BP v2 physical cell coverage drifted.")
    return MaterializedPhysicalBank(source_cache, prediction, store, partition)


__all__ = (
    "MaterializedPhysicalBank",
    "materialize_physical_bank",
    "physical_partition_hash",
)
