"""Independent reconstruction of the label-free physical probability seal."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.fixed_bank_a1_action_predictions import load_global_prediction_seal
from ...runtime.frozen_source_streams import load_frozen_source_streams
from .actions import action_library_by_target, build_action_library
from .execution_adapter import (
    build_exact_nine_surface,
    physical_partition_hash,
    probability_index_rows,
)
from .hashing import canonical_hash
from .persistence import read_rows
from .reports import seal_payload


def reconstruct_prelabel(
    root: Path,
    *,
    config: object,
    frame: object,
    generation_lock_hash: str,
) -> Mapping[str, object]:
    source = load_frozen_source_streams(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=generation_lock_hash,
    )
    # Validate the persisted action manifest before allowing it to bind the
    # neutral prediction loader.
    from ...runtime.fixed_bank_a1_prediction_contracts import validate_action_library

    _payload, library_hash = validate_action_library(action_library_by_target())
    prediction = load_global_prediction_seal(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_partition_hash=physical_partition_hash(frame),
        expected_source_lock_hash=source.lock_hash,
        expected_action_library_hash=library_hash,
        expected_target_cache_binding_hash=str(getattr(frame, "cache_binding_hash")),
    )
    surface = build_exact_nine_surface(prediction)
    expected_index = tuple(row.to_payload() for row in probability_index_rows(prediction))
    observed_index = read_rows(root / "tables/exact_nine_probability_index.csv")
    if observed_index != expected_index:
        raise ProtocolError("Directional-shrinkage exact-nine index is not reconstructive.")
    surface_hash = str(getattr(surface, "surface_hash"))
    expected_seal = seal_payload(
        "fixed_bank_dcse_physical_prelabel_seal_v1",
        bindings={
            "global_prediction_seal_hash": prediction.seal_hash,
            "prediction_store_hash": prediction.store.store_hash,
            "probability_surface_hash": surface_hash,
            "probability_index_hash": canonical_hash(expected_index),
        },
        physical_cell_count=len(prediction.store.cells),
        target_action_index_count=len(expected_index),
        exact_nine_reduction_dtype="float64",
        stored_probability_dtype="float32",
        labels_used=False,
        sealed_before_label_capabilities=True,
    )
    from ...runtime.artifact_io import read_json

    if read_json(root / "manifests/physical_prelabel_seal.json") != expected_seal:
        raise ProtocolError("Directional-shrinkage physical prelabel seal drifted.")
    if len(build_action_library()) != 90 or len(prediction.store.cells) != 810:
        raise ProtocolError("Directional-shrinkage physical topology drifted.")
    return {
        "source": source,
        "prediction": prediction,
        "probability_surface": surface,
        "physical_prelabel_seal": expected_seal,
        "physical_prelabel_seal_hash": expected_seal["seal_hash"],
        "probability_surface_hash": surface_hash,
        "probability_index_count": len(expected_index),
    }


__all__ = ("reconstruct_prelabel",)
