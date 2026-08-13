"""Independent reconstruction of the label-free physical probability surface."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from ...runtime.fixed_bank_a1_action_predictions import load_global_prediction_seal
from ...runtime.fixed_bank_a1_prediction_contracts import validate_action_library
from ...runtime.frozen_source_streams import load_frozen_source_streams
from .actions import action_library_by_target, build_action_library
from .constants import ACTION_COUNT_PER_TARGET, CENTERS, TARGET_PROBABILITY_CELL_COUNT
from .execution_adapter import (
    build_exact_nine_surface,
    physical_partition_hash,
    probability_index_rows,
)
from .hashing import canonical_hash
from .persistence import object_payload, read_rows
from .reports import seal_payload


def validate_action_products(root: Path) -> Mapping[str, object]:
    """Rebuild both experiment-owned action products exactly."""

    actions = build_action_library()
    rows = tuple(object_payload(action) for action in actions)
    if (
        len(rows) != len(CENTERS) * ACTION_COUNT_PER_TARGET
        or read_rows(root / "tables/action_library.csv") != rows
    ):
        raise ProtocolError("Case-directional action table is not reconstructive.")
    expected_seal = seal_payload(
        "fixed_bank_cdca_action_library_manifest_v1",
        bindings={"actions_hash": canonical_hash(rows)},
        action_count=len(rows),
        physical_actions_per_target=ACTION_COUNT_PER_TARGET,
        labels_used=False,
        target_expert_used=False,
    )
    if read_json(root / "manifests/action_library.json") != expected_seal:
        raise ProtocolError("Case-directional action manifest is not reconstructive.")
    return {
        "action_count": len(rows),
        "action_library_seal_hash": expected_seal["seal_hash"],
    }


def reconstruct_prelabel(
    root: Path,
    *,
    config: object,
    frame: object,
    generation_lock_hash: str,
) -> Mapping[str, object]:
    """Load the canonical source/prediction seals and rebuild exact-nine means."""

    source = load_frozen_source_streams(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=generation_lock_hash,
    )
    _action_payload, action_library_hash = validate_action_library(
        action_library_by_target()
    )
    prediction = load_global_prediction_seal(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_partition_hash=physical_partition_hash(frame),
        expected_source_lock_hash=source.lock_hash,
        expected_action_library_hash=action_library_hash,
        expected_target_cache_binding_hash=str(getattr(frame, "cache_binding_hash")),
    )
    surface = build_exact_nine_surface(prediction)
    index_rows = tuple(object_payload(row) for row in probability_index_rows(prediction))
    if read_rows(root / "tables/exact_nine_probability_index.csv") != index_rows:
        raise ProtocolError(
            "Case-directional exact-nine probability index is not reconstructive."
        )
    expected_seal = seal_payload(
        "fixed_bank_cdca_physical_prelabel_seal_v1",
        bindings={
            "global_prediction_seal_hash": prediction.seal_hash,
            "prediction_store_hash": prediction.store.store_hash,
            "probability_surface_hash": str(surface.surface_hash),
            "probability_index_hash": canonical_hash(index_rows),
        },
        physical_cell_count=len(prediction.store.cells),
        target_action_index_count=len(index_rows),
        labels_used=False,
        sealed_before_any_label_capability=True,
    )
    if read_json(root / "manifests/physical_prelabel_seal.json") != expected_seal:
        raise ProtocolError(
            "Case-directional physical prelabel seal is not reconstructive."
        )
    if (
        len(source.records) != 81
        or len(prediction.store.cells) != TARGET_PROBABILITY_CELL_COUNT
        or len(index_rows) != len(CENTERS) * ACTION_COUNT_PER_TARGET
    ):
        raise ProtocolError("Case-directional physical probability topology drifted.")
    return {
        "source": source,
        "prediction": prediction,
        "probability_surface": surface,
        "physical_prelabel_seal": expected_seal,
        "physical_prelabel_seal_hash": expected_seal["seal_hash"],
        "probability_surface_hash": str(surface.surface_hash),
        "probability_index_count": len(index_rows),
    }


__all__ = ("reconstruct_prelabel", "validate_action_products")
