"""Independent admission and reconstruction of the pre-label physical seal."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.fixed_bank_a1_action_predictions import load_global_prediction_seal
from ...runtime.fixed_bank_a1_prediction_contracts import (
    stable_digest,
    validate_action_library,
)
from ...runtime.frozen_source_streams import load_frozen_source_streams
from .actions import action_library_by_target, build_action_library
from .artifact_rows import reject_forbidden_persistence
from .artifact_writers import read_rows
from .hashing import canonical_hash
from .reports import seal_payload
from .reports import protocol_manifest_payload
from .runtime_adapter import (
    build_exact_nine_surface,
    physical_partition_hash,
    probability_index_rows,
)


def reconstruct_admission_and_prelabel(
    root: Path, *, config: object, protocol: object
) -> Mapping[str, object]:
    """Replay exact-six admission before opening any persisted science table."""

    from ...runtime.artifact_io import read_json
    from .inputs import (
        assert_input_fence,
        load_label_free_test_frame,
        load_validated_locks,
        validate_active_diagnostic_workspace_binding,
        validate_pre_gpu_firewall,
        validate_workspace_provenance,
    )
    from .workstation_preflight import load_validated_workstation_preflight

    assert_input_fence(config)
    workspace = validate_active_diagnostic_workspace_binding(config)
    provenance = validate_workspace_provenance(root, config)
    locks = load_validated_locks(config)
    frame = load_label_free_test_frame(config)
    firewall = dict(validate_pre_gpu_firewall(config, frame, locks))
    firewall["workspace_binding"] = workspace
    expected_protocol = protocol_manifest_payload(
        config,
        protocol=protocol,
        input_artifact_hashes={
            artifact_id: canonical_hash(provenance[artifact_id])
            for artifact_id in getattr(config, "input_artifact_ids")
        },
        cache_binding_hash=str(frame.cache_binding_hash),
        firewall=firewall,
    )
    if read_json(root / "manifests/protocol_manifest.json") != expected_protocol:
        raise ProtocolError("Dual-endpoint protocol manifest is not reconstructive.")
    _validate_action_products(root)
    preflight = load_validated_workstation_preflight(
        root, runtime=getattr(config, "runtime")
    )
    prelabel = reconstruct_prelabel(
        root,
        config=config,
        frame=frame,
        generation_lock_hash=str(locks.generation.generation_lock_hash),
    )
    return {
        "workspace": workspace,
        "provenance": provenance,
        "locks": locks,
        "frame": frame,
        "pre_gpu_firewall": firewall,
        "preflight": preflight,
        "prelabel": prelabel,
    }


def validate_path_free_json_members(root: Path) -> None:
    """Reject raw-label/path fields after content hashes have been checked."""

    from ...runtime.artifact_io import read_json

    excluded = {"provenance/input_artifacts.json"}
    for path in sorted(root.rglob("*.json")):
        if path.relative_to(root).as_posix() in excluded:
            continue
        reject_forbidden_persistence(read_json(path))


def _validate_action_products(root: Path) -> None:
    from ...runtime.artifact_io import read_json

    actions = tuple(build_action_library())
    _, neutral_hash = validate_action_library(action_library_by_target())
    rows = tuple(row.to_payload() for row in actions)
    if len(actions) != 90 or read_rows(root / "tables/action_library.csv") != rows:
        raise ProtocolError("Dual-endpoint action library table drifted.")
    # The neutral validator is the canonical compatibility check; binding the
    # returned hash here prevents a successor-only DTO from silently diverging.
    if not stable_digest(neutral_hash):
        raise ProtocolError("Dual-endpoint neutral action-library hash drifted.")
    expected = seal_payload(
        "fixed_bank_dual_endpoint_action_library_seal_v1",
        bindings={"actions_hash": canonical_hash(rows)},
        action_count=len(rows),
        actions_per_target=10,
        exact_nine_required=True,
        labels_used=False,
        target_expert_used=False,
    )
    if read_json(root / "manifests/action_library.json") != expected:
        raise ProtocolError("Dual-endpoint action library seal drifted.")


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
    _, library_hash = validate_action_library(action_library_by_target())
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
    if read_rows(root / "tables/exact_nine_probability_index.csv") != expected_index:
        raise ProtocolError("Dual-endpoint exact-nine index is not reconstructive.")
    surface_hash = str(getattr(surface, "surface_hash"))
    expected_seal = seal_payload(
        "fixed_bank_dual_endpoint_physical_prelabel_seal_v1",
        bindings={
            "global_prediction_seal_hash": prediction.seal_hash,
            "prediction_store_hash": prediction.store.store_hash,
            "probability_surface_hash": surface_hash,
            "probability_index_hash": canonical_hash(expected_index),
        },
        physical_cell_count=len(prediction.store.cells),
        target_action_index_count=len(expected_index),
        stored_probability_dtype="float32",
        exact_nine_reduction_dtype="float64",
        labels_used=False,
        sealed_before_any_label_capability=True,
    )
    from ...runtime.artifact_io import read_json

    if read_json(root / "manifests/physical_prelabel_seal.json") != expected_seal:
        raise ProtocolError("Dual-endpoint physical prelabel seal drifted.")
    if len(build_action_library()) != 90 or len(prediction.store.cells) != 810:
        raise ProtocolError("Dual-endpoint physical topology drifted.")
    return {
        "source": source,
        "prediction": prediction,
        "probability_surface": surface,
        "physical_prelabel_seal": expected_seal,
        "physical_prelabel_seal_hash": expected_seal["seal_hash"],
        "probability_surface_hash": surface_hash,
        "probability_index_count": len(expected_index),
    }


__all__ = (
    "reconstruct_admission_and_prelabel",
    "reconstruct_prelabel",
    "validate_path_free_json_members",
)
