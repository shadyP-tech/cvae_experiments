"""Independent reconstruction of the label-free physical origin chain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from ...runtime.fixed_bank_a1_prediction_contracts import validate_action_library
from ...runtime.fixed_bank_a1_prediction_store import load_global_prediction_seal
from ...runtime.frozen_source_streams import load_frozen_source_streams
from .actions import action_library_by_target
from .config_payloads import (
    canonical_action_library_payload,
    canonical_policy_menu_payload,
)
from .contracts import PhysicalProbabilitySurface
from .experiment_contracts import EXPECTED_GENERATION_LOCK_HASH
from .hashing import canonical_hash
from .inputs import (
    load_label_free_test_frame,
    load_validated_locks,
    validate_pre_gpu_firewall,
)
from .physical_fingerprint import (
    blocked_within_case_fingerprint,
    build_physical_fingerprint_surface,
)
from .physical_runtime import physical_partition_hash, probability_index_rows
from .posterior_contracts import PhysicalFingerprintSurface
from .probability_surface import build_physical_probability_surface
from .reports import protocol_manifest_payload
from .protocol import FROZEN_PROTOCOL_HASH
from .workspace_inputs import (
    validate_active_workspace_binding,
    validate_workspace_provenance,
)


@dataclass(frozen=True)
class PhysicalOriginTopology:
    surface: PhysicalProbabilitySurface
    fingerprints: Mapping[tuple[str, str], PhysicalFingerprintSurface]
    source_stream_lock_hash: str
    global_prediction_seal_hash: str
    frame: object


def validate_physical_origin(
    root: Path,
    *,
    config: object,
    protocol: Mapping[str, object],
    physical: Mapping[str, object],
    fingerprint_rows: Sequence[Mapping[str, object]],
) -> PhysicalOriginTopology:
    """Replay the six-input lineage and rebuild every label-free fingerprint."""

    validate_active_workspace_binding(config)
    provenance = validate_workspace_provenance(root, config)
    locks = load_validated_locks(config)
    frame = load_label_free_test_frame(config)
    pre_gpu = validate_pre_gpu_firewall(config, frame, locks)

    if read_json(root / "manifests/action_library.json") != (
        canonical_action_library_payload()
    ) or read_json(root / "manifests/policy_menu.json") != (
        canonical_policy_menu_payload()
    ):
        raise ProtocolError("CBPUPR canonical action/policy contract drifted.")
    expected_protocol = protocol_manifest_payload(
        config,
        protocol_hash=FROZEN_PROTOCOL_HASH,
        provenance=provenance,
        cache_binding_hash=canonical_hash(dict(frame.cache_binding)),
        pre_gpu_firewall=pre_gpu,
    )
    if dict(protocol) != expected_protocol:
        raise ProtocolError("CBPUPR protocol manifest origin drifted.")

    source_cache = load_frozen_source_streams(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
    )
    _action_payload, action_hash = validate_action_library(
        action_library_by_target()
    )
    prediction = load_global_prediction_seal(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_partition_hash=physical_partition_hash(frame),
        expected_source_lock_hash=source_cache.lock_hash,
        expected_action_library_hash=action_hash,
        expected_target_cache_binding_hash=frame.cache_binding_hash,
    )
    surface = build_physical_probability_surface(prediction.store)
    probability_rows = [
        row.to_payload() for row in probability_index_rows(prediction)
    ]
    if read_json(root / "tables/exact_nine_probability_index.json") != {
        "schema_version": "fixed_bank_cbpupr_exact_nine_probability_index_v1",
        "row_count": len(probability_rows),
        "rows": probability_rows,
    }:
        raise ProtocolError("CBPUPR exact-nine probability origin drifted.")
    physical_unhashed = {
        "schema_version": "fixed_bank_cbpupr_physical_surface_seal_v1",
        "surface_hash": surface.surface_hash,
        "probability_store_hash": surface.probability_store_hash,
        "source_stream_lock_hash": source_cache.lock_hash,
        "global_prediction_seal_hash": prediction.seal_hash,
        "probability_index_hash": canonical_hash(probability_rows),
        "target_probability_cell_count": 810,
        "labels_used": False,
    }
    expected_physical = {
        **physical_unhashed,
        "physical_surface_seal_hash": canonical_hash(physical_unhashed),
    }
    if dict(physical) != expected_physical:
        raise ProtocolError("CBPUPR physical surface seal origin drifted.")

    primary = {
        center: build_physical_fingerprint_surface(surface.centers[center])
        for center in surface.centers
    }
    blocked = {
        center: blocked_within_case_fingerprint(primary[center])
        for center in surface.centers
    }
    expected_fingerprints = [
        row.summary_payload()
        for row in (*primary.values(), *blocked.values())
    ]
    if list(fingerprint_rows) != expected_fingerprints:
        raise ProtocolError("CBPUPR physical fingerprint reconstruction drifted.")
    indexed = {
        (row.center, row.control_id): row
        for row in (*primary.values(), *blocked.values())
    }
    return PhysicalOriginTopology(
        surface,
        MappingProxyType(indexed),
        source_cache.lock_hash,
        prediction.seal_hash,
        frame,
    )


__all__ = ("PhysicalOriginTopology", "validate_physical_origin")
