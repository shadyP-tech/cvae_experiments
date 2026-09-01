from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import read_json
from midogpp_thesis.cvae.runtime.harp_v6_execution import (
    classifier_tasks,
    execution_profile,
    gpu_surface,
    json_payloads,
    physical,
    physical_contracts,
    resident_stream_contracts,
    resident_stream_store,
)


def test_physical_runtime_delegates_spawn_safe_worker_and_contracts() -> None:
    assert physical._classifier_task is classifier_tasks.execute_classifier_task
    assert (
        physical._load_task_checkpoint
        is classifier_tasks.load_classifier_task_checkpoint
    )
    assert physical._WorkstationProfile is execution_profile.WorkstationProfile
    assert (
        physical._DEFAULT_WORKSTATION_PROFILE
        is execution_profile.DEFAULT_WORKSTATION_PROFILE
    )
    assert physical.PhysicalInputReceipt is physical_contracts.PhysicalInputReceipt
    assert physical._SourceAdapter is physical_contracts.SourceAdapter
    assert physical._Frames is physical_contracts.StagedFrames


def test_gpu_surface_reexports_schema_and_store_without_api_drift() -> None:
    assert (
        gpu_surface.ResidentExpertStreamCache
        is resident_stream_contracts.ResidentExpertStreamCache
    )
    assert (
        gpu_surface.ResidentExpertStreamRecord
        is resident_stream_contracts.ResidentExpertStreamRecord
    )
    assert (
        gpu_surface.load_resident_expert_streams
        is resident_stream_store.load_resident_expert_streams
    )
    assert (
        gpu_surface.stage_resident_expert_streams
        is resident_stream_store.stage_resident_expert_streams
    )
    assert (
        gpu_surface.source_block_sha256
        is resident_stream_contracts.source_block_sha256
    )


def test_nested_immutable_compatibility_payload_round_trips_as_plain_json(
    tmp_path: Path,
) -> None:
    support = MappingProxyType(
        {
            "schema_version": "support-v6",
            "contexts": (
                MappingProxyType(
                    {
                        "center": "0",
                        "case_ids": ("case-a", "case-b"),
                    }
                ),
            ),
            "labels_present": False,
        }
    )
    payload = {
        "schema_version": "compatibility-v6",
        "support_binding": support,
        "replicas": [MappingProxyType({"source_center": "1", "seed": 17})],
    }
    normalized = json_payloads.plain_json_mapping(payload)
    path = tmp_path / "support_compatibility.json"

    gpu_surface._persist_or_validate_json(path, payload)
    first_bytes = path.read_bytes()
    gpu_surface._persist_or_validate_json(path, payload)

    assert read_json(path) == normalized
    assert path.read_bytes() == first_bytes
    assert canonical_hash(payload) == canonical_hash(normalized)
    assert isinstance(normalized["support_binding"], dict)
    assert isinstance(normalized["support_binding"]["contexts"], list)


@pytest.mark.parametrize(
    "payload",
    (
        MappingProxyType({"": 1}),
        MappingProxyType({"value": np.float32(1.0)}),
        MappingProxyType({"value": object()}),
    ),
)
def test_json_boundary_rejects_noncanonical_or_opaque_values(payload: object) -> None:
    with pytest.raises(ProtocolError, match="JSON payload"):
        json_payloads.plain_json_mapping(payload)  # type: ignore[arg-type]
