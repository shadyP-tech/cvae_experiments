from __future__ import annotations

from midogpp_thesis.cvae.runtime.harp_v5_execution import (
    classifier_tasks,
    execution_profile,
    gpu_surface,
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
