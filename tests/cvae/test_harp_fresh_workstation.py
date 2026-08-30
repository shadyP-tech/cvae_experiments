from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import pytest

from midogpp_thesis.cvae.frozen_policy_downstream.fresh_runtime_contract import (
    canonical_runtime_payload as canonical_peer_runtime_payload,
)
from midogpp_thesis.cvae.frozen_policy_downstream.harp_fresh import (
    production_runner,
)
from midogpp_thesis.cvae.frozen_policy_downstream.harp_fresh.config import (
    HARP_SOURCE_CACHE_FORMAT,
    canonical_harp_runtime_payload,
)
from midogpp_thesis.cvae.frozen_policy_downstream.harp_fresh.workstation import (
    REQUIRED_ENVIRONMENT,
    WorkstationProbes,
    WorkstationSnapshot,
    run_workstation_preflight,
    validate_workstation_snapshot,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.frozen_source_streams import SOURCE_ROWS_PER_CLASS


def _gpu_rows(*, free_mib: int = 22_000, name: str = "NVIDIA RTX A5000"):
    return (
        {
            "index": 0,
            "name": name,
            "memory_total_mib": 24_564,
            "memory_free_mib": free_mib,
        },
        {
            "index": 1,
            "name": name,
            "memory_total_mib": 24_564,
            "memory_free_mib": free_mib,
        },
    )


def _snapshot() -> WorkstationSnapshot:
    return WorkstationSnapshot(
        available_cpu_count=24,
        physical_ram_bytes=125 * 1024**3,
        artifact_disk_free_bytes=20 * 1024**3,
        gpu_rows=_gpu_rows(),
        spawn_available=True,
        parent_cuda_context_initialized=False,
    )


def _probes(*, scratch_writable: bool = True) -> WorkstationProbes:
    return WorkstationProbes(
        available_cpu_count=lambda: 24,
        physical_ram_bytes=lambda: 125 * 1024**3,
        disk_free_bytes=lambda _path: 20 * 1024**3,
        gpu_rows=_gpu_rows,
        spawn_available=lambda: True,
        parent_cuda_context_initialized=lambda: False,
        directory_writable=lambda path: (
            scratch_writable if path == Path("/data/local") else True
        ),
        atomic_replace_supported=lambda path: (
            scratch_writable if path == Path("/data/local") else True
        ),
    )


def test_harp_runner_owns_its_workstation_admission() -> None:
    source = inspect.getsource(production_runner)
    assert "residual_topup_fresh" not in source
    assert production_runner.WorkstationProbes.__module__.endswith(
        "harp_fresh.workstation"
    )


def test_harp_preflight_accepts_270_row_frozen_stream_runtime(tmp_path: Path) -> None:
    runtime = canonical_harp_runtime_payload()
    report = run_workstation_preflight(
        tmp_path,
        runtime=runtime,
        probes=_probes(),
        environment=REQUIRED_ENVIRONMENT,
    )
    assert SOURCE_ROWS_PER_CLASS == 270
    assert report["source_block_per_class"] == SOURCE_ROWS_PER_CLASS
    assert report["source_cache_format"] == HARP_SOURCE_CACHE_FORMAT
    assert report["generation_worker_count"] == 2
    assert report["classifier_workers"] == 4
    assert report["classifier_threads_per_worker"] == 3
    assert report["multiprocessing_start_method"] == "spawn"
    assert report["parent_cuda_context_initialized"] is False
    assert report["gpu_and_cpu_phases_disjoint"] is True


def test_harp_preflight_rejects_peer_source_runtime_semantics(tmp_path: Path) -> None:
    peer = canonical_peer_runtime_payload()
    assert peer["source_block_per_class"] == 256
    with pytest.raises(ProtocolError, match="270 rows per class"):
        run_workstation_preflight(
            tmp_path,
            runtime=peer,
            probes=_probes(),
            environment=REQUIRED_ENVIRONMENT,
        )

    wrong_format = canonical_harp_runtime_payload()
    wrong_format["source_cache_format"] = "float32_npy_memmap"
    with pytest.raises(ProtocolError, match="source-cache format"):
        run_workstation_preflight(
            tmp_path,
            runtime=wrong_format,
            probes=_probes(),
            environment=REQUIRED_ENVIRONMENT,
        )


@pytest.mark.parametrize(
    ("snapshot", "message"),
    [
        (replace(_snapshot(), spawn_available=False), "multiprocessing spawn"),
        (
            replace(_snapshot(), parent_cuda_context_initialized=True),
            "parent CUDA context",
        ),
        (
            replace(_snapshot(), gpu_rows=_gpu_rows(free_mib=17_999)),
            "less than 18 GiB free",
        ),
        (
            replace(_snapshot(), gpu_rows=_gpu_rows(name="NVIDIA RTX 4090")),
            "not an RTX A5000",
        ),
    ],
)
def test_harp_snapshot_preserves_hardware_and_parent_cuda_gates(
    snapshot: WorkstationSnapshot,
    message: str,
) -> None:
    with pytest.raises(ProtocolError, match=message):
        validate_workstation_snapshot(
            snapshot,
            runtime=canonical_harp_runtime_payload(),
        )


def test_harp_scratch_is_explicit_and_never_authoritative(tmp_path: Path) -> None:
    report = run_workstation_preflight(
        tmp_path,
        runtime=canonical_harp_runtime_payload(),
        probes=_probes(scratch_writable=False),
        environment=REQUIRED_ENVIRONMENT,
    )
    assert report["optional_local_scratch_enabled"] is False
    assert report["scratch_authoritative"] is False

    with pytest.raises(ProtocolError, match="not writable"):
        run_workstation_preflight(
            tmp_path,
            runtime=canonical_harp_runtime_payload(),
            probes=_probes(scratch_writable=False),
            environment=REQUIRED_ENVIRONMENT,
            enable_optional_local_scratch=True,
        )

    enabled = run_workstation_preflight(
        tmp_path,
        runtime=canonical_harp_runtime_payload(),
        probes=_probes(),
        environment=REQUIRED_ENVIRONMENT,
        enable_optional_local_scratch=True,
    )
    assert enabled["optional_local_scratch_root"] == "/data/local"
    assert enabled["scratch_authoritative"] is False
    assert enabled["canonical_publication_required"] is True
