from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.capacity_preflight import (
    MINIMUM_ARTIFACT_FREE_BYTES,
    MINIMUM_GPU_FREE_MIB,
    MINIMUM_RAM_AVAILABLE_BYTES,
    MINIMUM_SCRATCH_FREE_BYTES,
    validate_capacity_observation,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _observation(*, shared: bool = False) -> dict[str, object]:
    return {
        "gpus": [
            {
                "index": index,
                "name": "NVIDIA RTX A5000",
                "total_mib": 24_576,
                "free_mib": MINIMUM_GPU_FREE_MIB,
            }
            for index in range(2)
        ],
        "ram_available_bytes": MINIMUM_RAM_AVAILABLE_BYTES,
        "artifact_free_bytes": (
            MINIMUM_ARTIFACT_FREE_BYTES + MINIMUM_SCRATCH_FREE_BYTES
            if shared
            else MINIMUM_ARTIFACT_FREE_BYTES
        ),
        "scratch_free_bytes": MINIMUM_SCRATCH_FREE_BYTES,
        "artifact_device": 7,
        "scratch_device": 7 if shared else 8,
    }


def test_capacity_accepts_exact_minima_on_distinct_or_shared_filesystems() -> None:
    distinct = validate_capacity_observation(_observation())
    shared = validate_capacity_observation(_observation(shared=True))

    assert distinct.to_payload()["shared_artifact_scratch_filesystem"] is False
    assert shared.to_payload()["shared_artifact_scratch_filesystem"] is True
    assert distinct.filesystem_mutation_performed is False


def test_capacity_fails_closed_before_lease_when_any_resource_is_low() -> None:
    low_gpu = _observation()
    low_gpu["gpus"][0]["free_mib"] = MINIMUM_GPU_FREE_MIB - 1
    with pytest.raises(ProtocolError, match="capacity is insufficient"):
        validate_capacity_observation(low_gpu)

    low_ram = _observation()
    low_ram["ram_available_bytes"] = MINIMUM_RAM_AVAILABLE_BYTES - 1
    with pytest.raises(ProtocolError, match="capacity is insufficient"):
        validate_capacity_observation(low_ram)

    shared_overcommitted = _observation(shared=True)
    shared_overcommitted["artifact_free_bytes"] -= 1
    with pytest.raises(ProtocolError, match="capacity is insufficient"):
        validate_capacity_observation(shared_overcommitted)


def test_live_capacity_probe_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.capacity_preflight as capacity_module

    artifact = tmp_path / "output"
    artifact.mkdir()
    scratch = tmp_path / "future-scratch"
    monkeypatch.setattr(
        capacity_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                "0, NVIDIA RTX A5000, 24576, 20000\n"
                "1, NVIDIA RTX A5000, 24576, 20000\n"
            )
        ),
    )
    monkeypatch.setattr(
        capacity_module,
        "_read_mem_available_bytes",
        lambda: MINIMUM_RAM_AVAILABLE_BYTES,
    )
    monkeypatch.setattr(
        capacity_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            free=MINIMUM_ARTIFACT_FREE_BYTES + MINIMUM_SCRATCH_FREE_BYTES
        ),
    )

    receipt = capacity_module.preflight_resource_capacity(artifact, scratch)

    assert receipt.filesystem_mutation_performed is False
    assert not scratch.exists()
