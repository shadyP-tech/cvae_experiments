from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4 import (
    gpu_phase,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4 import (
    scratch,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.worker_dtos import (
    WORKER_DEPTH_ENV,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _runtime() -> dict[str, object]:
    return {
        "generation_devices": ["cuda:0", "cuda:1"],
        "persistent_source_workers": True,
        "persistent_generation_worker_count": 2,
        "source_workers_per_device": 1,
        "generation_workers_per_device": 1,
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "multiprocessing_start_method": "spawn",
        "gpu_generation_phase_precedes_cpu_phase": True,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "nested_process_pools_forbidden": True,
        "scratch_preference": [scratch.CANONICAL_SCRATCH_ROOT, "artifact_parent"],
    }


def test_gpu_then_prediction_phase_uses_one_depth_guard_and_restores_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sentinel = object()
    seen: dict[str, str | None] = {}

    def fake(*_args, **_kwargs):
        seen["depth"] = os.environ.get(WORKER_DEPTH_ENV)
        seen["cuda"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        return sentinel

    monkeypatch.delenv(WORKER_DEPTH_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "prior")
    monkeypatch.setattr(gpu_phase, "materialize_physical_bank", fake)
    result = gpu_phase.materialize_gpu_phase(
        SimpleNamespace(runtime=_runtime()),
        object(),
        object(),
        root=tmp_path,
        prediction_scratch_root=tmp_path / "predictions",
    )
    assert result is sentinel
    assert seen == {"depth": "gpu_then_prediction", "cuda": "0,1"}
    assert os.environ.get(WORKER_DEPTH_ENV) is None
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "prior"


def test_v4_scratch_identity_is_distinct_and_nonrecovering(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    artifact_root = tmp_path / "artifact"

    dedicated = tmp_path / "data-local" / "pdcaps-v4"
    dedicated.parent.mkdir()
    monkeypatch.setattr(scratch, "CANONICAL_SCRATCH_ROOT", str(dedicated))
    runtime = _runtime()
    runtime["scratch_preference"] = [str(dedicated), "artifact_parent"]
    lease = scratch.select_scratch(artifact_root, runtime)
    assert lease.role == "dedicated_local"
    assert lease.root == dedicated

    missing_parent = tmp_path / "missing-data-local" / "pdcaps-v4"
    monkeypatch.setattr(scratch, "CANONICAL_SCRATCH_ROOT", str(missing_parent))
    runtime["scratch_preference"] = [str(missing_parent), "artifact_parent"]
    fallback = scratch.select_scratch(artifact_root, runtime)
    assert fallback.role == "artifact_parent"
    assert fallback.root.name.endswith("pdcaps-v4-scratch")

    bad = dict(runtime)
    bad["scratch_preference"] = [str(missing_parent)[:-1] + "3", "artifact_parent"]
    with pytest.raises(ProtocolError, match="scratch identity"):
        scratch.select_scratch(artifact_root, bad)
