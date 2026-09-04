from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v15.runner_recovery import (
    persist_or_validate_json,
    stable_preflight_hash,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def test_v15_stable_preflight_hash_ignores_only_volatile_capacity() -> None:
    first = {
        "schema_version": "v15_preflight_test",
        "live_workstation": {
            "scratch_free_bytes": 10,
            "scratch_probe_path": "/scratch/first",
            "gpus": [
                {"index": 0, "name": "GPU-A", "memory_free_mib": 20},
                {"index": 1, "name": "GPU-B", "memory_free_mib": 30},
            ],
            "classifier_workers": 4,
        },
    }
    second = {
        "schema_version": "v15_preflight_test",
        "live_workstation": {
            "scratch_free_bytes": 999,
            "scratch_probe_path": "/scratch/second",
            "gpus": [
                {"index": 0, "name": "GPU-A", "memory_free_mib": 888},
                {"index": 1, "name": "GPU-B", "memory_free_mib": 777},
            ],
            "classifier_workers": 4,
        },
    }
    changed_topology = {
        **second,
        "live_workstation": {
            **second["live_workstation"],
            "classifier_workers": 3,
        },
    }

    assert stable_preflight_hash(first) == stable_preflight_hash(second)
    assert stable_preflight_hash(first) != stable_preflight_hash(changed_topology)


def test_v15_json_persistence_is_idempotent_and_refuses_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifests" / "seal.json"
    payload = {"schema_version": "v15_test", "nested": {"value": 1}}

    persist_or_validate_json(path, payload)
    first_bytes = path.read_bytes()
    persist_or_validate_json(path, payload)

    assert path.read_bytes() == first_bytes
    with pytest.raises(ProtocolError, match="overwrite drifted durable JSON"):
        persist_or_validate_json(
            path,
            {"schema_version": "v15_test", "nested": {"value": 2}},
        )

