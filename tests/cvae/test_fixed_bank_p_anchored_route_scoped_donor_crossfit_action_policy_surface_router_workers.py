from __future__ import annotations

import pickle
from types import MappingProxyType

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.execution import (
    ContiguousArray,
    HASH_MANIFEST_OPERATION,
    WorkerRequest,
    execute_outer_jobs,
    execute_outer_worker,
    validate_plain_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.execution.outer_worker import (
    WORKER_DEPTH_ENV,
)


def _request(center: str, ordinal: int) -> WorkerRequest:
    return WorkerRequest(
        center,
        ordinal,
        HASH_MANIFEST_OPERATION,
        (("case_count", ordinal + 2), ("roles", ("descriptor", "response"))),
        (
            ContiguousArray(
                "features",
                np.arange(12, dtype=np.float64).reshape(3, 4) + ordinal,
            ),
        ),
        1,
    )


def test_worker_contract_has_real_pickle_roundtrip_and_rejects_poison() -> None:
    request = _request("0", 0)
    restored = pickle.loads(pickle.dumps(request))
    assert restored.request_hash == request.request_hash
    assert restored.arrays[0].values.flags.c_contiguous
    assert restored.arrays[0].array_hash == request.arrays[0].array_hash
    result = execute_outer_worker(restored)
    returned = pickle.loads(pickle.dumps(result))
    assert returned.result_hash == result.result_hash
    assert returned.array_hashes == (("features", request.arrays[0].array_hash),)

    with pytest.raises(ProtocolError, match="prohibited mapping"):
        validate_plain_payload(MappingProxyType({"poison": 1}))
    with pytest.raises(ProtocolError, match="prohibited mapping"):
        WorkerRequest(
            "0",
            0,
            HASH_MANIFEST_OPERATION,
            (("poison", MappingProxyType({"nested": 1})),),
            (),
            1,
        )
    with pytest.raises(ProtocolError, match="mutable container"):
        validate_plain_payload(("nested", [1, 2]))

    def closure() -> None:
        return None

    with pytest.raises(ProtocolError, match="closure"):
        validate_plain_payload(("nested", closure))


def test_serial_and_spawn_science_hashes_are_identical() -> None:
    requests = (_request("0", 0), _request("1", 1))
    serial = execute_outer_jobs(requests, use_processes=False, max_workers=2)
    spawned = execute_outer_jobs(requests, use_processes=True, max_workers=2)
    assert serial.execution_mode == "serial"
    assert spawned.execution_mode == "spawn"
    assert serial.science_hash == spawned.science_hash
    assert tuple(row.result_hash for row in serial.results) == tuple(
        row.result_hash for row in spawned.results
    )
    assert serial.runtime_hash != spawned.runtime_hash
    assert all(not row.to_payload()["large_scientific_arrays_returned"] for row in spawned.results)


def test_nested_process_pool_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WORKER_DEPTH_ENV, "1")
    with pytest.raises(ProtocolError, match="nested process pools"):
        execute_outer_jobs((_request("0", 0),), use_processes=True)
