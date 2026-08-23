from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pickle

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.method_runtime import (
    build_outer_method_runtime,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.outer_execution import (
    execute_outer_runtime,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.outer_workers import (
    OuterWorkerContext,
    execute_outer_worker,
    fit_outer_control_pair,
    initialize_outer_worker,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.worker_dtos import (
    OuterRuntimeRequest,
    WORKER_DEPTH_ENV,
    assert_pickle_safe,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from tests.cvae.test_fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_outer_runtime import (
    _surface_and_responses,
)


def test_outer_request_is_plain_pickle_safe() -> None:
    request = OuterRuntimeRequest("0", 0)
    assert_pickle_safe(request, role="test request")
    restored = pickle.loads(pickle.dumps(request))
    assert restored == request
    assert restored.request_hash == request.request_hash


def test_one_spawned_v4_h_job_matches_serial_and_context_is_pickle_safe() -> None:
    surface_set, responses = _surface_and_responses()
    context = OuterWorkerContext(surface_set, responses, 6, True)
    request = OuterRuntimeRequest("0", 0)
    assert_pickle_safe(context, role="test outer worker context")
    serial = fit_outer_control_pair(request, context)

    try:
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_outer_worker,
            initargs=(context,),
        ) as executor:
            spawned = executor.submit(execute_outer_worker, request).result()
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"local sandbox cannot allocate spawned semaphores: {exc}")

    restored = pickle.loads(pickle.dumps(spawned))
    assert spawned.pair_hash == serial.pair_hash == restored.pair_hash
    methods = build_outer_method_runtime(
        surface_set=surface_set,
        identity_result=spawned.identity_result,
        cyclic_result=spawned.cyclic_result,
        center_sample_order=("0-a", "0-b"),
    )
    assert methods.identity_admission.admission_hash == (
        methods.decisions[1].outer_admission_hash
    )
    assert methods.cyclic_admission.admission_hash == (
        methods.decisions[-1].outer_admission_hash
    )


def test_outer_execution_rejects_nested_pool_before_science(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(WORKER_DEPTH_ENV, "gpu_then_prediction")
    with pytest.raises(ProtocolError, match="nested process pools"):
        execute_outer_runtime(None, None, use_processes=True)  # type: ignore[arg-type]
