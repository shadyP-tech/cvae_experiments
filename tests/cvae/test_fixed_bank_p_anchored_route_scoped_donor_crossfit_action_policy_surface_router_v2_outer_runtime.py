from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import multiprocessing
import pickle

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.action_surface import (
    ActionDraft,
    ActionResponse,
    RouteActionDraftSurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.contracts import (
    BankViability,
    FavorableUtility,
    RouteKey,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.identity import (
    ACTION_STRATA,
    METHOD_MENU,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.inventory import (
    ExpectedRouteInventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.surface_set import (
    seal_action_surface_set,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.method_runtime import (
    build_outer_method_runtime,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.outer_runtime import (
    WORKER_DEPTH_ENV,
    WORKER_DTO_KIND,
    OuterRuntimeRequest,
    _OuterWorkerContext,
    _execute_outer_worker,
    _fit_outer_control_pair,
    _initialize_outer_worker,
    execute_outer_runtime,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.validation_records import (
    validate_persisted_preterminal_records,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _hash(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _viability(value: object) -> BankViability:
    return BankViability(
        True,
        True,
        (("0", 30.0), ("1", 30.0)),
        5.0,
        _hash((value, "viability")),
    )


def _route(outer: str, center: str) -> RouteKey:
    target = outer == center
    return RouteKey(
        "target" if target else "pseudo",
        outer,
        center,
        f"case-{center}",
        outer,
        None if target else center,
        _hash((outer, center, "fit")),
    )


def _draft(outer: str, center: str, control: str) -> RouteActionDraftSurface:
    route = _route(outer, center)
    baseline = np.asarray([0.2, 0.8], dtype=np.float32)
    drafts = tuple(
        ActionDraft(
            route,
            family,
            direction,
            f"{family}::{direction}",
            (
                np.asarray([0.7, 0.8], dtype=np.float32)
                if direction == "zero_to_one"
                else np.asarray([0.2, 0.3], dtype=np.float32)
            ),
            FavorableUtility(
                0.01 * (CENTERS.index(center) + 1) + 0.001 * (index + 1),
                0.004,
                0.003,
            ),
            0.5,
            _viability((center, family, direction)),
            _hash((center, "endpoint")),
            _hash((center, control, "posterior")),
        )
        for index, (family, direction) in enumerate(ACTION_STRATA)
    )
    return RouteActionDraftSurface(
        route,
        (f"{center}-a", f"{center}-b"),
        baseline,
        drafts,
        _hash((center, "endpoint")),
        _hash((center, control, "posterior")),
        _hash("physical"),
        control,
    )


def _surface_and_responses():
    inventory = ExpectedRouteInventory.focused_fixture(
        tuple(
            (center, f"case-{center}", sample_id)
            for center in CENTERS
            for sample_id in (f"{center}-a", f"{center}-b")
        )
    )
    identity = tuple(
        _draft(outer, center, "IDENTITY")
        for outer in CENTERS
        for center in CENTERS
    )
    cyclic = tuple(
        _draft(outer, center, "WITHIN_CASE_CYCLIC_SHIFT")
        for outer in CENTERS
        for center in CENTERS
    )
    surface_set = seal_action_surface_set(
        identity,
        expected_inventory=inventory,
        cyclic_routes=cyclic,
    )
    by_control = []
    for control, surface in (
        ("IDENTITY", surface_set.identity),
        ("WITHIN_CASE_CYCLIC_SHIFT", surface_set.cyclic),
    ):
        responses = []
        for prediction in surface.predictions:
            route = prediction.key.route_key
            if route.outer_center != "0" or route.surface_role != "pseudo":
                continue
            center_index = CENTERS.index(route.route_center)
            stratum_index = ACTION_STRATA.index(prediction.key.stratum)
            level = 0.012 * (center_index + 1) + 0.001 * (stratum_index + 1)
            responses.append(
                ActionResponse(
                    prediction.key,
                    prediction.prediction_hash,
                    FavorableUtility(level, level / 2.0, level / 3.0),
                    2,
                    10,
                    10,
                    20,
                    _hash("P"),
                    _hash((control, route.route_center, "rows")),
                )
            )
        by_control.append((control, tuple(responses)))
    return surface_set, tuple(by_control)


@pytest.fixture(scope="module")
def worker_science():
    surface_set, responses = _surface_and_responses()
    context = _OuterWorkerContext(surface_set, responses, 6, True)
    request = OuterRuntimeRequest("0", 0)
    serial = _fit_outer_control_pair(request, context)
    return surface_set, context, request, serial


def test_one_spawned_h_job_fits_both_controls_with_pickle_safe_dtos(
    worker_science,
) -> None:
    surface_set, context, request, serial = worker_science
    try:
        executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_outer_worker,
            initargs=(context,),
        )
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"local sandbox cannot allocate spawned semaphores: {exc}")
    with executor:
        spawned = executor.submit(_execute_outer_worker, request).result()
    restored = pickle.loads(pickle.dumps(spawned))
    assert spawned.pair_hash == serial.pair_hash == restored.pair_hash
    assert spawned.identity_result.posterior_control_id == "IDENTITY"
    assert spawned.cyclic_result.posterior_control_id == (
        "WITHIN_CASE_CYCLIC_SHIFT"
    )
    assert spawned.to_payload()["worker_dto_kind"] == WORKER_DTO_KIND

    methods = build_outer_method_runtime(
        surface_set=surface_set,
        identity_result=spawned.identity_result,
        cyclic_result=spawned.cyclic_result,
        center_sample_order=("0-a", "0-b"),
    )
    assert tuple(row.method_id for row in methods.decisions) == METHOD_MENU
    assert methods.preterminal_hashes.centers == ("0",)
    assert len(methods.runtime_hash) == 64


def test_outer_runtime_rejects_nested_process_pool_before_science(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(WORKER_DEPTH_ENV, "1")
    with pytest.raises(ProtocolError, match="nested process pools"):
        execute_outer_runtime(None, None, use_processes=True)  # type: ignore[arg-type]


def _persisted_science_fixture(worker_science):
    surface_set, _context, _request_row, pair = worker_science
    methods = build_outer_method_runtime(
        surface_set=surface_set,
        identity_result=pair.identity_result,
        cyclic_result=pair.cyclic_result,
        center_sample_order=("0-a", "0-b"),
    )
    return (
        {
            "surface_set": surface_set.to_payload(),
            "identity_results": [pair.identity_result.to_payload()],
            "cyclic_results": [pair.cyclic_result.to_payload()],
            "identity_legacy_controls": [
                methods.identity_legacy_control.to_payload()
            ],
            "cyclic_legacy_controls": [
                methods.cyclic_legacy_control.to_payload()
            ],
            "method_decisions": [row.to_payload() for row in methods.decisions],
            "method_compositions": [
                row.to_payload() for row in methods.compositions
            ],
        },
        methods,
    )


def _replace_nested(payload: object, path: tuple[object, ...], value: object) -> None:
    target = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]


def test_persisted_science_records_are_reconstructed_and_nested_poison_fails(
    worker_science,
) -> None:
    science, methods = _persisted_science_fixture(worker_science)

    checks = validate_persisted_preterminal_records(
        science, methods.preterminal_hashes
    )
    assert checks["semantic_record_reconstruction_without_refit"] is True
    assert checks["method_decision_count"] == len(METHOD_MENU)
    assert checks["target_decision_counts_by_control"] == {
        "IDENTITY": 1,
        "WITHIN_CASE_CYCLIC_SHIFT": 1,
    }
    assert checks["pseudo_decision_counts_by_control"] == {
        "IDENTITY": len(CENTERS) - 1,
        "WITHIN_CASE_CYCLIC_SHIFT": len(CENTERS) - 1,
    }

    poisoned = deepcopy(science)
    poisoned["identity_results"][0]["target_action_decisions"][0][
        "route_key"
    ]["excluded_outer_center"] = "1"
    with pytest.raises(ProtocolError, match="H/J/d decision drifted"):
        validate_persisted_preterminal_records(
            poisoned, methods.preterminal_hashes
        )

    reordered = deepcopy(science)
    pseudo_rows = reordered["identity_results"][0]["pseudo_action_decisions"]
    pseudo_rows[0], pseudo_rows[1] = pseudo_rows[1], pseudo_rows[0]
    with pytest.raises(ProtocolError):
        validate_persisted_preterminal_records(
            reordered, methods.preterminal_hashes
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (
            (
                "identity_results",
                0,
                "calibration_families",
                "target_models",
                0,
                "coefficients",
                0,
            ),
            999.0,
        ),
        (
            (
                "identity_results",
                0,
                "target_reliabilities",
                0,
                "center_metric_means",
                0,
                1,
            ),
            999.0,
        ),
        (
            (
                "identity_results",
                0,
                "target_action_decisions",
                0,
                "selection",
                "selected_utility",
                "bacc_gain",
            ),
            999.0,
        ),
        (
            (
                "identity_results",
                0,
                "pseudo_policy_response_surfaces",
                0,
                "cells",
                0,
                "normalized_depth",
            ),
            0.5,
        ),
        (
            (
                "identity_results",
                0,
                "target_policy_selection",
                "calibrated_cells",
                0,
                "corrected_utility",
                "bacc_gain",
            ),
            999.0,
        ),
        (
            (
                "identity_results",
                0,
                "nested_policy_calibration",
                "final_calibration",
                "models",
                0,
                "coefficients",
                0,
            ),
            999.0,
        ),
        (
            (
                "identity_results",
                0,
                "policy_calibration_families",
                "pseudo_calibrations_by_center",
                0,
                1,
                "models",
                0,
                "coefficients",
                0,
            ),
            999.0,
        ),
    ),
)
def test_persisted_outer_nested_dto_poison_is_rejected(
    worker_science,
    path: tuple[object, ...],
    replacement: object,
) -> None:
    science, methods = _persisted_science_fixture(worker_science)
    poisoned = deepcopy(science)
    _replace_nested(poisoned, path, replacement)

    with pytest.raises(ProtocolError):
        validate_persisted_preterminal_records(
            poisoned, methods.preterminal_hashes
        )
