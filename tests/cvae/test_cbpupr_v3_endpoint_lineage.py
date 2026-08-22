from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import multiprocessing as mp

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.candidate_orchestration import (
    build_outer_endpoint_jobs,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.constants import (
    CENTERS,
    DIRECTION_IDS,
    candidate_sources,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.contracts import (
    BinaryLabel,
    CenterProbabilitySurface,
    PhysicalProbabilitySurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.endpoint_preparation import (
    prepare_center,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.outer_endpoint_runtime import (
    OuterEndpointJob,
    _compute_outer_endpoint_payload,
    _job_from_payload,
    _job_payload,
    _products_from_payload,
    _products_payload,
    _validate_products_against_job,
    compute_outer_endpoint_products,
    execute_outer_endpoint_jobs,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.outer_plans import (
    build_outer_plans,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _physical_surface() -> PhysicalProbabilitySurface:
    store_hash = "a" * 64
    centers: dict[str, CenterProbabilitySurface] = {}
    for center_index, center in enumerate(CENTERS):
        sample_ids = tuple(
            f"center-{center}::case-{case_index}::sample-{sample_index}"
            for case_index in range(2)
            for sample_index in range(2)
        )
        case_ids = ("case-0", "case-0", "case-1", "case-1")
        baseline = np.asarray((0.2, 0.8, 0.3, 0.7), dtype=np.float32)
        arrays: dict[str, np.ndarray] = {}
        for action_index, action in enumerate(physical_action_ids(center)):
            direction = np.float32(0.0 if action_index < 2 else 0.35)
            if action_index % 2 == 0:
                direction *= np.float32(-1.0)
            action_values = np.clip(
                baseline
                + direction
                + np.float32(center_index) * np.float32(0.0001),
                0.01,
                0.99,
            )
            arrays[action] = np.stack(
                tuple(
                    np.clip(
                        action_values
                        + np.float32(seed_index - 4) * np.float32(0.001),
                        0.0,
                        1.0,
                    )
                    for seed_index in range(9)
                )
            )
        centers[center] = CenterProbabilitySurface(
            center,
            sample_ids,
            case_ids,
            arrays,
            store_hash,
        )
    return PhysicalProbabilitySurface(
        centers,
        store_hash,
        strict_canonical_topology=False,
    )


def _production_jobs() -> tuple[
    PhysicalProbabilitySurface,
    tuple[OuterEndpointJob, ...],
]:
    surface = _physical_surface()
    identities = tuple(
        SimpleNamespace(
            center=center,
            case_id=case,
            sample_id=sample,
            group_id=case,
        )
        for center in CENTERS
        for sample, case in zip(
            surface.centers[center].sample_ids,
            surface.centers[center].case_ids,
            strict=True,
        )
    )
    plans = build_outer_plans(
        identities,
        probability_surface_hash=surface.surface_hash,
        strict_canonical_topology=False,
    )
    prepared = {
        center: prepare_center(surface.centers[center]) for center in CENTERS
    }
    support_by_route = {
        plan.key: tuple(
            BinaryLabel(
                plan.target_center,
                case,
                surface.centers[plan.target_center].sample_ids[position],
                position % 2,
                (
                    f"outer_support::H={plan.target_center}::"
                    f"excluded_c={plan.case_id}"
                ),
            )
            for case in plan.support_case_ids
            for position in surface.centers[plan.target_center].positions(case)
        )
        for plan in plans.outer_plans
    }
    ordinary_priors = {
        center: {
            (source, direction): 0.0
            for source in candidate_sources(center)
            for direction in DIRECTION_IDS
        }
        for center in CENTERS
    }
    jobs = build_outer_endpoint_jobs(
        surface,
        plan_seal=plans,
        prepared_centers=prepared,
        support_by_route=support_by_route,
        ordinary_priors=ordinary_priors,
    )
    return surface, jobs


def _spawn_compute(payload: dict[str, object]) -> dict[str, object]:
    try:
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            return executor.submit(
                _compute_outer_endpoint_payload, payload
            ).result(timeout=60)
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"OS spawn boundary is unavailable: {exc}")


def test_production_factory_binds_distinct_global_and_center_hashes() -> None:
    surface, jobs = _production_jobs()

    assert tuple(job.target_center for job in jobs) == CENTERS
    all_surface_hashes = {
        surface.surface_hash,
        *(row.surface_hash for row in surface.centers.values()),
    }
    assert len(all_surface_hashes) == 10
    for job in jobs:
        assert job.physical_surface_hash == surface.surface_hash
        assert job.center_surface_hash == surface.centers[job.target_center].surface_hash
        assert job.physical_surface_hash != job.center_surface_hash
        assert all(
            plan.probability_surface_hash == job.physical_surface_hash
            for plan in job.outer_plans
        )


def test_endpoint_serial_and_spawn_round_trip_preserve_both_hash_roles() -> None:
    _surface, jobs = _production_jobs()
    job = jobs[0]

    serialized_job = _job_payload(job)
    restored_job = _job_from_payload(serialized_job)
    assert restored_job.physical_surface_hash == job.physical_surface_hash
    assert restored_job.center_surface_hash == job.center_surface_hash

    serial = compute_outer_endpoint_products(job)
    spawned = _products_from_payload(_spawn_compute(serialized_job))
    _validate_products_against_job(job, serial)
    _validate_products_against_job(job, spawned)

    assert serial.physical_surface_hash == spawned.physical_surface_hash
    assert serial.center_surface_hash == spawned.center_surface_hash
    assert serial.state_hashes == spawned.state_hashes
    assert tuple(row.prediction_hash for row in serial.predictions) == tuple(
        row.prediction_hash for row in spawned.predictions
    )


def test_endpoint_lineage_tampering_fails_before_dispatch_and_after_spawn() -> None:
    _surface, jobs = _production_jobs()
    job = jobs[0]
    job_payload = _job_payload(job)

    with pytest.raises(ProtocolError, match="surface hash roles collapsed"):
        _compute_outer_endpoint_payload(
            {
                **job_payload,
                "physical_surface_hash": job.center_surface_hash,
            }
        )
    with pytest.raises(ProtocolError, match="surface or plan lineage"):
        _compute_outer_endpoint_payload(
            {
                **job_payload,
                "center_surface_hash": "f" * 64,
            }
    )

    poisoned_jobs = list(jobs)
    poisoned = copy.copy(poisoned_jobs[0])
    object.__setattr__(
        poisoned,
        "physical_surface_hash",
        poisoned.center_surface_hash,
    )
    poisoned_jobs[0] = poisoned
    with pytest.raises(ProtocolError, match="surface hash roles collapsed"):
        execute_outer_endpoint_jobs(poisoned_jobs, use_processes=False)

    products = compute_outer_endpoint_products(job)
    product_payload = _products_payload(products)
    tampered_products = _products_from_payload(
        {**product_payload, "physical_surface_hash": "f" * 64}
    )
    with pytest.raises(ProtocolError, match="worker result topology"):
        _validate_products_against_job(job, tampered_products)
