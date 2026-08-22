from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.config import (
    load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.constants import (
    CENTERS,
    DIRECTION_IDS,
    ENDPOINT_METHOD_IDS,
    FINGERPRINT_STATISTIC_IDS,
    candidate_sources,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.contracts import (
    BinaryLabel,
    CenterProbabilitySurface,
    EndpointCasePrediction,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.endpoint_fitting import (
    EndpointState,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.endpoint_preparation import (
    CenterCaseOutcomes,
    prepare_center,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.outer_endpoint_runtime import (
    OuterEndpointJob,
    OuterEndpointProducts,
    _compute_outer_endpoint_payload,
    _job_payload as _endpoint_job_payload,
    _products_from_payload as _endpoint_products_from_payload,
    _validate_products_against_job as _validate_endpoint_products_against_job,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.outer_plans import (
    WholeCaseOuterPlan,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.physical_fingerprint import (
    blocked_within_case_fingerprint,
    build_physical_fingerprint_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.posterior_fit import (
    predict_route_posterior,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.posterior_contracts import (
    CasePosteriorPrediction,
    TargetLocalPosteriorModel,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.pseudo_endpoint_evidence import (
    PseudoEndpointEvidence,
    PseudoSourcePriorEvidence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.probability_surface import (
    _canonicalize_center_columns,
    build_physical_probability_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.target_local_runtime import (
    TargetCenterPosteriorJob,
    TargetCenterPosteriorProducts,
    _compute_worker_payload as _compute_posterior_payload,
    _job_payload as _posterior_job_payload,
    _products_from_payload as _posterior_products_from_payload,
    _validate_products_against_job as _validate_posterior_products_against_job,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router import runner
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.hashing import canonical_hash
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_array


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_center_"
    "balanced_posterior_utility_prefix_router_v2.yaml"
)


def _spawn_call(function: object, payload: object) -> object:
    try:
        with ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context("spawn")
        ) as executor:
            return executor.submit(function, payload).result(timeout=60)
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"OS spawn boundary is unavailable: {exc}")


def _surface() -> CenterProbabilitySurface:
    samples = tuple(f"sample-{index}" for index in range(6))
    cases = ("case-0", "case-0", "case-1", "case-1", "case-2", "case-2")
    base = np.asarray([0.2, 0.8, 0.3, 0.7, 0.4, 0.6], dtype=np.float32)
    arrays = {}
    for action_index, action in enumerate(physical_action_ids("0")):
        values = base if action_index == 0 else np.clip(
            base + np.float32(0.45 if action_index % 2 else -0.45), 0.01, 0.99
        )
        arrays[action] = np.stack(
            [np.clip(values + np.float32(seed - 4) * 0.001, 0.0, 1.0) for seed in range(9)]
        )
    return CenterProbabilitySurface("0", samples, cases, arrays, "a" * 64)


_RAW_SAMPLE_SUFFIXES = (
    "sample-z",
    "sample-b",
    "sample-y",
    "sample-a",
    "sample-x",
    "sample-c",
)
_RAW_CASE_IDS = ("case-1", "case-0", "case-1", "case-0", "case-2", "case-2")
_CANONICAL_ORDER = (3, 1, 2, 0, 5, 4)


class _FakePredictionCell:
    def __init__(self, target: str, action: str, probabilities: np.ndarray) -> None:
        self.target_center = target
        self.action_id = action
        self.probabilities = probabilities

    @property
    def label(self) -> object:
        raise AssertionError("label access is forbidden while building the surface")


class _FakePredictionStore:
    def __init__(
        self,
        *,
        label_fixture: tuple[int, ...],
        poison_action: str | None = None,
        row_order: tuple[int, ...] = tuple(range(6)),
    ) -> None:
        if tuple(sorted(row_order)) != tuple(range(6)):
            raise ValueError("row_order must be an exact permutation")
        rows: dict[str, tuple[str, ...]] = {}
        cases: dict[str, tuple[str, ...]] = {}
        cells: list[_FakePredictionCell] = []
        for center_index, target in enumerate(CENTERS):
            rows[target] = tuple(
                f"{target}::{_RAW_SAMPLE_SUFFIXES[index]}" for index in row_order
            )
            cases[target] = tuple(_RAW_CASE_IDS[index] for index in row_order)
            for action_index, action in enumerate(physical_action_ids(target)):
                for seed_index in range(9):
                    probabilities = _sentinel_probabilities(
                        center_index, action_index, seed_index
                    )
                    if target == "0" and action == poison_action:
                        probabilities = probabilities[[1, 0, 2, 3, 4, 5]]
                    probabilities = probabilities[list(row_order)]
                    cells.append(_FakePredictionCell(target, action, probabilities))
        self.cells = tuple(cells)
        self.rows_by_center = rows
        self.case_ids_by_center = cases
        self.store_hash = "d" * 64
        self._label_fixture = label_fixture

    @property
    def labels(self) -> object:
        raise AssertionError("label access is forbidden while building the surface")


def _sentinel_probabilities(
    center_index: int, action_index: int, seed_index: int
) -> np.ndarray:
    offset = ((center_index * 10 + action_index) * 9 + seed_index) * 6
    return np.asarray(
        [(offset + column + 1) / 5_000.0 for column in range(6)],
        dtype=np.float32,
    )


def test_public_physical_surface_builder_is_label_free_and_canonical() -> None:
    zeros = _FakePredictionStore(label_fixture=(0,) * 54)
    ones = _FakePredictionStore(label_fixture=(1,) * 54)
    with pytest.raises(AssertionError, match="label access is forbidden"):
        _ = zeros.labels
    with pytest.raises(AssertionError, match="label access is forbidden"):
        _ = zeros.cells[0].label

    first = build_physical_probability_surface(
        zeros, strict_canonical_topology=False
    )
    replay = build_physical_probability_surface(
        zeros, strict_canonical_topology=False
    )
    changed_labels = build_physical_probability_surface(
        ones, strict_canonical_topology=False
    )
    source_reordered = build_physical_probability_surface(
        _FakePredictionStore(
            label_fixture=(0,) * 54,
            row_order=(2, 5, 0, 3, 1, 4),
        ),
        strict_canonical_topology=False,
    )
    assert (
        first.surface_hash
        == replay.surface_hash
        == changed_labels.surface_hash
        == source_reordered.surface_hash
    )

    for center_index, target in enumerate(CENTERS):
        center = first.centers[target]
        assert center.sample_ids == tuple(
            f"{target}::{_RAW_SAMPLE_SUFFIXES[index]}"
            for index in _CANONICAL_ORDER
        )
        assert center.case_ids == tuple(
            _RAW_CASE_IDS[index] for index in _CANONICAL_ORDER
        )
        for action_index, action in enumerate(physical_action_ids(target)):
            raw = np.stack(
                [
                    _sentinel_probabilities(center_index, action_index, seed_index)
                    for seed_index in range(9)
                ]
            )
            expected = np.ascontiguousarray(raw[:, _CANONICAL_ORDER])
            assert center.seed_probabilities[action].tobytes(order="C") == (
                expected.tobytes(order="C")
            )
            assert center.seed_probabilities[action].tobytes(order="C") == (
                changed_labels.centers[target]
                .seed_probabilities[action]
                .tobytes(order="C")
            )
        first_fingerprint = build_physical_fingerprint_surface(center)
        replay_fingerprint = build_physical_fingerprint_surface(
            replay.centers[target]
        )
        reordered_center = source_reordered.centers[target]
        assert reordered_center.sample_ids == center.sample_ids
        assert reordered_center.case_ids == center.case_ids
        assert all(
            reordered_center.seed_probabilities[action].tobytes(order="C")
            == center.seed_probabilities[action].tobytes(order="C")
            for action in physical_action_ids(target)
        )
        assert center.surface_hash == replay.centers[target].surface_hash
        assert center.surface_hash == reordered_center.surface_hash
        assert (
            first_fingerprint.fingerprint_hash
            == replay_fingerprint.fingerprint_hash
            == build_physical_fingerprint_surface(
                reordered_center
            ).fingerprint_hash
        )

    poisoned_action = physical_action_ids("0")[0]
    poisoned = build_physical_probability_surface(
        _FakePredictionStore(
            label_fixture=(0,) * 54,
            poison_action=poisoned_action,
        ),
        strict_canonical_topology=False,
    )
    clean_center = first.centers["0"]
    poisoned_center = poisoned.centers["0"]
    assert sha256_array(clean_center.seed_probabilities[poisoned_action]) != (
        sha256_array(poisoned_center.seed_probabilities[poisoned_action])
    )
    assert all(
        sha256_array(clean_center.seed_probabilities[action])
        == sha256_array(poisoned_center.seed_probabilities[action])
        for action in physical_action_ids("0")[1:]
    )
    assert poisoned_center.surface_hash != clean_center.surface_hash
    assert poisoned.surface_hash != first.surface_hash
    assert (
        build_physical_fingerprint_surface(poisoned_center).fingerprint_hash
        != build_physical_fingerprint_surface(clean_center).fingerprint_hash
    )


def test_physical_surface_canonicalizes_one_shared_label_free_permutation() -> None:
    base = _surface()
    samples = (
        "sample-z",
        "sample-b",
        "sample-y",
        "sample-a",
        "sample-x",
        "sample-c",
    )
    cases = ("case-1", "case-0", "case-1", "case-0", "case-2", "case-2")
    raw_arrays = {
        action: np.asarray(values, dtype=np.float32).copy()
        for action, values in base.seed_probabilities.items()
    }
    expected_order = (3, 1, 2, 0, 5, 4)

    canonical_samples, canonical_cases, canonical_arrays = (
        _canonicalize_center_columns(samples, cases, raw_arrays)
    )

    assert canonical_samples == (
        "sample-a",
        "sample-b",
        "sample-y",
        "sample-z",
        "sample-c",
        "sample-x",
    )
    assert canonical_cases == (
        "case-0",
        "case-0",
        "case-1",
        "case-1",
        "case-2",
        "case-2",
    )
    for action, values in raw_arrays.items():
        expected = np.ascontiguousarray(values[:, expected_order], dtype=np.float32)
        assert canonical_arrays[action].tobytes(order="C") == expected.tobytes(
            order="C"
        )

    surface = CenterProbabilitySurface(
        "0", canonical_samples, canonical_cases, canonical_arrays, "a" * 64
    )
    fingerprint = build_physical_fingerprint_surface(surface)
    plan = WholeCaseOuterPlan(
        "0",
        "case-0",
        "case-0",
        ("case-1", "case-2"),
        ("sample-b", "sample-a"),
        surface.surface_hash,
    )
    assert tuple(
        fingerprint.sample_ids[index]
        for index in fingerprint.positions("case-0")
    ) == plan.evaluation_sample_ids

    replay_samples, replay_cases, replay_arrays = _canonicalize_center_columns(
        canonical_samples, canonical_cases, canonical_arrays
    )
    assert replay_samples == canonical_samples
    assert replay_cases == canonical_cases
    assert all(
        replay_arrays[action].tobytes(order="C")
        == canonical_arrays[action].tobytes(order="C")
        for action in canonical_arrays
    )

    with pytest.raises(ProtocolError, match="identity, row, or action order"):
        CenterProbabilitySurface("0", samples, cases, raw_arrays, "a" * 64)

    bad_width = dict(raw_arrays)
    first_action = next(iter(bad_width))
    bad_width[first_action] = bad_width[first_action][:, :-1]
    with pytest.raises(ProtocolError, match="column topology"):
        _canonicalize_center_columns(samples, cases, bad_width)
    with pytest.raises(ProtocolError, match="identities cannot be canonicalized"):
        _canonicalize_center_columns(
            (*samples[:-1], samples[0]), cases, raw_arrays
        )
    with pytest.raises(ProtocolError, match="identities cannot be canonicalized"):
        _canonicalize_center_columns(samples[:-1], cases, raw_arrays)

    poisoned = dict(raw_arrays)
    poisoned[first_action] = poisoned[first_action][:, (1, 0, 2, 3, 4, 5)]
    _, _, poisoned_arrays = _canonicalize_center_columns(samples, cases, poisoned)
    poisoned_surface = CenterProbabilitySurface(
        "0", canonical_samples, canonical_cases, poisoned_arrays, "a" * 64
    )
    assert poisoned_surface.surface_hash != surface.surface_hash


def test_config_and_catalog_inventory_are_exact() -> None:
    config = load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config(
        CONFIG
    )
    assert config.runtime["expected_total_posterior_model_fit_count"] == 436
    catalog = yaml.safe_load((ROOT / "experiments/midogpp/artifact_catalog.yaml").read_text())
    row = next(
        value
        for value in catalog["artifacts"]
        if value["artifact_id"] == config.output_artifact_id
    )
    registry = yaml.safe_load((ROOT / "experiments/midogpp/registry.yaml").read_text())
    experiment = next(
        value
        for value in registry["experiments"]
        if value["experiment_id"] == config.experiment_id
    )
    assert tuple(row["required_files"]) == REQUIRED_FILES
    assert all(isinstance(note, str) for note in experiment["notes"])
    assert experiment["status"] == "failed"
    assert "cannot be recovered or rerun" in experiment["runner"]["argv"][-1]
    assert experiment["config_path"].endswith(
        "center_balanced_posterior_utility_prefix_router_v2.yaml"
    )
    assert row["evidence_label"] == "REJECTED"
    assert row["availability"] == "workstation_failed_preterminal"
    assert row["semantic_identities"]["run_state_status"] == "FAILED"
    assert row["semantic_identities"]["terminal_access_journal_count"] == "0"
    assert row["semantic_identities"]["execution_authorized"] == "true"
    assert row["semantic_identities"]["further_execution_authorized"] == "false"
    assert row["semantic_identities"]["authorization_exhausted"] == "true"
    assert row["semantic_identities"]["mechanical_repair_only"] == "true"
    assert row["semantic_identities"]["routing_success_claimed"] == "false"
    assert "tables/support_fold_plans.json" not in REQUIRED_FILES
    assert "tables/route_posterior_ensembles.json" not in REQUIRED_FILES
    assert (
        config.protocol[
            "pseudo_outer_H_frozen_label_free_expert_fingerprint_covariates_present"
        ]
        is True
    )
    assert config.protocol["pseudo_posterior_is_outer_H_covariate_invariant"] is False
    assert config.claim_boundary["fresh_evidence"] is False
    assert config.claim_boundary["may_feed_another_experiment"] is False


def test_cli_registers_the_new_router_surface() -> None:
    args = build_parser().parse_args(
        [
            "fixed-bank-p-anchored-route-scoped-center-balanced-posterior-utility-prefix-router",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/cbpupr-test-output",
        ]
    )
    assert args.surface.endswith("center-balanced-posterior-utility-prefix-router")


def test_input_fence_rejects_predecessor_semantic_paths() -> None:
    config = load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config(
        CONFIG
    )
    poisoned = SimpleNamespace(
        **{
            **config.__dict__,
            "test_cache_root": Path(
                "/tmp/fixed_bank_p_anchored_route_scoped_boundary_projected_v2"
            ),
        }
    )
    with pytest.raises(ProtocolError, match="predecessor diagnostic input"):
        assert_input_fence(poisoned)


def test_outer_endpoint_worker_uses_plain_spawn_payload() -> None:
    surface = _surface()
    prepared = prepare_center(surface)
    support = ("case-0", "case-1")
    outcomes = CenterCaseOutcomes(
        "0",
        support,
        np.ones((2, 8, 2), dtype=np.int64),
        np.full((2, 8, 2), 2, dtype=np.int64),
        np.asarray([1, 1], dtype=np.int64),
        np.asarray([1, 1], dtype=np.int64),
    )
    plan = WholeCaseOuterPlan(
        "0",
        "case-2",
        "case-2",
        support,
        ("sample-4", "sample-5"),
        surface.surface_hash,
    )
    job = OuterEndpointJob(
        "0",
        prepared,
        (("case-2", outcomes),),
        (plan,),
        tuple(
            ((source, direction), 0.0)
            for source in ("1", "2", "3", "5", "6", "7", "8", "9")
            for direction in DIRECTION_IDS
        ),
    )
    payload = _endpoint_job_payload(job)
    reverse = tuple(reversed(range(len(surface.sample_ids))))
    raw_surface = payload["surface"]
    assert isinstance(raw_surface, dict)
    assert raw_surface["surface_hash"] == surface.surface_hash
    with pytest.raises(ProtocolError, match="worker surface hash"):
        _compute_outer_endpoint_payload(
            {
                **payload,
                "surface": {**raw_surface, "surface_hash": "f" * 64},
            }
        )
    changed_seed_rows = list(raw_surface["seed_probabilities"])
    changed_action, changed_values = changed_seed_rows[0]
    changed_values = np.asarray(changed_values, dtype=np.float32).copy()
    changed_values[0, 0] = np.float32(1.0) - changed_values[0, 0]
    changed_seed_rows[0] = (changed_action, changed_values)
    with pytest.raises(ProtocolError, match="worker surface hash"):
        _compute_outer_endpoint_payload(
            {
                **payload,
                "surface": {
                    **raw_surface,
                    "seed_probabilities": tuple(changed_seed_rows),
                },
            }
        )
    poisoned_surface = {
        **raw_surface,
        "sample_ids": tuple(raw_surface["sample_ids"][index] for index in reverse),
        "case_ids": tuple(raw_surface["case_ids"][index] for index in reverse),
        "seed_probabilities": tuple(
            (action, np.asarray(values, dtype=np.float32)[:, reverse])
            for action, values in raw_surface["seed_probabilities"]
        ),
    }
    with pytest.raises(ProtocolError, match="identity, row, or action order"):
        _compute_outer_endpoint_payload({**payload, "surface": poisoned_surface})
    raw_plans = tuple(payload["outer_plans"])
    raw_plan = raw_plans[0]
    assert isinstance(raw_plan, dict)
    with pytest.raises(ProtocolError, match="plan hash"):
        _compute_outer_endpoint_payload(
            {
                **payload,
                "outer_plans": (
                    {**raw_plan, "plan_hash": "f" * 64},
                    *raw_plans[1:],
                ),
            }
        )
    with pytest.raises(ProtocolError, match="plan hash"):
        _compute_outer_endpoint_payload(
            {
                **payload,
                "outer_plans": (
                    {**raw_plan, "group_id": "changed-valid-group"},
                    *raw_plans[1:],
                ),
            }
        )
    with pytest.raises(ProtocolError, match="plan hash"):
        _compute_outer_endpoint_payload(
            {
                **payload,
                "outer_plans": (
                    {**raw_plan, "labels_used": True},
                    *raw_plans[1:],
                ),
            }
        )
    result = _spawn_call(_compute_outer_endpoint_payload, payload)
    assert result["endpoint_model_fit_count"] == 16
    assert result["target_center"] == "0"
    assert len(result["predictions"]) == 1
    products = _endpoint_products_from_payload(result)
    _validate_endpoint_products_against_job(job, products)
    assert products.predictions[0].sample_ids == plan.evaluation_sample_ids
    _case_id, state = products.states[0]
    state_payload = state.to_payload()
    assert EndpointState.from_payload(state_payload).to_payload() == state_payload

    raw_states = tuple(result["states"])
    state_case, raw_state = raw_states[0]
    assert isinstance(raw_state, dict)
    with pytest.raises(ProtocolError, match="endpoint-state hash"):
        _endpoint_products_from_payload(
            {
                **result,
                "states": (
                    (state_case, {**raw_state, "state_hash": "f" * 64}),
                    *raw_states[1:],
                ),
            }
        )
    changed_state = json.loads(json.dumps(raw_state))
    changed_state["model_coefficients"][0][0][0] += 0.125
    with pytest.raises(ProtocolError, match="endpoint-state hash"):
        _endpoint_products_from_payload(
            {
                **result,
                "states": ((state_case, changed_state), *raw_states[1:]),
            }
        )

    with pytest.raises(ProtocolError, match="product topology"):
        _endpoint_products_from_payload(
            {**result, "endpoint_model_fit_count": 17}
        )

    raw_predictions = tuple(result["predictions"])
    raw_prediction = raw_predictions[0]
    assert isinstance(raw_prediction, dict)
    with pytest.raises(ProtocolError, match="prediction hash"):
        _endpoint_products_from_payload(
            {
                **result,
                "predictions": (
                    {**raw_prediction, "prediction_hash": "f" * 64},
                    *raw_predictions[1:],
                ),
            }
        )
    changed_endpoint_probabilities = list(raw_prediction["probabilities"])
    changed_method, changed_endpoint_values = changed_endpoint_probabilities[0]
    changed_endpoint_values = list(changed_endpoint_values)
    changed_endpoint_values[0] = (
        0.0 if float(changed_endpoint_values[0]) > 0.5 else 1.0
    )
    changed_endpoint_probabilities[0] = (
        changed_method,
        tuple(changed_endpoint_values),
    )
    with pytest.raises(ProtocolError, match="prediction hash"):
        _endpoint_products_from_payload(
            {
                **result,
                "predictions": (
                    {
                        **raw_prediction,
                        "probabilities": tuple(changed_endpoint_probabilities),
                    },
                    *raw_predictions[1:],
                ),
            }
        )
    forged_prediction = EndpointCasePrediction(
        str(raw_prediction["center"]),
        str(raw_prediction["case_id"]),
        tuple(raw_prediction["sample_ids"]),
        {
            str(method): tuple(values)
            for method, values in raw_prediction["probabilities"]
        },
        "f" * 64,
    )
    forged_prediction_payload = {
        **raw_prediction,
        "state_hash": forged_prediction.state_hash,
        "prediction_hash": forged_prediction.prediction_hash,
    }
    with pytest.raises(ProtocolError, match="prediction/state lineage"):
        _endpoint_products_from_payload(
            {
                **result,
                "predictions": (
                    forged_prediction_payload,
                    *raw_predictions[1:],
                ),
            }
        )

    original_prediction = products.predictions[0]
    wrong_case_samples = EndpointCasePrediction(
        original_prediction.center,
        original_prediction.case_id,
        ("sample-2", "sample-3"),
        dict(original_prediction.probabilities),
        original_prediction.state_hash,
    )
    with pytest.raises(ProtocolError, match="plan lineage"):
        _validate_endpoint_products_against_job(
            job,
            OuterEndpointProducts(
                products.target_center,
                (wrong_case_samples,),
                products.states,
                products.state_hashes,
                products.endpoint_model_fit_count,
            ),
        )


def test_route_posterior_worker_uses_plain_spawn_payload() -> None:
    surface = _surface()
    primary = build_physical_fingerprint_surface(surface)
    blocked = blocked_within_case_fingerprint(primary)
    identities = (
        ("case-0", "sample-0", 0),
        ("case-0", "sample-1", 1),
        ("case-1", "sample-2", 0),
        ("case-1", "sample-3", 1),
        ("case-2", "sample-4", 0),
        ("case-2", "sample-5", 1),
    )
    route_labels = tuple(
        (
            held,
            tuple(
                BinaryLabel(
                    "0",
                    case,
                    sample,
                    value,
                    f"outer_support::H=0::excluded_c={held}",
                )
                for case, sample, value in identities
                if case != held
            ),
        )
        for held in ("case-0", "case-1", "case-2")
    )
    job = TargetCenterPosteriorJob(
        "0", primary, blocked, route_labels
    )
    payload = _posterior_job_payload(job)
    reverse = tuple(reversed(range(len(primary.sample_ids))))
    raw_primary = payload["primary"]
    assert isinstance(raw_primary, dict)
    assert raw_primary["fingerprint_hash"] == primary.fingerprint_hash
    with pytest.raises(ProtocolError, match="fingerprint worker hash"):
        _compute_posterior_payload(
            {
                **payload,
                "primary": {**raw_primary, "fingerprint_hash": "f" * 64},
            }
        )
    changed_features = np.asarray(
        raw_primary["feature_values"], dtype=np.float64
    ).copy()
    changed_features[0, 0] += 0.125
    with pytest.raises(ProtocolError, match="fingerprint worker hash"):
        _compute_posterior_payload(
            {
                **payload,
                "primary": {**raw_primary, "feature_values": changed_features},
            }
        )
    poisoned_primary = {
        **raw_primary,
        "sample_ids": tuple(raw_primary["sample_ids"][index] for index in reverse),
        "case_ids": tuple(raw_primary["case_ids"][index] for index in reverse),
        "feature_values": np.asarray(
            raw_primary["feature_values"], dtype=np.float64
        )[list(reverse)],
    }
    with pytest.raises(ProtocolError, match="physical fingerprint topology"):
        _compute_posterior_payload({**payload, "primary": poisoned_primary})
    result = _spawn_call(_compute_posterior_payload, payload)
    assert result["target_center"] == "0"
    assert result["model_fit_count"] == 6
    assert len(result["models"]) == 6
    assert len(result["predictions"]) == 6
    products = _posterior_products_from_payload(result)
    _validate_posterior_products_against_job(job, products)
    for prediction in products.predictions:
        expected_samples = tuple(
            sample
            for sample, case in zip(surface.sample_ids, surface.case_ids)
            if case == prediction.held_case_id
        )
        assert prediction.sample_ids == expected_samples
    model = products.models[0]
    fingerprint = primary if model.control_id == primary.control_id else blocked
    replayed = predict_route_posterior(fingerprint, model)
    persisted = next(
        row
        for row in products.predictions
        if (row.held_case_id, row.control_id)
        == (model.held_case_id, model.control_id)
    )
    assert model.to_payload() == result["models"][0]
    assert replayed.to_payload() == persisted.to_payload()
    assert np.asarray(replayed.natural_probabilities, dtype=np.float32).tobytes() == (
        np.asarray(persisted.natural_probabilities, dtype=np.float32).tobytes()
    )

    raw_models = tuple(result["models"])
    first_raw_model = raw_models[0]
    assert isinstance(first_raw_model, dict)
    with pytest.raises(ProtocolError, match="model worker hash"):
        _posterior_products_from_payload(
            {
                **result,
                "models": (
                    {**first_raw_model, "model_hash": "f" * 64},
                    *raw_models[1:],
                ),
            }
        )
    with pytest.raises(ProtocolError, match="model worker hash"):
        _posterior_products_from_payload(
            {
                **result,
                "models": (
                    {
                        **first_raw_model,
                        "intercept": float(first_raw_model["intercept"]) + 0.125,
                    },
                    *raw_models[1:],
                ),
            }
        )

    raw_predictions = tuple(result["predictions"])
    first_raw_prediction = raw_predictions[0]
    assert isinstance(first_raw_prediction, dict)
    with pytest.raises(ProtocolError, match="prediction worker hash"):
        _posterior_products_from_payload(
            {
                **result,
                "predictions": (
                    {**first_raw_prediction, "prediction_hash": "f" * 64},
                    *raw_predictions[1:],
                ),
            }
        )
    changed_probabilities = list(first_raw_prediction["natural_probabilities"])
    changed_probabilities[0] = (
        0.001 if float(changed_probabilities[0]) > 0.5 else 0.999
    )
    with pytest.raises(ProtocolError, match="prediction worker hash"):
        _posterior_products_from_payload(
            {
                **result,
                "predictions": (
                    {
                        **first_raw_prediction,
                        "natural_probabilities": tuple(changed_probabilities),
                    },
                    *raw_predictions[1:],
                ),
            }
        )
    with pytest.raises(ProtocolError, match="product count"):
        _posterior_products_from_payload(
            {**result, "model_fit_count": 7}
        )

    mismatched_model_hash = str(raw_models[1]["model_hash"])
    mismatched_prediction = CasePosteriorPrediction(
        str(first_raw_prediction["target_center"]),
        str(first_raw_prediction["held_case_id"]),
        str(first_raw_prediction["control_id"]),
        tuple(first_raw_prediction["sample_ids"]),
        tuple(first_raw_prediction["natural_probabilities"]),
        mismatched_model_hash,
        str(first_raw_prediction["fingerprint_hash"]),
    )
    with pytest.raises(ProtocolError, match="model/prediction lineage"):
        _posterior_products_from_payload(
            {
                **result,
                "predictions": (
                    mismatched_prediction.to_payload(),
                    *raw_predictions[1:],
                ),
            }
        )
    mismatched_key_prediction = CasePosteriorPrediction(
        str(first_raw_prediction["target_center"]),
        "case-not-in-worker-models",
        str(first_raw_prediction["control_id"]),
        tuple(first_raw_prediction["sample_ids"]),
        tuple(first_raw_prediction["natural_probabilities"]),
        str(first_raw_prediction["model_hash"]),
        str(first_raw_prediction["fingerprint_hash"]),
    )
    with pytest.raises(ProtocolError, match="model/prediction lineage"):
        _posterior_products_from_payload(
            {
                **result,
                "predictions": (
                    mismatched_key_prediction.to_payload(),
                    *raw_predictions[1:],
                ),
            }
        )

    original_model = products.models[0]
    original_prediction = products.predictions[0]
    forged_model = TargetLocalPosteriorModel(
        original_model.target_center,
        original_model.held_case_id,
        original_model.control_id,
        original_model.training_case_ids,
        original_model.feature_names,
        original_model.feature_mean,
        original_model.feature_scale,
        original_model.coefficients,
        original_model.intercept,
        original_model.training_row_count,
        original_model.training_n_positive,
        original_model.training_n_negative,
        "e" * 64,
        original_model.training_identity_hash,
        original_model.iterations,
        original_model.converged,
    )
    forged_source_prediction = CasePosteriorPrediction(
        original_prediction.target_center,
        original_prediction.held_case_id,
        original_prediction.control_id,
        original_prediction.sample_ids,
        original_prediction.natural_probabilities,
        forged_model.model_hash,
        forged_model.fingerprint_hash,
    )
    with pytest.raises(ProtocolError, match="source lineage"):
        _validate_posterior_products_against_job(
            job,
            TargetCenterPosteriorProducts(
                products.target_center,
                (forged_model, *products.models[1:]),
                (forged_source_prediction, *products.predictions[1:]),
                products.model_fit_count,
            ),
        )

    replay_poisoned_probabilities = list(
        original_prediction.natural_probabilities
    )
    replay_poisoned_probabilities[0] = (
        0.001 if replay_poisoned_probabilities[0] > 0.5 else 0.999
    )
    replay_poisoned_prediction = CasePosteriorPrediction(
        original_prediction.target_center,
        original_prediction.held_case_id,
        original_prediction.control_id,
        original_prediction.sample_ids,
        tuple(replay_poisoned_probabilities),
        original_prediction.model_hash,
        original_prediction.fingerprint_hash,
    )
    with pytest.raises(ProtocolError, match="worker replay"):
        _validate_posterior_products_against_job(
            job,
            TargetCenterPosteriorProducts(
                products.target_center,
                products.models,
                (replay_poisoned_prediction, *products.predictions[1:]),
                products.model_fit_count,
            ),
        )


def test_pseudo_endpoint_evidence_binds_h_excluded_priors_and_prediction() -> None:
    outer, target = "0", "1"
    sources = candidate_sources(target)
    priors = {
        (source, direction): 0.0 if source == outer else 0.125
        for source in sources
        for direction in DIRECTION_IDS
    }
    capability_hashes = {
        source: canonical_hash([outer, target, source])
        for source in sources
        if source != outer
    }
    source_evidence = PseudoSourcePriorEvidence(
        outer, target, priors, capability_hashes
    )
    probabilities = {
        method: (0.2, 0.8) for method in ENDPOINT_METHOD_IDS
    }
    prediction = EndpointCasePrediction(
        target,
        "case-1",
        ("sample-1", "sample-2"),
        probabilities,
        "a" * 64,
    )
    evidence = PseudoEndpointEvidence(
        outer, prediction, source_evidence.source_prior_hash
    )
    assert source_evidence.to_payload()["source_excluded_centers"] == [outer, target]
    assert evidence.to_payload()["source_prior_hash"] == source_evidence.source_prior_hash
    assert evidence.to_payload()["prediction_hash"] == prediction.prediction_hash


def test_runner_requires_durable_preterminal_barrier_before_terminal_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class Firewall:
        def open_target_terminal_labels(self) -> tuple[object, ...]:
            events.append("terminal_open")
            return (object(),)

        def audit_payload(self) -> dict[str, object]:
            events.append("final_audit")
            return {"terminal_opened": True}

    digests = {name: name[0] * 64 for name in "abcde"}
    preterminal = SimpleNamespace(
        preterminal_hash=digests["a"],
        candidates=SimpleNamespace(
            firewall=Firewall(),
            target_candidate_seal_hash=digests["b"],
            pre_evaluation_seal_hash=digests["c"],
        ),
        decisions=SimpleNamespace(
            replay_calibration_seal_hash=digests["d"],
            aggregate_seal_hash=digests["e"],
        ),
    )
    monkeypatch.setattr(
        runner,
        "persist_label_capability_report",
        lambda _root, _payload: events.append("capability_persisted"),
    )
    monkeypatch.setattr(
        runner,
        "persist_terminal_label_access_intent",
        lambda _root, *, expected_checks: (
            events.append("terminal_access_intent")
            or {"terminal_access_intent_hash": "f" * 64}
        ),
    )
    monkeypatch.setattr(
        runner,
        "persist_terminal_label_access_opened_receipt",
        lambda _root, *, intent, labels: events.append(
            "terminal_access_opened_receipt"
        ),
    )
    checks = {"preterminal_hash": digests["a"]}
    monkeypatch.setattr(
        runner,
        "validate_preterminal_gate_artifacts",
        lambda _root, *, expected_checks: dict(expected_checks),
    )

    with pytest.raises(ProtocolError, match="durable barrier"):
        runner._open_terminal_after_durable_preterminal(
            tmp_path, preterminal, expected_checks=checks
        )
    assert events == []

    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/decision_barrier.json").write_text("{}")
    (tmp_path / "manifests/preterminal_aggregate_seal.json").write_text("{}")
    with pytest.raises(ProtocolError, match="lineage drifted"):
        runner._open_terminal_after_durable_preterminal(
            tmp_path, preterminal, expected_checks=checks
        )
    assert events == []

    decision_barrier = {
        "schema_version": "fixed_bank_cbpupr_decision_barrier_v1",
        "candidate_seal_hash": digests["b"],
        "pre_evaluation_seal_hash": digests["c"],
        "replay_calibration_seal_hash": digests["d"],
        "pseudo_evaluation_opened_after_candidate_seal": True,
        "target_evaluation_opened": False,
    }
    (tmp_path / "manifests/decision_barrier.json").write_text(
        json.dumps(
            {
                **decision_barrier,
                "decision_barrier_hash": canonical_hash(decision_barrier),
            }
        )
    )
    (tmp_path / "manifests/preterminal_aggregate_seal.json").write_text(
        json.dumps(
            {
                "schema_version": "fixed_bank_cbpupr_preterminal_aggregate_seal_v1",
                "aggregate_seal_hash": digests["e"],
                "preterminal_hash": digests["a"],
                "target_evaluation_opened": False,
            }
        )
    )
    labels, audit = runner._open_terminal_after_durable_preterminal(
        tmp_path, preterminal, expected_checks=checks
    )
    assert len(labels) == 1
    assert audit == {"terminal_opened": True}
    assert events == [
        "terminal_access_intent",
        "terminal_open",
        "terminal_access_opened_receipt",
        "final_audit",
        "capability_persisted",
    ]
