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
    _compute_outer_endpoint_payload,
    _job_payload as _endpoint_job_payload,
    _products_from_payload as _endpoint_products_from_payload,
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
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.pseudo_endpoint_evidence import (
    PseudoEndpointEvidence,
    PseudoSourcePriorEvidence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.target_local_runtime import (
    TargetCenterPosteriorJob,
    _compute_worker_payload as _compute_posterior_payload,
    _job_payload as _posterior_job_payload,
    _products_from_payload as _posterior_products_from_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router import runner
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.hashing import canonical_hash
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.protocol import ProtocolError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_center_"
    "balanced_posterior_utility_prefix_router_v1.yaml"
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
    result = _spawn_call(_compute_outer_endpoint_payload, payload)
    assert result["endpoint_model_fit_count"] == 16
    assert result["target_center"] == "0"
    assert len(result["predictions"]) == 1
    products = _endpoint_products_from_payload(result)
    _case_id, state = products.states[0]
    state_payload = state.to_payload()
    assert EndpointState.from_payload(state_payload).to_payload() == state_payload


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
    result = _spawn_call(_compute_posterior_payload, _posterior_job_payload(job))
    assert result["target_center"] == "0"
    assert result["model_fit_count"] == 6
    assert len(result["models"]) == 6
    assert len(result["predictions"]) == 6
    products = _posterior_products_from_payload(result)
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

    with pytest.raises(ProtocolError, match="durable barrier"):
        runner._open_terminal_after_durable_preterminal(tmp_path, preterminal)
    assert events == []

    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/decision_barrier.json").write_text("{}")
    (tmp_path / "manifests/preterminal_aggregate_seal.json").write_text("{}")
    with pytest.raises(ProtocolError, match="lineage drifted"):
        runner._open_terminal_after_durable_preterminal(tmp_path, preterminal)
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
        tmp_path, preterminal
    )
    assert len(labels) == 1
    assert audit == {"terminal_opened": True}
    assert events == ["terminal_open", "final_audit", "capability_persisted"]
