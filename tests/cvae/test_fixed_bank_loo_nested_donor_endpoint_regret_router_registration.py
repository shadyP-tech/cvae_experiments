from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router import (
    fresh_process_validation,
    validation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.actions import (
    action_library_by_target,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.artifact_io import (
    reject_sensitive_persistence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    assert_closed_world,
    validate_content_index,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.config import (
    load_nested_donor_endpoint_regret_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.constants import (
    CENTERS,
    ENDPOINT_METHOD_IDS,
    PORTFOLIO_METHOD_ID,
    REGRET_FEATURE_NAMES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.contracts import (
    CandidateDescriptor,
    DonorRegretRow,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.donor_regret_model import (
    fit_full_and_delete_donor_models,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.experiment_contracts import (
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.hashing import (
    canonical_hash,
    canonical_json,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.learn_then_test import (
    CenterBlockPolicyEvidence,
    learn_then_test_center_harm,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.label_capabilities import (
    validate_preterminal_aggregate_seal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.protocol import (
    build_frozen_protocol,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.physical_runtime import (
    runtime_summary_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.selection import (
    select_model_based_route,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import read_json, sha256_file
from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_contracts import (
    validate_action_library,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_loo_nested_donor_endpoint_regret_router_v1.yaml"
)
AMENDMENT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "uniform_b_v2_consumed_test_fixed_bank_loo_nested_donor_endpoint_regret_router_ledger_amendment_v1.json"
)
PACKAGE = (
    ROOT
    / "src/midogpp_thesis/cvae/diagnostics/"
    "fixed_bank_loo_nested_donor_endpoint_regret_router"
)


def _write_content_members(root: Path) -> None:
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture::{member}\n".encode("utf-8"))


def _replace_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate(*, support: float = 1.0, dispersion: float = 1.0) -> CandidateDescriptor:
    values = [0.0] * len(REGRET_FEATURE_NAMES)
    values[0] = support
    values[2] = dispersion
    values[3] = 0.25
    values[9] = 1.0
    values[-1] = 1.0
    return CandidateDescriptor(
        "0",
        "case-0",
        "B",
        REGRET_FEATURE_NAMES,
        tuple(values),
        ("1" * 64,),
    )


def _models() -> tuple[object, object]:
    values = (0.0,) * len(REGRET_FEATURE_NAMES)
    rows = tuple(
        DonorRegretRow(
            center,
            f"case-{center}",
            "B",
            values,
            0.1,
            -0.1,
            1,
            "2" * 64,
        )
        for center in CENTERS
        if center != "0"
    )
    return fit_full_and_delete_donor_models(rows, outer_target_center="0")


def test_registration_config_and_exact_six_inputs_are_frozen() -> None:
    config = load_nested_donor_endpoint_regret_config(CONFIG)
    protocol = build_frozen_protocol()
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))

    assert config.contract_hash == "1f60b4352a67c60f"
    assert protocol.protocol_hash == (
        "474ef49cf7b2fd6ce60ac10d473d5ffdb49abf028737b1aa5ee1d644f782884b"
    )
    assert sha256_file(AMENDMENT) == EXPECTED_LEDGER_AMENDMENT_SHA256
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(INPUT_ARTIFACT_IDS) == len(set(INPUT_ARTIFACT_IDS)) == 6
    assert experiment.claim_scope == "diagnostic_only"
    assert output.required_files == REQUIRED_FILES
    assert "tables/center_block_feasibility.json" in output.required_files
    assert all(isinstance(note, str) for note in experiment.notes)
    assert "oracle_and_diagnostic_evidence" in output.forbidden_reuse
    assert config.protocol["route_decision_label_blind"] is False
    assert config.protocol["center_block_ltt_statistical_authorization_enabled"] is False
    assert config.evaluation["route_pipeline_refit_inside_null_replicate"] is False
    assert config.claim_boundary["protected_fallback_label_blind"] is False
    assert amendment["route_decision_label_blind"] is False
    assert amendment["protected_fallback_label_blind"] is False
    assert amendment["full_selection_inference_claimed"] is False
    assert experiment.runner_argv[-2:] == (
        "--artifact-root",
        f"output://{OUTPUT_ARTIFACT_ID}",
    )


def test_action_library_satisfies_neutral_exact_810_contract() -> None:
    actions = action_library_by_target()
    payload, digest = validate_action_library(actions)

    assert tuple(actions) == CENTERS
    assert all(len(rows) == 10 for rows in actions.values())
    assert sum(len(rows) for rows in payload.values()) == 90
    assert len(digest) == 16


def test_cli_surface_is_registered_without_eager_execution() -> None:
    parsed = build_parser().parse_args(
        [
            "fixed-bank-loo-nested-donor-endpoint-regret-router",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/nested-regret-fixture",
        ]
    )

    assert parsed.surface == "fixed-bank-loo-nested-donor-endpoint-regret-router"


def test_config_rejects_scientific_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["protocol"]["support_dispersion_multiplier"] = 0.75
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError, match="config section drifted"):
        load_nested_donor_endpoint_regret_config(path)


def test_input_fence_rejects_predecessor_diagnostic_path() -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router.inputs import (
        assert_input_fence,
    )

    config = load_nested_donor_endpoint_regret_config(CONFIG)
    assert_input_fence(config)
    poisoned = replace(
        config,
        test_cache_root=Path(
            "/tmp/fixed_bank_loo_directional_shrinkage_ensemble/cache"
        ),
    )
    with pytest.raises(ProtocolError, match="predecessor diagnostic input"):
        assert_input_fence(poisoned)


@pytest.mark.parametrize("tamper", ("header", "row", "bytes"))
def test_content_index_rejects_tamper(tmp_path: Path, tamper: str) -> None:
    _write_content_members(tmp_path)
    write_content_index(
        tmp_path,
        config_contract_hash="a" * 16,
        protocol_contract_hash="b" * 64,
    )
    index_path = tmp_path / "manifests/content_index.json"
    payload = read_json(index_path)
    if tamper == "header":
        payload["unexpected"] = False
        payload["content_hash"] = canonical_hash(
            {key: value for key, value in payload.items() if key != "content_hash"}
        )
        _replace_json(index_path, payload)
        error = "header drifted"
    elif tamper == "row":
        payload["members"][0]["sha256"] = "0" * 64
        payload["content_hash"] = canonical_hash(
            {key: value for key, value in payload.items() if key != "content_hash"}
        )
        _replace_json(index_path, payload)
        error = "indexed member drifted"
    else:
        (tmp_path / CONTENT_INDEX_MEMBERS[0]).write_bytes(b"tampered\n")
        error = "indexed member drifted"

    with pytest.raises(ProtocolError, match=error):
        validate_content_index(
            tmp_path,
            config_contract_hash="a" * 16,
            protocol_contract_hash="b" * 64,
        )


@pytest.mark.parametrize("binding", ("plan_seal_hash", "decision_barrier_hash"))
def test_preterminal_aggregate_seal_rejects_coherently_rehashed_binding_tamper(
    binding: str,
) -> None:
    plan_hash = "1" * 64
    barrier_hash = "2" * 64
    payload = {
        "schema_version": "fixed_bank_nested_regret_preterminal_aggregate_v1",
        "protocol_hash": "3" * 64,
        "probability_surface_hash": "4" * 16,
        "plan_seal_hash": plan_hash,
        "decision_barrier_hash": barrier_hash,
        "descriptor_hash": "5" * 64,
        "outer_excluded_donor_descriptor_hash": "6" * 64,
        "donor_model_hash": "7" * 64,
        "policy_menu_hash": "8" * 64,
        "ltt_authorization_hash": "9" * 64,
        "terminal_labels_used": False,
    }
    sealed = {**payload, "aggregate_seal_hash": canonical_hash(payload)}
    assert (
        validate_preterminal_aggregate_seal(
            sealed,
            expected_plan_seal_hash=plan_hash,
            expected_decision_barrier_hash=barrier_hash,
        )
        == sealed["aggregate_seal_hash"]
    )

    tampered_payload = dict(payload)
    tampered_payload[binding] = "a" * 64
    tampered = {
        **tampered_payload,
        "aggregate_seal_hash": canonical_hash(tampered_payload),
    }
    with pytest.raises(ProtocolError, match="exact plan and decision barrier"):
        validate_preterminal_aggregate_seal(
            tampered,
            expected_plan_seal_hash=plan_hash,
            expected_decision_barrier_hash=barrier_hash,
        )


def test_closed_world_and_sensitive_persistence_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "foreign").mkdir()
    (tmp_path / "foreign/file.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="foreign directory"):
        assert_closed_world(tmp_path, allow_incomplete=True)
    with pytest.raises(ProtocolError, match="persisted key is forbidden"):
        reject_sensitive_persistence({"nested": {"label": 1}})
    with pytest.raises(ProtocolError, match="exposes a path"):
        reject_sensitive_persistence({"identity": "/Users/example/image.png"})


def test_closed_world_rejects_empty_unauthorized_nested_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "arrays/foreign/empty").mkdir(parents=True)
    with pytest.raises(ProtocolError, match="foreign directory"):
        assert_closed_world(tmp_path, allow_incomplete=True)


def test_two_process_attestation_rejects_coherently_rehashed_child_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {"schema_version": "fixture_v1", "status": "PASS", "value": 7}
    parent = os.getpid()
    worker_payloads = iter(
        (
            {"process_id": parent + 10_001, "checks": expected},
            {"process_id": parent + 10_002, "checks": expected},
        )
    )

    def fake_worker(_root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=(sys.executable,),
            returncode=0,
            stdout=canonical_json(next(worker_payloads)) + "\n",
            stderr="",
        )

    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(fresh_process_validation, "_run_worker", fake_worker)
    report = fresh_process_validation.require_two_fresh_process_validations(
        tmp_path, expected_checks=expected
    )
    persisted = read_json(tmp_path / "reports/fresh_process_attestation.json")
    assert (
        fresh_process_validation.verify_attested_validation(
            report,
            expected_checks=expected,
            persisted_attestation=persisted,
        )
        == report
    )

    tampered = deepcopy(report)
    attestation = tampered[fresh_process_validation.ATTESTATION_KEY]
    attestation["child_process_results"][0]["result_hash"] = "0" * 64
    unsigned = {
        key: value for key, value in attestation.items() if key != "attestation_hash"
    }
    attestation["attestation_hash"] = canonical_hash(unsigned)
    with pytest.raises(ProtocolError, match="result drifted"):
        fresh_process_validation.verify_attested_validation(
            tampered,
            expected_checks=expected,
            persisted_attestation=attestation,
        )


def test_runtime_summary_is_exact_and_cross_bound(tmp_path: Path) -> None:
    preflight = {
        "status": "PASS",
        "scratch_root_id": ".nested-regret-scratch",
        "scratch_role": "artifact_parent",
    }
    runtime = {
        "persistent_generation_worker_count": 2,
        "classifier_workers": 4,
        "route_model_workers": 4,
        "classifier_threads_per_worker": 3,
        "expected_endpoint_model_fit_count": 46_048,
    }
    source = SimpleNamespace(
        lock_hash="1" * 16,
        records=tuple(range(81)),
        lock_payload={"source_stream_lock_hash": "1" * 16},
    )
    prediction = SimpleNamespace(
        seal_hash="2" * 16,
        store=SimpleNamespace(cells=tuple(range(810))),
    )
    physical = SimpleNamespace(
        canonical_source_cache=source,
        local_source_cache=source,
        prediction=prediction,
        scratch=SimpleNamespace(
            root=tmp_path / ".nested-regret-scratch",
            role="artifact_parent",
        ),
    )
    payload = runtime_summary_payload(
        physical,
        preflight=preflight,
        runtime=runtime,
    )
    path = tmp_path / "reports/runtime_summary.json"
    path.parent.mkdir()
    _replace_json(path, payload)

    assert (
        validation._validate_runtime_summary(
            tmp_path,
            preflight=preflight,
            runtime=runtime,
            source=source,
            prediction=prediction,
        )
        == payload
    )
    payload["unsealed_field"] = True
    _replace_json(path, payload)
    with pytest.raises(ProtocolError, match="runtime summary drifted"):
        validation._validate_runtime_summary(
            tmp_path,
            preflight=preflight,
            runtime=runtime,
            source=source,
            prediction=prediction,
        )


def test_eight_center_ltt_is_correctly_unable_to_authorize() -> None:
    evidence = CenterBlockPolicyEvidence(
        "NDR_MODEL_BASED",
        (0.1,) * 8,
        (-0.1,) * 8,
    )
    report = learn_then_test_center_harm((evidence,))

    assert report["tests"][0][
        "optimistic_independent_binomial_tail_probability"
    ] == pytest.approx(0.9**8)
    assert report["authorized_policy_ids"] == []
    assert report["statistical_authorization_enabled"] is False
    assert report["fallback_when_none_authorized"] == PORTFOLIO_METHOD_ID


def test_primary_router_authorizes_only_when_every_frozen_gate_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full, deleted = _models()
    authorized = select_model_based_route(
        _candidate(), full_models=full, delete_donor_models=deleted
    )
    support_failure = select_model_based_route(
        _candidate(support=0.5, dispersion=1.0),
        full_models=full,
        delete_donor_models=deleted,
    )

    assert authorized.selected_method == "B"
    assert authorized.reason.startswith("authorized_center_balanced")
    assert support_failure.selected_method == PORTFOLIO_METHOD_ID
    assert support_failure.reason == "fallback_P_insufficient_nested_support_margin"

    from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_nested_donor_endpoint_regret_router import (
        selection,
    )

    donors = tuple(center for center in CENTERS if center != "0")

    def unstable_prediction(model: object, _features: object) -> float:
        if getattr(model, "response_name") == "log_loss_delta":
            return -0.1
        if len(getattr(model, "training_centers")) == 8:
            return 0.1
        omitted = next(center for center in donors if center not in model.training_centers)
        return -0.1 if omitted in donors[:2] else 0.1

    monkeypatch.setattr(selection, "predict_unseen_center", unstable_prediction)
    deletion_failure = select_model_based_route(
        _candidate(), full_models=full, delete_donor_models=deleted
    )
    assert deletion_failure.selected_method == PORTFOLIO_METHOD_ID
    assert deletion_failure.reason == "fallback_P_insufficient_delete_donor_bacc_consensus"


def test_delete_donor_models_refit_and_exclude_the_deleted_center() -> None:
    full, deleted = _models()
    assert set(deleted) == set(CENTERS) - {"0"}
    for deleted_center, response_models in deleted.items():
        for response_name, model in response_models.items():
            assert response_name in {"bacc_regret", "log_loss_delta"}
            assert deleted_center not in model.training_centers
            assert len(model.training_centers) == 7
            assert model.model_hash != full[response_name].model_hash


def test_package_has_no_predecessor_diagnostic_imports() -> None:
    forbidden = (
        "fixed_bank_loo_opportunity_gated_dual_endpoint_router",
        "fixed_bank_loo_directional_shrinkage_ensemble",
        "fixed_bank_disagreement_regret_prediction_only",
        "fixed_bank_actionability_recoverability",
    )
    observed: list[tuple[Path, str]] = []
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            observed.extend((path, module) for module in modules)

    assert not [
        (path, module)
        for path, module in observed
        if any(name in module for name in forbidden)
    ]
    assert ENDPOINT_METHOD_IDS[-1] == PORTFOLIO_METHOD_ID
