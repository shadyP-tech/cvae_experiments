from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router import (
    experiment_contracts as contracts,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.actions import (
    action_library_by_target,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.config import (
    load_fixed_bank_case_directional_correctness_abstention_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.protocol import (
    build_frozen_science_protocol,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_contracts import (
    validate_action_library,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_case_directional_correctness_"
    "abstention_router_v1.yaml"
)
AMENDMENT = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "uniform_b_v2_consumed_test_fixed_bank_case_directional_correctness_"
    "abstention_router_ledger_amendment_v1.json"
)


def test_config_freezes_H_minus_c_model_score_and_workstation() -> None:
    config = load_fixed_bank_case_directional_correctness_abstention_router_config(
        CONFIG
    )
    assert config.contract_hash == "a41dce9dfd086f4a"
    assert config.input_artifact_ids == contracts.INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 6
    assert config.protocol["route_labels_never_enter_own_fit_scaler_state_or_decision"] is True
    model = config.case_correctness_router
    assert model["one_model_per_H_c_e_direction"] is True
    assert model["fit_scope"] == "same_H_whole_cases_except_c_only"
    assert model["ridge_alpha"] == 1.0
    assert model["max_iterations"] == 50
    assert model["case_proxy_weight_fraction"] == "1/2"
    assert model["donor_prior_weight_fraction"] == "1/2"
    assert model["candidate_pool"] == "all_eight_non_target_sources_plus_OFF"
    assert model["predicted_held_case_exact_bacc_claimed"] is False
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["route_model_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["parent_cuda_context_forbidden"] is True
    assert config.runtime["cross_run_recovery_allowed"] is False
    for key in (
        "fresh_evidence",
        "routing_success_claimed",
        "routing_quality_claimed",
        "downstream_utility_claimed",
        "promotion_eligible",
        "may_feed_stage50",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_another_stage90",
        "may_feed_another_experiment",
    ):
        assert config.claim_boundary[key] is False


def test_runner_imports_and_action_library_satisfies_neutral_runtime() -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router import runner

    payload, library_hash = validate_action_library(action_library_by_target())
    assert runner.run_fixed_bank_case_directional_correctness_abstention_router
    assert tuple(payload) == contracts.CENTERS
    assert len(library_hash) == 16
    assert all(
        row["target_expert_excluded"] is True
        and row["labels_used"] is False
        and row["counts_by_class"]
        and row["sample_weight_by_source"]
        for actions in payload.values()
        for row in actions
    )


def test_amendment_is_direct_hash_bound_and_single_consumer() -> None:
    raw = AMENDMENT.read_bytes()
    payload = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == contracts.EXPECTED_LEDGER_AMENDMENT_SHA256
    assert payload["parent_sha256"] == contracts.EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
    assert payload["authorized_consumer_experiment_ids"] == [contracts.EXPERIMENT_ID]
    assert payload["route_labels_never_enter_own_fit_scaler_state_or_decision"] is True
    assert payload["all_72_donor_grants_complete_before_route_support"] is True
    assert payload["all_218_predictions_and_decisions_sealed_before_terminal_label_access"] is True
    assert payload["previous_stage90_outputs_used"] is False
    assert payload["fresh_evidence"] is False
    assert payload["generic_consumer_authorized"] is False


def test_workspace_catalog_and_exact_43_file_inventory() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(contracts.EXPERIMENT_ID)
    output = workspace.artifacts[contracts.OUTPUT_ARTIFACT_ID]
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == contracts.INPUT_ARTIFACT_IDS
    assert experiment.run_recovery_strategy is None
    assert experiment.runner_argv[4] == (
        "fixed-bank-case-directional-correctness-abstention-router"
    )
    assert output.required_files == REQUIRED_FILES
    assert len(REQUIRED_FILES) == 43
    assert output.semantic_identities["config_contract_hash"] == "a41dce9dfd086f4a"
    assert output.semantic_identities["protocol_contract_hash"] == (
        build_frozen_science_protocol().protocol_hash
    )
    assert output.semantic_identities["fresh_evidence"] == "false"
    assert output.semantic_identities["predicted_held_case_exact_bacc_claimed"] == "false"
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    for artifact_id in contracts.INPUT_ARTIFACT_IDS[2:]:
        artifact = workspace.artifacts[artifact_id]
        assert artifact.semantic_identities["authorized_consumer_experiment_ids"] == contracts.EXPERIMENT_ID


def test_config_and_input_fence_fail_closed_on_claim_or_lineage_drift(
    tmp_path: Path,
) -> None:
    config = load_fixed_bank_case_directional_correctness_abstention_router_config(
        CONFIG
    )
    assert_input_fence(config)
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["case_correctness_router"]["ridge_alpha"] = 0.5
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="section drifted"):
        load_fixed_bank_case_directional_correctness_abstention_router_config(drifted)
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["claim_boundary"]["fresh_evidence"] = True
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="section drifted"):
        load_fixed_bank_case_directional_correctness_abstention_router_config(drifted)


def test_cli_registers_and_lazily_dispatches_surface(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    name = "fixed-bank-case-directional-correctness-abstention-router"
    parsed = cli.build_parser().parse_args(
        [name, "--config", str(CONFIG), "--artifact-root", "/tmp/cdca-v1"]
    )
    assert parsed.surface == name
    import midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router as surface

    sentinel = object()
    calls = []
    monkeypatch.setattr(
        surface,
        "load_fixed_bank_case_directional_correctness_abstention_router_config",
        lambda _: sentinel,
    )
    monkeypatch.setattr(
        surface,
        "run_fixed_bank_case_directional_correctness_abstention_router",
        lambda config, *, artifact_root: calls.append((config, artifact_root))
        or Path("/tmp/cdca-result"),
    )
    assert cli.main(
        [name, "--config", str(CONFIG), "--artifact-root", "/tmp/cdca-v1"]
    ) == 0
    assert calls == [(sentinel, Path("/tmp/cdca-v1"))]
    assert capsys.readouterr().out.strip() == "/tmp/cdca-result"
