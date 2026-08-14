from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router import (
    experiment_contracts as contracts,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.bundle import (
    CONTENT_INDEX_MEMBERS,
    INDEX_EXCLUDED,
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.config import (
    load_fixed_bank_loo_opportunity_gated_dual_endpoint_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.protocol import (
    build_frozen_science_protocol,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_loo_opportunity_gated_"
    "dual_endpoint_router_v1.yaml"
)
AMENDMENT = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "uniform_b_v2_consumed_test_fixed_bank_loo_opportunity_gated_"
    "dual_endpoint_router_ledger_amendment_v1.json"
)


def test_config_freezes_endpoints_portfolio_claims_and_workstation() -> None:
    config = load_fixed_bank_loo_opportunity_gated_dual_endpoint_router_config(
        CONFIG
    )
    assert config.contract_hash == "693cfbe7aec131dd"
    assert config.input_artifact_ids == contracts.INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 6

    identification = config.identification_endpoint
    assert identification["fit_scope"] == "same_H_whole_cases_except_c_only"
    assert identification["eligibility_requires_positive_held_flip_count"] is True
    assert identification["eligibility_requires_strict_positive_case_proxy"] is True
    assert identification["case_scale"] == "mean_absolute_over_exact_eight_candidates"
    assert identification["normalization_epsilon_used"] is False
    assert identification["case_proxy_weight_fraction"] == "4/5"
    assert identification["donor_prior_weight_fraction"] == "1/5"
    assert identification["nonfinite_policy"] == "entire_route_fails_closed_to_OFF"

    robust = config.robust_endpoint
    assert robust["K_grid"] == [4, 5, 6]
    assert robust["w_rational_grid"] == ["1/2", "3/5", "7/10"]
    assert robust["arm_count"] == 9
    assert robust["all_arm_identities_retained_when_selected_endpoints_duplicate"] is True

    assert config.portfolio["identification_weight_fraction"] == "3/5"
    assert config.portfolio["robust_weight_fraction"] == "2/5"
    assert config.portfolio["sole_final_threshold"] == 0.5
    assert config.portfolio["CVAE_or_generative_mixture"] is False
    assert config.portfolio["NELBO_or_compatibility_estimate"] is False
    assert "portfolio_robust_weight_sensitivity_fractions" not in config.controls
    assert "weight_sensitivity_is_descriptive_only" not in config.controls
    assert "identification_baselines" not in config.evaluation

    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["route_model_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["parent_cuda_context_forbidden"] is True
    assert config.runtime["cross_run_recovery_allowed"] is False
    assert config.runtime["terminal_recovery_allowed"] is False
    assert config.runtime["two_fresh_process_validation_required"] is True

    boundary = config.claim_boundary
    assert boundary["method_development_is_posthoc"] is True
    assert boundary["weights_selected_on_same_evaluation_surface"] is True
    assert boundary["observed_vs_B_is_descriptive_only"] is True
    assert boundary["delete_center_results_are_descriptive_only"] is True
    assert boundary["incremental_vs_R_is_inconclusive"] is True
    assert boundary["source_identification_is_established"] is False
    assert boundary["nominal_coverage_claimed"] is False
    assert boundary["nominal_significance_claimed"] is False


def test_amendment_is_direct_hash_bound_single_consumer_and_terminal() -> None:
    raw = AMENDMENT.read_bytes()
    payload = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == contracts.EXPECTED_LEDGER_AMENDMENT_SHA256
    assert payload["parent_sha256"] == contracts.EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
    assert payload["authorized_consumer_experiment_ids"] == [
        contracts.EXPERIMENT_ID
    ]
    assert payload["route_labels_never_enter_own_fit_scaler_state_or_decision"] is True
    assert payload["all_72_donor_grants_complete_before_route_support"] is True
    assert payload[
        "all_218_predictions_and_decisions_sealed_before_terminal_label_access"
    ] is True
    assert payload["previous_stage90_outputs_used"] is False
    assert payload["previous_stage90_amendments_used"] is False
    assert payload["fresh_evidence"] is False
    assert payload["generic_consumer_authorized"] is False


def test_workspace_catalog_and_exact_closed_world_inventory() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(contracts.EXPERIMENT_ID)
    output = workspace.artifacts[contracts.OUTPUT_ARTIFACT_ID]

    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == contracts.INPUT_ARTIFACT_IDS
    assert experiment.run_recovery_strategy is None
    assert experiment.runner_argv[4] == (
        "fixed-bank-loo-opportunity-gated-dual-endpoint-router"
    )
    assert output.required_files == REQUIRED_FILES
    assert len(REQUIRED_FILES) == 49
    assert set(REQUIRED_FILES) == set(CONTENT_INDEX_MEMBERS) | set(INDEX_EXCLUDED)
    assert set(CONTENT_INDEX_MEMBERS).isdisjoint(INDEX_EXCLUDED)
    assert output.semantic_identities["config_contract_hash"] == "693cfbe7aec131dd"
    assert output.semantic_identities["protocol_contract_hash"] == (
        build_frozen_science_protocol().protocol_hash
    )
    assert output.semantic_identities["fresh_evidence"] == "false"
    assert output.semantic_identities["registered_recovery_strategy"] == "none"
    assert output.semantic_identities["source_identification_is_established"] == "false"
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    for artifact_id in contracts.INPUT_ARTIFACT_IDS[2:]:
        artifact = workspace.artifacts[artifact_id]
        assert artifact.semantic_identities[
            "authorized_consumer_experiment_ids"
        ] == contracts.EXPERIMENT_ID


def test_config_and_input_fence_fail_closed_on_science_or_claim_drift(
    tmp_path: Path,
) -> None:
    config = load_fixed_bank_loo_opportunity_gated_dual_endpoint_router_config(
        CONFIG
    )
    assert_input_fence(config)
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["identification_endpoint"]["case_proxy_weight_fraction"] = "3/4"
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="section drifted"):
        load_fixed_bank_loo_opportunity_gated_dual_endpoint_router_config(drifted)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["claim_boundary"]["fresh_evidence"] = True
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="section drifted"):
        load_fixed_bank_loo_opportunity_gated_dual_endpoint_router_config(drifted)


def test_cli_registers_and_lazily_dispatches_surface(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    name = "fixed-bank-loo-opportunity-gated-dual-endpoint-router"
    parsed = cli.build_parser().parse_args(
        [name, "--config", str(CONFIG), "--artifact-root", "/tmp/ogde-v1"]
    )
    assert parsed.surface == name

    import midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router as surface

    sentinel = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        surface,
        "load_fixed_bank_loo_opportunity_gated_dual_endpoint_router_config",
        lambda _: sentinel,
    )
    monkeypatch.setattr(
        surface,
        "run_fixed_bank_loo_opportunity_gated_dual_endpoint_router",
        lambda config, *, artifact_root: calls.append((config, artifact_root))
        or Path("/tmp/ogde-result"),
    )
    assert cli.main(
        [name, "--config", str(CONFIG), "--artifact-root", "/tmp/ogde-v1"]
    ) == 0
    assert calls == [(sentinel, Path("/tmp/ogde-v1"))]
    assert capsys.readouterr().out.strip() == "/tmp/ogde-result"
