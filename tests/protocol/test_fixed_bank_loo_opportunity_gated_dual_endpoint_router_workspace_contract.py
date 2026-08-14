from __future__ import annotations

from pathlib import Path

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
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.protocol import (
    build_frozen_science_protocol,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_loo_opportunity_gated_"
    "dual_endpoint_router_v1.yaml"
)


def test_successor_is_exact_six_input_terminal_consumed_diagnostic() -> None:
    config = load_fixed_bank_loo_opportunity_gated_dual_endpoint_router_config(
        CONFIG
    )
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(contracts.EXPERIMENT_ID)
    output = workspace.artifacts[contracts.OUTPUT_ARTIFACT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == contracts.INPUT_ARTIFACT_IDS
    assert len(experiment.input_artifact_ids) == 6
    assert experiment.run_recovery_strategy is None
    assert output.required_files == REQUIRED_FILES
    assert len(REQUIRED_FILES) == 49
    assert output.semantic_identities["config_contract_hash"] == config.contract_hash
    assert output.semantic_identities["protocol_contract_hash"] == (
        build_frozen_science_protocol().protocol_hash
    )
    assert output.semantic_identities["fresh_evidence"] == "false"
    assert output.semantic_identities["routing_success_claimed"] == "false"
    assert output.semantic_identities["source_identification_is_established"] == "false"
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False


def test_successor_aliases_are_single_consumer_and_predecessor_free() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(contracts.EXPERIMENT_ID)

    forbidden = (
        "loo_directional_shrinkage_ensemble",
        "case_directional_correctness_abstention_router",
        "support_static_router",
    )
    assert not any(
        fragment in artifact_id
        for artifact_id in experiment.input_artifact_ids
        for fragment in forbidden
    )
    for artifact_id in contracts.INPUT_ARTIFACT_IDS[2:]:
        artifact = workspace.artifacts[artifact_id]
        assert artifact.semantic_identities[
            "authorized_consumer_experiment_ids"
        ] == contracts.EXPERIMENT_ID
        assert artifact.semantic_identities["fresh_evidence"] == "false"


def test_label_barriers_science_and_workstation_topology_are_frozen() -> None:
    config = load_fixed_bank_loo_opportunity_gated_dual_endpoint_router_config(
        CONFIG
    )
    protocol = config.protocol
    identification = config.identification_endpoint

    assert protocol["all_72_donor_grants_complete_before_route_support"] is True
    assert protocol[
        "route_labels_never_enter_own_fit_scaler_state_or_decision"
    ] is True
    assert protocol[
        "all_218_predictions_and_decisions_sealed_before_terminal_labels"
    ] is True
    assert identification["support_calibrated_output_name"] == "expected_BACC_proxy"
    assert identification["output_is_NELBO"] is False
    assert identification["output_is_held_case_utility"] is False
    assert config.portfolio["composition_level"] == (
        "common_row_aligned_prediction_probability"
    )
    assert config.runtime["persistent_generation_worker_count"] == 2
    assert config.runtime["route_model_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["cuda_visible_devices_cleared_before_cpu_phase"] is True
    assert config.runtime["cross_run_recovery_allowed"] is False
    assert config.runtime["terminal_recovery_allowed"] is False
    assert config.runtime["two_fresh_process_validation_required"] is True


def test_bundle_inventory_has_exact_terminal_validation_exclusions() -> None:
    assert INDEX_EXCLUDED == frozenset(
        {
            "manifests/content_index.json",
            "reports/run_state.json",
            "reports/fresh_process_attestation.json",
            "reports/validation_report.json",
        }
    )
    assert CONTENT_INDEX_MEMBERS == tuple(
        member for member in REQUIRED_FILES if member not in INDEX_EXCLUDED
    )
    assert "manifests/identification_endpoint_seal.json" in REQUIRED_FILES
    assert "manifests/robust_endpoint_seal.json" in REQUIRED_FILES
    assert "manifests/portfolio_prediction_seal.json" in REQUIRED_FILES
    assert "tables/whole_pipeline_delete_one_center.csv" in REQUIRED_FILES
    assert "tables/attribution_controls.csv" in REQUIRED_FILES
