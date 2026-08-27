from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.config import (
    CONFIG_TOP_LEVEL,
    frozen_config_contract_payload,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.identity import (
    CLI_SURFACE,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.runner import (
    run_planned_router,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / (
        "uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_"
        "equivalence_pairwise_primitive_utility_router_v1.yaml"
    )
)
DIRECT_ORIGINAL_INPUTS = INPUT_ARTIFACT_IDS
FORBIDDEN_DOWNSTREAM_PURPOSES = {
    "real_feature_reference_evidence",
    "cvae_preservation_evidence",
    "expert_bank_evidence",
    "generation_evidence",
    "all_candidate_utility_diagnostic",
    "routing_evidence",
    "expert_selection_evidence",
    "nelbo_compatibility_evidence",
    "synthetic_downstream_utility_evidence",
    "oracle_and_diagnostic_evidence",
}


def test_registration_is_three_direct_original_inputs_and_planned_only() -> None:
    config = load_config(CONFIG)
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == DIRECT_ORIGINAL_INPUTS
    assert len(set(experiment.input_artifact_ids)) == 3
    assert all("stage90" not in value for value in experiment.input_artifact_ids)
    assert all("support_calibrated" not in value for value in experiment.input_artifact_ids)
    assert all("amendment" not in value for value in experiment.input_artifact_ids)
    assert output.availability == "planned_execution_not_authorized"
    assert output.evidence_label == "NEEDS_EVIDENCE_EXECUTION_NOT_AUTHORIZED"
    assert output.semantic_identities["execution_authorized"] == "false"
    assert output.semantic_identities["consumed_test_reuse_authorized"] == "false"
    assert output.semantic_identities["target_terminal_labels_may_open"] == "false"
    assert output.semantic_identities["config_contract_hash"] == config.contract_hash
    assert output.semantic_identities["protocol_contract_hash"] == (
        config.protocol["protocol_hash"]
    )
    assert output.semantic_identities["consumed_test_cache_resolution_present"] == "false"
    assert output.semantic_identities[
        "parent_consumption_ledger_resolution_present"
    ] == "false"


def test_config_is_exact_six_section_path_free_non_authorized_contract() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config = load_config(CONFIG)

    assert set(payload) == set(CONFIG_TOP_LEVEL)
    assert payload == frozen_config_contract_payload()
    assert config.input_artifact_ids == DIRECT_ORIGINAL_INPUTS
    assert tuple(payload["inputs"]["direct_input_artifact_ids"]) == (
        DIRECT_ORIGINAL_INPUTS
    )
    assert payload["inputs"]["direct_input_count"] == 3
    assert payload["inputs"]["input_path_resolution_deferred"] is True
    assert payload["inputs"]["test_cache_capability_registered"] is False
    assert payload["inputs"]["test_label_capability_registered"] is False
    assert payload["inputs"][
        "test_consumption_ledger_capability_registered"
    ] is False
    assert payload["inputs"]["test_cache_resolution_status"] == (
        "PENDING_SEPARATE_FUTURE_AUTHORIZATION"
    )
    assert payload["inputs"]["parent_consumption_ledger_resolution_status"] == (
        "PENDING_SEPARATE_FUTURE_AUTHORIZATION"
    )
    assert payload["inputs"]["authorization_amendment_status"] == (
        "ABSENT_NOT_AUTHORIZED"
    )
    terminal_manifest = payload["inputs"]["canonical_terminal_manifest_contract"]
    assert terminal_manifest["annotation_artifact_id"] == (
        "midogpp_dataset_contract_annotation_patch_v1"
    )
    assert terminal_manifest["manifest_content_sha256"] == (
        "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
    )
    assert terminal_manifest["terminal_row_count"] == 9928
    assert terminal_manifest["terminal_case_count"] == 218
    assert terminal_manifest["terminal_case_inventory_hash"] == (
        "d22568075a287af71d0f4477ba5e6265e43278cba4865f7775741cdbcdf2bcc6"
    )
    assert terminal_manifest["input_resolution_authorized"] is False
    assert payload["protocol"]["target_H_labels_closed_preterminal"] is True
    assert payload["protocol"]["nested_K_rotation_centers"] == "EXACT_C_MINUS_H"
    assert payload["protocol"][
        "typed_opportunity_receipt_required_at_pairwise_fit_and_selection"
    ] is True
    assert payload["protocol"]["opportunity_candidate_action_inventory"] == (
        "EXACT_FROZEN_CANDIDATE_ACTION_IDS"
    )
    assert payload["protocol"][
        "typed_row_posterior_prediction_required_for_primitive_and_denominator"
    ] is True
    assert payload["protocol"][
        "primitive_action_id_exact_match_to_opportunity_member"
    ] is True
    assert payload["protocol"][
        "primitive_protected_baseline_probability_hash_exact_match_to_opportunity"
    ] is True
    assert payload["protocol"][
        "primitive_candidate_probability_hash_exact_match_to_opportunity_member"
    ] is True
    assert payload["protocol"][
        "primitive_denominator_posterior_model_hash_exact_match"
    ] is True
    assert payload["protocol"][
        "pairwise_fit_exact_matches_utility_action_and_probability_surface_to_opportunity"
    ] is True
    assert payload["protocol"][
        "selection_exact_matches_utility_action_and_probability_surface_to_opportunity"
    ] is True
    assert payload["protocol"][
        "selection_ledger_entry_requires_typed_opportunity_receipt"
    ] is True
    assert payload["protocol"][
        "selection_ledger_entry_exact_matches_opportunity_receipt_center_and_case"
    ] is True
    assert payload["protocol"][
        "selection_ledger_entry_exact_matches_decision_opportunity_receipt_hash"
    ] is True
    assert payload["protocol"][
        "prelabel_selection_decision_ledger_case_count"
    ] == 218
    assert payload["protocol"][
        "prelabel_selection_decision_ledger_requires_exact_dataset_case_manifest_hash"
    ] is True
    assert payload["protocol"]["outer_selection_lineage_inventory"] == (
        "EXACT_ONE_PER_ELIGIBLE_H"
    )
    assert payload["protocol"][
        "each_case_decision_must_match_its_H_specific_source_pool_model_and_calibration"
    ] is True
    assert payload["protocol"][
        "preterminal_phase_uses_canonical_per_H_lineage_surface_hashes"
    ] is True
    assert payload["protocol"][
        "canonical_terminal_manifest_receipt_required_by_selection_ledger"
    ] is True
    assert payload["protocol"][
        "canonical_terminal_manifest_receipt_hash_bound_in_preterminal_phase"
    ] is True
    assert payload["protocol"][
        "terminal_label_gate_requires_canonical_terminal_manifest_receipt"
    ] is True
    assert payload["protocol"][
        "terminal_label_capability_openable_in_current_contract"
    ] is False
    assert payload["protocol"]["pairwise_fit_response_metric"] == (
        "EXPECTED_BACC_GAIN_ONLY"
    )
    assert payload["protocol"]["brier_or_log_may_enter_pairwise_ranking_response"] is False
    assert payload["protocol"]["residual_calibration_one_sided_alpha"] == 0.2
    assert payload["protocol"]["terminal_admission_aggregation"] == (
        "CASE_THEN_EQUAL_CENTER"
    )
    assert payload["source_provenance"]["adapter_scope_role"] == (
        "diagnostic_adapter"
    )
    assert payload["source_provenance"]["core_scope_role"] == (
        "neutral_scientific_core"
    )
    assert payload["source_provenance"]["source_scopes_are_disjoint"] is True
    assert len(payload["source_provenance"]["combined_source_seal_hash"]) == 64
    assert payload["source_provenance"]["recompute_and_exact_match_on_load"] is True
    assert payload["protocol"][
        "combined_diagnostic_adapter_and_neutral_core_source_seal_required"
    ] is True
    assert payload["protocol"][
        "combined_source_seal_recomputed_and_exact_matched_on_config_load"
    ] is True
    assert payload["runtime"]["output_or_scratch_creation_allowed"] is False
    assert payload["runtime"]["threadpool_limiter_scope"] == (
        "worker_process_lifetime"
    )
    assert payload["runtime"]["threadpool_info_evidence_required"] is True
    assert payload["claim_boundary"]["fresh_evidence"] is False
    assert payload["claim_boundary"]["may_feed_another_experiment"] is False
    assert all(not key.endswith(("_path", "_root")) for key in payload["inputs"])


def test_cli_and_workspace_refuse_planned_execution_before_mutation(
    tmp_path: Path,
) -> None:
    parsed = cli.build_parser().parse_args(
        [CLI_SURFACE, "--config", str(CONFIG), "--artifact-root", "/tmp/unused"]
    )
    assert parsed.surface == CLI_SURFACE

    workspace = MidogppWorkspace.load(ROOT)
    with pytest.raises(WorkspaceError, match="status='planned'"):
        workspace._render_run(  # noqa: SLF001 - exact pre-run refusal seam
            EXPERIMENT_ID,
            require_inputs=False,
            validate_workspace=True,
            include_all_declared_inputs=True,
        )

    config = load_config(CONFIG)
    direct_output = tmp_path / "direct" / "output"
    with pytest.raises(ProtocolError, match="execution is not authorized"):
        run_planned_router(config, artifact_root=direct_output)
    assert not (tmp_path / "direct").exists()

    cli_output = tmp_path / "cli" / "output"
    with pytest.raises(ProtocolError, match="execution is not authorized"):
        cli.main(
            [
                CLI_SURFACE,
                "--config",
                str(CONFIG),
                "--artifact-root",
                str(cli_output),
            ]
        )
    assert not (tmp_path / "cli").exists()


def test_output_is_forbidden_as_any_downstream_input() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    consumers = [
        experiment.experiment_id
        for experiment in workspace.experiments.values()
        if OUTPUT_ARTIFACT_ID in experiment.input_artifact_ids
    ]

    assert consumers == []
    assert set(output.forbidden_reuse) == FORBIDDEN_DOWNSTREAM_PURPOSES
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert output.semantic_identities["may_feed_stage50"] == "false"
    assert output.semantic_identities["may_feed_stage60"] == "false"
    assert output.semantic_identities["may_feed_stage70"] == "false"
    assert output.semantic_identities["may_feed_another_stage90"] == "false"
    assert output.semantic_identities["may_feed_another_experiment"] == "false"
