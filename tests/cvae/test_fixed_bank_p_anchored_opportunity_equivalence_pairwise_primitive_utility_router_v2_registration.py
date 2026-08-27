from __future__ import annotations

import json
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.config import (
    frozen_config_contract_payload,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    DIRECT_INPUT_ARTIFACT_IDS,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    PACKAGE_NAME,
    V1_EXPERIMENT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.source_seal import (
    _reject_unsealed_project_imports,
    build_source_contract_receipt,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_"
    "equivalence_pairwise_primitive_utility_router_v2.yaml"
)


def test_v2_registration_is_exact_six_input_planned_successor() -> None:
    config = load_config(CONFIG)
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    amendment = workspace.artifacts[AUTHORIZATION_AMENDMENT_ARTIFACT_ID]

    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert experiment.input_artifact_ids == DIRECT_INPUT_ARTIFACT_IDS
    assert len(experiment.input_artifact_ids) == 6
    assert len(set(experiment.input_artifact_ids)) == 6
    assert output.availability == "planned_execution_not_authorized"
    assert output.semantic_identities["execution_authorized"] == "false"
    assert output.semantic_identities["consumed_test_reuse_authorized"] == "false"
    assert output.semantic_identities[
        "parsed_probability_matrix_science_receipt_implemented"
    ] == "true"
    assert output.semantic_identities["probability_matrix_shape"] == "9928x7"
    assert output.semantic_identities[
        "canonical_admitted_row_binding_implemented"
    ] == "true"
    assert output.semantic_identities[
        "typed_preterminal_decision_ledger_implemented"
    ] == "true"
    assert output.semantic_identities[
        "artifact_only_fresh_process_attestation_implemented"
    ] == "true"
    assert output.semantic_identities[
        "one_shot_terminal_aggregate_capability_implemented"
    ] == "true"
    assert output.semantic_identities["bank_content_index_file_sha256"] == (
        EXPECTED_BANK_CONTENT_INDEX_SHA256
    )
    assert output.semantic_identities[
        "generation_content_index_file_sha256"
    ] == EXPECTED_GENERATION_CONTENT_INDEX_SHA256
    assert output.semantic_identities["structural_service_injection_allowed"] == (
        "false"
    )
    assert output.semantic_identities["canonical_scientific_service_implemented"] == (
        "false"
    )
    assert output.semantic_identities["canonical_terminal_evaluator_implemented"] == (
        "false"
    )
    assert output.semantic_identities["authorized_execution_available"] == "false"
    assert amendment.availability == "planned"
    assert amendment.required_files == ()
    assert amendment.semantic_identities["amendment_status"] == "ABSENT_NOT_ISSUED"
    assert output.semantic_identities["config_contract_hash"] == config.contract_hash
    assert output.semantic_identities["protocol_contract_hash"] == (
        config.protocol["protocol_hash"]
    )

    predecessor = workspace.get_experiment(V1_EXPERIMENT_ID)
    assert predecessor.status == "planned"
    assert len(predecessor.input_artifact_ids) == 3


def test_checked_in_config_is_exact_non_authorized_payload() -> None:
    config = load_config(CONFIG)
    assert config.to_payload() == frozen_config_contract_payload()
    assert config.execution_authorized is False
    assert config.source_contract_hash is None
    assert config.expected_authorization_amendment_sha256 is None
    assert config.inputs["direct_input_count"] == 6
    assert tuple(config.inputs["direct_input_artifact_ids"]) == (
        DIRECT_INPUT_ARTIFACT_IDS
    )
    assert config.inputs["expert_bank_content_index_file_sha256"] == (
        EXPECTED_BANK_CONTENT_INDEX_SHA256
    )
    assert config.inputs["generation_lock_content_index_file_sha256"] == (
        EXPECTED_GENERATION_CONTENT_INDEX_SHA256
    )
    assert config.protocol["probability_matrix_shape"] == [9928, 7]
    assert config.protocol["probability_matrix_column_ids"][0] == "P_PROTECTED"
    assert config.protocol["canonical_admitted_row_binding_required"] is True
    assert config.protocol["typed_preterminal_decision_ledger_required"] is True
    assert config.protocol[
        "two_artifact_only_fresh_process_attestations_required"
    ] is True
    assert config.protocol["terminal_evaluated_case_count"] == 218
    assert config.protocol[
        "probability_shards_within_admitted_scratch_root_required"
    ] is True
    assert config.protocol["bank_content_index_file_sha256"] == (
        EXPECTED_BANK_CONTENT_INDEX_SHA256
    )
    assert config.protocol["generation_content_index_file_sha256"] == (
        EXPECTED_GENERATION_CONTENT_INDEX_SHA256
    )
    assert config.protocol[
        "admitted_input_location_binding_exact_matched_by_service_factory"
    ] is True
    assert config.protocol["shared_protocol_source_member_sealed"] is True
    assert config.protocol[
        "production_workstation_observations_caller_injectable"
    ] is False
    assert config.protocol["preterminal_service_manifest_path_exposed"] is False
    assert config.protocol[
        "structural_scientific_service_injection_allowed"
    ] is False


def test_catalog_source_pins_match_live_closed_world_source() -> None:
    source = build_source_contract_receipt(ROOT)
    output = MidogppWorkspace.load(ROOT).artifacts[OUTPUT_ARTIFACT_ID]
    semantics = output.semantic_identities
    assert int(semantics["adapter_source_member_count"]) == source.adapter_member_count
    assert semantics["adapter_source_tree_sha256"] == source.adapter_tree_sha256
    assert int(semantics["neutral_core_source_member_count"]) == (
        source.neutral_member_count
    )
    assert semantics["neutral_core_source_tree_sha256"] == (
        source.neutral_tree_sha256
    )
    assert semantics["shared_protocol_source_sha256"] == (
        source.shared_protocol_sha256
    )
    assert int(semantics["combined_source_member_count"]) == (
        source.adapter_member_count + source.neutral_member_count + 1
    )
    assert semantics["combined_source_seal_sha256"] == (
        source.combined_source_sha256
    )
    assert semantics["combined_source_receipt_sha256"] == source.receipt_hash


def test_source_seal_rejects_relative_imports_outside_both_sealed_trees(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "src/midogpp_thesis/cvae/diagnostics"
        / PACKAGE_NAME
        / "unsealed_import_probe.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        "from ....real_feature import forbidden_runtime\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="unsealed project module"):
        _reject_unsealed_project_imports(source, repository_root=tmp_path)


def test_cli_inspection_is_mutation_free_and_direct_run_remains_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "output"
    status = cli.main(
        [
            "fixed-bank-p-anchored-opportunity-equivalence-pairwise-primitive-utility-router-v2",
            "--config",
            str(CONFIG),
            "--artifact-root",
            str(output),
            "--inspect-plan",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["execution_authorized"] is False
    assert payload["direct_input_count"] == 6
    assert payload["parsed_probability_matrix_shape"] == [9928, 7]
    assert payload["mutation_performed"] is False
    assert not output.exists()

    with pytest.raises(ProtocolError, match="not authorized"):
        cli.main(
            [
                "fixed-bank-p-anchored-opportunity-equivalence-pairwise-primitive-utility-router-v2",
                "--config",
                str(CONFIG),
                "--artifact-root",
                str(output),
            ]
        )
    assert not output.exists()
