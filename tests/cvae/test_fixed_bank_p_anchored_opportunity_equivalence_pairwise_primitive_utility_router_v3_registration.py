from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.config import (
    load_config as load_v2_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.identity import (
    EXPERIMENT_ID as V2_EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID as V2_OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.source_seal import (
    build_source_contract_receipt as build_v2_source_contract_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.config import (
    build_planned_config,
    frozen_config_contract_payload,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    CANONICAL_OUTPUT_RELATIVE_ROOT,
    CLI_SURFACE,
    DIRECT_INPUT_ARTIFACT_IDS,
    EXPERIMENT_ID,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    OUTPUT_ARTIFACT_ID,
    SOURCE_SUPERVISION_ARTIFACT_ID,
    SOURCE_SUPERVISION_REQUIRED_MEMBERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.lifecycle_source_seal import (
    build_lifecycle_source_seal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_seal import (
    build_source_seal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.workspace_binding import (
    canonical_output_root,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = (
    ROOT / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
)
CONFIG = CONFIG_DIRECTORY / (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_"
    "equivalence_pairwise_primitive_utility_router_v3.yaml"
)
V2_CONFIG = CONFIG_DIRECTORY / (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_"
    "equivalence_pairwise_primitive_utility_router_v2.yaml"
)

EXPECTED_V3_DIRECT_INPUT_IDS = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
    "midogpp_output_uniform_b_v2_generation_lock_v1",
    "midogpp_stage90_oe_ppur_source_training_action_supervision_v3",
    "midogpp_stage90_fixed_bank_p_anchored_opportunity_equivalence_pairwise_"
    "primitive_utility_router_test_cache_v3",
    "midogpp_stage90_fixed_bank_p_anchored_opportunity_equivalence_pairwise_"
    "primitive_utility_router_test_manifest_v3",
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router_parent_v3",
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router_amendment_v3",
)
EXPECTED_V2_DIRECT_INPUT_IDS = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
    "midogpp_output_uniform_b_v2_generation_lock_v1",
    "midogpp_stage90_fixed_bank_p_anchored_opportunity_equivalence_pairwise_"
    "primitive_utility_router_test_cache_v2",
    "midogpp_stage90_fixed_bank_p_anchored_opportunity_equivalence_pairwise_"
    "primitive_utility_router_test_manifest_v2",
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router_parent_v2",
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router_amendment_v2",
)

# These are the already-published v2 contract and closed-world source identities.
# V3 registration must not silently reseal, rewrite, or expand its predecessor.
EXPECTED_V2_CONFIG_HASH = (
    "039cc66b9e7bebb35eacadf388f36f35aa62c8f867693a343583872f93084801"
)
EXPECTED_V2_PROTOCOL_HASH = (
    "d6cd84f2734b62f7d3381456111ca9af35cb0ef62d72c7e8b59941024b16bc4b"
)
EXPECTED_V2_ADAPTER_TREE_HASH = (
    "d0b1ee53645511d487a2e47c57441691172bdb4b5ba989cc4b19952c1b1d3ef1"
)
EXPECTED_V2_SOURCE_SEAL_HASH = (
    "96606a67071946a200a740dff2100bd2d45f16388530f60a646388555e492d54"
)
EXPECTED_V2_SOURCE_RECEIPT_HASH = (
    "0cd84d858df7296cc22ed306d29c6f7f354689e100ab634f252707e05ebc1827"
)


def test_v3_registration_is_exact_seven_input_planned_successor() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert DIRECT_INPUT_ARTIFACT_IDS == EXPECTED_V3_DIRECT_INPUT_IDS
    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert experiment.config_path == CONFIG.relative_to(ROOT).as_posix()
    assert experiment.input_artifact_ids == EXPECTED_V3_DIRECT_INPUT_IDS
    assert len(experiment.input_artifact_ids) == 7
    assert len(set(experiment.input_artifact_ids)) == 7
    assert output.availability == "planned_execution_not_authorized"
    assert output.canonical_path == CANONICAL_OUTPUT_RELATIVE_ROOT
    assert canonical_output_root() == ROOT / CANONICAL_OUTPUT_RELATIVE_ROOT
    assert output.semantic_identities["exact_direct_input_count"] == "7"
    assert output.semantic_identities["execution_authorized"] == "false"
    assert output.semantic_identities["consumed_test_reuse_authorized"] == "false"
    assert output.semantic_identities["v2_input_or_state_used"] == "false"


def test_v3_unissued_inputs_are_registered_without_materialization() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    source = workspace.artifacts[SOURCE_SUPERVISION_ARTIFACT_ID]
    amendment = workspace.artifacts[AUTHORIZATION_AMENDMENT_ARTIFACT_ID]

    assert source.availability == "workstation_build_required"
    assert source.required_files == SOURCE_SUPERVISION_REQUIRED_MEMBERS
    assert source.semantic_identities["source_split"] == "SOURCE_ONLY"
    assert source.semantic_identities["source_bundle_materialized"] == "false"
    assert source.semantic_identities["target_rows_present"] == "false"
    assert source.semantic_identities["target_labels_used"] == "false"
    assert source.semantic_identities["execution_authorized"] == "false"

    assert amendment.availability == "planned"
    assert amendment.required_files == ()
    assert amendment.expected_file_hashes == {}
    assert amendment.semantic_identities["amendment_status"] == (
        "ABSENT_NOT_ISSUED"
    )
    assert amendment.semantic_identities["amendment_file_present"] == "false"
    assert amendment.semantic_identities[
        "expected_amendment_sha256_present"
    ] == "false"
    assert amendment.semantic_identities["execution_authorized"] == "false"
    assert amendment.semantic_identities[
        "consumed_test_reuse_authorized"
    ] == "false"


def test_v2_registry_config_and_closed_world_package_are_unchanged() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    experiment = workspace.get_experiment(V2_EXPERIMENT_ID)
    output = workspace.artifacts[V2_OUTPUT_ARTIFACT_ID]
    config = load_v2_config(V2_CONFIG)
    source = build_v2_source_contract_receipt(ROOT)

    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert experiment.input_artifact_ids == EXPECTED_V2_DIRECT_INPUT_IDS
    assert config.contract_hash == EXPECTED_V2_CONFIG_HASH
    assert config.protocol["protocol_hash"] == EXPECTED_V2_PROTOCOL_HASH
    assert source.adapter_member_count == 25
    assert source.adapter_tree_sha256 == EXPECTED_V2_ADAPTER_TREE_HASH
    assert source.combined_source_sha256 == EXPECTED_V2_SOURCE_SEAL_HASH
    assert source.receipt_hash == EXPECTED_V2_SOURCE_RECEIPT_HASH
    assert output.semantic_identities["config_contract_hash"] == (
        EXPECTED_V2_CONFIG_HASH
    )
    assert output.semantic_identities["protocol_contract_hash"] == (
        EXPECTED_V2_PROTOCOL_HASH
    )
    assert output.semantic_identities["combined_source_seal_sha256"] == (
        EXPECTED_V2_SOURCE_SEAL_HASH
    )
    assert output.semantic_identities["combined_source_receipt_sha256"] == (
        EXPECTED_V2_SOURCE_RECEIPT_HASH
    )


def test_checked_in_v3_config_is_the_exact_path_free_planned_payload() -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config = load_config(CONFIG)
    expected = frozen_config_contract_payload()

    assert raw == expected
    assert config == build_planned_config()
    assert config.to_payload() == expected
    assert config.execution_authorized is False
    assert config.direct_input_artifact_ids == EXPECTED_V3_DIRECT_INPUT_IDS
    assert config.source_supervision_content_sha256 is None
    assert config.source_supervision_row_order_sha256 is None
    assert config.authorization_amendment_sha256 is None
    assert expected["inputs"]["source_supervision"]["direct_input_ordinal"] == 3
    assert expected["inputs"]["authorization_amendment_sha256"] is None
    assert expected["paths_present"] is False


def test_v3_catalog_config_protocol_and_source_seal_pins_match_live() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    semantics = workspace.artifacts[OUTPUT_ARTIFACT_ID].semantic_identities
    config = load_config(CONFIG)
    source = build_source_seal(ROOT)
    lifecycle = build_lifecycle_source_seal(ROOT)

    assert semantics["config_contract_hash"] == config.contract_hash
    assert semantics["protocol_contract_hash"] == config.protocol_hash
    assert int(semantics["adapter_source_member_count"]) == (
        source.adapter_member_count
    )
    assert semantics["adapter_source_tree_sha256"] == source.adapter_tree_sha256
    assert int(semantics["neutral_core_source_member_count"]) == (
        source.neutral_member_count
    )
    assert semantics["neutral_core_source_tree_sha256"] == (
        source.neutral_tree_sha256
    )
    assert int(semantics["production_source_member_count"]) == (
        source.production_member_count
    )
    assert semantics["production_source_tree_sha256"] == (
        source.production_tree_sha256
    )
    assert semantics["shared_protocol_source_sha256"] == (
        source.shared_protocol_sha256
    )
    assert int(semantics["combined_source_member_count"]) == (
        source.adapter_member_count
        + source.neutral_member_count
        + source.production_member_count
        + 1
    )
    assert semantics["combined_source_seal_sha256"] == (
        source.combined_source_sha256
    )
    assert semantics["combined_source_receipt_sha256"] == source.receipt_hash
    assert int(semantics["lifecycle_source_member_count"]) == (
        lifecycle.member_count
    )
    assert semantics["lifecycle_source_seal_sha256"] == (
        lifecycle.lifecycle_source_seal_sha256
    )
    assert semantics["lifecycle_source_receipt_hash"] == lifecycle.receipt_hash
    assert semantics["bank_content_index_file_sha256"] == (
        EXPECTED_BANK_CONTENT_INDEX_SHA256
    )
    assert semantics["generation_content_index_file_sha256"] == (
        EXPECTED_GENERATION_CONTENT_INDEX_SHA256
    )


def test_v3_cli_inspection_and_closed_run_are_mutation_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "output"
    scratch = tmp_path / "scratch"
    common = [
        CLI_SURFACE,
        "--config",
        str(CONFIG),
        "--artifact-root",
        str(output),
        "--scratch-root",
        str(scratch),
    ]

    status = cli.main([*common, "--inspect-plan"])
    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["execution_authorized"] is False
    assert payload["direct_input_count"] == 7
    assert payload["direct_input_artifact_ids"] == list(
        EXPECTED_V3_DIRECT_INPUT_IDS
    )
    assert payload["source_supervision_direct_input_ordinal"] == 3
    assert payload["authorization_amendment_input_ordinal"] == 7
    assert payload["authorization_amendment_issued"] is False
    assert payload["filesystem_mutation_performed"] is False
    assert payload["labels_opened"] is False
    assert payload["experiment_launched"] is False
    assert not output.exists()
    assert not scratch.exists()

    with pytest.raises(ProtocolError, match="not authorized"):
        cli.main(common)
    assert not output.exists()
    assert not scratch.exists()
