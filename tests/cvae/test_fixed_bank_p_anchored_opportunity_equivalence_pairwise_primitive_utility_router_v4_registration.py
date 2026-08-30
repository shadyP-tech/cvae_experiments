from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.config import (
    build_planned_config,
    frozen_config_contract_payload,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    CANONICAL_OUTPUT_RELATIVE_ROOT,
    DIRECT_INPUT_ARTIFACT_IDS,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    PRESERVED_V3_AMENDMENT_SHA256,
    SOURCE_SUPERVISION_ARTIFACT_ID,
    SOURCE_SUPERVISION_REQUIRED_MEMBERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.input_contract import (
    build_planned_seven_input_contract,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.output_persistence import (
    COMPLETE_CATALOG_MEMBERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.protocol import (
    frozen_protocol_payload,
)
from midogpp_thesis.cvae.diagnostics.oe_ppur_v4_preparation.predecessor import (
    capture_predecessor_preservation,
)
from midogpp_thesis.cvae.diagnostics.oe_ppur_v4_preparation.workspace import (
    V3_AMENDMENT_ARTIFACT_ID,
    V3_AMENDMENT_FILENAME,
    V3_EXPERIMENT_ID,
    V3_OUTPUT_ARTIFACT_ID,
    _validate_predecessor_workspace_metadata,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_"
    "equivalence_pairwise_primitive_utility_router_v4.yaml"
)
CATALOG = ROOT / "experiments/midogpp/artifact_catalog.yaml"
REGISTRY = ROOT / "experiments/midogpp/registry.yaml"
DOC = (
    ROOT
    / "docs/wiki/03-experiments/midogpp-uniform-b-v2-consumed-test-fixed-bank-"
    "p-anchored-opportunity-equivalence-pairwise-primitive-utility-router-v4.md"
)

EXPECTED_SOURCE_MEMBER_HASHES = {
    "manifests/source_training_surface.json": (
        "2313db90779d1b509db620faa5425ddad2a2e0824c1d709a3489ce7f7f99294b"
    ),
    "manifests/source_pool_lineage.json": (
        "c3599f8f56c89382494a19c019432dee5a8dc12d45c638a5f8388875c658edf5"
    ),
    "tables/source_rows.csv": (
        "a324215960961074d924d5b67198263b5afdc906b6800eb96835b448d5d45a31"
    ),
    "arrays/source_action_probabilities.npy": (
        "979d7575ef933bb4b208ce58ca469a88d8861d23fb9bcb682cbe7a6b7f4fb649"
    ),
    "manifests/content_index.json": (
        "1cb9c1a2b548b7b31250b57b5be4a9870ef97ce299877a54c8de6780898f4d5f"
    ),
    "reports/validation_report.json": (
        "881377105eb62cd09c2a17aa27cdeb1ab59e01e57b4a2af44672b54fab44b71a"
    ),
}

def test_v4_registration_is_exact_seven_input_nonrunnable_successor() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert experiment.config_path == CONFIG.relative_to(ROOT).as_posix()
    assert experiment.input_artifact_ids == DIRECT_INPUT_ARTIFACT_IDS
    assert len(experiment.input_artifact_ids) == 7
    assert len(set(experiment.input_artifact_ids)) == 7
    assert output.availability == "planned_execution_not_authorized"
    assert output.canonical_path == CANONICAL_OUTPUT_RELATIVE_ROOT
    assert output.semantic_identities["exact_direct_input_count"] == "7"
    assert output.semantic_identities["execution_authorized"] == "false"
    assert output.semantic_identities["launch_authorized"] == "false"
    assert output.semantic_identities["target_labels_closed"] == "true"
    assert output.semantic_identities["v3_operational_state_reuse"] == "false"
    assert output.semantic_identities["nfs_safe_publication_topology"] == (
        "NFS_SAFE_IN_PLACE_COMMIT"
    )
    assert set(output.required_files) == set(COMPLETE_CATALOG_MEMBERS)
    assert len(output.required_files) == len(COMPLETE_CATALOG_MEMBERS)


def test_v4_source_alias_binds_every_member_without_inheriting_authority() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    source = workspace.artifacts[SOURCE_SUPERVISION_ARTIFACT_ID]

    assert source.canonical_path == (
        "artifacts/midogpp/90_oracles_and_diagnostics/"
        "oe_ppur_source_training_action_supervision/v3"
    )
    assert source.required_files == SOURCE_SUPERVISION_REQUIRED_MEMBERS
    assert set(source.expected_file_hashes) == set(EXPECTED_SOURCE_MEMBER_HASHES)
    for member, expected in EXPECTED_SOURCE_MEMBER_HASHES.items():
        expectation = source.expected_file_hashes[member]
        assert expectation.algorithm == "sha256"
        assert expectation.digest == expected

    semantics = source.semantic_identities
    assert semantics["source_split"] == "SOURCE_ONLY"
    assert semantics["source_bundle_materialized"] == "true"
    assert semantics["target_rows_present"] == "false"
    assert semantics["target_labels_used"] == "false"
    assert semantics["source_content_provenance_only"] == "true"
    assert semantics["predecessor_no_feed_fence_acknowledged"] == "true"
    assert semantics["v4_source_content_reuse_exception"] == (
        "USER_AUTHORIZED_V4_ONLY_HASH_EXACT_CONTENT_PROVENANCE"
    )
    assert semantics["consumer_resolution_fence_only"] == "true"
    assert semantics["v3_amendment_authority_inherited"] == "false"
    assert semantics["v3_output_authority_inherited"] == "false"
    assert semantics["v3_lease_or_scratch_inherited"] == "false"
    assert semantics["execution_authorized"] == "false"
    assert semantics["consumed_test_reuse_authorized"] == "false"
    assert semantics["execution_authorized_by_this_artifact"] == "false"


def test_v4_catalog_contract_pins_match_live_planned_contracts() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    amendment = workspace.artifacts[AUTHORIZATION_AMENDMENT_ARTIFACT_ID]
    config = load_config(CONFIG)
    planned_inputs = build_planned_seven_input_contract()
    protocol = frozen_protocol_payload()

    assert yaml.safe_load(CONFIG.read_text(encoding="utf-8")) == (
        frozen_config_contract_payload()
    )
    assert config == build_planned_config()
    assert output.semantic_identities["config_contract_hash"] == (
        config.contract_hash
    )
    assert output.semantic_identities["protocol_contract_hash"] == (
        protocol["protocol_hash"]
    )
    assert output.semantic_identities["planned_seven_input_contract_hash"] == (
        planned_inputs.receipt_hash
    )
    assert amendment.availability == "planned"
    assert amendment.required_files == ()
    assert amendment.expected_file_hashes == {}
    assert amendment.semantic_identities["amendment_status"] == (
        "FRESH_PREFLIGHT_AND_PUBLICATION_REQUIRED"
    )
    assert amendment.semantic_identities["launch_authorized"] == "false"
    assert amendment.semantic_identities["execution_authorized"] == "false"
    assert amendment.semantic_identities[
        "v3_amendment_sha256_preservation_witness"
    ] == PRESERVED_V3_AMENDMENT_SHA256
    assert amendment.semantic_identities["v3_amendment_status"] == (
        "ISSUED_UNRENDERED_UNCLAIMED_NO_RUN"
    )


def test_v3_sealed_metadata_matches_issued_unrendered_live_witness(
    tmp_path: Path,
) -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    source = workspace.artifacts[
        "midogpp_stage90_oe_ppur_source_training_action_supervision_v3"
    ]
    amendment = workspace.artifacts[V3_AMENDMENT_ARTIFACT_ID]
    output = workspace.artifacts[V3_OUTPUT_ARTIFACT_ID]
    experiment = workspace.get_experiment(V3_EXPERIMENT_ID)

    assert source.availability == "workstation_only"
    assert source.semantic_identities["source_bundle_materialized"] == "true"
    assert {
        member: expectation.digest
        for member, expectation in source.expected_file_hashes.items()
    } == EXPECTED_SOURCE_MEMBER_HASHES
    assert amendment.availability == "workstation_only"
    assert amendment.required_files == (V3_AMENDMENT_FILENAME,)
    assert amendment.expected_file_hashes[V3_AMENDMENT_FILENAME].digest == (
        PRESERVED_V3_AMENDMENT_SHA256
    )
    assert amendment.semantic_identities["amendment_status"] == (
        "AUTHORIZED_SINGLE_USE_NOT_CONSUMED"
    )
    assert amendment.semantic_identities["rendered_launch_envelope_present"] == (
        "false"
    )
    assert output.availability == "planned"
    assert output.semantic_identities["output_root_present"] == "false"
    assert output.semantic_identities["lease_claimed"] == "false"
    assert output.semantic_identities["scratch_present"] == "false"
    assert output.semantic_identities["experiment_launched"] == "false"
    assert experiment.status == "planned"
    assert experiment.runnable is False

    amendment_path = ROOT / "tests/cvae/fixtures/oe_ppur_v3_amendment_7.json"
    assert hashlib.sha256(amendment_path.read_bytes()).hexdigest() == (
        PRESERVED_V3_AMENDMENT_SHA256
    )
    witness = capture_predecessor_preservation(
        amendment_path=amendment_path,
        output_root=tmp_path / "v3-output",
        lease_path=tmp_path / "v3-lease",
        scratch_root=tmp_path / "v3-scratch",
    )
    _validate_predecessor_workspace_metadata(workspace, witness)


def test_v4_documentation_keeps_the_terminal_no_launch_boundary() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "POST_HOC_CONSUMED_TEST_SENSITIVITY" in text
    assert "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE" in text
    assert PRESERVED_V3_AMENDMENT_SHA256 in text
    assert "issued/unrendered/unclaimed/no-run" in text
    assert "NFS_SAFE_IN_PLACE_COMMIT" in text
    assert "Scientific execution adapter: implemented" in text
    assert "Preflight and amendment publication are preparation-only" in text
    assert "They do\nnot authorize the `run` subcommand" in text
    assert "RUN_TERMINAL_CONSUMED_TEST" in text
    assert "-m midogpp_thesis.oe_ppur_v4 run" in text
    assert "did not issue an\namendment, render an authority file" in text
