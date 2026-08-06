from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


METADATA_INPUT_ID = "midogpp_routing_metadata_profiles_v1"
COMPATIBILITY_EXPERIMENT_ID = (
    "midogpp.routing_compatibility.uniform_b_v2_metadata_exact_match_lock.v1"
)
COMPATIBILITY_OUTPUT_ID = (
    "midogpp_output_uniform_b_v2_metadata_exact_match_compatibility_v1"
)
COMPATIBILITY_CONFIG = (
    "experiments/midogpp/stages/60_routing_and_composition/configs/"
    "uniform_b_v2_metadata_exact_match_compatibility_v1.yaml"
)
POLICY_EXPERIMENT_ID = (
    "midogpp.routing_and_composition."
    "uniform_b_v2_metadata_tie_union_policy_lock.v1"
)
POLICY_OUTPUT_ID = "midogpp_output_uniform_b_v2_metadata_tie_union_policy_lock_v1"
POLICY_CONFIG = (
    "experiments/midogpp/stages/60_routing_and_composition/configs/"
    "uniform_b_v2_metadata_tie_union_policy_lock_v1.yaml"
)
BANK_ID = "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
GENERATION_LOCK_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
CONTROL_ID = "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1"
DOMAIN_MAPPING_SHA256 = (
    "79d703ccf3085ae3968698c2ac44a3eabc2713b434762cc6b2fd2fa90126a211"
)
COMPATIBILITY_SEMANTIC_IDENTITIES = {
    "metadata_compatibility_contract": (
        "midogpp_uniform_b_v2_metadata_exact_match_compatibility_lock_v1"
    ),
    "config_contract_hash": "89191838fbb3f1c8",
    "metadata_profile_lock_hash": "de23d1c8de734503",
    "compatibility_lock_hash": "4b46b3d157b07781",
    "metadata_profile_table_hash": "eee8dececd62bef8",
    "compatibility_score_table_hash": "aec9e0b5b09a1fe5",
    "domain_mapping_sha256": DOMAIN_MAPPING_SHA256,
}
POLICY_SEMANTIC_IDENTITIES = {
    "policy_lock_contract": (
        "midogpp_uniform_b_v2_metadata_tie_union_policy_lock_v1"
    ),
    "config_contract_hash": "df69d7481b0fd62a",
    "policy_lock_hash": "27f16953b32c46cd",
    "policy_plan_hash": "ca10b4ed038ccdba",
    "selection_table_hash": "ba611c0180149d79",
    "assignment_table_hash": "b3bc2187806f8788",
    "compatibility_lock_hash": "4b46b3d157b07781",
    "compatibility_score_table_hash": "aec9e0b5b09a1fe5",
    "equal_union_policy_lock_hash": "4b9ea514308b084f",
    "generation_lock_hash": "34e551425710362e",
    "expert_bank_lock_hash": "9972a41dcd4814cd",
}
COMPATIBILITY_FILE_SHA256 = {
    "config.resolved.yaml": "09931064fee3fb2e5fee05b3a0c4692e8cfec6953965abf0dd1957628a8c7580",
    "provenance/input_artifacts.json": "aa1dc63a7cf7fc3f581d45fceb57e95fb29af80e897e0b5ede24bf1f94df2de1",
    "manifests/protocol_manifest.json": "49c11f7bf9ac23d259c68d24647c7ea7ab3a584749323e8ff9dd7ea0a6b3b26a",
    "manifests/metadata_profile_lock.json": "755593f8167d37c0bb4f125020c3c21779258ba15f566e38c016d9198f69f002",
    "manifests/compatibility_lock.json": "ceab51603a13d6f43c2a313b46bedf0975cbc47ce49045742df3993c3359ba53",
    "manifests/content_index.json": "f78c2b0ec4b31e75d6e4a43b626f6887363ce7562783e47ce9f15e67ad362904",
    "reports/compatibility_decision.json": "79559962dfc6fa2b63f219471ed27123d2ea273ce2e2a087c4fad30a19c1aaea",
    "reports/leakage_report.json": "c81ac1ebebc53b18f1199d3f28b3871371508fe345c1f0a9eb2e4754cb9fb492",
    "reports/run_state.json": "8616bb36c45ce095002ccc431d7fea9d0b7903a88aaa3b0802689215f4fb58d2",
    "reports/validation_report.json": "feef2f8d306ad05ecd2a17d10536b44585704920f77e016b6abbef6c56dfb002",
    "tables/metadata_profiles.csv": "16fbbb70f142222d6bc782c30a8779f69a4884a9c11b6444eee91f53835bf57c",
    "tables/compatibility_scores.csv": "927a99e2d645eca384651c6d6ccb6d77284a9cb896f53525da57be7a654046b0",
}
POLICY_FILE_SHA256 = {
    "config.resolved.yaml": "f1cb0945fbef86828be150db5ea58d50e23873322689e8260213afd5224992da",
    "provenance/input_artifacts.json": "30db64eb2999003300a20245fe1f0fcfa6805f45f5261655608e605bb53c7717",
    "manifests/protocol_manifest.json": "2039d71bbffb51fedf7271e5903268462240ebb33bd8f966e1a419cf532b11b3",
    "manifests/policy_lock.json": "43b8f7aaac782502cf4f530d3afc7dcef6ab0ffc7fb3f06bb576dac5561a0e9f",
    "manifests/metadata_tie_union_policy_plan.json": "f725eb8728aa98501b45598e05d5f38341b1a1eb381bbd6740ad2203c18eb3f9",
    "manifests/content_index.json": "137e3c166c6555a277038b43fc047c5a5d9b8336aac9f54a1cd5f757c485b17f",
    "reports/policy_decision.json": "0e3e48f13e014677c316c92bc475c67fb08e2ef9e6ef9d0d1f4f526419532d54",
    "reports/leakage_report.json": "65ddeaf339e1c9b1bcc2ee21def68679bbea42b913b9ab11a2a323036ea8a072",
    "reports/run_state.json": "64128c704bcebb2e8b69eb3c04d362611b4a3bd952a2ebaf40b6960545f92754",
    "reports/validation_report.json": "2391dfa905a5974f69b3096dd5c3e80f2e092577bca3936f30be0756636fdd28",
    "tables/policy_selections.csv": "68ca6e2615fa06698f009751b42d3a7f46444f37d931641812fd35cae28928ff",
    "tables/policy_assignments.csv": "30237fe07fc94d54549e10f32ccdb271a1571fc808f045d103020181bc98ca9c",
}


def test_routing_metadata_input_is_narrow_and_hash_authorized() -> None:
    workspace = MidogppWorkspace.load()

    workspace.validate()
    metadata = workspace.artifacts[METADATA_INPUT_ID]

    assert metadata.stage == "dataset_contract"
    assert metadata.claim_scope == "dataset_contract_and_split_provenance"
    assert metadata.required_files == ("domain_mapping.json",)
    assert metadata.authoritative_files == ()
    assert metadata.may_feed_recipe_selection is False
    assert metadata.may_feed_deployable_selection is True
    assert metadata.expected_file_hashes["domain_mapping.json"].algorithm == "sha256"
    assert (
        metadata.expected_file_hashes["domain_mapping.json"].digest
        == DOMAIN_MAPPING_SHA256
    )
    assert "manifest.csv" not in metadata.provenance_files
    assert "split_manifest.csv" not in metadata.provenance_files


def test_metadata_compatibility_is_a_nonselecting_stage60_proxy() -> None:
    workspace = MidogppWorkspace.load()

    workspace.validate()
    experiment = workspace.get_experiment(COMPATIBILITY_EXPERIMENT_ID)
    output = workspace.artifacts[COMPATIBILITY_OUTPUT_ID]
    stage60 = workspace.stages["60_routing_and_composition"]

    assert "routing_compatibility_only" in stage60["allowed_claim_scopes"]
    assert (
        "predeclared_compatibility_artifact_from_unconsumed_source_inner_evidence_or_frozen_query_metadata_available_at_routing_time_or_disjoint_unlabeled_target_support_for_ranked_selected_or_weighted_policies"
        in stage60["required_upstream"]
    )
    assert experiment.status == "active"
    assert experiment.stage == "60_routing_and_composition"
    assert experiment.claim_scope == "routing_compatibility_only"
    assert experiment.config_path == COMPATIBILITY_CONFIG
    assert experiment.input_artifact_ids == (METADATA_INPUT_ID,)
    assert experiment.output_artifact_id == COMPATIBILITY_OUTPUT_ID
    assert experiment.runner_argv == (
        "{python}",
        "-m",
        "midogpp_thesis",
        "cvae-routing",
        "uniform-b-v2-metadata-exact-match-compatibility",
        "--config",
        "{resolved_config}",
        "--artifact-root",
        f"output://{COMPATIBILITY_OUTPUT_ID}",
    )
    assert output.claim_scope == "routing_compatibility_only"
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is True
    assert dict(output.semantic_identities) == COMPATIBILITY_SEMANTIC_IDENTITIES
    assert {
        relative: expectation.digest
        for relative, expectation in output.expected_file_hashes.items()
    } == COMPATIBILITY_FILE_SHA256
    assert {item.algorithm for item in output.expected_file_hashes.values()} == {
        "sha256"
    }
    assert "tables/metadata_profiles.csv" in output.required_files
    assert "tables/compatibility_scores.csv" in output.required_files


def test_metadata_policy_consumes_only_four_authorized_locks() -> None:
    workspace = MidogppWorkspace.load()

    workspace.validate()
    experiment = workspace.get_experiment(POLICY_EXPERIMENT_ID)
    output = workspace.artifacts[POLICY_OUTPUT_ID]

    assert experiment.status == "active"
    assert experiment.stage == "60_routing_and_composition"
    assert experiment.claim_scope == "routing_and_composition"
    assert experiment.config_path == POLICY_CONFIG
    assert experiment.input_artifact_ids == (
        BANK_ID,
        GENERATION_LOCK_ID,
        CONTROL_ID,
        COMPATIBILITY_OUTPUT_ID,
    )
    assert experiment.output_artifact_id == POLICY_OUTPUT_ID
    assert experiment.runner_argv == (
        "{python}",
        "-m",
        "midogpp_thesis",
        "cvae-routing",
        "uniform-b-v2-metadata-tie-union-policy-lock",
        "--config",
        "{resolved_config}",
        "--artifact-root",
        f"output://{POLICY_OUTPUT_ID}",
    )
    assert output.claim_scope == "routing_and_composition"
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is True
    assert dict(output.semantic_identities) == POLICY_SEMANTIC_IDENTITIES
    assert {
        relative: expectation.digest
        for relative, expectation in output.expected_file_hashes.items()
    } == POLICY_FILE_SHA256
    assert {item.algorithm for item in output.expected_file_hashes.values()} == {
        "sha256"
    }
    assert "tables/policy_selections.csv" in output.required_files
    assert "tables/policy_assignments.csv" in output.required_files


@pytest.mark.parametrize(
    ("config_path", "expected_inputs"),
    (
        (COMPATIBILITY_CONFIG, {METADATA_INPUT_ID}),
        (
            POLICY_CONFIG,
            {BANK_ID, GENERATION_LOCK_ID, CONTROL_ID, COMPATIBILITY_OUTPUT_ID},
        ),
    ),
)
def test_metadata_routing_configs_resolve_only_declared_inputs(
    config_path: str,
    expected_inputs: set[str],
) -> None:
    workspace = MidogppWorkspace.load()
    payload = yaml.safe_load(
        (workspace.repo_root / config_path).read_text(encoding="utf-8")
    )
    used_inputs: set[str] = set()

    workspace.resolve_value(payload, require_inputs=False, used_inputs=used_inputs)

    assert used_inputs == expected_inputs


def test_stage70_can_consume_control_and_frozen_metadata_policy_together() -> None:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    catalog = deepcopy(source.catalog_payload)
    output_id = "test_uniform_b_v2_paired_metadata_downstream_output"
    catalog["artifacts"].append(
        {
            "artifact_id": output_id,
            "stage": "70_frozen_policy_downstream",
            "canonical_path": (
                "artifacts/midogpp/70_frozen_policy_downstream/"
                "test_uniform_b_v2_paired_metadata/v1"
            ),
            "availability": "generated_on_run",
            "migration": "canonical_output",
            "evidence_label": "TODO_VERIFY_ARTIFACT",
            "claim_scope": "synthetic_downstream_utility",
        }
    )
    registry["experiments"].append(
        {
            "experiment_id": "test.uniform_b_v2.paired_metadata.downstream",
            "stage": "70_frozen_policy_downstream",
            "status": "planned",
            "claim_scope": "synthetic_downstream_utility",
            "output_artifact_id": output_id,
            "input_artifact_ids": [CONTROL_ID, POLICY_OUTPUT_ID],
            "runner": {"argv": ["{python}", "-c", "pass"]},
        }
    )

    _clone_workspace(source, registry=registry, catalog=catalog).validate()


@pytest.mark.parametrize(
    "experiment_id",
    (COMPATIBILITY_EXPERIMENT_ID, POLICY_EXPERIMENT_ID),
)
@pytest.mark.parametrize(
    "forbidden_stage",
    ("50_all_candidate_utility_matrix", "90_oracles_and_diagnostics"),
)
def test_metadata_routing_rejects_stage50_and_stage90_inputs(
    experiment_id: str,
    forbidden_stage: str,
) -> None:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    catalog = deepcopy(source.catalog_payload)
    forbidden_id = f"test_forbidden_metadata_{forbidden_stage}"
    catalog["artifacts"].append(
        {
            "artifact_id": forbidden_id,
            "stage": forbidden_stage,
            "canonical_path": f"artifacts/midogpp/{forbidden_stage}/forbidden/v1",
            "availability": "generated_on_run",
            "migration": "canonical_output",
            "evidence_label": "VALIDATED_TEST_INPUT",
            "claim_scope": "diagnostic_only",
            "may_feed_deployable_selection": True,
        }
    )
    experiment = _experiment(registry, experiment_id)
    experiment["input_artifact_ids"] = [*experiment["input_artifact_ids"], forbidden_id]

    with pytest.raises(WorkspaceError, match="consumes forbidden upstream stage"):
        _clone_workspace(source, registry=registry, catalog=catalog).validate()


def test_broad_labeled_contract_cannot_replace_narrow_metadata_input() -> None:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    experiment = _experiment(registry, COMPATIBILITY_EXPERIMENT_ID)
    experiment["input_artifact_ids"] = ["midogpp_dataset_contract_annotation_patch_v1"]

    with pytest.raises(WorkspaceError, match="may_feed_deployable_selection=true"):
        _clone_workspace(source, registry=registry).validate()


def _experiment(registry: dict[str, object], experiment_id: str) -> dict[str, object]:
    experiments = registry["experiments"]
    assert isinstance(experiments, list)
    return next(item for item in experiments if item["experiment_id"] == experiment_id)


def _clone_workspace(
    source: MidogppWorkspace,
    *,
    registry: dict[str, object] | None = None,
    catalog: dict[str, object] | None = None,
) -> MidogppWorkspace:
    return MidogppWorkspace(
        repo_root=source.repo_root,
        registry=registry or deepcopy(source.registry_payload),
        catalog=catalog or deepcopy(source.catalog_payload),
        workspace=source.workspace_payload,
        protocol_defaults=source.protocol_defaults_payload,
    )
