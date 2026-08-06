from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


EXPERIMENT_ID = (
    "midogpp.routing_and_composition.uniform_b_v2_equal_union_policy_lock.v1"
)
OUTPUT_ID = "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1"
BANK_ID = "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
GENERATION_LOCK_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
CONFIG_PATH = (
    "experiments/midogpp/stages/60_routing_and_composition/configs/"
    "uniform_b_v2_equal_union_policy_lock_v1.yaml"
)
CLAIM_SCOPE = "routing_and_composition"
REQUIRED_BUNDLE = {
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/policy_lock.json",
    "manifests/equal_union_policy_plan.json",
    "manifests/content_index.json",
    "reports/policy_decision.json",
    "reports/leakage_report.json",
    "reports/run_state.json",
    "reports/validation_report.json",
    "tables/policy_assignments.csv",
}
EXPECTED_FILE_HASHES = {
    "config.resolved.yaml": "9c438af6a6d90210340c5dca211073f4e8951a0eed6320dd1b81147568cfe288",
    "provenance/input_artifacts.json": "885805ed6b4f4f9030aa51090ae977df94262b2f0c88ab2803861c31da982234",
    "manifests/protocol_manifest.json": "ba70c80e8ae023137508dbc8297f154fed1535184a92c82fd29c7415f1d9bbae",
    "manifests/policy_lock.json": "59dc4019941b5fcd72717ef465fcebb0c6dd99c631d66c4591023bbdb0b1b476",
    "manifests/equal_union_policy_plan.json": "161e02d8cd065a50d6c11d3f6d4106ab41df26247b79dde3519bb2e9cdeda9b9",
    "manifests/content_index.json": "d283a201ae24e99f1b6e79e98404b59ec85c19e13f2ffdc02a62c4c42aa94b61",
    "reports/policy_decision.json": "2414847d43df853f0e00169ae796bee1a39f6f21f125f2b696c88a1435038680",
    "reports/leakage_report.json": "9b6f4f317bd03b4f4851754136f2282a415f4887e8c94ae8af467c714affa125",
    "reports/run_state.json": "42d2e7d4f250709f071014ee2bb8d6e32ae559d69d2c4aefe175e12666e144c9",
    "reports/validation_report.json": "b1482658a8eebce2c7c164af54399b821e78e7a2654f2d245544453bb47340fe",
    "tables/policy_assignments.csv": "e27a7bb238f9a67a632972e86301db886c685268901fbe1eefcd22290d2a7671",
}


def test_equal_union_policy_lock_is_the_registered_stage60_control() -> None:
    workspace = MidogppWorkspace.load()

    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ID]
    bank = workspace.artifacts[BANK_ID]
    generation_lock = workspace.artifacts[GENERATION_LOCK_ID]
    stage60 = workspace.stages["60_routing_and_composition"]

    assert experiment.status == "active"
    assert experiment.runnable is True
    assert experiment.stage == "60_routing_and_composition"
    assert experiment.claim_scope == CLAIM_SCOPE
    assert experiment.config_path == CONFIG_PATH
    assert experiment.output_artifact_id == OUTPUT_ID
    assert experiment.input_artifact_ids == (BANK_ID, GENERATION_LOCK_ID)
    assert experiment.runner_argv == (
        "{python}",
        "-m",
        "midogpp_thesis",
        "cvae-routing",
        "uniform-b-v2-equal-union-policy-lock",
        "--config",
        "{resolved_config}",
        "--artifact-root",
        f"output://{OUTPUT_ID}",
    )
    assert stage60["performs_deployable_selection"] is True
    assert (
        "predeclared_compatibility_artifact_from_unconsumed_source_inner_evidence_or_frozen_query_metadata_available_at_routing_time_or_disjoint_unlabeled_target_support_for_ranked_selected_or_weighted_policies"
        in stage60["required_upstream"]
    )
    assert (
        "fixed_all_eligible_source_controls_may_omit_compatibility_only_when_no_ranking_selection_or_weighting_occurs"
        in stage60["hard_requirements"]
    )
    assert bank.may_feed_deployable_selection is True
    assert generation_lock.may_feed_deployable_selection is True
    assert output.claim_scope == CLAIM_SCOPE
    assert output.migration == "canonical_output"
    assert output.semantic_identities == {
        "policy_lock_contract": "midogpp_uniform_b_v2_equal_union_policy_lock_v1",
        "config_contract_hash": "e581a1bf98762031",
        "policy_lock_hash": "4b9ea514308b084f",
        "policy_plan_hash": "9ec24122d7d0cdf1",
        "assignment_table_hash": "c85415c1b953c04e",
        "generation_lock_hash": "34e551425710362e",
        "expert_bank_lock_hash": "9972a41dcd4814cd",
    }
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is True
    assert set(output.required_files) == REQUIRED_BUNDLE
    assert {
        relative: expectation.digest
        for relative, expectation in output.expected_file_hashes.items()
        if expectation.algorithm == "sha256"
    } == EXPECTED_FILE_HASHES
    assert "synthetic_downstream_utility_evidence" not in output.forbidden_reuse


def test_equal_union_policy_lock_config_resolves_only_its_two_declared_inputs() -> None:
    workspace = MidogppWorkspace.load()
    config = yaml.safe_load(
        (workspace.repo_root / CONFIG_PATH).read_text(encoding="utf-8")
    )
    used_inputs: set[str] = set()

    resolved = workspace.resolve_value(
        config,
        require_inputs=False,
        used_inputs=used_inputs,
    )

    assert used_inputs == {BANK_ID, GENERATION_LOCK_ID}
    assert resolved["experiment"]["artifact_root"].endswith(
        "artifacts/midogpp/60_routing_and_composition/"
        "uniform_b_v2_equal_union_policy_lock/v1"
    )


def test_equal_union_policy_lock_is_consumable_by_stage70() -> None:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    catalog = deepcopy(source.catalog_payload)
    downstream_output = "test_uniform_b_v2_equal_union_downstream_output"
    catalog["artifacts"].append(
        {
            "artifact_id": downstream_output,
            "stage": "70_frozen_policy_downstream",
            "canonical_path": (
                "artifacts/midogpp/70_frozen_policy_downstream/"
                "test_uniform_b_v2_equal_union/v1"
            ),
            "availability": "generated_on_run",
            "migration": "canonical_output",
            "evidence_label": "TODO_VERIFY_ARTIFACT",
            "claim_scope": "synthetic_downstream_utility",
        }
    )
    registry["experiments"].append(
        {
            "experiment_id": "test.uniform_b_v2.equal_union.downstream",
            "stage": "70_frozen_policy_downstream",
            "status": "planned",
            "claim_scope": "synthetic_downstream_utility",
            "output_artifact_id": downstream_output,
            "input_artifact_ids": [OUTPUT_ID],
            "runner": {"argv": ["{python}", "-c", "pass"]},
        }
    )

    _clone_workspace(source=source, registry=registry, catalog=catalog).validate()


@pytest.mark.parametrize(
    "forbidden_stage",
    ("50_all_candidate_utility_matrix", "90_oracles_and_diagnostics"),
)
def test_stage60_rejects_stage50_and_stage90_inputs(forbidden_stage: str) -> None:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    catalog = deepcopy(source.catalog_payload)
    forbidden_id = f"test_forbidden_{forbidden_stage}"
    catalog["artifacts"].append(
        {
            "artifact_id": forbidden_id,
            "stage": forbidden_stage,
            "canonical_path": f"artifacts/midogpp/{forbidden_stage}/test_forbidden/v1",
            "availability": "generated_on_run",
            "migration": "canonical_output",
            "evidence_label": "VALIDATED_TEST_INPUT",
            "claim_scope": "diagnostic_only",
            "may_feed_deployable_selection": True,
        }
    )
    experiment = _experiment_payload(registry, EXPERIMENT_ID)
    experiment["input_artifact_ids"] = [*experiment["input_artifact_ids"], forbidden_id]

    with pytest.raises(WorkspaceError, match="consumes forbidden upstream stage"):
        _clone_workspace(source=source, registry=registry, catalog=catalog).validate()


def test_stage60_rejects_an_input_not_authorized_for_deployable_selection() -> None:
    source = MidogppWorkspace.load()
    catalog = deepcopy(source.catalog_payload)
    _artifact_payload(catalog, GENERATION_LOCK_ID)[
        "may_feed_deployable_selection"
    ] = False

    with pytest.raises(WorkspaceError, match="may_feed_deployable_selection=true"):
        _clone_workspace(source=source, catalog=catalog).validate()


def _experiment_payload(
    registry: dict[str, object],
    experiment_id: str,
) -> dict[str, object]:
    experiments = registry["experiments"]
    assert isinstance(experiments, list)
    return next(item for item in experiments if item["experiment_id"] == experiment_id)


def _artifact_payload(catalog: dict[str, object], artifact_id: str) -> dict[str, object]:
    artifacts = catalog["artifacts"]
    assert isinstance(artifacts, list)
    return next(item for item in artifacts if item["artifact_id"] == artifact_id)


def _clone_workspace(
    *,
    source: MidogppWorkspace,
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
