from __future__ import annotations

from midogpp_thesis.workspace.runtime import MidogppWorkspace


EXPERIMENT_ID = "midogpp.oracle.uniform_b_source_expert_adaptation_pilot.v1"
OUTPUT_ID = "midogpp_output_uniform_b_source_expert_adaptation_pilot_v1"


def test_uniform_b_adaptation_pilot_is_registered_nonadoptive_diagnostic() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "failed"
    assert experiment.runnable is False
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == (
        "midogpp_dataset_contract_annotation_patch_v1",
        "midogpp_virchow2_uniform_b_canonical_train_cache_seed42",
        "midogpp_virchow2_xyxy_feature_cache_seed42",
    )
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert {
        "expert_bank_evidence",
        "generation_evidence",
        "routing_evidence",
    }.issubset(output.forbidden_reuse)

def test_uniform_b_adaptation_v2_is_a_separate_nonadoptive_amendment() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(
        "midogpp.oracle.uniform_b_source_expert_adaptation_pilot.v2"
    )
    output = workspace.artifacts[
        "midogpp_output_uniform_b_source_expert_adaptation_pilot_v2"
    ]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert output.canonical_path.endswith("uniform_b_source_expert_adaptation_pilot_v2")
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert "tables/real_reference_preflight.csv" in output.required_files

