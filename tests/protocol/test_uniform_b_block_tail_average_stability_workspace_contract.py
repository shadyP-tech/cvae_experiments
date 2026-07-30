from __future__ import annotations

from midogpp_thesis.workspace.runtime import MidogppWorkspace


def test_stability_probe_workspace_contract() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(
        "midogpp.oracle.uniform_b_block_tail_average_stability_probe.v1"
    )
    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == (
        "midogpp_output_uniform_b_source_expert_adaptation_pilot_v2",
    )
    artifact = workspace.artifacts[
        "midogpp_output_uniform_b_block_tail_average_stability_probe_v1"
    ]
    assert artifact.claim_scope == "diagnostic_only"
    assert artifact.may_feed_recipe_selection is False
    assert artifact.may_feed_deployable_selection is False
